import torch
import torch.nn as nn

from audio_loading_utils import UrbanSoundDataset, TransformSubset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from support_scripts.weighted_sampling_distributor import analyze_weaknesses_produce_weights

import torchaudio
from torchaudio import transforms as T
import custom_augmentations as cT
import torch_audiomentations as eT

from architecture import MobileNetAudio
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

import math
import numpy as np
from tester import perform_testing
from sklearn.metrics import precision_score, recall_score, accuracy_score

from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
from zoneinfo import ZoneInfo



METADATA_CSV_PATH = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
DATA_PATH = 'data/UrbanSound8K/audio'

RAND_STATE = 44
N_CLASSES = 10
NUM_WORKERS = 4
SAMPLE_RATE = 16000


def curr_time():
    return datetime.now(ZoneInfo('Europe/Kiev'))


def printshare(msg, logfile="training_log.txt"):
    print(msg)

    with open(logfile, "a") as f:
        print(msg, file=f)


def cosannealing_decay_warmup(warmup_steps, T_0, T_mult, decay_factor, base_lr, eta_min):
    # returns the func that performs all the calculations.
    # useful for keeping all the params in one place = scheduler def.
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return ((epoch + 1) / warmup_steps) ** 2

        annealing_step = epoch - warmup_steps

        # calculating which cycle (zero-based) are we in,
        # current cycle length (T_current) and position inside the cycle (t)
        if T_mult == 1:
            cycle = annealing_step // T_0
            t = annealing_step % T_0
            T_current = T_0

        else:
            # fast log-based computation
            cycle = int(math.log((annealing_step * (T_mult - 1)) / T_0 + 1, T_mult))
            sum_steps_of_previous_cycles = T_0 * (T_mult ** cycle - 1) // (T_mult - 1)
            t = annealing_step - sum_steps_of_previous_cycles
            T_current = T_0 * (T_mult ** cycle)


        # enable decay
        eta_max = base_lr * (decay_factor ** cycle)

        # cosine schedule between (eta_min, max_lr]
        lr = eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(math.pi * t / T_current))
        return lr/base_lr

    return lr_lambda






def perform_training(net,
                     training_set,
                     validation_set,
                     epochs, w_decay, batch_size, sub_batch_size,
                     lr, lr_lambda: cosannealing_decay_warmup,
                     pretrained: bool | str = False):

    assert batch_size % sub_batch_size == 0 #screws up gradient accumulation otherwise

    printshare("training preparation...")

    #======== creating balanced-batch dataloaders ========
    t_class_counts = np.bincount(training_set.targets)
    t_class_weights = 1.0 / t_class_counts #weights based on dataset imbalance
    # extra modifiers based on current situation and model`s weaknesses
    t_class_modifiers = analyze_weaknesses_produce_weights("checkpoints/stats/ep_266_p_96.3_r_96.1_a_96.2_stats.pth")
    t_class_weights *= t_class_modifiers

    t_sample_weights = [t_class_weights[t] for t in training_set.targets]
    train_sampler = WeightedRandomSampler(t_sample_weights, num_samples=len(t_sample_weights), replacement=True)
    train_loader = DataLoader(training_set, batch_size=sub_batch_size, sampler=train_sampler, num_workers=NUM_WORKERS)

    v_class_counts = np.bincount(validation_set.targets)
    v_class_weights = 1.0 / v_class_counts
    v_sample_weights = [v_class_weights[t] for t in validation_set.targets]
    val_sampler = WeightedRandomSampler(v_sample_weights, num_samples=len(v_sample_weights), replacement=True)
    val_loader = DataLoader(validation_set, batch_size=sub_batch_size, sampler=val_sampler, num_workers=NUM_WORKERS)

    #========= loading the checkpoint and preparing optimizers =========

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=w_decay)
        #[
        #    {"params": net.features[-2].parameters()},  # last residual block
        #    {"params": net.features[-1].parameters()},  # last conv
        #    {"params": net.classifier.parameters()}  # classifier
        #],

    #used LambdaLR to implement CosineAnnealing with warm restarts and decay.
    #yup, we need the base_lr to be passed in, cause it looks like this is the safest way.
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    #scheduler = CosineAnnealingLR(
    #    optimizer=optimizer,
    #    T_max=50,
    #    eta_min=1e-8,
    #)

    curr_epoch = 0
    if isinstance(pretrained, str):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(pretrained)
        mid_se_keys = ["mid_se.fc.0.weight", "mid_se.fc.0.bias", "mid_se.fc.2.weight", "mid_se.fc.2.bias"]

        if 'model' not in checkpoint:
            missing, unexpected = net.load_state_dict(checkpoint, strict=False)
            printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

        else:
            missing, unexpected = net.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['weight_decay'] = w_decay

            #scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = checkpoint['epoch']
            curr_epoch = checkpoint['epoch'] + 1

            printshare("all the dicts set up successfully.")


        printshare(f"[DEBUG] model missing statedict vals: {missing};")
        printshare(f"[DEBUG] model unexpected statedict vals: {unexpected}")

    #manual testing cycle
    #while(True):

    #    image, _ = training_set[225]
    #    transform = v2.ToPILImage()
    #    for i in range(16):
    #        img = transform(image[i])
    #        plt.imshow(img)
    #        plt.title(f"Augmented sample #0")
    #        plt.axis('off')
    #        plt.show()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/stats", exist_ok=True)
    printshare("done.")

    #========== training itself ==========
    while curr_epoch < epochs:
        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] epoch {curr_epoch + 1}/{epochs} processing...")
        train_targets, train_predictions, train_loss = perform_training_epoch(
            net=net,
            full_batch_size=batch_size, sub_batch_size=sub_batch_size,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        train_precision = round(100 * precision_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
        train_recall = round(100 * recall_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
        train_accuracy = round(100 * accuracy_score(y_true=train_targets, y_pred=train_predictions), 3)

        printshare(f"training done. precision: {train_precision}%; recall: {train_recall}%; accuracy: {train_accuracy}%")


        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] processing validation phase...")
        val_targets, val_predictions, val_loss = perform_validation_epoch(
            net=net,
            val_loader=val_loader,
            criterion=criterion,
        )
        val_precision = round(100 * precision_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_recall = round(100 * recall_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
        val_accuracy = round(100 * accuracy_score(y_true=val_targets, y_pred=val_predictions), 3)

        printshare(f"validation done. precision: {val_precision}%; recall: {val_recall}%; accuracy: {val_accuracy}%\n\n")

        torch.save({ # model
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': curr_epoch,

        }, f'checkpoints/ep_{curr_epoch+1}_p_{round(val_precision, 1)}_r_{round(val_recall, 1)}_a_{round(val_accuracy, 1)}_model.pth')

        torch.save({ # stats
            'epoch': curr_epoch,
            'train_targets': train_targets,
            'train_predictions': train_predictions,
            'train_loss': train_loss,
            'val_loss': val_loss,
        },
            f'checkpoints/stats/ep_{curr_epoch + 1}_p_{round(val_precision, 1)}_r_{round(val_recall, 1)}_a_{round(val_accuracy, 1)}_stats.pth')

        curr_epoch += 1

    printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] training successfully finished.")
    return net


def perform_training_epoch(net, full_batch_size, sub_batch_size,
                           train_loader, criterion, optimizer, scheduler):
    targets = []
    predictions = []
    batch_losses = []
    net.train()

    accum_steps = math.ceil(full_batch_size / sub_batch_size)  # number of sub-batches per "big batch"

    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)

        pred_vals, pred_classes = torch.max(outputs.data, 1)
        targets.extend(labels.detach().cpu().numpy())
        predictions.extend(pred_classes.detach().cpu().numpy())

        loss = criterion(outputs, labels)
        loss = loss / accum_steps # smoothing the magnitude
        loss.backward()
        batch_losses.append(loss.item())

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    epoch_loss = sum(batch_losses) / len(batch_losses)
    return targets, predictions, epoch_loss


def perform_validation_epoch(net, val_loader, criterion):
    net.eval()
    with torch.no_grad():
        targets = []
        predictions = []
        batch_losses = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)

            batch_losses.append(criterion(outputs, labels).item())

            pred_vals, pred_classes = torch.max(outputs.data, 1)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        return targets, predictions, epoch_loss





def custom_loader(path):
    return Image.open(path, formats=["JPEG"])




if __name__ == '__main__':

    net = MobileNetAudio(n_classes=N_CLASSES, width_mult=0.25)
    net.cuda(0)

    train_transform = torch.nn.Sequential(
        #cT.RandomizedPitchShift(sample_rate=SAMPLE_RATE, n_fft=256, shift_range=2),
        cT.RandomRepeat(sample_rate=SAMPLE_RATE, target_duration_s=4),
        cT.UnsqueezeBatch(),
        eT.Gain(min_gain_in_db=-6, max_gain_in_db=6, output_type='tensor'),
        eT.Shift(min_shift=-0.2, max_shift=0.2, output_type='tensor'),
        eT.PolarityInversion(p=0.5, output_type='tensor'),
        cT.SqueezeBatch(),
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, n_mels=64),
        T.AmplitudeToDB(),

        T.SpecAugment(n_time_masks=5, time_mask_param=6, n_freq_masks=3, freq_mask_param=2, zero_masking=True),
    )

    valtest_transform = torch.nn.Sequential(
        cT.RandomRepeat(sample_rate=SAMPLE_RATE, target_duration_s=4, random_state=RAND_STATE),
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=512, n_mels=64),
        T.AmplitudeToDB(),
    )

    dataset = UrbanSoundDataset(csv_path=METADATA_CSV_PATH, root_dir=DATA_PATH,
                                  target_sr=SAMPLE_RATE, transform=None, normalize_wavs=True)

    train_indices, valtest_indices = train_test_split(
        np.arange(len(dataset)),
        train_size=0.75,
        stratify=dataset.targets,
        random_state=RAND_STATE
    )

    val_indices, test_indices = train_test_split(
        valtest_indices,
        test_size=0.5,  # 12.5% val 12.5% test
        stratify=dataset.targets[valtest_indices],
        random_state=RAND_STATE
    )

    train_set = TransformSubset(dataset=dataset, indices=train_indices, transform=train_transform)
    val_set = TransformSubset(dataset=dataset, indices=val_indices, transform=valtest_transform)
    test_set = TransformSubset(dataset=dataset, indices=test_indices, transform=valtest_transform)

    #from support_scripts.data_stats_collector import duration_hist
    #duration_hist(dataset)
    #print(len(dataset))
    #from support_scripts.data_stats_collector import classcount_hist
    #classcount_hist(class_counts=np.bincount(dataset.targets), class_labels=dataset.classes)

    perform_training(net, train_set, val_set,
                     epochs=500, w_decay=0.001, batch_size=64, sub_batch_size=64,
                     lr=1e-3, lr_lambda=cosannealing_decay_warmup(
                       warmup_steps=0, T_0=10, T_mult=1.2, decay_factor=0.8, base_lr=1e-3, eta_min=1e-8),
                     pretrained='checkpoints/ep_266_p_96.3_r_96.1_a_96.2_model.pth')

    #perform_testing(net=net, batch_size=64, testing_set=test_set,
    #                weights_file='checkpoints/ep_266_p_96.3_r_96.1_a_96.2_model.pth')

    from tester import perform_manual_testing
    perform_manual_testing(net, weights_file='checkpoints/ep_266_p_96.3_r_96.1_a_96.2_model.pth',
                           sample_rate=SAMPLE_RATE, transform=valtest_transform, class_names=dataset.classes)



