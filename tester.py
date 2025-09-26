import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from audio_loading_utils import UrbanSoundDataset, TransformSubset
import torchaudio


def perform_testing(net, batch_size, testing_set, weights_file: str):
    print("performing testing...")
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=8)

    net_data = torch.load(weights_file)
    net.load_state_dict(net_data['model'], strict=True)

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        targets = []
        predictions = []
        for inputs, labels in testing_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            pred_vals, pred_classes = torch.max(outputs.data, 1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)
    if hasattr(testing_set, "dataset"): #subset
        class_names = testing_set.dataset.classes
    else: #dataset
        class_names = testing_set.classes
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    ax = plt.subplot()
    plt.rcParams.update({'font.size': 6})
    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    plt.show()

    print(f"Finished.")
    print(classification_report(y_true=targets, y_pred=predictions, target_names=class_names))

def perform_manual_testing(net, weights_file: str, sample_rate: int, transform: torch.nn.Sequential, class_names: list[str]):
    own_set = []

    wav_names = ["dog_bark.wav", "idle_engine.wav", "street_music.wav"]
    for wav_name in wav_names:
        wav_path = f"test_samples/{wav_name}"
        wav_file, sr = torchaudio.load(wav_path, normalize=True)
        wav_file_resampled = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav_file)
        wav_stectrogram = transform(wav_file_resampled)
        own_set.append(wav_stectrogram)

    net_data = torch.load(weights_file)
    net.load_state_dict(net_data['model'], strict=True)

    net.eval()
    with (torch.no_grad()):

        inputs = torch.stack(own_set)
        inputs = inputs.cuda()
        outputs = net(inputs)

        _, pred_classes = torch.max(outputs.data, dim=1)

        for wav_name, pred_class in zip(wav_names, pred_classes):
            print(f"for {wav_name} the model predicted: {class_names[pred_class]}")
