from torch.utils.data import Dataset, Subset
import torchaudio
import pandas as pd
import numpy as np
import torch
class UrbanSoundDataset(Dataset):
    """
    PyTorch Dataset wrapper for the UrbanSound8K dataset.

    This dataset reads metadata from a CSV file (UrbanSound8K's `metadata/UrbanSound8K.csv`),
    loads the corresponding audio clips, and optionally applies resampling and transforms.

    Args:
        csv_path (str): Path to the UrbanSound8K metadata CSV file.
        root_dir (str): Root directory of the UrbanSound8K dataset containing the fold subfolders.
        target_sr (int, optional): Target sample rate for resampling audio (default: 16000).
        transform (callable, optional): Optional transform function applied to each waveform
            (e.g., spectrogram extraction, augmentation) (default: None).
        normalize_wavs (bool, optional): normalize parameter for torchaudio.load() (default: True).
    """

    def __init__(self, csv_path, root_dir, target_sr=16000, transform=None, normalize_wavs=True):
        self.meta = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform
        self.target_sr = target_sr
        self.normalize_wavs = normalize_wavs

        self.classes = self.meta.drop_duplicates(subset=['classID', 'class']).sort_values('classID')['class'].to_numpy()
        self.targets = self.meta['classID'].astype(int).to_numpy()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        metadata_row = self.meta.iloc[idx]
        wav_path = f"{self.root}/fold{metadata_row['fold']}/{metadata_row['slice_file_name']}"

        waveform, sr = torchaudio.load(wav_path, normalize=self.normalize_wavs)

        if waveform.shape[0] > 1: #non-mono
            #convert to mono by taking the mean across the C dim
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
        if self.transform:
            waveform = self.transform(waveform)

        label = metadata_row['classID']

        return waveform, label


# transform wrapper for separating val&test out of train augmentation
class TransformSubset(Subset):
    def __init__(self, dataset: UrbanSoundDataset, indices, transform=None):
        super().__init__(dataset, indices)

        self.transform = transform

        if hasattr(self.dataset, "targets"):
            self.targets = np.array(self.dataset.targets[indices])
        elif hasattr(self.dataset, "labels"):
            self.targets = np.array([self.dataset.labels[i] for i in self.indices])
        else:
            raise AttributeError(
                "Underlying dataset has no 'targets' or 'labels' attribute"
            )

    def __getitem__(self, idx):
        items = super().__getitem__(idx) #each item is (wav, label)

        if self.transform is None:
            return items

        if isinstance(items, list):
            items = [(self.transform(wav), label) for wav, label in items]
        else: # tuple
            items = (self.transform(items[0]), items[1])

        return items

    def __getitems__(self, indices: list[int]):
        items = super().__getitems__(indices)

        if self.transform is None:
            return items

        return [(self.transform(wav), label) for wav, label in items]
