import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_parser.parser import read_zip_binary#, read_binary

class ECGDataset(Dataset):
    
    def __init__(self, signals_path: str, labels_path: str=None, dataset: str='test',
                 transform=None, train_val_split_csv: str=None):
        self.dataset = dataset
        self.X, self.y, self.split = self._load_data_from_files(signals_path, labels_path, train_val_split_csv)
        if dataset in ("train", "val"):
            filtered_y = self.y[self.split['split'] == dataset]
            filtered_X = [signal for i, signal in enumerate(self.X) if self.split['split'].iloc[i] == dataset]
            self.X = filtered_X
            self.y = filtered_y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        y = -1 if self.y is None else self.y['label'].iloc[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    def _load_data_from_files(self, signals_path: str, labels_path: str | None=None,
                              train_val_split_csv: str | None=None):
        X = read_zip_binary(signals_path)
        #X = read_binary(signals_path)
        y = None if labels_path is None else pd.read_csv(labels_path, header=None, names=['label'])
        split_df = None if train_val_split_csv is None else pd.read_csv(train_val_split_csv, index_col=0)
        return X, y, split_df


class Normalizer:
    
    def __init__(self, eps=1e-6):
        self.eps = eps
        
    def __call__(self, sample: list[int]):
        sample_arr = np.array(sample)
        mean = np.mean(sample_arr)
        std = np.std(sample_arr)
        return (sample_arr - mean) / (std + self.eps)


class PadderTruncator:
    
    def __init__(self, target_length: int=9000, mode: str="left"):
        self.target_length = target_length
        self.mode = mode
        
    def __call__(self, sample: np.ndarray):
        sample_len = sample.shape[0]
        if sample_len == self.target_length:
            return sample
        elif sample_len < self.target_length:
            missing_zeros = self.target_length - sample_len
            if self.mode == 'left':
                return np.pad(sample, (0, missing_zeros))
            elif self.mode == 'center':
                left = missing_zeros // 2
                right = missing_zeros - left
                return np.pad(sample, (left, right))
            else:
                raise ValueError(f'Unsupported padding in PadderTruncator for mode: {self.mode}.')
        else:
            if self.mode == 'left':
                return sample[:self.target_length]
            elif self.mode == 'center':
                middle = sample_len // 2
                left = middle - self.target_length // 2
                right = middle + self.target_length // 2
                if self.target_length % 2 == 1:
                    right += 1
                return sample[left:right]
            else:
                raise ValueError(f'Unsupported truncating in PadderTruncator for mode: {self.mode}.')
