import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import resample
from data_parser.parser import read_zip_binary, read_binary

class ECGAugmentDataset(Dataset):
    
    def __init__(self, signals_path: str, labels_path: str, dataset: str="test",
                 transform=None, train_val_split_csv: str=None, augment: bool=False,
                 augment_mode: str="mild", n_augs: int=1, random_state: int=42):
        np.random.seed(random_state)
        self.augment = Augmentation(augment_mode, n_augs) if augment and dataset != "test" else None
        self.transform = transform
        self.X, self.y, self.split = self._load_data_from_file(signals_path, labels_path, train_val_split_csv)
        if dataset in ('train', 'val'):
            filtered_y = self.y[self.split['split'] == dataset]
            filtered_X = [signal for i, signal in enumerate(self.X) if self.split['split'].iloc[i] == dataset]
            self.y = filtered_y
            self.X = filtered_X
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self.augment.augment(x)
        if self.transform:
            x = self.transform(x)
        y = -1 if self.y is None else self.y["label"].iloc[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        
    def _load_data_from_file(self, signals_path: str, labels_path: str|None=None,
                             train_val_split_csv: str|None=None):
        #X = read_zip_binary(signals_path)
        X = read_binary(signals_path)
        y = None if labels_path is None else pd.read_csv(labels_path, header=None, names=['label'])
        split_df = None if train_val_split_csv is None else pd.read_csv(train_val_split_csv, index_col=0) 
        return X, y, split_df
    

class Augmentation():
    
    def __init__(self, mode: str="mild", num_augs: int = 1):
        if mode not in ("mild", "moderate", "high"):
            raise ValueError(f"Augmentation mode has to be either mild, moderate or high, but was: {mode}.")
        if num_augs not in (1, 2):
            raise ValueError(f"Number of augmentations can be only 1 or 2, but was: {num_augs}.")
        self.n_augs = 1
        self.mode = mode
        self.params = {
                "mild" : {
                    "shift_min_timesteps": 50,
                    "shift_max_timesteps": 100,
                    "resample_min_factor": 0.9,
                    "resample_max_factor": 1.1,
                    "gaussian_noise_std_frac": 0.01,
                    "crop_fraction": 0.9
                    },
                "moderate": {
                    "shift_min_timesteps": 100,
                    "shift_max_timesteps": 300,
                    "resample_min_factor": 0.8,
                    "resample_max_factor": 1.2,
                    "gaussian_noise_std_frac": 0.05,
                    "crop_fraction": 0.8
                    },
                "high": {
                    "shift_min_timesteps": 300,
                    "shift_max_timesteps": 600,
                    "resample_min_factor": 0.7,
                    "resample_max_factor": 1.3,
                    "gaussian_noise_std_frac": 0.1,
                    "crop_fraction": 0.7
                    }
            }
        
    def augment(self, signal: list[int]) -> list[int]:
        idxs = np.random.choice(list(range(4)), size=self.n_augs, replace=False)
        for idx in idxs:
            signal = self._perform_augmentation_by_idx(signal, idx)
        return signal
    
    def _perform_augmentation_by_idx(self, signal: list[int], idx: int):
        if idx == 0:
            left = np.random.uniform() <= 0.5
            timestamp_min = self.params[self.mode]["shift_min_timesteps"]
            timestamp_max = self.params[self.mode]["shift_max_timesteps"]
            timesteps = np.random.randint(timestamp_min, timestamp_max + 1)
            return self._shift_in_time(signal, left=left, timesteps=timesteps)
        elif idx == 1:
            factor_min = self.params[self.mode]["resample_min_factor"]
            factor_max = self.params[self.mode]["resample_max_factor"]
            factor = np.random.uniform(factor_min, factor_max)
            return self._stretch_or_compress_in_time(signal, factor=factor)
        elif idx == 2:
            fraction_of_signals_std = self.params[self.mode]["gaussian_noise_std_frac"]
            return self._add_gaussian_noise(signal, fraction_of_signals_std=fraction_of_signals_std)
        elif idx == 3:
            fraction = self.params[self.mode]["crop_fraction"]
            return self._crop(signal, fraction=fraction)
        else:
            raise ValueError(f"Index of augmentation too large, should be 0, 1, 2 or 3, but was: {idx}.")
    
    def _shift_in_time(self, signal: list[int], left: bool=True, timesteps: int=1):
        if left:
            return signal[min(timesteps, len(signal)):] + timesteps * [0]
        else:
            return timesteps * [0] + signal[:-timesteps]
    
    def _stretch_or_compress_in_time(self, signal: list[int], factor: float=1.1):
        new_length = int(len(signal) * factor)
        return resample(signal, new_length)
    
    def _add_gaussian_noise(self, signal: list[int], fraction_of_signals_std: float):
        signal_arr = np.array(signal)
        signal_std = np.std(signal_arr)
        noise_std = signal_std * fraction_of_signals_std
        augmented_signal = signal_arr + np.random.normal(scale=noise_std, size=signal_arr.shape[0])
        return augmented_signal.tolist()
    
    def _crop(self, signal: list[int], fraction: float=0.8):
        signal_len = len(signal)
        crop_len = int(signal_len * fraction)
        if crop_len >= signal_len:
            return signal
        start_idx = np.random.randint(0, signal_len - crop_len + 1)
        end_idx = start_idx + crop_len
        return signal[start_idx:end_idx]
    