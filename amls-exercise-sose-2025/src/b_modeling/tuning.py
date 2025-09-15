import torch
import json
import random
import pandas as pd
import numpy as np
import itertools as it
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from b_modeling.training import train_model
from sklearn.utils.class_weight import compute_class_weight
from b_modeling.dataset import Normalizer, PadderTruncator, ECGDataset
from b_modeling.model import ECGClassifier
from typing import Tuple

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_dataloaders_and_weigths(x_train_path: str, y_train_path: str,
                                split_path: str, device: str) -> Tuple[DataLoader, DataLoader, torch.tensor]:
    split_df = pd.read_csv(split_path)
    train_df = split_df[split_df['split'] == 'train']
    weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=np.array(train_df['label'])), dtype=torch.float32)
    weights = weights.to(device)
    
    transform = transforms.Compose([Normalizer(), PadderTruncator()])
    train_dataset = ECGDataset(x_train_path, y_train_path, dataset='train', transform=transform, train_val_split_csv=split_path)
    val_dataset = ECGDataset(x_train_path, y_train_path, dataset='val', transform=transform, train_val_split_csv=split_path)
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return loader_train, loader_val, weights

def tune_hyperparameters(params: dict, res_out_path: str, n_epochs: int=20, device='cpu',
                         x_train_path = "../../../data/X_train.zip",
                         y_train_path = "../../../data/y_train.csv",
                         split_path = "../../train_val_split_jakub.csv"):
    train_loader, val_loader, weights = prepare_dataloaders_and_weigths(x_train_path, y_train_path, split_path, device)
    
    res_dict = {}
    
    for param_set in it.product(*params.values()):
        print(f'\nNew param set: {param_set}\n')
        model = ECGClassifier(channels=param_set[2], res_blocks=param_set[3],
                              lstm_hidden_size=param_set[4], fc_hidden_dim=param_set[5], dropout_proba=param_set[6])
        optimizer = Adam(model.parameters(), lr=param_set[0], weight_decay=param_set[1])
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        res_train = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, n_epochs=n_epochs)
        res_dict[str(param_set)] = res_train
        
    with open(res_out_path, 'w') as f:
        json.dump(res_dict, f)

def further_tuning(params: dict, res_out_path: str, n_epochs: int=40, device: str='cpu', random_state: int=42, n_repetitions: int=3,
                   x_train_path = "../../../data/X_train.zip",
                   y_train_path = "../../../data/y_train.csv",
                   split_path = "../../train_val_split_jakub.csv"):
        set_seed(random_state)
        train_loader, val_loader, weights = prepare_dataloaders_and_weigths(x_train_path, y_train_path, split_path, device)
        
        res_dict = {}
        combinations = list(it.product(*params.values()))
        for i, param_set in enumerate(combinations):
            print(f"\nCombination {i+1}/{len(combinations)}\n")
            res_param_set = {}
            for j in range(n_repetitions):
                model = ECGClassifier(channels=param_set[2], res_blocks=param_set[3],
                                      lstm_hidden_size=param_set[4], fc_hidden_dim=param_set[5], dropout_proba=param_set[6])
                optimizer = Adam(model.parameters(), lr=param_set[0], weight_decay=param_set[1])
                step_size = 10 if param_set[3] == 10 else 20 if param_set[3] == 7 else 25
                scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
                loss_fn = nn.CrossEntropyLoss(weight=weights)
                res_train = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, n_epochs=n_epochs, scheduler=scheduler, n_epochs_early_stop=5)
                res_param_set[j] = res_train
            res_dict[str(param_set)] = res_param_set
        
        with open(res_out_path, 'w') as f:
            json.dump(res_dict, f)


if __name__ == "__main__":
    params = {
        'lr': [1e-4, 1e-3, 1e-2],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'channels': [32, 64],
        'res_blocks': [5, 10],
        'lstm_hidden_size': [64, 128],
        'fc_hidden_dim': [64],
        'dropout_proba': [0.3]
        }
    
    params_test = {
        'lr': [1e-3],
        'weight_decay': [1e-5],
        'channels': [16],
        'res_blocks': [1],
        'lstm_hidden_size': [32],
        'fc_hidden_dim': [16],
        'dropout_proba': [0.3]
        }
    
    tune_hyperparameters(params_test, "results.json", device='cuda', n_epochs=20)

