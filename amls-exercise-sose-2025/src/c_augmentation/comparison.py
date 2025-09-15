import torch
import json
import pandas as pd
import numpy as np
import itertools as it
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from b_modeling.dataset import Normalizer, PadderTruncator
from b_modeling.model import ECGClassifier
from b_modeling.training import train_model
from c_augmentation.dataset import ECGAugmentDataset
from typing import Tuple



def prepare_weights(split_path: str, device: str) -> torch.tensor:
    split_df = pd.read_csv(split_path)
    train_df = split_df[split_df["split"] == "train"]
    weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=np.array(train_df['label'])), dtype=torch.float32)
    weights = weights.to(device)
    return weights

def prepare_dataloaders(x_train_path: str, y_train_path: str, split_path: str,
                        augment_mode: str, n_augs: int, random_state: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([Normalizer(), PadderTruncator()])
    train_dataset = ECGAugmentDataset(x_train_path, y_train_path, dataset="train", transform=transform,
                                      train_val_split_csv=split_path, augment=True, augment_mode=augment_mode,
                                      n_augs=n_augs, random_state=random_state*2)
    val_dataset = ECGAugmentDataset(x_train_path, y_train_path, dataset="val", transform=transform,
                                    train_val_split_csv=split_path, augment=False, random_state=random_state*3)   
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return loader_train, loader_val

def run_comparison_data_augmentation(
        augment_params: dict,
        optim_params: dict,
        model_params: dict,
        res_out_path: str,
        n_epochs: int=40,
        random_state: int=42,
        device: str="cpu",
        x_train_path: str="../../../data/X_train.zip",
        y_train_path: str="../../../data/y_train.csv",
        split_path: str="../../train_val_split_jakub.csv"
        ):
    
    weights = prepare_weights(split_path, device)
    res_dict = {}
    
    for i, aug_param_set in enumerate(list(it.product(*augment_params.values()))):
        augment_mode, n_augs = aug_param_set
        train_loader, val_loader = prepare_dataloaders(x_train_path, y_train_path, split_path, augment_mode, n_augs, random_state + i)
        
        model = ECGClassifier(**model_params)
        optimizer = Adam(model.parameters(), **optim_params)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        res_train = train_model(model, train_loader, val_loader, optimizer, loss_fn, device,
                                n_epochs=n_epochs, scheduler=scheduler, n_epochs_early_stop=5)
        
        res_dict[str(aug_param_set)] = res_train
        
    with open(res_out_path, 'w') as f:
        json.dump(res_dict, f)    

if __name__ == "__main__":
    augment_params = {
        "mode": ["mild", "moderate", "high"],
        "n_augs": [1, 2]
        }      
    
    model_params = {
        "channels": 64,
        "res_blocks": 10,
        "lstm_hidden_size": 64,
        "fc_hidden_dim": 64,
        "dropout_proba": 0.3
        }
    
    optim_params = {
        "lr": 0.001,
        "weight_decay": 1e-5
        }
    
    run_comparison_data_augmentation(augment_params, optim_params, model_params, "results_augment.json")
  