import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import copy
from torch import nn
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torcheval.metrics.functional import multiclass_f1_score
from b_modeling.dataset import Normalizer, PadderTruncator, ECGDataset
from b_modeling.model import ECGClassifier

def train_model(model: nn.Module, train_loader, val_loader, optimizer, loss_fn, device,
                n_epochs=10, n_epochs_early_stop=3, scheduler=None, path_to_save_model: str|None=None):
    
    model.to(device)
    early_stopping = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    
    best_val_f1_score = 0.0
    best_model_state = None
    
    for epoch in range(1, n_epochs + 1):
        
        print(f"Epoch: {epoch}/{n_epochs}")
        # training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # optional
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            y_true.append(targets)
            y_pred.append(preds)

            train_loop.set_postfix(loss=loss.item(), acc=correct/total)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = correct / total
        train_f1 = multiclass_f1_score(torch.cat(y_pred), torch.cat(y_true), num_classes=4, average="macro")
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Train F1: {train_f1:.4f}')
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        train_f1s.append(train_f1.item())
        
        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        val_loop = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                y_true.append(targets)
                y_pred.append(preds)
                
                val_loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = correct / total
        val_f1 = multiclass_f1_score(torch.cat(y_pred), torch.cat(y_true), num_classes=4, average="macro")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Val F1: {val_f1:.4f}")
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        val_f1s.append(val_f1.item())
        
        scheduler.step()
        
        if best_val_f1_score < val_f1.item():
            best_val_f1_score = val_f1.item()
            best_model_state = copy.deepcopy(model.state_dict())
        
        if epoch > 1:
            if avg_val_loss >= val_losses[-2]:
                early_stopping += 1
                if early_stopping == n_epochs_early_stop:
                    break
            else:
                early_stopping = 0
    
    if path_to_save_model:
        torch.save(best_model_state, path_to_save_model)

    res = {
        'train': {'losses': train_losses, 'accs': train_accs, 'f1s': train_f1s},
        'val': {'losses': val_losses, 'accs': val_accs, 'f1s': val_f1s}
        }
    return res
        

if __name__ == "__main__":
    
    # define paths
    x_train_path = "../../../data/X_train.zip"
    y_train_path = "../../../data/y_train.csv"
    split_path = "../../train_val_split_jakub.csv"
    
    # calcluate weights
    split_df = pd.read_csv(split_path)
    train_df = split_df[split_df['split'] == 'train']
    weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=np.array(train_df['label'])), dtype=torch.float32)
    
    # first experiment
    transforms = transforms.Compose([Normalizer(), PadderTruncator()])
    train_dataset = ECGDataset(x_train_path, y_train_path, dataset='train', transform=transforms, train_val_split_csv=split_path)
    val_dataset = ECGDataset(x_train_path, y_train_path, dataset='val', transform=transforms, train_val_split_csv=split_path)
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    model = ECGClassifier()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    train_model(model, loader_train, loader_val, optimizer, loss_fn, "cpu")
