import sys
import json
import torch
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from b_modeling.tuning import set_seed, prepare_dataloaders_and_weigths
from b_modeling.training import train_model
from b_modeling.model import ECGClassifier
from b_modeling.dataset import Normalizer, PadderTruncator, ECGDataset
from c_augmentation.comparison import prepare_weights, prepare_dataloaders
from config import *

def run_model_training(
        path_to_x_train: str = PATH_TO_X_TRAIN,
        path_to_y_train: str = PATH_TO_Y_TRAIN,
        path_to_train_val_split: str = PATH_TO_TRAIN_VAL_SPLIT,
        res_out_path: str = RES_OUT_PATH,
        model_out_path: str = MODEL_OUT_PATH,
        device: str = DEVICE,
        augment: bool = AUGMENT,
        epoch_params: dict = EPOCH_PARAMS,
        optim_params: dict = OPTIM_PARAMS,
        scheduler_params: dict = SCHEDULER_PARAMS,
        model_params: dict = MODEL_PARAMS,
        augment_params: dict = AUGMENT_PARAMS,
        random_state: int = RANDOM_STATE
        ):    
    
    set_seed(random_state)
    
    if augment:
        mode = augment_params["mode"]
        n_augs = augment_params["n_augs"]
        weights = prepare_weights(path_to_train_val_split, device)
        train_loader, val_loader = prepare_dataloaders(path_to_x_train, path_to_y_train,
                                                       path_to_train_val_split, mode, n_augs, random_state)
    else:
        train_loader, val_loader, weights = prepare_dataloaders_and_weigths(path_to_x_train, path_to_y_train,
                                                                            path_to_train_val_split, device)
    model = ECGClassifier(**model_params)
    optimizer = Adam(model.parameters(), **optim_params)
    scheduler = StepLR(optimizer, **scheduler_params)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    res = train_model(
            model, train_loader, val_loader, optimizer, loss_fn, device,
            n_epochs=epoch_params["n_epochs"], n_epochs_early_stop=epoch_params["n_epochs_early_stop"],
            scheduler=scheduler, path_to_save_model=model_out_path
        )
    
    with open(res_out_path, 'w') as f:
        json.dump(res, f)
        
def run_model_inference(
        path_to_x_test: str = PATH_TO_X_TEST,
        path_to_model_state: str = PATH_TO_MODEL_STATE,
        pred_out_path: str = PRED_OUT_PATH,
        device: str = DEVICE,
        model_params: dict = MODEL_PARAMS
        ):
    
    transform = transforms.Compose([Normalizer(), PadderTruncator()])
    test_set = ECGDataset(path_to_x_test, dataset="test", transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    model = ECGClassifier(**model_params)
    if device == "cuda":
        model.load_state_dict(torch.load(path_to_model_state))
    elif device == "cpu":
        model.load_state_dict(torch.load(path_to_model_state, map_location=torch.device('cpu')))
    else:
        raise ValueError(f"Device must be either cpu or cuda, but was: {device}.")
    model.eval()
    model.to(device)
    
    preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            preds.append(pred)
            
    preds = torch.cat(preds, dim=0).cpu().numpy()
    df = pd.DataFrame(preds, columns=["pred"])
    df.to_csv(pred_out_path)

if __name__ == "__main__":
    args = sys.argv[1:]
    arg = args[0].lower()
    if arg == "train":
        run_training = True
    elif arg == "test":
        run_training = False
    else:
        raise ValueError(f"Argument needs to be either 'train' or 'test' for training or testing, respectively. Was: {arg}.")
    if run_training:
        run_model_training()
    else:
        run_model_inference()
