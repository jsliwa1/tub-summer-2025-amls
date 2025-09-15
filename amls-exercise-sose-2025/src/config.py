# params
AUGMENT = True
RANDOM_STATE = 42
DEVICE = "cpu"

# model
MODEL_PARAMS = {
        "res_blocks": 10,
        "channels": 64,
        "lstm_hidden_size": 64,
        "fc_hidden_dim": 64,
        "dropout_proba": 0.3
    }

# optimizer
OPTIM_PARAMS = {
        "lr": 0.001,
        "weight_decay": 1e-5
    }

# scheduler
SCHEDULER_PARAMS = {
        "step_size": 15,
        "gamma": 0.1
    }

# epochs
EPOCH_PARAMS = {
        "n_epochs": 1,
        "n_epochs_early_stop": 5
    }

# augmentations
AUGMENT_PARAMS = {
        "mode": "mild",
        "n_augs": 2
    }

# paths
PATH_TO_X_TRAIN = "../data/X_train.zip"
PATH_TO_Y_TRAIN = "../data/y_train.csv"
PATH_TO_X_TEST = "../data/X_test.zip"
PATH_TO_TRAIN_VAL_SPLIT = "train_val_split_jakub.csv"
RES_OUT_PATH = "results/results.json"
MODEL_OUT_PATH = "models/best_model.pt"
PATH_TO_MODEL_STATE = "models/base_model.pt"
PRED_OUT_PATH = "predictions/pred.csv"
