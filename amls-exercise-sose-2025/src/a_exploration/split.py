import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_parser.parser import read_zip_binary
from sklearn.model_selection import train_test_split

def load_train_data(path_to_data_folder: str) -> (list[list[int]], pd.DataFrame):
    X_train = read_zip_binary(path_to_data_folder + "X_train.zip")
    y_train = pd.read_csv(path_to_data_folder + "y_train.csv", header=None)
    y_train.columns = ["label"]
    return X_train, y_train

def create_raw_df(X_train: list[list[int]], y_train: pd.DataFrame) -> pd.DataFrame:
    dict_data = {'label': y_train['label'].tolist(), 'length': [len(x) for x in X_train], 'signal': X_train}
    return pd.DataFrame(dict_data)

def train_val_split(
        X_train: list[list[int]],
        y_train: pd.DataFrame,
        val_fraction: float,
        random_state: int = 10000
        ) -> (pd.DataFrame, pd.DataFrame):
    df = create_raw_df(X_train, y_train)    
    conditions = [df['length'] < 9000, df['length'] == 9000, df['length'] > 9000]
    qs = [8, 1, 4]
    train_idx = pd.Index([])
    val_idx = pd.Index([])
    for i, condition in enumerate(conditions):
        sub_df = df[condition].copy()
        if i == 2:
            dummy_row = sub_df[(sub_df['label'] == 1) & (sub_df['length'] > 18000)]
            dummy_row.index = pd.Index([len(df)])
            sub_df = pd.concat([sub_df, dummy_row])
        sub_df['length_bin'] = pd.qcut(sub_df['length'], q=qs[i], duplicates='drop')
        sub_df['stratify_key'] = sub_df['label'].astype(str) + '_' + sub_df['length_bin'].astype(str)
        train_df, val_df = train_test_split(sub_df, test_size=val_fraction, stratify=sub_df['stratify_key'], random_state=random_state)
        train_idx = train_idx.union(train_df.index)
        val_idx = val_idx.union(val_df.index)
        if  i == 2:
            train_idx = train_idx.difference(pd.Index([len(df)]))
            val_idx = val_idx.difference(pd.Index([len(df)]))
    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]
    return df_train, df_val

def _visualize_distributions_split(train_df: pd.DataFrame, val_df: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    fig.suptitle("Distribution of time series length by class after train-val split", fontsize=18)
    labels=['0: normal', '1: AF', '2: other', '3: noisy']
    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.hist(train_df[train_df['label'] == i]['length'], bins = 16, density=True, label="train", alpha=0.5)
        ax.hist(val_df[val_df['label'] == i]['length'], bins = 16, density=True, label="val", alpha=0.5)
        ax.set_ylim((0, 0.001))
        ax.set_xticks(ticks=np.arange(0, 21000, 3000), labels=np.arange(0, 21000, 3000), rotation=45)
        ax.set_title(labels[i])
        ax.legend()
    plt.show()
    
def _save_train_val_split_to_csv(train_df: pd.DataFrame, val_df: pd.DataFrame, path_to_csv_split: str):
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    full_df = pd.concat([train_df, val_df])
    full_df = full_df.sort_index()
    full_df[['label', 'length', 'split']].to_csv(path_to_csv_split)

if __name__ == "__main__":
    PATH_TO_DATA_FOLDER = "../../../data/"
    PATH_TO_SPLIT_CSV = "../../train_val_split_jakub.csv"
    X_train, y_train = load_train_data(PATH_TO_DATA_FOLDER)
    seed = 2453245
    train_df, val_df = train_val_split(X_train, y_train, 0.2, random_state=seed)
    _visualize_distributions_split(train_df, val_df)
    _save_train_val_split_to_csv(train_df, val_df, PATH_TO_SPLIT_CSV)
    
    