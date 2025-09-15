import json
import ast
import pandas as pd
from b_modeling.tuning_analysis import plot_training_stats

def results_to_df(path_to_results_folder: str="../../augmentation_comparison_results/") -> pd.DataFrame:
    filename = "results_augmentation_final.json"
    file = path_to_results_folder + filename
    with open(file, "r") as f:
        res = json.load(f)
    df = pd.DataFrame(columns=["mode", "n_augs", "split", "metric", "epoch", "value"])
    
    for param_set in res.keys():
        mode, n_augs = ast.literal_eval(param_set)
        for split in res[param_set].keys():
            for metric in res[param_set][split].keys():
                metric_val = "loss" if metric == "losses" else metric[:-1]
                for i, val in enumerate(res[param_set][split][metric]):
                    if metric != "accs":
                        df.loc[len(df)] = {
                            "mode": mode,
                            "n_augs": n_augs,
                            "split": split,
                            "metric": metric_val,
                            "epoch": i,
                            "value": val
                            }
    return df

def _find_epoch_max_val_f1(df: pd.DataFrame) -> pd.DataFrame:
    val_f1_df = df[(df["split"] == "val") & (df["metric"] == "f1")]
    return val_f1_df.groupby(['mode', 'n_augs'])[["epoch", "value"]].apply(
        lambda x: x.loc[x['value'].idxmax(), ['epoch', 'value']])[['epoch']].reset_index()

def prepare_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    val_f1_df = _find_epoch_max_val_f1(df)
    return df.merge(val_f1_df, on=["mode", "n_augs", "epoch"], how="inner")

def prepare_final_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    epochs = df.groupby(["mode", "n_augs"])[["epoch"]].agg("min")
    pivot = df.pivot_table(index=["mode", "n_augs"], columns=["split", "metric"], values="value")
    return pd.concat([pivot, epochs], axis=1)
    



df = results_to_df()
comp_df = prepare_comparison_df(df)
pivot_df = prepare_final_pivot_table(comp_df)

params = ["mode", "n_augs"]
param_values = [("mild", 1), ("mild", 2), ("moderate", 1), ("moderate", 2), ("high", 1), ("high", 2)]
plot_training_stats(df, params, param_values, epochs=40)