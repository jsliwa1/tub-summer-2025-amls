import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results_to_dict(path_to_res_folder: str="../../parameter_tuning_results/",
                         params: list[tuple[int]]=[(5, 64), (5, 128), (10, 64), (10, 128)],
                         first_run: bool=True):
    results = {}
    for param in params:
        filename = f"results_{param[0]}_{param[1]}.json" if first_run else f"results_{param[0]}_blocks_{param[1]}_channels.json"
        file = path_to_res_folder + filename
        with open(file, 'r') as f:
            tmp_res = json.load(f)
        results = {**results, **tmp_res}
    return results      

def results_to_df(results: dict, iteration: int|None=None) -> pd.DataFrame:
    df = pd.DataFrame(columns=['lr', 'wd', 'channels', 'res_blocks', 'lstm_hidden',
                               'split', 'metric', 'epoch', 'value'])
    for param_set in results.keys():
        lr, wd, channels, res_blocks, lstm_hidden, _, _ = ast.literal_eval(param_set)
        splits = results[param_set].keys() if iteration is None else results[param_set][iteration].keys()
        for split in splits:
            metrics = results[param_set][split].keys() if iteration is None else results[param_set][iteration][split].keys()
            for metric in metrics:
                metric_val = "loss" if metric == "losses" else metric[:-1]
                values = results[param_set][split][metric] if iteration is None else results[param_set][iteration][split][metric]
                for i, val in enumerate(values):
                    df.loc[len(df)] = {'lr': lr, 'wd': wd, 'channels': channels,
                                       'res_blocks': res_blocks, 'lstm_hidden': lstm_hidden,
                                       'split': split, 'metric': metric_val, 
                                       'epoch': i, 'value': val}
    return df

def _find_epoch_max_val_f1(df: pd.DataFrame) -> pd.DataFrame:
    val_f1_df = df[(df['split'] == 'val') & (df['metric'] == 'f1')]
    return val_f1_df.groupby(['lr','wd','channels','res_blocks','lstm_hidden'])[['epoch','value']].apply(
        lambda x: x.loc[x['value'].idxmax(), ['epoch', 'value']])[['epoch']].reset_index()

def _find_epoch_min_val_loss(df: pd.DataFrame) -> pd.DataFrame:
    val_loss_df = df[(df['split'] == 'val') & (df['metric'] == 'loss')]
    return val_loss_df.groupby(['lr','wd','channels','res_blocks','lstm_hidden'])[['epoch','value']].apply(
        lambda x: x.loc[x['value'].idxmin(), ['epoch', 'value']])[['epoch']].reset_index()

def prepare_comparison_df(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    val_f1_df = _find_epoch_max_val_f1(df)
    val_loss_df = _find_epoch_min_val_loss(df)
    res_tmp_1 = df.merge(val_f1_df, on=params + ['epoch'], how='inner')
    res_tmp_2 = df.merge(val_loss_df, on=params + ['epoch'], how='inner')
    return res_tmp_1.merge(res_tmp_2, on=params + ['split', 'metric'], how='inner')

def prepare_final_pivot_table(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    epochs = df.groupby(params)[['epoch_x', 'epoch_y']].agg('min')
    pivot = df.pivot_table(index=params, columns=['split', 'metric'], values=['value_x', 'value_y'])
    return pd.concat([pivot, epochs], axis=1)
    
def plot_training_stats(df: pd.DataFrame, params: list[str], param_values: list[tuple], epochs: int=20):
    fig, axs = plt.subplots(2, len(param_values), figsize=(24, 8), dpi=300)
    idx_df = df.set_index(params)
    for i, param_set in enumerate(param_values):
        params_df = idx_df.loc[param_set]
        for j, metric in enumerate(['loss', 'f1']):
            ax = axs[j, i]
            train_series = params_df[(params_df['split'] == 'train') & (params_df['metric'] == metric)]['value']
            val_series = params_df[(params_df['split'] == 'val') & (params_df['metric'] == metric)]['value']
            ax.set_title(f"{metric} for {param_set}")
            ax.plot(np.arange(0, len(train_series)), train_series, label="train")
            ax.plot(np.arange(0, len(val_series)), val_series, label="val")
            ax.set_xlim((0, epochs))
            ax.set_ylim((0, 1.25)) if j == 0 else ax.set_ylim((0, 1))
            ax.set_xlabel("epoch")
            ax.set_ylabel(f"{metric}")
            ax.legend(loc="lower left") if j == 0 else ax.legend(loc="lower right")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def prepare_final_model_selection(params: list[str], args_tuples: list[tuple[int]]) -> pd.DataFrame:
    results = load_results_to_dict(params=args_tuples, first_run=False)
    iter_results_dfs = [results_to_df(results, str(i)) for i in range(1)]
    iter_comp_dfs = [prepare_comparison_df(df, params) for df in iter_results_dfs]
    iter_pivot_dfs = [prepare_final_pivot_table(df, params) for df in iter_comp_dfs]
    # some further merging and averaging logic for multiple runs
    return iter_pivot_dfs, iter_results_dfs
    
yesterday = False
today = False

if yesterday:
    params = ['lr', 'wd', 'channels', 'res_blocks', 'lstm_hidden']
    results = load_results_to_dict()
    df = results_to_df(results)
    comp_df = prepare_comparison_df(df, params)
    final_df = prepare_final_pivot_table(comp_df, params)
    
    param_values = [
        (0.001, 0.001, 32, 5, 64),
        (0.001, 1e-5, 32, 5, 64),
        (0.0001, 1e-5, 64, 5, 64),
        (0.001, 0.0001, 64, 5, 128),
        (0.001, 0.0001, 64, 10, 128)
        ]
    plot_training_stats(df, params, param_values)

if today:
    params = ['lr', 'wd', 'channels', 'res_blocks', 'lstm_hidden']
    args_tuples = [(5, 32), (5, 64), (7, 32), (7, 64), (10, 32), (10, 64)]
    pivot_df, res_df = prepare_final_model_selection(params, args_tuples)
    
    param_values = [
        (0.001, 0.0001, 32, 7, 64),
        (0.001, 1e-5, 32, 7, 64),
        (0.001, 0.0001, 32, 10, 64),
        (0.001, 1e-5, 32, 10, 64),
        (0.001, 0.0001, 64, 10, 64),
        (0.001, 1e-5, 64, 10, 64)
        ]
    plot_training_stats(res_df[0], params, param_values, epochs=40)




























