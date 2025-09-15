#import sys
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

#src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#if src_path not in sys.path:
#    sys.path.insert(0, src_path)

from data_parser.parser import read_zip_binary

PATH_TO_DATA_FOLDER = "../../../data/"

# load the data
X_train = read_zip_binary(PATH_TO_DATA_FOLDER + "X_train.zip")
X_test = read_zip_binary(PATH_TO_DATA_FOLDER + "X_test.zip")
y_train = pd.read_csv(PATH_TO_DATA_FOLDER + "y_train.csv", header=None)
y_train.columns = ['label']

# create a df representing data with columns: [label, length, signal]
dict_data = {'label': y_train['label'].tolist(), 'length': [len(x) for x in X_train], 'signal': X_train}
df = pd.DataFrame(dict_data)

# calculate summary statistics of time series length by class
time_stats = df.groupby('label')['length'].agg(['mean', 'std', 'max', 'median', 'min'])
all_time_stats = pd.DataFrame(df['length'].agg(['mean', 'std', 'max', 'median', 'min'])).T
full_time_stats = pd.concat([time_stats, all_time_stats])
full_time_stats.reset_index(drop=True, inplace=True)

# plot distributions of time series length by class
labels=['0: normal', '1: AF', '2: other', '3: noisy']
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
fig.suptitle("Distribution of time series length by class", fontsize=18)
for i in range(4):
    ax = axs[i // 2, i % 2]
    ax.hist(df[df['label'] == i]['length'], bins = 18, density=True)
    ax.set_ylim((0, 0.001))
    ax.set_xticks(ticks=np.arange(0, 21000, 3000), labels=np.arange(0, 21000, 3000), rotation=45)
    ax.set_title(labels[i])
plt.show()

# descriptive stats of time series per class
def calc_stats(signal):
    return pd.Series({
        'mean': np.mean(signal),
        'std': np.std(signal),
        'median': np.median(signal),
        'rms': np.sqrt(np.sum(np.array(signal) ** 2) / len(signal)),
        'min': min(signal),
        'max': max(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal)
        })

stats_df = df['signal'].apply(calc_stats)
df = pd.concat([df, stats_df], axis=1)
df['range'] = df['max'] - df['min']
final_stats = df.groupby('label')[['mean', 'std', 'median', 'rms', 'min', 'max', 'range', 'skew', 'kurtosis']].agg(['mean', 'std']).T
print(final_stats)

# visualize ECG time series for different classes
instance = True
fig_ylim = [(-750, 750), (-750, 750), (-750, 750), (-1000, 1000)]
indices = [(4500, 7500), (0, 3000), (0, 3000), (0, 3000)]
fig, axs = plt.subplots(4, 1, figsize=(20, 12), dpi=300)
for i in range(4):
    ecg = df[(df['label'] == i) & (df['length'] == 9000)].head(1)
    axs[i].plot(np.arange(1, 3001), ecg['signal'].tolist()[0][indices[i][0]:indices[i][1]])
    instance_mean = ecg['mean'].iloc[0] 
    instance_std = ecg['std'].iloc[0]
    class_mean = final_stats.loc[('mean', 'mean')][i]
    class_std = final_stats.loc[('std', 'mean')][i]
    mean = instance_mean if instance else class_mean
    std = instance_std if instance else class_std
    axs[i].axhline(y=mean, color='red', linestyle='--', linewidth=1)
    axs[i].axhline(y=mean + 3 * std, color='orange', linestyle='--', linewidth=1)
    axs[i].axhline(y=mean - 3 * std, color='orange', linestyle='--', linewidth=1)
    axs[i].set_ylim(fig_ylim[i])
plt.show()


# create boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
sns.boxplot(x='label', y='std', data=df, ax=axes[0], showfliers=False)
axes[0].set_title('Std Dev')
sns.boxplot(x='label', y='range', data=df, ax=axes[1], showfliers=False)
axes[1].set_title('Range')
sns.boxplot(x='label', y='skew', data=df, ax=axes[2], showfliers=False)
axes[2].set_title('Skew')
plt.tight_layout()
plt.show()




