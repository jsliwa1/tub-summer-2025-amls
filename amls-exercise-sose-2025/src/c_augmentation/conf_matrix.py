import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pred_df = pd.read_csv("../../predictions/base_val.csv")
y_pred = pred_df[["pred"]]
split_df = pd.read_csv("../../train_val_split_jakub.csv", index_col=0)
y_true = split_df[split_df["split"] == "val"][["label"]]
merged_df = y_true.join(y_pred, how="inner")

class_names = ["Normal (0)", "AF (1)", "Other (2)", "Noisy (3)"]
cm = confusion_matrix(merged_df["label"], merged_df["pred"])
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(6, 5), dpi=200)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

def calc_f1_perf_metric(y_true: pd.Series, y_pred: pd.Series) -> float:
    conf_mat = confusion_matrix(y_true, y_pred)
    F1s = []
    for i in range(conf_mat.shape[0]):
        F1 = 2 * conf_mat[i, i] / (np.sum(conf_mat[i, :]) + np.sum(conf_mat[:, i]))
        F1s.append(F1)
    return np.mean(F1s)
