"""Evaluation metrics for the EMS benchmark (paper Section V-A).

All classification metrics are computed at a FIXED decision threshold of 0.5
on the sigmoid/score outputs; AUC is threshold-free.
"""
import numpy as np
from sklearn import metrics as skm


def compute_metrics(y_true, y_score, threshold=0.5):
    """Return dict with Acc/Sen/Spe/Pre/F1/AUC at the given threshold.

    EMS convention: SZ (label 1) is the positive class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * sen / max(sen + pre, 1e-12)
    auc = float(skm.roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
    return {"acc": acc, "sen": sen, "spe": spe, "pre": pre, "f1": f1, "auc": auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_table_mean_std(list_of_metrics):
    """Average metric dicts -> {metric: (mean, std)}."""
    out = {}
    for key in ["acc", "sen", "spe", "pre", "f1", "auc"]:
        vals = [m[key] for m in list_of_metrics if key in m and not np.isnan(m[key])]
        out[key] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)
    return out
