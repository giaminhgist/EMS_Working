"""Six evaluation metrics at fixed threshold 0.5 + AUC (threshold-free).

EMS convention: SZ = positive class (label 1).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5


def compute_metrics(y_true, y_score, threshold=THRESHOLD):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)          # sensitivity / recall
    spe = tn / max(tn + fp, 1)          # specificity
    bal = (sen + spe) / 2.0             # balanced accuracy
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * sen / max(sen + pre, 1e-12)
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    return {"acc": acc, "auc": auc, "balanced_acc": bal, "sen": sen,
            "spe": spe, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summarize(list_of_metrics):
    """Mean ± std over a list of metric dicts."""
    out = {}
    for key in ["acc", "auc", "balanced_acc", "sen", "spe", "f1"]:
        vals = [m[key] for m in list_of_metrics if key in m and not np.isnan(m[key])]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": float("nan"), "std": float("nan")}
    return out
