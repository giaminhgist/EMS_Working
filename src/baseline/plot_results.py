"""Generate result figures: AUC comparison bars + ROC curves per protocol.

Usage:
    python plot_results.py            # reads docs/baseline/results/summary.csv + preds

Outputs -> docs/baseline/results/figures/
    auc_comparison_P1.png, auc_comparison_P2.png
    roc_P1.png, roc_P2.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

METHOD_LABELS = {
    "svm_rbf": "SVM-RBF", "svm_linear": "SVM-Lin", "rf": "RF", "qda": "QDA",
    "gnb": "GaussianNB", "lr": "LogReg-L2", "lr_l1": "LogReg-L1(concat)",
    "knn": "KNN", "fnn": "FNN-agg", "fnn_cat": "FNN-catagg",
}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_summary():
    s = pd.read_csv(RESULTS / "summary.csv")
    return s


def plot_auc_comparison(summary):
    for protocol, title in [("P1", "Protocol P1 — official 4-fold CV (val folds, n=160)"),
                            ("P2", "Protocol P2 — 120/40 subject split (test 40, 3 seeds)")]:
        sub = summary[summary.protocol == protocol]
        methods = sub.method.unique()
        means = [sub[sub.method == m]["auc"].mean() for m in methods]
        stds = [sub[sub.method == m]["auc"].std() for m in methods]
        order = np.argsort(means)[::-1]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(range(len(methods)), [means[i] for i in order],
               yerr=[stds[i] for i in order], capsize=4, alpha=0.85,
               color=[COLORS[i % 10] for i in order])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods[order]],
                           rotation=25, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC (mean ± std)")
        ax.set_title(title)
        for i, j in enumerate(order):
            ax.text(i, means[j] + stds[j] + 0.008, f"{means[j]:.3f}",
                    ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"auc_comparison_{protocol}.png")
        plt.close(fig)
        print(f"saved {FIG_DIR / f'auc_comparison_{protocol}.png'}")


def collect_scores_p1(method_rep_dirs):
    """Concatenate val predictions over the 4 folds -> (y, scores) per method."""
    meta = pd.read_csv(Path("/root/EMS-Minh/processed_dataset/metadata.csv"))
    meta = meta[meta.partition == "train"].set_index("subject_id")
    out = {}
    for mdir in method_rep_dirs:
        preds = pd.read_csv(mdir / "seed42" / "val_preds.csv")
        y = meta.loc[preds.subject_id, "label"].values
        out[mdir.parent.name] = (y, preds.prob.values)
    return out


def collect_scores_p2(method_rep_dirs, seed=42):
    out = {}
    for mdir in method_rep_dirs:
        preds = pd.read_csv(mdir / f"seed{seed}" / "test_preds.csv")
        out[mdir.parent.name] = (preds.label.values, preds.prob.values)
    return out


def plot_roc(curves, fname, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for i, (name, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i % 10], lw=1.8,
                label=f"{METHOD_LABELS.get(name, name)} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"saved {FIG_DIR / fname}")


def main():
    summary = load_summary()
    plot_auc_comparison(summary)
    dirs = sorted((RESULTS / "P1").glob("*__*"))
    c1 = collect_scores_p1(dirs)
    plot_roc(c1, "roc_P1.png", "ROC — protocol P1 (val fold predictions pooled over 4 folds)")
    dirs2 = sorted((RESULTS / "P2").glob("*__*"))
    c2 = collect_scores_p2(dirs2, seed=42)
    plot_roc(c2, "roc_P2.png", "ROC — protocol P2 (40-subject test, seed 42)")


if __name__ == "__main__":
    main()
