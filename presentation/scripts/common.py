"""Shared utilities for the presentation figure suite.

Resolves the repo root from this file, provides provenance-safe access to
run artifacts (config-checked), metric recomputation, a consistent visual
style (colorblind-friendly, 16:9-friendly, light background), and figure
saving at 300 dpi PNG + vector SVG.

Conventions (match src/trainer/metrics.py):
  - SZ is the positive class (label 1); threshold fixed at 0.5.
  - Fold-mean metrics = mean of per-fold metrics (reported P1 numbers).
  - Pooled metrics = metrics computed on pooled out-of-fold probabilities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
assert (REPO / "processed_dataset").exists(), f"repo root not found at {REPO}"
PRES = REPO / "presentation"
FIG = PRES / "figures"
TAB = PRES / "tables"
CACHE = PRES / "cache"
SCRIPTS = PRES / "scripts"
OUT_PROP = REPO / "outputs" / "proposal"
OUT_EVAL = REPO / "outputs" / "evaluation"
BASE_RES = REPO / "docs" / "baseline" / "results"
PROCESSED = REPO / "processed_dataset"
RAW = REPO / "original_dataset" / "EMS"

for d in [FIG, TAB, CACHE]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from data.common import (load_metadata, official_folds, protocol2_split,  # noqa: E402
                         load_stimulus_features, image_list, category_of)
from trainer.metrics import compute_metrics  # noqa: E402

# --------------------------------------------------------------------- #
# Run metadata (must match experiment/matrix.json + docs/model_spec.md)
# --------------------------------------------------------------------- #
SEEDS = [42, 1234, 2024, 2026, 7]
FOLDS = ["Set_0", "Set_1", "Set_2", "Set_3"]
SEEDS_HELDOUT = [42, 2024, 2026]

# ablation -> (deviation, comparator, pool, lambda_norm)
ABLATION_META = {
    "mlp_attn":    dict(deviation="learned", comparator="mlp", pool="attention", lambda_norm=0.0),
    "mlp_norm01":  dict(deviation="learned", comparator="mlp", pool="attention", lambda_norm=0.1),
    "sub_attn":    dict(deviation="learned", comparator="sub", pool="attention", lambda_norm=0.0),
    "zsub_attn":   dict(deviation="learned", comparator="zsub", pool="attention", lambda_norm=0.0),
    "mlp_mean":    dict(deviation="learned", comparator="mlp", pool="mean", lambda_norm=0.0),
    "mlp_deepset": dict(deviation="learned", comparator="mlp", pool="deepset", lambda_norm=0.0),
    "z_mean":      dict(deviation="z", pool="mean", lambda_norm=0.0),
    "diff_mean":   dict(deviation="diff", pool="mean", lambda_norm=0.0),
    "mahal_mean":  dict(deviation="mahal", pool="mean", lambda_norm=0.0),
}

ABLATION_LABELS = {
    "mlp_attn": "learned + attn (main proposal)",
    "mlp_norm01": "learned + attn + λ=0.1",
    "sub_attn": "latent sub + attn",
    "zsub_attn": "latent zsub + attn",
    "mlp_mean": "learned + mean",
    "mlp_deepset": "learned + deepset",
    "z_mean": "hard z-dev (fixed)",
    "diff_mean": "hard diff (fixed)",
    "mahal_mean": "hard z+mahal (fixed)",
}

BASELINE_LABELS = {
    "svm_rbf": "SVM-RBF", "lr": "LogReg-L2", "rf": "Random Forest",
    "fnn": "FNN-agg", "svm_linear": "SVM-Lin", "knn": "KNN",
    "gnb": "GaussianNB", "qda": "QDA", "lr_l1": "LogReg-L1 (concat)",
    "fnn_cat": "FNN-catagg",
}

# --------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------- #
HC_COLOR = "#0072B2"      # blue
SZ_COLOR = "#D55E00"      # vermillion
REF_COLOR = "#949494"     # grey (normative reference)
LEARNED_COLOR = "#009E73"  # green (learned path)
HARD_COLOR = "#CC79A7"    # pink (hard-deviation path)
COLOR_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#999999", "#8C510A"]

CAT_COLORS = {"social": "#0072B2", "natural": "#009E73",
              "synthetic": "#E69F00", "manipulated": "#CC79A7"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)


def savefig(fig, outdir: Path, name: str, tight=True):
    """Save figure as 300-dpi PNG and vector SVG."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"{name}.svg", bbox_inches="tight")
    else:
        fig.savefig(outdir / f"{name}.png", dpi=300)
        fig.savefig(outdir / f"{name}.svg")
    plt.close(fig)
    print(f"  saved {outdir / name}.png/.svg")


def wide_fig(nrows=1, ncols=1, w=4.6, h=3.2, **kw):
    """Figure sized for a 16:9 slide (small multiples in w x h inches)."""
    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), **kw)


# --------------------------------------------------------------------- #
# Run-dir resolution (provenance-safe: config must match)
# --------------------------------------------------------------------- #
def find_run_dirs(ablation, seed, fold):
    """All run dirs matching prefix; sorted by timestamp (oldest first)."""
    pat = f"{ablation}__seed{seed}__fold{fold}__*"
    dirs = sorted(OUT_PROP.glob(pat))
    return dirs


def pick_run_dir(ablation, seed, fold):
    """Pick the run dir whose config.json matches ABLATION_META; newest wins."""
    meta = ABLATION_META[ablation]
    defaults = {"deviation": "learned", "comparator": "mlp", "pool": "attention",
                "lambda_norm": 0.0}
    candidates = []
    for d in find_run_dirs(ablation, seed, fold):
        cfg = json.loads((d / "config.json").read_text())
        extra = cfg.get("extra", {})
        ok = all(extra.get(k, defaults[k]) == v for k, v in meta.items()
                 if k in ("deviation", "comparator", "pool", "lambda_norm"))
        if ok:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"no run dir for {ablation} seed{seed} fold {fold}")
    return candidates[-1]


def load_allfolds_summary(ablation, seed):
    p = OUT_PROP / f"{ablation}__seed{seed}__allfolds_summary.json"
    return json.loads(p.read_text())


def load_run_preds(ablation, seed):
    """Pooled out-of-fold validation predictions for one (ablation, seed).

    Returns DataFrame [subject_id, label, prob, fold] for the 160 subjects.
    """
    rows = []
    for fold in FOLDS:
        d = pick_run_dir(ablation, seed, fold)
        df = pd.read_csv(d / "predictions.csv")
        df["fold"] = fold
        rows.append(df)
    preds = pd.concat(rows, ignore_index=True)
    preds = preds.sort_values("subject_id").reset_index(drop=True)
    return preds


def metrics_from_preds(df):
    """Per-fold metrics + pooled metrics from a predictions DataFrame."""
    out = {}
    for fold in FOLDS:
        sub = df[df.fold == fold]
        out[fold] = compute_metrics(sub.label.values, sub.prob.values)
    out["fold_mean"] = {k: float(np.mean([out[f][k] for f in FOLDS]))
                        for k in ["acc", "auc", "balanced_acc", "sen", "spe", "f1"]}
    out["fold_std"] = {k: float(np.std([out[f][k] for f in FOLDS]))
                       for k in ["acc", "auc", "balanced_acc", "sen", "spe", "f1"]}
    out["pooled"] = compute_metrics(df.label.values, df.prob.values)
    return out


def seed_metrics(ablation, metric="auc", seeds=SEEDS):
    """Fold-mean metric per seed + mean/std over seeds (the reported P1 number)."""
    vals = {}
    for s in seeds:
        summ = load_allfolds_summary(ablation, s)
        vals[s] = summ["mean_metrics"][metric]["mean"]
    return vals, float(np.mean(list(vals.values()))), float(np.std(list(vals.values())))


def verify_p1_recompute(ablation):
    """Recompute fold-mean AUC/Acc from predictions.csv and compare to summaries."""
    rows = []
    for s in SEEDS:
        summ = load_allfolds_summary(ablation, s)
        preds = load_run_preds(ablation, s)
        m = metrics_from_preds(preds)
        rows.append(dict(seed=s,
                         reported_auc=summ["mean_metrics"]["auc"]["mean"],
                         recomputed_auc=m["fold_mean"]["auc"],
                         reported_acc=summ["mean_metrics"]["acc"]["mean"],
                         recomputed_acc=m["fold_mean"]["acc"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Baseline access (docs/baseline/results/)
# --------------------------------------------------------------------- #
def baseline_summary_df():
    return pd.read_csv(BASE_RES / "summary.csv")


def baseline_p1_preds(method):
    """Pooled out-of-fold val predictions (160 rows); label joined from metadata.

    The committed val_preds.csv has no label/fold columns -> join by subject_id.
    """
    d = BASE_RES / "P1" / f"{method}__agg" / "seed42"
    preds = pd.read_csv(d / "val_preds.csv")
    meta = load_metadata()
    preds["label"] = preds["subject_id"].map(meta["label"])
    preds["fold"] = preds["subject_id"].map(meta["official_fold"])
    assert preds["label"].notna().all(), "label join failed for P1 baseline preds"
    return preds


def baseline_p2_preds(method, seed):
    d = BASE_RES / "P2" / f"{method}__agg" / f"seed{seed}"
    return pd.read_csv(d / "test_preds.csv")


# --------------------------------------------------------------------- #
# Model reconstruction from checkpoint
# --------------------------------------------------------------------- #
import torch  # noqa: E402

from proposal.model import NormativeModel, NormativeDataset  # noqa: E402


def load_checkpoint_model(ablation, seed, fold, device="cpu"):
    """Rebuild the model from a run's config and best.pt (bank buffers kept)."""
    meta = ABLATION_META[ablation]
    run_dir = pick_run_dir(ablation, seed, fold)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    dropout = cfg.get("dropout", 0.3)
    if meta["deviation"] == "learned":
        model = NormativeModel(comparator=meta["comparator"], pool=meta["pool"],
                               lambda_norm=meta["lambda_norm"], dropout=dropout,
                               deviation="learned")
    else:
        model = NormativeModel(deviation=meta["deviation"], pool=meta["pool"],
                               dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, run_dir, cfg


def make_datasets(ablation, seed, fold, stim_subset=None):
    """(train_ds, val_ds) with leakage-safe stats, as used in training."""
    meta = ABLATION_META[ablation]
    train_ids, val_ids = None, None
    for name, tr, va in official_folds():
        if name == fold:
            train_ids, val_ids = tr, va
    train_ds = NormativeDataset(train_ids, train_ids, deviation=meta["deviation"],
                                stim_subset=stim_subset)
    val_ds = NormativeDataset(val_ids, train_ids, deviation=meta["deviation"],
                              stim_subset=stim_subset)
    return train_ds, val_ds, train_ids, val_ids


# --------------------------------------------------------------------- #
# Feature groups (src/features.py order)
# --------------------------------------------------------------------- #
FEATURE_GROUPS = {
    "spa_pos": ["spa_fix_count", "spa_mean_x", "spa_mean_y", "spa_std_x", "spa_std_y",
                "spa_dispersion", "spa_bbox_area"],
    "spa_center": ["spa_center_dist_mean", "spa_center_dist_std", "spa_center_frac",
                   "spa_q1", "spa_q2", "spa_q3", "spa_q4", "spa_entropy",
                   "spa_max_grid_frac", "spa_skew_x"],
    "geo": ["geo_scanpath_len", "geo_sacc_amp_mean", "geo_sacc_amp_std", "geo_sacc_amp_max",
            "geo_dx_mean", "geo_dy_mean", "geo_angle_var", "geo_revisit_rate",
            "geo_nn_dist_mean", "geo_hull_area"],
    "tem": ["tem_dur_mean", "tem_dur_std", "tem_dur_total", "tem_dur_max", "tem_first_dur",
            "tem_last_dur", "tem_ifi_mean", "tem_ifi_std", "tem_velocity_mean",
            "tem_fix_rate", "tem_trans_entropy"],
    "pup": ["pup_mean", "pup_std", "pup_min", "pup_max", "pup_median", "pup_slope",
            "pup_first_last_diff"],
}
FEATURE_NAMES = [f for g in FEATURE_GROUPS.values() for f in g]
assert len(FEATURE_NAMES) == 45

GROUP_LABELS = {"spa_pos": "Spatial — position & dispersion",
                "spa_center": "Spatial — center & regions",
                "geo": "Scanpath geometry",
                "tem": "Temporal",
                "pup": "Pupil"}


def feature_group_of(name):
    for g, names in FEATURE_GROUPS.items():
        if name in names:
            return g
    raise KeyError(name)


def load_subject_features(partition="train"):
    """DataFrame (subject_id, image) x 45 features."""
    return load_stimulus_features(partition)


def subject_level_features(partition="train"):
    """Subject x 45 table: per-subject mean over valid stimuli (NaN-safe)."""
    feat = load_subject_features(partition)
    subj = feat.groupby(level="subject_id").mean()
    return subj
