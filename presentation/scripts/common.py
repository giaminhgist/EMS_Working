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
FIG = PRES / "figure"
TAB = PRES / "tables"
CACHE = PRES / "cache"
SCRIPTS = PRES / "scripts"
OUT_PROP = REPO / "outputs" / "proposal"
PROCESSED = REPO / "processed_dataset"
RAW = REPO / "original_dataset" / "EMS"

for d in [FIG, TAB, CACHE]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from data.common import (load_metadata, official_folds,  # noqa: E402
                         load_stimulus_features, image_list, category_of)
from trainer.metrics import compute_metrics  # noqa: E402

# --------------------------------------------------------------------- #
# Run metadata (must match experiment/matrix.json + docs/model_spec.md)
# --------------------------------------------------------------------- #
FOLDS = ["Set_0", "Set_1", "Set_2", "Set_3"]

# ablation -> (deviation, comparator, pool, lambda_norm)
# (configs used by the remaining figures)
ABLATION_META = {
    "mlp_attn":    dict(deviation="learned", comparator="mlp", pool="attention", lambda_norm=0.0),
    "mlp_norm01":  dict(deviation="learned", comparator="mlp", pool="attention", lambda_norm=0.1),
    "z_mean":      dict(deviation="z", pool="mean", lambda_norm=0.0),
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
