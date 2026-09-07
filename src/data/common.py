"""Shared data utilities for the normative-gaze research codebase.

Paths, subject/fold utilities, and the two evaluation protocols. Reuses the
processed dataset (stimulus features + metadata) produced by src/preprocess.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "processed_dataset"
RAW = ROOT / "original_dataset" / "EMS"
OUTPUTS = ROOT / "outputs"

NUM_STIMULI = 100
NUM_FEATURES = 45
SCREEN_W, SCREEN_H = 1024.0, 768.0

SEEDS_HELDOUT = [42, 2024, 2026]


def load_metadata():
    meta = pd.read_csv(PROCESSED / "metadata.csv", index_col="subject_id")
    return meta


def train_subject_ids():
    meta = load_metadata()
    return meta[meta.partition == "train"].index.tolist()


def official_folds():
    """Yield (fold_name, train_ids, val_ids) for the official 4-fold protocol."""
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    folds = {}
    for name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
        folds[name] = meta[meta.official_fold == name].index.tolist()
    for name, val_ids in folds.items():
        train_ids = [s for n, ids in folds.items() if n != name for s in ids]
        yield name, train_ids, val_ids


def protocol2_split(seed):
    """Stratified 120/40 split of the 160 labelled subjects (reproducible)."""
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    ids = meta.index.to_numpy()
    y = meta.label.to_numpy()
    tv, te = train_test_split(ids, test_size=40, stratify=y, random_state=seed)
    return {"train_val_ids": list(tv), "test_ids": list(te)}


def labels_of(subject_ids):
    meta = load_metadata()
    return meta.loc[subject_ids, "label"].to_numpy(dtype=np.int64)


def load_stimulus_features(partition="train"):
    """DataFrame (subject_id, image) x 45 features; NaN for missing pairs."""
    return pd.read_pickle(PROCESSED / f"stimulus_features_{partition}.pkl")


def image_list():
    feat = load_stimulus_features("train")
    return sorted(feat.index.get_level_values(1).unique())


def category_of(image):
    prefix = image.split("_")[0]
    mapping = {"act": "social", "por": "social", "soc": "social",
               "ind": "natural", "land": "natural", "outman": "natural",
               "sat": "natural", "art": "synthetic", "cat": "synthetic",
               "pat": "synthetic", "low": "manipulated", "mood": "manipulated",
               "noi": "manipulated", "patch": "manipulated",
               "rand": "manipulated"}
    return mapping[prefix]


def subject_label(sid):
    return 0 if int(sid) < 200 else 1
