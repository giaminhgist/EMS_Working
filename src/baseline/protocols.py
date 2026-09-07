"""The two evaluation protocols used by all EMS-Baseline baselines.

Protocol 1 — official split:
    The original EMS protocol: Train_Valid.xlsx assigns the 160 labelled
    subjects to 4 folds (Set_0..Set_3). For each fold: train on the other 3,
    validate on the fold. Results are the mean ± std over the 4 folds.
    (Official test labels are withheld by the authors, so no test metrics
    can be computed — only validation metrics + saved test predictions.)

Protocol 2 — 120/40 subject split:
    The 160 labelled subjects are split (stratified, seed-controlled) into
    120 train/val subjects and 40 held-out test subjects. The 120 are
    further split 90/30 (stratified) for inner model selection (FNN early
    stopping); final metrics are computed on the 40-subject test set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PROCESSED, SEED


def load_metadata():
    meta = pd.read_csv(PROCESSED / "metadata.csv", index_col="subject_id")
    return meta


def train_subjects():
    meta = load_metadata()
    return meta[meta.partition == "train"].index.tolist()


def protocol1_folds():
    """Yield (fold_name, train_ids, val_ids) following the official 4 folds."""
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    folds = {}
    for name in ["Set_0", "Set_1", "Set_2", "Set_3"]:
        folds[name] = meta[meta.official_fold == name].index.tolist()
    for name, val_ids in folds.items():
        train_ids = [s for n, ids in folds.items() if n != name for s in ids]
        yield name, train_ids, val_ids


def protocol2_split(seed=SEED):
    """Stratified 120/40 split of the 160 train subjects (reproducible).

    Returns dict with train_val_ids (120), test_ids (40), and an inner
    stratified 90/30 split of the 120.
    """
    meta = load_metadata()
    meta = meta[meta.partition == "train"]
    ids = meta.index.to_numpy()
    y = meta.label.to_numpy()
    tv, te = train_test_split(ids, test_size=40, stratify=y, random_state=seed)
    y_tv = meta.loc[tv].label.to_numpy()
    tr, va = train_test_split(tv, test_size=30, stratify=y_tv, random_state=seed)
    return {"train_val_ids": list(tv), "test_ids": list(te),
            "inner_train_ids": list(tr), "inner_val_ids": list(va)}
