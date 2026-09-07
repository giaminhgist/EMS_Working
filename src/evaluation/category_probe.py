"""Eval 3 — Stimulus-category discriminability probing.

Hypothesis: SZ-specific gaze abnormality is stimulus-dependent, and the
normative deviation makes it visible at the SINGLE-stimulus level, especially
for social and manipulated categories (which the EMS paper designed to stress
schizophrenic viewing deficits).

Method: for each fold, compute out-of-fold stimulus-conditioned z-deviations
for the val subjects (norms from the training fold only). Pool the 4 folds ->
(160 subjects x 100 stimuli x 45) deviations. For every stimulus: a
per-subject logistic probe (stratified 4-fold inner CV over subjects) gives a
per-stimulus AUC; report mean AUC per stimulus category, comparing
z-deviation vs standardized raw features.

Usage (from EMS-Project/src):
    python evaluation/category_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import official_folds, category_of, image_list, OUTPUTS  # noqa: E402
from data.tabular import (subject_matrices, hc_normative_stats,  # noqa: E402
                          feature_norm_stats, apply_deviation)


def pooled_stimulus_matrices(mode):
    """Out-of-fold pooled (160, 100, D) matrices. mode: 'raw' | 'z'."""
    mats = subject_matrices([s for _, _, v in official_folds() for s in v])
    subj_order = [s for _, _, v in official_folds() for s in v]
    D_all, mask_all = [], []
    for _, train_ids, val_ids in official_folds():
        if mode == "raw":
            mu, sigma = feature_norm_stats(train_ids)
            sigma_shrink = sigma
        else:
            mu, sigma, sigma_shrink = hc_normative_stats(train_ids)
        for s in val_ids:
            X, mask = mats[s]
            D, m = apply_deviation(X, mask, mu, sigma, sigma_shrink, mode)
            D_all.append(D)
            mask_all.append(m)
    return np.stack(D_all), np.stack(mask_all), subj_order


def probe_stimulus(X, y, mask):
    keep = mask.astype(bool)
    if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
        return np.nan
    Xk = X[keep]
    yk = y[keep]
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(Xk, yk):
        sc = StandardScaler().fit(Xk[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xk[tr]), yk[tr])
        aucs.append(roc_auc_score(yk[va], clf.predict_proba(sc.transform(Xk[va]))[:, 1]))
    return float(np.mean(aucs))


def main():
    from data.common import labels_of
    results = {}
    for mode in ["raw", "z"]:
        D, mask, subj_order = pooled_stimulus_matrices(mode)
        y = labels_of(subj_order)
        per_stim = {}
        images = image_list()
        for s, img in enumerate(images):
            per_stim[img] = probe_stimulus(D[:, s, :], y, mask[:, s])
        cat_auc = {}
        for img, auc in per_stim.items():
            cat = category_of(img)
            cat_auc.setdefault(cat, []).append(auc)
        results[mode] = {c: float(np.nanmean(v)) for c, v in cat_auc.items()}
        results[mode]["overall"] = float(np.nanmean(list(per_stim.values())))
        print(f"{mode}: " + "  ".join(f"{c}={results[mode][c]:.3f}" for c in results[mode]))
        pd.Series(per_stim, name=f"{mode}_per_stimulus_auc") \
            .to_csv(OUTPUTS / "evaluation" / f"category_probe_{mode}.csv")
    pd.DataFrame(results).T.to_csv(OUTPUTS / "evaluation" / "category_probe_summary.csv")
    print(f"saved -> {OUTPUTS / 'evaluation'}")


if __name__ == "__main__":
    main()
