"""Tabular (45-dim hand-crafted feature) access for the proposal.

Provides per-subject stimulus matrices X (S x 45) + presence masks, plus
leakage-safe normative statistics: everything is fitted ONLY on the given
train subjects (HC-only where stated), then applied to any other subjects.
"""
import numpy as np
import pandas as pd

from data.common import NUM_FEATURES, NUM_STIMULI, load_stimulus_features, image_list

EPS = 1e-6


def subject_matrices(subject_ids, partition="train"):
    """Return {subject_id: (X (S,45) float32, mask (S,) bool)}.

    X rows are NaN for missing (subject, stimulus) pairs; mask marks valid ones.
    """
    feat = load_stimulus_features(partition)
    images = image_list()
    feat_ids = set(feat.index.get_level_values(0))
    # official-test subjects have synthetic ids 400..447 -> route to the test pkl
    if partition == "train" and any(int(s) >= 400 for s in subject_ids):
        return subject_matrices(subject_ids, partition="test")
    out = {}
    for sid in subject_ids:
        key = sid - 400 if partition == "test" else sid  # test pkl uses file ids 0..47
        sub = feat.loc[key] if key in feat_ids else None
        if sub is None or len(sub) == 0:
            X = np.full((NUM_STIMULI, NUM_FEATURES), np.nan, dtype=np.float32)
            mask = np.zeros(NUM_STIMULI, dtype=bool)
        else:
            X = sub.reindex(images).to_numpy(dtype=np.float32)
            mask = ~np.isnan(X).all(axis=1)
        out[int(sid)] = (X, mask)
    return out


def feature_norm_stats(subject_ids, partition="train"):
    """Per-feature mean/std across the given subjects (unsupervised scaling).

    Used to scale raw inputs before deviation computation. Fits on the passed
    subject ids only — call with the training fold to avoid leakage.
    """
    mats = subject_matrices(subject_ids, partition)
    X = np.concatenate([m[0][m[1]] for m in mats.values()], axis=0)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.maximum(std, EPS)
    return mean, std


def hc_normative_stats(train_ids):
    """Stimulus-conditioned HC normative statistics from train-fold HC only.

    Returns (mu (S,45), sigma (S,45), sigma_shrink (S,45)) with mu, sigma
    computed per stimulus over the HC subjects of `train_ids`.
    """
    mats = subject_matrices(train_ids)
    hc_ids = [s for s in train_ids if s < 200]
    X_all = np.stack([mats[s][0] for s in hc_ids], axis=0)          # (N_HC, S, 45)
    mask_all = np.stack([mats[s][1] for s in hc_ids], axis=0)
    mu = np.full((NUM_STIMULI, NUM_FEATURES), np.nan, dtype=np.float32)
    sigma = np.full((NUM_STIMULI, NUM_FEATURES), np.nan, dtype=np.float32)
    for s in range(NUM_STIMULI):
        col = X_all[:, s, :]
        m = mask_all[:, s]
        if m.sum() == 0:
            continue
        mu[s] = np.nanmean(col[m], axis=0)
        sigma[s] = np.nanstd(col[m], axis=0)
    sigma = np.nan_to_num(sigma, nan=0.0)
    mu = np.nan_to_num(mu, nan=0.0)
    # shrinkage toward the per-feature median sigma across stimuli (Marquand-style)
    med_sigma = np.nanmedian(sigma, axis=0)
    sigma_shrink = 0.9 * sigma + 0.1 * med_sigma
    sigma_shrink = np.maximum(sigma_shrink, EPS)
    return mu, np.maximum(sigma, EPS), sigma_shrink


def apply_deviation(X, mask, mu, sigma, sigma_shrink, mode):
    """Convert a subject matrix to per-stimulus deviation vectors.

    mode: 'raw'  -> standardized raw features (mu/sigma here must be the global
                     feature stats from feature_norm_stats)
          'diff' -> (x - mu) with mu = HC stimulus norms
          'z'    -> (x - mu) / sigma
          'mahal'-> z-deviation concatenated with the per-stimulus Mahalanobis
                     scalar (shrinkage diagonal covariance)
    Returns (S, D_out, mask) with D_out = 45 (raw/diff/z) or 46 (mahal).
    """
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)
    if mode == "raw":
        D = (X - mu) / sigma
    elif mode == "diff":
        D = X - mu
    elif mode == "z":
        D = (X - mu) / sigma
    elif mode == "mahal":
        z = (X - mu) / sigma
        m_s = np.sqrt(np.mean(((X - mu) / sigma_shrink) ** 2, axis=1, keepdims=True))
        D = np.concatenate([z, m_s.astype(np.float32)], axis=1)
    else:
        raise ValueError(mode)
    D = D * mask[:, None].astype(np.float32)
    return D, mask
