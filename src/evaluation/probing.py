"""Eval 1 — Embedding separability probing.

Hypothesis: subject embeddings built from stimulus-conditioned normative
DEVIATIONS separate HC/SZ better than embeddings from raw features; the gap
should grow from the hard-deviation ablations (z / mahal) to the learned
latent deviation of the main proposal.

Method: pool the out-of-fold validation embeddings of the 4 official folds
(each embedding comes from a model that never saw its subject), then:
  - linear probe: LogisticRegression, stratified 4-fold inner CV -> AUC ± std
  - silhouette score (HC vs SZ label) on the pooled embedding space

Usage (from EMS-Minh/src):
    python evaluation/probing.py --runs proposal/mlp_attn proposal/z_mean proposal/mahal_mean
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import OUTPUTS  # noqa: E402


def latest_fold_runs(proposal, ablation, seed=42):
    """Latest run dir per official fold for one ablation.

    Only dirs carrying an official fold name (__foldSet_X__) are used, so
    test-protocol runs (fold=heldout_split / official) never leak in.
    """
    base = OUTPUTS / proposal
    runs = {}
    for d in base.glob(f"{ablation}__seed{seed}__fold*__*"):
        name = d.name.split("__fold")[1].split("__")[0]
        if not name.startswith("Set_"):
            continue
        if name not in runs or d.stat().st_mtime > runs[name].stat().st_mtime:
            runs[name] = d
    return list(runs.values())


def load_pooled_embeddings(proposal, ablation, seed=42):
    embs, ys, sids = [], [], []
    for d in latest_fold_runs(proposal, ablation, seed):
        z = np.load(d / "embeddings.npz")
        embs.append(z["emb"]); ys.append(z["label"]); sids.append(z["subject_id"])
    return (np.concatenate(embs), np.concatenate(ys), np.concatenate(sids))


def probe(X, y):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[va]))[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="'{proposal}/{ablation}' pairs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = []
    for spec in args.runs:
        proposal, ablation = spec.split("/")
        try:
            X, y, sids = load_pooled_embeddings(proposal, ablation, args.seed)
        except Exception as e:
            print(f"[skip] {spec}: {e}")
            continue
        auc, auc_std = probe(X, y)
        sil = silhouette_score(StandardScaler().fit_transform(X), y) if len(np.unique(y)) > 1 else np.nan
        rows.append({"run": spec, "n": len(y), "probe_auc": auc, "probe_auc_std": auc_std,
                     "silhouette": sil})
        print(f"{spec:40s} n={len(y):3d}  probe_AUC={auc:.4f}±{auc_std:.4f}  silhouette={sil:.4f}")
    df = pd.DataFrame(rows)
    out = OUTPUTS / "evaluation" / "probing.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
