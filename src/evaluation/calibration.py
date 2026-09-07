"""Eval 2 — Calibration (ECE + Brier) on pooled out-of-fold predictions.

Hypothesis: models built on normative DEVIATIONS are better calibrated than
models on raw features, because deviations are a relative quantity with a
stable interpretation across stimuli (near-zero = healthy), which should map
to probabilities closer to the observed frequencies.

Method: pool the 4 official folds' val predictions (each out-of-fold), bin
probabilities into 10 equal-width bins, compute expected calibration error
(ECE) and Brier score.

Usage (from EMS-Minh/src):
    python evaluation/calibration.py --runs proposal/z_mean proposal/diff_mean proposal/mlp_attn
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import OUTPUTS  # noqa: E402
from evaluation.probing import latest_fold_runs  # noqa: E402


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = []
    for spec in args.runs:
        proposal, ablation = spec.split("/")
        ys, ps = [], []
        for d in latest_fold_runs(proposal, ablation, args.seed):
            df = pd.read_csv(d / "predictions.csv")
            ys.append(df.label.values); ps.append(df.prob.values)
        y, p = np.concatenate(ys), np.concatenate(ps)
        brier = float(np.mean((p - y) ** 2))
        rows.append({"run": spec, "n": len(y), "ece": ece(y, p), "brier": brier,
                     "auc": None})
        # AUC as reference
        from sklearn.metrics import roc_auc_score
        rows[-1]["auc"] = roc_auc_score(y, p)
        print(f"{spec:40s} n={len(y):3d}  ECE={rows[-1]['ece']:.4f}  "
              f"Brier={brier:.4f}  AUC={rows[-1]['auc']:.4f}")
    out = OUTPUTS / "evaluation" / "calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
