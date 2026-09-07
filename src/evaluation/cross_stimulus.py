"""Eval 4 — Cross-stimulus generalization (fewer stimuli at test time).

Hypothesis: a stimulus-conditioned normative DEVIATION is a transferable
quantity: a subject's deviation estimated from a small random subset of
stimuli should retain more discriminative power than a model that must learn
the comparison, so the hard z-deviation ablation should degrade LESS than the
main proposal (learned latent deviation) when the number of stimuli shrinks
100 -> 50 -> 25.

Method: train the hard z-deviation ablation (deviation=z, pool=mean) and the
main proposal (deviation=learned, comparator=mlp, pool=attention) under the
official 4-fold protocol with random stimulus subsets of size 25/50/100 (same
subset across folds, stimulus seed 42) and report fold-mean validation AUC.

Usage (from EMS-Minh/src):
    python evaluation/cross_stimulus.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.common import OUTPUTS  # noqa: E402

HERE = Path(__file__).resolve().parent

# mode -> extra CLI flags of proposal/train.py
CONFIGS = {
    "z": ["--deviation", "z", "--pool", "mean"],                          # hard ablation
    "learned": ["--deviation", "learned", "--comparator", "mlp",           # main proposal
                "--pool", "attention"],
}


def run(mode, n_stim):
    out = OUTPUTS / "proposal" / f"xstim_{mode}_{n_stim}__seed42__allfolds_summary.json"
    if not out.exists():  # skip re-training if the summary already exists
        cmd = [sys.executable, str(HERE.parent / "proposal" / "train.py"),
               "--ablation", f"xstim_{mode}_{n_stim}",
               "--n_stim", str(n_stim), "--stim_seed", "42",
               "--fold", "all", "--epochs", "100", "--patience", "30"] \
            + CONFIGS[mode]
        subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out.read_text())["mean_metrics"]


def main():
    rows = []
    for mode in CONFIGS:
        for n in [25, 50, 100]:
            m = run(mode, n)
            rows.append({"mode": mode, "n_stim": n, **{k: v["mean"] for k, v in m.items()}})
            print(f"{mode} n_stim={n:3d}: AUC={m['auc']['mean']:.4f}  "
                  f"Acc={m['acc']['mean']:.4f}  F1={m['f1']['mean']:.4f}")
    df = pd.DataFrame(rows)
    out = OUTPUTS / "evaluation" / "cross_stimulus.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # degradation of AUC from 100 -> 25 stimuli
    for mode in CONFIGS:
        sub = df[df["mode"] == mode].sort_values("n_stim")
        drop = sub.auc.iloc[-1] - sub.auc.iloc[0]
        print(f"{mode}: AUC drop 100->25 stimuli = {drop:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
