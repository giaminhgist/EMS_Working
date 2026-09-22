"""Compare re-run artifacts vs backed-up reference (reproducibility check).

Usage (from repo root):
    uv run python outputs/repro_check_20260922/compare.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BACKUP = REPO / "outputs" / "repro_check_20260922" / "backup_original"
PROPOSAL = REPO / "outputs" / "proposal"
BASELINE_NEW = REPO / "docs" / "baseline" / "results" / "summary.csv"
BASELINE_REF = BACKUP / "baseline_results" / "summary.csv"

METRICS = ["auc", "acc", "balanced_acc", "sen", "spe", "f1"]
ABLATIONS = ["mlp_attn", "mlp_norm01", "sub_attn", "zsub_attn", "mlp_mean",
             "mlp_deepset"]
SEEDS = [42, 1234, 2024, 2026, 7]


def classify(d):
    if abs(d) <= 1e-6:
        return "exact"
    if abs(d) <= 1e-3:
        return "noise"
    if abs(d) <= 1e-2:
        return "small"
    return "FLAG"


def compare_proposal():
    rows = []
    for ab in ABLATIONS:
        for seed in SEEDS:
            fname = f"{ab}__seed{seed}__allfolds_summary.json"
            ref_p, new_p = BACKUP / "proposal_summaries" / fname, PROPOSAL / fname
            ref, new = json.loads(ref_p.read_text()), json.loads(new_p.read_text())
            for m in METRICS:
                for agg in ("mean", "std"):
                    d = new["mean_metrics"][m][agg] - ref["mean_metrics"][m][agg]
                    rows.append({"ablation": ab, "seed": seed, "metric": m,
                                 "agg": agg, "ref": ref["mean_metrics"][m][agg],
                                 "new": new["mean_metrics"][m][agg], "delta": d,
                                 "class": classify(d)})
    return pd.DataFrame(rows)


def compare_baseline():
    ref = pd.read_csv(BASELINE_REF)
    new = pd.read_csv(BASELINE_NEW)
    key = ["protocol", "method", "rep", "seed"]
    cols = [c for c in ref.columns if c in key
            or c.split("_")[0] in ["acc", "sen", "spe", "pre", "f1", "auc"]]
    m = ref[key + ["auc", "acc"]].merge(new[key + ["auc", "acc"]],
                                        on=key, suffixes=("_ref", "_new"))
    m["delta_auc"] = m.auc_new - m.auc_ref
    m["delta_acc"] = m.acc_new - m.acc_ref
    m["class"] = m[["delta_auc", "delta_acc"]].abs().max(axis=1).map(classify)
    return m.sort_values(key)


if __name__ == "__main__":
    p = compare_proposal()
    b = compare_baseline()
    out = REPO / "outputs" / "repro_check_20260922"
    p.to_csv(out / "proposal_deltas.csv", index=False)
    b.to_csv(out / "baseline_deltas.csv", index=False)
    print("== Proposal: worst deltas per (ablation, metric, agg) ==")
    print(p.assign(abs_delta=p.delta.abs()).sort_values(
        "abs_delta", ascending=False).drop_duplicates(
        ["ablation", "metric", "agg"]).head(20).to_string(index=False))
    print("\n== Proposal: delta class counts ==")
    print(p["class"].value_counts().to_string())
    flagged = p[p["class"] == "FLAG"]
    print(f"\n== FLAGGED proposal rows: {len(flagged)} ==")
    if len(flagged):
        print(flagged.to_string(index=False))
    print("\n== Baseline (P1 seed 42 / P2 3 seeds): delta classes ==")
    print(b["class"].value_counts().to_string())
    print(b.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
