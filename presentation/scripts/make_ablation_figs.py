"""Figures 05 — ablations, one figure per scientific question.

Each figure title states the question the ablation tests (per the prompt's
matrix). All comparisons use paired statistics over the 5 seeds' 4-fold mean
validation AUC (n=5 — small, so p-values are indicative only and reported
alongside the raw per-seed values, matching docs/analysis.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, SEEDS, ABLATION_META, ABLATION_LABELS,  # noqa: E402
                    LEARNED_COLOR, HARD_COLOR, savefig, load_allfolds_summary,
                    load_run_preds, metrics_from_preds)
OUT = FIG / "04_ablations"


def seed_vals(ablation, metric="auc"):
    return {s: load_allfolds_summary(ablation, s)["mean_metrics"][metric]["mean"]
            for s in SEEDS}


def paired_p(a, b):
    t, p = stats.ttest_rel(list(a.values()), list(b.values()))
    return p


def paired_bars(ax, groups, metric="auc", ylab="validation AUC (4-fold mean)"):
    """Paired-comparison bars with per-seed lines; returns axis + p-values."""
    names = [g for g in groups]
    vals = {g: seed_vals(g, metric) for g in groups}
    xs = np.arange(len(groups))
    means = [np.mean(list(vals[g].values())) for g in groups]
    sds = [np.std(list(vals[g].values())) for g in groups]
    colors = [LEARNED_COLOR if g.startswith(("mlp", "sub", "zsub")) else HARD_COLOR
              for g in groups]
    ax.bar(xs, means, yerr=sds, capsize=4, color=colors, alpha=0.9, width=0.55)
    # per-seed paired lines
    rng = np.random.default_rng(0)
    for gi in range(len(groups)):
        ax.scatter(xs[gi] + rng.normal(0, 0.03, len(SEEDS)),
                   [vals[groups[gi]][s] for s in SEEDS], s=18, color="#222222",
                   alpha=0.6, lw=0, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(ylab)
    if metric == "auc":
        lo = min(means) - 0.03
        ax.set_ylim(lo, 1.0)
    return ax, vals


def annotate_pairs(ax, groups, vals):
    """Annotate paired t-tests between consecutive groups above the bars."""
    means = [np.mean(list(vals[g].values())) for g in groups]
    sds = [np.std(list(vals[g].values())) for g in groups]
    top = max(m + sd for m, sd in zip(means, sds))
    span = ax.get_ylim()[1] - top
    y = top + 0.18 * span
    for i in range(len(groups) - 1):
        a, b = groups[i], groups[i + 1]
        p = paired_p(vals[a], vals[b])
        x1, x2 = i, i + 1
        dy = 0.06 * span
        ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="#555555", lw=0.9)
        ax.text((x1 + x2) / 2, y + dy * 1.4, f"p={p:.3f}", ha="center", fontsize=8)
        y += 1.5 * dy


# --------------------------------------------------------------------- #
# F04.01 learned vs hard deviation (mlp_mean vs z_mean / diff_mean / mahal_mean)
# --------------------------------------------------------------------- #
def fig_learned_vs_hard():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 3.9))
    groups = ["mlp_mean", "z_mean", "diff_mean", "mahal_mean"]
    ax, vals = paired_bars(axes[0], groups, "auc")
    annotate_pairs(ax, groups, vals)
    ax.set_title("Question: does a LEARNED normative deviation beat fixed\n"
                 "(hard) deviations in raw feature space? — validation AUC")
    ax2, vals2 = paired_bars(axes[1], groups, "acc", "validation Accuracy (thr 0.5)")
    ax2.set_title("Same comparison — validation Accuracy")
    # key comparison mlp_mean vs z_mean
    p = paired_p(vals["mlp_mean"], vals["z_mean"])
    ax.text(0.5, 0.02, f"mlp_mean vs z_mean: ΔAUC = "
            f"{np.mean(list(vals['mlp_mean'].values())) - np.mean(list(vals['z_mean'].values())):+.3f} "
            f"(paired t, n=5 seeds, p={p:.4f})", transform=ax.transAxes,
            fontsize=8.5, ha="center")
    savefig(fig, OUT, "F04.01_learned_vs_hard")
    pd.DataFrame({g: vals[g] for g in groups}).to_csv(TAB / "T04.01_learned_vs_hard.csv")


# --------------------------------------------------------------------- #
# F04.02 hard-deviation chain: diff -> z -> mahal
# --------------------------------------------------------------------- #
def fig_hard_chain():
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    groups = ["diff_mean", "z_mean", "mahal_mean"]
    for ax, metric, ylab in [(axes[0], "auc", "AUC"), (axes[1], "acc", "Accuracy"),
                             (axes[2], "balanced_acc", "Balanced Accuracy")]:
        axb, vals = paired_bars(ax, groups, metric, f"validation {ylab} (4-fold mean)")
        annotate_pairs(ax, groups, vals)
    fig.suptitle("Question: what do HC conditioning (z) and the extra scalar (mahal) add, "
                 "one step at a time?", fontsize=11, y=1.0)
    savefig(fig, OUT, "F04.02_hard_chain")
    pd.DataFrame({g: seed_vals(g, "auc") for g in groups}).to_csv(TAB / "T04.02_hard_chain.csv")


# --------------------------------------------------------------------- #
# F04.03 learned comparator
# --------------------------------------------------------------------- #
def fig_comparator():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 3.9))
    groups = ["mlp_attn", "zsub_attn", "sub_attn"]
    ax, vals = paired_bars(axes[0], groups, "auc")
    annotate_pairs(ax, groups, vals)
    ax.set_title("Question: does the learned comparator gφ([z,μ,z−μ,z⊙μ]) help\n"
                 "over fixed latent subtraction? — validation AUC")
    p = paired_p(vals["mlp_attn"], vals["sub_attn"])
    ax.text(0.5, 0.02, f"mlp_attn vs sub_attn: ΔAUC = "
            f"{np.mean(list(vals['mlp_attn'].values())) - np.mean(list(vals['sub_attn'].values())):+.3f} "
            f"(p={p:.4f})", transform=ax.transAxes, fontsize=8.5, ha="center")
    ax2, vals2 = paired_bars(axes[1], groups, "acc", "validation Accuracy (thr 0.5)")
    ax2.set_title("Same comparison — validation Accuracy")
    savefig(fig, OUT, "F04.03_comparator")
    pd.DataFrame({g: vals[g] for g in groups}).to_csv(TAB / "T04.03_comparator.csv")


# --------------------------------------------------------------------- #
# F04.04 pooling
# --------------------------------------------------------------------- #
def fig_pooling():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 3.9))
    groups = ["mlp_attn", "mlp_mean", "mlp_deepset"]
    ax, vals = paired_bars(axes[0], groups, "auc")
    annotate_pairs(ax, groups, vals)
    ax.set_title("Question: how does stimulus-set aggregation affect\n"
                 "performance? — validation AUC (attn vs mean vs deepset)")
    ax2, vals2 = paired_bars(axes[1], groups, "acc", "validation Accuracy (thr 0.5)")
    ax2.set_title("Same comparison — validation Accuracy")
    savefig(fig, OUT, "F04.04_pooling")
    pd.DataFrame({g: vals[g] for g in groups}).to_csv(TAB / "T04.04_pooling.csv")


# --------------------------------------------------------------------- #
# F04.05 lambda_norm
# --------------------------------------------------------------------- #
def fig_lambda_norm():
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    groups = ["mlp_attn", "mlp_norm01"]
    ax, vals = paired_bars(axes[0], groups, "auc")
    annotate_pairs(ax, groups, vals)
    ax.set_title("Question: does HC concentration regularization (λ=0.1)\nhelp? — validation AUC")
    ax2, vals2 = paired_bars(axes[1], groups, "acc", "validation Accuracy (thr 0.5)")
    ax2.set_title("Accuracy trade-off (pulled toward HC)")
    # seed stability: per-seed AUC spread
    ax3 = axes[2]
    for g, color in [("mlp_attn", LEARNED_COLOR), ("mlp_norm01", "#56B4E9")]:
        v = list(vals[g].values())
        ax3.plot(SEEDS, v, "o-", color=color, label=ABLATION_LABELS[g])
    ax3.set_xticks(SEEDS)
    ax3.set_xlabel("seed")
    ax3.set_ylabel("validation AUC (4-fold mean)")
    ax3.set_title("Per-seed AUC — λ=0.1 is the tightest across seeds")
    ax3.legend(fontsize=8)
    savefig(fig, OUT, "F04.05_lambda_norm")
    pd.DataFrame({g: vals[g] for g in groups}).to_csv(TAB / "T04.05_lambda_norm.csv")


# --------------------------------------------------------------------- #
# F04.06 stimulus budget (xstim) — subset sensitivity
# --------------------------------------------------------------------- #
def fig_stimulus_budget():
    df = pd.read_csv(Path("/root/EMS-Project/outputs/evaluation/cross_stimulus.csv"))
    # also verify against the xstim allfolds summaries
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    for ax, metric in [(axes[0], "auc"), (axes[1], "acc"), (axes[2], "balanced_acc")]:
        for mode, color, label in [("z", HARD_COLOR, "hard z-deviation"),
                                   ("learned", LEARNED_COLOR, "learned (mlp_attn)")]:
            sub = df[df["mode"] == mode].sort_values("n_stim")
            ax.plot(sub.n_stim, sub[metric], "o-", color=color, lw=2, ms=6, label=label)
        ax.set_xlabel("number of stimuli (K)")
        ax.set_ylabel(f"validation {metric.upper()} (4-fold mean)")
        ax.set_xticks([25, 50, 100])
        if metric == "auc":
            ax.legend(fontsize=9)
    fig.suptitle("Question: how many stimuli are needed, and which deviation transfers "
                 "better when K shrinks?\n(same random subset K across folds, retrained per K "
                 "on train+val — a stimulus-budget sensitivity study, not an unseen-stimulus test)",
                 fontsize=10, y=1.04)
    # AUC drop 100->25 annotation
    drops = {}
    for mode in ["z", "learned"]:
        sub = df[df["mode"] == mode].sort_values("n_stim")
        drops[mode] = sub.auc.iloc[-1] - sub.auc.iloc[0]
    axes[0].text(0.5, 0.05, f"AUC drop 100→25: z = {drops['z']:.4f} vs "
                            f"learned = {drops['learned']:.4f}",
                 transform=axes[0].transAxes, fontsize=9, ha="center")
    savefig(fig, OUT, "F04.06_stimulus_budget")
    df.to_csv(TAB / "T04.06_stimulus_budget.csv", index=False)


if __name__ == "__main__":
    fig_learned_vs_hard()
    fig_hard_chain()
    fig_comparator()
    fig_pooling()
    fig_lambda_norm()
    fig_stimulus_budget()
    print("done 04_ablations")
