"""Figures 02 — the 45 hand-crafted features.

Statistical unit: SUBJECT-level aggregation of each feature over valid
stimuli (mean), n=80 per group — never individual fixations or
subject-stimulus pairs treated as independent.

Implementation notes honored here (see src/features.py):
  - tem_ifi_* = mean of (dur_i + dur_{i+1})/2 (onset timestamps unavailable)
  - tem_velocity_mean = saccade amplitude / that IFI proxy (px/ms) — NOT a
    measured saccade velocity
  - tem_fix_rate = n_fix / (sum(dur) + mean_ifi*(n-1)) * 1000  (fix/s, time
    reconstructed in code)
  - tem_trans_entropy = occupancy entropy of fixations minus the last one on a
    4x3 grid — NOT a transition-matrix entropy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, TAB, HC_COLOR, SZ_COLOR, FEATURE_NAMES,  # noqa: E402
                    FEATURE_GROUPS, feature_group_of, savefig,
                    load_subject_features, load_metadata)

OUT = FIG / "02_features"
CAT_ORDER = ["social", "natural", "synthetic", "manipulated"]

meta = load_metadata()
LABELS = meta.loc[meta.partition == "train", "label"]


def subject_level():
    """Subject x 45 feature table (mean over valid stimuli)."""
    feat = load_subject_features("train")
    return feat.groupby(level="subject_id").mean()


def welch_d(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return p, d


# --------------------------------------------------------------------- #
# F02.01 feature overview (formulas + group table)
# --------------------------------------------------------------------- #
def fig_feature_overview():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    # toy scanpath illustrating spa_dispersion / geo_scanpath_len / tem_dur_mean
    rng = np.random.default_rng(3)
    pts = np.array([[150, 380], [280, 260], [430, 330], [610, 240], [700, 420],
                    [560, 560], [820, 520], [880, 300]])
    dur = np.array([220, 180, 260, 210, 300, 240, 280, 250])
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#333333", lw=1.4, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=dur * 0.12, color=HC_COLOR, alpha=0.8,
               zorder=3)
    centroid = pts.mean(0)
    ax.scatter(*centroid, marker="x", s=80, color=SZ_COLOR, zorder=4)
    for p in pts:
        ax.plot([p[0], centroid[0]], [p[1], centroid[1]], "--", color="#999999",
                lw=0.8, zorder=1)
    step = pts[2:3]
    ax.annotate("saccade\namplitude\n(p₃→p₄)",
                xy=(0.5 * (pts[2] + pts[3])[0], 0.5 * (pts[2] + pts[3])[1] - 40),
                fontsize=8, ha="center", color="#333333")
    ax.annotate("spa_dispersion = mean of dashed distances",
                xy=(150, 665), fontsize=8.5, color=SZ_COLOR)
    ax.annotate("geo_scanpath_len = Σ successive distances (px)",
                xy=(150, 630), fontsize=8.5, color="#333333")
    ax.annotate("tem_dur_mean = mean fixation duration (dot size ∝ duration)",
                xy=(150, 595), fontsize=8.5, color=HC_COLOR)
    ax.set_xlim(40, 1000)
    ax.set_ylim(680, 150)
    ax.set_title("(a) How a few key features are computed\n(one stimulus, one subject)",
                 fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    # (b) grouped table of the 45 features
    ax = axes[1]
    ax.axis("off")
    ax.set_title("(b) The 45 features per (subject, stimulus)", fontsize=10.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    rows = [
        ("spa_pos", "Position & dispersion (7)", "fix count, mean/std x/y, dispersion, bbox area"),
        ("spa_center", "Center & regions (10)", "center distance/frac, quadrants q1–q4, grid entropy"),
        ("geo", "Scanpath geometry (10)", "scanpath length, saccade amp mean/std/max, angle var, revisit rate, hull area"),
        ("tem", "Temporal (11)", "duration mean/std/total/max, first/last, IFI proxy, velocity proxy, fix rate, occupancy entropy"),
        ("pup", "Pupil (7)", "mean/std/min/max/median, slope over time, first-last diff"),
    ]
    y = 9.2
    for key, title, desc in rows:
        ax.add_patch(plt.Rectangle((0.2, y - 0.62), 0.34, 1.24,
                                   facecolor=plt.cm.tab10(list(FEATURE_GROUPS).index(key)),
                                   alpha=0.85))
        ax.text(1.0, y, title, fontsize=9.5, va="center", fontweight="bold")
        ax.text(1.0, y - 0.38, desc, fontsize=8, va="center", color="#333333")
        y -= 1.85
    savefig(fig, OUT, "F02.01_feature_overview")


# --------------------------------------------------------------------- #
# F02.02 HC vs SZ effect sizes + rainclouds + category profiles
# --------------------------------------------------------------------- #
def fig_effect_sizes():
    S = subject_level().reindex(columns=FEATURE_NAMES)
    hc = S[LABELS == 0]
    sz = S[LABELS == 1]
    rows = []
    for f in FEATURE_NAMES:
        a, b = hc[f].dropna(), sz[f].dropna()
        p, d = welch_d(a, b)
        rows.append({"feature": f, "group": feature_group_of(f),
                     "cohen_d": d, "p_welch": p,
                     "hc_mean": a.mean(), "sz_mean": b.mean(),
                     "hc_n": len(a), "sz_n": len(b)})
    es = pd.DataFrame(rows).set_index("feature")
    es = es.sort_values("cohen_d")
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.8, 9.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.26,
                          left=0.06, right=0.975, top=0.955, bottom=0.04)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    ax = fig.add_subplot(gs_top[0])
    colors = [plt.cm.tab10(list(FEATURE_GROUPS).index(feature_group_of(f)))
              for f in es.index]
    ax.barh(range(45), es.cohen_d.values, color=colors)
    for i, (f, row) in enumerate(es.iterrows()):
        stars = "***" if row.p_welch < 0.001 else ("**" if row.p_welch < 0.01 else
                                                    ("*" if row.p_welch < 0.05 else "n.s."))
        x = row.cohen_d + (0.02 if row.cohen_d >= 0 else -0.02)
        ha = "left" if row.cohen_d >= 0 else "right"
        ax.text(x, i, stars, va="center", ha=ha, fontsize=6.5,
                color="#333333" if stars != "n.s." else "#aaaaaa")
    ax.set_yticks(range(45))
    ax.set_yticklabels(es.index, fontsize=6.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d (HC − SZ, subject level; positive = HC larger)")
    ax.set_title("(a) Effect size of every feature (subject-level, n=80/80)",
                 fontsize=10.5, pad=20)
    # rainclouds for representative features (one per group)
    ax = fig.add_subplot(gs_top[1])
    reps = ["spa_dispersion", "spa_entropy", "geo_sacc_amp_mean", "tem_dur_mean", "pup_mean"]
    gs_rain = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=ax.get_subplotspec(),
                                               hspace=0.4)
    axes2 = [fig.add_subplot(gs_rain[i]) for i in range(5)]
    ax.axis("off")
    ax.set_title("(b) Representative features, one per group",
                 fontsize=10.5, pad=20)
    for i, f in enumerate(reps):
        a, b = hc[f].dropna(), sz[f].dropna()
        aax = axes2[i]
        for pos, data, color in [(-0.32, a, HC_COLOR), (0.32, b, SZ_COLOR)]:
            vp = aax.violinplot(data, positions=[pos], widths=0.42,
                                showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.35)
            aax.boxplot(data, positions=[pos], widths=0.14, showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color="k", lw=0.8),
                        boxprops=dict(facecolor="white", color=color, lw=0.8),
                        whiskerprops=dict(color=color, lw=0.8),
                        capprops=dict(color=color, lw=0.8))
        p, d = welch_d(a, b)
        aax.set_title(f, fontsize=8, loc="left", pad=6)
        aax.set_xticks([-0.32, 0.32])
        aax.set_xticklabels(["HC", "SZ"], fontsize=7)
        aax.tick_params(axis="y", labelsize=6.5)
        aax.set_title(f"d={d:.2f}, p={p:.1g}", fontsize=7, loc="right", pad=6)
    # --- bottom row: (c) feature x category profiles -------------------
    feat = load_subject_features("train")
    feat = feat.join(meta["label"], on="subject_id")
    cats = feat.index.get_level_values("image").map(
        lambda im: im.split("_")[0]).map(
        lambda p: {"act": "social", "por": "social", "soc": "social",
                   "ind": "natural", "land": "natural", "outman": "natural",
                   "sat": "natural", "art": "synthetic", "cat": "synthetic",
                   "pat": "synthetic", "low": "manipulated", "mood": "manipulated",
                   "noi": "manipulated", "patch": "manipulated", "rand": "manipulated"}[p])
    feat["category"] = cats
    figs = ["tem_dur_mean", "spa_dispersion", "geo_scanpath_len", "pup_mean"]
    subj_cat = feat.groupby(["subject_id", "category", "label"])[figs].mean().reset_index()
    gs_bot = gs[1].subgridspec(1, 4, wspace=0.3)
    axes_c = [fig.add_subplot(gs_bot[i]) for i in range(4)]
    for i, f in enumerate(figs):
        axc = axes_c[i]
        x = np.arange(4)
        for gi, (lab, color) in enumerate([(0, HC_COLOR), (1, SZ_COLOR)]):
            grp = subj_cat[subj_cat.label == lab].groupby("category")[f]
            axc.bar(x + (gi - 0.5) * 0.35, grp.mean().reindex(CAT_ORDER).values,
                    width=0.35, color=color, yerr=grp.sem().reindex(CAT_ORDER).values,
                    capsize=2.5, label="HC" if lab == 0 else "SZ")
        axc.set_xticks(x)
        axc.set_xticklabels([c[:6] for c in CAT_ORDER], fontsize=8)
        axc.set_title(f, fontsize=9.5)
        if i == 0:
            axc.set_ylabel("subject-level mean (±SEM)")
    pos = gs[1].get_position(fig)
    handles, labels = axes_c[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=2, loc="lower right",
               bbox_to_anchor=(pos.x1, pos.y1 + 0.025))
    fig.text(pos.x0, pos.y1 + 0.027, "(c) Feature values depend on stimulus "
             "category — per-subject category means, HC vs SZ",
             fontsize=10.5, ha="left", va="bottom")

    savefig(fig, OUT, "F02.02_effect_sizes")
    es.to_csv(TAB / "T02.02_effect_sizes.csv")
    subj_cat.groupby(["category", "label"])[figs].agg(["mean", "sem"]) \
        .to_csv(TAB / "T02.03_category_profiles.csv")


if __name__ == "__main__":
    fig_feature_overview()
    fig_effect_sizes()
    print("done 02_features")
