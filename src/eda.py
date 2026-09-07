"""Exploratory data analysis for the EMS dataset.

Loads the raw EMS dataset, computes summary statistics and produces all
figures + markdown reports under docs/EDA/.

Usage:
    python src/eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, RAW, DOCS_EDA, TRAIN_FIX_DIR, TEST_FIX_DIR,
                    IMAGES_DIR, SCREEN_W, SCREEN_H, load_all, image_categories,
                    official_folds, subject_labels, subject_label, train_subject_ids)

FIG_DIR = DOCS_EDA / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.3,
})
HC_COLOR, SZ_COLOR = "#1f77b4", "#d62728"
CAT_COLORS = {"social": "#4c72b0", "natural": "#55a868", "synthetic": "#c44e52",
              "manipulated": "#8172b3"}


def load_full():
    """Concatenate train+test fixations with label column."""
    tr = load_all("train")
    tr["label"] = tr["subject_id"].map(lambda s: subject_label(s))
    tr["partition"] = "train"
    te = load_all("test")
    te["label"] = np.nan  # official test labels are not released
    te["partition"] = "test"
    return pd.concat([tr, te], ignore_index=True)


def fig_fixations_per_subject(df):
    tr = df[df.partition == "train"]
    cnt = tr.groupby(["subject_id", "label"]).size().reset_index(name="n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(cnt[cnt.label == 0]["n"], bins=30, color=HC_COLOR, alpha=0.75,
                 label=f"HC (n={ (cnt.label == 0).sum() })")
    axes[0].hist(cnt[cnt.label == 1]["n"], bins=30, color=SZ_COLOR, alpha=0.75,
                 label=f"SZ (n={ (cnt.label == 1).sum() })")
    axes[0].set_xlabel("fixations per subject"); axes[0].set_ylabel("# subjects")
    axes[0].set_title("Total fixations per subject (train)")
    axes[0].legend()
    g = cnt.groupby("label")["n"]
    axes[1].bar([0, 1], g.mean(), yerr=g.std(), color=[HC_COLOR, SZ_COLOR],
                tick_label=["HC", "SZ"], alpha=0.85, capsize=6)
    axes[1].set_ylabel("mean fixations / subject")
    axes[1].set_title("Mean ± std fixations per subject")
    t, p = stats.ttest_ind(cnt[cnt.label == 0]["n"], cnt[cnt.label == 1]["n"],
                           equal_var=False)
    fig.suptitle(f"Fixation counts per subject  (Welch t-test p={p:.3g})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_fixations_per_subject.png")
    plt.close(fig)
    return cnt.groupby("label")["n"].describe()


def fig_fixations_per_stimulus(df):
    tr = df[df.partition == "train"]
    cats = image_categories()
    per_stim = tr.groupby("IMAGE").size()
    order = per_stim.sort_values().index
    colors = [CAT_COLORS[cats[img]] for img in order]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(100), per_stim[order], color=colors)
    for c in CAT_COLORS:
        ax.bar([], [], color=CAT_COLORS[c], label=c)
    ax.set_xticks([])
    ax.set_xlabel("stimuli (sorted by fixation count)")
    ax.set_ylabel("total fixations (160 subjects)")
    ax.set_title("Fixations per stimulus, colored by category")
    ax.legend(title="category")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_fixations_per_stimulus.png")
    plt.close(fig)
    cat_stats = (tr.groupby("category").size()
                 .reindex(["social", "natural", "synthetic", "manipulated"]))
    return cat_stats


def fig_density_maps(df):
    tr = df[df.partition == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (label, name, color) in zip(axes, [(0, "HC", HC_COLOR), (1, "SZ", SZ_COLOR)]):
        pts = tr[tr.label == label][["FIX_X", "FIX_Y"]].values
        ax.hist2d(pts[:, 0], pts[:, 1], bins=[64, 48], range=[[0, SCREEN_W], [0, SCREEN_H]],
                  cmap="inferno")
        ax.invert_yaxis()
        ax.set_title(f"{name} fixation density (n={len(pts):,})")
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    fig.suptitle("Spatial distribution of fixations on the 1024×768 screen")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_density_map_hc_vs_sz.png")
    plt.close(fig)

    # per-quadrant occupancy
    def quadrants(sub):
        x, y = sub["FIX_X"], sub["FIX_Y"]
        q1 = ((x < SCREEN_W / 2) & (y < SCREEN_H / 2)).mean()
        q2 = ((x >= SCREEN_W / 2) & (y < SCREEN_H / 2)).mean()
        q3 = ((x < SCREEN_W / 2) & (y >= SCREEN_H / 2)).mean()
        q4 = ((x >= SCREEN_W / 2) & (y >= SCREEN_H / 2)).mean()
        return pd.Series({"TL": q1, "TR": q2, "BL": q3, "BR": q4})
    occ = tr.groupby("label").apply(quadrants)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    occ.T.plot.bar(ax=ax, color=[HC_COLOR, SZ_COLOR], alpha=0.85)
    ax.set_ylabel("fraction of fixations"); ax.set_xlabel("screen quadrant")
    ax.set_title("Quadrant occupancy (train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03b_quadrant_occupancy.png")
    plt.close(fig)
    return occ


def fig_duration_distribution(df):
    tr = df[df.partition == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, color, name in [(0, HC_COLOR, "HC"), (1, SZ_COLOR, "SZ")]:
        d = tr[tr.label == label]["FIX_DURATION"]
        axes[0].hist(d.clip(upper=1500), bins=60, color=color, alpha=0.6, label=name)
    axes[0].set_xlabel("fixation duration (ms, clipped at 1500)")
    axes[0].set_ylabel("count"); axes[0].legend()
    axes[0].set_title("Fixation duration distribution")
    g = tr.groupby("label")["FIX_DURATION"]
    desc = g.describe()
    axes[1].bar([0, 1], g.mean(), yerr=g.std(), color=[HC_COLOR, SZ_COLOR],
                tick_label=["HC", "SZ"], capsize=6, alpha=0.85)
    axes[1].set_ylabel("mean duration (ms)"); axes[1].set_title("Mean ± std duration")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_duration_distribution.png")
    plt.close(fig)
    return desc


def fig_pupil_distribution(df):
    tr = df[df.partition == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, color, name in [(0, HC_COLOR, "HC"), (1, SZ_COLOR, "SZ")]:
        d = tr[tr.label == label]["FIX_PUPIL"]
        axes[0].hist(d, bins=60, color=color, alpha=0.6, label=name)
    axes[0].set_xlabel("pupil size (a.u.)"); axes[0].set_ylabel("count")
    axes[0].legend(); axes[0].set_title("Pupil size distribution")
    g = tr.groupby("label")["FIX_PUPIL"]
    axes[1].bar([0, 1], g.mean(), yerr=g.std(), color=[HC_COLOR, SZ_COLOR],
                tick_label=["HC", "SZ"], capsize=6, alpha=0.85)
    axes[1].set_ylabel("mean pupil (a.u.)"); axes[1].set_title("Mean ± std pupil size")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_pupil_distribution.png")
    plt.close(fig)
    return g.describe()


def fig_outlier_summary(df):
    tr = df[df.partition == "train"]
    rules = {
        "X < 0": (tr.FIX_X < 0).sum(),
        "X ≥ 1024": (tr.FIX_X >= SCREEN_W).sum(),
        "Y < 0": (tr.FIX_Y < 0).sum(),
        "Y ≥ 768": (tr.FIX_Y >= SCREEN_H).sum(),
        "duration ≤ 0": (tr.FIX_DURATION <= 0).sum(),
        "duration > 2000 ms": (tr.FIX_DURATION > 2000).sum(),
        "duration > 5000 ms": (tr.FIX_DURATION > 5000).sum(),
        "pupil ≤ 0": (tr.FIX_PUPIL <= 0).sum(),
    }
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(list(rules.keys())[::-1], list(rules.values())[::-1], color="#555")
    for i, v in enumerate(list(rules.values())[::-1]):
        ax.text(v, i, f" {v:,} ({v/len(tr)*100:.2f}%)", va="center")
    ax.set_xlabel("count"); ax.set_title(f"Outlier rule counts on train set (n={len(tr):,})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_outlier_summary.png")
    plt.close(fig)
    return pd.Series(rules, name="count")


def fig_scanpath_examples(df):
    """Sample scanpaths: first 2 HC and first 2 SZ subjects, one stimulus each."""
    tr = df[df.partition == "train"]
    imgs = image_categories().index
    np.random.seed(0)
    chosen = np.random.choice(imgs, size=3, replace=False)
    subj = {"HC": [s for s in train_subject_ids() if s < 200][:2],
            "SZ": [s for s in train_subject_ids() if s >= 200][:2]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for col, img in enumerate(chosen):
        for row, (gname, sids) in enumerate(subj.items()):
            ax = axes[row, col]
            for sid in sids:
                sub = tr[(tr.subject_id == sid) & (tr.IMAGE == img)].sort_values("FIX_INDEX")
                color = HC_COLOR if gname == "HC" else SZ_COLOR
                ax.plot(sub.FIX_X, sub.FIX_Y, "o-", ms=3, lw=1, color=color,
                        alpha=0.9, label=f"{gname}-{sid}")
                ax.plot(sub.FIX_X.iloc[0], sub.FIX_Y.iloc[0], "s", ms=7,
                        color=color)
            ax.invert_yaxis()
            ax.set_xlim(0, SCREEN_W); ax.set_ylim(SCREEN_H, 0)
            ax.set_title(f"{img.split('_')[0]}-{img}  ({image_categories()[img]})")
            if col == 0:
                ax.set_ylabel(gname)
            ax.legend(fontsize=7)
    fig.suptitle("Example scanpaths (square = first fixation)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_scanpath_examples.png")
    plt.close(fig)


def fig_spatial_features_by_group(df):
    tr = df[df.partition == "train"]
    per = (tr.groupby(["subject_id", "IMAGE", "label"])
           .agg(n_fix=("FIX_INDEX", "size"),
                mean_x=("FIX_X", "mean"), mean_y=("FIX_Y", "mean"),
                std_x=("FIX_X", "std"), std_y=("FIX_Y", "std"),
                dur_mean=("FIX_DURATION", "mean"),
                pupil_mean=("FIX_PUPIL", "mean"))
           .reset_index())
    metrics = ["n_fix", "std_x", "std_y", "dur_mean", "pupil_mean"]
    titles = ["# fixations / stimulus", "spread of X (px)", "spread of Y (px)",
              "mean duration (ms)", "mean pupil (a.u.)"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for ax, m, t in zip(axes, metrics, titles):
        data = [per[per.label == 0][m], per[per.label == 1][m]]
        bp = ax.boxplot(data, tick_labels=["HC", "SZ"], patch_artist=True,
                        widths=0.5)
        for patch, color in zip(bp["boxes"], [HC_COLOR, SZ_COLOR]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(t)
        t, p = stats.ttest_ind(*data, equal_var=False)
        ax.text(0.5, 0.95, f"p={p:.2g}", transform=ax.transAxes, ha="center",
                va="top", fontsize=8)
    fig.suptitle("Per-stimulus spatial/temporal statistics by group (train, 16k samples each)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_spatial_features_by_group.png")
    plt.close(fig)
    return per


def fig_category_comparison(df):
    tr = df[df.partition == "train"]
    per = (tr.groupby(["subject_id", "category", "label"])
           .agg(n_fix=("FIX_INDEX", "size"))
           .reset_index())
    cats = ["social", "natural", "synthetic", "manipulated"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(cats)); w = 0.35
    for i, (label, name, color) in enumerate([(0, "HC", HC_COLOR), (1, "SZ", SZ_COLOR)]):
        means = [per[(per.label == label) & (per.category == c)]["n_fix"].mean() for c in cats]
        stds = [per[(per.label == label) & (per.category == c)]["n_fix"].std() for c in cats]
        ax.bar(x + i * w - w / 2, means, w, yerr=stds, label=name, color=color,
               alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("mean fixations per subject-stimulus")
    ax.set_title("Fixations per stimulus category (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_category_comparison.png")
    plt.close(fig)


def main():
    print("loading raw dataset ...")
    df = load_full()
    n_subj_train = df[df.partition == "train"].subject_id.nunique()
    n_subj_test = df[df.partition == "test"].subject_id.nunique()
    print(f"  {len(df):,} fixations | train subjects={n_subj_train} test subjects={n_subj_test}")
    print(f"  stimuli per subject-train: {df[df.partition=='train'].groupby('subject_id').IMAGE.nunique().unique()}")
    print(f"  stimuli per subject-test : {df[df.partition=='test'].groupby('subject_id').IMAGE.nunique().unique()}")

    print("computing EDA statistics ...")
    subj_desc = fig_fixations_per_subject(df)
    cat_counts = fig_fixations_per_stimulus(df)
    quad = fig_density_maps(df)
    dur_desc = fig_duration_distribution(df)
    pup_desc = fig_pupil_distribution(df)
    out_sum = fig_outlier_summary(df)
    fig_scanpath_examples(df)
    per = fig_spatial_features_by_group(df)
    fig_category_comparison(df)

    labels = subject_labels()
    folds = official_folds()
    fold_composition = {k: {"HC": sum(1 for s in v if s < 200),
                            "SZ": sum(1 for s in v if s >= 200)} for k, v in folds.items()}

    # extra stats for markdown
    tr = df[df.partition == "train"]
    n_stim_missing = (tr.groupby("subject_id").IMAGE.nunique() < 100).sum()
    long_desc = {
        "n_train_subjects": n_subj_train, "n_test_subjects": n_subj_test,
        "n_train_fixations": len(tr),
        "n_test_fixations": len(df[df.partition == "test"]),
        "n_fix_mean": tr.groupby("subject_id").size().mean(),
        "n_fix_min": tr.groupby("subject_id").size().min(),
        "n_fix_max": tr.groupby("subject_id").size().max(),
        "n_stimuli": df.IMAGE.nunique(),
        "n_stim_missing_subjects": int(n_stim_missing),
        "subj_desc": subj_desc, "cat_counts": cat_counts, "quad": quad,
        "dur_desc": dur_desc, "pup_desc": pup_desc, "out_sum": out_sum,
        "per": per, "fold_composition": fold_composition,
    }
    pd.to_pickle(long_desc, DOCS_EDA / "eda_stats.pkl")
    print("saved stats ->", DOCS_EDA / "eda_stats.pkl")
    print("\n=== key stats ===")
    print(subj_desc.to_string())
    print("\ncategory fixation counts:\n", cat_counts.to_string())
    print("\nquadrant occupancy:\n", quad.to_string())
    print("\noutlier rule counts:\n", out_sum.to_string())
    print("\nfold composition:", fold_composition)


if __name__ == "__main__":
    main()
