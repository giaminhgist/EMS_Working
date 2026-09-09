"""Generate figure_manifest.csv and the contact-sheet gallery."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRES, FIG, savefig  # noqa: E402

# figure id -> metadata
META = [
    # (id, group, title, science question, main/supp, script, sources,
    #  protocol, seed/fold, sample note)
    ("F01.01", "01_dataset", "Dataset overview: subjects, stimuli, official folds",
     "Who is in the dataset, what do they view, how are folds composed?",
     "main", "make_dataset_figs.py", "metadata.csv; Images/*; fixations cache",
     "n/a", "n/a", "160 subjects / 100 stimuli"),
    ("F01.02", "01_dataset", "Gaze signatures: subject-level distributions + scanpaths",
     "How do groups differ in gaze statistics, and what do their scanpaths look like?",
     "main", "make_dataset_figs.py", "fixations cache (cleaned, train); Images/*",
     "n/a", "n/a", "n=80 HC / 80 SZ subjects; 4 stimuli x 2 subjects"),
    ("F02.01", "02_features", "The 45 hand-crafted features: groups + examples",
     "What features encode the gaze signal?",
     "main", "make_feature_figs.py", "src/features.py definitions",
     "n/a", "n/a", "45 features"),
    ("F02.02", "02_features", "HC-SZ effect sizes for all 45 features + rainclouds + category profiles",
     "Which features discriminate groups, and how does the signal depend on category?",
     "main", "make_feature_figs.py", "stimulus_features_train.pkl",
     "n/a", "n/a", "n=80/80 subjects; 4 features x 4 categories"),
    ("F03.01", "03_methodology", "Architecture diagram (matches model.py)",
     "How does the proposed pipeline work exactly?",
     "main", "make_methodology_figs.py", "src/proposal/model.py (verified)",
     "n/a", "n/a", "tensor shapes as in code"),
    ("F03.02", "03_methodology", "Evaluation protocols P1 / P2 / official test",
     "How are models trained and evaluated?",
     "main", "make_methodology_figs.py", "data/common.py; test_model/eval.py",
     "n/a", "n/a", "protocols"),
    ("F04.01", "04_ablations", "Learned vs hard normative deviation",
     "Does learning the deviation beat fixed hard deviations?",
     "main", "make_ablation_figs.py", "allfolds_summary.json (mlp_mean, z_mean, diff_mean, mahal_mean)",
     "P1 4-fold", "5 seeds", "paired t on 5 seed values"),
    ("F04.02", "04_ablations", "Hard-deviation chain: diff -> z -> mahal",
     "What do HC conditioning and the mahal scalar add step by step?",
     "main", "make_ablation_figs.py", "allfolds_summary.json (hard configs)",
     "P1 4-fold", "5 seeds", "paired t on 5 seed values"),
    ("F04.03", "04_ablations", "Learned comparator vs fixed latent subtraction",
     "Does the learned comparator g_phi help?",
     "main", "make_ablation_figs.py", "allfolds_summary.json (mlp/sub/zsub_attn)",
     "P1 4-fold", "5 seeds", "paired t on 5 seed values"),
    ("F04.04", "04_ablations", "Stimulus pooling: attention vs mean vs deepset",
     "How does stimulus-set aggregation affect results?",
     "main", "make_ablation_figs.py", "allfolds_summary.json (mlp_attn/mean/deepset)",
     "P1 4-fold", "5 seeds", "paired t on 5 seed values"),
    ("F04.05", "04_ablations", "HC concentration regularization lambda_norm",
     "Does lambda=0.1 help, and is there an accuracy trade-off?",
     "main", "make_ablation_figs.py", "allfolds_summary.json (mlp_attn, mlp_norm01)",
     "P1 4-fold", "5 seeds", "paired t on 5 seed values"),
    ("F04.06", "04_ablations", "Stimulus budget sensitivity K=25/50/100",
     "How many stimuli are needed, which deviation transfers better?",
     "supp", "make_ablation_figs.py", "outputs/evaluation/cross_stimulus.csv; xstim_* summaries",
     "P1 4-fold retrain per K", "seed 42", "same subset across folds; NOT an unseen-stimulus test"),
    ("F05.01", "05_latent_distribution", "Learned latent space vs the HC normative bank (PCA + heatmap + lambda diagnostics)",
     "Where do SZ live relative to the HC norm, and is the norm stable?",
     "main", "make_latent_figs.py", "cache/latent mlp_norm01 + mlp_attn seed42 (z, bank, ref_z)",
     "P1 out-of-fold", "seed 42", "160 eval + train-HC ref subjects x 100 stimuli"),
    ("F06.01", "06_importance_xai", "Feature + stimulus importance (permutation, attention, leave-one-out)",
     "Which features and stimuli drive the prediction?",
     "main", "make_xai_figs.py", "checkpoints mlp_norm01 + z_mean seed42 (permutation on Set_1 val); attn exports; LOO on Set_0 val model",
     "P1 fold Set_1 val / Set_0 val", "seed 42", "40 val subjects, 8-10 repeats"),
]


def build_manifest():
    rows = []
    for fid, grp, title, question, tier, script, sources, proto, seed, n in META:
        rows.append({
            "figure_id": fid,
            "group": grp,
            "title": title,
            "science_question": question,
            "tier": tier,
            "script": f"presentation/scripts/{script}",
            "png": f"presentation/figures/{grp}/{fid}_{title.split(':')[0].lower().replace(' ', '_')}",
            "source_data": sources,
            "protocol": proto,
            "seed_fold": seed,
            "sample_note": n,
        })
    # fill actual filenames from disk
    for r in rows:
        stem = r["figure_id"]
        cand = list(Path(PRES / "figures" / r["group"]).glob(f"{stem}_*.png"))
        r["png"] = str(cand[0].relative_to(PRES)) if cand else "MISSING"
    df = pd.DataFrame(rows)
    df.to_csv(PRES / "figure_manifest.csv", index=False)
    missing = df[df.png == "MISSING"]
    if len(missing):
        print("WARNING missing figures:\n", missing[["figure_id", "title"]])
    print(f"figure_manifest.csv: {len(df)} figures")
    return df


def build_gallery():
    df = build_manifest()
    n = len(df)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    with PdfPages(PRES / "figures_gallery.pdf") as pdf:
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.2, rows_n * 3.6))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(df.iterrows()):
            r = row[1]
            ax = axes[i // cols, i % cols]
            ax.axis("off")
            if r.png != "MISSING":
                img = plt.imread(PRES / r.png)
                ax.imshow(img)
            ax.set_title(f"{r.figure_id} [{r.tier}] — {r.group}\n{r.title}",
                         fontsize=7.5, pad=4)
        for j in range(n, rows_n * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle("EMS-Project presentation figures — contact sheet", fontsize=14, y=0.995)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print("figures_gallery.pdf written")


if __name__ == "__main__":
    build_manifest()
    build_gallery()
