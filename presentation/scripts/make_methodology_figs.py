"""Figures 03 — methodology diagrams (editable vector output).

F03.01 architecture: the exact pipeline implemented in src/proposal/model.py
(verified against code): input [B,S,45] + mask, encoder 45→128→128, latent
HC bank (buffer, refreshed every epoch, no grad), comparator
[z‖μ‖z−μ‖z⊙μ] → 512→256→128→64, masked pooling (attention/mean/deepset),
head → P(SZ); hard-deviation branch feeds [z|x−μ|[z‖M]] ∈ R^45/46 through the
SAME pooling + head. Loss: BCE at the head + optional λ_norm on HC z (L2 to
bank), attached in _deviate.

F03.02 protocols: P1 official 4-fold (5 seeds), P2 held-out 120/40 (3 seeds,
inner 90/30), official test (train on 160, inner 75/25, 48 probabilities).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, HC_COLOR, SZ_COLOR, REF_COLOR, savefig  # noqa: E402

OUT = FIG / "03_methodology"

C_BOX = "#eef3f8"       # encoder
C_DEV = "#e8f4ee"       # deviation / learned path
C_BANK = "#f5efe8"      # bank
C_HEAD = "#f7f0f4"      # head
C_HARD = "#faf0f5"      # hard path
C_LOSS = "#fdf3f3"


def box(ax, x, y, w, h, text, fc, ec="#5a5a5a", fs=8.5, lw=1.1, rounded=True,
        text_fs=None):
    style = ("round,pad=0.02,rounding_size=0.06" if rounded else "square,pad=0.02")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=text_fs or fs, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.3, style="-|>", ls="-",
          connectionstyle=None, mutation_scale=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=lw,
                                 color=color, linestyle=ls, mutation_scale=13,
                                 connectionstyle=connectionstyle))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ---- learned path (top row) ----
    box(ax, 0.4, 9.6, 2.7, 2.2, "x$_{i,s}$\nstimulus features\n[B, S, 45]", "#f2f2f2")
    box(ax, 0.4, 6.0, 2.7, 2.2, "mask\n[B, S]", "#f2f2f2")
    box(ax, 0.4, 3.0, 2.7, 2.2, "input scaling\n(X−μ)/σ fit on\ntrain subjects only", "#f2f2f2", fs=8)
    box(ax, 3.9, 6.0, 3.2, 2.2, "shared encoder f$_θ$\n45→128→128\n(LN+GELU)", C_BOX)
    arrow(ax, 3.1, 10.7, 3.9, 7.6)
    box(ax, 7.9, 6.0, 3.0, 2.2, "z\n[B, S, 128]", "#eef0f2")
    arrow(ax, 7.1, 7.1, 7.9, 7.1)
    # bank
    box(ax, 7.9, 9.4, 3.0, 2.3, "HC latent bank\nμ$_s^z$, σ$_s^z$  [S, 128]\n(buffer, no grad)", C_BANK)
    arrow(ax, 10.4, 9.4, 10.4, 8.3, color=REF_COLOR, lw=1.2)
    ax.text(11.15, 8.85, "train-fold HC only,\nrefreshed every epoch", fontsize=7.5,
            color="#6b6b6b")
    box(ax, 11.6, 6.0, 3.4, 2.2, "comparator g$_φ$\n[z ‖ μ$^z$ ‖ z−μ$^z$ ‖ z⊙μ$^z$]\n512→256→128→64", C_DEV)
    arrow(ax, 10.9, 7.1, 11.6, 7.1)
    box(ax, 15.7, 6.0, 2.2, 2.2, "d\n[B, S, 64]\n× mask", "#eef0f2")
    arrow(ax, 15.0, 7.1, 15.7, 7.1)
    box(ax, 18.6, 6.0, 3.0, 2.2, "stimulus pooling\nattention | mean |\ndeepset (masked)", C_BOX)
    arrow(ax, 17.9, 7.1, 18.6, 7.1)
    box(ax, 22.3, 6.0, 2.0, 2.2, "h\n[B, 64]", "#eef0f2")
    arrow(ax, 21.6, 7.1, 22.3, 7.1)
    box(ax, 25.0, 6.0, 2.6, 2.2, "head\nBN+ReLU+Drop\n→ sigmoid", C_HEAD)
    arrow(ax, 24.3, 7.1, 25.0, 7.1)
    box(ax, 25.0, 9.4, 2.6, 1.4, "P(SZ)", "#eef0f2", fs=9)
    arrow(ax, 26.3, 8.2, 26.3, 9.4)
    # losses
    box(ax, 23.6, 2.6, 4.0, 1.5, "L$_{cls}$ = BCE(P(SZ), y)", C_LOSS)
    arrow(ax, 25.9, 6.0, 25.9, 4.1, color="#b05050", lw=1.3)
    box(ax, 7.9, 2.6, 4.6, 1.5, "L$_{norm}$ = λ·mean$_{HC,s}$‖z−μ$^z$‖²\n(per batch, graph-attached)", C_LOSS, fs=7.8)
    arrow(ax, 9.4, 6.0, 9.4, 4.1, color="#b05050", lw=1.3, style="-|>")
    ax.text(27.2, 3.35, "losses act here;\nbank has no gradient", fontsize=7.5,
            color="#8a8a8a", va="center")

    # ---- hard-deviation path (bottom row) ----
    box(ax, 3.9, 0.6, 3.2, 2.0, "HC stimulus norms\nμ$_s$, σ$_s$ (σ$_{shrink}$)\n[100, 45] from train HC", C_HARD, fs=7.8)
    box(ax, 7.9, 0.6, 4.4, 2.0, "hard deviation (no learning)\nz = (x−μ)/σ   |   diff = x−μ\nmahal = [z ‖ M$_s$], M$_s$=RMS((x−μ)/σ$_{shrink}$)", C_HARD, fs=7.6)
    box(ax, 13.1, 0.6, 2.2, 2.0, "D\n[B, S, 45|46]\n× mask", "#eef0f2", fs=8.5)
    arrow(ax, 7.1, 1.6, 7.9, 1.6)
    arrow(ax, 12.3, 1.6, 13.1, 1.6)
    arrow(ax, 15.3, 1.6, 18.6, 5.6, color="#9c5a7d", lw=1.3,
          connectionstyle="arc3,rad=-0.28")
    ax.text(16.6, 3.3, "SAME pooling + head\n(ablation: only the deviation\ncomputation changes)", fontsize=7.8,
            color="#7a4a63")
    # input feeding hard norms
    arrow(ax, 1.75, 9.6, 4.9, 1.9, color="#9c5a7d", lw=1.1,
          connectionstyle="arc3,rad=0.25")
    ax.text(0.55, 11.9, "raw features (no encoder)", fontsize=7.5, color="#7a4a63")

    # comparator variants
    box(ax, 11.6, 3.0, 3.4, 1.5, "sub: proj(z−μ)   zsub: proj((z−μ)/σ)\n(linear 128→64)", "#f5f5f5", fs=7.6)
    arrow(ax, 12.5, 6.0, 12.5, 4.5, color="#888888", lw=1.0, style="-|>")

    ax.set_title("Learned Stimulus-Conditioned Normative Modeling — architecture "
                 "(shapes and formulas exactly as in src/proposal/model.py)", fontsize=11.5)
    savefig(fig, OUT, "F03.01_architecture")


def fig_protocols():
    fig, ax = plt.subplots(figsize=(13.8, 5.0))
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # P1
    box(ax, 0.3, 6.6, 2.2, 2.6, "P1 — official\n4-fold CV\nSet_0..3\n160 labelled\nsubjects", "#eef3f8", fs=8.5)
    x = 3.0
    for i, fold in enumerate(["Set_0", "Set_1", "Set_2", "Set_3"]):
        box(ax, x, 6.6 + (0.6 if i % 2 == 0 else -0.6), 2.2, 1.9,
            f"fold {fold}\ntrain 120 / val 40", "#ffffff", ec="#5a5a5a", fs=8)
        arrow(ax, x + 2.2, 7.7, x + 2.65, 7.7)
        x += 2.75
    box(ax, x, 6.6, 3.4, 2.2, "val AUC selects epoch\n(early stop patience 30)\nbest.pt → val predictions", "#ffffff", ec="#5a5a5a", fs=7.8)
    ax.text(14.5, 5.7, "5 seeds: 42 / 1234 / 2024 / 2026 / 7  →  mean ± SD over seeds of the 4-fold mean",
            fontsize=8, ha="center", color="#444444")

    # P2
    box(ax, 0.3, 2.6, 2.2, 2.4, "P2 — held-out\n120 / 40 split\n(stratified)", "#eef3f8", fs=8.5)
    box(ax, 3.0, 2.6, 2.6, 2.4, "120 train/val\n→ inner 90/30\n(early stop)", "#ffffff", ec="#5a5a5a", fs=8)
    arrow(ax, 5.6, 3.8, 6.0, 3.8)
    box(ax, 6.0, 2.6, 2.6, 2.4, "40 held-out\nsubjects\nreal labels", "#ffffff", ec=SZ_COLOR, fs=8)
    arrow(ax, 8.6, 3.8, 9.0, 3.8)
    box(ax, 9.0, 2.6, 3.2, 2.4, "test AUC / Acc /\nBalAcc, threshold 0.5\n3 seeds: 42 / 2024 / 2026", "#ffffff", ec="#5a5a5a", fs=7.8)

    # official test
    box(ax, 13.4, 2.6, 2.6, 2.4, "Official test\n48 subjects\nlabels withheld", "#eef3f8", fs=8.5)
    arrow(ax, 16.0, 3.8, 16.4, 3.8)
    box(ax, 16.4, 2.6, 3.6, 2.4, "retrain on all 160\n(inner 75/25 early stop)\n→ probabilities only", "#ffffff", ec=REF_COLOR, fs=8)
    arrow(ax, 20.0, 3.8, 20.4, 3.8)
    box(ax, 20.4, 2.6, 3.2, 2.4, "official_test_preds.csv\nTest_000..047 → P(SZ)\n(benchmark format)", "#ffffff", ec="#5a5a5a", fs=7.8)

    # leakage guard strip
    box(ax, 24.6, 6.6, 3.0, 2.6, "leakage guard:\nall normative stats\nfit on train-fold\nHC only", C_BANK, fs=8)
    arrow(ax, 24.6, 7.9, 22.9, 7.9, color=REF_COLOR, lw=1.1, style="-|>")

    ax.set_title("Evaluation protocols (as implemented in src/data/common.py, "
                 "src/proposal/train.py, src/test_model/eval.py)", fontsize=11.5)
    savefig(fig, OUT, "F03.02_protocols")


if __name__ == "__main__":
    fig_architecture()
    fig_protocols()
    print("done 03_methodology")
