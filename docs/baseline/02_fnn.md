# 02 — Basic FNN Baseline

A basic feed-forward network (FNN) is trained on the same hand-crafted features
to serve as the "simple deep baseline" of EMS-Baseline. Two input variants exist:

- **`fnn`** — input = `agg` (91 dims: mean+std per feature over stimuli + n_valid_stim)
- **`fnn_cat`** — input = `catagg` (181 dims: per-category means over the 4 stimulus
  categories + n_valid_stim)

## Architecture

```
Input x ∈ R^B×D                        D = 91 (fnn) or 181 (fnn_cat)
  │
  ├─ Linear(D → 128)  → BatchNorm1d(128) → ReLU → Dropout(0.3)
  ├─ Linear(128 → 64) → BatchNorm1d(64)  → ReLU → Dropout(0.3)
  ├─ Linear(64 → 32)  → BatchNorm1d(32)  → ReLU → Dropout(0.3)
  └─ Linear(32 → 1)
  │
Sigmoid → ŷ ∈ R^B×1                     P(SZ | subject)
```

## Tensor shapes

| Stage | Input | Output |
|---|---|---|
| input | (B, D) | — |
| fc1 | (B, 91) or (B, 181) | (B, 128) |
| bn1 | (B, 128) | (B, 128) |
| relu1 | (B, 128) | (B, 128) |
| drop1 | (B, 128) | (B, 128) |
| fc2 | (B, 128) | (B, 64) |
| bn2 | (B, 64) | (B, 64) |
| relu2 | (B, 64) | (B, 64) |
| drop2 | (B, 64) | (B, 64) |
| fc3 | (B, 64) | (B, 32) |
| bn3 | (B, 32) | (B, 32) |
| relu3 | (B, 32) | (B, 32) |
| drop3 | (B, 32) | (B, 32) |
| head | (B, 32) | (B, 1) |
| sigmoid | (B, 1) | (B, 1) |

Trainable parameters (fnn, D=91): 91·128+128 + 128·64+64 + 64·32+32 + 32·1+1
≈ **22,049** (plus ~448 BN scale/shift).

## Training configuration

| Hyperparameter | Value |
|---|---|
| Loss | Binary cross-entropy (`torch.nn.BCELoss` on sigmoid output) |
| Optimizer | Adam, lr = 1e-3, weight_decay = 1e-4 |
| Batch size | 16 |
| Max epochs | 150 |
| Early stopping | patience 20 epochs on the inner validation AUC |
| Input preprocessing | median imputation + z-score standardization, fitted on training subjects only (same as ML methods) |
| Device | CUDA (NVIDIA A40) if available, else CPU |
| Seed | 42 (torch + numpy + split) |

## Evaluation

- Threshold **0.5** on the sigmoid output for Acc/Sen/Spe/Pre/F1; AUC is
  threshold-free.
- Protocol P1: per fold, the model is trained on the 3 training folds with a
  stratified 75/25 inner split for early stopping, then evaluated on the held-out
  fold; the 4 fold metrics are averaged.
- Protocol P2: trained on the 120 train/val subjects (90 train / 30 inner val),
  evaluated on the 40 held-out subjects.

## Implementation

`src/baseline/models.py` — class `BasicFNN`, `train_fnn`, `predict_fnn`.
