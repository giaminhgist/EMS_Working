# EXP-PROP-003-s2024 — proposal/sub_attn

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation sub_attn --comparator sub --pool attention --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Ablation: plain latent subtraction (no comparator) isolates the comparator's contribution.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9254 ± 0.0519 |
| acc | 0.8250 ± 0.0771 |
| balanced_acc | 0.8159 ± 0.0818 |
| sen | 0.8256 ± 0.1371 |
| spe | 0.8063 ± 0.0819 |
| f1 | 0.8158 ± 0.0961 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9747 | 0.9000 | 0.8939 | 0.9545 | 0.8333 | 0.9130 |
| Set_1 | 0.8984 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9750 | 0.9000 | 0.9000 | 0.9500 | 0.8500 | 0.9048 |
| Set_3 | 0.8535 | 0.7250 | 0.7197 | 0.7727 | 0.6667 | 0.7556 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
