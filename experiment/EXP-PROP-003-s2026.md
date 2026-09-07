# EXP-PROP-003-s2026 — proposal/sub_attn

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation sub_attn --comparator sub --pool attention --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Ablation: plain latent subtraction (no comparator) isolates the comparator's contribution.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9162 ± 0.0549 |
| acc | 0.7812 ± 0.0569 |
| balanced_acc | 0.7830 ± 0.0501 |
| sen | 0.7730 ± 0.1911 |
| spe | 0.7931 ± 0.1208 |
| f1 | 0.7632 ± 0.0971 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9596 | 0.8500 | 0.8384 | 0.9545 | 0.7222 | 0.8750 |
| Set_1 | 0.8490 | 0.7250 | 0.7188 | 0.6875 | 0.7500 | 0.6667 |
| Set_2 | 0.9800 | 0.8250 | 0.8250 | 0.9500 | 0.7000 | 0.8444 |
| Set_3 | 0.8763 | 0.7250 | 0.7500 | 0.5000 | 1.0000 | 0.6667 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
