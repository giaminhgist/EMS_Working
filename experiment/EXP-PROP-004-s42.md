# EXP-PROP-004-s42 — proposal/zsub_attn

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation zsub_attn --comparator zsub --pool attention --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Ablation: standardized latent subtraction as intermediate comparator variant.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9136 ± 0.0592 |
| acc | 0.8062 ± 0.0370 |
| balanced_acc | 0.8021 ± 0.0439 |
| sen | 0.8298 ± 0.1233 |
| spe | 0.7743 ± 0.1188 |
| f1 | 0.8029 ± 0.0684 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9596 | 0.8000 | 0.7828 | 0.9545 | 0.6111 | 0.8400 |
| Set_1 | 0.8333 | 0.7500 | 0.7396 | 0.6875 | 0.7917 | 0.6875 |
| Set_2 | 0.9800 | 0.8500 | 0.8500 | 0.9500 | 0.7500 | 0.8636 |
| Set_3 | 0.8813 | 0.8250 | 0.8359 | 0.7273 | 0.9444 | 0.8205 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
