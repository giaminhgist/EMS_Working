# EXP-PROP-004-s2026 — proposal/zsub_attn

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation zsub_attn --comparator zsub --pool attention --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Ablation: standardized latent subtraction as intermediate comparator variant.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9327 ± 0.0595 |
| acc | 0.8438 ± 0.1036 |
| balanced_acc | 0.8495 ± 0.0985 |
| sen | 0.8497 ± 0.1410 |
| spe | 0.8493 ± 0.1079 |
| f1 | 0.8357 ± 0.1140 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Set_1 | 0.8620 | 0.7500 | 0.7604 | 0.8125 | 0.7083 | 0.7222 |
| Set_2 | 0.9825 | 0.8750 | 0.8750 | 0.9500 | 0.8000 | 0.8837 |
| Set_3 | 0.8864 | 0.7500 | 0.7626 | 0.6364 | 0.8889 | 0.7368 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
