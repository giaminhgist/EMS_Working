# EXP-PROP-007-s2024 — proposal/z_mean

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Hard z-deviation + mean pooling (fixed normative subtraction, no encoder/comparator) — the controlled hard-deviation ablation the learned pipeline must beat.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9164 ± 0.0543 |
| acc | 0.8000 ± 0.0848 |
| balanced_acc | 0.7966 ± 0.0940 |
| sen | 0.7759 ± 0.1466 |
| spe | 0.8174 ± 0.1111 |
| f1 | 0.7833 ± 0.1208 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9798 | 0.9250 | 0.9318 | 0.8636 | 1.0000 | 0.9268 |
| Set_1 | 0.8724 | 0.7000 | 0.6771 | 0.5625 | 0.7917 | 0.6000 |
| Set_2 | 0.9600 | 0.8250 | 0.8250 | 0.9500 | 0.7000 | 0.8444 |
| Set_3 | 0.8535 | 0.7500 | 0.7525 | 0.7273 | 0.7778 | 0.7619 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
