# EXP-PROP-007-s1234 — proposal/z_mean

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Hard z-deviation + mean pooling (fixed normative subtraction, no encoder/comparator) — the controlled hard-deviation ablation the learned pipeline must beat.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9069 ± 0.0549 |
| acc | 0.8063 ± 0.0569 |
| balanced_acc | 0.8082 ± 0.0551 |
| sen | 0.8372 ± 0.0738 |
| spe | 0.7792 ± 0.0692 |
| f1 | 0.8080 ± 0.0646 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9722 | 0.9000 | 0.8990 | 0.9091 | 0.8889 | 0.9091 |
| Set_1 | 0.8568 | 0.7750 | 0.7812 | 0.8125 | 0.7500 | 0.7429 |
| Set_2 | 0.9500 | 0.8000 | 0.8000 | 0.9000 | 0.7000 | 0.8182 |
| Set_3 | 0.8485 | 0.7500 | 0.7525 | 0.7273 | 0.7778 | 0.7619 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
