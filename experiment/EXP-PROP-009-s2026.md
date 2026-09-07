# EXP-PROP-009-s2026 — proposal/mahal_mean

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Hard deviation augmented with a per-stimulus Mahalanobis (shrinkage) scalar captures multivariate deviation and may help over z alone.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9061 ± 0.0623 |
| acc | 0.8063 ± 0.0778 |
| balanced_acc | 0.8079 ± 0.0773 |
| sen | 0.8102 ± 0.0974 |
| spe | 0.8056 ± 0.0810 |
| f1 | 0.8013 ± 0.0892 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9823 | 0.9250 | 0.9268 | 0.9091 | 0.9444 | 0.9302 |
| Set_1 | 0.8411 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7059 |
| Set_2 | 0.9525 | 0.8250 | 0.8250 | 0.9000 | 0.7500 | 0.8372 |
| Set_3 | 0.8485 | 0.7250 | 0.7298 | 0.6818 | 0.7778 | 0.7317 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
