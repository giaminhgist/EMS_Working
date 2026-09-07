# EXP-PROP-009-s42 — proposal/mahal_mean

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Hard deviation augmented with a per-stimulus Mahalanobis (shrinkage) scalar captures multivariate deviation and may help over z alone.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9062 ± 0.0576 |
| acc | 0.8000 ± 0.0848 |
| balanced_acc | 0.8009 ± 0.0848 |
| sen | 0.7977 ± 0.0878 |
| spe | 0.8042 ± 0.0857 |
| f1 | 0.7949 ± 0.0921 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9697 | 0.9250 | 0.9268 | 0.9091 | 0.9444 | 0.9302 |
| Set_1 | 0.8490 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7059 |
| Set_2 | 0.9575 | 0.8250 | 0.8250 | 0.8500 | 0.8000 | 0.8293 |
| Set_3 | 0.8485 | 0.7000 | 0.7020 | 0.6818 | 0.7222 | 0.7143 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
