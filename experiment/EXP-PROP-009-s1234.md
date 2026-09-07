# EXP-PROP-009-s1234 — proposal/mahal_mean

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Hard deviation augmented with a per-stimulus Mahalanobis (shrinkage) scalar captures multivariate deviation and may help over z alone.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9170 ± 0.0534 |
| acc | 0.8250 ± 0.0685 |
| balanced_acc | 0.8255 ± 0.0663 |
| sen | 0.8330 ± 0.0966 |
| spe | 0.8181 ± 0.0505 |
| f1 | 0.8192 ± 0.0849 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9798 | 0.9250 | 0.9217 | 0.9545 | 0.8889 | 0.9333 |
| Set_1 | 0.8620 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7059 |
| Set_2 | 0.9600 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8662 | 0.7750 | 0.7803 | 0.7273 | 0.8333 | 0.7805 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
