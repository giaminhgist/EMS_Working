# EXP-PROP-009-s7 — proposal/mahal_mean

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Hard deviation augmented with a per-stimulus Mahalanobis (shrinkage) scalar captures multivariate deviation and may help over z alone.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9112 ± 0.0630 |
| acc | 0.7625 ± 0.1606 |
| balanced_acc | 0.7812 ± 0.1303 |
| sen | 0.8798 ± 0.0903 |
| spe | 0.6826 ± 0.2757 |
| f1 | 0.7927 ± 0.1237 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9848 | 0.9250 | 0.9217 | 0.9545 | 0.8889 | 0.9333 |
| Set_1 | 0.8542 | 0.5000 | 0.5729 | 0.9375 | 0.2083 | 0.6000 |
| Set_2 | 0.9625 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8434 | 0.7750 | 0.7803 | 0.7273 | 0.8333 | 0.7805 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
