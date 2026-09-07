# EXP-PROP-009-s2024 — proposal/mahal_mean

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Hard deviation augmented with a per-stimulus Mahalanobis (shrinkage) scalar captures multivariate deviation and may help over z alone.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9106 ± 0.0556 |
| acc | 0.8125 ± 0.0650 |
| balanced_acc | 0.8113 ± 0.0655 |
| sen | 0.7935 ± 0.0897 |
| spe | 0.8292 ± 0.0532 |
| f1 | 0.8020 ± 0.0883 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9697 | 0.8750 | 0.8712 | 0.9091 | 0.8333 | 0.8889 |
| Set_1 | 0.8594 | 0.7250 | 0.7188 | 0.6875 | 0.7500 | 0.6667 |
| Set_2 | 0.9625 | 0.8750 | 0.8750 | 0.8500 | 0.9000 | 0.8718 |
| Set_3 | 0.8510 | 0.7750 | 0.7803 | 0.7273 | 0.8333 | 0.7805 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
