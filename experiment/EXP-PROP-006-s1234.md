# EXP-PROP-006-s1234 — proposal/mlp_deepset

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_deepset --comparator mlp --pool deepset --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: DeepSets pooling (mean||max) as a set-aggregation alternative.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9514 ± 0.0263 |
| acc | 0.8313 ± 0.0541 |
| balanced_acc | 0.8342 ± 0.0451 |
| sen | 0.7545 ± 0.1434 |
| spe | 0.9139 ± 0.0551 |
| f1 | 0.8100 ± 0.0712 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9495 | 0.7500 | 0.7727 | 0.5455 | 1.0000 | 0.7059 |
| Set_1 | 0.9193 | 0.8500 | 0.8333 | 0.7500 | 0.9167 | 0.8000 |
| Set_2 | 0.9925 | 0.9000 | 0.9000 | 0.9500 | 0.8500 | 0.9048 |
| Set_3 | 0.9444 | 0.8250 | 0.8308 | 0.7727 | 0.8889 | 0.8293 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
