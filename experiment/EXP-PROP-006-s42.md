# EXP-PROP-006-s42 — proposal/mlp_deepset

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_deepset --comparator mlp --pool deepset --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: DeepSets pooling (mean||max) as a set-aggregation alternative.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9295 ± 0.0484 |
| acc | 0.7937 ± 0.0817 |
| balanced_acc | 0.7994 ± 0.0765 |
| sen | 0.6301 ± 0.1421 |
| spe | 0.9688 ± 0.0541 |
| f1 | 0.7441 ± 0.1092 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9520 | 0.7750 | 0.7955 | 0.5909 | 1.0000 | 0.7429 |
| Set_1 | 0.8464 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9675 | 0.9250 | 0.9250 | 0.8500 | 1.0000 | 0.9189 |
| Set_3 | 0.9520 | 0.7000 | 0.7273 | 0.4545 | 1.0000 | 0.6250 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
