# EXP-PROP-006-s2024 — proposal/mlp_deepset

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_deepset --comparator mlp --pool deepset --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: DeepSets pooling (mean||max) as a set-aggregation alternative.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9486 ± 0.0439 |
| acc | 0.8688 ± 0.0715 |
| balanced_acc | 0.8675 ± 0.0699 |
| sen | 0.8239 ± 0.1207 |
| spe | 0.9111 ± 0.0252 |
| f1 | 0.8571 ± 0.0802 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9773 | 0.8750 | 0.8763 | 0.8636 | 0.8889 | 0.8837 |
| Set_1 | 0.8802 | 0.8500 | 0.8333 | 0.7500 | 0.9167 | 0.8000 |
| Set_2 | 0.9950 | 0.9750 | 0.9750 | 1.0000 | 0.9500 | 0.9756 |
| Set_3 | 0.9419 | 0.7750 | 0.7854 | 0.6818 | 0.8889 | 0.7692 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
