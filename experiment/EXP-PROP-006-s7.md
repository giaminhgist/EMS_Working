# EXP-PROP-006-s7 — proposal/mlp_deepset

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_deepset --comparator mlp --pool deepset --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: DeepSets pooling (mean||max) as a set-aggregation alternative.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9173 ± 0.0597 |
| acc | 0.8125 ± 0.0650 |
| balanced_acc | 0.8061 ± 0.0750 |
| sen | 0.7261 ± 0.1716 |
| spe | 0.8861 ± 0.0237 |
| f1 | 0.7785 ± 0.1123 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9672 | 0.8500 | 0.8535 | 0.8182 | 0.8889 | 0.8571 |
| Set_1 | 0.8333 | 0.7500 | 0.7083 | 0.5000 | 0.9167 | 0.6154 |
| Set_2 | 0.9800 | 0.9000 | 0.9000 | 0.9500 | 0.8500 | 0.9048 |
| Set_3 | 0.8889 | 0.7500 | 0.7626 | 0.6364 | 0.8889 | 0.7368 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
