# EXP-PROP-006-s2026 — proposal/mlp_deepset

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_deepset --comparator mlp --pool deepset --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: DeepSets pooling (mean||max) as a set-aggregation alternative.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9130 ± 0.0604 |
| acc | 0.8250 ± 0.0848 |
| balanced_acc | 0.8255 ± 0.0899 |
| sen | 0.7676 ± 0.1025 |
| spe | 0.8833 ± 0.0806 |
| f1 | 0.8065 ± 0.1137 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9672 | 0.8750 | 0.8813 | 0.8182 | 0.9444 | 0.8780 |
| Set_1 | 0.8385 | 0.7000 | 0.6875 | 0.6250 | 0.7500 | 0.6250 |
| Set_2 | 0.9775 | 0.9250 | 0.9250 | 0.9000 | 0.9500 | 0.9231 |
| Set_3 | 0.8687 | 0.8000 | 0.8081 | 0.7273 | 0.8889 | 0.8000 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
