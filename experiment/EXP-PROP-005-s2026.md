# EXP-PROP-005-s2026 — proposal/mlp_mean

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_mean --comparator mlp --pool mean --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Ablation: mean pooling over stimuli instead of attention.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9406 ± 0.0421 |
| acc | 0.8375 ± 0.0893 |
| balanced_acc | 0.8420 ± 0.0842 |
| sen | 0.8528 ± 0.1028 |
| spe | 0.8313 ± 0.0941 |
| f1 | 0.8362 ± 0.0925 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9874 | 0.9250 | 0.9217 | 0.9545 | 0.8889 | 0.9333 |
| Set_1 | 0.8958 | 0.7750 | 0.7917 | 0.8750 | 0.7083 | 0.7568 |
| Set_2 | 0.9775 | 0.9250 | 0.9250 | 0.9000 | 0.9500 | 0.9231 |
| Set_3 | 0.9015 | 0.7250 | 0.7298 | 0.6818 | 0.7778 | 0.7317 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
