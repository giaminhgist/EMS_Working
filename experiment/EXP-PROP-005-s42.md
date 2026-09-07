# EXP-PROP-005-s42 — proposal/mlp_mean

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_mean --comparator mlp --pool mean --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Ablation: mean pooling over stimuli instead of attention.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9347 ± 0.0489 |
| acc | 0.8500 ± 0.0791 |
| balanced_acc | 0.8458 ± 0.0785 |
| sen | 0.8298 ± 0.1463 |
| spe | 0.8618 ± 0.0216 |
| f1 | 0.8362 ± 0.0964 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9848 | 0.9500 | 0.9444 | 1.0000 | 0.8889 | 0.9565 |
| Set_1 | 0.8724 | 0.8000 | 0.7812 | 0.6875 | 0.8750 | 0.7333 |
| Set_2 | 0.9800 | 0.9000 | 0.9000 | 0.9500 | 0.8500 | 0.9048 |
| Set_3 | 0.9015 | 0.7500 | 0.7576 | 0.6818 | 0.8333 | 0.7500 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
