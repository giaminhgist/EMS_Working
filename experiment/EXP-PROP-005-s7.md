# EXP-PROP-005-s7 — proposal/mlp_mean

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_mean --comparator mlp --pool mean --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Ablation: mean pooling over stimuli instead of attention.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9462 ± 0.0436 |
| acc | 0.8563 ± 0.0596 |
| balanced_acc | 0.8593 ± 0.0599 |
| sen | 0.8270 ± 0.0796 |
| spe | 0.8917 ± 0.0682 |
| f1 | 0.8488 ± 0.0654 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 1.0000 | 0.9000 | 0.9091 | 0.8182 | 1.0000 | 0.9000 |
| Set_1 | 0.8958 | 0.8250 | 0.8229 | 0.8125 | 0.8333 | 0.7879 |
| Set_2 | 0.9775 | 0.9250 | 0.9250 | 0.9500 | 0.9000 | 0.9268 |
| Set_3 | 0.9116 | 0.7750 | 0.7803 | 0.7273 | 0.8333 | 0.7805 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
