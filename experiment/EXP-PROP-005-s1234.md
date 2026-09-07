# EXP-PROP-005-s1234 — proposal/mlp_mean

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_mean --comparator mlp --pool mean --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Ablation: mean pooling over stimuli instead of attention.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9520 ± 0.0341 |
| acc | 0.8187 ± 0.0480 |
| balanced_acc | 0.8254 ± 0.0478 |
| sen | 0.8841 ± 0.1487 |
| spe | 0.7667 ± 0.1612 |
| f1 | 0.8257 ± 0.0533 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9949 | 0.8000 | 0.7778 | 1.0000 | 0.5556 | 0.8462 |
| Set_1 | 0.9115 | 0.8000 | 0.8333 | 1.0000 | 0.6667 | 0.8000 |
| Set_2 | 0.9750 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| Set_3 | 0.9268 | 0.7750 | 0.7904 | 0.6364 | 0.9444 | 0.7568 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
