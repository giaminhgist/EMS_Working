# EXP-PROP-003-s42 — proposal/sub_attn

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation sub_attn --comparator sub --pool attention --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Ablation: plain latent subtraction (no comparator) isolates the comparator's contribution.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9129 ± 0.0615 |
| acc | 0.8063 ± 0.0647 |
| balanced_acc | 0.7996 ± 0.0708 |
| sen | 0.7645 ± 0.1516 |
| spe | 0.8347 ± 0.1002 |
| f1 | 0.7869 ± 0.0931 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9621 | 0.9000 | 0.9040 | 0.8636 | 0.9444 | 0.9048 |
| Set_1 | 0.8333 | 0.7750 | 0.7396 | 0.5625 | 0.9167 | 0.6667 |
| Set_2 | 0.9825 | 0.8250 | 0.8250 | 0.9500 | 0.7000 | 0.8444 |
| Set_3 | 0.8737 | 0.7250 | 0.7298 | 0.6818 | 0.7778 | 0.7317 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
