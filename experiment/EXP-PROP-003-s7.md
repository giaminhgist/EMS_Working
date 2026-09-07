# EXP-PROP-003-s7 — proposal/sub_attn

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation sub_attn --comparator sub --pool attention --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Ablation: plain latent subtraction (no comparator) isolates the comparator's contribution.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9130 ± 0.0614 |
| acc | 0.7938 ± 0.0569 |
| balanced_acc | 0.7906 ± 0.0563 |
| sen | 0.7688 ± 0.1619 |
| spe | 0.8125 ± 0.0533 |
| f1 | 0.7756 ± 0.0919 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9646 | 0.8500 | 0.8434 | 0.9091 | 0.7778 | 0.8696 |
| Set_1 | 0.8411 | 0.7500 | 0.7292 | 0.6250 | 0.8333 | 0.6667 |
| Set_2 | 0.9825 | 0.8500 | 0.8500 | 0.9500 | 0.7500 | 0.8636 |
| Set_3 | 0.8636 | 0.7250 | 0.7399 | 0.5909 | 0.8889 | 0.7027 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
