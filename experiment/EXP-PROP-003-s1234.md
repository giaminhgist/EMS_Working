# EXP-PROP-003-s1234 — proposal/sub_attn

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation sub_attn --comparator sub --pool attention --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Ablation: plain latent subtraction (no comparator) isolates the comparator's contribution.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9176 ± 0.0538 |
| acc | 0.8250 ± 0.0500 |
| balanced_acc | 0.8231 ± 0.0501 |
| sen | 0.8455 ± 0.0857 |
| spe | 0.8007 ± 0.0204 |
| f1 | 0.8226 ± 0.0675 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9621 | 0.8750 | 0.8712 | 0.9091 | 0.8333 | 0.8889 |
| Set_1 | 0.8646 | 0.7750 | 0.7708 | 0.7500 | 0.7917 | 0.7273 |
| Set_2 | 0.9800 | 0.8750 | 0.8750 | 0.9500 | 0.8000 | 0.8837 |
| Set_3 | 0.8636 | 0.7750 | 0.7753 | 0.7727 | 0.7778 | 0.7907 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
