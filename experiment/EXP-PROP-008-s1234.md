# EXP-PROP-008-s1234 — proposal/diff_mean

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation diff_mean --deviation diff --pool mean --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Unstandardized hard deviation (x-mu) retains feature scale and should be weaker than z-scores.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.8715 ± 0.0386 |
| acc | 0.8000 ± 0.0395 |
| balanced_acc | 0.7976 ± 0.0408 |
| sen | 0.7875 ± 0.1033 |
| spe | 0.8076 ± 0.1244 |
| f1 | 0.7949 ± 0.0380 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.8763 | 0.7500 | 0.7374 | 0.8636 | 0.6111 | 0.7917 |
| Set_1 | 0.8438 | 0.8250 | 0.8125 | 0.7500 | 0.8750 | 0.7742 |
| Set_2 | 0.9325 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8333 | 0.7750 | 0.7904 | 0.6364 | 0.9444 | 0.7568 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
