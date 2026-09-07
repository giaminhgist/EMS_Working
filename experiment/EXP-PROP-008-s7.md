# EXP-PROP-008-s7 — proposal/diff_mean

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation diff_mean --deviation diff --pool mean --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Unstandardized hard deviation (x-mu) retains feature scale and should be weaker than z-scores.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.8583 ± 0.0533 |
| acc | 0.7750 ± 0.0771 |
| balanced_acc | 0.7770 ± 0.0749 |
| sen | 0.7338 ± 0.1175 |
| spe | 0.8201 ± 0.1255 |
| f1 | 0.7648 ± 0.0683 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.8131 | 0.6750 | 0.6692 | 0.7273 | 0.6111 | 0.7111 |
| Set_1 | 0.8594 | 0.8500 | 0.8438 | 0.8125 | 0.8750 | 0.8125 |
| Set_2 | 0.9450 | 0.8500 | 0.8500 | 0.8500 | 0.8500 | 0.8500 |
| Set_3 | 0.8157 | 0.7250 | 0.7449 | 0.5455 | 0.9444 | 0.6857 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
