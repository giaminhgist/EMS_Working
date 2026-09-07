# EXP-PROP-008-s2024 — proposal/diff_mean

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation diff_mean --deviation diff --pool mean --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Unstandardized hard deviation (x-mu) retains feature scale and should be weaker than z-scores.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.8482 ± 0.0550 |
| acc | 0.7000 ± 0.1299 |
| balanced_acc | 0.6847 ± 0.1462 |
| sen | 0.8486 ± 0.0578 |
| spe | 0.5208 ± 0.3052 |
| f1 | 0.7492 ± 0.0659 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.7879 | 0.6250 | 0.6086 | 0.7727 | 0.4444 | 0.6939 |
| Set_1 | 0.8516 | 0.8250 | 0.8229 | 0.8125 | 0.8333 | 0.7879 |
| Set_2 | 0.9350 | 0.8250 | 0.8250 | 0.9000 | 0.7500 | 0.8372 |
| Set_3 | 0.8182 | 0.5250 | 0.4823 | 0.9091 | 0.0556 | 0.6780 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
