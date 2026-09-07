# EXP-PROP-008-s2026 — proposal/diff_mean

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation diff_mean --deviation diff --pool mean --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Unstandardized hard deviation (x-mu) retains feature scale and should be weaker than z-scores.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.8714 ± 0.0516 |
| acc | 0.7875 ± 0.0375 |
| balanced_acc | 0.7884 ± 0.0394 |
| sen | 0.7761 ± 0.0966 |
| spe | 0.8007 ± 0.0984 |
| f1 | 0.7809 ± 0.0482 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.8712 | 0.7500 | 0.7424 | 0.8182 | 0.6667 | 0.7826 |
| Set_1 | 0.8385 | 0.7750 | 0.7708 | 0.7500 | 0.7917 | 0.7273 |
| Set_2 | 0.9550 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8207 | 0.7750 | 0.7904 | 0.6364 | 0.9444 | 0.7568 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
