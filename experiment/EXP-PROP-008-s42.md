# EXP-PROP-008-s42 — proposal/diff_mean

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation diff_mean --deviation diff --pool mean --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Unstandardized hard deviation (x-mu) retains feature scale and should be weaker than z-scores.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.8684 ± 0.0382 |
| acc | 0.7687 ± 0.0446 |
| balanced_acc | 0.7709 ± 0.0512 |
| sen | 0.7918 ± 0.0962 |
| spe | 0.7500 ± 0.1375 |
| f1 | 0.7717 ± 0.0381 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.8636 | 0.7000 | 0.6869 | 0.8182 | 0.5556 | 0.7500 |
| Set_1 | 0.8542 | 0.7750 | 0.7812 | 0.8125 | 0.7500 | 0.7429 |
| Set_2 | 0.9300 | 0.8250 | 0.8250 | 0.9000 | 0.7500 | 0.8372 |
| Set_3 | 0.8258 | 0.7750 | 0.7904 | 0.6364 | 0.9444 | 0.7568 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
