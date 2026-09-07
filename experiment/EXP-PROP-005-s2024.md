# EXP-PROP-005-s2024 — proposal/mlp_mean

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_mean --comparator mlp --pool mean --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Ablation: mean pooling over stimuli instead of attention.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9458 ± 0.0329 |
| acc | 0.8625 ± 0.0545 |
| balanced_acc | 0.8669 ± 0.0467 |
| sen | 0.8435 ± 0.1398 |
| spe | 0.8903 ± 0.0570 |
| f1 | 0.8537 ± 0.0639 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9798 | 0.9250 | 0.9167 | 1.0000 | 0.8333 | 0.9362 |
| Set_1 | 0.9271 | 0.8750 | 0.8854 | 0.9375 | 0.8333 | 0.8571 |
| Set_2 | 0.9750 | 0.8750 | 0.8750 | 0.8000 | 0.9500 | 0.8649 |
| Set_3 | 0.9015 | 0.7750 | 0.7904 | 0.6364 | 0.9444 | 0.7568 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
