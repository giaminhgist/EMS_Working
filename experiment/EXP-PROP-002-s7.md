# EXP-PROP-002-s7 — proposal/mlp_norm01

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_norm01 --comparator mlp --pool attention --lambda_norm 0.1 --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Normative regularization pulls HC latent codes to the bank center; should stabilize training and lift AUC (proposal + lambda_norm=0.1).

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9406 ± 0.0434 |
| acc | 0.8062 ± 0.0108 |
| balanced_acc | 0.7941 ± 0.0261 |
| sen | 0.7986 ± 0.1823 |
| spe | 0.7896 ± 0.1697 |
| f1 | 0.7931 ± 0.0625 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9899 | 0.8000 | 0.7778 | 1.0000 | 0.5556 | 0.8462 |
| Set_1 | 0.8958 | 0.8000 | 0.7604 | 0.5625 | 0.9583 | 0.6923 |
| Set_2 | 0.9775 | 0.8250 | 0.8250 | 0.9500 | 0.7000 | 0.8444 |
| Set_3 | 0.8990 | 0.8000 | 0.8131 | 0.6818 | 0.9444 | 0.7895 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
