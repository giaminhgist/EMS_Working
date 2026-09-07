# EXP-PROP-002-s2024 — proposal/mlp_norm01

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_norm01 --comparator mlp --pool attention --lambda_norm 0.1 --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Normative regularization pulls HC latent codes to the bank center; should stabilize training and lift AUC (proposal + lambda_norm=0.1).

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9515 ± 0.0310 |
| acc | 0.8250 ± 0.0791 |
| balanced_acc | 0.8241 ± 0.0726 |
| sen | 0.8239 ± 0.1642 |
| spe | 0.8243 ± 0.2197 |
| f1 | 0.8220 ± 0.0709 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9823 | 0.9250 | 0.9217 | 0.9545 | 0.8889 | 0.9333 |
| Set_1 | 0.9219 | 0.8750 | 0.8542 | 0.7500 | 0.9583 | 0.8276 |
| Set_2 | 0.9825 | 0.7250 | 0.7250 | 1.0000 | 0.4500 | 0.7843 |
| Set_3 | 0.9192 | 0.7750 | 0.7955 | 0.5909 | 1.0000 | 0.7429 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
