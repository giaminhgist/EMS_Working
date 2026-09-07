# EXP-PROP-002-s1234 — proposal/mlp_norm01

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_norm01 --comparator mlp --pool attention --lambda_norm 0.1 --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Normative regularization pulls HC latent codes to the bank center; should stabilize training and lift AUC (proposal + lambda_norm=0.1).

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9529 ± 0.0201 |
| acc | 0.7812 ± 0.0974 |
| balanced_acc | 0.7835 ± 0.0857 |
| sen | 0.7858 ± 0.2736 |
| spe | 0.7812 ± 0.1849 |
| f1 | 0.7563 ± 0.1584 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9747 | 0.7750 | 0.7500 | 1.0000 | 0.5000 | 0.8302 |
| Set_1 | 0.9401 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8485 |
| Set_2 | 0.9700 | 0.8500 | 0.8500 | 0.9500 | 0.7500 | 0.8636 |
| Set_3 | 0.9268 | 0.6250 | 0.6591 | 0.3182 | 1.0000 | 0.4828 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
