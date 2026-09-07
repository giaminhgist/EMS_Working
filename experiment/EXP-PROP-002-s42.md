# EXP-PROP-002-s42 — proposal/mlp_norm01

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_norm01 --comparator mlp --pool attention --lambda_norm 0.1 --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Normative regularization pulls HC latent codes to the bank center; should stabilize training and lift AUC (proposal + lambda_norm=0.1).

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9541 ± 0.0326 |
| acc | 0.8438 ± 0.0541 |
| balanced_acc | 0.8297 ± 0.0658 |
| sen | 0.8190 ± 0.1593 |
| spe | 0.8403 ± 0.1279 |
| f1 | 0.8275 ± 0.0965 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9874 | 0.8500 | 0.8333 | 1.0000 | 0.6667 | 0.8800 |
| Set_1 | 0.9297 | 0.7750 | 0.7396 | 0.5625 | 0.9167 | 0.6667 |
| Set_2 | 0.9850 | 0.9250 | 0.9250 | 0.8500 | 1.0000 | 0.9189 |
| Set_3 | 0.9141 | 0.8250 | 0.8207 | 0.8636 | 0.7778 | 0.8444 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
