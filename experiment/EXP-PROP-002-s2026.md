# EXP-PROP-002-s2026 — proposal/mlp_norm01

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_norm01 --comparator mlp --pool attention --lambda_norm 0.1 --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Normative regularization pulls HC latent codes to the bank center; should stabilize training and lift AUC (proposal + lambda_norm=0.1).

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9432 ± 0.0393 |
| acc | 0.6937 ± 0.1595 |
| balanced_acc | 0.7036 ± 0.1458 |
| sen | 0.6648 ± 0.3437 |
| spe | 0.7424 ± 0.3449 |
| f1 | 0.6302 ± 0.2748 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9848 | 0.8750 | 0.8813 | 0.8182 | 0.9444 | 0.8780 |
| Set_1 | 0.9062 | 0.8250 | 0.8125 | 0.7500 | 0.8750 | 0.7742 |
| Set_2 | 0.9800 | 0.5750 | 0.5750 | 1.0000 | 0.1500 | 0.7018 |
| Set_3 | 0.9015 | 0.5000 | 0.5455 | 0.0909 | 1.0000 | 0.1667 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
