# EXP-PROP-007-s2026 — proposal/z_mean

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Hard z-deviation + mean pooling (fixed normative subtraction, no encoder/comparator) — the controlled hard-deviation ablation the learned pipeline must beat.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9055 ± 0.0593 |
| acc | 0.8000 ± 0.0500 |
| balanced_acc | 0.8003 ± 0.0490 |
| sen | 0.8102 ± 0.0732 |
| spe | 0.7903 ± 0.0305 |
| f1 | 0.7971 ± 0.0663 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9646 | 0.8500 | 0.8485 | 0.8636 | 0.8333 | 0.8636 |
| Set_1 | 0.8464 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7059 |
| Set_2 | 0.9650 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8460 | 0.7500 | 0.7525 | 0.7273 | 0.7778 | 0.7619 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
