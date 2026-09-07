# EXP-PROP-007-s42 — proposal/z_mean

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Hard z-deviation + mean pooling (fixed normative subtraction, no encoder/comparator) — the controlled hard-deviation ablation the learned pipeline must beat.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9009 ± 0.0667 |
| acc | 0.8125 ± 0.0800 |
| balanced_acc | 0.8146 ± 0.0790 |
| sen | 0.8216 ± 0.0834 |
| spe | 0.8076 ± 0.0859 |
| f1 | 0.8087 ± 0.0928 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9722 | 0.9250 | 0.9268 | 0.9091 | 0.9444 | 0.9302 |
| Set_1 | 0.8281 | 0.7250 | 0.7292 | 0.7500 | 0.7083 | 0.6857 |
| Set_2 | 0.9625 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8409 | 0.7500 | 0.7525 | 0.7273 | 0.7778 | 0.7619 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
