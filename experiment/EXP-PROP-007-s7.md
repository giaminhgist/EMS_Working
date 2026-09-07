# EXP-PROP-007-s7 — proposal/z_mean

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Hard z-deviation + mean pooling (fixed normative subtraction, no encoder/comparator) — the controlled hard-deviation ablation the learned pipeline must beat.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9029 ± 0.0608 |
| acc | 0.8188 ± 0.0758 |
| balanced_acc | 0.8129 ± 0.0807 |
| sen | 0.7903 ± 0.1198 |
| spe | 0.8354 ± 0.0829 |
| f1 | 0.8053 ± 0.0941 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9697 | 0.9250 | 0.9268 | 0.9091 | 0.9444 | 0.9302 |
| Set_1 | 0.8411 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9575 | 0.8500 | 0.8500 | 0.9000 | 0.8000 | 0.8571 |
| Set_3 | 0.8434 | 0.7250 | 0.7247 | 0.7273 | 0.7222 | 0.7442 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
