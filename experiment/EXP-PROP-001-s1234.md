# EXP-PROP-001-s1234 — proposal/mlp_attn

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Learned latent normative deviation with a learned comparator (the main proposal) beats all ablations under the official 4-fold protocol.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9424 ± 0.0297 |
| acc | 0.8562 ± 0.0596 |
| balanced_acc | 0.8499 ± 0.0669 |
| sen | 0.8256 ± 0.1371 |
| spe | 0.8743 ± 0.0253 |
| f1 | 0.8397 ± 0.0943 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9495 | 0.9000 | 0.8939 | 0.9545 | 0.8333 | 0.9130 |
| Set_1 | 0.8932 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9725 | 0.9250 | 0.9250 | 0.9500 | 0.9000 | 0.9268 |
| Set_3 | 0.9545 | 0.8250 | 0.8308 | 0.7727 | 0.8889 | 0.8293 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
