# EXP-PROP-001-s2024 — proposal/mlp_attn

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Learned latent normative deviation with a learned comparator (the main proposal) beats all ablations under the official 4-fold protocol.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9388 ± 0.0310 |
| acc | 0.7875 ± 0.0760 |
| balanced_acc | 0.7970 ± 0.0709 |
| sen | 0.8253 ± 0.2166 |
| spe | 0.7688 ± 0.1999 |
| f1 | 0.7829 ± 0.0984 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9571 | 0.8750 | 0.8712 | 0.9091 | 0.8333 | 0.8889 |
| Set_1 | 0.9141 | 0.8500 | 0.8646 | 0.9375 | 0.7917 | 0.8333 |
| Set_2 | 0.9800 | 0.7250 | 0.7250 | 1.0000 | 0.4500 | 0.7843 |
| Set_3 | 0.9040 | 0.7000 | 0.7273 | 0.4545 | 1.0000 | 0.6250 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
