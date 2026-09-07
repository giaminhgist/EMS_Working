# EXP-PROP-001-s42 — proposal/mlp_attn

Status: **done** · Seed 42 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all --epochs 150 --patience 30 --seed 42`

Hypothesis: Learned latent normative deviation with a learned comparator (the main proposal) beats all ablations under the official 4-fold protocol.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9415 ± 0.0219 |
| acc | 0.8125 ± 0.0625 |
| balanced_acc | 0.8217 ± 0.0599 |
| sen | 0.8594 ± 0.1375 |
| spe | 0.7840 ± 0.1139 |
| f1 | 0.8158 ± 0.0697 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9419 | 0.9000 | 0.9040 | 0.8636 | 0.9444 | 0.9048 |
| Set_1 | 0.9349 | 0.8000 | 0.8229 | 0.9375 | 0.7083 | 0.7895 |
| Set_2 | 0.9750 | 0.8250 | 0.8250 | 1.0000 | 0.6500 | 0.8511 |
| Set_3 | 0.9141 | 0.7250 | 0.7348 | 0.6364 | 0.8333 | 0.7179 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
