# EXP-PROP-001-s7 — proposal/mlp_attn

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Learned latent normative deviation with a learned comparator (the main proposal) beats all ablations under the official 4-fold protocol.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9208 ± 0.0472 |
| acc | 0.8125 ± 0.0625 |
| balanced_acc | 0.7981 ± 0.0727 |
| sen | 0.8440 ± 0.1635 |
| spe | 0.7521 ± 0.1190 |
| f1 | 0.8050 ± 0.1005 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9621 | 0.8750 | 0.8662 | 0.9545 | 0.7778 | 0.8936 |
| Set_1 | 0.8646 | 0.7500 | 0.7188 | 0.5625 | 0.8750 | 0.6429 |
| Set_2 | 0.9725 | 0.8750 | 0.8750 | 0.9500 | 0.8000 | 0.8837 |
| Set_3 | 0.8838 | 0.7500 | 0.7323 | 0.9091 | 0.5556 | 0.8000 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
