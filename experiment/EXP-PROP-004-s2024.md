# EXP-PROP-004-s2024 — proposal/zsub_attn

Status: **done** · Seed 2024 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation zsub_attn --comparator zsub --pool attention --fold all --epochs 150 --patience 30 --seed 2024`

Hypothesis: Ablation: standardized latent subtraction as intermediate comparator variant.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9202 ± 0.0434 |
| acc | 0.8188 ± 0.0370 |
| balanced_acc | 0.8109 ± 0.0444 |
| sen | 0.8142 ± 0.1274 |
| spe | 0.8076 ± 0.0560 |
| f1 | 0.8085 ± 0.0735 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9470 | 0.8250 | 0.8157 | 0.9091 | 0.7222 | 0.8511 |
| Set_1 | 0.8802 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9775 | 0.8750 | 0.8750 | 0.9500 | 0.8000 | 0.8837 |
| Set_3 | 0.8763 | 0.8000 | 0.8030 | 0.7727 | 0.8333 | 0.8095 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
