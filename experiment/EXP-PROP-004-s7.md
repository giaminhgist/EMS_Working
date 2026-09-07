# EXP-PROP-004-s7 — proposal/zsub_attn

Status: **done** · Seed 7 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation zsub_attn --comparator zsub --pool attention --fold all --epochs 150 --patience 30 --seed 7`

Hypothesis: Ablation: standardized latent subtraction as intermediate comparator variant.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9181 ± 0.0631 |
| acc | 0.8250 ± 0.0884 |
| balanced_acc | 0.8143 ± 0.0983 |
| sen | 0.7466 ± 0.1567 |
| spe | 0.8819 ± 0.1009 |
| f1 | 0.7969 ± 0.1257 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9773 | 0.9000 | 0.8990 | 0.9091 | 0.8889 | 0.9091 |
| Set_1 | 0.8542 | 0.7500 | 0.7083 | 0.5000 | 0.9167 | 0.6154 |
| Set_2 | 0.9850 | 0.9250 | 0.9250 | 0.8500 | 1.0000 | 0.9189 |
| Set_3 | 0.8561 | 0.7250 | 0.7247 | 0.7273 | 0.7222 | 0.7442 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
