# EXP-PROP-004-s1234 — proposal/zsub_attn

Status: **done** · Seed 1234 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation zsub_attn --comparator zsub --pool attention --fold all --epochs 150 --patience 30 --seed 1234`

Hypothesis: Ablation: standardized latent subtraction as intermediate comparator variant.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9258 ± 0.0513 |
| acc | 0.8313 ± 0.0693 |
| balanced_acc | 0.8292 ± 0.0706 |
| sen | 0.7688 ± 0.1619 |
| spe | 0.8896 ± 0.0346 |
| f1 | 0.8064 ± 0.1012 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9646 | 0.9000 | 0.8990 | 0.9091 | 0.8889 | 0.9091 |
| Set_1 | 0.8672 | 0.7750 | 0.7500 | 0.6250 | 0.8750 | 0.6897 |
| Set_2 | 0.9875 | 0.9000 | 0.9000 | 0.9500 | 0.8500 | 0.9048 |
| Set_3 | 0.8838 | 0.7500 | 0.7677 | 0.5909 | 0.9444 | 0.7222 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
