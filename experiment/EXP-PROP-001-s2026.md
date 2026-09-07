# EXP-PROP-001-s2026 — proposal/mlp_attn

Status: **done** · Seed 2026 · Protocol: official 4-fold CV (Set_0..3, threshold 0.5)

## Setup

Command: `python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all --epochs 150 --patience 30 --seed 2026`

Hypothesis: Learned latent normative deviation with a learned comparator (the main proposal) beats all ablations under the official 4-fold protocol.

## Results (validation folds, mean ± std over 4 folds)

| Metric | Mean ± std |
|---|---|
| auc | 0.9443 ± 0.0420 |
| acc | 0.7812 ± 0.0569 |
| balanced_acc | 0.7713 ± 0.0675 |
| sen | 0.7614 ± 0.2392 |
| spe | 0.7812 ± 0.1849 |
| f1 | 0.7544 ± 0.1144 |

| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |
|---|---|---|---|---|---|---|
| Set_0 | 0.9899 | 0.7750 | 0.7500 | 1.0000 | 0.5000 | 0.8302 |
| Set_1 | 0.8854 | 0.7250 | 0.6875 | 0.5000 | 0.8750 | 0.5926 |
| Set_2 | 0.9775 | 0.8750 | 0.8750 | 1.0000 | 0.7500 | 0.8889 |
| Set_3 | 0.9242 | 0.7500 | 0.7727 | 0.5455 | 1.0000 | 0.7059 |

## Observations / Conclusion

_To be filled after analysis (see experiment_tracker.md)._
