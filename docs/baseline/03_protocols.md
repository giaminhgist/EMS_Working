# 03 — Evaluation Protocols

Every EMS-Baseline baseline is evaluated under two protocols; all metrics use the
fixed decision threshold **0.5** (except AUC, which is threshold-free).

## Protocol P1 — official EMS 4-fold cross-validation

Reproduces the original EMS benchmark protocol on the 160 labelled subjects:

```
Train_Valid.xlsx:  Set_0 (40)  Set_1 (40)  Set_2 (40)  Set_3 (40)
                    └───────────┬───────────┘
for each fold i ∈ {0..3}:
    train ← other 3 folds (120 subjects)
    val   ← fold i         (40 subjects)
report: mean ± std of Acc/Sen/Spe/Pre/F1/AUC over the 4 folds
```

- Fold composition (official): Set_0 18HC/22SZ, Set_1 24HC/16SZ,
  Set_2 20HC/20SZ, Set_3 18HC/22SZ.
- FNN early stopping uses a stratified 90/30 inner split of the 120 training
  subjects (seed 42).
- Additionally, each method trained on all 160 subjects produces predictions
  for the 48 official test subjects (`official_test_preds.csv`, ids `Test_000..047`)
  — official test **labels are withheld** by the EMS authors, so no test metrics
  can be computed locally.

## Protocol P2 — 120 / 40 subject split

Since the official test labels are not released, this protocol builds a fully
local, subject-disjoint train/test split from the 160 labelled subjects:

```
160 subjects (80 HC / 80 SZ)
   │  stratified split (seed s ∈ {42, 2024, 2026})
   ├── 120 train/val subjects (60 HC / 60 SZ)
   │      └── inner stratified split 90 train / 30 val  (FNN early stopping)
   └── 40 test subjects      (20 HC / 20 SZ)  ← final metrics here
report: mean ± std over the 3 seeds
```

- No test subject ever appears in training/validation (subject-disjoint).
- Stratification preserves the 1:1 HC:SZ ratio in every partition.

## Metric definitions (EMS paper, Section V-A)

Positive class = SZ (label 1):

- Acc = (TP+TN)/(TP+TN+FP+FN)
- Sen = TP/(TP+FN) — sensitivity
- Spe = TN/(TN+FP) — specificity
- Pre = TP/(TP+FP) — precision
- F1 = 2·Pre·Sen/(Sen+Pre)
- AUC = ROC area under the curve

At threshold 0.5, `pred = (score ≥ 0.5)`.
