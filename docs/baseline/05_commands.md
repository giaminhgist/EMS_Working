# 05 — Command Reference

Complete list of commands to reproduce the EMS-Baseline experiments
(from repo root `/root/EMS-Project`).

```
EMS-Project/
├── src/
│   ├── eda.py                  # EDA: figures + stats -> docs/EDA/
│   ├── preprocess.py           # clean + 45 hand-crafted features -> processed_dataset/
│   └── baseline/               # <-- BASELINE CODE (all experiments)
│       ├── run_experiment.py   # one method x protocol x seed
│       ├── run_all.py          # full suite -> docs/baseline/results/summary.csv
│       ├── plot_results.py     # AUC bars + ROC figures
│       └── ... (features_builder, protocols, models, metrics_utils, common)
├── processed_dataset/          # stimulus_features_{train,test}.pkl, metadata.csv
└── docs/
    ├── EDA/                    # EDA markdown + figures
    └── baseline/               # <-- THIS documentation (01..05) + results/
        └── results/            # <-- experiment outputs + summary.csv + figures/
```

## 0. Data preparation

```bash
# EDA: figures + statistics -> docs/EDA/ (figures/, *.md, eda_stats.pkl)
python src/eda.py

# Preprocessing: clean raw fixations + compute the 45 hand-crafted features
# -> processed_dataset/stimulus_features_{train,test}.pkl,
#    metadata.csv, feature_names.txt, quality_report.txt
python src/preprocess.py
```

## 1. Accessing hand-crafted features

```python
import pandas as pd

feat = pd.read_pickle("processed_dataset/stimulus_features_train.pkl")
# MultiIndex (subject_id, image) — ids are the original NON-contiguous ids
vec = feat.loc[(216, "mood_37.jpg")]        # 45-dim vector for one stimulus
sub = feat.loc[216]                          # (100, 45) matrix for one subject
missing_pairs = feat[feat.isna().any(axis=1)]  # NaN rows = pairs without data

meta = pd.read_csv("processed_dataset/metadata.csv", index_col="subject_id")
# labels (0=HC, 1=SZ) and official 4-fold membership for the 160 train subjects;
# official test subjects have synthetic ids 400..447 with original file ids
# in `file_id` (their labels are withheld by the EMS authors)
```

**Note on the folds:** `stimulus_features_train.pkl` already contains **all 160
train subjects = all 4 official folds** (16,000 rows = 160 × 100 stimuli).
The fold assignment is *not* stored in the pkl itself but in
`metadata.csv → official_fold` (Set_0..Set_3, 40 subjects each). The baseline
code selects the train/val subjects of each fold from this single pkl
(`src/baseline/protocols.py` → `protocol1_folds()`).

## 2. Running one experiment

```bash
cd src/baseline

# Protocol P1 — official EMS 4-fold CV on the 160 labelled subjects
python run_experiment.py --method svm_rbf --protocol P1
python run_experiment.py --method lr      --protocol P1
python run_experiment.py --method fnn     --protocol P1

# Protocol P2 — stratified 120 train/val + 40 test subjects (one seed)
python run_experiment.py --method svm_rbf --protocol P2 --seed 42
```

Available `--method` values (see [01_methods_ml.md](01_methods_ml.md),
[02_fnn.md](02_fnn.md)):

```
svm_rbf  svm_linear  rf  qda  gnb  lr  lr_l1  knn  fnn  fnn_cat
```

## 3. Running the full suite

```bash
cd src/baseline

# all 10 methods x P1 (1 seed) + all 10 methods x P2 (seeds 42/2024/2026)
python run_all.py

# subsets
python run_all.py --methods svm_rbf,lr,rf --protocols P2
python run_all.py --protocols P1
python run_all.py --only-collect        # rebuild summary.csv from existing runs
```

## 4. Figures

```bash
cd src/baseline
python plot_results.py     # -> docs/baseline/results/figures/{auc_comparison,roc}_{P1,P2}.png
```

## Outputs

```
docs/baseline/results/
├── summary.csv                        # all runs: method, protocol, seed, 6 metrics
├── P1/{method}__{rep}/seed42/
│   ├── metrics.json                   # fold metrics + mean±std
│   ├── val_preds.csv                  # per-subject validation predictions (4 folds pooled)
│   └── official_test_preds.csv        # predictions for the 48 withheld test subjects
├── P2/{method}__{rep}/seed{seed}/
│   ├── metrics.json                   # metrics on the 40-subject test set
│   └── test_preds.csv                 # per-subject test predictions + labels
└── figures/                           # AUC bars + ROC curves
```

Docs: [README.md](README.md) · [01_methods_ml.md](01_methods_ml.md) ·
[02_fnn.md](02_fnn.md) · [03_protocols.md](03_protocols.md) ·
[04_results.md](04_results.md)
