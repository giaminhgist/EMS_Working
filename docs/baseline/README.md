# EMS-Baseline Documentation

Hand-crafted-feature baselines for subject-level schizophrenia recognition on the
EMS dataset. All models consume the 45 spatial+temporal hand-crafted features
(see [`../../docs/EDA/README.md`](../../docs/EDA/README.md) — Section 6),
aggregated to subject level in three possible representations.

| Document | Content |
|---|---|
| [01_methods_ml.md](01_methods_ml.md) | Classical ML baselines (SVM, RF, QDA, NB, LR, KNN) |
| [02_fnn.md](02_fnn.md) | Basic feed-forward network (architecture + tensor shapes) |
| [03_protocols.md](03_protocols.md) | Evaluation protocols P1 (official folds) & P2 (120/40 split) |
| [04_results.md](04_results.md) | Experiment results (generated after running) |
| [05_commands.md](05_commands.md) | Complete command reference (data prep → experiment → figures) |

## Quick start

```bash
# 0. (already done) download raw dataset -> original_dataset/EMS
# 1. EDA
python src/eda.py
# 2. preprocess + hand-crafted features
python src/preprocess.py
# 3. run one baseline (code lives in EMS-Minh/src/baseline/)
cd src/baseline
python run_experiment.py --method svm_rbf --protocol P1
python run_experiment.py --method fnn     --protocol P2 --seed 42
# 4. run the full suite (all methods x protocols) and rebuild summary.csv
python run_all.py
# 5. figures (AUC curves, protocol comparison)
python plot_results.py
```

## Repository layout

```
src/baseline/                # (EMS-Minh/src/baseline)
├── common.py            # paths, seeds, threshold=0.5, category mapping
├── features_builder.py  # subject-level matrices: agg(91) / catagg(181) / concat(4500)
├── metrics_utils.py     # Acc/Sen/Spe/Pre/F1 @0.5 + AUC
├── protocols.py         # P1 official 4-fold, P2 stratified 120/40 split
├── models.py            # sklearn pipelines + BasicFNN
├── run_experiment.py    # run one method x protocol x seed
├── run_all.py           # driver over the whole suite -> results/summary.csv
└── plot_results.py      # AUC figures
docs/baseline/results/   # {P1,P2}/{method}__{rep}/seed*/{metrics.json,preds}
```

## Common settings

- **Decision threshold**: fixed at **0.5** for all methods and protocols
  (AUC is threshold-free). EMS label convention: SZ = positive (1), HC = 0.
- **Feature preprocessing**: median imputation + z-score standardization,
  fitted on the *training* subjects only (no leakage).
- **Seeds**: 42 for protocol P1 (fixed official folds); P2 runs the stratified
  split with seeds 42 / 2024 / 2026 and reports mean ± std.
- **Missing data**: (subject, stimulus) pairs absent after cleaning (e.g.
  subjects 216, 259 viewed only 63/68 stimuli) contribute NaN → skipped in
  subject-level aggregation; `n_valid_stim` is kept as an extra feature.
