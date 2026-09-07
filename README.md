# EMS-Project — Learned Normative Gaze Modeling for Schizophrenia Recognition

Research codebase implementing the methodology of
[`Original_Proposal.md`](Original_Proposal.md) as a reproducible
experiment pipeline on the public **EMS** eye-tracking dataset
(Song et al., *IEEE TNNLS* 2024 — [paper](https://ieeexplore.ieee.org/document/10645682)).

**Core hypothesis:**
> Gaze abnormalities in schizophrenia are better modeled as **stimulus-conditioned
> deviations from healthy (HC) normative behavior** than as raw hand-crafted
> features fed to a classifier.

The codebase implements **one main proposal** — *Learned Stimulus-Conditioned
Normative Modeling* — plus a family of config-toggled ablations, including
fixed (hard) normative-deviation variants in raw feature space.

Literature/novelty analysis: [`docs/research/design.md`](docs/research/design.md).
Experiment tracking: [`experiment_tracker.md`](experiment_tracker.md).

---

## 1. Dataset, EDA, processed data

- **Raw data**: `original_dataset/EMS/` — 208 subjects (104 SZ / 104 HC),
  free viewing of 100 stimuli × 5 s, Eyelink 1000 Plus @1 kHz, 1,024×768 screen.
  Train/Valid: 160 labelled subjects (id < 200 = HC). Official test: 48 subjects,
  **labels withheld** by the authors.
- **EDA**: full report in [`docs/EDA/README.md`](docs/EDA/README.md)
  (headline: SZ make fewer/longer fixations, narrower spatial spread, smaller
  pupil; 1.9 % off-screen fixations removed).
- **Processed data** (`processed_dataset/`):
  - `stimulus_features_{train,test}.pkl` — 45 hand-crafted features per
    (subject_id, image), MultiIndex preserving **non-contiguous ids**
  - `metadata.csv` — labels, official 4-fold membership (Set_0..3),
    test file ids (synthetic subject ids 400–447; labels NaN)
  - `quality_report.txt` — NaN audit (2.04 % cells, from missing stimulus pairs)
- **Reproduction**: `python src/eda.py` → `python src/preprocess.py`

> **Data access & licensing**: the EMS dataset is *not* included in this
> repository (usage agreement: non-commercial research only). Download it from
> the [official repo](https://github.com/YingjieSong1/EMS) into
> `original_dataset/EMS/`, then regenerate `processed_dataset/` with the two
> commands above. `outputs/`, `original_dataset/` and `processed_dataset/` are
> git-ignored; aggregate results are summarized in
> [`experiment_tracker.md`](experiment_tracker.md) and `experiment/`.

## 2. Method (locked version)

See [`docs/research/design.md`](docs/research/design.md) for the full
specification and novelty evidence. Short form:

- **Encoder** f_θ: 45 → 128 → 128 (LN+GELU), shared across stimuli.
- **HC latent normative bank**: per-stimulus μ_s^z, σ_s^z recomputed
  **every epoch** from train-fold HC encodings.
- **Comparator** g_φ: `mlp` ([z, μ^z, z−μ^z, z⊙μ^z] → 64) | `sub` (z−μ^z) |
  `zsub` ((z−μ^z)/(σ^z+ε)).
- **Stimulus aggregation**: `attention` | `mean` | `deepset` (mean‖max).
- **Optional regularization**: λ_norm HC latent-concentration loss
  (per-batch, graph-attached).
- **Hard-deviation ablations** (`--deviation z | diff | mahal`): per-stimulus
  deviations from HC normative statistics computed directly in raw feature
  space (no encoder/comparator/bank), fed through the **same pooling + head**.

**Leakage guarantees:** every normative statistic (per-feature input scaling,
HC stimulus norms, latent bank) is fitted on the **training fold's HC subjects
only** (input scaling unsupervised on the training fold) and applied to
held-out folds/test subjects.

## 3. Ablations (config-toggled, no separate codebase)

| Flag | Values | Ablated component |
|---|---|---|
| `--deviation` | learned / z / diff / mahal | learned latent deviation vs fixed hard deviation |
| `--comparator` | mlp / sub / zsub | learned comparator |
| `--pool` | attention / mean / deepset | stimulus set aggregation |
| `--lambda_norm` | 0 / 0.1 | HC concentration regularization |
| `--n_stim --stim_seed` | 25/50/100 | stimulus subset (cross-stimulus eval) |

## 4. Project structure

```
EMS-Project/
├── original_dataset/            # raw EMS (xlsx fixations + 100 stimuli images)
├── processed_dataset/           # 45-dim features, metadata, quality report
├── docs/
│   ├── EDA/README.md            # full EDA (sections 1–7) + figures
│   ├── baseline/                # hand-crafted baselines (docs + results/)
│   ├── research/design.md       # novelty check + locked methodology
│   ├── model_spec.md            # architecture, tensor shapes, loss, ablations, evals
│   └── analysis.md              # component ablation analysis + conclusions
├── src/
│   ├── data/                    # shared builders: tabular deviations, folds,
│   │                            #   leakage-safe normative stats
│   ├── trainer/                 # generic trainer: per-epoch metrics.jsonl,
│   │                            #   early stopping, best.pt, config.json
│   ├── proposal/                # main proposal model + train entry
│   ├── test_model/eval.py       # held-out test (120/40 ×3 seeds) + official-test preds
│   ├── evaluation/              # probing / calibration / category / cross-stimulus
│   ├── baseline/                # classical ML + FNN baselines (threshold 0.5)
│   ├── eda.py, preprocess.py, features.py
│   └── run_matrix.py            # prioritized experiment matrix runner
├── outputs/                     # every run: metrics, checkpoints, predictions
├── experiment/                  # matrix.json + per-experiment reports + status
└── experiment_tracker.md        # hypothesis/status/metrics table
```

## 5. Commands

```bash
# data prep
python src/eda.py
python src/preprocess.py

# train (all folds of the official 4-fold protocol)
cd src
python proposal/train.py --ablation mlp_attn --comparator mlp --pool attention --fold all
python proposal/train.py --ablation mlp_norm01 --lambda_norm 0.1 --fold all
python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all

# single fold
python proposal/train.py --ablation z_mean --deviation z --fold Set_1

# held-out test (120 train/val + 40 test, 3 seeds, real labels)
python test_model/eval.py --ablation mlp_attn --mode heldout

# official test (labels withheld -> probabilities in benchmark format)
python test_model/eval.py --ablation mlp_norm01 --lambda_norm 0.1 --mode official

# representation / generalization evaluations
python evaluation/probing.py --runs proposal/mlp_attn proposal/z_mean proposal/mahal_mean
python evaluation/calibration.py --runs proposal/z_mean proposal/diff_mean proposal/mlp_attn
python evaluation/category_probe.py
python evaluation/cross_stimulus.py

# full prioritized experiment matrix (from experiment/matrix.json)
python run_matrix.py --workers 4
```

## 6. Config examples

Every run saves its full config to `config.json`. CLI flags beyond the common
ones are captured automatically:

```bash
# common flags: --seed --fold --epochs --patience --lr --weight_decay --batch_size --dropout
python proposal/train.py \
    --ablation mlp_norm01        \   # run tag (part of the run-dir name)
    --comparator mlp             \   # learned comparator [z, mu, z-mu, z*mu]
    --pool attention             \   # stimulus set pooling
    --lambda_norm 0.1            \   # HC concentration regularization weight
    --fold all --seed 42 --epochs 150 --patience 30 --lr 1e-3
```

Example `config.json`:

```json
{
  "proposal": "proposal", "ablation": "mlp_norm01",
  "seed": 42, "fold": "Set_0", "epochs": 150, "patience": 30,
  "lr": 0.001, "weight_decay": 0.0001, "batch_size": 16, "dropout": 0.3,
  "extra": {"comparator": "mlp", "pool": "attention", "lambda_norm": 0.1}
}
```

## 7. Output conventions

```
outputs/proposal/{ablation}__seed{seed}__fold{fold}__{YYYYMMDD-HHMMSS}/
├── config.json        # full configuration (queryable, deterministic)
├── metrics.jsonl      # one JSON line per epoch: {epoch, train{6 metrics+loss}, val{...}}
├── best.pt            # checkpoint of the best validation-AUC epoch (+config)
├── predictions.csv    # val subjects: subject_id, label, prob (threshold 0.5)
├── embeddings.npz     # subject embeddings (for probing evals)
└── summary.json       # best epoch, val metrics, epochs trained

outputs/proposal/{ablation}__seed{seed}__allfolds_summary.json   # 4-fold mean±std
outputs/proposal/test_heldout/{ablation}/result.json             # 120/40 test (3 seeds)
outputs/proposal/test_official/{ablation}/official_test_preds.csv
```

Metrics (fixed threshold **0.5**): Accuracy, ROC-AUC, Balanced Accuracy,
Sensitivity, Specificity, F1 — identical convention to the hand-crafted
baselines in `docs/baseline/results/`, so numbers are directly comparable.

## 8. Reproducibility

- Seeds: matrix experiments run on seeds **42 / 1234 / 2024 / 2026 / 7**
  (data splits, torch, numpy); held-out-protocol splits use seeds 42/2024/2026
  as in the baseline suite.
- No data leakage: normative stats from train-fold HC only; early stopping on
  validation AUC; official test labels never used.
- Deterministic run-dir naming (proposal, ablation, seed, fold, timestamp) —
  re-running produces a new timestamped dir, results queryable by prefix.
- Environment: `uv sync` (Python ≥3.12, PyTorch cu128 per `pyproject.toml`);
  no PyG/DGL dependency.

## 9. Status & headline results

All matrix experiments (9 configs × 5 seeds), held-out/official tests and the
4 evaluation studies have run. Headline (official 4-fold validation, mean over
5 seeds; held-out test = 120/40 protocol, 3 seeds):

| Model | Val AUC | Val Acc | Held-out test AUC / Acc |
|---|---|---|---|
| Proposal mlp_norm01 (λ=0.1) | **0.9484±0.0055** | 0.7900 | **0.9150** / 0.7750 |
| Proposal mlp_mean | 0.9439±0.0059 | **0.8450** | 0.8808 / **0.8083** |
| Proposal mlp_attn | 0.9376±0.0086 | 0.8100 | 0.8650 / 0.7667 |
| Hard ablation z_mean | 0.9065±0.0054 | 0.8075 | 0.8733 / 0.7833 |
| Hard ablation mahal_mean | 0.9102±0.0040 | 0.8012 | — |
| Hard ablation diff_mean | 0.8635±0.0091 | 0.7662 | — |
| Best hand-crafted baseline (SVM-RBF) | 0.8793 | 0.7938 | — |
| Paper MSNet (deep, saliency features) | 0.8972 | 0.8313 | 0.8854 |

Key evaluation findings: learned latent deviation embeddings most separable
(probe AUC 0.9356); hard deviations and mlp_norm01 best calibrated
(ECE 0.083–0.089 vs 0.152 for mlp_attn); the fixed z-deviation transfers
better across stimulus subsets (AUC drop 0.013 vs 0.019).

See [`experiment_tracker.md`](experiment_tracker.md) for the full matrix,
per-experiment reports in [`experiment/`](experiment/), and next steps.
