# Research Design — Learned Normative Gaze Modeling

Novelty-check summary and **locked implementation version** for the proposal
in `Original_Proposal.md`. All deviations are computed against
**HC-only normative statistics fitted on the training fold** (zero leakage).

## Novelty check summary

| Method | Verdict | Key evidence |
|---|---|---|
| Learned Stimulus-Conditioned Normative Modeling | **Potentially novel** (in domain) | Deep normative modeling exists in neuroimaging (NormVAE, [Kumar & Sotiras 2021](https://arxiv.org/abs/2102.05420)-style latent deviations), and hard normative z-score / Mahalanobis pipelines exist (PCNtoolkit, [Rutherford et al., Nature Protocols 2022](https://experiments.springernature.com/articles/10.1038/s41596-022-00696-5); Marquand et al. 2019) — but **no learned latent normative bank + learned comparator for eye-tracking SZ** was found. Building blocks are known → methodology incremental, domain application new. |

Shared gap: the EMS benchmark (Song et al. TNNLS 2024/25) compares raw-feature
classifiers and deep scanpath models; no method treats disease signal as
*stimulus-conditioned deviation from healthy norms*.

## Data flow (verified)

```
original_dataset/EMS/{Train_Valid,Test}/Fixations/*.xlsx
        │  IMAGE, FIX_INDEX, FIX_DURATION, FIX_X, FIX_Y, FIX_PUPIL
        ▼  src/preprocess.py (clean_fixations: off-screen, dur∉(40,2000], pupil ±4SD)
processed_dataset/stimulus_features_{train,test}.pkl   (16000/4800 × 45, MultiIndex (subject_id, image))
        │  metadata.csv: label (0=HC,1=SZ), official_fold (Set_0..3), file_id (test)
        ▼
45-dim x_{i,s}
```

Protocols: **4-fold protocol** = official 4-fold CV (Set_0..3, 40
subjects/fold); **held-out protocol** = stratified 120/40 subject split
(seeds 42/2024/2026). Metrics: Acc, ROC-AUC, Balanced Acc, Sens, Spec, F1 at
threshold 0.5.

## Locked implementation (`src/proposal/`)

### Main proposal — learned latent normative deviation

- Input normalization: per-feature z from **train-fold subjects** (unsupervised).
- Encoder f_θ: 45→128→128 (LN+GELU), shared across stimuli.
- Latent HC bank per stimulus: μ_s^z, σ_s^z recomputed **every epoch** on
  train-fold HC encodings (cheap; keeps the bank aligned with the moving encoder).
- Comparator `comparator`: `mlp` (default: g_φ([z, μ^z, z−μ^z, z⊙μ^z]) 512→256→128→64)
  | `sub` (z−μ^z) | `zsub` ((z−μ^z)/(σ^z+ε)).
- Subject aggregation `pool`: `attention` (default) | `mean` | `deepset` (mean∥max).
- Optional `lambda_norm` L_norm regularization (default 0, ablation 0.1).

### Hard-deviation ablations (`--deviation z | diff | mahal`)

Fixed deviations in raw feature space, computed by the dataset from
**train-fold HC** statistics (μ_s, σ_s per stimulus-feature; ε=1e-6):

- `diff` — x − μ_s (unstandardized)
- `z` — (x − μ_s)/(σ_s + ε) (stimulus-conditioned normative z-score)
- `mahal` — z-deviation concatenated with the per-stimulus Mahalanobis scalar
  (shrinkage diagonal covariance, λ=0.1 toward median σ)

These tensors go through the **same pooling + MLP head** as the learned path
(the encoder/comparator/bank are simply absent) — isolating exactly the
"learned vs fixed deviation" contrast. D_out = 45 (z/diff) or 46 (mahal).

### Shared infrastructure

- `src/data/`: fold utils (reuse official Train_Valid.xlsx + metadata.csv),
  tabular builders, leakage-safe normative stats.
- `src/trainer/`: generic supervised trainer — per-epoch train/val metrics →
  `outputs/proposal/{ablation}__seed{s}__fold{f}__{ts}/metrics.jsonl`,
  `best.pt` (val-AUC early stopping, patience 30, max 200 epochs),
  `config.json`, `predictions.csv`. Deterministic naming for querying.
- `src/test_model/`: loads checkpoint+config, evaluates held-out split
  (120/40 protocol; official test → probabilities only), saves test report.
- `src/evaluation/`: (1) embedding separability probing (linear probe AUC +
  silhouette), (2) calibration (ECE, Brier), (3) stimulus-category
  discriminability (per-stimulus logistic probe by category), (4)
  cross-stimulus subset robustness (K=25/50/100: hard-z ablation vs main
  proposal). Each documented with its hypothesis.

## Key scientific decisions (with rationale)

1. **Normative stats fit only on train-fold HC** — standard NM protocol
   (Rutherford et al. 2022); prevents any test-subject information leaking
   into deviations.
2. **Keep the stimulus axis until subject pooling** (per the proposal's core
   principle) — the model reasons stimulus-wise then pools.
3. **Bank refreshed every epoch** — the encoder drifts during training;
   a static bank would compare against stale norms. Per-epoch refresh is
   cheap (one forward pass over train-fold HC subjects).
4. **Hard deviations share the learned path's pooling + head** — ablation
   hygiene: every difference in scores traces to the deviation mechanism,
   not to a different classifier.
5. **Threshold 0.5 for all classification metrics** — matches the EMS-Baseline
   convention, making results directly comparable with the hand-crafted
   baselines (`docs/baseline/results/`).
