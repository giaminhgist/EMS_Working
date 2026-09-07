# Experiment Tracker — Learned Normative Gaze Modeling (EMS)

Tracks every experiment of the main proposal in `Original_Proposal.md`.
Run status is maintained in `experiment/matrix_status.csv`; per-run metrics in
`outputs/proposal/…/metrics.jsonl`, fold summaries in
`outputs/proposal/{ablation}__seed{seed}__allfolds_summary.json`; individual
reports in [`experiment/`](experiment/).

Legend: ⬜ planned · ▶ running · ✅ done · ❌ failed · 🔬 analysis written

## Phase B — Main proposal + ablations, official 4-fold, 5 seeds (42/1234/2024/2026/7)

| ID | Hypothesis | Config | Status | Val AUC (mean±std over seeds) | Conclusion |
|---|---|---|---|---|---|
| EXP-PROP-001 | learned comparator beats all ablations | deviation=learned, comparator=mlp, pool=attention | ⬜ | — | — |
| EXP-PROP-002 | λ_norm=0.1 stabilizes training | +lambda_norm=0.1 | ⬜ | — | — |
| EXP-PROP-003 | ablation: plain latent subtraction | comparator=sub | ⬜ | — | — |
| EXP-PROP-004 | ablation: standardized latent subtraction | comparator=zsub | ⬜ | — | — |
| EXP-PROP-005 | ablation: mean pooling | pool=mean | ⬜ | — | — |
| EXP-PROP-006 | ablation: DeepSets pooling | pool=deepset | ⬜ | — | — |
| EXP-PROP-007 | hard z-deviation + mean pool (fixed, no learning) | deviation=z, pool=mean | ⬜ | — | — |
| EXP-PROP-008 | unstandardized hard diff weaker than z | deviation=diff, pool=mean | ⬜ | — | — |
| EXP-PROP-009 | Mahalanobis scalar adds value over z alone | deviation=mahal, pool=mean | ⬜ | — | — |

## Phase E — Held-out test (120/40, seeds 42/2024/2026) for the main configs

| ID | Model | Status | Test AUC | Conclusion |
|---|---|---|---|---|
| EXP-TEST-001 | proposal mlp_attn | ⬜ | — | — |
| EXP-TEST-002 | proposal mlp_norm01 | ⬜ | — | — |
| EXP-TEST-003 | proposal mlp_mean | ⬜ | — | — |
| EXP-TEST-004 | proposal z_mean (hard ablation) | ⬜ | — | — |
| EXP-TEST-005 | official test predictions (best config) | ⬜ | — | — |

## Phase F — Representation / generalization evaluations

| ID | Test | Hypothesis | Status | Result |
|---|---|---|---|---|
| EXP-EVAL-001 | Embedding separability probing | learned latent deviation embeddings separate HC/SZ better (probe AUC + silhouette) | ⬜ | — |
| EXP-EVAL-002 | Calibration (ECE, Brier) | deviation models better calibrated | ⬜ | — |
| EXP-EVAL-003 | Stimulus-category discriminability | per-stimulus AUC by category; deviation signal at subject aggregation | ⬜ | — |
| EXP-EVAL-004 | Cross-stimulus generalization (K=25/50/100) | hard z-deviation degrades less than the learned proposal when stimuli shrink | ⬜ | — |

## Reference numbers (existing hand-crafted baselines, `docs/baseline/results/`)

- Official 4-fold val: SVM-RBF AUC 0.8793 · LogReg 0.8711 · RF 0.8631 · FNN-agg 0.7700
- Held-out test (40 subjects): LogReg-L2 AUC 0.9075 · SVM-RBF 0.9058
- Paper (Song et al. TNNLS): MSNet val AUC 0.8972 / test 0.8854; best traditional ESR_SVM 0.8498

## Priority order (executed as feasible)

1. EXP-PROP-001/002/007 (core comparisons) — first wave
2. Remaining ablations — second wave
3. Phase E (held-out + official tests) — after Phase B
4. Phase F (evaluations) — after Phase E

## Current best models

_To be filled after Phase B–F complete._

## Next highest-value experiments

1. Interpretability export for the proposal (stimulus attention, feature-level
   deviation maps) — hooks already exist in the model.
2. Ensembles of learned + hard deviation embeddings.
3. Wider hyperparameter sweep on the comparator (gated matching block) and
   deeper encoders.
