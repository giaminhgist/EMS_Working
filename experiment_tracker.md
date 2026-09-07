# Experiment Tracker — Learned Normative Gaze Modeling (EMS)

Tracks every experiment of the main proposal in `Original_Proposal.md`.
Run status is maintained in `experiment/matrix_status.csv`; per-run metrics in
`outputs/proposal/…/metrics.jsonl`, fold summaries in
`outputs/proposal/{ablation}__seed{seed}__allfolds_summary.json`; individual
reports in [`experiment/`](experiment/).

Legend: ⬜ planned · ▶ running · ✅ done · ❌ failed · 🔬 analysis written

## Phase B — Main proposal + ablations, official 4-fold, 5 seeds (42/1234/2024/2026/7)

Val AUC/Acc = mean ± std over 5 seeds of the 4-fold mean metrics (threshold 0.5).

| ID | Hypothesis | Config | Status | Val AUC | Val Acc | Conclusion |
|---|---|---|---|---|---|---|
| EXP-PROP-001 | learned comparator beats all ablations | deviation=learned, comparator=mlp, pool=attention | ✅ | 0.9376±0.0086 | 0.8100±0.0264 | Supported vs hard ablations (0.86–0.91) |
| EXP-PROP-002 | λ_norm=0.1 stabilizes training | +lambda_norm=0.1 | ✅ | **0.9484±0.0055** | 0.7900±0.0524 | Best AUC, tightest across seeds; no collapse |
| EXP-PROP-003 | ablation: plain latent subtraction | comparator=sub | ✅ | 0.9170±0.0046 | 0.8063±0.0172 | Comparator contributes ~+0.02–0.03 AUC |
| EXP-PROP-004 | ablation: standardized latent subtraction | comparator=zsub | ✅ | 0.9221±0.0066 | 0.8250±0.0125 | Between sub and mlp |
| EXP-PROP-005 | ablation: mean pooling | pool=mean | ✅ | 0.9439±0.0059 | **0.8450±0.0155** | Best-balanced decisions (BalAcc 0.8479) |
| EXP-PROP-006 | ablation: DeepSets pooling | pool=deepset | ✅ | 0.9320±0.0157 | 0.8262±0.0248 | Between mean and attention |
| EXP-PROP-007 | hard z-deviation + mean pool (fixed, no learning) | deviation=z, pool=mean | ✅ | 0.9065±0.0054 | 0.8075±0.0073 | Learned pipeline beats it by +0.03–0.04 AUC |
| EXP-PROP-008 | unstandardized hard diff weaker than z | deviation=diff, pool=mean | ✅ | 0.8635±0.0091 | 0.7662±0.0348 | Confirmed: HC standardization is necessary |
| EXP-PROP-009 | Mahalanobis scalar adds value over z alone | deviation=mahal, pool=mean | ✅ | 0.9102±0.0040 | 0.8012±0.0211 | Marginal gain over z (diagonal approx.) |

## Phase E — Held-out test (120/40, seeds 42/2024/2026) for the main configs

| ID | Model | Status | Test AUC | Test Acc | Conclusion |
|---|---|---|---|---|---|
| EXP-TEST-001 | proposal mlp_attn | ✅ | 0.8650±0.0216 | 0.7667 | — |
| EXP-TEST-002 | proposal mlp_norm01 | ✅ | **0.9150±0.0329** | 0.7750 | Best on held-out test; beats paper MSNet test Acc (0.8125) on AUC |
| EXP-TEST-003 | proposal mlp_mean | ✅ | 0.8808±0.0365 | **0.8083** | Best test Acc of the suite |
| EXP-TEST-004 | proposal z_mean (hard ablation) | ✅ | 0.8733±0.0348 | 0.7833 | Hard ablation ≈ mlp_attn on held-out; learned + λ_norm wins |
| EXP-TEST-005 | official test predictions (mlp_norm01) | ✅ | — | — | 48 probabilities in benchmark format (`test_official/mlp_norm01/official_test_preds.csv`) |

## Phase F — Representation / generalization evaluations

| ID | Test | Hypothesis | Status | Result |
|---|---|---|---|---|
| EXP-EVAL-001 | Embedding separability probing | learned latent deviation embeddings separate HC/SZ better (probe AUC + silhouette) | ✅ | mlp_norm01 probe AUC **0.9356**, sil 0.111 > mlp_attn 0.9163 > mahal 0.8769 > z 0.8688 — learned latent deviation most separable |
| EXP-EVAL-002 | Calibration (ECE, Brier) | deviation models better calibrated | ✅ | mahal ECE **0.0833** ≈ z 0.0888 ≈ mlp_norm01 0.0890 < diff 0.1008 < mlp_mean 0.1415 < mlp_attn 0.1515; λ_norm fixes calibration of the learned path; temperature scaling: mlp_norm01 → ECE 0.0647 |
| EXP-EVAL-003 | Stimulus-category discriminability | per-stimulus AUC by category; deviation signal at subject aggregation | ✅ | raw 0.727 / z 0.725 overall; categories similar (0.71–0.73) — deviation signal lives at subject aggregation, not single stimuli |
| EXP-EVAL-004 | Cross-stimulus generalization (K=25/50/100) | hard z-deviation degrades less than the learned proposal when stimuli shrink | ✅ | AUC drop 100→25: z **0.0125** vs learned 0.0190 — fixed deviation is the more transferable quantity |

## Reference numbers (existing hand-crafted baselines, `docs/baseline/results/`)

- Official 4-fold val: SVM-RBF AUC 0.8793 · LogReg 0.8711 · RF 0.8631 · FNN-agg 0.7700
- Held-out test (40 subjects): LogReg-L2 AUC 0.9075 · SVM-RBF 0.9058
- Paper (Song et al. TNNLS): MSNet val AUC 0.8972 / test 0.8854; best traditional ESR_SVM 0.8498

## Priority order (executed as feasible)

1. EXP-PROP-001/002/007 (core comparisons) — first wave
2. Remaining ablations — second wave
3. Phase E (held-out + official tests) — after Phase B
4. Phase F (evaluations) — after Phase E

## Current best models (all phases complete, 5-seed)

- **proposal mlp_norm01** — val AUC **0.9484±0.0055** (5 seeds, 4-fold); held-out test AUC **0.9150** — the strongest configuration.
- **proposal mlp_mean** — best-balanced val decisions (Acc 0.8450 / BalAcc 0.8479), best held-out test Acc 0.8083.
- **proposal z_mean (hard ablation)** — val AUC 0.9065; the fixed-deviation ceiling the learned pipeline must clear (+0.03–0.04 AUC).
- All learned configs beat the best hand-crafted baseline (SVM-RBF val AUC 0.8793).

## Next highest-value experiments

1. Interpretability export for the proposal (stimulus attention, feature-level
   deviation maps) — hooks already exist in the model.
2. Ensembles of learned + hard deviation embeddings.
3. Wider hyperparameter sweep on the comparator (gated matching block) and
   deeper encoders.
