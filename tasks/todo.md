# Todo — Reproducibility re-run (baseline + EXP-PROP-001..006)

## T01 — Environment setup + data integrity check
State: done
Evidence: `uv sync` OK (torch 2.11.0+cu128, CPU mode); processed_dataset
readable — 16000 rows (160×100×45), 80 HC / 80 SZ train-val + 48 test NaN
labels, official_fold present. 8-CPU cgroup slice on shared host (load ~105).

## T02 — Backup reference outputs before overwrite
State: done
Evidence: backup at `outputs/repro_check_20260922/backup_original/`
(proposal summaries ×51, matrix_logs, matrix_status.csv, baseline_results);
`diff -r` / cmp verified byte-identical.

## T03 — Re-run classical baselines
State: done
Changes: 2-line path fix in `src/common.py` + `src/baseline/common.py`
(hardcoded `/root/EMS-Project` → `__file__`-derived; required to run at all).
Evidence: run_all.py completed, summary.csv rewritten.
Check result: 8/10 methods bit-exact (Δ=0.0 incl. svm_rbf 0.879296, lr
0.871066, rf 0.863095 — matches tracker). fnn/fnn_cat shift (ΔAUC ≤ 0.0925,
P2) — GPU→CPU RNG divergence; CPU-side determinism verified by identical
double re-run of fnn P2 seed42.

## T04 — Re-run EXP-PROP-001..006 (5 seeds × 4 folds)
State: done
Evidence: 30/30 jobs done, 0 failed (matrix_status.csv rows started
2026-09-22). Run config: OMP_NUM_THREADS=1, 8 workers (thread oversubscription
on shared host made default settings 13–40× slower). Partial dirs from the
aborted first attempt cleaned.
Check result: 5-seed aggregate ΔAUC −0.0102…+0.0074 (see report); ranking
preserved (mlp_norm01 best 0.9453, mlp_mean 2nd).

## T05 — Comparison + reproducibility report
State: done
Changes: `docs/reproducibility/repro_20260922.md` (new);
`experiment_tracker.md` — added short "Reproducibility (re-run 2026-09-22)"
pointer section (no recorded number edited); artifacts in
`outputs/repro_check_20260922/` (backups, compare.py, delta CSVs).
Check: report written with per-experiment delta tables, verdicts, and
environment notes; remaining hardcoded paths (`src/preprocess.py:122`,
`src/baseline/plot_results.py:70`) flagged as unrelated follow-up.
