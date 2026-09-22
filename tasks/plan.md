# Plan — Reproducibility re-run (baseline + EXP-PROP-001..006)

Date: 2026-09-22
Request: re-run the recorded experiments to verify reproducibility from the
current repo state (same code, same data, same seeds).

## Question

Are the recorded Phase-B results in `experiment_tracker.md` —
classical baselines and EXP-PROP-001..006 (5 seeds × official 4-fold) —
reproducible by re-executing the recorded commands in the current environment?

## Scope (stated assumptions)

- **baseline** = `src/baseline/run_all.py` (all 10 methods, P1 seed 42, P2
  seeds 42/2024/2026), i.e. the classical baselines quoted in the tracker.
- **exp 001 to 006** = `EXP-PROP-001..006` from `experiment/matrix.json`
  (learned-config experiments). Hard-deviation ablations 007–009 are
  **excluded** (user said "001 to 006"); cheap to add later if wanted.
- Same data: `processed_dataset/` is assumed unchanged (git-ignored, local).
- Environment: no GPU on this machine (`nvidia-smi` absent, no CUDA).
  `pyproject.toml` pins torch to the cu128 index; `uv sync` installs that
  wheel, which runs on CPU. `train.py` falls back to `cpu` automatically.
  Device of the original 2026-09-07 runs is not recorded in the run configs.
  → If original runs were on GPU, tiny float differences (≲1e-3) are possible
  even with fixed seeds; this is reported, not "failure".

## Success criterion / evidence

For every re-run: recorded reference metric (mean ± std over 4 folds) vs
re-run metric. Expect agreement to float precision (same code + data + seeds);
deviations > ~1e-2 in AUC/Acc/BalAcc/F1 are treated as non-reproducible and
investigated before concluding.

## Risks

- Reference outputs are overwritten by re-running (paths are not
  timestamped): `outputs/proposal/*_allfolds_summary.json`,
  `outputs/.matrix_logs/*.log`, `experiment/matrix_status.csv`,
  `docs/baseline/results/**`. → Mitigation: full backup to
  `outputs/repro_check_20260922/backup_original/` **before** any run.
- torch cu128 wheel is a large download (~2.5 GB); needs network.
- P1 baseline uses only seed 42 by design (single-seed reference); P2 uses 3
  seeds. Comparison respects the same protocol.

## Decisions

- Reference numbers are read from the **backup**, never from the overwritten
  live files.
- Pass/fail is reported per experiment with the actual numeric deltas, not a
  single global verdict.

## Unresolved

- Whether the original runs used GPU or CPU (not recorded; not resolvable).
  Handled as stated above.
