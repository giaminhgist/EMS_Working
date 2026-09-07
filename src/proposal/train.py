"""Train the main proposal (Learned Stimulus-Conditioned Normative Modeling).

Usage (from EMS-Project/src):
    python proposal/train.py --ablation mlp_attn --fold all
    python proposal/train.py --ablation sub_mean --comparator sub --pool mean --fold all
    python proposal/train.py --ablation mlp_norm01 --lambda_norm 0.1 --fold all
    # hard-deviation ablations (fixed deviation in feature space)
    python proposal/train.py --ablation z_mean --deviation z --pool mean --fold all
    python proposal/train.py --ablation mahal_mean --deviation mahal --pool mean --fold all

Per fold, a run directory is created:
    outputs/proposal/{ablation}__seed{seed}__fold{fold}__{ts}/
A fold summary (mean ± std over folds) is saved next to it as
{ablation}__seed{seed}__allfolds_summary.json.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Project/src
from data.common import official_folds  # noqa: E402
from trainer.config import parse_args  # noqa: E402
from trainer.trainer import train_model  # noqa: E402
from trainer.metrics import summarize  # noqa: E402
from proposal.model import NormativeModel, NormativeDataset  # noqa: E402


def main():
    cfg, device_arg = parse_args(default_proposal="proposal")
    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    deviation = cfg.extra.get("deviation", "learned")
    comparator = cfg.extra.get("comparator", "mlp")
    pool = cfg.extra.get("pool", "attention")
    lambda_norm = cfg.extra.get("lambda_norm", 0.0)

    # optional stimulus subset (cross-stimulus generalization experiments)
    stim_subset = None
    if cfg.extra.get("n_stim"):
        import numpy as np
        rng = np.random.default_rng(int(cfg.extra.get("stim_seed", 42)))
        stim_subset = sorted(rng.choice(100, size=int(cfg.extra["n_stim"]),
                                       replace=False).tolist())
    n_stim = len(stim_subset) if stim_subset is not None else 100

    folds = list(official_folds()) if cfg.fold == "all" else [
        (cfg.fold, *next(f for f in official_folds() if f[0] == cfg.fold)[1:])]
    fold_metrics, run_dirs = [], []
    for fold_name, train_ids, val_ids in folds:
        cfg.fold = fold_name  # per-fold run dirs get the real fold name
        train_ds = NormativeDataset(train_ids, train_ids, deviation, stim_subset)
        val_ds = NormativeDataset(val_ids, train_ids, deviation, stim_subset)
        model = NormativeModel(comparator=comparator, pool=pool,
                               lambda_norm=lambda_norm, dropout=cfg.dropout,
                               deviation=deviation, n_stim=n_stim)
        summary = train_model(model, train_ds, val_ds, cfg, device)
        fold_metrics.append({**summary["val_metrics"], "fold": fold_name,
                             "best_val_auc": summary["best_val_auc"]})
        run_dirs.append(summary["run_dir"])
        print(f"{fold_name}: AUC={summary['val_metrics']['auc']:.4f} "
              f"Acc={summary['val_metrics']['acc']:.4f} "
              f"BalAcc={summary['val_metrics']['balanced_acc']:.4f} "
              f"F1={summary['val_metrics']['f1']:.4f} "
              f"({summary['epochs_trained']} epochs)")

    agg = summarize(fold_metrics)
    out = {"proposal": cfg.proposal, "ablation": cfg.ablation, "seed": cfg.seed,
           "folds": [m["fold"] for m in fold_metrics],
           "fold_metrics": fold_metrics, "mean_metrics": agg}
    if len(folds) > 1:
        base = Path(run_dirs[0]).parent
        (base / f"{cfg.ablation}__seed{cfg.seed}__allfolds_summary.json") \
            .write_text(json.dumps(out, indent=2))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
