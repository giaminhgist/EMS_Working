"""Train Proposal 2 (Learned Stimulus-Conditioned Normative Modeling).

Usage (from EMS-Minh/src):
    python proposal2_latent_norm/train.py --ablation mlp_attn --fold all
    python proposal2_latent_norm/train.py --ablation sub_mean --comparator sub --pool mean --fold all
    python proposal2_latent_norm/train.py --ablation mlp_norm01 --lambda_norm 0.1 --fold all
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Minh/src
from data.common import official_folds  # noqa: E402
from trainer.config import parse_args  # noqa: E402
from trainer.trainer import train_model  # noqa: E402
from trainer.metrics import summarize  # noqa: E402
from proposal2_latent_norm.model import LatentNormModel, LatentNormDataset  # noqa: E402


def main():
    cfg, device_arg = parse_args(default_proposal="proposal2_latent_norm")
    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    comparator = cfg.extra.get("comparator", "mlp")
    pool = cfg.extra.get("pool", "attention")
    lambda_norm = cfg.extra.get("lambda_norm", 0.0)

    folds = list(official_folds()) if cfg.fold == "all" else [
        (cfg.fold, *next(f for f in official_folds() if f[0] == cfg.fold)[1:])]
    fold_metrics, run_dirs = [], []
    for fold_name, train_ids, val_ids in folds:
        cfg.fold = fold_name  # per-fold run dirs get the real fold name
        train_ds = LatentNormDataset(train_ids, train_ids)
        val_ds = LatentNormDataset(val_ids, train_ids)
        model = LatentNormModel(comparator=comparator, pool=pool,
                                lambda_norm=lambda_norm, dropout=cfg.dropout)
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
           "fold_metrics": fold_metrics, "mean_metrics": agg}
    if len(folds) > 1:
        base = Path(run_dirs[0]).parent
        (base / f"{cfg.ablation}__seed{cfg.seed}__allfolds_summary.json") \
            .write_text(json.dumps(out, indent=2))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
