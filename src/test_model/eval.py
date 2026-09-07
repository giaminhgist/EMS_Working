"""Test-set evaluation for the trained normative model.

Two modes (never touches the official-test labels, which are withheld):

  heldout:  retrain the model's configuration on the 120 train/val subjects
            (stratified 120/40 split, seeds 42/2024/2026), evaluate on the
            40 held-out subjects -> test metrics with real labels.
  official: retrain on all 160 labelled subjects, produce probabilities for
            the 48 official test subjects (Test_000..047, labels unknown).

Usage (from EMS-Minh/src):
    python test_model/eval.py --proposal proposal --ablation mlp_attn --mode heldout
    python test_model/eval.py --proposal proposal --ablation z_mean \\
        --deviation z --pool mean --mode heldout
    python test_model/eval.py --proposal proposal --ablation mlp_norm01 \\
        --lambda_norm 0.1 --mode official

The model + dataset are reconstructed from the same config flags used for
training; results are written to outputs/{proposal}/test_{mode}/{ablation}/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # EMS-Minh/src
from data.common import (OUTPUTS, SEEDS_HELDOUT, protocol2_split,  # noqa: E402
                         load_metadata)
from trainer.trainer import train_model, _evaluate  # noqa: E402
from trainer.config import RunConfig, _auto_type  # noqa: E402
from trainer.metrics import summarize  # noqa: E402

PROPOSAL_LOADERS = {"proposal": ("proposal.model", "NormativeModel", "NormativeDataset")}


def make_model(cfg, proposal):
    mod, cls, _ = PROPOSAL_LOADERS[proposal]
    m = __import__(mod, fromlist=[cls])
    Model = getattr(m, cls)
    deviation = cfg.extra.get("deviation", "learned")
    if deviation == "learned":
        return Model(comparator=cfg.extra.get("comparator", "mlp"),
                     pool=cfg.extra.get("pool", "attention"),
                     lambda_norm=cfg.extra.get("lambda_norm", 0.0),
                     dropout=cfg.dropout, deviation="learned")
    return Model(deviation=deviation, pool=cfg.extra.get("pool", "mean"),
                 dropout=cfg.dropout)


def make_dataset(cfg, proposal, subject_ids, train_ids):
    mod, _, dcls = PROPOSAL_LOADERS[proposal]
    m = __import__(mod, fromlist=[dcls])
    Dataset = getattr(m, dcls)
    return Dataset(subject_ids, train_ids, cfg.extra.get("deviation", "learned"))


def run_heldout(cfg, device, out_dir):
    out = {"mode": "heldout", "seeds": {}, "mean_metrics": None}
    per_seed = []
    for seed in SEEDS_HELDOUT:
        cfg.seed = seed
        split = protocol2_split(seed)
        tv, te = split["train_val_ids"], split["test_ids"]
        meta = load_metadata()
        # inner 90/30 stratified split for early stopping
        from sklearn.model_selection import train_test_split
        y_tv = meta.loc[tv].label.values
        inner_tr, inner_va = train_test_split(tv, test_size=30, stratify=y_tv,
                                              random_state=seed)
        train_ds = make_dataset(cfg, cfg.proposal, inner_tr, inner_tr)
        val_ds = make_dataset(cfg, cfg.proposal, inner_va, inner_tr)
        test_ds = make_dataset(cfg, cfg.proposal, te, inner_tr)
        model = make_model(cfg, cfg.proposal)
        summary = train_model(model, train_ds, val_ds, cfg, device)
        te_metrics, te_ys, te_probs, te_sids, _ = _evaluate(model, test_ds, device)
        out["seeds"][str(seed)] = {"inner_val": summary["val_metrics"],
                                   "test": te_metrics}
        per_seed.append(te_metrics)
        import pandas as pd
        pd.DataFrame({"subject_id": te_sids, "label": te_ys, "prob": te_probs}) \
            .to_csv(out_dir / f"preds_seed{seed}.csv", index=False)
        print(f"  seed {seed}: test AUC={te_metrics['auc']:.4f} "
              f"Acc={te_metrics['acc']:.4f} BalAcc={te_metrics['balanced_acc']:.4f} "
              f"F1={te_metrics['f1']:.4f}")
    out["mean_metrics"] = summarize(per_seed)
    return out


def run_official(cfg, device, out_dir):
    from data.common import train_subject_ids
    meta = load_metadata()
    all_train = train_subject_ids()
    test_ids = meta[meta.partition == "test"].index.tolist()
    from sklearn.model_selection import train_test_split
    y_all = meta.loc[all_train].label.values
    inner_tr, inner_va = train_test_split(all_train, test_size=0.25, stratify=y_all,
                                          random_state=cfg.seed)
    train_ds = make_dataset(cfg, cfg.proposal, inner_tr, inner_tr)
    val_ds = make_dataset(cfg, cfg.proposal, inner_va, inner_tr)
    model = make_model(cfg, cfg.proposal)
    train_model(model, train_ds, val_ds, cfg, device)
    test_ds = make_dataset(cfg, cfg.proposal, test_ids, inner_tr)
    _, _, te_probs, te_sids, _ = _evaluate(model, test_ds, device)
    file_ids = [f"Test_{int(meta.loc[s, 'file_id']):03d}" for s in te_sids]
    import pandas as pd
    pd.DataFrame({"subject_id": file_ids, "prob": te_probs}) \
        .to_csv(out_dir / "official_test_preds.csv", index=False)
    print(f"official test predictions saved for {len(file_ids)} subjects")
    return {"mode": "official", "n_test": len(file_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", default="proposal")
    ap.add_argument("--ablation", default="base")
    ap.add_argument("--mode", default="heldout", choices=["heldout", "official"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="")
    args, unknown = ap.parse_known_args()
    extra = {}
    it = iter(unknown)
    for a in it:
        assert a.startswith("--")
        try:
            extra[a[2:]] = _auto_type(next(it))
        except StopIteration:
            extra[a[2:]] = 1
    cfg = RunConfig(proposal=args.proposal, ablation=args.ablation, seed=args.seed,
                    fold="heldout_split", epochs=args.epochs, patience=args.patience,
                    extra=extra)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUTPUTS / cfg.proposal / f"test_{args.mode}" / cfg.ablation
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "heldout":
        result = run_heldout(cfg, device, out_dir)
    else:
        result = run_official(cfg, device, out_dir)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
