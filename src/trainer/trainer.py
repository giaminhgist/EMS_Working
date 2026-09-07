"""Generic supervised trainer for the normative-gaze proposal.

Responsibilities:
  - train/val loops with per-epoch metrics appended to metrics.jsonl
  - early stopping on validation AUC (patience), best checkpoint to best.pt
  - deterministic seeding; optional per-epoch model hook
    (model.on_epoch_start()) used e.g. by the proposal to refresh the latent
    HC bank
  - final val predictions + embeddings to predictions.csv / embeddings.npz
  - returns the run summary (best epoch metrics + all epochs)

Dataset contract (implemented per model):
  ds.subjects      -> list of subject ids
  ds.label_of(sid) -> 0/1
  ds.__getitem__(idx) -> (x, y, sid) with x model-specific
  ds.device_x(x)   -> moves x to device (optional; default = recursive .to)
Model contract:
  model(x) -> (prob (B,1), aux dict) where aux may contain "emb" (B,D)
  model.on_epoch_start()            # optional hook
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trainer.metrics import compute_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


@torch.no_grad()
def _evaluate(model, ds, device):
    model.eval()
    ys, probs, embs, sids = [], [], [], []
    for idx in range(len(ds)):
        x, y, sid = ds[idx]
        xb = ds.collate([x])  # add batch dim for uniform shapes
        prob, aux = model(_to_device(xb, device))
        ys.append(y)
        probs.append(float(prob.item()))
        sids.append(sid)
        if aux.get("emb") is not None:
            embs.append(aux["emb"].cpu().numpy().ravel())
    m = compute_metrics(np.array(ys), np.array(probs))
    return m, np.array(ys), np.array(probs), sids, (np.stack(embs) if embs else None)


def train_model(model, train_ds, val_ds, cfg, device):
    """Full training loop. Returns dict with run summary."""
    seed_all(cfg.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCELoss()
    run_dir = cfg.run_dir(create=True)
    cfg.save(run_dir)
    log_path = run_dir / "metrics.jsonl"
    log_path.touch()

    best_auc, best_state, best_epoch, wait = -1.0, None, -1, 0
    history = []
    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        if hasattr(model, "refresh_bank"):      # rebuild the HC latent bank
            model.refresh_bank(train_ds)
        order = np.random.permutation(len(train_ds))
        tr_loss, tr_n = 0.0, 0
        tr_ys, tr_probs = [], []
        for i in range(0, len(order), cfg.batch_size):
            batch = order[i:i + cfg.batch_size]
            xs, ys, sids = zip(*(train_ds[j] for j in batch))
            xs = _to_device(_collate(xs, train_ds), device)
            yb = torch.tensor(ys, dtype=torch.float32, device=device).view(-1, 1)
            opt.zero_grad()
            prob, aux = model(xs, y=yb)
            loss = loss_fn(prob, yb)
            if aux.get("norm_loss") is not None:  # HC latent-concentration regularization
                loss = loss + aux["norm_loss"]
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(batch)
            tr_n += len(batch)
            tr_ys += list(ys)
            tr_probs += [p.item() for p in prob.view(-1)]
        tr_metrics = compute_metrics(np.array(tr_ys), np.array(tr_probs))
        tr_metrics["loss"] = tr_loss / max(tr_n, 1)

        # ---- val ----
        val_metrics, val_ys, val_probs, val_sids, val_embs = _evaluate(model, val_ds, device)
        row = {"epoch": epoch, "train": tr_metrics, "val": val_metrics}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        history.append(row)

        auc = val_metrics["auc"]
        if auc > best_auc:
            best_auc, best_epoch, wait = auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    # ---- restore best + final val evaluation ----
    model.load_state_dict(best_state)
    val_metrics, val_ys, val_probs, val_sids, val_embs = _evaluate(model, val_ds, device)
    torch.save({"model_state": best_state, "epoch": best_epoch,
                "val_auc": best_auc, "config": cfg.to_dict()}, run_dir / "best.pt")

    import pandas as pd
    pd.DataFrame({"subject_id": val_sids, "label": val_ys, "prob": val_probs}) \
        .to_csv(run_dir / "predictions.csv", index=False)
    if val_embs is not None:
        np.savez(run_dir / "embeddings.npz", subject_id=np.array(val_sids),
                 label=val_ys, emb=val_embs)
    summary = {"run_dir": str(run_dir), "best_epoch": best_epoch,
               "best_val_auc": best_auc, "val_metrics": val_metrics,
               "epochs_trained": epoch + 1, "history": history}
    (run_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return summary


def _collate(batch, ds):
    """Default collate: stack first elements; override via ds.collate()."""
    if hasattr(ds, "collate"):
        return ds.collate(batch)
    xs = [b[0] for b in batch]
    if isinstance(xs[0], torch.Tensor):
        return torch.stack(xs, dim=0)
    return xs
