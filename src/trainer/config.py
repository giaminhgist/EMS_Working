"""Run configuration: dataclass + CLI parsing + deterministic run-directory naming.

Every run writes its full config to config.json inside the run directory:
  outputs/{proposal}/{ablation}__seed{s}__fold{f}__{timestamp}/
making results queryable by proposal/ablation/seed/fold.
"""
import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from data.common import OUTPUTS


@dataclass
class RunConfig:
    # identity
    proposal: str = ""                 # e.g. "proposal1_norm_sub"
    ablation: str = ""                 # short tag describing the variant
    seed: int = 42
    fold: str = "Set_0"                # official fold name (or "P2_split")
    # training
    epochs: int = 200
    patience: int = 30                 # early stopping on val AUC
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    dropout: float = 0.3
    # model-specific options are added by each proposal via kwargs
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        extra = d.pop("extra", {})
        cfg = cls(**d)
        cfg.extra = extra
        return cfg

    def run_dir(self, create=True):
        ts = time.strftime("%Y%m%d-%H%M%S")
        d = OUTPUTS / self.proposal / f"{self.ablation}__seed{self.seed}__fold{self.fold}__{ts}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, run_dir: Path):
        (run_dir / "config.json").write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, run_dir: Path):
        return cls.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))


def parse_args(default_proposal=""):
    """CLI: --proposal --ablation --seed --fold --epochs --lr ... plus --X Y
    pairs captured into cfg.extra (e.g. --deviation z --pool mean)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", default=default_proposal)
    ap.add_argument("--ablation", default="base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", default="Set_0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default="")
    args, unknown = ap.parse_known_args()
    extra = {}
    it = iter(unknown)
    for a in it:
        assert a.startswith("--"), f"unexpected arg {a}"
        key = a[2:]
        try:
            val = next(it)
        except StopIteration:
            val = "1"
        extra[key] = _auto_type(val)
    cfg = RunConfig(proposal=args.proposal, ablation=args.ablation, seed=args.seed,
                    fold=args.fold, epochs=args.epochs, patience=args.patience,
                    lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch_size, dropout=args.dropout, extra=extra)
    return cfg, args.device


def _auto_type(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s
