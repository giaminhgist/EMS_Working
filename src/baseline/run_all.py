"""Driver: run every baseline method under both protocols and aggregate results.

Usage:
    python run_all.py                 # all methods x protocols
    python run_all.py --methods svm_rbf,fnn   # subset
    python run_all.py --protocols P1

Writes docs/baseline/results/summary.csv (one row per method x protocol x seed).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

ALL_METHODS = ["svm_rbf", "svm_linear", "rf", "qda", "gnb", "lr", "lr_l1",
               "knn", "fnn", "fnn_cat"]
P2_SEEDS = [42, 2024, 2026]


def run_one(method, protocol, seed=42):
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_experiment.py"),
           "--method", method, "--protocol", protocol, "--seed", str(seed)]
    subprocess.run(cmd, check=True)


def collect():
    rows = []
    for protocol in ["P1", "P2"]:
        for mdir in sorted((RESULTS / protocol).glob("*")):
            if not mdir.is_dir():
                continue
            for jf in sorted(mdir.glob("seed*/metrics.json")):
                d = json.loads(jf.read_text())
                row = {k: d[k] for k in ["method", "rep", "protocol", "seed",
                                         "threshold"]}
                for k in ["acc", "sen", "spe", "pre", "f1", "auc"]:
                    if isinstance(d.get(k), dict):      # P1: {mean, std}
                        row[k] = d[k]["mean"]
                        row[k + "_std"] = d[k]["std"]
                    else:                               # P2: scalar
                        row[k] = d[k]
                        row[k + "_std"] = float("nan")
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(["protocol", "method", "seed"])
    df.to_csv(RESULTS / "summary.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--protocols", default="P1,P2")
    ap.add_argument("--only-collect", action="store_true",
                    help="skip running, just rebuild summary.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")
    protocols = args.protocols.split(",")

    if not args.only_collect:
        for protocol in protocols:
            for method in methods:
                seeds = [42] if protocol == "P1" else P2_SEEDS
                for seed in seeds:
                    run_one(method, protocol, seed)
    df = collect()
    print(df.to_string(index=False))
    print(f"\nsaved -> {RESULTS}/summary.csv")


if __name__ == "__main__":
    main()
