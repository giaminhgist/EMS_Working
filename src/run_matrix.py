"""Prioritized experiment matrix runner.

Executes experiment commands (from experiment/matrix.json) in priority order
using N parallel workers, logs status to experiment/matrix_status.csv and
per-experiment stdout/stderr to outputs/.matrix_logs/{id}.log.

Usage (from EMS-Minh/src):
    python run_matrix.py                 # all pending
    python run_matrix.py --ids EXP-PROP-001-s42 EXP-PROP-007-s42
    python run_matrix.py --workers 4
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.common import ROOT  # noqa: E402

MATRIX = ROOT / "experiment" / "matrix.json"
STATUS = ROOT / "experiment" / "matrix_status.csv"
LOG_DIR = ROOT / "outputs" / ".matrix_logs"


def load_matrix():
    return json.loads(MATRIX.read_text())["experiments"]


def expand(exps):
    """Expand entries with a "seeds" field into one job per seed.

    Expanded ids get a -s{seed} suffix and --seed is appended to the command.
    """
    for e in exps:
        seeds = e.get("seeds") or [None]
        for s in seeds:
            row = {k: v for k, v in e.items() if k != "seeds"}
            row["cmd"] = list(e["cmd"])
            if s is not None:
                row["cmd"] += ["--seed", str(s)]
                row["id"] = f"{e['id']}-s{s}"
            yield row


def load_status():
    if STATUS.exists():
        return pd.read_csv(STATUS, index_col=0)
    return pd.DataFrame(columns=["status", "started", "finished", "duration_s",
                                 "exit", "tail"])


def run_one(exp):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{exp['id']}.log"
    started = datetime.now().isoformat(timespec="seconds")
    with open(log, "a") as f:
        f.write(f"### {exp['id']} started {started}\n$ {' '.join(exp['cmd'])}\n")
    t0 = datetime.now()
    proc = subprocess.run(exp["cmd"], capture_output=True, text=True, cwd=ROOT / "src")
    dur = (datetime.now() - t0).total_seconds()
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
    with open(log, "a") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
        f.write(f"\n### exit={proc.returncode} duration={dur:.0f}s\n")
    return {"id": exp["id"], "status": "done" if proc.returncode == 0 else "failed",
            "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(dur, 1), "exit": proc.returncode, "tail": tail[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-run done experiments")
    args = ap.parse_args()

    exps = list(expand(load_matrix()))
    if args.ids:
        exps = [e for e in exps if e["id"] in args.ids]
    status = load_status()
    if status.index.name is None:
        status.index.name = "id"
    todo = [e for e in exps
            if args.force or e["id"] not in status.index
            or status.loc[e["id"], "status"] in ("failed", "pending")]
    print(f"{len(todo)}/{len(exps)} experiments to run with {args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, e): e for e in todo}
        for fut in as_completed(futures):
            exp = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"id": exp["id"], "status": "failed", "started": "",
                       "finished": "", "duration_s": 0, "exit": -1,
                       "tail": str(exc)[:400]}
            exp_id = row.pop("id")
            status.loc[exp_id] = row
            status.to_csv(STATUS)
            print(f"[{row['status']:6s}] {exp_id}  ({row['duration_s']:.0f}s)  "
                  f"{row['tail'].strip().splitlines()[-1][:100] if row['tail'] else ''}")
    print(f"status -> {STATUS}")


if __name__ == "__main__":
    main()
