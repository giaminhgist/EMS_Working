"""Generate per-experiment reports in experiment/ from fold summaries + matrix.

Usage (from EMS-Minh): python src/make_reports.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.common import ROOT  # noqa: E402
from run_matrix import expand  # noqa: E402

EXP_DIR = ROOT / "experiment"


def collect():
    exps = list(expand(json.loads((EXP_DIR / "matrix.json").read_text())["experiments"]))
    status = pd.read_csv(EXP_DIR / "matrix_status.csv", index_col="id") \
        if (EXP_DIR / "matrix_status.csv").exists() else None
    reports = {}
    for summary_file in sorted((ROOT / "outputs").glob("proposal*/**/*allfolds_summary.json")):
        d = json.loads(summary_file.read_text())
        e = next((e for e in exps
                  if e["cmd"][1] == f"{d['proposal']}/train.py"
                  and "--ablation" in e["cmd"]
                  and e["cmd"][e["cmd"].index("--ablation") + 1] == d["ablation"]
                  and ("--seed" not in e["cmd"]
                       or e["cmd"][e["cmd"].index("--seed") + 1] == str(d["seed"]))),
                 None)
        if e is None:
            continue
        reports[e["id"]] = {"summary": d, "hypothesis": e["hypothesis"],
                            "cmd": " ".join(e["cmd"])}
    return reports, status


def main():
    reports, status = collect()
    for eid in sorted(reports):
        r = reports[eid]
        d = r["summary"]
        mm = d["mean_metrics"]
        folds = pd.DataFrame(d["fold_metrics"])[
            ["fold", "auc", "acc", "balanced_acc", "sen", "spe", "f1"]]
        st = "done" if status is None or eid not in status.index else status.loc[eid, "status"]
        lines = [
            f"# {eid} — {d['proposal']}/{d['ablation']}",
            "",
            f"Status: **{st}** · Seed {d['seed']} · Protocol: official 4-fold CV "
            "(Set_0..3, threshold 0.5)",
            "",
            "## Setup",
            "",
            f"Command: `{' '.join(r['cmd'])}`",
            "",
            f"Hypothesis: {r['hypothesis']}",
            "",
            "## Results (validation folds, mean ± std over 4 folds)",
            "",
            "| Metric | Mean ± std |",
            "|---|---|",
        ]
        for k in ["auc", "acc", "balanced_acc", "sen", "spe", "f1"]:
            lines.append(f"| {k} | {mm[k]['mean']:.4f} ± {mm[k]['std']:.4f} |")
        lines += [
            "",
            "| Fold | AUC | Acc | BalAcc | Sen | Spe | F1 |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in folds.iterrows():
            lines.append(f"| {row['fold']} | {row['auc']:.4f} | {row['acc']:.4f} | "
                         f"{row['balanced_acc']:.4f} | {row['sen']:.4f} | {row['spe']:.4f} | {row['f1']:.4f} |")
        lines += [
            "",
            "## Observations / Conclusion",
            "",
            "_To be filled after analysis (see experiment_tracker.md)._",
            "",
        ]
        (EXP_DIR / f"{eid}.md").write_text("\n".join(lines))
        print(f"wrote {eid}.md")
    print(f"{len(reports)} reports written")


if __name__ == "__main__":
    main()
