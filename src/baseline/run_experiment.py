"""Run one baseline method under one protocol.

Usage:
    python run_experiment.py --method svm_rbf --protocol P1 [--seed 42]
    python run_experiment.py --method fnn --protocol P2 --rep agg --seed 42

Protocols (see protocols.py):
    P1 : official 4-fold cross-validation on the 160 labelled subjects
         (mean±std over folds) + predictions for the 48 official test subjects.
    P2 : stratified 120 (train/val) / 40 (test) subject split; metrics on the
         40-subject test set.

All classification metrics use the fixed threshold 0.5 (AUC is threshold-free).
Outputs are written to docs/baseline/results/{protocol}/{method}__{rep}/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, SEED, THRESHOLD, NUM_STIMULI  # noqa: E402
import protocols  # noqa: E402
from features_builder import build_agg, build_catagg, build_concat  # noqa: E402
from metrics_utils import compute_metrics, metrics_table_mean_std  # noqa: E402
from models import fit_predict_ml, train_fnn, predict_fnn  # noqa: E402

METHOD_REP = {  # which subject-level representation each method uses
    "svm_rbf": "agg", "svm_linear": "agg", "rf": "agg", "qda": "agg",
    "gnb": "agg", "lr": "agg", "knn": "agg", "fnn": "agg",
    "fnn_cat": "catagg", "lr_l1": "concat",
}


def build_matrix(rep, subject_ids, partition="train"):
    if rep == "agg":
        return build_agg(subject_ids, partition)
    if rep == "catagg":
        return build_catagg(subject_ids, partition)
    if rep == "concat":
        return build_concat(subject_ids, partition)
    raise ValueError(rep)


def run_protocol1(method, rep, out_dir, seed=SEED):
    """Official 4-fold CV on the 160 train subjects + official-test predictions."""
    fold_results = []
    all_preds = {}
    for fold_name, train_ids, val_ids in protocols.protocol1_folds():
        X = build_matrix(rep, sorted(train_ids + val_ids))
        X_train, y_train = X.loc[train_ids], protocols.load_metadata().loc[train_ids, "label"]
        X_val, y_val = X.loc[val_ids], protocols.load_metadata().loc[val_ids, "label"]
        if method.startswith("fnn"):
            from sklearn.model_selection import train_test_split
            inner_tr, inner_va = train_test_split(train_ids, test_size=0.25,
                                                  stratify=y_train, random_state=seed)
            model, _ = train_fnn(X.loc[inner_tr].values, y_train.loc[inner_tr].values,
                                 X.loc[inner_va].values, y_train.loc[inner_va].values,
                                 in_dim=X.shape[1], seed=seed)
            pv = predict_fnn(model, X_val.values)
        else:
            pv = fit_predict_ml(method, X_train.values, y_train.values, X_val.values)
        m = compute_metrics(y_val.values, pv, THRESHOLD)
        m["fold"] = fold_name
        fold_results.append(m)
        for sid, p in zip(val_ids, pv):
            all_preds[int(sid)] = float(p)
        print(f"  {fold_name}: auc={m['auc']:.4f} acc={m['acc']:.4f} "
              f"sen={m['sen']:.4f} spe={m['spe']:.4f} f1={m['f1']:.4f}")

    mean_m = metrics_table_mean_std(fold_results)
    summary = {k: {"mean": m, "std": s} for k, (m, s) in mean_m.items()}
    summary["fold_metrics"] = fold_results

    # official test predictions (labels withheld -> probabilities only)
    meta = protocols.load_metadata()
    all_train_ids = protocols.train_subjects()
    X = build_matrix(rep, all_train_ids)
    y = meta.loc[all_train_ids, "label"]
    test_ids = meta[meta.partition == "test"].index.tolist()
    X_test = build_matrix(rep, test_ids, partition="test")
    if method.startswith("fnn"):
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(np.arange(len(all_train_ids)),
                                          test_size=0.25, stratify=y,
                                          random_state=seed)
        model, _ = train_fnn(X.iloc[tr_idx].values, y.iloc[tr_idx].values,
                             X.iloc[va_idx].values, y.iloc[va_idx].values,
                             in_dim=X.shape[1], seed=seed)
        p_test = predict_fnn(model, X_test.values)
    else:
        p_test = fit_predict_ml(method, X.values, y.values, X_test.values)
    file_ids = [f"Test_{int(meta.loc[s, 'file_id']):03d}" for s in test_ids]
    pd.DataFrame({"subject_id": file_ids, "prob": p_test}).to_csv(
        out_dir / "official_test_preds.csv", index=False)

    # per-subject validation predictions
    pd.DataFrame([{"subject_id": k, "prob": v} for k, v in all_preds.items()]) \
        .sort_values("subject_id").to_csv(out_dir / "val_preds.csv", index=False)
    return summary


def run_protocol2(method, rep, out_dir, seed=SEED):
    """Stratified 120/40 split; metrics on the 40-subject test set."""
    split = protocols.protocol2_split(seed=seed)
    meta = protocols.load_metadata()
    tv_ids = sorted(split["train_val_ids"])
    test_ids = sorted(split["test_ids"])
    X = build_matrix(rep, tv_ids + test_ids)
    y = meta.loc[tv_ids + test_ids, "label"]
    X_tv, X_test = X.loc[tv_ids], X.loc[test_ids]
    y_tv, y_test = y.loc[tv_ids], y.loc[test_ids]

    if method.startswith("fnn"):
        inner_tr = sorted(split["inner_train_ids"])
        inner_va = sorted(split["inner_val_ids"])
        model, _ = train_fnn(X_tv.loc[inner_tr].values, y_tv.loc[inner_tr].values,
                             X_tv.loc[inner_va].values, y_tv.loc[inner_va].values,
                             in_dim=X.shape[1], seed=seed)
        p_test = predict_fnn(model, X_test.values)
    else:
        p_test = fit_predict_ml(method, X_tv.values, y_tv.values, X_test.values)

    m = compute_metrics(y_test.values, p_test, THRESHOLD)
    print(f"  test(40): auc={m['auc']:.4f} acc={m['acc']:.4f} "
          f"sen={m['sen']:.4f} spe={m['spe']:.4f} f1={m['f1']:.4f}")
    pd.DataFrame({"subject_id": test_ids, "label": y_test.values,
                  "prob": p_test}).to_csv(out_dir / "test_preds.csv", index=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--protocol", required=True, choices=["P1", "P2"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    method = args.method
    rep = METHOD_REP[method]
    out_dir = RESULTS / args.protocol / f"{method}__{rep}" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {method} [{rep}] protocol {args.protocol} seed {args.seed} ==")
    if args.protocol == "P1":
        summary = run_protocol1(method, rep, out_dir, seed=args.seed)
    else:
        summary = run_protocol2(method, rep, out_dir, seed=args.seed)

    result = {"method": method, "rep": rep, "protocol": args.protocol,
              "seed": args.seed, "threshold": THRESHOLD, **summary}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
