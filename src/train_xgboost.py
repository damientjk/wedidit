"""Train CPU XGBoost on frozen train.csv; compare with the saved LR on val.csv.

Run: python src/train_xgboost.py
Native export uses models/xgboost/model.json to preserve the teammate's demo.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train import ROOT, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, validate_frame


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    return {"machine": platform.machine(), "system": platform.system(),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")},
            "s390x_runtime_verified": platform.machine().lower() == "s390x"}


def classification(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "true_positives": int(tp), "false_positives": int(fp),
            "true_negatives": int(tn), "false_negatives": int(fn),
            "claims_flagged": int(tp + fp),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def threshold_metrics(y, scores, threshold):
    return {"threshold": float(threshold), **classification(y, scores >= threshold)}


def smoke_check():
    """Run on LinuxONE to check native imports, CPU training and JSON reload."""
    x = np.array([[0.0], [1.0], [2.0], [3.0]] * 8)
    y = np.array([0, 0, 1, 1] * 8)
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, tree_method="hist", n_jobs=1)
    model.fit(x, y)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "smoke.json"
        model.save_model(path)
        loaded = xgb.XGBClassifier()
        loaded.load_model(path)
        np.testing.assert_allclose(model.predict_proba(x), loaded.predict_proba(x))
    print(json.dumps({"native_smoke_check": "passed", **environment()}, indent=2))


def main():
    data = ROOT / "data" / "processed"
    models = ROOT / "models"
    report = ROOT / "reports" / "xgboost"
    # Protect baseline, demo, and frozen inputs; never open test.csv.
    protected_paths = [models / "baseline.joblib", models / "metrics.json",
                       models / "model.json", data / "train.csv", data / "val.csv"]
    before = {path: sha256(path) for path in protected_paths}
    train = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    val = pd.read_csv(data / "val.csv", encoding="utf-8-sig")
    validate_frame(train, "train", "train")
    validate_frame(val, "validation", "val")
    if "split" not in train or "split" not in val:
        raise ValueError("Frozen split metadata is required")
    if set(train.claim_id) & set(val.claim_id):
        raise ValueError("Train/validation claim IDs overlap")
    negatives = int(train[TARGET].eq(0).sum())
    positives = int(train[TARGET].eq(1).sum())
    weight = negatives / positives
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop")
    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="aucpr",
                  tree_method="hist", n_jobs=1, random_state=42,
                  scale_pos_weight=weight)
    pipeline = Pipeline([("preprocessor", preprocessor),
                         ("classifier", xgb.XGBClassifier(**params))])
    pipeline.fit(train[FEATURES], train[TARGET])
    baseline = joblib.load(models / "baseline.joblib")
    if list(baseline.feature_names_in_) != FEATURES:
        raise ValueError("Baseline feature schema differs")
    y = val[TARGET].to_numpy()
    scores = {"logistic_regression": baseline.predict_proba(val[FEATURES])[:, 1],
              "xgboost": pipeline.predict_proba(val[FEATURES])[:, 1]}
    results = {}
    threshold_rows = []
    workload_rows = []
    # Compare identical top-k queues; claim_id is a label-independent tie-break.
    budgets = sorted({round(len(val) * fraction) for fraction in (0.10, 0.20, 0.25, 0.45)} |
                     {int((scores["logistic_regression"] >= t).sum()) for t in (0.30, 0.50, 0.60)})
    for name, probability in scores.items():
        grid = [threshold_metrics(y, probability, i / 20) for i in range(2, 19)]
        threshold_rows.extend({"model": name, **row} for row in grid)
        eligible = [r for r in grid if r["recall"] >= 0.90]
        low = max(eligible, key=lambda r: r["threshold"]) if eligible else None
        best = max(grid, key=lambda r: (r["f1"], r["recall"]))
        tiers = None
        if low and low["threshold"] < best["threshold"]:
            labels = np.where(probability < low["threshold"], "fast_track",
                              np.where(probability < best["threshold"], "review", "investigate"))
            tiers = {tier: {"claims": int((labels == tier).sum()),
                            "fraud": int(y[labels == tier].sum())}
                     for tier in ("fast_track", "review", "investigate")}
        results[name] = {
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "at_0.5": threshold_metrics(y, probability, 0.5),
            "best_grid_f1": best,
            "candidate_policy": {"review_threshold": low["threshold"] if low else None,
                                 "investigate_threshold": best["threshold"],
                                 "tiers": tiers,
                                 "selection": "review: largest grid threshold with recall >=90%; investigate: best grid F1; validation-selected only"},
        }
        order = np.lexsort((val.claim_id.to_numpy(), -probability))
        for k in budgets:
            pred = np.zeros(len(y), dtype=bool)
            pred[order[:k]] = True
            cutoff = float(probability[order[k - 1]])
            workload_rows.append({"model": name, "workload_fraction": k / len(y),
                                  "score_cutoff": cutoff,
                                  "cutoff_tie_count": int((probability == cutoff).sum()),
                                  **classification(y, pred)})
    native_dir = models / "xgboost"
    native_dir.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models / "xgboost_pipeline.joblib")
    pipeline.named_steps["classifier"].save_model(native_dir / "model.json")
    joblib.dump(pipeline.named_steps["preprocessor"], native_dir / "preprocessor.joblib")
    manifest = {"raw_features": FEATURES,
                "encoded_features": pipeline.named_steps["preprocessor"].get_feature_names_out().tolist(),
                "environment": environment(),
                "inference": "model.json accepts only the matching fitted preprocessor output, in exact encoded column order and sparse/dense representation; never raw claims. Prefer xgboost_pipeline.joblib.",
                "portability": "Native JSON is portable; joblib requires compatible Python/library versions. Verify on target s390x host."}
    (native_dir / "preprocessing.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Verify both inference paths, including sparse representation parity.
    reloaded = joblib.load(models / "xgboost_pipeline.joblib")
    np.testing.assert_allclose(reloaded.predict_proba(val[FEATURES])[:, 1], scores["xgboost"])
    native = xgb.XGBClassifier()
    native.load_model(native_dir / "model.json")
    prep = joblib.load(native_dir / "preprocessor.joblib")
    np.testing.assert_allclose(native.predict_proba(prep.transform(val[FEATURES]))[:, 1], scores["xgboost"])
    assert all(sha256(path) == value for path, value in before.items()), "Protected input changed"
    summary = {"target": TARGET, "selected_features": FEATURES,
               "samples": {"train": len(train), "validation": len(val)},
               "training_class_counts": {"non_fraud": negatives, "fraud": positives},
               "scale_pos_weight": weight, "parameters": params,
               "environment": environment(), "validation_fraud_prevalence": float(y.mean()),
               "validation_results": results, "equal_workload": workload_rows,
               "input_sha256": {str(path.relative_to(ROOT)): value for path, value in before.items()},
               "notes": ["No test.csv access; no fitting on validation; one fixed training configuration, no early stopping or hyperparameter sweep.",
                         "Scores are uncalibrated; thresholds selected on validation require final held-out assessment.",
                         "Equal workloads use top-k scores, with claim_id ascending to break ties; a threshold alone may flag a different number when scores tie.",
                         "Native export is models/xgboost/model.json to preserve teammate demo models/model.json.",
                         "Full pipeline and native-plus-preprocessor reload predictions matched on validation."]}
    (models / "xgboost_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(threshold_rows).drop(columns="confusion_matrix").to_csv(report / "thresholds.csv", index=False)
    pd.DataFrame(workload_rows).drop(columns="confusion_matrix").to_csv(report / "equal_workload.csv", index=False)
    print(json.dumps({"scale_pos_weight": weight, "environment": environment(), "results": results}, indent=2))
    print(pd.DataFrame(workload_rows).drop(columns="confusion_matrix").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility-check", action="store_true", help="Run tiny CPU/native JSON smoke test without dataset access")
    args = parser.parse_args()
    smoke_check() if args.compatibility_check else main()
