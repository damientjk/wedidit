"""Bounded training-only CV, then one frozen-validation comparison.

Run: python src/experiment_capacity.py
Writes a new timestamped models/experiments/capacity_* directory only.
"""
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import hashlib
import json
from datetime import datetime, timezone
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from train import ROOT, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, validate_frame
from train_xgboost import environment

CAPACITIES = (0.10, 0.20, 0.30)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queue(y, scores, ids, k):
    order = np.lexsort((np.asarray(ids), -np.asarray(scores)))
    selected = order[:k]
    tp = int(np.asarray(y)[selected].sum())
    positives = int(np.asarray(y).sum())
    return {"reviewed": k, "fraud_caught": tp, "missed_fraud": positives - tp,
            "false_positives": k - tp, "precision": tp / k,
            "recall": tp / positives, "score_cutoff": float(scores[order[k - 1]])}


def candidates():
    configs = []
    for c in (0.01, 0.1, 1.0, 10.0):
        for balanced in (False, True):
            configs.append({"family": "logistic", "C": c, "balanced": balanced})
    for depth, trees in ((2, 200), (2, 500), (3, 200), (3, 500), (4, 300), (5, 200)):
        for weight in ("one", "sqrt_ratio", "half_ratio", "ratio"):
            configs.append({"family": "xgboost", "depth": depth, "trees": trees, "weight": weight})
    for leaves in (7, 15):
        for balanced in (False, True):
            configs.append({"family": "hist_gradient_boosting", "leaves": leaves, "balanced": balanced})
    return [{"candidate": f"{c['family']}_{i:02d}", **c} for i, c in enumerate(configs)]


def build(config, y):
    family = config["family"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if family == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=family != "hist_gradient_boosting")),
        ]), CATEGORICAL_FEATURES),
    ])
    if family == "logistic":
        classifier = LogisticRegression(C=config["C"], class_weight="balanced" if config["balanced"] else None,
                                        max_iter=1000, random_state=42)
    elif family == "xgboost":
        ratio = float((np.asarray(y) == 0).sum() / (np.asarray(y) == 1).sum())
        weight = {"one": 1.0, "sqrt_ratio": np.sqrt(ratio), "half_ratio": ratio / 2, "ratio": ratio}[config["weight"]]
        classifier = XGBClassifier(n_estimators=config["trees"], max_depth=config["depth"],
                                   learning_rate=0.04, min_child_weight=5, reg_lambda=5,
                                   subsample=1, colsample_bytree=1, scale_pos_weight=float(weight),
                                   objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                                   max_bin=128, n_jobs=1, random_state=42)
    else:
        classifier = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=config["leaves"],
                       learning_rate=0.05, l2_regularization=5, min_samples_leaf=30,
                       class_weight="balanced" if config["balanced"] else None,
                       early_stopping=False, random_state=42)
    return Pipeline([("preprocessor", prep), ("classifier", classifier)])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    started = time.monotonic()
    output = ROOT / "models" / "experiments" / ("capacity_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot existing models before creating experimental outputs; no test access.
    protected = {p: digest(p) for p in (ROOT / "models").rglob("*")
                 if p.is_file() and "experiments" not in p.relative_to(ROOT / "models").parts}
    train_path = ROOT / "data" / "processed" / "train.csv"
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    validate_frame(train, "train", "train")
    assert train["split"].eq("train").all()
    x, y, ids = train[FEATURES], train[TARGET].to_numpy(), train.claim_id.to_numpy()
    configs = candidates()
    protocol = {"features": FEATURES, "target": TARGET, "training_sha256": digest(train_path),
                "cv": "3-fold StratifiedKFold(shuffle=True, random_state=42)",
                "selection": "descending mean fold recall across 10%,20%,30%; ties: mean AP, recall20, candidate name",
                "finalists": "one winner per family, fixed before validation is opened; global CV winner recommended independently of validation",
                "capacities": CAPACITIES, "queue": "round(n*capacity), matching prior 617/3084 benchmark; ties by claim_id ascending",
                "resource_limits": "sequential, 1 native/BLAS thread; max 500 XGB trees; bounded 19-column one-hot (~130 columns); no parallel search",
                "candidates": configs, "environment": environment()}
    write_json(output / "protocol.json", protocol)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(x, y))
    fold_rows = []
    for config in configs:
        for fold, (fit_idx, hold_idx) in enumerate(folds):
            model = build(config, y[fit_idx])
            model.fit(x.iloc[fit_idx], y[fit_idx])
            scores = model.predict_proba(x.iloc[hold_idx])[:, 1]
            row = {"candidate": config["candidate"], "family": config["family"], "fold": fold,
                   "ap": float(average_precision_score(y[hold_idx], scores))}
            for capacity in CAPACITIES:
                q = queue(y[hold_idx], scores, ids[hold_idx], round(len(hold_idx) * capacity))
                row[f"recall{round(capacity*100)}"] = q["recall"]
            row["capacity_recall"] = float(np.mean([row[f"recall{round(c*100)}"] for c in CAPACITIES]))
            fold_rows.append(row)
            del model
        print(f"CV completed {config['candidate']}", flush=True)
    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(output / "cv_folds.csv", index=False)
    ranking = folds_frame.groupby(["candidate", "family"], as_index=False)[["capacity_recall", "ap", "recall10", "recall20", "recall30"]].mean()
    ranking = ranking.sort_values(["capacity_recall", "ap", "recall20", "candidate"], ascending=[False, False, False, True])
    ranking.to_csv(output / "cv_ranking.csv", index=False)
    finalists = ranking.drop_duplicates("family").candidate.tolist()
    chosen = ranking.iloc[0].candidate
    write_json(output / "selection.json", {"recommended_by_training_cv": chosen, "finalists": finalists,
                                           "validation_not_opened_yet": True})
    fitted = {}
    config_map = {c["candidate"]: c for c in configs}
    for name in finalists:
        model = build(config_map[name], y)
        model.fit(x, y)
        joblib.dump(model, output / f"{name}.joblib")
        fitted[name] = model
    # First and only validation comparison after all hyperparameters are locked.
    val_path = ROOT / "data" / "processed" / "val.csv"
    val = pd.read_csv(val_path, encoding="utf-8-sig")
    validate_frame(val, "val", "val")
    assert val["split"].eq("val").all()
    assert set(train.claim_id).isdisjoint(set(val.claim_id))
    fitted["working_logistic"] = joblib.load(ROOT / "models" / "baseline.joblib")
    fitted["working_xgboost"] = joblib.load(ROOT / "models" / "xgboost_pipeline.joblib")
    vy, vids = val[TARGET].to_numpy(), val.claim_id.to_numpy()
    validation, capacities, tiers = [], [], {}
    prediction_frame = pd.DataFrame({"claim_id": vids, "actual_fraud": vy})
    for name, model in fitted.items():
        scores = model.predict_proba(val[FEATURES])[:, 1]
        prediction_frame[name] = scores
        row = {"model": name, "cv_selected": name == chosen,
               "average_precision": float(average_precision_score(vy, scores)),
               "roc_auc": float(roc_auc_score(vy, scores))}
        for capacity in CAPACITIES:
            q = queue(vy, scores, vids, round(len(val) * capacity))
            capacities.append({"model": name, "capacity": capacity, **q})
            row[f"caught{round(capacity*100)}"] = q["fraud_caught"]
        q = queue(vy, scores, vids, 786)
        row["caught_at_786"] = q["fraud_caught"]
        capacities.append({"model": name, "capacity": 786/len(val), **q})
        validation.append(row)
        order = np.lexsort((vids, -scores))
        k10, k20 = round(len(val)*0.10), round(len(val)*0.20)
        tiers[name] = {}
        for tier, subset in (("investigate", order[:k10]), ("review", order[k10:k20]),
                             ("expedited_standard", order[k20:])):
            fraud = int(vy[subset].sum())
            tiers[name][tier] = {"claims": len(subset), "fraud": fraud, "non_fraud": int(len(subset)-fraud)}
        if name in finalists:
            loaded = joblib.load(output / f"{name}.joblib")
            np.testing.assert_allclose(loaded.predict_proba(val[FEATURES])[:, 1], scores)
    table = pd.DataFrame(validation).sort_values(["caught20", "average_precision"], ascending=False)
    table.to_csv(output / "validation_comparison.csv", index=False)
    pd.DataFrame(capacities).to_csv(output / "validation_capacity.csv", index=False)
    prediction_frame.to_csv(output / "validation_predictions.csv", index=False)
    assert all(digest(p) == value for p, value in protected.items())
    write_json(output / "results.json", {"protocol": protocol, "recommended_by_training_cv": chosen,
               "finalists": finalists, "configurations_selected": {n: config_map[n] for n in finalists},
               "validation": validation, "capacity_metrics": capacities, "tiers": tiers,
               "train_rows": len(train), "train_fraud": int(y.sum()), "validation_rows": len(val),
               "validation_fraud": int(vy.sum()), "validation_sha256": digest(val_path),
               "protected_model_sha256": {str(p.relative_to(ROOT)): h for p, h in protected.items()},
               "elapsed_seconds": time.monotonic()-started,
               "caveats": ["Prior development has already used validation; it is not an unbiased final holdout.",
                           "One validation comparison this experiment; no further tuning based on these results.",
                           "No test data used; no deployment; uncertainty remains with 184 validation fraud cases.",
                           "Fixed top-k controls model escalations only; safeguard/manual-rule referrals must be counted separately."]})
    print("OUTPUT", output, flush=True)
    print("TRAINING_CV_SELECTION", chosen, flush=True)
    print(ranking.head(10).to_string(index=False), flush=True)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(tiers[chosen], indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
