"""Train the demo models: logistic regression baseline + XGBoost.

Writes everything as JSON (no pickles, so files load the same on s390x):
  models/model.json    XGBoost booster
  models/lr.json       scaler + coefficients for logistic regression
  models/trees.json    first few XGBoost trees, for the live tree view
  models/demo_metrics.json  AUC / precision / recall on val and test
                       (models/metrics.json belongs to Marcus's real model)

Run:  python src/train_demo.py
"""
import json
import platform
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from synth_data import FEATURES, make, split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
TIERS = {"review": 0.30, "investigate": 0.60}  # ensemble probability cut-offs
N_TREES_SHOWN = 6


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
    }


def main():
    OUT.mkdir(exist_ok=True)
    s = split(make())
    Xtr, ytr = s["train"][FEATURES], s["train"]["fraud"]

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)

    pos = ytr.mean()
    model = xgb.XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.15,
        scale_pos_weight=(1 - pos) / pos, eval_metric="auc",
    ).fit(Xtr, ytr)
    model.get_booster().save_model(str(OUT / "model.json"))

    baseline = {f: float(np.median(Xtr[f])) for f in FEATURES}
    (OUT / "lr.json").write_text(json.dumps({
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "baseline": baseline,
        "tiers": TIERS,
    }, indent=2))

    dump = model.get_booster().get_dump(dump_format="json")[:N_TREES_SHOWN]
    (OUT / "trees.json").write_text(json.dumps([json.loads(t) for t in dump]))

    metrics = {"platform": platform.machine(), "data": "synthetic placeholder",
               "rows": {k: len(v) for k, v in s.items()},
               "fraud_rate_train": round(float(pos), 4), "threshold": TIERS["review"],
               "n_trees": model.n_estimators, "xgboost_version": xgb.__version__}
    for name in ("val", "test"):
        X, y = s[name][FEATURES], s[name]["fraud"]
        p_lr = lr.predict_proba(scaler.transform(X))[:, 1]
        p_xgb = model.predict_proba(X)[:, 1]
        metrics[name] = {
            "logistic_regression": evaluate(y, p_lr, TIERS["review"]),
            "xgboost": evaluate(y, p_xgb, TIERS["review"]),
            "ensemble": evaluate(y, (p_lr + p_xgb) / 2, TIERS["review"]),
        }
    (OUT / "demo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
