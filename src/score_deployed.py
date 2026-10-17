"""Score val and test with the deployed model, in the shape risk.py expects.

Also carries over Marcus's out-of-fold training predictions for the same model
configuration, so risk.py can fit the calibrator on training data only.

Must run with the team's pinned versions (requirements.txt: scikit-learn
1.4.1.post1, xgboost 2.0.3); a different scikit-learn can change a loaded
pipeline's output. A reproduction check against Marcus's saved validation
scores guards against that.

Usage: python src/score_deployed.py
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("score_deployed")

DATA = Path("data/processed")
EXPERIMENTS = Path("models/experiments")
MODEL = EXPERIMENTS / "capacity_20261017T042040511095Z/xgboost_09.joblib"
SAVED_VAL = EXPERIMENTS / "capacity_20261017T042040511095Z/validation_predictions.csv"
TRAIN_OOF = EXPERIMENTS / "probability_20261017T050105523504Z/training_oof_predictions.csv"
OOF_COLUMN = "xgboost_raw"  # same locked configuration as xgboost_09
OUT = Path("outputs/xgb_scores.csv")
LABEL = "FraudFound_P"
AMOUNT = "claim_amount_usd(synth)"
KEEP = ["claim_id", "split", LABEL, AMOUNT]


def score_split(model, name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv")
    return df[KEEP].assign(score=model.predict_proba(df)[:, 1])


def check_reproduces_saved(val: pd.DataFrame) -> None:
    saved = pd.read_csv(SAVED_VAL)[["claim_id", "xgboost_09"]]
    merged = val.merge(saved, on="claim_id", validate="1:1")
    gap = np.abs(merged["score"] - merged["xgboost_09"]).max()
    if len(merged) != len(val) or gap > 1e-6:
        raise RuntimeError(f"Loaded model does not reproduce saved validation scores (max gap {gap:.2e})")
    log.info("Reproduction check passed: max gap %.1e on %d validation claims", gap, len(merged))


def train_oof() -> pd.DataFrame:
    oof = pd.read_csv(TRAIN_OOF)[["claim_id", OOF_COLUMN]].rename(columns={OOF_COLUMN: "score"})
    train = pd.read_csv(DATA / "train.csv")[KEEP]
    return train.merge(oof, on="claim_id", validate="1:1")


def main() -> None:
    model = joblib.load(MODEL)
    val = score_split(model, "val")
    check_reproduces_saved(val)
    scored = pd.concat([train_oof(), val, score_split(model, "test")], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT, index=False)
    log.info("Wrote %s: %s", OUT, scored["split"].value_counts().to_dict())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
