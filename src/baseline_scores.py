"""Stand-in scores for building risk.py before the real model lands.

Trains a plain logistic regression on train.csv and writes validation and
test scores to outputs/baseline_scores.csv (claim_id, split, label, amount,
score). risk.py only needs that file shape, so Marcus's scores can be
swapped in later without code changes.

Usage: python src/baseline_scores.py
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("baseline_scores")

DATA = Path("data/processed")
OUT = Path("outputs/baseline_scores.csv")
LABEL = "FraudFound_P"
AMOUNT = "claim_amount_usd(synth)"
NOT_FEATURES = [
    LABEL, "claim_id", "split", "RepNumber",
    "incident_type(synth)", "accident_severity(synth)", AMOUNT,
]


def encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode; reindex to the training columns so val/test can't add any."""
    x = pd.get_dummies(df.drop(columns=NOT_FEATURES), dtype=float)
    return x if columns is None else x.reindex(columns=columns, fill_value=0.0)


def main() -> None:
    train, val, test = (pd.read_csv(DATA / f"{s}.csv") for s in ("train", "val", "test"))
    x_train = encode(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model.fit(x_train, train[LABEL])

    scored = pd.concat([
        df[["claim_id", "split", LABEL, AMOUNT]].assign(
            score=model.predict_proba(encode(df, list(x_train.columns)))[:, 1])
        for df in (val, test)
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT, index=False)
    log.info("Wrote %d scored rows to %s", len(scored), OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
