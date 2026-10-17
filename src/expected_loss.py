"""Calibration and expected-loss ranking for the investigate queue.

Expected loss = P(fraud) x claim amount. That product needs real probabilities,
so scores are first Platt-calibrated: a one-feature logistic regression on
logit(score). Platt, not isotonic: ~185 fraud cases is too few for isotonic.

Evaluation uses cross-fitted calibration within validation (each fold is
calibrated by a model fit on the other folds), so the calibrator never scores
claims it was fitted on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

LABEL = "FraudFound_P"
AMOUNT = "claim_amount_usd(synth)"
EPS = 1e-6
FOLDS = 5
SEED = 42


def _logit(scores) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(scores, labels) -> LogisticRegression:
    # C large = effectively unregularised, the textbook Platt fit.
    return LogisticRegression(C=1e6).fit(_logit(scores), np.asarray(labels))


def apply_platt(model: LogisticRegression, scores) -> np.ndarray:
    return model.predict_proba(_logit(scores))[:, 1]


def cross_fitted_probability(scores: pd.Series, labels: pd.Series, folds: int = FOLDS) -> pd.Series:
    out = pd.Series(np.nan, index=scores.index)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fit_idx, apply_idx in splitter.split(scores, labels):
        model = fit_platt(scores.iloc[fit_idx], labels.iloc[fit_idx])
        out.iloc[apply_idx] = apply_platt(model, scores.iloc[apply_idx])
    return out


def reliability_table(prob: pd.Series, labels: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed fraud rate per probability decile."""
    decile = pd.qcut(prob.rank(method="first"), bins, labels=False)
    return (pd.DataFrame({"predicted": prob, "observed": labels, "decile": decile})
            .groupby("decile").agg(claims=("observed", "size"), predicted=("predicted", "mean"),
                                   observed=("observed", "mean"))
            .round(4))


def ranking_comparison(df: pd.DataFrame, prob: pd.Series, capacity: float) -> pd.DataFrame:
    """Investigate the top `capacity` share ranked two ways; compare what each catches."""
    n = max(1, int(round(capacity * len(df))))
    fraud = df[LABEL] == 1
    fraud_usd = df.loc[fraud, AMOUNT].sum()
    rankings = {"probability": prob, "expected_loss": prob * df[AMOUNT]}
    rows = []
    for name, key in rankings.items():
        top = key.sort_values(ascending=False).index[:n]
        rows.append({
            "ranked_by": name,
            "claims_investigated": n,
            "fraud_cases_caught": int(fraud[top].sum()),
            "share_fraud_cases": round(fraud[top].sum() / fraud.sum(), 3),
            "fraud_usd_caught": round(df.loc[top, AMOUNT][fraud[top]].sum()),
            "share_fraud_usd": round(df.loc[top, AMOUNT][fraud[top]].sum() / fraud_usd, 3),
            "median_amount_investigated": round(df.loc[top, AMOUNT].median()),
        })
    return pd.DataFrame(rows)
