"""Synthetic motor-claim data for the demo.

PLACEHOLDER until Yu Heng's processed Kaggle data lands in data/processed/.
The columns match the five facts the PDF reader can pull from a claim, so the
same model serves both the form and the PDF flow. Fraud is generated from a
known rule (including interactions a linear model cannot fully capture), so
XGBoost should beat logistic regression.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "claims_last_12m",
    "days_policy_to_incident",
    "vehicle_age",
    "claim_amount",
    "police_report",
]


def make(n=15000, seed=7):
    rng = np.random.default_rng(seed)
    claims = rng.poisson(0.6, n).clip(0, 8)
    days = rng.exponential(420, n).round().clip(0, 3650)
    vage = rng.gamma(2.2, 3.0, n).round().clip(0, 25)
    amount = rng.lognormal(8.4, 0.75, n).round(-1).clip(200, 60000)
    police = rng.binomial(1, 0.72, n)

    z = (
        -3.4
        + 0.55 * claims
        + 1.7 * (days < 30)
        + 0.6 * (days < 120)
        + 0.05 * (vage - 6)
        + 0.9 * np.log(amount / 4500).clip(-2, None)
        + 0.8 * (1 - police)
        + 1.1 * ((claims >= 3) & (days < 60))       # repeat claimant on a fresh policy
        + 1.0 * ((amount > 12000) & (police == 0))  # big claim, no police report
    )
    fraud = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({
        "claims_last_12m": claims,
        "days_policy_to_incident": days,
        "vehicle_age": vage,
        "claim_amount": amount,
        "police_report": police,
        "fraud": fraud,
    })


def split(df, seed=7):
    """Stratified 9k / 3k / 3k split on the fraud label."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("fraud"):
        idx = rng.permutation(g.index.to_numpy())
        a, b = int(len(idx) * 0.6), int(len(idx) * 0.8)
        parts["train"] += list(idx[:a])
        parts["val"] += list(idx[a:b])
        parts["test"] += list(idx[b:])
    return {k: df.loc[sorted(v)].reset_index(drop=True) for k, v in parts.items()}
