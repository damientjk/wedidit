"""Final headline numbers on the test set, with bootstrap confidence intervals.

Test has only ~185 fraud cases, so "we catch 95%" could easily be 92% or 98%.
The bootstrap estimates that uncertainty: resample the test claims with
replacement many times, recompute each metric, and report the middle 95%.
"""
import numpy as np
import pandas as pd

LABEL = "FraudFound_P"
AMOUNT = "claim_amount_usd(synth)"
N_BOOT = 1000
SEED = 42


def headline_metrics(df: pd.DataFrame) -> dict:
    """df needs: tier, FraudFound_P, claim amount."""
    fraud = df[LABEL] == 1
    to_human = df["tier"] != "fast_track"
    fraud_usd = df.loc[fraud, AMOUNT].sum()
    return {
        "share_fast_tracked": (~to_human).mean(),
        "share_fraud_to_human": (to_human & fraud).sum() / fraud.sum(),
        "share_fraud_investigated": (df["tier"].eq("investigate") & fraud).sum() / fraud.sum(),
        "fraud_rate_fast_track": df.loc[~to_human, LABEL].mean(),
        "share_fraud_usd_to_human": df.loc[to_human & fraud, AMOUNT].sum() / fraud_usd,
    }


def bootstrap(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = [headline_metrics(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)]
    boot = pd.DataFrame(draws)
    point = pd.Series(headline_metrics(df))
    return pd.DataFrame({
        "estimate": point,
        "ci_low": boot.quantile(0.025),
        "ci_high": boot.quantile(0.975),
    }).round(3)


def headline_sentence(table: pd.DataFrame) -> str:
    ft, fh = table.loc["share_fast_tracked"], table.loc["share_fraud_to_human"]
    return (f"Fast-track {ft.estimate:.0%} of claims [{ft.ci_low:.0%}-{ft.ci_high:.0%}] "
            f"while routing {fh.estimate:.0%} of fraud [{fh.ci_low:.0%}-{fh.ci_high:.0%}] to a human.")
