"""Risk layer: turn model scores into fast_track / review / investigate.

Input: a CSV with claim_id, split, FraudFound_P, claim_amount_usd(synth), score
(outputs/baseline_scores.csv now; Marcus's scores later, same shape).
Stage 1 (validation): choose thresholds, check calibration, compare rankings.
Stage 2 (test, once): apply the frozen thresholds; report fairness and the
headline with bootstrap confidence intervals.

Usage: python src/risk.py [scores_csv] [--val-only]
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from expected_loss import (
    apply_platt, cross_fitted_probability, fit_platt, ranking_comparison, reliability_table,
)
from fairness import fairness_report
from headline import bootstrap, headline_sentence

log = logging.getLogger("risk")

SCORES = Path("outputs/baseline_scores.csv")
CLEAN_DATA = Path("data/processed/claims_clean.csv")
SENSITIVE_COLUMNS = ["Sex", "MaritalStatus", "AgeOfPolicyHolder"]
OUT_DIR = Path("outputs")
LABEL = "FraudFound_P"
AMOUNT = "claim_amount_usd(synth)"

# Knee of the baseline triage curve: 2%->5% leakage buys ~23 pts of fast-track,
# 5%->10% buys only ~5. Recheck on the final model (leakage_sensitivity.csv).
MAX_FRAUD_LEAKED = 0.05  # policy: fast-track may let through at most 5% of fraud
LEAKAGE_LIMITS = (0.02, 0.05, 0.10, 0.15, 0.20)
INVESTIGATE_CAPACITY = 0.10  # staffing assumption: investigators take the riskiest 10%
CAPACITIES = (0.05, 0.10, 0.15, 0.20)
# Sensitivity only: what a pure cost rule would pick. 1995 USD; x2.73 for S$ today.
INVESTIGATION_COSTS_USD = (37, 100, 200, 300, 500, 800)


def triage_curve(scores: pd.Series, is_fraud: pd.Series) -> pd.DataFrame:
    """One row per cut-off: share of claims fast-tracked and of fraud leaked below it.

    The cut-off is the score of the next claim, so `score < cutoff` is fast-tracked.
    """
    order = np.argsort(scores.to_numpy(), kind="stable")
    s = scores.to_numpy()[order]
    f = is_fraud.to_numpy()[order]
    n = len(s)
    cutoffs = np.append(s[1:], np.inf)
    curve = pd.DataFrame({
        "cutoff": cutoffs,
        "share_fast_tracked": np.arange(1, n + 1) / n,
        "share_fraud_leaked": np.cumsum(f) / f.sum(),
    })
    # A cut-off can't split tied scores: only keep rows where the next score is higher.
    splittable = np.append(s[1:] > s[:-1], True)
    return curve[splittable].reset_index(drop=True)


def fast_track_cutoff(curve: pd.DataFrame, max_leaked: float = MAX_FRAUD_LEAKED) -> float:
    """Highest cut-off whose fast-track lane leaks at most `max_leaked` of all fraud."""
    ok = curve[curve["share_fraud_leaked"] <= max_leaked]
    if ok.empty:
        raise ValueError("No cut-off keeps fraud leakage under the limit")
    return float(ok["cutoff"].max())


def investigation_cost_curve(df: pd.DataFrame, cost_per_case: float) -> pd.DataFrame:
    """Total cost for each candidate cut-off B, on validation data.

    Investigating every claim with score >= B costs `cost_per_case` each and
    stops that fraud. Fraud below B is paid out (its synthetic amount).
    """
    d = df.sort_values("score", ascending=False)
    n_investigated = np.arange(1, len(d) + 1)
    fraud_stopped = np.cumsum(d[AMOUNT].to_numpy() * d[LABEL].to_numpy())
    fraud_total = fraud_stopped[-1]
    return pd.DataFrame({
        "cutoff": d["score"].to_numpy(),
        "share_investigated": n_investigated / len(d),
        "total_cost": n_investigated * cost_per_case + (fraud_total - fraud_stopped),
    })


def cost_optimal_cutoff(df: pd.DataFrame, cost_per_case: float) -> float:
    curve = investigation_cost_curve(df, cost_per_case)
    return float(curve.loc[curve["total_cost"].idxmin(), "cutoff"])


def investigate_cutoff(scores: pd.Series, capacity: float = INVESTIGATE_CAPACITY) -> float:
    """Lowest score among the riskiest `capacity` share of claims."""
    if not 0 < capacity < 1:
        raise ValueError("capacity must be between 0 and 1")
    n = max(1, int(round(capacity * len(scores))))
    return float(np.sort(scores.to_numpy())[-n])


def cost_sensitivity(df: pd.DataFrame, costs=INVESTIGATION_COSTS_USD) -> pd.DataFrame:
    """What a pure cost rule would choose at each investigation cost (for the README)."""
    total_fraud = df[LABEL].sum()
    rows = []
    for cost in costs:
        b = cost_optimal_cutoff(df, cost)
        inv = df["score"] >= b
        rows.append({
            "cost_usd_1995": cost, "cost_sgd_today": round(cost * 2.73, -1), "cutoff": round(b, 4),
            "share_investigated": round(inv.mean(), 3),
            "share_fraud_caught": round(df.loc[inv, LABEL].sum() / total_fraud, 3),
            "fraud_rate_investigated": round(df.loc[inv, LABEL].mean(), 3),
        })
    return pd.DataFrame(rows)


def leakage_sensitivity(df: pd.DataFrame, curve: pd.DataFrame, limits=LEAKAGE_LIMITS) -> pd.DataFrame:
    """Fast-track share bought by each leakage limit, to locate the knee."""
    fraud = df[LABEL] == 1
    fraud_usd = df.loc[fraud, AMOUNT].sum()
    rows = []
    for limit in limits:
        ft = df["score"] < fast_track_cutoff(curve, limit)
        rows.append({
            "max_fraud_leaked": limit, "cutoff": round(fast_track_cutoff(curve, limit), 4),
            "share_fast_tracked": round(ft.mean(), 3),
            "share_fraud_leaked": round((ft & fraud).sum() / fraud.sum(), 3),
            "share_fraud_usd_leaked": round(df.loc[ft & fraud, AMOUNT].sum() / fraud_usd, 3),
        })
    return pd.DataFrame(rows)


def capacity_sensitivity(df: pd.DataFrame, capacities=CAPACITIES) -> pd.DataFrame:
    """Fraud caught and hit rate for each investigator capacity."""
    total_fraud = df[LABEL].sum()
    rows = []
    for cap in capacities:
        inv = df["score"] >= investigate_cutoff(df["score"], cap)
        rows.append({
            "capacity": cap, "cutoff": round(investigate_cutoff(df["score"], cap), 4),
            "share_fraud_caught": round(df.loc[inv, LABEL].sum() / total_fraud, 3),
            "fraud_rate_investigated": round(df.loc[inv, LABEL].mean(), 3),
        })
    return pd.DataFrame(rows)


def make_tier(fast_track_below: float, investigate_from: float):
    """Same rule as demo_server.tier(), with the chosen numbers baked in."""
    def tier(score: float) -> str:
        if score >= investigate_from:
            return "investigate"
        return "review" if score >= fast_track_below else "fast_track"
    return tier


def tier_summary(df: pd.DataFrame, tier) -> pd.DataFrame:
    t = df["score"].map(tier)
    fraud = df[LABEL] == 1
    return pd.DataFrame({
        "share_of_claims": t.value_counts(normalize=True),
        "share_of_fraud": t[fraud].value_counts() / fraud.sum(),
        "fraud_rate_in_tier": df.groupby(t)[LABEL].mean(),
    }).reindex(["fast_track", "review", "investigate"]).fillna(0).round(3)


def plot_triage_curve(curve: pd.DataFrame, a: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    point = curve[curve["cutoff"] == a].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve["share_fast_tracked"], curve["share_fraud_leaked"], label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random guessing")
    ax.axhline(MAX_FRAUD_LEAKED, ls=":", color="orange", label=f"{MAX_FRAUD_LEAKED:.0%} leakage limit")
    ax.scatter([point["share_fast_tracked"]], [point["share_fraud_leaked"]], color="purple", zorder=3,
               label=f"cut-off A = {a:.3f}")
    ax.set_xlabel("Share of claims fast-tracked")
    ax.set_ylabel("Share of fraud leaked into fast-track")
    ax.set_title("Triage curve (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def choose_thresholds(val: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Stage 1, validation only: cut-off A from the leakage knee, B from capacity."""
    curve = triage_curve(val["score"], val[LABEL])
    a = fast_track_cutoff(curve)
    b = investigate_cutoff(val["score"])
    if b <= a:
        raise ValueError(f"Investigate cut-off {b:.4f} is not above fast-track cut-off {a:.4f}")
    return a, b, curve


def calibrate(scores: pd.DataFrame, val: pd.DataFrame):
    """Return (validation probabilities, calibrator for test).

    Preferred: fit Platt on training out-of-fold scores, so validation stays a
    clean check. Fallback (no train rows): cross-fit within validation, and fit
    on all of validation for test.
    """
    train = scores.query("split == 'train'")
    if len(train):
        platt = fit_platt(train["score"], train[LABEL])
        log.info("Calibration: Platt fitted on %d training out-of-fold scores", len(train))
        return pd.Series(apply_platt(platt, val["score"]), index=val.index), platt
    log.info("Calibration: no train rows; cross-fitting Platt within validation")
    return cross_fitted_probability(val["score"], val[LABEL]), fit_platt(val["score"], val[LABEL])


def validation_tables(val: pd.DataFrame, curve: pd.DataFrame, val_prob: pd.Series) -> dict:
    """Evidence for each choice: sensitivity, calibration, and ranking comparison."""
    return {
        "leakage_sensitivity": leakage_sensitivity(val, curve),
        "capacity_sensitivity": capacity_sensitivity(val),
        "cost_sensitivity": cost_sensitivity(val),
        "reliability": reliability_table(val_prob, val[LABEL]),
        "ranking_comparison": pd.concat(
            [ranking_comparison(val, val_prob, cap).assign(capacity=cap) for cap in CAPACITIES]),
    }


def score_test(test: pd.DataFrame, tier, platt) -> pd.DataFrame:
    """Stage 2: apply the frozen thresholds and the frozen calibrator to test."""
    prob = apply_platt(platt, test["score"])
    out = test.assign(tier=test["score"].map(tier), probability=prob.round(4),
                      expected_loss_usd=(prob * test[AMOUNT]).round(0))
    # Investigators work the queue in expected-loss order: 1 = look at first.
    queue = out.loc[out["tier"] == "investigate", "expected_loss_usd"].rank(ascending=False, method="first")
    return out.assign(queue_rank=queue.astype("Int64"))


def write_outputs(tables: dict, curve: pd.DataFrame, a: float, b: float, test_scored: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, table in tables.items():
        keep_index = name in {"fairness", "headline"}
        log.info("%s:\n%s", name, table.to_string(index=keep_index))
        table.to_csv(OUT_DIR / f"{name}.csv", index=keep_index)
    plot_triage_curve(curve, a, OUT_DIR / "triage_curve.png")
    # Same keys as models/lr.json["tiers"], so Damien can paste it in.
    (OUT_DIR / "tiers.json").write_text(json.dumps({"review": round(a, 4), "investigate": round(b, 4)}, indent=2))
    cols = ["claim_id", LABEL, AMOUNT, "score", "probability", "tier", "expected_loss_usd", "queue_rank"]
    test_scored[cols].to_csv(OUT_DIR / "test_tiers.csv", index=False)


def main(scores_path: Path = SCORES, val_only: bool = False) -> None:
    scores = pd.read_csv(scores_path)
    val = scores.query("split == 'val'").reset_index(drop=True)

    a, b, curve = choose_thresholds(val)
    tier = make_tier(a, b)
    log.info("A (fast_track below) = %.4f, B (investigate from) = %.4f", a, b)
    log.info("Validation tiers:\n%s", tier_summary(val, tier))
    val_prob, platt = calibrate(scores, val)
    tables = validation_tables(val, curve, val_prob)
    if val_only:
        for name, table in tables.items():
            log.info("%s:\n%s", name, table.to_string(index=False))
        log.info("--val-only: test not evaluated, nothing written")
        return

    attributes = pd.read_csv(CLEAN_DATA, usecols=["claim_id", *SENSITIVE_COLUMNS])
    test = scores.query("split == 'test'").merge(attributes, on="claim_id", validate="1:1")
    test_scored = score_test(test, tier, platt)
    tables["fairness"] = fairness_report(test_scored["tier"], test_scored[LABEL], test_scored)
    tables["headline"] = bootstrap(test_scored)
    write_outputs(tables, curve, a, b, test_scored)
    log.info("HEADLINE (test): %s", headline_sentence(tables["headline"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = [a for a in sys.argv[1:] if a != "--val-only"]
    main(*[Path(p) for p in args[:1]], val_only="--val-only" in sys.argv)
