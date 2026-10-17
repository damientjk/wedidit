"""Fairness check: does the triage treat groups differently?

For each group of a sensitive attribute we report, side by side:
- base_fraud_rate: how much fraud the group actually has;
- flag_rate: share of the group's claims sent to a human (review or investigate);
- investigate_rate: share sent to investigators;
- recall: share of the group's fraud that reaches a human.

A higher flag rate is only a concern if the group's base fraud rate doesn't
explain it. Small groups are marked: their rates are too noisy to judge.
"""
import pandas as pd

LABEL = "FraudFound_P"
MIN_GROUP_SIZE = 30
MIN_FRAUD_FOR_RECALL = 10

# 9 holder-age bands -> 4, so groups are big enough to read.
AGE_GROUPS = {
    "16 to 17": "under 26", "18 to 20": "under 26", "21 to 25": "under 26",
    "26 to 30": "26 to 35", "31 to 35": "26 to 35",
    "36 to 40": "36 to 50", "41 to 50": "36 to 50",
    "51 to 65": "over 50", "over 65": "over 50",
}
ATTRIBUTES = {
    "Sex": lambda df: df["Sex"],
    "MaritalStatus": lambda df: df["MaritalStatus"],
    "AgeGroup": lambda df: df["AgeOfPolicyHolder"].map(AGE_GROUPS),
}


def group_rates(tiers: pd.Series, labels: pd.Series, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "group": groups, "fraud": labels,
        "flagged": tiers.ne("fast_track"), "investigated": tiers.eq("investigate"),
    })
    fraud = df[df["fraud"] == 1]
    table = df.groupby("group").agg(
        claims=("fraud", "size"), fraud_cases=("fraud", "sum"), base_fraud_rate=("fraud", "mean"),
        flag_rate=("flagged", "mean"), investigate_rate=("investigated", "mean"),
    )
    table["recall"] = fraud.groupby("group")["flagged"].mean()
    table.loc[table["fraud_cases"] < MIN_FRAUD_FOR_RECALL, "recall"] = float("nan")
    table["note"] = ""
    table.loc[table["claims"] < MIN_GROUP_SIZE, "note"] = "too small to judge"
    return table.round(3)


def fairness_report(tiers: pd.Series, labels: pd.Series, attributes_df: pd.DataFrame) -> pd.DataFrame:
    parts = [group_rates(tiers, labels, get(attributes_df)).assign(attribute=name)
             for name, get in ATTRIBUTES.items()]
    return pd.concat(parts).reset_index().set_index(["attribute", "group"])
