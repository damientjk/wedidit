"""Clean fraud_oracle.csv, add the (synth) claim fields, freeze the split.

Usage: python prepare_data.py [raw_csv] [out_csv]
Output keeps raw string categories (the API receives the same strings);
encoding is done downstream with a fixed lookup table, not fitted here.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from synth_amount import synthesise

log = logging.getLogger("prepare_data")

RAW_PATH = Path("data/raw/fraud_oracle.csv")
OUT_PATH = Path("data/processed/claims_clean.csv")
LABEL = "FraudFound_P"
SPLIT_SEED = 42
SPLIT_SHARES = {"train": 0.6, "val": 0.2, "test": 0.2}

MAKE_SPELLING = {"Accura": "Acura", "Nisson": "Nissan", "Porche": "Porsche", "Mecedes": "Mercedes"}
MIN_OBSERVED_AGE = 16
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def load(path: Path) -> pd.DataFrame:
    # utf-8-sig strips the BOM that otherwise corrupts the first column name.
    return pd.read_csv(path, encoding="utf-8-sig")


def drop_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["DayOfWeekClaimed"].isin(DAYS) & df["MonthClaimed"].isin(MONTHS)
    log.info("Dropping %d row(s) with an invalid claim date", (~valid).sum())
    return df.loc[valid]


def fix_age(df: pd.DataFrame) -> pd.DataFrame:
    # Age 0 occurs only in the "16 to 17" holder band, i.e. it is a missing code.
    zero = df["Age"].eq(0)
    log.info("Imputing Age=%d for %d rows with Age 0", MIN_OBSERVED_AGE, zero.sum())
    return df.assign(Age=df["Age"].mask(zero, MIN_OBSERVED_AGE), age_was_missing=zero.astype(int))


def fix_make(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Make=df["Make"].replace(MAKE_SPELLING))


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    # PolicyType = VehicleCategory + BasePolicy, except it mislabels 4,987 Sport
    # liability policies as Sedan. Keep the two clean columns, drop the composite.
    return df.drop(columns=["PolicyType"]).rename(columns={"PolicyNumber": "claim_id"})


def stratified_split(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Assign train/val/test within each (label, Year) stratum. No sklearn needed."""
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=df.index, name="split")
    names = list(SPLIT_SHARES)
    bounds = np.cumsum(list(SPLIT_SHARES.values()))[:-1]
    for _, idx in df.groupby([LABEL, "Year"]).groups.items():
        shuffled = rng.permutation(np.asarray(idx))
        cuts = np.round(bounds * len(shuffled)).astype(int)
        for name, part in zip(names, np.split(shuffled, cuts)):
            split.loc[part] = name
    return split


def validate(df: pd.DataFrame) -> None:
    if df.isna().any().any():
        raise ValueError("Missing values after cleaning")
    if (df["Age"] < MIN_OBSERVED_AGE).any():
        raise ValueError("Implausible Age after cleaning")
    if not df["claim_id"].is_unique:
        raise ValueError("Duplicate claim_id")
    if not set(df["split"]) == set(SPLIT_SHARES):
        raise ValueError("Every row must be in train/val/test")
    if (df["claim_amount_usd(synth)"] < 0).any():
        raise ValueError("Negative claim amount")


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = drop_redundant(fix_make(fix_age(drop_invalid_dates(raw))))
    df = df.join(synthesise(df.drop(columns=[LABEL])))
    df = df.assign(split=stratified_split(df)).reset_index(drop=True)
    validate(df)
    return df


def summarise(df: pd.DataFrame) -> None:
    log.info("Rows %d, fraud rate %.3f", len(df), df[LABEL].mean())
    log.info("Split:\n%s", df.groupby("split")[LABEL].agg(["count", "sum", "mean"]).round(4))
    amount = "claim_amount_usd(synth)"
    log.info("Amount (USD 1995) by coverage:\n%s",
             df.groupby("BasePolicy")[amount].describe(percentiles=[.5, .9, .99]).round(0))
    log.info("Amount by incident type:\n%s",
             df.groupby("incident_type(synth)")[amount].agg(["count", "median", "mean"]).round(0))
    log.info("Severity mix:\n%s", df["accident_severity(synth)"].value_counts(normalize=True).round(3))
    log.info("Amount by label (differences come only via features):\n%s",
             df.groupby(LABEL)[amount].agg(["mean", "median"]).round(0))


def write_splits(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write one CSV per split (train.csv, val.csv, test.csv) next to the full file."""
    paths = []
    for name in SPLIT_SHARES:
        path = out_dir / f"{name}.csv"
        df.loc[df["split"].eq(name)].to_csv(path, index=False)
        paths.append(path)
    return paths


def main(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> None:
    df = prepare(load(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    split_paths = write_splits(df, out_path.parent)
    summarise(df)
    log.info("Wrote %s and %s", out_path, ", ".join(map(str, split_paths)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = [Path(a) for a in sys.argv[1:3]]
    main(*args)
