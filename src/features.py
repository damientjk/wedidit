"""
src/features.py (owner: Ethan)

Shared feature engineering. Imported by BOTH src/train.py and the API, so a
claim is turned into numbers the same way in training and at scoring time.

Design rules (from data/README.md):
- Input is the raw claim with the original string values.
- Pure functions, no file I/O, nothing is fitted, so nothing leaks from test data.
- Fixed lookups only. Unknown or missing values become NaN (XGBoost handles NaN).
- Never uses the label (FraudFound_P), claim_id, RepNumber, split, Year, or any
  column ending in "(synth)".
- Works on a full DataFrame or a one-row DataFrame (a single API request).

Usage:
    from features import build_features, FEATURE_NAMES
    X = build_features(df)          # df has the raw claim columns
    model.fit(X[FEATURE_NAMES], y)
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns that must never become features
# ---------------------------------------------------------------------------
FORBIDDEN = {
    "FraudFound_P",  # label
    "claim_id",      # row id
    "RepNumber",     # arbitrary id
    "split",         # bookkeeping
    "Year",          # 1994-96 only, a new claim would not be in range
    "PolicyType",    # dropped upstream (redundant)
}
SYNTH_SUFFIX = "(synth)"  # synthetic columns, never features

# Raw columns the API request is expected to contain
RAW_COLUMNS = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea",
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Sex",
    "MaritalStatus", "Age", "Fault", "VehicleCategory", "VehiclePrice",
    "Deductible", "DriverRating", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder",
    "PoliceReportFiled", "WitnessPresent", "AgentType", "NumberOfSuppliments",
    "AddressChange_Claim", "NumberOfCars", "BasePolicy", "age_was_missing",
]

# ---------------------------------------------------------------------------
# Fixed lookups (ordinal bands -> numbers)
# ---------------------------------------------------------------------------
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

PRICE_MID = {  # US$ band midpoints (1995 USD)
    "less than 20000": 15000, "20000 to 29000": 25000, "30000 to 39000": 35000,
    "40000 to 59000": 50000, "60000 to 69000": 65000, "more than 69000": 80000,
}
DAYS_POLICY_ACCIDENT = {  # days from policy start to accident (band midpoint)
    "none": 0, "1 to 7": 4, "8 to 15": 11, "15 to 30": 22, "more than 30": 45,
}
DAYS_POLICY_CLAIM = {
    "none": 0, "8 to 15": 11, "15 to 30": 22, "more than 30": 45,
}
PAST_CLAIMS = {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 5}
VEHICLE_AGE_YEARS = {
    "new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
    "6 years": 6, "7 years": 7, "more than 7": 9,
}
HOLDER_AGE_MID = {
    "16 to 17": 16.5, "18 to 20": 19, "21 to 25": 23, "26 to 30": 28,
    "31 to 35": 33, "36 to 40": 38, "41 to 50": 45.5, "51 to 65": 58, "over 65": 70,
}
SUPPLEMENTS = {"none": 0, "1 to 2": 1, "3 to 5": 4, "more than 5": 6}
ADDRESS_CHANGE = {  # recency: higher = more recent change (more suspicious)
    "no change": 0, "4 to 8 years": 1, "2 to 3 years": 2, "1 year": 3, "under 6 months": 4,
}
NUM_CARS = {"1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3.5, "5 to 8": 6.5, "more than 8": 9}
DAYS_OF_WEEK = {d: i + 1 for i, d in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}

# Nominal columns: fixed one-hot lists (no fitting needed)
ONEHOT = {
    "MaritalStatus": ["Single", "Married", "Widow", "Divorced"],
    "VehicleCategory": ["Sport", "Utility", "Sedan"],
    "BasePolicy": ["Liability", "Collision", "All Perils"],
    "Make": ["Acura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda",
             "Jaguar", "Lexus", "Mazda", "Mercedes", "Mercury", "Nissan",
             "Pontiac", "Porsche", "Saab", "Saturn", "Toyota", "VW"],
}
# Both spellings accepted (data/README.md says the brands were corrected upstream)
MAKE_ALIASES = {"Accura": "Acura", "Nisson": "Nissan", "Porche": "Porsche", "Mecedes": "Mercedes"}

YES_NO = {"Yes": 1, "No": 0}


def _map(series, mapping):
    """Map with a fixed dict, unknown or missing becomes NaN."""
    return series.map(mapping).astype(float)


def _col(df, name):
    """Return the column, or an all-NaN column if the request left it out."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype="object")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw claims (one or many rows) into a numeric feature table."""
    df = df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)

    # --- Policy and history -------------------------------------------------
    out["fault_policy_holder"] = _map(_col(df, "Fault"), {"Policy Holder": 1, "Third Party": 0})
    out["past_claims"] = _map(_col(df, "PastNumberOfClaims"), PAST_CLAIMS)
    out["days_policy_to_incident"] = _map(_col(df, "Days_Policy_Accident"), DAYS_POLICY_ACCIDENT)
    out["days_policy_to_claim"] = _map(_col(df, "Days_Policy_Claim"), DAYS_POLICY_CLAIM)
    out["address_change_recency"] = _map(_col(df, "AddressChange_Claim"), ADDRESS_CHANGE)
    out["deductible"] = pd.to_numeric(_col(df, "Deductible"), errors="coerce")
    out["driver_rating"] = pd.to_numeric(_col(df, "DriverRating"), errors="coerce")

    # --- Vehicle ------------------------------------------------------------
    out["vehicle_price"] = _map(_col(df, "VehiclePrice"), PRICE_MID)
    out["vehicle_age_years"] = _map(_col(df, "AgeOfVehicle"), VEHICLE_AGE_YEARS)

    # --- Claimant -----------------------------------------------------------
    out["holder_age_mid"] = _map(_col(df, "AgeOfPolicyHolder"), HOLDER_AGE_MID)
    out["age"] = pd.to_numeric(_col(df, "Age"), errors="coerce")
    out["age_was_missing"] = pd.to_numeric(_col(df, "age_was_missing"), errors="coerce")
    out["is_male"] = _map(_col(df, "Sex"), {"Male": 1, "Female": 0})
    out["agent_external"] = _map(_col(df, "AgentType"), {"External": 1, "Internal": 0})

    # --- Incident details ---------------------------------------------------
    out["accident_area_rural"] = _map(_col(df, "AccidentArea"), {"Rural": 1, "Urban": 0})
    out["police_report_filed"] = _map(_col(df, "PoliceReportFiled"), YES_NO)
    out["witness_present"] = _map(_col(df, "WitnessPresent"), YES_NO)
    out["supplements"] = _map(_col(df, "NumberOfSuppliments"), SUPPLEMENTS)
    out["num_cars"] = _map(_col(df, "NumberOfCars"), NUM_CARS)
    out["week_of_month"] = pd.to_numeric(_col(df, "WeekOfMonth"), errors="coerce")
    out["accident_day_of_week"] = _map(_col(df, "DayOfWeek"), DAYS_OF_WEEK)

    # --- Time to report: months between accident and claim (wraps at year end)
    m_acc = _map(_col(df, "Month"), MONTHS)
    m_clm = _map(_col(df, "MonthClaimed"), MONTHS)
    out["months_to_claim"] = (m_clm - m_acc) % 12

    # --- Interaction the data README flags as dominant ----------------------
    allperils = (_col(df, "BasePolicy") == "All Perils").astype(float)
    out["holder_fault_allperils"] = out["fault_policy_holder"] * allperils

    # --- One-hot for nominal columns (fixed lists) --------------------------
    for col, values in ONEHOT.items():
        s = _col(df, col)
        if col == "Make":
            s = s.replace(MAKE_ALIASES)
        for v in values:
            name = f"{col}_{v}".replace(" ", "_")
            out[name] = (s == v).astype(float)

    return out


# Stable column order for training and serving
FEATURE_NAMES = list(build_features(pd.DataFrame({"Fault": ["Policy Holder"]})).columns)


def check_columns(df: pd.DataFrame):
    """Return raw columns the claim is missing (call at load time, not silently)."""
    return [c for c in RAW_COLUMNS if c not in df.columns]


def assert_no_forbidden(feature_names):
    """Safety net: raise if a forbidden or synthetic column leaked into features."""
    bad = [f for f in feature_names if f in FORBIDDEN or f.endswith(SYNTH_SUFFIX)]
    if bad:
        raise ValueError(f"Forbidden columns in features: {bad}")


assert_no_forbidden(FEATURE_NAMES)
