"""Synthetic claim fields for fraud_oracle.csv.

The source data has no incident type, accident severity or claim amount. This
module fabricates all three, coherently, from each claim's own attributes
(coverage, fault, vehicle value, area, police report, supplements):

    incident type -> damage, injury, third-party loss -> severity -> paid amount

Rules:
- The fraud label is never read, so these columns cannot leak it.
- None of them is a model feature: they are derived from features the model
  already sees plus noise, and exist for expected-loss ranking and the demo.
- Money is in 1995 USD, the price level of VehiclePrice.
"""
import numpy as np
import pandas as pd

SEED = 42
LABEL = "FraudFound_P"

COLLISION_VEHICLE = "Collision - other vehicle"
SINGLE_VEHICLE = "Single-vehicle collision"
ANIMAL = "Animal strike"
PARKED = "Hit while parked"
THEFT = "Theft"
VANDALISM = "Vandalism"
FIRE = "Fire"
WEATHER = "Weather/flood"
GLASS = "Glass only"
INCIDENTS = [COLLISION_VEHICLE, SINGLE_VEHICLE, ANIMAL, PARKED, THEFT, VANDALISM, FIRE, WEATHER, GLASS]
COLLISIONS = [COLLISION_VEHICLE, SINGLE_VEHICLE, ANIMAL, PARKED]

# Base incident mix by (who is at fault, coverage). Comprehensive events
# (theft, fire, ...) only exist under All Perils and have no third party.
MIX_OTHER_PARTY_AT_FAULT = {COLLISION_VEHICLE: 0.75, PARKED: 0.25}
MIX_LIABILITY = {COLLISION_VEHICLE: 0.82, SINGLE_VEHICLE: 0.18}
MIX_COLLISION = {COLLISION_VEHICLE: 0.70, SINGLE_VEHICLE: 0.22, ANIMAL: 0.08}
MIX_ALL_PERILS = {
    COLLISION_VEHICLE: 0.42, SINGLE_VEHICLE: 0.14, ANIMAL: 0.05, THEFT: 0.09,
    VANDALISM: 0.09, FIRE: 0.03, WEATHER: 0.06, GLASS: 0.12,
}
RURAL_TILT = {ANIMAL: 3.0, SINGLE_VEHICLE: 1.5, THEFT: 0.6, VANDALISM: 0.6}
POLICE_YES_TILT = {THEFT: 4.0, COLLISION_VEHICLE: 1.3, VANDALISM: 1.5, GLASS: 0.2}
POLICE_NO_TILT = {THEFT: 0.3}
WITNESS_TILT = {COLLISION_VEHICLE: 2.0, THEFT: 0.3}

VEHICLE_PRICE_MID = {
    "less than 20000": 15_000, "20000 to 29000": 24_500, "30000 to 39000": 34_500,
    "40000 to 59000": 49_500, "60000 to 69000": 64_500, "more than 69000": 80_000,
}
DEPRECIATION = {
    "new": 0.95, "2 years": 0.80, "3 years": 0.70, "4 years": 0.62,
    "5 years": 0.55, "6 years": 0.48, "7 years": 0.42, "more than 7": 0.33,
}
SUPPLEMENT_FACTOR = {"none": 1.00, "1 to 2": 1.12, "3 to 5": 1.25, "more than 5": 1.40}

# Collision repair: lognormal median at a US$20k car, scaled by value^0.4;
# mean ~US$2.2k, near mid-90s US collision severity.
REPAIR_MEDIAN_USD = 1_600
REPAIR_VALUE_ELASTICITY = 0.4
REPAIR_SIGMA = 0.8
COLLISION_TYPE_FACTOR = {COLLISION_VEHICLE: 1.0, SINGLE_VEHICLE: 1.15, ANIMAL: 0.9, PARKED: 0.75}
TOTAL_LOSS_BASE_P = 0.04
TOTAL_LOSS_SHARE = 0.75  # repair above 75% of vehicle value is written off

THEFT_RECOVERED_P = 0.55
THEFT_RECOVERED_MEDIAN_USD = 1_800
VANDALISM_MEDIAN_USD = 700
FIRE_TOTAL_P = 0.65
FIRE_PARTIAL_MEDIAN_USD = 3_000
WEATHER_MEDIAN_USD = 2_200
WEATHER_TOTAL_P = 0.12
GLASS_MEDIAN_USD = 300

TP_PROPERTY_MEDIAN_USD = {COLLISION_VEHICLE: 1_500, SINGLE_VEHICLE: 1_200}  # other car / fence, pole
INJURY_P = {COLLISION_VEHICLE: 0.22, SINGLE_VEHICLE: 0.08, ANIMAL: 0.05}  # IRC 1995: ~29.5 BI per 100 PD
INJURY_MEDIAN_USD = 6_000
INJURY_SIGMA = 1.1
LIABILITY_LIMIT_USD = 100_000
RECOVERY_MEDIAN_USD = 350  # towing/rental before recovery from the at-fault party's insurer

MAJOR_DAMAGE_SHARE = 0.5
MAJOR_LOSS_USD = 10_000
MAJOR_INJURY_USD = 10_000  # minor injuries (e.g. whiplash) are common and cheap
MODERATE_LOSS_USD = 2_500


def _lognormal(rng, median, sigma, n):
    return rng.lognormal(np.log(median), sigma, n)


def vehicle_value_usd(df: pd.DataFrame) -> np.ndarray:
    price = df["VehiclePrice"].map(VEHICLE_PRICE_MID)
    keep = df["AgeOfVehicle"].map(DEPRECIATION)
    if price.isna().any() or keep.isna().any():
        raise ValueError("Unknown VehiclePrice or AgeOfVehicle category")
    return (price * keep).to_numpy()


def _mix_matrix(df: pd.DataFrame) -> np.ndarray:
    other_at_fault = df["Fault"].eq("Third Party").to_numpy()
    policy = df["BasePolicy"].to_numpy()
    weights = np.zeros((len(df), len(INCIDENTS)))
    for mix, mask in [
        (MIX_OTHER_PARTY_AT_FAULT, other_at_fault),
        (MIX_LIABILITY, ~other_at_fault & (policy == "Liability")),
        (MIX_COLLISION, ~other_at_fault & (policy == "Collision")),
        (MIX_ALL_PERILS, ~other_at_fault & (policy == "All Perils")),
    ]:
        for incident, w in mix.items():
            weights[mask, INCIDENTS.index(incident)] = w
    return weights


def _apply_tilt(weights: np.ndarray, mask: np.ndarray, tilt: dict) -> np.ndarray:
    factors = np.ones(len(INCIDENTS))
    for incident, f in tilt.items():
        factors[INCIDENTS.index(incident)] = f
    return np.where(mask[:, None], weights * factors, weights)


def draw_incident_type(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    police = df["PoliceReportFiled"].eq("Yes").to_numpy()
    weights = _mix_matrix(df)
    weights = _apply_tilt(weights, df["AccidentArea"].eq("Rural").to_numpy(), RURAL_TILT)
    weights = _apply_tilt(weights, police, POLICE_YES_TILT)
    weights = _apply_tilt(weights, ~police, POLICE_NO_TILT)
    weights = _apply_tilt(weights, df["WitnessPresent"].eq("Yes").to_numpy(), WITNESS_TILT)
    cum = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
    pick = (rng.random(len(df))[:, None] > cum).sum(axis=1)
    return np.array(INCIDENTS)[np.minimum(pick, len(INCIDENTS) - 1)]


def _crash_multiplier(df: pd.DataFrame) -> np.ndarray:
    rural = np.where(df["AccidentArea"].eq("Rural"), 1.25, 1.0)
    police = np.where(df["PoliceReportFiled"].eq("Yes"), 1.5, 1.0)
    return rural * police


def _own_damage(df, incident, acv, rng):
    """Gross damage to the insured vehicle (before deductible) and total-loss flag."""
    n = len(df)
    supplements = df["NumberOfSuppliments"].map(SUPPLEMENT_FACTOR).to_numpy()
    crash = _crash_multiplier(df) * supplements
    type_factor = pd.Series(incident).map(COLLISION_TYPE_FACTOR).fillna(1.0).to_numpy()
    repair = _lognormal(rng, REPAIR_MEDIAN_USD, REPAIR_SIGMA, n) * (acv / 20_000) ** REPAIR_VALUE_ELASTICITY
    collision = repair * crash * type_factor
    old = df["AgeOfVehicle"].isin(["7 years", "more than 7"]).to_numpy()
    p_total = np.clip(TOTAL_LOSS_BASE_P * np.where(old, 1.5, 1.0) * crash * type_factor, 0, 0.5)
    u = rng.random(n)

    is_collision = np.isin(incident, COLLISIONS)
    total = np.select(
        [is_collision, incident == THEFT, incident == FIRE, incident == WEATHER],
        [u < p_total, u > THEFT_RECOVERED_P, u < FIRE_TOTAL_P, u < WEATHER_TOTAL_P],
        False,
    )
    partial = np.select(
        [is_collision, incident == THEFT, incident == VANDALISM, incident == FIRE, incident == WEATHER],
        [collision,
         _lognormal(rng, THEFT_RECOVERED_MEDIAN_USD, 0.8, n) * supplements,
         _lognormal(rng, VANDALISM_MEDIAN_USD, 0.7, n) * supplements,
         _lognormal(rng, FIRE_PARTIAL_MEDIAN_USD, 0.8, n) * supplements,
         _lognormal(rng, WEATHER_MEDIAN_USD, 0.8, n) * supplements],
        _lognormal(rng, GLASS_MEDIAN_USD, 0.35, n),
    )
    total = total | ((incident != GLASS) & (partial > TOTAL_LOSS_SHARE * acv))
    damage = np.where(total, acv * rng.uniform(0.9, 1.0, n), np.minimum(partial, acv))
    return damage, total


def _third_party_loss(df, incident, rng):
    """Other people's property damage and bodily injury caused by the accident."""
    n = len(df)
    crash = _crash_multiplier(df)
    prop_median = pd.Series(incident).map(TP_PROPERTY_MEDIAN_USD).fillna(0).to_numpy()
    prop = np.where(prop_median > 0, _lognormal(rng, np.maximum(prop_median, 1), 0.7, n) * crash, 0.0)
    p_injury = np.clip(pd.Series(incident).map(INJURY_P).fillna(0).to_numpy() * crash, 0, 0.7)
    injury = np.where(rng.random(n) < p_injury, _lognormal(rng, INJURY_MEDIAN_USD, INJURY_SIGMA, n), 0.0)
    return prop, injury


def _severity(damage, total, acv, prop, injury) -> np.ndarray:
    loss = damage + prop
    return np.select(
        [total,
         (injury >= MAJOR_INJURY_USD) | (damage >= MAJOR_DAMAGE_SHARE * acv) | (loss >= MAJOR_LOSS_USD),
         (injury > 0) | (loss >= MODERATE_LOSS_USD)],
        ["Total loss", "Major", "Moderate"],
        "Minor",
    )


def _paid_usd(df, incident, damage, prop, injury, rng):
    policy = df["BasePolicy"].to_numpy()
    holder_at_fault = df["Fault"].eq("Policy Holder").to_numpy()
    deductible = np.where(incident == GLASS, 0, df["Deductible"].to_numpy())  # glass deductible waived
    own = np.maximum(damage - deductible, 0)
    liability = np.minimum(prop + injury, LIABILITY_LIMIT_USD)
    recovery = _lognormal(rng, RECOVERY_MEDIAN_USD, 0.6, len(df))
    return (
        np.where(np.isin(policy, ["Collision", "All Perils"]), own, 0.0)
        + np.where(np.isin(policy, ["Liability", "All Perils"]) & holder_at_fault, liability, 0.0)
        + np.where((policy == "Liability") & ~holder_at_fault, recovery, 0.0)
    )


def synthesise(features: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return incident_type(synth), accident_severity(synth), claim_amount_usd(synth).

    `features` must not contain the fraud label; this is enforced.
    """
    if LABEL in features.columns:
        raise ValueError("Pass features only: the label must not influence synthetic fields")
    rng = np.random.default_rng(seed)
    acv = vehicle_value_usd(features)
    incident = draw_incident_type(features, rng)
    damage, total = _own_damage(features, incident, acv, rng)
    prop, injury = _third_party_loss(features, incident, rng)
    return pd.DataFrame(
        {
            "incident_type(synth)": incident,
            "accident_severity(synth)": _severity(damage, total, acv, prop, injury),
            "claim_amount_usd(synth)": np.round(_paid_usd(features, incident, damage, prop, injury, rng)),
        },
        index=features.index,
    )
