# Missing fields in the claim form

Owner: Da Feng. Measured 17 Oct 2026 on LinuxONE (s390x) with the served model `xgboost_09`, on 600 validation claims (`data/processed/val.csv`, random_state 0). Script: `docs/missing_fields_analysis.py`.

## What the model does today with a blank field

- **Model:** `xgboost_09`, an XGBoost classifier on 19 Kaggle claim fields (Marcus, `src/explain.py`).
- **Blank field:** the pipeline's `SimpleImputer`, fitted on the training set only, fills in the median (Deductible) or the most common value (all other fields).
- **Flag:** the response sets `requires_data_quality_check` and `operational_routing_allowed: false` and lists `missing_features`. The UI marks those fields `typ.` and labels the score *provisional*.
- With **every** field blank the score is 0.22, which already falls in *review*. That's why the form shows no score until at least one field is filled in.

## How much each blank field matters

The columns below are:
- **Tier flips:** the share of claims whose tier (fast track / review / investigate) changes when only this field is left blank.
- **Range crosses a tier:** the share of claims where trying every possible value of the field moves the score across a tier boundary.

| Field | Mean abs TreeSHAP (log-odds) | Mean score change if blank | Tier flips if blank | Range crosses a tier |
|---|---|---|---|---|
| BasePolicy | 0.872 | 0.090 | **36.7%** | **98.7%** |
| VehiclePrice | 0.086 | 0.013 | 6.8% | 24.2% |
| MonthClaimed | 0.108 | 0.011 | 6.2% | 47.8% |
| AgeOfVehicle | 0.054 | 0.007 | 3.7% | 41.0% |
| WeekOfMonth | 0.027 | 0.004 | 2.5% | 8.8% |
| DayOfWeek | 0.038 | 0.006 | 2.3% | 8.2% |
| Make | 0.036 | 0.004 | 2.2% | 41.7% |
| VehicleCategory | 0.049 | 0.002 | 2.0% | 21.7% |
| Deductible | 0.027 | 0.002 | 1.8% | 47.3% |
| WeekOfMonthClaimed | 0.020 | 0.006 | 1.7% | 3.2% |
| PastNumberOfClaims | 0.020 | 0.002 | 1.5% | 4.0% |
| AccidentArea | 0.030 | 0.003 | 1.3% | 16.5% |
| DriverRating | 0.005 | 0.001 | 0.2% | 0.7% |
| Days_Policy_Accident | 0.010 | 0.000 | 0.2% | **51.2%** |
| Month | 0.002 | 0.000 | 0.0% | 0.0% |
| NumberOfCars | 0.002 | 0.000 | 0.0% | 6.0% |
| DayOfWeekClaimed | 0.000 | 0.000 | 0.0% | 0.0% |
| AgentType | 0.000 | 0.000 | 0.0% | 0.0% |
| Days_Policy_Claim | 0.000 | 0.000 | 0.0% | 0.0% |

How to read it:
- **BasePolicy decides the score.** If it's left blank, more than a third of claims land in the wrong tier.
- **Some fields look harmless on average but aren't.** Days_Policy_Accident, Deductible and Make rarely flip the tier when blank, because most claims have the typical value. But when a claim has an unusual value (for example a policy only 1 to 7 days old), the score can cross a tier. So a blank there is only safe if the claim really is typical.
- **Five fields have no effect on this model:** Month, DayOfWeekClaimed, AgentType, Days_Policy_Claim and NumberOfCars. Leaving them blank changes nothing.

## Options for handling blanks

| # | Approach | Pros | Cons | Verdict |
|---|---|---|---|---|
| 1 | **Typical value + flag** (current) | Simple, already in the model, honest "provisional" label | Silently assumes the claim is typical | Keep as the fallback |
| 2 | **Required core fields** | Stops the one field that matters most from being skipped | Extra friction for the agent | **Do it**: make BasePolicy required |
| 3 | **Score range for blanks** (best / worst case over every possible value) | Honest, cheap (at most 19 values per field), tells the agent *which* blank to chase | More API calls; needs a small UI element | **Recommended** for the "range crosses a tier" fields |
| 4 | **Auto-fill from the policy record** (policy type, deductible, vehicle make/price/age, agent, number of cars) and **derive dates** (months, weeks, weekdays from the accident and claim dates) | How insurers actually do it; the agent only types incident facts | Needs a policy system; out of scope for the demo | Say it in the video as the production design |
| 5 | **Predictive imputation** (KNN / IterativeImputer / one small model per field, trained on train.csv only) | Uses correlations (Make with VehiclePrice and VehicleCategory, Month with MonthClaimed) | Hides uncertainty, needs its own validation, and must never predict BasePolicy | Only for weak, correlated fields, and as future work |
| 6 | **Expected score over plausible values** (weight each value by how often it appears for similar claims) | Combines 3 and 5 into one number plus a spread | More complex to explain | Future work |
| 7 | **Retrain with missingness**: drop the imputer and let XGBoost learn a default branch for missing values, training with randomly blanked fields | The model itself learns what "unknown" means | Needs Marcus to retrain and re-validate | Future work |
| 8 | **Leave blank** | Zero friction | Fine only for the five zero-effect fields | OK for those five |

## Recommended field groups for the form

- **Required (no score until filled):** BasePolicy
- **Strongly recommended (show the score range while blank):** VehiclePrice, MonthClaimed, AgeOfVehicle, Days_Policy_Accident, Deductible, Make
- **Optional:** WeekOfMonth, DayOfWeek, VehicleCategory, WeekOfMonthClaimed, PastNumberOfClaims, AccidentArea, DriverRating
- **No effect on this model (can be collapsed):** Month, DayOfWeekClaimed, AgentType, Days_Policy_Claim, NumberOfCars

Note: the "no effect" fields are a finding about `xgboost_09`, not about fraud. A retrained model could use them.
