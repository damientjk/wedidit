# Data

**Source:** Kaggle `shivamb` *fraud_oracle.csv*: US vehicle claims from 1994–96, 15,420 rows, label `FraudFound_P` (about 6%).
Put the raw file at `data/raw/fraud_oracle.csv`. It is gitignored until the Kaggle licence is confirmed to allow redistribution.

**Build:** `python prepare_data.py` writes these files to `data/processed/`:
- `claims_clean.csv`: all rows, with a `split` column.
- `train.csv`, `val.csv`, `test.csv`: one file per split, with the same columns.

Fit models only on `train.csv`. Set thresholds and calibration on `val.csv`. Touch `test.csv` once, for the final numbers.
**Test:** `python -m unittest discover tests`.

## Cleaning steps

| # | Problem found | Action | Why |
|---|---|---|---|
| 1 | The file starts with a UTF-8 BOM, which corrupts the first column name to `﻿Month`. | Read with `utf-8-sig`. | Otherwise `df["Month"]` fails. |
| 2 | One row (`PolicyNumber` 1517) has `DayOfWeekClaimed` = `MonthClaimed` = "0", `Age` = 0 and `Days_Policy_Claim` = "none". | Drop it. | This is a corrupt record and it is not fraud. Imputing four fields for one row adds nothing. |
| 3 | `Age` = 0 in the other 319 rows. All of them have `AgeOfPolicyHolder` "16 to 17" and are Single. Everywhere else, `Age` is consistent with its band. The bands are offset: "18 to 20" holds `Age` 16–17. | Set `Age` = 16, the minimum observed age, and add the flag `age_was_missing`. | 0 is a missing-value code, not an age. The flag keeps the signal: these rows have a 9.7% fraud rate against 6% overall. A model can use it without the age itself being invented. |
| 4 | Brands are misspelled: Accura, Nisson, Porche, Mecedes. | Correct the spellings. | These names show up in the UI and in the LLM case brief. |
| 5 | `PolicyType` is always `VehicleCategory` + `BasePolicy`, but it labels 4,987 Sport-car liability policies as "Sedan - Liability". | Drop `PolicyType`. | The column is redundant and internally inconsistent. The two clean source columns carry the same information. |
| 6 | `PolicyNumber` runs exactly 1…15,420 in file order. | Rename it to `claim_id`. | It is a row ID, not a feature. Keep it so the API, UI and evaluation can join on it. |
| 7 | Duplicates. | None found, even with the ID column ignored. | This was checked, so nothing needed removing. |
| 8 | Missing values. | None remain after steps 2–3; `validate()` asserts this. | |

**Checked and deliberately left alone:**
- **`RepNumber`:** fraud rate per rep ranges from 4.8% to 7.0%, with about 950 claims per rep, so it is not a shortcut. It is an arbitrary ID, so it should be excluded from features.
- **Rare categories:** `Deductible` 300 has n = 8, and some brands have n < 10. They are left as they are; grouping them is a modeling choice.
- **`Days_Policy_Accident` = "none":** about 55 rows, plausibly an accident on the policy's first day. Kept as a category.
- **No encoding here.** The output keeps the raw strings the API will receive. Ordinal mapping belongs in one shared fixed lookup that both `train.py` and `api.py` import, so it needs no fitting and nothing leaks from test data.

**Worth knowing before modeling:** `Fault` = Third Party claims are 0.9% fraud, against 7.9% when the policy holder is at fault. Liability-only policies are 0.8% fraud. Expect these to dominate the model's explanations.

## Split

The split is a stratified random 60/20/20, stratified on (`FraudFound_P`, `Year`) with seed 42. It is frozen in the `split` column, so everyone uses the same rows.

| split | rows | fraud | rate |
|---|---|---|---|
| train | 9,251 | 554 | 5.99% |
| val | 3,084 | 184 | 5.97% |
| test | 3,084 | 185 | 6.00% |

This was chosen over splitting by `Year` because 1996 has only 213 fraud cases and a lower fraud rate (5.2% against 6.7% in 1994). A model trained without 1996 also can't learn the `Year` level properly. The Year split is still worth running as a robustness check. With about 185 test fraud cases, report recall with a bootstrap CI.

## Synthetic columns: `incident_type(synth)`, `accident_severity(synth)`, `claim_amount_usd(synth)`

The source data has no incident type, accident severity or claim amount. `synth_amount.py` **fabricates** all three. Every synthetic column ends in `(synth)`.

They exist for:
- expected-loss ranking: P(fraud) × `claim_amount_usd(synth)`;
- a realistic demo (UI, LLM case brief).

**They are not real data and must not be model features.**

### Design rules
1. **The fraud label is never read.** `synthesise()` raises an error if `FraudFound_P` is passed. Fraud claims average more (US$4,047 against US$3,383) only because fraud concentrates in All Perils / policy-holder-at-fault claims, which cost more. Fraud amounts are deliberately not inflated, because that would build the expected-loss result into the data.
2. **They carry no new information.** They are a function of existing columns plus noise. Using them as features would add no real signal, and the model's explanations would cite patterns we invented.
3. **The three columns are coherent.** They are built in one chain: incident type → damage and injury → severity → amount paid.
4. **Money is in 1995 USD,** the same price level as `VehiclePrice`. To compare with S$ costs today, multiply by about 2.73 (US CPI 1995→now, about 2.1×, times about 1.30 SGD/USD). For example, the S$50–150 review cost in `risk.py` is about US$18–55 in these units.
5. **They are deterministic.** Seed 42 always reproduces them.

### `incident_type(synth)`
The incident type depends on coverage and fault:

| Situation | Possible incidents (base mix) |
|---|---|
| Other party at fault (any policy) | Collision - other vehicle 75%, Hit while parked 25% |
| Policy holder at fault, Liability | Collision - other vehicle 82%, Single-vehicle collision 18% |
| Policy holder at fault, Collision | Collision - other vehicle 70%, Single-vehicle 22%, Animal strike 8% |
| Policy holder at fault, All Perils | The collision types above (61% combined) plus Theft 9%, Vandalism 9%, Glass only 12%, Weather/flood 6%, Fire 3% |

The base mix is then adjusted by these modifiers:
- **Rural:** animal strikes ×3, single-vehicle ×1.5, theft and vandalism ×0.6.
- **Police report filed:** theft ×4. Without a report, theft ×0.3 (theft claims normally need one).
- **Witness present:** two-vehicle collision ×2.

Theft, fire, vandalism, weather and glass occur only under All Perils, and never when a third party is at fault.

**Result:**

| Incident | Claims | Median paid |
|---|---|---|
| Collision - other vehicle | 10,753 | US$1,689 |
| Single-vehicle collision | 2,150 | US$1,872 |
| Hit while parked | 1,035 | US$604 |
| Animal strike | 522 | US$1,120 |
| Glass only | 335 | US$303 |
| Vandalism | 255 | US$359 |
| Weather/flood | 182 | US$2,528 |
| Theft | 94 | US$3,064 |
| Fire | 93 | US$8,877 |

### `accident_severity(synth)`
Severity measures the size of the accident, not the payout. A severe accident can have a small claim when the other party's insurer pays.

| Severity | Rule (first match wins) | Share | Median paid |
|---|---|---|---|
| Total loss | Vehicle written off | 8.3% | US$9,324 |
| Major | Injury ≥ US$10k, damage ≥ 50% of vehicle value, or total loss ≥ US$10k | 10.4% | US$5,030 |
| Moderate | Any injury, or total loss ≥ US$2.5k | 52.6% | US$2,044 |
| Minor | Everything else | 28.7% | US$580 |

"Total loss" in the rules means own damage plus third-party property damage.

### `claim_amount_usd(synth)`
This is the amount the insurer pays, after the deductible:
- **Vehicle value:** the `VehiclePrice` band midpoint × depreciation by `AgeOfVehicle` (new 0.95 → more than 7 years 0.33).
- **Collision damage:**
  - Lognormal, with a median of US$1,600 at a US$20k car, scaled by value^0.4. σ = 0.8, giving a mean of about US$2.2k, in line with mid-90s US collision severity.
  - Scaled by incident: single-vehicle ×1.15, animal ×0.9, parked ×0.75.
  - Multiplied by: Rural ×1.25, police report ×1.5, supplements ×1.0–1.4 (supplements are extra damage found after the first estimate).
  - Total-loss chance: 4% base, ×1.5 for cars 7+ years old. Any repair above 75% of the car's value is also written off, paying 90–100% of its value.
- **Other incidents:**
  - Theft: 55% recovered (repair, median US$1,800), otherwise total loss.
  - Fire: 65% total loss.
  - Weather: median US$2,200, 12% total loss.
  - Vandalism: median US$700.
  - Glass: median US$300, with the deductible waived.
- **Third-party liability** (Liability and All Perils, policy holder at fault):
  - Property damage: median US$1,500 for the other car, US$1,200 for a fence or pole.
  - Bodily injury: chance is 22% for a two-car collision, 8% single-vehicle, 5% animal (IRC 1995: about 29.5 injury claims per 100 property-damage claims). Lognormal, median US$6k, σ = 1.1.
  - Capped at a US$100k limit.
- **Liability policy, other party at fault:** towing, rental and handling only (median US$350), because the other party's insurer pays the rest.
- **Coverage decides what is paid:** Liability pays only third parties, Collision pays only own damage, All Perils pays both. 3.1% of claims pay US$0 because the damage was below the deductible.

**Result:**

| | Median | Mean | p99 |
|---|---|---|---|
| All claims | US$1,543 | US$3,423 | US$31,100 |
| All Perils | US$1,973 | US$4,210 | US$33,892 |
| Collision | US$1,366 | US$2,730 | US$19,345 |
| Liability | US$1,482 | US$3,550 | US$34,011 |

### Limitations to state openly
- **Expected-loss results are illustrative.** They show how the triage *policy* behaves, not real savings.
- **Every parameter is an assumption.** Before quoting a dollar-weighted result, rerun it with another seed or with the parameters moved ±25%.
- **Don't report fraud rates by `incident_type(synth)` or `accident_severity(synth)`.** Glass-only claims show about 19% fraud and theft about 17%. That isn't a finding: those incidents occur only in the All Perils, policy-holder-at-fault group, which already has a high fraud rate.
