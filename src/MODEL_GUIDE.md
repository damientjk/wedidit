# Vehicle insurance fraud models: development and team handoff

This guide describes Marcus's dataset audit, model development, current
risk-prioritization deployment candidate and separate probability-estimation
experiments for the IBM Z Datathon 2026 project.

**Current deployment candidate:** the locked tuned, uncalibrated XGBoost in
`models/experiments/capacity_20261017T042040511095Z/xgboost_09.joblib`.
It produces a **relative fraud-risk score** to help a human prioritize claims.
`src/explain.py` loads this model, not the calibrated experiment. Calling the
score a verified fraud probability would be incorrect. Deployment candidate
does not mean it has been deployed or verified on LinuxONE.

**Separate probability-estimation experiment:** isotonic-calibrated XGBoost.
It was selected using training-only nested cross-validation. Logistic
regression and a logistic/XGBoost combination were also evaluated. It is saved
separately and has **not replaced the deployment candidate or been connected
to the existing UI**. Its probability losses improve, but its ranking metrics
decline, so it must not be silently substituted for the tuned risk scorer.

Neither model confirms fraud, approves/rejects claims or automatically clears
them. Human-review budgets used in experiments were evaluation tools, not
mandatory product routing rules.

## 1. Dataset and target

Yu Heng's [data README](../data/README.md) is the primary reference for source,
cleaning and frozen splits. The source is Kaggle `shivamb`'s
`fraud_oracle.csv`, described there as vehicle insurance claims from 1994-1996.

The raw source has 15,420 rows. The cleaned dataset has **15,419 rows and 37
columns**, after removing one corrupt record. The raw CSV is not present in
this checkout; `data/raw/` is gitignored pending licence confirmation. The
actual available model inputs are the existing processed files:

| File | Rows | Non-fraud | Fraud | Fraud rate |
|---|---:|---:|---:|---:|
| `data/processed/claims_clean.csv` | 15,419 | 14,496 | 923 | 5.99% |
| `data/processed/train.csv` | 9,251 | 8,697 | 554 | 5.99% |
| `data/processed/val.csv` | 3,084 | 2,900 | 184 | 5.97% |
| `data/processed/test.csv` | Frozen final holdout | Not used in modeling work | Not evaluated | — |

The target is **`FraudFound_P`**, with this exact capitalization:

- `0`: actual non-fraud label.
- `1`: actual fraud label.

The existing split is stratified random 60/20/20 on `(FraudFound_P, Year)`,
seed 42, frozen in the `split` column. Prior read-only dataset inspection
verified consistent schemas, no remaining nulls, unique claim IDs, disjoint
split IDs, and exact agreement between split files and the full cleaned CSV.
Training/validation use their existing files; the final test file has not been
used for fitting, tuning, calibration or performance evaluation. The initial
dataset audit inspected all CSVs to verify their split integrity.

Important cleaning decisions already made by Yu Heng include correcting make
spellings, dropping redundant/inconsistent `PolicyType`, renaming the sequential
`PolicyNumber` to `claim_id`, and fixing missing-age codes with an accompanying
flag. Age and its missingness flag are excluded from prediction here.

## 2. The agreed 19 prediction features

| Treatment | Exact feature names |
|---|---|
| Numeric | `Deductible` |
| Calendar categories | `Month`, `WeekOfMonth`, `DayOfWeek`, `MonthClaimed`, `WeekOfMonthClaimed`, `DayOfWeekClaimed` |
| Nominal categories | `Make`, `AccidentArea`, `VehicleCategory`, `BasePolicy`, `AgentType` |
| Ordered categories/bands, encoded categorically | `VehiclePrice`, `DriverRating`, `Days_Policy_Accident`, `Days_Policy_Claim`, `PastNumberOfClaims`, `AgeOfVehicle`, `NumberOfCars` |

Preserve source types: the week fields and `DriverRating` are numeric integer
categories; other categorical fields are source strings. Do not convert bands
to invented exact durations, ages, prices or counts. `PastNumberOfClaims` is
not documented as a count specifically over the last 12 months.

### Exclusions and leakage concerns

| Columns | Reason excluded |
|---|---|
| `FraudFound_P` | Target, never an input |
| `claim_id`, `RepNumber` | Identifiers, not eligible predictors |
| `split` | Evaluation metadata |
| `Year` | Kept out of prediction; historical/generalization concern |
| `Age`, `AgeOfPolicyHolder`, `age_was_missing`, `Sex`, `MaritalStatus` | Fairness-sensitive fields/proxies; age quality issues |
| `Fault`, `PoliceReportFiled`, `WitnessPresent`, `AddressChange_Claim` | Claim-time availability/provenance unresolved; later updates may leak information |
| `NumberOfSuppliments` | Likely post-estimate information, unsuitable for initial-claim scoring |
| `incident_type(synth)`, `accident_severity(synth)`, `claim_amount_usd(synth)` | Fabricated demo fields; explicitly not model features |

Confirm that prior-claim history excludes the current claim and that ratings
and vehicle counts are snapshots available when scoring. Excluding explicit
demographics does not remove all demographic proxies. No new engineered
features or changes to Ethan's feature schema were made; additions require
coordination. Ethan's `src/features.py` was absent during the experiments but
has since arrived from `origin/main`. It uses a different engineered schema,
including fields excluded above. The saved models here do **not** call
`build_features`; they own their fitted 19-input preprocessing. Passing Ethan's
encoded output into these pipelines would create a train/serve mismatch.
Reconcile any future shared schema with Ethan and Damien before retraining or
changing inference. His file has not been modified by this work.

## 3. Modeling work completed

### Logistic regression baseline

[train.py](train.py) uses a scikit-learn Pipeline and ColumnTransformer:

- Numeric: median imputation, then StandardScaler.
- Categorical: most-frequent imputation, then OneHotEncoder with unknown
  categories ignored.
- LogisticRegression: `max_iter=1000`, `class_weight="balanced"`, seed 42.

Baseline artifacts are `models/baseline.joblib` and `models/metrics.json`.
The latter now contains the real-data baseline results, replacing its original
synthetic-demo metrics when baseline training was authorized.

Validation results at threshold 0.5: ROC-AUC **0.6941**, precision **9.18%**,
recall **69.57%**, F1 **0.1621**; confusion matrix `[[1633,1267],[56,128]]`.

### XGBoost and capacity experiments

[train_xgboost.py](train_xgboost.py) added a CPU XGBoost pipeline with train-only
imputation/one-hot encoding and `scale_pos_weight=8697/554`. Its original
validation ROC-AUC was **0.7125**, AP **0.1245**.

[experiment_capacity.py](experiment_capacity.py) compared 36 configurations
using three-fold stratified training-only CV, then evaluated locked family
finalists once on validation. It selected a tuned XGBoost candidate by mean
recall at fixed 10%, 20%, 30% workloads. The tuned uncalibrated candidate has
validation ROC-AUC **0.7169**, AP **0.1256**. A HistGradientBoosting finalist
had ROC-AUC **0.7189**, AP **0.1404**, and caught **82/184 fraud cases** among
617 highest-scoring validation claims. These were prioritization experiments,
not probability-calibration evidence.

[improve_ranking.py](improve_ranking.py) subsequently explored precision at a
fixed 20% workload using training-only CV. It found a modest gain at that
workload, but worse AP/other capacities; that candidate did not replace the
locked model. Its validation and test performance were not evaluated.

Earlier workload/triage scripts remain for reproducibility. Their queue rules
are not automatic insurance decisions or mandatory product workload settings.

## 4. Separate calibrated probability experiment

[train_probability.py](train_probability.py) compares:

1. Logistic regression: the training-CV-selected `C=0.1`, unweighted model.
2. Locked tuned XGBoost: depth 2, 200 trees, learning rate 0.04,
   `min_child_weight=5`, `reg_lambda=5`, `max_bin=128`,
   `scale_pos_weight=sqrt(8697/554)`.
3. An equal-weight soft-voting combination of those two pipelines.

Each family is evaluated raw, sigmoid-calibrated and isotonic-calibrated.
Class-weighted training scores are calibrated using natural, unweighted
calibration labels. No oversampling is applied to calibration data.

### Selection procedure

- Outer CV: three stratified shuffled folds, seed 2026, on training only.
- Inner calibration: three shuffled folds, seed `100 + outer_fold`.
- CalibratedClassifierCV uses `ensemble=False`: inner out-of-fold predictions
  fit the calibrator; the base model then refits on all outer-training rows.
- All preprocessing stays inside the estimator pipeline and each fold.
- Select the calibrated candidate with the lowest mean outer-fold **log loss**;
  break ties using lower **Brier score**, then name.
- Retain raw models and a train-prevalence-only predictor as controls.
- Lock selections before opening validation. Fit final calibration using
  training-only out-of-fold predictions, seed 42, and refit the base model
  using all training rows.
- Compare the locked calibrated family finalists and raw controls once on
  frozen validation. No validation-based retuning follows.

This selected **isotonic-calibrated XGBoost**. Its mean outer-CV log loss was
**0.208667**, versus **0.210453** for sigmoid XGBoost, **0.210936** for sigmoid
logistic regression, and **0.211345** for the sigmoid blend. The combination
was tested rather than assumed to be better; it was not selected.

### Measured validation results

Lower log loss and Brier score are better. ROC-AUC/AP assess discrimination.

| Model | Log loss | Brier score | ROC-AUC | AP | Mean estimate |
|---|---:|---:|---:|---:|---:|
| **Probability experiment selection: XGBoost + isotonic** | **0.207653** | **0.054382** | 0.710078 | 0.110821 | 5.956% |
| Tuned XGBoost, raw | 0.280215 | 0.075632 | 0.716921 | 0.125552 | 18.694% |
| Logistic + sigmoid | 0.208409 | 0.054461 | 0.710096 | 0.114257 | 6.010% |
| Logistic, raw tuned model | 0.208345 | 0.054502 | 0.710096 | 0.114257 | 6.005% |
| LR/XGBoost blend + sigmoid | 0.209008 | 0.054611 | 0.718368 | 0.120899 | 6.023% |
| LR/XGBoost blend, raw | 0.230786 | 0.059569 | 0.718368 | 0.120899 | 12.350% |
| Constant training fraud rate | 0.226039 | 0.056103 | 0.500000 | 0.059663 | 5.989% |

Actual validation fraud prevalence is **5.966%**. Mean agreement alone does
not prove calibration; see the saved calibration curve and probability-bin
counts. Small-bin estimates remain uncertain.

For the selected model:

| Actual class | Mean probability | Median probability |
|---|---:|---:|
| Fraud | 9.083% | 8.962% |
| Non-fraud | 5.758% | 6.544% |

Calibration improves the probability objective, but it does not add predictive
information. Isotonic introduces ties and reduced AP/ROC-AUC relative to raw
XGBoost. There is still substantial overlap between actual classes.

No validation claim received a calibrated probability >=0.5. At that diagnostic
cutoff accuracy is 94.03%, recall/F1 are zero, and the confusion matrix is
`[[2900,0],[184,0]]` (rows actual, columns predicted; labels `[0,1]`). This
does not mean the calibrated model is useless; it means a 50% decision cutoff
is unsuitable for these estimates and this low-prevalence dataset. Do not
optimize the product by simply forcing fraud scores above 0.5.

## 5. Where the artifacts are

The separately saved probability experiment is
[`models/experiments/probability_20261017T050105523504Z/`](../models/experiments/probability_20261017T050105523504Z/).

| Artifact | Purpose |
|---|---|
| `calibrated_probability.joblib` | Selected complete model: fitted preprocessing, classifier and calibrator |
| `manifest.json` | Model ID/hash, feature schema, positive class and dependency versions |
| `xgboost_isotonic.joblib` | Selected model under its comparison name |
| `logistic_sigmoid.joblib`, `blend_sigmoid.joblib` | Other calibrated family finalists |
| `protocol.json`, `selection.json` | Predeclared procedure and training-only selection |
| `cv_folds.csv`, `cv_comparison.csv` | Cross-validation measurements |
| `training_oof_predictions.csv` | Held-out training-fold predictions |
| `validation_comparison.csv`, `validation_predictions.csv` | Locked validation comparison |
| `calibration_bins.csv`, `calibration.png` | Reliability evidence |
| `metrics.json`, `README.md` | Complete evaluation, audit hashes and detailed notes |

The earlier locked base models are in
[`models/experiments/capacity_20261017T042040511095Z/`](../models/experiments/capacity_20261017T042040511095Z/).
`train_probability.py` currently clones `logistic_02.joblib` and
`xgboost_09.joblib` from that directory to reproduce their fixed configurations.
Include these source artifacts when sharing a reproducible training checkout.
Loading the final calibrated artifact does not require those source models.

**Do not confuse model generations:**

- `models/model.json` is the teammate's original five-feature synthetic demo
  model and was preserved.
- `models/xgboost_pipeline.joblib` is the first real-data XGBoost model.
- `.../capacity_.../xgboost_09.joblib` is the **current risk-prioritization
  deployment candidate**, locked tuned and uncalibrated.
- `.../probability_.../calibrated_probability.joblib` is the selected model
  **within the separate probability-estimation experiment**, not a deployment
  replacement.

## 6. Current risk scorer: inference and handoff

With the repository available and `src` on the Python import path:

```python
from explain import score

# claim is a flat dictionary containing the 19 original source features.
result = score(claim)
what_if = score(claim, exclude=["Make", "Deductible"])
```

`score` returns the agreed `risk_score`, `tier`, `contributions`, `excluded`
keys, plus quality/explanation/sensitivity metadata. Runtime files required:

- `src/explain.py`;
- `reports/explain_runtime/manifest.json` (model hash and provisional bands);
- `models/experiments/capacity_20261017T042040511095Z/xgboost_09.joblib`.

The full joblib contains fitted preprocessing; no training CSV is read during
scoring. Missing values are imputed, unknown categories are flagged, and
`operational_routing_allowed` is false for quality issues or sensitivity
scenarios. It is not permission to approve/reject a claim when true.

Native TreeSHAP impacts aggregate all encoded columns back to the 19 original
features. They are **log-odds**, not probability-point changes. Exclusions
replace values with fitted train-only references and are hypothetical
sensitivity analysis, not a new observed score or a retrained model.

The provisional API tiers are `fast_track`/low below .25596985,
`review`/medium from .25596985 to below .31254968, and `investigate`/high from
.31254968. These reflect validation score distributions, not calibrated
probabilities, confirmed fraud labels or guaranteed future queue percentages.
Low priority means standard checks and safeguards still apply.

**Damien (API/deployment handoff):** use the whole locked pipeline via the
scorer; preserve the 19 raw input types and transform exactly once. Coordinate
the different `src/features.py` schema rather than substituting it. Translate
invalid-input errors appropriately, propagate quality/sensitivity metadata,
and verify model loading and score parity on s390x. The existing five-feature
demo API/model remains separate and has not been overwritten. Do not switch
the scorer to the calibrated experiment without an explicit model decision.

**Da Feng (UI handoff):** retain the four response keys, display a relative
risk score and provisional priority, and show human-decision wording. Do not
label the raw score as a fraud probability or treat low priority as clearance.
Display signed contribution impacts in log-odds, not the existing probability
point formatter. What-if results need a sensitivity banner and must not
overwrite the observed claim score. Coordinate feature labels, missing/unknown
warnings, and the existing five-feature PDF/demo interface before integration.

See the [detailed scorer contract](../reports/explain_runtime/HANDOFF.md),
[measured example response](../reports/explain_runtime/example_response.json),
and [inference tests](../tests/test_explain.py).

## 7. Loading the separate calibrated experiment

From the repository root, with a trusted artifact and compatible dependencies:

```python
from pathlib import Path
import json
import joblib
import pandas as pd

directory = Path("models/experiments/probability_20261017T050105523504Z")
metadata = json.loads((directory / "manifest.json").read_text())
model = joblib.load(directory / metadata["model_file"])

# claim is a dictionary of the 19 approved input values, never the target.
missing = set(metadata["features"]) - set(claim)
if missing:
    raise ValueError(f"Missing claim fields: {sorted(missing)}")
row = pd.DataFrame([claim], columns=metadata["features"])
positive_column = list(model.classes_).index(metadata["positive_label"])
estimated_fraud_probability = float(model.predict_proba(row)[0, positive_column])
```

The artifact includes the calibrator; serving just its XGBoost trees would
return the wrong output scale. Unknown categories and NaN values are supported
by preprocessing, but incomplete or unfamiliar claims still require data-quality
checks before their estimates are interpreted.

### Explanation and UI status

[explain.py](explain.py) still loads the **uncalibrated** locked tuned XGBoost
through its separate manifest. It has not been switched to the probability
candidate. Its native TreeSHAP impacts are log-odds contributions for that raw
model, not additive explanations of the new calibrated probability.

The existing teammate API/PDF demo uses five different synthetic features and
is not wired to this 19-feature probability artifact. Integrating either new
model, aligning extraction/schema and adapting explanations are separate
coordinated tasks. No UI implementation or deployment was performed here.

## 8. Reproduction and verification

From the repository root in a compatible Python environment:

```bash
python src/train.py                 # overwrites baseline.joblib and metrics.json
python src/train_xgboost.py         # regenerates first XGBoost artifacts
python src/experiment_capacity.py  # creates a separate capacity experiment
python src/train_probability.py    # creates a separate probability experiment
python -m unittest discover -s tests -p test_explain.py -v
```

Do not run the first two commands casually when preserving existing artifacts;
they intentionally write their documented outputs. The probability script
uses the fixed source-run paths noted above, so it is not automatically
redirected to the newest capacity experiment.

Verified local training runtime: Python 3.12.14, NumPy 1.26.4, pandas 2.1.4,
SciPy 1.11.4, scikit-learn 1.4.1.post1, XGBoost 2.0.3, joblib 1.6.0.
Matplotlib 3.9.4 generates reports. Temporary dependencies were used during
verification; Damien's requirements files were not changed. Other machines
need their own compatible environment; they cannot rely on those temporary
packages being installed.

The repository's Ubuntu 24.04 approach uses distribution packages for s390x
native dependencies. Package availability was checked, but the new calibrated
artifact has **not been executed on LinuxONE**. Verify on-target loading,
version compatibility and prediction parity before release; do not assume
Windows joblib portability from native-model portability.

Checks completed:

- Saved calibrated model reloaded and matched full validation predictions.
- Log loss, Brier score, ROC-AUC and AP independently recomputed from saved
  predictions in a fresh process.
- Missing/unknown-input checks returned finite probabilities in `[0,1]`.
- Existing model artifact hashes remained unchanged.
- Earlier raw explanation tests passed, including score parity, contribution
  additivity, missing/unknown categories and exclusions. Those tests concern
  the raw explanation model, not the new calibrated probability mapping.

## 9. Limits and next steps

The available predictors provide only moderate separation. Calibration makes
probability estimates better aligned with observed frequencies; it does not
prove an individual estimate is correct or create a high-precision fraud
detector. The small, historical dataset may not represent current insurers,
and changed prevalence can invalidate calibration.

Validation has already been examined repeatedly during project development,
so its metrics are exploratory. Base hyperparameters were previously selected
using these training rows; the latest nested calibration CV does not erase
that historical selection. Freeze the final model and interpretation before
an approved once-only evaluation on the untouched final test holdout.

Coordinate any real claim-time feature additions, verify the model on LinuxONE,
and obtain approval before replacing a working model or deploying. Nothing
in this work certifies fraud, certifies absence of fraud, or authorizes an
insurance claim decision.
