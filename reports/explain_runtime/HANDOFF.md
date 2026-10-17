# Locked tuned XGBoost explanation handoff

Implementation: src/explain.py. No API/server/UI/trainer/requirements files were
changed. No retraining, test data access, push, deployment or working artifact
replacement occurred. All pre-existing model hashes and the validation hash
matched the locked experiment after verification.

## Model and dependencies

Load models/experiments/capacity_20261017T042040511095Z/xgboost_09.joblib,
the complete fitted preprocessing + XGBClassifier pipeline. This is the locked
depth-2, 200-tree candidate with sqrt-ratio class weighting, not the older
models/xgboost_pipeline.joblib or five-feature models/model.json.
reports/explain_runtime/manifest.json pins its SHA-256, path, model ID and bands.
All paths resolve from the repository, independent of current working directory.
Loading is lazy, cached once per process; call score once during startup to
warm/check artifacts. Do not mutate the shared model or fit it in the API.

Verified original local runtime: Python 3.12.14, NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4,
scikit-learn 1.4.1.post1, XGBoost 2.0.3, joblib 1.6.0. No shap package needed.
Temporary packages were used; requirements.txt is unchanged. Use compatible
versions for trusted joblib loading. Subsequent team documentation reports
LinuxONE artifact/score parity checks; see [the evaluation report](../../docs/evaluation.md)
and retain timestamped evidence for the currently running release. Concurrent
service throughput is not established by those small parity checks. The
separate calibrated experiment is not the served model.

## Python interface

With src on the Python import path (`uvicorn ... --app-dir src` does this):

```python
from explain import score

result = score(claim)
what_if = score(claim, exclude=["Make", "Deductible"])
```

The default is exclude=None rather than a mutable list; behavior is equivalent
to an empty exclusion list. The existing flat payload style is retained:
claim contains approved feature keys, optional claim_id, and optionally exclude.
An explicitly supplied exclude argument overrides embedded claim['exclude'].
The existing server handler can pass the flat JSON to score and translate
ValueError into HTTP 422. This is a handoff proposal, not an edited endpoint.

Keep the 19 raw input names from the fitted pipeline:
Deductible, Month, WeekOfMonth, DayOfWeek, MonthClaimed, WeekOfMonthClaimed,
DayOfWeekClaimed, Make, AccidentArea, VehicleCategory, BasePolicy, AgentType,
VehiclePrice, DriverRating, Days_Policy_Accident, Days_Policy_Claim,
PastNumberOfClaims, AgeOfVehicle, NumberOfCars.

Deductible must be a finite number. WeekOfMonth, WeekOfMonthClaimed and
DriverRating are integer numeric categories (integral floats are normalized).
Remaining fields are exact source category strings. Do not translate bands
into midpoints or fabricate exact durations. Target, unapproved columns,
non-scalar values, numeric strings for numeric fields, infinities and unknown
exclusion names raise ValueError. Missing fields, nulls, NaN and blank strings
use fitted imputers and are reported; categorical unknowns use fitted
OneHotEncoder(handle_unknown='ignore') and are reported.

## Response contract

The four agreed top-level keys are preserved:

```json
{
  "risk_score": 0.03716346248984337,
  "tier": "fast_track",
  "contributions": [
    {"feature": "Make", "label": "Vehicle make", "value": "Honda", "impact": 0.0,
     "excluded": false, "imputed": false}
  ],
  "excluded": []
}
```

The contribution entry above illustrates schema only, not a measured Make
impact. example_response.json contains the actual measured response and all
19 values; example_what_if.json contains the actual sensitivity response.

- risk_score: uncalibrated relative fraud-risk ranking score in [0,1]. Display
  as a risk score, not a validated probability or fraud finding.
- tier: existing API names fast_track/review/investigate, mapped to low/medium/high.
- contributions: all 19 original features, sorted by absolute signed impact,
  plus human-readable label, effective value, excluded/imputed flags. Each
  impact is exact native TreeSHAP aggregated over *all* one-hot columns owned
  by that feature, including inactive categories.
- excluded: deduplicated original feature names in canonical input order.

Additional metadata includes model_id, score_type, priority_band, tier_status,
recommendation, contribution_method, impact_unit, base_value, raw_margin,
sensitivity_analysis, missing_features, unknown_categories,
requires_data_quality_check, operational_routing_allowed, safety_note.

The API must propagate/act on the quality/sensitivity metadata, rather than
discard it and route automatically from tier alone. operational_routing_allowed
is false for any missing/unknown inputs or exclusion scenario. This means the
result must not be used for automatic priority routing; it never means that
approval/rejection is allowed when true. For unknown/missing inputs the band
is still returned as provisional model output, with input-verification text.

## Explanation units and what-if semantics

XGBoost Booster.predict(pred_contribs=True, approx_contribs=False) gives native
TreeSHAP contributions on raw log-odds. See the
[XGBoost 2.0.3 API](https://xgboost.readthedocs.io/en/release_2.0.0/python/python_api.html#xgboost.Booster.predict).
The final native column is the bias, returned as base_value. The scorer checks:

`base_value + sum(contribution.impact) ~= raw_margin`

`sigmoid(raw_margin) ~= risk_score`

Positive impact raises the model margin relative to its TreeSHAP baseline;
negative lowers it. These are not probability percentage-point changes,
causal findings, evidence sufficient to prove fraud, or differences from the
median reference claim. The bias is a TreeSHAP model baseline, not necessarily
the training fraud prevalence or the reference claim score. Do not sum only
the five displayed features and expect the full prediction; keep the remainder
available in a details view.

What-if exclusion replaces original values with their *fitted training*
median/modes, preserves all encoded columns and rescores. It does not retrain,
delete columns or remove all effects/proxies of a feature. categorical
references come directly from the fitted most-frequent imputer, numeric
references from the fitted median imputer. No runtime dataset read occurs.

sensitivity_analysis identifies the mode, original_risk_score, score_change
(scenario score minus original score), and reference_values. Display a
"Sensitivity analysis" banner. Never use the scenario tier to overwrite the
observed claim's priority. Contributions describe the scenario's effective
inputs; excluded fields can have nonzero impacts at their reference values.
They remain in the response so additivity is preserved; grey them in the UI
instead of forcing their impact to zero or silently dropping their effects.

## Chosen current score bands

Review begins at the team's chosen **0.1754** validation-based leakage-control
boundary (commit bc85518). Investigation remains **0.31254968**, displayed
approximately as **0.31255**. The model and fitted preprocessing are unchanged:

| Priority | API tier | Score band |
|---|---|---|
| Low | fast_track | score < 0.1754 |
| Medium | review | 0.1754 <= score < 0.31254968 |
| High | investigate | score >= 0.31254968 |

Under this policy validation has 1,111 low, 1,665 review and 308 investigate
claims. Review-only workload is 53.99%; combined workload is 63.98%, with
175/184 fraud labels captured (95.11%). These are observed validation values,
not fixed percentages, probability cutoffs or future coverage guarantees.
Bands are not exact batch top-K assignment. Version and monitor the policy.

Historical note: the original review boundary was **0.25596985**, yielding
about 30% combined validation workload. That value remains in the original
[frozen test protocol](../../docs/figures/test/frozen_protocol.json) by design.
The current 0.1754 boundary was adopted after the original test evaluation.
Its saved-test result (174/185 fraud captured, 94.05% recall, 63.13% workload)
is [post-evaluation policy reanalysis](../../docs/figures/routing-update/README.md),
not a new untouched holdout assessment. Do not overwrite the historical
protocol or mix original and current-policy metrics/charts.

Recommended text is returned per tier: specialist investigation, lightweight
verification, or expedited standard processing with checks/safeguards. No tier
certifies fraud, certifies fraud absence, approves payment or rejects a claim.

## Damien's current integration contract

1. `/score` now calls this function and maps ValueError to 422. Preserve that
   wiring; handle artifact/version failures as service failures, not low scores.
2. `/model/final` exposes the 19 names, explain.LABELS and band/model metadata.
   `/model/info` remains the separate demo interface. The five-feature LR coefficients, PDF extractor,
   lr_terms and tree visualizations cannot describe this model. Do not blend
   the synthetic LR with the tuned XGBoost.
3. Keep the final UI's impact label/formatter in signed log-odds (or a clearly marked relative
   explanation bar); keep risk_score separate. Show actual what-if score_change
   as the scenario score difference, not as a SHAP contribution.
4. Show provisional band, quality warnings, excluded/imputed state, human-only
   recommendation and sensitivity banner. Preserve the original observed score.
5. Retain LinuxONE parity evidence for the current release and thresholds,
   then perform any separately authorized load/concurrency checks.
   Request approval before any push/deployment or working-model replacement.

## Verification

Run `python -m unittest discover -s tests -p test_explain.py -v` in a compatible
environment. Eight tests pass: 12 actual validation claims match pipeline
predictions; original-feature contributions add back to the native margin and
score; missing/unknown inputs flag quality; one/all exclusions equal direct
reference replacement without mutation; invalid input is rejected; no runtime
CSV reads/fitting; strict JSON serialization and band metadata are valid.

Measured validation claim 1: risk_score=.03716346248984337, provisional
fast_track. Local warm scoring including native explanations: median 2.29ms,
p95 2.82ms over 20 runs (Windows AMD64, one thread, excluding transport/loading).
No test data or fresh model fitting was used. Full measured details are in
verification.json. No API or UI changes have been made by this task.
