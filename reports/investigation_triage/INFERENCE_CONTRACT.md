# Proposed inference contract for Damien (not an API change)

Use the locked xgboost_09.joblib pipeline under
models/experiments/capacity_20261017T042040511095Z. HistGB challenger is
hist_gradient_boosting_32.joblib alongside it. Verify trusted artifact hashes
against the experiment and validate on LinuxONE with compatible dependencies.
Do not load arbitrary uploaded joblib files. Preserve the entire fitted
preprocessor and encoded order/representation. The existing demo model.json,
five-field PDF extractor, LR JSON and trees JSON are not compatible artifacts
for this contract; do not mix them with the 19-column models.

## Input

`TriageCandidate(model_path, reference_claim).score(claims)` accepts a list of
objects with numeric-string-compatible stable claim_id and a features object.
The reference_claim is training_reference.json (train-only numeric median and
categorical modes), used only for explanations, not for feature creation.
Require these exact keys and preserve the fitted input types:

| Keys | Type |
|---|---|
|Deductible|finite numeric|
|WeekOfMonth, WeekOfMonthClaimed, DriverRating|integer categorical levels, not stringified numbers|
|Month, DayOfWeek, MonthClaimed, DayOfWeekClaimed, Make, AccidentArea, VehicleCategory, BasePolicy, AgentType, VehiclePrice, Days_Policy_Accident, Days_Policy_Claim, PastNumberOfClaims, AgeOfVehicle, NumberOfCars|string categories with original source spelling|

Never supply FraudFound_P, IDs as predictors, demographics, post-investigation
columns or synthetic fields. Additional/missing feature keys are rejected.
Null/incomplete input is rejected to a standard data-quality exception workflow;
it must not silently receive a low score or clearance. The pipeline retains its
trained imputers, but this proposed intake adapter requires complete records.
Unknown categorical levels are accepted by fitted one-hot encoding but returned
as quality flags, requiring explicit operational handling. Never infer the
19 inputs by relabeling the five numeric demo/PDF fields. DriverRating and
prior-claim history must use the correct claim-time definitions.

## Immediate output and final queue assignment

score returns claim_id, model_id, risk_score in [0,1], score_type (uncalibrated
ranking score), top_contributing_features, unknown_categories, quality flags,
recommendation and initially null tier with tier_status=pending_closed_batch_ranking.
The model does not establish a calibrated absolute fraud probability.

`assign_batch(scored, investigate_slots, review_slots)` returns rank, final tier
(investigate/review/fast_track), tier_status and recommendation. Explicit integer
slots bound workload. Sorting is descending risk_score, then ascending numeric
claim_id for stable ties. Require unique IDs, one model/version per batch,
closed batch ID/time window, and atomic queue assignment/idempotence in the
eventual API. Return that policy version and window metadata in the service.
The example response shows the tested adapter output; endpoint integration,
authentication, version manifests and persistence are Damien's coordinated work.

True instantaneous final tier assignment and exact top-percentile capacity over
an unknown future stream cannot both be guaranteed. Score immediately to order
a live priority queue; finalize tiers in short closed microbatches or use a
separately validated rolling quota/scheduling policy. Until then tiers are
provisional, never unconditional clearance. Do not copy this validation batch's
score cutoffs into the API as universal probability thresholds.

## Explanations

For each of the 19 input features, replace its observed value with the
train-only median/mode reference and rescore; delta = original score minus
reference-replaced score. Return the five largest absolute deltas with feature,
observed value, reference value and signed score_delta_on_restoring_observed_value.
Positive means the observed value increased the score relative to that reference;
negative means it decreased it. Values are non-additive and interaction-dependent,
not causal findings, not SHAP values, not proof of fraud, and not sufficient
evidence for adverse action. The replacements are explanatory probes only,
not additional model inputs or synthetic claim fields. No SHAP dependency added.

## Required recommendations

- investigate: prioritize specialist investigation; flag is not a finding of fraud.
- review: request lightweight verification; flag is not a finding of fraud.
- fast_track: expedite standard processing with coverage/document checks and
  applicable safeguards; no certification of absence of fraud or payment approval.

Mandatory referral rules and unknown/missing-input exceptions must never be
silently fast-tracked. Add explicit override/audit reason codes and account for
their unique cases in the human capacity budget. Audits reduce slots available
for top-ranked referrals at the same total workload. The current offline metrics
do not measure combined model-plus-safeguard coverage or staff-hours.
Keep score, model/policy versions, batch rank, tier, feature/quality status and
human disposition in an audit trail under normal access/retention controls.

## Verification and remaining work

Both candidates passed local scoring/explanation and exact batch-count checks.
Incomplete input rejection was checked. Local warm single-claim medians including
explanations: XGB 16.62ms, HistGB 11.55ms; ten repetitions each, Windows AMD64.
No network, process loading, concurrent load or LinuxONE latency was measured.
Verify actual s390x artifact loading, preprocessing, ranking parity and latency
before release. Confirm intake availability for all 19 fields, choose staffing
and safeguard budgets, then request deployment approval. Existing API and UI
remain unchanged; this adapter is an experimental integration reference only.
