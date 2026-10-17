# Human-assisted investigation prioritization

This is a separate exploratory analysis of the two locked candidate models,
not a new training/tuning round or deployment. Source artifacts are in
models/experiments/capacity_20261017T042040511095Z. FraudFound_P=1 is the
retrospective positive label, not a label assigned to a live flagged claim.
The test file was not read or hashed. Existing artifacts remain unchanged.

## Audit and reproducibility

Training remains 9251 rows (554 positives), validation 3084 (184 positives).
The training/validation file hashes match the locked experiment. Current models,
baseline and native demo files match the experiment's protected SHA-256 values.
The 19 raw inputs, target and class proportions are unchanged. Prior audits
verified matching 37-column CSV schemas, no nulls, unique IDs and disjoint
train/validation IDs. Missing-value handling exists despite the clean inputs.
Neither IDs, split nor labels reach model fitting. Numeric Deductible uses
median imputation; categorical fields use most-frequent imputation and one-hot
encoding with unknown categories ignored. Only logistic models apply scaling.
HistGB uses dense one-hot output; XGBoost representation must remain consistent.

The prior search used 3-fold shuffled stratified training-only CV (seed 42),
36 configurations, 108 fits, fold-local preprocessing and weight ratios,
one CPU thread and a bounded grid. The exact protocol, fold results and locked
selection were saved before validation evaluation. xgboost_09 is the CV winner;
HistGB is the validation AP winner. New policy exploration reuses those saved
validation predictions, not a new fitting step. Validation has been used in
earlier development and is exploratory; these are not unbiased final results.
The small dataset, historical 1994-96 source and absence of reliable entity
keys limit temporal/entity generalization. LinuxONE target execution remains
unverified; earlier documentation establishes package availability only.

## Policy semantics

- investigate: highest risk requiring specialist investigation, not confirmed fraud.
- review: additional suspicious cases for lighter human verification.
- fast_track: expedited standard processing with coverage/document checks,
  mandatory referral rules and audits, never unconditional clearance/payment.

Capacity metrics set binary positive to investigate OR review. TP means a
retrospectively labelled fraud captured in those queues; it does not establish
that humans would actually confirm it. Investigate recall is TP in investigate
divided by all 184 positives. Combined coverage is both queues' TP/184.
False positives are legitimate cases escalated; false negatives are fraud left
in fast_track. Specialist and lightweight case costs are different and have
not been supplied, so escalated case counts are not staff-hours or dollar costs.
Audits, mandatory referrals and data-quality exceptions must be counted in total
human workload; the numerical results below cover model-driven queues only.

## Operational comparisons

policies.csv and analysis.json give all requested counts and metrics, including
separate tier fraud/false positives and confusion matrices. Queue counts use
round(N*combined_budget), then subtract round(N*investigation_budget). Thus
rounding stays consistent with the earlier 617-claim 20% benchmark.

| Model | Investigate + review budgets | I / R claims | I / R fraud | Fast-track fraud | Escalation % | Precision % | Coverage % | F1 | Accuracy % | Balanced acc % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGB | 5%+15% |154/463|23/58|103|20.01|13.13|44.02|.2022|79.28|62.77|
| HistGB |5%+15%|154/463|26/56|102|20.01|13.29|44.57|.2047|79.35|63.06|
| XGB |10%+20%|308/617|39/68|77|29.99|11.57|58.15|.1930|70.98|64.97|
| HistGB |10%+20%|308/617|45/62|77|29.99|11.57|58.15|.1930|70.98|64.97|
| XGB |10%+30%|308/926|39/88|57|40.01|10.29|69.02|.1791|62.26|65.42|
| HistGB |10%+30%|308/926|45/84|55|40.01|10.45|70.11|.1819|62.39|66.00|
| XGB |15%+35%|463/1079|63/79|42|50.00|9.21|77.17|.1645|53.24|64.45|
| HistGB |15%+35%|463/1079|65/78|41|50.00|9.27|77.72|.1657|53.31|64.74|
| XGB |20%+40%|617/1233|81/82|21|59.99|8.81|88.59|.1603|44.62|65.21|
| HistGB |20%+40%|617/1233|82/77|25|59.99|8.59|86.41|.1563|44.36|64.05|

XGB AP=.125552, ROC-AUC=.716921; HistGB AP=.140389, ROC-AUC=.718931.
These two ranking metrics do not vary with queue budgets. Investigate-only
recall at 10% is 39/184=21.20% for XGB and 45/184=24.46% for HistGB.
At 5% it is 23/184=12.50% and 26/184=14.13%, respectively.

Classification-at-score-0.5 is a separate diagnostic, not the queue policy:

| Model | TP/FP/TN/FN | Precision | Recall | F1 | Accuracy | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| XGB |2/4/2896/182|.3333|.0109|.0211|.9397|.5047|
| HistGB |0/0/2900/184|0|0|0|.9403|.5000|

Zero predicted positives uses precision=0. High accuracy here merely reflects
imbalance; it is not the optimization objective. Scores are uncalibrated.

## Pareto frontier and coverage goals

model_frontiers.csv contains the minimum observed top-K workload for each
attainable fraud count per model. combined_frontier.csv removes policies
dominated by another locked candidate: equal/higher fraud count at lower/equal
escalations. Tier splits do not change combined capture at fixed total K.
For a score tie, stable numeric claim_id breaks the tie without labels. The
minimum K calculations use retrospective labels; they cannot predict the
workload needed to achieve a recall target on unseen claims.

| Coverage goal | XGB minimum reviews | XGB workload | HistGB minimum reviews | HistGB workload |
|---|---:|---:|---:|---:|
|70%|1332|43.19%|1232|39.95%|
|80%|1606|52.08%|1637|53.08%|
|90%|1906|61.80%|1980|64.20%|
|95%|1973|63.98%|2009|65.14%|

No 70%+ goal is feasible in this validation sample at the assumed practical
30% escalation ceiling. Even 60% escalation does not reach 90% coverage.
No near-complete fraud detection claim is supported.

Recommendation: use tuned XGBoost as the practical experimental candidate,
consistent with the predeclared CV selection, with an assumed maximum of 30%
human escalations and about 10% specialist capacity. This is a planning budget,
not a staffing commitment approved by the team. At the requested 10%+20% policy,
925 escalations capture 107 fraud. Its observed Pareto-equivalent cutoff is
920 reviews (29.83%), capturing the same 107, versus HistGB needing 923 for
that count. A frontier-aligned candidate on this validation batch is:

- investigate 308: 39 fraud, 269 legitimate;
- review 612: 68 fraud, 544 legitimate;
- fast_track 2164: 77 fraud, 2087 legitimate.

Combined coverage 58.15%; investigate-only recall 21.20%; escalation precision
11.63%; FP=813; F1=.193841; accuracy=.711414; balanced accuracy=.650588;
confusion matrix [[2087,813],[77,107]]. This model/policy is recorded in
inference_checks.json. The five-slot plateau saving is exploratory and should
not be claimed to generalize. Prefer a stable approved quota near 30% in
operation rather than changing quotas to chase validation labels. HistGB
remains a strong alternative if specialist yield at 10% is the main concern:
45 vs 39 fraud in that tier; the evidence does not establish significant
superiority of either model.

At a strict 20% staffing ceiling, use top-617 (rounding convention) or floor
the percentage for a hard contractual ceiling. HistGB captures 82 at 617,
XGB 81. At 925 both catch 107. Extra review increases coverage but many fraud
cases remain outside the queues; humans are not guaranteed to detect every
labelled fraud routed to them. Safeguard slots must be reserved within the
same total capacity, reducing model slots; that combined policy is unmeasured.

## Leakage-safe feature opportunities, requiring coordination

No schema changes or new model features were made. Ethan's src/features.py is
absent in this checkout and was not created or edited.

Keep excluding target, claim_id, RepNumber, split, Year, synthetic amount,
synthetic severity/type, demographics and missing-age proxy. Fault, later
police/witness findings, NumberOfSuppliments and AddressChange_Claim remain
excluded until claim-time provenance is established. Confirm prior claims
exclude the current claim, and DriverRating/NumberOfCars are intake snapshots.
Yu Heng's imputed ages and inconsistent holder bands remain relevant to audit,
but are not prediction inputs. Vehicle, area and rating fields may proxy
demographics despite the explicit exclusions.

Potential deterministic encodings from existing real intake columns include
cyclic month/week-day encodings, coverage-by-vehicle or coverage-by-policy-age
interactions and fixed ordered-band encodings. These add representations, not
new observed information; tree models can already learn interactions. A
coarse accident-month versus claim-month consistency flag is possible only
with documented year-wrap/meaning rules. Exact reporting delay cannot be
reconstructed from these partial calendar fields. No actual claim amount,
severity or incident type exists; never fabricate them. Treat every proposed
representation as a schema change coordinated with Ethan/Damien and assess
with training-only CV; do not use this validation analysis for another search.

## Inference delivery

See INFERENCE_CONTRACT.md, src/triage_candidate.py, training_reference.json
and example_response.json. No server or teammate files were modified. Both
candidate pipelines were loaded and scored with the same preprocessing.
Single-claim scoring including 19-feature occlusion explanations had local
warm medians of 16.62ms XGB and 11.55ms HistGB (10 repetitions, Windows AMD64,
one CPU thread, excluding transport/loading). These establish local feasibility,
not production latency or s390x compatibility. Remaining blockers are on-host
LinuxONE verification, complete 19-feature intake extraction, quota/safeguard
budget approval and an approved final once-only held-out evaluation. No push,
deployed model replacement, or deployment occurred.
