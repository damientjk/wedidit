# Capacity-constrained experiment

Run `python src/experiment_capacity.py` to create a separate timestamped run.
This run used 9,251 frozen training rows (554 fraud), 3,084 frozen validation
rows (184 fraud), the 19 approved features and target `FraudFound_P`.
Training has 37 columns; none of the selected features is blank.

## Selection protocol

The protocol was written before fitting, and selection.json was written before
the first validation read. Three-fold shuffled stratified CV, seed 42, selected
hyperparameters only on training data. The score is mean fold recall across
10%, 20%, 30% queues, with mean AP then recall20 as tie-breakers. All folds use
their own fitted preprocessing; XGBoost weight ratios use only fold-training
labels. No validation early stopping or resampling. Validation was opened once
after locking one finalist per family and refitting those finalists on all train.

36 configurations / 108 CV fits, then 3 full-training fits. Sequential search,
one native/BLAS thread; 19 raw features, small one-hot matrices; HistGB uses
dense one-hot rather than ordinal categories. XGBoost limits max_bin to 128 and
trees to 500. Thread and search bounds are explicit; no OS-level hard RAM cap
was imposed. Run elapsed time was 21.375 seconds on Windows AMD64.

Compared 8 logistic configurations (C=.01,.1,1,10; balanced/unweighted),
24 XGBoost configurations (depth/tree pairs 2/200,2/500,3/200,3/500,4/300,5/200;
weights 1, sqrt(negative/positive), half ratio, full ratio), and 4 HistGB
configurations (7/15 leaves; balanced/unweighted).

CV winner: xgboost_09, depth 2, 200 trees, learning rate .04,
min_child_weight=5, reg_lambda=5, max_bin=128,
scale_pos_weight=sqrt(8697/554), approximately 3.962. It has mean capacity
recall .400751 and AP .127607. The next XGBoost has .400672, a negligible
CV margin; this is not evidence of a reliably superior configuration.

Other family finalists: logistic_02 (C=.1, unweighted),
hist_gradient_boosting_32 (7 leaves, 200 iterations, learning rate .05,
L2=5, min_samples_leaf=30, unweighted, early stopping disabled).
CV HistGB mean capacity recall=.398293, AP=.132435.

## Frozen validation comparison

Rank below uses validation caught@20%, then AP, solely to describe finalists.
This does not retroactively change the predeclared training-CV selection.

| Rank | Model | AP | ROC-AUC | Caught@308 | Caught@617 | Caught@925 | Caught@786 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | HistGB finalist | .140389 | .718931 | 45 | 82 | 107 | 95 |
| 2 | XGBoost finalist | .125552 | .716921 | 39 | 81 | 107 | 100 |
| 3 | Logistic finalist | .114257 | .710096 | 39 | 75 | 103 | 90 |
| 4 | Working XGBoost | .124494 | .712493 | 40 | 73 | 107 | 92 |
| 5 | Working logistic | .101912 | .694106 | 32 | 73 | 98 | 87 |

`validation_capacity.csv` includes every model's caught/missed fraud, false
positives, precision, recall, score cutoff and exact workload for 10%,20%,30%
and the historical 786-claim benchmark. All models rank by descending score;
ties use claim_id ascending, without labels. IDs are not model predictors.
Counts use round(n*capacity), matching the existing benchmark: 308,617,925.
617/3084 is 20.006%, a rounding convention, not an exact mathematical 20% cap.
If a contractual hard percentage ceiling is required, floor the batch quota
consistently for every model; these results should not be relabeled as 616.
Persisted cutoffs describe this batch only; do not deploy them as universal
probability thresholds. Scores are not calibrated probabilities.

Recommendation: retain xgboost_09 as the predeclared training-CV-selected
experimental model. It beats the current working XGBoost at 617 reviews
(81 vs 73; 8 extra fraud, 536 vs 544 false positives), and at 786 reviews
(100 vs 92). It does not improve at 10% (39 vs 40) or 30% (107 vs 107).
HistGB is the strongest validation challenger for a 10%-20% operation:
45/82 caught, AP .140389. At 617 its lead over tuned XGBoost is only one case;
do not claim a statistically established winner or run another validation-driven
hyperparameter search. Both should remain experimental pending operational
selection and an approved final assessment. No deployment occurred.

At a 30% queue both boosted finalists catch 107/184 fraud (58.15%), leaving
77 missed and 818 false positives. More review is not necessarily better
ranking: working XGBoost already achieves the same 30% result.

## Three-tier queue (20% total model escalations)

| Model | Tier | Claims | Fraud | Non-fraud |
|---|---|---:|---:|---:|
| CV-selected XGBoost | investigate (top 10%) | 308 | 39 | 269 |
| CV-selected XGBoost | review (next 10%) | 309 | 42 | 267 |
| CV-selected XGBoost | expedited standard (rest) | 2467 | 103 | 2364 |
| HistGB challenger | investigate (top 10%) | 308 | 45 | 263 |
| HistGB challenger | review (next 10%) | 309 | 37 | 272 |
| HistGB challenger | expedited standard (rest) | 2467 | 102 | 2365 |

The extra 308 reviews from 20% to 30% catch another 26 fraud with selected
XGBoost (8.44% marginal precision) or 25 with HistGB (8.12%).
Do not call the remaining claims safe, automatically approve payment, or reject
claims solely from model scores. Expedited standard processing retains normal
coverage/document checks, established mandatory referral rules and a small
audited sample of low-ranked cases. Missing/invalid input should route to the
standard exception workflow, not silently receive a low score. Define actual
case-batch/time-window quotas, include rule overrides and audits in the same
total human-capacity budget, and report unique escalated claims without double
counting. The measured 617 model referrals exclude any additional safeguard
work; allocating safeguard slots inside that budget reduces available model
slots and requires a separately evaluated operating policy. No safeguard
performance or human-review sensitivity has been measured here.

## Feature and leakage review

No additions or edits to Ethan's feature code; src/features.py was absent in
this checkout. The imported fixed whitelist is the same 19 features as before.
Deductible is numeric; weeks and DriverRating are categorical despite integer
storage. No target encoding, oversampling, feature selection or fitted
preprocessing crosses a CV boundary.

- Exclude target, split, claim_id, RepNumber, synthetic columns and Year.
- Continue excluding Sex, MaritalStatus, Age, AgeOfPolicyHolder and
  age_was_missing from prediction. They may be retained for separate fairness
  audits; feature exclusion does not eliminate demographic proxies.
- Continue excluding Fault, NumberOfSuppliments, PoliceReportFiled,
  WitnessPresent and AddressChange_Claim until capture time/definition is
  established. Final fault, later witness/report findings, and repair
  supplements can leak post-investigation or post-intake information.
- Confirm PastNumberOfClaims excludes the current claim; the dataset does not
  establish a 12-month history window. Confirm DriverRating and NumberOfCars
  are values available at scoring. Do not change their semantics.
- Calendar fields are available at reporting but do not encode a reliable
  exact reporting delay; month/week/day combinations lack complete dated
  timestamps. Avoid fabricating an exact delay or policy-duration number.
- Candidate future features, only after coordination and data availability:
  verified prior-claim counts with an explicit cutoff/window; policy history
  as of intake; true reported amount; authenticated reporting delay; intake
  document completeness. This dataset cannot supply real claim amount,
  incident type or severity; never substitute the synthetic fields.
- Random frozen splits do not establish future-year generalization or
  separation by real customer/vehicle identity. claim_id is only a row ID.
  No reliable entity key is available for grouped CV. Age imputation and
  source inconsistencies remain documented in Yu Heng's README.

Validation has been used in earlier development. These are development
comparisons, not unbiased final performance estimates. With 184 fraud cases,
small count differences are uncertain; no significance claim is made.
LinuxONE s390x execution remains unverified for these experimental pipelines;
the package availability evidence in reports/xgboost/README.md is not an
on-host execution test. Load/reproduce with compatible target dependencies
before any separately approved deployment.

## Artifacts and integrity

All three finalist pipelines, protocol, CV fold results/ranking, locked
selection, validation predictions, comparison, queue metrics and results are
in this directory. Saved pipelines matched validation scores after reload.
Every pre-existing model artifact was protected by before/after SHA-256 checks.
The script read train.csv first, then val.csv exactly once after locking
finalists. test.csv was not opened, hashed, trained on or evaluated. Working
models, teammate files and requirements.txt were not modified. No automatic
replacement or deployment is implemented; deployment requires user approval.
