# Calibrated fraud-probability model selection

Training/calibration work only. No UI, API, explanation scorer, deployed model,
requirements or teammate files were changed. Existing artifacts were protected
by SHA-256 checks. test.csv was never opened or hashed. Outputs are separate.

Reproduce with `python src/train_probability.py`. Each run creates its own
timestamped probability_* directory; it does not overwrite this run.

## Model selection and leakage controls

Data: frozen 9251 training rows, 554 fraud; frozen 3084 validation rows,
184 fraud. Same 19 feature names and FraudFound_P target (1=fraud).
No age/demographic, ID, split, synthetic or post-investigation fields added.
claim_id is for audit only, never a predictor. The target is not an input.

Base models are clones of previously training-CV-selected artifacts:

- logistic_02: numeric median imputation/scaling, categorical mode imputation
  and one-hot encoding, LogisticRegression(C=.1, class_weight=None,
  max_iter=1000, random_state=42).
- xgboost_09: median numeric imputation, categorical mode imputation/one-hot,
  depth2, 200 trees, learning rate .04, min_child_weight5, lambda5, max_bin128,
  scale_pos_weight=sqrt(8697/554), one CPU worker.
- blend: equal-weight soft-voting combination of those two pipelines.

Compared raw outputs, sigmoid and isotonic calibration for all three families.
Calibration data retains natural class prevalence, with no calibration sample
weights or oversampling. A prevalence-only predictor is a sanity baseline.

Outer CV: 3 stratified shuffled training folds, seed2026. Within each outer
training partition, CalibratedClassifierCV uses 3 shuffled inner folds,
seed100+outer_fold, ensemble=False. This fits the calibrator on inner
out-of-fold model outputs, then refits the base estimator on all outer-training
rows. All imputers/encoders/scalers are inside estimator pipelines, so inner
and outer holdouts never fit preprocessing or calibration. Logistic calibration
uses its decision_function; XGBoost and soft voting use predict_proba, matching
scikit-learn's documented behavior. Fits are sequential, one native thread.

Selection was locked in selection.json before validation was first opened:
minimum mean outer-fold log loss among six calibrated models, tie-break lower
Brier score then name. Raw models were controls, not forced to appear worse;
all results are reported. The best calibration per family was also locked for
the validation comparison. Final calibrators use training-only inner OOF
predictions, seed42; their base estimators refit on all 9251 training rows.

Prior base-model hyperparameter selection already used these training rows;
this nested calibration comparison is not an independent unbiased evaluation
of the entire historical model search. Validation has also been used in
earlier development; the single final comparison here remains exploratory.
No parameter/calibration choice was changed after viewing validation results.

## Actual results

Training-only outer-CV comparison (lower log loss/Brier is better):

| Candidate | Log loss | Brier | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| XGBoost isotonic, selected | .208667 | .054556 | .702603 | .116154 |
| XGBoost sigmoid | .210453 | .054681 | .701212 | .123757 |
| Logistic sigmoid | .210936 | .054810 | .696294 | .114178 |
| Logistic raw | .211021 | .054870 | .696294 | .114178 |
| Blend sigmoid | .211345 | .054830 | .702170 | .121019 |
| Blend isotonic | .212657 | .054727 | .694547 | .110096 |
| Logistic isotonic | .214079 | .054910 | .690779 | .104220 |
| Prevalence control | .226652 | .056299 | .500000 | .059885 |
| Blend raw | .232020 | .059957 | .702170 | .121019 |
| XGBoost raw | .280539 | .075987 | .701213 | .123757 |

Frozen validation comparison, after locking selection:

| Candidate | Log loss | Brier | ROC-AUC | AP | Mean estimated probability |
|---|---:|---:|---:|---:|---:|
| XGBoost isotonic, selected | .207653 | .054382 | .710078 | .110821 | 5.956% |
| XGBoost raw | .280215 | .075632 | .716921 | .125552 | 18.694% |
| Logistic sigmoid | .208409 | .054461 | .710096 | .114257 | 6.010% |
| Logistic raw | .208345 | .054502 | .710096 | .114257 | 6.005% |
| Blend sigmoid | .209008 | .054611 | .718368 | .120899 | 6.023% |
| Blend raw | .230786 | .059569 | .718368 | .120899 | 12.350% |
| Prevalence control | .226039 | .056103 | .500000 | .059663 | 5.989% |

Observed validation fraud prevalence: 5.966%. Mean alignment alone does not
prove calibration; calibration_bins.csv contains counts, mean estimates and
observed rates using both quantile and fixed bins. Duplicate isotonic outputs
reduce the number of distinct quantile bins. calibration.png visualizes
reliability. Proper scoring metrics improve over both raw XGBoost and the
constant-prevalence control, but gains over the constant control are modest.

Selected-model fixed probability bands:

| Estimated band | Claims | Actual fraud | Mean estimate | Observed rate |
|---|---:|---:|---:|---:|
| 0-1% |973|7|.72%|.72%|
| >2-5% |108|2|4.65%|1.85%|
| >5-10% |1698|136|7.39%|8.01%|
| >10-15% |187|21|13.51%|11.23%|
| >15-20% |96|13|15.84%|13.54%|
| >20-30% |22|5|25.79%|22.73%|

Empty bands are omitted. Small bins are uncertain, especially the 22-claim
highest band. These are estimates, not individually verified probabilities.
The source data's 1990s geography/prevalence may not represent a modern
insurance portfolio, so calibration must be reassessed on the actual population.

Actual fraud claims: mean estimated probability 9.083%, median 8.962%.
Actual legitimate claims: mean 5.758%, median 6.544%. Considerable overlap
remains; calibration does not invent new predictive information.

At the diagnostic threshold .5 the selected model predicts no positives:
accuracy .940337, precision0, recall0, F1=0, specificity1, confusion matrix
[[2900,0],[184,0]] (rows actual, columns predicted, labels [0,1]). A .5 decision
rule is not recommended and not part of this probability artifact. Low honest
probabilities can be appropriate at ~6% prevalence. No automatic approval,
rejection, routing or score bands were implemented.

Isotonic is monotone but introduces score ties: AP and ROC-AUC fell relative
to raw XGBoost. Calibration improves the probability objective here, not every
ranking or threshold metric. A different model should not be declared superior
merely because more of its outputs exceed .5. The blend was evaluated, but
was not forced into the final model because its probability losses were worse.

## Saved model files

- calibrated_probability.joblib: selected complete calibrated model.
- xgboost_isotonic.joblib: same selected candidate under its comparison name.
- logistic_sigmoid.joblib, blend_sigmoid.joblib: calibrated family finalists.
- manifest.json: model hash, feature schema, class order and environment.
- protocol.json, selection.json, cv_folds.csv, cv_comparison.csv,
  training_oof_predictions.csv: training-only selection evidence.
- validation_predictions.csv, validation_comparison.csv, calibration_bins.csv,
  calibration.png, metrics.json: locked validation evaluation and audit details.

The serialized object includes fitted preprocessing, classifier and calibrator.
Do not load it as an XGBoost native Booster or extract and serve just a tree.
No custom estimator classes are used, so standard installed dependencies are
sufficient to load it. With a compatible trusted-artifact environment:

```python
from pathlib import Path
import json
import joblib
import pandas as pd

directory = Path("models/experiments/probability_20261017T050105523504Z")
metadata = json.loads((directory / "manifest.json").read_text())
model = joblib.load(directory / metadata["model_file"])
row = pd.DataFrame([claim], columns=metadata["features"])
positive_column = list(model.classes_).index(metadata["positive_label"])
fraud_probability = float(model.predict_proba(row)[0, positive_column])
```

`claim` contains the same 19 real input values; never FraudFound_P. Preserve
source types, including integer categories and string bands. Preprocessing
handles NaN and unknown categories, but inputs still need external data-quality
checks before interpreting probabilities for incomplete/out-of-distribution
claims. Output is an estimated fraud probability in [0,1]. No threshold, tier,
human-capacity policy or UI implementation is included.

The old src/explain.py remains attached to the uncalibrated locked model and
must not be mistaken for this new probability model. Raw tree SHAP values do
not directly explain the calibrated probability. Any eventual integration or
explanation change is separate work requiring coordination and approval.

Verified local dependencies: Python3.12.14, NumPy1.26.4, pandas2.1.4,
SciPy1.11.4, scikit-learn1.4.1.post1, XGBoost2.0.3, joblib1.6.0.
Matplotlib3.9.4 is used for training reports only. No requirements.txt edit.
Actual LinuxONE s390x loading and parity still require target execution;
cross-platform joblib compatibility must not be assumed from local success.

Verification completed: saved model reload matches full validation predictions;
missing numeric/categorical inputs and unknown makes return finite probabilities
within [0,1]. Artifact/input hashes preserve previous models; test is untouched.
The next approved final assessment should use test only once after all model
and interpretation decisions are frozen. Nothing was pushed or deployed.

Implementation follows the installed scikit-learn 1.4 interface:
[CalibratedClassifierCV](https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
