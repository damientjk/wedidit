# XGBoost validation comparison

Run `python src/train_xgboost.py` from the repository root. Paths resolve from
the script. Uses only frozen `data/processed/train.csv` (9,251 rows) and
`val.csv` (3,084 rows, 184 fraud). No test CSV is read. No validation fitting,
early stopping, or hyperparameter sweep. All preprocessing is fitted on train.

The same 19 features and `FraudFound_P` target are imported from Marcus's
`src/train.py`. Deductible gets median imputation. Categoricals get
most-frequent imputation and one-hot encoding with unknown categories ignored.
The CPU histogram classifier uses 200 trees, depth 3, learning rate 0.05,
seed 42, one worker, and `scale_pos_weight = 8697 / 554 = 15.6985559566787`.

## Measured validation results

| Model | ROC-AUC | AP | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.694106 | 0.101912 | 0.091756 | 0.695652 | 0.162128 | 1395 |
| XGBoost | 0.712493 | 0.124494 | 0.099502 | 0.760870 | 0.175990 | 1407 |

Confusion matrices use rows=actual, columns=predicted, labels=[0,1]. LR:
`[[1633,1267],[56,128]]`; XGBoost: `[[1633,1267],[44,140]]`.
Validation fraud prevalence is 0.059663.

`equal_workload.csv` compares identical top-k queues. Sort by descending score,
then ascending claim_id to break score ties without looking at labels.
`score_cutoff` is descriptive: ties can require a queue policy rather than
thresholding alone. At k=1395 LR catches 128 fraud, XGBoost 139; at k=617,
both catch 73. Results are descriptive; no statistical significance claim.

`thresholds.csv` contains both models' precision, recall, F1, confusion counts,
and flagged counts for thresholds 0.10 through 0.90 in steps of 0.05.
For zero predicted positives, precision is reported as 0 (`zero_division=0`).

## Candidate tiers

These are validation-selected candidates, not deployed settings. The selection
rule uses the largest grid threshold retaining at least 90% recall for review,
and the best grid F1 for investigate. Neither rule encodes business costs or
actual officer capacity. The LR comparison policy is derived by this common
rule and does not change the earlier LR threshold analysis or artifacts.

XGBoost candidates: fast_track <0.35; review >=0.35 and <0.55;
investigate >=0.55.

| Tier | Claims | Fraud | Non-fraud |
|---|---:|---:|---:|
| fast_track | 1139 | 17 | 1122 |
| review | 919 | 51 | 868 |
| investigate | 1026 | 116 | 910 |

Review plus investigation covers 167/184 fraud (90.76%) but handles 1945/3084
claims (63.07%). Fast-tracking leaves 17 fraud cases without model-triggered
review. An investigation cutoff of 0.60 reduces that queue to 631 claims,
including 76 fraud and 555 legitimate claims; with review starting at 0.35,
40 fraud cases move from investigation into review. The 0.55 cutoff has the
best grid F1, 0.191736, but still low precision (11.31%). Balanced-weight
model outputs are uncalibrated scores, not validated absolute probabilities.
Freeze business-approved thresholds before any later final held-out evaluation.

## Saved artifacts and inference contract

- `models/xgboost_pipeline.joblib`: preprocessing plus XGBClassifier; preferred
  Python inference interface, taking a DataFrame with the 19 raw columns.
- `models/xgboost_metrics.json`: parameters, versions, input hashes, validation
  metrics, threshold candidates, and equal-workload comparisons.
- `models/xgboost/model.json`: native XGBoost export. The existing teammate demo
  `models/model.json` is deliberately preserved, as confirmed by Marcus.
- `models/xgboost/preprocessor.joblib`: the matching fitted preprocessor.
- `models/xgboost/preprocessing.json`: exact raw/encoded column ordering and
  environment. Native JSON requires this matching preprocessing at inference;
  it cannot score the raw 19-column claims directly. Preserve output order,
  category vocabulary, imputations, and sparse/dense representation. Replacing
  sparse implicit zeros with dense zeros can change XGBoost predictions.

Both the full pipeline and native JSON plus saved preprocessing were reloaded
and matched validation predictions. SHA-256 checks protected the baseline,
baseline metrics, teammate native model, and train/validation inputs.
No API, demo trainer, requirements, or teammate files were edited.

## LinuxONE compatibility evidence and remaining blocker

Local execution: Windows AMD64, Python 3.12.14; NumPy 1.26.4, pandas 2.1.4,
SciPy 1.11.4, scikit-learn 1.4.1.post1, XGBoost 2.0.3, joblib 1.6.0.
Temporary dependencies were used; requirements.txt was not edited.

Official Ubuntu 24.04 Noble package listings were checked on 2026-10-17:

- [python3-xgboost 2.0.3-1build2, including s390x](https://packages.ubuntu.com/python3-xgboost)
- [scikit-learn native library 1.4.1.post1, s390x](https://packages.ubuntu.com/noble/s390x/python3-sklearn-lib)
- [NumPy 1.26.4, s390x](https://packages.ubuntu.com/en/noble/s390x/python3-numpy)
- [SciPy 1.11.4 compiled s390x files](https://packages.ubuntu.com/noble/s390x/python3-scipy/filelist)
- [pandas native library 2.1.4, s390x](https://packages.ubuntu.com/noble/s390x/python3-pandas-lib/download)

These establish a packaged CPU dependency route for Ubuntu 24.04 s390x,
consistent with the teammate's existing Dockerfile. They do not establish
that this model runs on the team's actual host. No connected LinuxONE runtime
or Docker runtime was available here. Actual s390x execution remains unverified.
Do not substitute x86 Windows/Linux wheels. Native JSON is the portable model
format; joblib portability across OS, endian order, and library versions still
needs target verification. Prefer training/serializing on the target system.

On the Ubuntu 24.04 LinuxONE host, an administrator can provision dependencies:

```bash
sudo apt-get update
sudo apt-get install python3-numpy python3-pandas python3-scipy python3-sklearn python3-xgboost python3-joblib
python3 src/train_xgboost.py --compatibility-check
```

The smoke check imports the stack, trains a tiny CPU model and verifies native
JSON reload predictions without reading any claim data or writing repo outputs.
Confirm its `machine` is `s390x`; a successful AMD64 run is not s390x verification.
Before serving, verify the saved full pipeline and native-plus-preprocessor on
the same frozen validation rows on the target host, with compatible versions.
The existing demo Dockerfile runs `train_demo.py` and does not copy the real
data/models, so it is not a deployment path for these artifacts without a
separately authorized teammate change.
