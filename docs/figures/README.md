# Evaluation figures

## Final test: `test/`

The user confirmed locked tuned XGBoost and authorised once-only evaluation. `test/` contains its final test report inputs, frozen protocol, verification, predictions, five figures, feature summary and model/capacity comparisons. All were generated without retraining or modifying thresholds.

Use `python docs/figures/evaluate_test_once.py --reuse` to regenerate summaries and plots from the saved test predictions. This does not load models or reopen test data. First evaluation refuses to overwrite existing predictions. The original baseline is a fixed comparator; other candidates/calibration models were not evaluated on test.

## Earlier validation: this directory


These figures describe the locked model outputs from commit 48e14f9 on 3,084 real historical-data validation claims, including 184 fraud labels. They are not synthetic-demo charts or final-test results.

- `roc_curve.png`: original LR baseline, tuned LR, locked XGBoost and histogram-boosting reference.
- `precision_recall_curve.png`: the same models and a fraud-prevalence reference.
- `confusion_matrix.png`: candidate at the manifest investigation threshold 0.31254968.
- `feature_contributions.png`: top 12 mean absolute TreeSHAP contributions in log-odds across all validation claims; full 19-feature table in CSV.
- `tier_distribution.png`: observed provisional score-band counts.
- CSV files and `evaluation_summary.json`: computed values and audit record.

Reproduce from the repository root with `python docs/figures/generate_evaluation.py` in the numerical environment recorded in the summary. On macOS the XGBoost OpenMP library may need to be located, for example via the installed scikit-learn `.dylibs` directory. Do not run any training scripts to reproduce these plots. The generator reads validation inputs only and leaves models/test data unchanged.

Final test evaluation is pending final model confirmation. Deployment parity remains Damien's handoff.
