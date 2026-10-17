# Training-only precision improvement experiment

The goal is better precision/recall at the same 20% human workload, not making
more uncalibrated scores exceed .5. No validation/test datasets were opened,
no existing artifacts changed, and no UI/model/deployment replacement occurred.
The approved 19 features and FraudFound_P target remain unchanged.

Run `python src/improve_ranking.py` to create a new isolated experiment.
The bounded search compared 27 configurations: locked tuned XGBoost control,
18 additional XGBoost structures/regularizations/weights, 4 HistGB settings
and 4 random forest settings. Three-fold stratified CV with seed42 selected
two leaders; these and the control were checked with three-fold seed137.
All preprocessing and class ratios were fitted within each training fold.
Fitting was sequential, one CPU thread, XGBoost at most 600 trees and forests
200 trees. 90 CV fits plus one final full-training fit took 101.391 seconds.

Selection: mean fold precision in a top-20% queue, with AP as tie-breaker.
Risk scores rank claims; equal scores use claim_id as label-independent tie
breaker, never as a predictor. Results below average six fold evaluations
(two seeds). Those folds reuse training rows and are not an independent
holdout; search/selection optimism remains. No validation gain is claimed.

| Model | Precision@20% | Recall@20% | AP | Recall@10% | Recall@30% |
|---|---:|---:|---:|---:|---:|
| New XGBoost candidate | .121286 | .405195 | .118753 | .226484 | .532437 |
| New HistGB candidate | .119665 | .399824 | .125830 | .230141 | .545114 |
| Locked XGBoost control, refitted within CV | .113992 | .380871 | .126225 | .241858 | .553261 |

The selected candidate improves precision@20% by 0.729 percentage points and
recall@20% by 2.432 points, relative to control on the same training folds.
It loses overall AP and recall at 10%/30%. This is a workload-specific modest
gain, not a demonstrated general improvement. Do not compare these CV scores
directly to earlier 3084-row validation results, which use a different sample.

Selected candidate_pipeline.joblib: XGBoost depth3, 600 trees, learning rate
.04, min_child_weight10, lambda10, alpha1, subsample.85, max_bin128,
scale_pos_weight=8697/554 on full training. Config ID xgb_09 is local to this
new run and is NOT the earlier locked xgboost_09 model. The latter remains
the model referenced by src/explain.py and its manifest. Both preprocessing
and classifier are included. The pipeline reloaded and matched predictions.

## Recommendation

Preserve the working UI model. The current evidence does not justify a model
replacement or stronger general fraud-detection claim. Keep this candidate
for a controlled future assessment only if the 20% workload objective is
confirmed. The already-used validation set was deliberately not queried
again, and the final once-only test remains untouched.

The old threshold-.5 recall of 1.09% does not mean top-ranked fraud capture
is 1.09%: scores are uncalibrated and concentrated below .5. Current provisional
UI bands are already distribution-derived (~.256 review and .313 investigate).
Moving .5 lower or increasing class weights can raise threshold recall by
escalating more claims; that alone is not improved discrimination. Calibration
is not a reliable cure for low ranking precision, and on a 6% fraud dataset
honest calibrated probabilities may remain far below .5 for most cases.

Large precision gains likely require additional verified intake information,
not just score rescaling. Candidate data improvements require coordination:
genuine observed claim amount, reliable prior-claims/time-window history,
authenticated intake timing and document checks, plus evaluation of claim-time
fault provenance if such information is actually available. Do not introduce
synthetic amount/type/severity or final investigation findings. No features
were added and Ethan's feature schema was not changed.
