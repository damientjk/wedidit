# ClaimLens integration verification

Checked 17 October 2026, 14:09 SGT. Repository source reviewed: origin/main at `48e14f9`. The running model-test container contains additional server-side integration work not yet present in that reviewed commit. This is a timestamped verification, not a claim that later changes have been tested.

**Overall status: model and test scoring API pass; the final end-to-end dashboard/PDF flow is not ready.** No application source, container or deployment was modified during these checks.

## Verified passes

- LinuxONE responds as s390x; both the team demo and model-test containers are healthy.
- The test container's locked `xgboost_09` hash is `bfa2cc3ca893eb9f0f2d066472cd6b7897155ce4d2270c5a849711fc8ffb76e9`, exactly matching the evaluated model.
- Model-test `/score` passes **20/20** checks against cached final-test scores, with maximum difference **0**. These were parity checks using already-evaluated claims, not another round of model selection or test-metric tuning.
- Responses contain all 19 contributions, the correct model ID and `log_odds` units.
- What-if outputs equal explicit fitted-reference substitution, preserve the original score and disable operational routing for the hypothetical scenario.
- Missing inputs flag quality checks and disable routing; invalid exclusion names and invalid deductible values return HTTP 422.

## Reproduced integration failures

| Surface | Observation | Status |
| --- | --- | --- |
| Team service, server port 8000 | `/model/info` lists the old five features; 19-field requests do not match final-model scores | Still the demo, not the final scorer |
| Model-test service, server port 8002 | `/score` correctly invokes the new model, but `/model/info` still advertises five demo features | Metadata is inconsistent with scoring |
| Model-test dashboard startup | Browser throws `TypeError: Cannot read properties of undefined (reading 'xgboost')` in `app.js:59`; screen remains “Loading model...” | Reproduced browser failure |
| Dashboard interpretation | Served UI still says it averages LR and XGBoost; impact formatter uses probability points | Inconsistent with XGBoost-only / TreeSHAP log-odds |
| Three PDF samples on model-test service | All return HTTP 200, but final analysis contains five demo features and no `xgboost_09` model identity | Does not establish final-model PDF integration |
| Versioned Docker recipe | Reviewed origin/main still runs `train_demo.py` during build | Final build recipe not yet verified in shared source |

The API's four agreed response keys are preserved, but that alone does not make the existing UI compatible: metadata, feature extraction, metrics access and contribution interpretation also changed.

The three checked PDF samples are `claim_safe.pdf`, `claim_borderline.pdf`, and `claim_suspicious.pdf`. Their successful HTTP status is not an end-to-end pass for the final model.

## Timing and environment

The 20 sequential on-server HTTP checks to the model-test service had median 4.43 ms and p95 5.74 ms. This small diagnostic includes HTTP, has no deliberate warm-up, and is not a concurrency/throughput benchmark or a guarantee. It is distinct from a direct-scorer “about 3 ms” measurement.

Observed test-container versions: NumPy 1.26.4; pandas 2.1.4+dfsg; SciPy 1.11.4; scikit-learn 1.4.1.post1; XGBoost 2.0.3; joblib 1.3.2. Artifact loading and score parity passed despite joblib differing from the local evaluation environment's 1.6.0.

No test CSV was bundled inside the inspected test container; runtime scoring does not require a CSV. The team repository at the reviewed commit already contains `data/processed/test.csv`. No newer CSV commit was visible at verification time. The parity checks used the frozen dataset and cached predictions from the completed evaluation, without inventing another test split.

## Evidence

- [API and artifact verification](api-verification.json)
- [Sample PDF endpoint results](sample-verification.json)
- [Browser console error](browser-errors.json)
- [Dashboard startup screenshot](dashboard-startup.png)

This verifies the scorer and partial API integration. It does not certify the final dashboard, examples, PDF route, build recipe, live service replacement or model-serving latency under load. Those remain with their existing component owners; Yizhuo's model evaluation stays unchanged.

## Source update before publication

After the checks above, commit `393ad1c` reached `main`. It wires `/score` to `explain.score`, introduces `/model/final` for the 19-field metadata, exposes `/demo/score` separately, and changes the Dockerfile to copy committed model artifacts instead of retraining. `/model/info` and `/analyze` remain demo interfaces by design in that commit. The served UI was not changed in that commit. The timestamped failures above describe the earlier running containers; this newer source/deployment still needs a fresh end-to-end check. These source changes do not alter the evaluated model or its test results.
