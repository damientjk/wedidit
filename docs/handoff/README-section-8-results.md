## 8. Results

Claim! uses **tuned XGBoost (`xgboost_09`) alone**, with no ensemble. The model was locked before the original held-out test evaluation on **3,084 real vehicle claims, including 185 fraud labels (6.00%)**. Synthetic fields are excluded from prediction.

The comparison below uses the same test claims. Precision and recall refer to reviewing the **617 highest-scoring claims (approximately 20% capacity)** for each model, rather than applying the live score thresholds.

| Model | Test ROC-AUC | Test average precision (AP) | Precision at 20% capacity | Recall at 20% capacity | Fraud labels captured |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original logistic-regression baseline | 0.6822 | 0.0943 | 10.21% | 34.05% | 63 / 185 |
| Locked tuned XGBoost | 0.6877 | 0.1061 | 10.53% | 35.14% | 65 / 185 |

At matched capacity, XGBoost captures two additional fraud-labelled claims. The gain is modest; it does not establish statistical significance or operational savings. AP is average precision, not accuracy.

**Current live score bands**

| Tier | Score rule |
| --- | --- |
| `fast_track` | score < 0.1754 |
| `review` | 0.1754 ≤ score < 0.31254968 |
| `investigate` | score ≥ 0.31254968 (displayed approximately as 0.31255) |

At the unchanged investigation threshold, test precision is **12.38%** and recall is **21.08%**, flagging 315 claims containing 39 fraud labels. This is a different operating point from the matched-capacity comparison above.

**Validation trade-off after the review-policy update:** on 3,084 validation claims with 184 fraud labels, lowering the review boundary from 0.25596985 to 0.1754 reduces fraud labels routed to fast-track from **77 to 9**. Review-only workload rises from **20.01% to 53.99%**, with investigation unchanged at **9.99%**. Combined review plus investigation therefore reaches **63.98%** of validation claims. These are validation routing results, not an improvement in test ROC-AUC or a guarantee of future workload.

The review boundary changed after the original test evaluation. Revised test routing is reported separately as **post-evaluation reanalysis of cached predictions**, not a new untouched holdout evaluation. Scores are relative risk rankings, not calibrated fraud probabilities. Officers retain the final decision; fast-track does not mean automatic approval or clearance.

See the [full evaluation and charts](docs/evaluation.md), [original test results](docs/figures/test/evaluation_summary.json), and [routing-policy reanalysis](docs/figures/routing-update/README.md).
