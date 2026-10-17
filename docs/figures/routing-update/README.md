# Review-boundary policy update

Source: team update and bc85518. Current live model: xgboost_09 alone; review begins at 0.1754, investigate at exact API value 0.31254968 (display approximately 0.31255).

The generator applies these fixed values to cached predictions. It does not load or fit models, run inference, reopen the raw test CSV, or search thresholds. Original files under docs/figures/test remain untouched; their protocol describes the original 0.25596985 review boundary.

Validation: fast-tracked fraud labels fall from 77/184 to 9/184. Review-only workload rises from 20.01% to 53.99%; investigation remains 9.99%; combined workload is 63.98%.

Saved test reanalysis: fast-tracked fraud labels fall from 94/185 to 11/185. Review-only workload rises from 18.61% to 52.92%; investigation remains 10.21%; combined workload is 63.13%. These are revised-policy descriptions after original test results were available, not a new independent holdout evaluation. The team attributes the supplied cutoff to validation.

See routing_summary.json for exact metrics, source hashes and the audit note. Original scores, ranking metrics and model hashes are unchanged.
