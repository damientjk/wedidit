# Run inside the claimlens image from the repo root: docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/docs/missing_fields_analysis.py:/app/a.py:ro" claimlens:latest python3 a.py
"""For each field: how much does leaving it blank (median/mode imputation) change the score and tier?"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import explain
manifest, pipe, features, reference, encoder, categorical, _ = explain._load()
opts = {f: list(c) for f, c in zip(categorical, encoder.categories_)}
opts["Deductible"] = [300.0, 400.0, 500.0, 700.0]
df = pd.read_csv("data/processed/val.csv").sample(600, random_state=0)
def claim(r):
    return {f: (int(r[f]) if f in explain.INTEGER_CATEGORIES else float(r[f]) if f == "Deductible" else str(r[f])) for f in features}
claims = [claim(r) for _, r in df.iterrows()]
t0 = time.time()
base = [explain.score(c) for c in claims]
shap = {f: np.mean([abs(next(x["impact"] for x in b["contributions"] if x["feature"] == f)) for b in base]) for f in features}
out = []
for f in features:
    d, flips, rng, cross = [], 0, [], 0
    for c, b in zip(claims, base):
        m = explain.score({**c, f: None})
        d.append(abs(m["risk_score"] - b["risk_score"])); flips += m["tier"] != b["tier"]
        scores = [explain.score({**c, f: v})["risk_score"] for v in opts[f]]
        tiers = {("investigate" if s >= manifest["investigate_min_score"] else "review" if s >= manifest["review_min_score"] else "fast_track") for s in scores}
        rng.append(max(scores) - min(scores)); cross += len(tiers) > 1
    out.append(dict(field=f, mean_abs_shap=round(shap[f], 3), blank_mean_score_change=round(float(np.mean(d)), 4),
                    blank_tier_flip_pct=round(100 * flips / len(claims), 1), range_mean=round(float(np.mean(rng)), 3),
                    range_crosses_tier_pct=round(100 * cross / len(claims), 1)))
out.sort(key=lambda x: -x["blank_tier_flip_pct"])
# all-blank vs half-blank
allblank = explain.score({})["risk_score"]
print(json.dumps({"n": len(claims), "seconds": round(time.time() - t0), "tiers": [manifest["review_min_score"], manifest["investigate_min_score"]],
                  "all_blank_score": allblank, "fields": out}, indent=1))
