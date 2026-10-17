"""Demo server: serves the UI and the scoring endpoints.

  GET  /health
  GET  /model/info       metrics, LR weights, first XGBoost trees (for the live view)
  GET  /model/final      the final model's input fields, labels and tier thresholds
  POST /score            claim JSON (+ "exclude") -> final XGBoost model (src/explain.py)
  POST /demo/score       the 5-feature demo model, used by the PDF walkthrough UI
  POST /analyze          PDF upload -> risk after every sentence (demo model)
  GET  /                 the UI;  /samples/*.pdf  example claims

Run:  uvicorn demo_server:app --app-dir src --port 8000
"""
import json
import math
import platform
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import explain
from pdf_pipeline import LABELS, analyse

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MAX_PDF_BYTES = 10 * 1024 * 1024

LR = json.loads((MODELS / "lr.json").read_text())
FEATURES = LR["features"]
BASELINE = LR["baseline"]
TIERS = LR["tiers"]
BOOSTER = xgb.Booster()
BOOSTER.load_model(str(MODELS / "model.json"))


def tier(p):
    if p >= TIERS["investigate"]:
        return "investigate"
    return "review" if p >= TIERS["review"] else "fast_track"


def lr_parts(features):
    terms = {}
    for f, m, s, c in zip(FEATURES, LR["mean"], LR["scale"], LR["coef"]):
        terms[f] = c * (float(features[f]) - m) / s
    return terms


def raw_score(features):
    terms = lr_parts(features)
    z = LR["intercept"] + sum(terms.values())
    p_lr = 1 / (1 + math.exp(-z))
    row = pd.DataFrame([[float(features[f]) for f in FEATURES]], columns=FEATURES)
    p_xgb = float(BOOSTER.predict(xgb.DMatrix(row))[0])
    risk = (p_lr + p_xgb) / 2
    return {"risk_score": round(risk, 4), "tier": tier(risk), "p_lr": round(p_lr, 4),
            "p_xgb": round(p_xgb, 4), "z_lr": round(z, 4),
            "lr_terms": {f: round(v, 4) for f, v in terms.items()}}


def clean(claim):
    out = {}
    for f in FEATURES:
        v = claim.get(f, BASELINE[f])
        try:
            out[f] = float(v)
        except (TypeError, ValueError):
            raise HTTPException(422, f"{f} must be a number")
    return out


app = FastAPI(title="we did it: claim risk scoring")


@app.get("/health")
def health():
    return {"status": "ok", "platform": platform.machine(), "xgboost": xgb.__version__}


@app.get("/model/info")
def model_info():
    return {
        "features": FEATURES, "labels": LABELS, "baseline": BASELINE, "tiers": TIERS,
        "lr": {k: LR[k] for k in ("coef", "intercept", "mean", "scale")},
        "trees": json.loads((MODELS / "trees.json").read_text()),
        "metrics": json.loads((MODELS / "metrics.json").read_text()),
        "platform": platform.machine(),
    }


@app.get("/model/final")
def model_final():
    manifest, _, features, reference, *_ = explain._load()
    return {
        "model_id": manifest["model_id"], "features": features,
        "labels": {f: explain.LABELS.get(f, f) for f in features}, "reference": reference,
        "tiers": {"review": manifest["review_min_score"], "investigate": manifest["investigate_min_score"]},
        "impact_unit": "log-odds", "platform": platform.machine(),
    }


@app.post("/score")
def score(claim: dict):
    try:
        return explain.score(claim)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/demo/score")
def demo_score(claim: dict):
    exclude = [f for f in claim.get("exclude", []) if f in FEATURES]
    feats = clean(claim)
    for f in exclude:
        feats[f] = BASELINE[f]  # neutralise the factor
    full = raw_score(feats)
    contributions = []
    for f in FEATURES:
        if f in exclude:
            continue
        base = raw_score({**feats, f: BASELINE[f]})
        contributions.append({"feature": f, "value": feats[f],
                              "impact": round(full["risk_score"] - base["risk_score"], 4)})
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return {**full, "contributions": contributions, "excluded": exclude}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF is larger than 10 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Please upload a PDF file")
    try:
        result = analyse(data, raw_score, BASELINE)
    except Exception as e:  # unreadable or encrypted PDF
        raise HTTPException(422, f"Could not read this PDF: {e}")
    if not result["steps"]:
        raise HTTPException(422, "No text found in this PDF (scanned image PDFs are not supported yet)")
    return {"filename": file.filename, **result}


app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
app.mount("/", StaticFiles(directory=ROOT / "ui", html=True), name="ui")
