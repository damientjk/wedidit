"""Evaluate the saved baseline on validation only; python src/evaluate_thresholds.py."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baseline_thresholds"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model_path = ROOT / "models" / "baseline.joblib"
    metrics_path = ROOT / "models" / "metrics.json"
    val_path = ROOT / "data" / "processed" / "val.csv"
    protected = {p: digest(p) for p in (model_path, metrics_path, val_path)}
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    frame = pd.read_csv(val_path, encoding="utf-8-sig")
    if not frame["split"].eq("val").all():
        raise ValueError("Expected only frozen validation rows")
    y = frame[metadata["target"]]
    if set(y) != {0, 1}:
        raise ValueError("Expected binary validation labels")
    scores = model.predict_proba(frame[metadata["selected_features"]])[:, list(model.classes_).index(1)]
    rows = []
    for step in range(2, 19):
        threshold = step / 20
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": threshold,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "true_positives": int(tp), "false_positives": int(fp),
            "true_negatives": int(tn), "false_negatives": int(fn),
            "claims_flagged": int(tp + fp),
        })
    table = pd.DataFrame(rows)
    ap = float(average_precision_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "thresholds.csv", index=False)
    summary = {
        "dataset": "frozen validation only", "rows": len(frame),
        "fraud_count": int(y.sum()), "fraud_prevalence": float(y.mean()),
        "average_precision": ap, "zero_division": 0,
        "score_note": "Balanced-class logistic scores have not been probability calibrated.",
        "input_sha256": {str(p.relative_to(ROOT)): value for p, value in protected.items()},
        "thresholds": rows,
    }
    # Candidate policy: keep low scores in fast-track and reserve investigation
    # for the highest-scoring group; these are validation-selected, not final.
    tiers = np.where(scores < 0.30, "fast_track", np.where(scores < 0.60, "review", "investigate"))
    summary["candidate_policy"] = {"review_threshold": 0.30, "investigate_threshold": 0.60}
    summary["candidate_tiers"] = {
        tier: {"claims": int((tiers == tier).sum()),
               "fraud": int(y.to_numpy()[tiers == tier].sum())}
        for tier in ("fast_track", "review", "investigate")
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.step(recall, precision, where="post", label=f"Baseline (AP = {ap:.4f})")
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"Fraud prevalence = {y.mean():.2%}")
    for threshold in (0.30, 0.50, 0.60):
        row = table.loc[table.threshold.eq(threshold)].iloc[0]
        ax.scatter(row.recall, row.precision, s=35, zorder=3)
        ax.annotate(f"t={threshold:.2f}", (row.recall, row.precision), xytext=(5, 8), textcoords="offset points")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01),
           title="Logistic regression: frozen validation precision-recall curve")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall.png", dpi=180)
    plt.close(fig)
    assert all(digest(p) == before for p, before in protected.items()), "Protected input changed"
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps({k: v for k, v in summary.items() if k != "thresholds"}, indent=2))


if __name__ == "__main__":
    main()
