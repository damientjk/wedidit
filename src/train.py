"""Train the real-data logistic regression baseline: python src/train.py.

Uses the frozen training/validation CSVs; never reads the test CSV. Running
main replaces models/metrics.json (previously used by the synthetic demo).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TARGET = "FraudFound_P"
NUMERIC_FEATURES = ["Deductible"]
CATEGORICAL_FEATURES = [
    "Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed",
    "WeekOfMonthClaimed", "DayOfWeekClaimed", "Make", "AccidentArea",
    "VehicleCategory", "BasePolicy", "AgentType", "VehiclePrice",
    "DriverRating", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "NumberOfCars",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_frame(frame, name, expected_split=None):
    required = FEATURES + [TARGET, "claim_id"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    if frame.empty or frame[TARGET].isna().any() or set(frame[TARGET]) != {0, 1}:
        raise ValueError(f"{name}: target must contain both binary classes 0 and 1")
    if frame["claim_id"].isna().any() or not frame["claim_id"].is_unique:
        raise ValueError(f"{name}: claim_id must be present and unique")
    if expected_split and "split" in frame and not frame["split"].eq(expected_split).all():
        raise ValueError(f"{name}: contains rows outside the {expected_split} split")


def load_data(data_dir=DATA_DIR):
    train_path, val_path = data_dir / "train.csv", data_dir / "val.csv"
    if train_path.exists() and val_path.exists():
        train = pd.read_csv(train_path, encoding="utf-8-sig")
        val = pd.read_csv(val_path, encoding="utf-8-sig")
        strategy = "existing frozen train/val CSVs"
    elif (data_dir / "claims_clean.csv").exists():
        # Honor the frozen assignments, including when per-split files are absent.
        pool = pd.read_csv(data_dir / "claims_clean.csv", encoding="utf-8-sig")
        if ("split" not in pool or pool["split"].isna().any()
                or not set(pool["split"]) <= {"train", "val", "test"}):
            raise ValueError("Cannot identify held-out test rows safely in claims_clean.csv")
        train = pool.loc[pool["split"].eq("train")].copy()
        val = pool.loc[pool["split"].eq("val")].copy()
        strategy = "frozen train/val assignments from claims_clean.csv"
        if val.empty:
            validate_frame(train, "training pool", "train")
            train, val = train_test_split(
                train, test_size=0.2, stratify=train[TARGET], random_state=42,
            )
            strategy = "stratified 80/20 of frozen training pool; random_state=42"
    elif train_path.exists():
        # Only split the known training pool: never draw from the held-out test.
        pool = pd.read_csv(train_path, encoding="utf-8-sig")
        validate_frame(pool, "training pool", "train")
        train, val = train_test_split(
            pool, test_size=0.2, stratify=pool[TARGET], random_state=42,
        )
        strategy = "stratified 80/20 of train.csv; random_state=42"
    else:
        raise FileNotFoundError("No safe training pool found; test.csv is never used")

    validate_frame(train, "train", "train")
    # Fallback validation rows inherit their original training-pool split value.
    validate_frame(val, "validation", None if strategy.startswith("stratified") else "val")
    if set(train["claim_id"]) & set(val["claim_id"]):
        raise ValueError("Training and validation claim IDs overlap")
    return train, val, strategy


def build_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ], remainder="drop")),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])


def main(data_dir=DATA_DIR, model_dir=MODEL_DIR):
    train, val, strategy = load_data(Path(data_dir))
    pipeline = build_pipeline()
    pipeline.fit(train[FEATURES], train[TARGET])
    probabilities = pipeline.predict_proba(val[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y = val[TARGET]
    metrics = {
        "model_type": "LogisticRegression",
        "data": "Yu Heng cleaned vehicle claims",
        "target": TARGET,
        "label_mapping": {"0": "non-fraud", "1": "fraud"},
        "split_strategy": strategy,
        "sample_counts": {"train": len(train), "validation": len(val)},
        "class_counts": {
            name: {str(label): int((frame[TARGET] == label).sum()) for label in (0, 1)}
            for name, frame in (("train", train), ("validation", val))
        },
        "selected_features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_parameters": {
            "max_iter": 1000, "class_weight": "balanced", "random_state": 42,
        },
        "threshold": 0.5,
        "validation": {
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            "confusion_matrix_layout": "rows=actual, columns=predicted; labels=[0,1]",
        },
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "baseline.joblib")
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


if __name__ == "__main__":
    main()
