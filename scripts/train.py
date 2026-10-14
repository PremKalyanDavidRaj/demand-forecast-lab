"""Train on historical data and compare validation forecasts."""

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "features.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

CONFIG = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def main():
    if not SOURCE.exists():
        raise SystemExit("Run python -m scripts.build_features first.")

    data = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()

    if train.empty or validation.empty:
        raise ValueError("Training and validation splits must be nonempty.")
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise ValueError("Training must precede validation.")
    if train["target"].isna().any() or validation["target"].isna().any():
        raise ValueError("Targets must be observed.")

    # Baseline: same hour last week.
    # If missing, use yesterday's same hour, then training median.
    training_median = float(train["target"].median())
    weekly_missing = validation["lag_168"].isna()
    both_missing = weekly_missing & validation["lag_24"].isna()
    baseline = (
        validation["lag_168"]
        .fillna(validation["lag_24"])
        .fillna(training_median)
    )

    print("Training histogram gradient boosting...")
    started = time.perf_counter()
    model = HistGradientBoostingRegressor(**CONFIG)
    model.fit(train[FEATURE_COLUMNS], train["target"])
    training_seconds = time.perf_counter() - started

    # Counts cannot be negative. Apply this consistently at inference too.
    predictions = np.maximum(
        model.predict(validation[FEATURE_COLUMNS]), 0.0
    )

    baseline_metrics = metrics(validation["target"], baseline)
    model_metrics = metrics(validation["target"], predictions)

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "demand_forecaster.joblib"

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_median": training_median,
        "configuration": CONFIG,
        "training_end": train["timestamp"].max().isoformat(),
        "prediction_postprocessing": "clip predictions at zero",
    }, model_path)

    results = validation[["timestamp", "target"]].copy()
    results["baseline_prediction"] = baseline.to_numpy()
    results["model_prediction"] = predictions
    results.to_csv(REPORTS / "validation_predictions.csv", index=False)

    baseline_mae = baseline_metrics["mae"]
    improvement = (
        100 * (baseline_mae - model_metrics["mae"]) / baseline_mae
        if baseline_mae > 0 else None
    )

    report = {
        "evaluation_split": "validation",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "baseline": baseline_metrics,
        "gradient_boosting": model_metrics,
        "mae_improvement_percent": improvement,
        "baseline_weekly_lag_missing_rows": int(weekly_missing.sum()),
        "baseline_training_median_fallback_rows": int(both_missing.sum()),
        "training_seconds": round(training_seconds, 3),
        "configuration": CONFIG,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "features_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "scope": (
            "Daily rolling-origin validation forecasts with a fixed model. "
            "Both methods scored on the same observed target hours. "
            "Calibration and test splits were not scored. "
            "Errors are measured in rentals per hour."
        ),
    }

    (REPORTS / "validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("\nSaved models/demand_forecaster.joblib")


if __name__ == "__main__":
    main()
