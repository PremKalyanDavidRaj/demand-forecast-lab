"""Compare a small, predefined set of forecasting approaches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.train import CONFIG, metrics
from src.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "features.csv"
REPORTS = ROOT / "reports"


def baseline(frame, fallback):
    return (
        frame["lag_168"]
        .fillna(frame["lag_24"])
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def main():
    data = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    train = data.loc[data["split"] == "train"].copy()
    validation = data.loc[data["split"] == "validation"].copy()
    recent = train.loc[train["timestamp"] >= "2012-01-01"].copy()

    if train.empty or recent.empty or validation.empty:
        raise ValueError("A required split is empty.")
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise ValueError("Training must precede validation.")

    fallback = float(train["target"].median())
    validation_baseline = baseline(validation, fallback)
    actual = validation["target"].to_numpy()

    results = [{
        "name": "weekly_baseline",
        "training_rows": 0,
        **metrics(actual, validation_baseline),
    }]

    predictions = validation[["timestamp", "target"]].copy()
    predictions["weekly_baseline"] = validation_baseline

    experiments = [
        ("original_direct", train, False),
        ("recent_2012_direct", recent, False),
        ("weekly_residual", train, True),
    ]

    for name, training, use_residuals in experiments:
        print(f"Training {name}...")
        target = training["target"].to_numpy(dtype=float)

        if use_residuals:
            target = target - baseline(training, fallback)

        model = HistGradientBoostingRegressor(**CONFIG)
        model.fit(training[FEATURE_COLUMNS], target)
        predicted = model.predict(validation[FEATURE_COLUMNS])

        if use_residuals:
            predicted = validation_baseline + predicted

        predicted = np.maximum(predicted, 0.0)
        predictions[name] = predicted

        results.append({
            "name": name,
            "training_rows": len(training),
            **metrics(actual, predicted),
        })

    results.sort(key=lambda item: item["mae"])
    report = {
        "evaluation_split": "validation",
        "selection_metric": "mae",
        "validation_rows": len(validation),
        "configuration": CONFIG,
        "baseline_fallback_training_median": fallback,
        "features_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "results": results,
        "note": (
            "Development comparison on the same validation period. "
            "Not an independent final benchmark. Saved model unchanged. "
            "Calibration and test splits not scored."
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "model_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions.to_csv(
        REPORTS / "comparison_predictions.csv", index=False
    )

    print("\nValidation comparison:")
    print(pd.DataFrame(results).to_string(index=False, float_format="%.3f"))
    print("\nYour saved model has NOT been replaced.")


if __name__ == "__main__":
    main()
