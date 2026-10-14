"""Freeze the validation-selected forecasting approach."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
COMPARISON = ROOT / "reports" / "model_comparison.json"


def main():
    comparison = json.loads(COMPARISON.read_text())
    feature_hash = hashlib.sha256(FEATURES.read_bytes()).hexdigest()

    if feature_hash != comparison["features_sha256"]:
        raise ValueError("Features changed. Re-run the model comparison.")

    winner = min(comparison["results"], key=lambda row: row["mae"])
    if winner["name"] != "weekly_baseline":
        raise ValueError("This selection script expects the baseline winner.")

    data = pd.read_csv(FEATURES, parse_dates=["timestamp"])
    train = data.loc[data["split"] == "train"]
    if train.empty:
        raise ValueError("Training split is empty.")

    artifact = {
        "model_type": "weekly_baseline",
        "training_median": float(train["target"].median()),
        "training_end": train["timestamp"].max().isoformat(),
        "forecast_horizon_hours": 24,
        "forecast_origin": "start of each calendar day",
        "required_features": ["lag_168", "lag_24"],
        "fallback_order": [
            "same hour last week",
            "same hour yesterday",
            "training target median",
        ],
        "selection_metric": "validation MAE",
        "selected_validation_metrics": winner,
        "features_sha256": feature_hash,
        "comparison_sha256": hashlib.sha256(
            COMPARISON.read_bytes()
        ).hexdigest(),
        "note": (
            "Selected using validation only. Calibration and test scoring "
            "have not been performed. Experimental gradient-boosting "
            "artifact is not the selected forecasting approach."
        ),
    }

    encoded = json.dumps(artifact, indent=2) + "\n"
    model_path = ROOT / "models" / "selected_forecaster.json"
    report_path = ROOT / "reports" / "model_selection.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(encoded, encoding="utf-8")
    report_path.write_text(encoded, encoding="utf-8")

    print(encoded)
    print("Saved models/selected_forecaster.json")


if __name__ == "__main__":
    main()
