"""Historical daily forecasting service."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.features import build_features
from src.intervals import predict_with_intervals

ROOT = Path(__file__).resolve().parents[1]


class ForecastService:
    def __init__(self):
        artifact_path = ROOT / "models" / "calibrated_forecaster.json"
        history_path = ROOT / "data" / "processed" / "hourly.csv"
        report_path = ROOT / "reports" / "calibration_metrics.json"

        if not artifact_path.exists() or not history_path.exists():
            raise ValueError(
                "Prepare the hourly data and run model selection "
                "and calibration first."
            )

        artifact_bytes = artifact_path.read_bytes()
        self.artifact = json.loads(artifact_bytes)
        report = json.loads(report_path.read_text())

        self.model_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if self.model_sha256 != report["calibrated_artifact_sha256"]:
            raise ValueError("Calibrated artifact hash does not match.")

        self.history = pd.read_csv(
            history_path,
            parse_dates=["timestamp"],
            index_col="timestamp",
        )[["cnt"]].sort_index()

        if self.history.empty or self.history.index.has_duplicates:
            raise ValueError("History must be nonempty with unique timestamps.")

        # Serve dates strictly after the calibration period.
        self.first_date = (
            pd.Timestamp(self.artifact["interval"]["calibration_end"])
            .normalize() + pd.Timedelta(days=1)
        )
        self.last_date = (
            self.history.index.max().normalize() + pd.Timedelta(days=1)
        )

    def forecast(self, forecast_date: date):
        origin = pd.Timestamp(forecast_date)

        if not self.first_date <= origin <= self.last_date:
            raise ValueError(
                f"Choose a date from {self.first_date.date()} "
                f"through {self.last_date.date()}. "
                "This app uses historical data, not a live demand feed."
            )

        # Include enough history for every lag and 24 forecast hours.
        grid = pd.date_range(
            origin - pd.Timedelta(hours=336),
            origin + pd.Timedelta(hours=23),
            freq="h",
        )

        # Remove forecast-day and future observations BEFORE feature building.
        available_history = self.history.loc[self.history.index < origin]
        window = available_history.reindex(grid)
        features = build_features(window).loc[origin:]

        if len(features) != 24:
            raise ValueError("Expected exactly 24 forecast hours.")

        point, lower, upper = predict_with_intervals(
            features, self.artifact
        )

        rows = []
        for index, timestamp in enumerate(features.index):
            if pd.notna(features.loc[timestamp, "lag_168"]):
                source = "same_hour_last_week"
            elif pd.notna(features.loc[timestamp, "lag_24"]):
                source = "same_hour_yesterday"
            else:
                source = "training_median"

            rows.append({
                "timestamp": timestamp.isoformat(),
                "prediction": float(point[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "prediction_source": source,
            })

        return {
            "forecast_date": origin.date().isoformat(),
            "model_type": self.artifact["model_type"],
            "model_sha256": self.model_sha256,
            "nominal_interval_coverage": (
                self.artifact["interval"]["nominal_coverage"]
            ),
            "forecast": rows,
            "note": (
                "Historical daily forecast using only earlier recorded counts. "
                "Timestamp labels follow the original dataset without timezone "
                "conversion. Intervals target individual hourly counts; "
                "90% future coverage is not guaranteed."
            ),
        }
