"""Evaluate the frozen forecasting approach on the final test period."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.intervals import predict_with_intervals

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
ARTIFACT = ROOT / "models" / "calibrated_forecaster.json"
CALIBRATION_REPORT = ROOT / "reports" / "calibration_metrics.json"
REPORTS = ROOT / "reports"


def main():
    if not ARTIFACT.exists():
        raise SystemExit("Run python -m scripts.calibrate first.")

    artifact_bytes = ARTIFACT.read_bytes()
    artifact = json.loads(artifact_bytes)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    calibration_report = json.loads(CALIBRATION_REPORT.read_text())

    if artifact_hash != calibration_report["calibrated_artifact_sha256"]:
        raise ValueError("Artifact differs from the recorded calibration.")

    feature_hash = hashlib.sha256(FEATURES.read_bytes()).hexdigest()
    if feature_hash != artifact["features_sha256"]:
        raise ValueError("Features changed since model selection.")

    data = pd.read_csv(FEATURES, parse_dates=["timestamp"])
    test = data.loc[data["split"] == "test"].sort_values("timestamp").copy()

    if test.empty or test["timestamp"].duplicated().any():
        raise ValueError("Test timestamps must be nonempty and unique.")
    if not (
        test["timestamp"].ge("2012-11-01")
        & test["timestamp"].lt("2013-01-01")
    ).all():
        raise ValueError("Unexpected test dates.")
    if pd.Timestamp(artifact["interval"]["calibration_end"]) >= (
        test["timestamp"].min()
    ):
        raise ValueError("Calibration must end before test evaluation.")

    actual = test["target"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Test targets must be observed and nonnegative.")

    point, lower, upper = predict_with_intervals(test, artifact)
    error = actual - point
    covered = (actual >= lower) & (actual <= upper)

    expected = pd.date_range(
        "2012-11-01", "2012-12-31 23:00", freq="h"
    )
    missing = expected.difference(pd.DatetimeIndex(test["timestamp"]))

    predictions = test[["timestamp", "target"]].copy()
    predictions["prediction"] = point
    predictions["lower"] = lower
    predictions["upper"] = upper
    predictions["covered"] = covered
    predictions["absolute_error"] = np.abs(error)
    predictions["hour"] = predictions["timestamp"].dt.hour

    by_hour = predictions.groupby("hour").agg(
        rows=("target", "size"),
        mae=("absolute_error", "mean"),
        coverage=("covered", "mean"),
    )

    report = {
        "evaluation_split": "test",
        "model_type": artifact["model_type"],
        "first_timestamp": test["timestamp"].min().isoformat(),
        "last_timestamp": test["timestamp"].max().isoformat(),
        "expected_calendar_hours": len(expected),
        "observed_target_hours_scored": len(test),
        "missing_target_hours_not_scored": len(missing),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "nominal_interval_coverage": artifact["interval"]["nominal_coverage"],
        "observed_interval_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "below_lower_bound_rows": int(np.sum(actual < lower)),
        "above_upper_bound_rows": int(np.sum(actual > upper)),
        "weekly_lag_missing_rows": int(test["lag_168"].isna().sum()),
        "training_median_fallback_rows": int(
            (test["lag_168"].isna() & test["lag_24"].isna()).sum()
        ),
        "model_sha256": artifact_hash,
        "features_sha256": feature_hash,
        "scope": (
            "Frozen forecast rule and interval margin. Daily rolling-origin "
            "evaluation; earlier observed test counts can become history "
            "for later days. No same-day actuals enter that day's features. "
            "Missing targets excluded. Coverage is pooled over observed "
            "hours, not simultaneous coverage of an entire day."
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    by_hour.to_csv(REPORTS / "test_metrics_by_hour.csv")
    (REPORTS / "test_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nHourly breakdown saved to reports/test_metrics_by_hour.csv")


if __name__ == "__main__":
    main()
