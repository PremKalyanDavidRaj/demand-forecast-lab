"""Calibrate interval width without evaluating the test period."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.intervals import predict_with_intervals
from src.predict import predict

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
SELECTED = ROOT / "models" / "selected_forecaster.json"
OUTPUT = ROOT / "models" / "calibrated_forecaster.json"
REPORT = ROOT / "reports" / "calibration_metrics.json"


def main():
    if not SELECTED.exists():
        raise SystemExit("Run python -m scripts.select_model first.")

    artifact = json.loads(SELECTED.read_text())
    feature_hash = hashlib.sha256(FEATURES.read_bytes()).hexdigest()
    if feature_hash != artifact["features_sha256"]:
        raise ValueError("Features changed since model selection.")

    data = pd.read_csv(FEATURES, parse_dates=["timestamp"])
    calibration = data.loc[data["split"] == "calibration"].copy()

    if calibration.empty:
        raise ValueError("Calibration split is empty.")
    if not (
        calibration["timestamp"].ge("2012-09-01")
        & calibration["timestamp"].lt("2012-11-01")
    ).all():
        raise ValueError("Unexpected calibration dates.")

    actual = calibration["target"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Calibration targets must be observed and nonnegative.")

    predictions = predict(calibration, artifact)
    errors = np.abs(actual - predictions)
    margin = float(np.quantile(errors, 0.90, method="higher"))

    artifact["interval"] = {
        "method": "pooled absolute-error empirical quantile",
        "nominal_coverage": 0.90,
        "quantile_method": "higher",
        "absolute_error_margin": margin,
        "calibration_start": calibration["timestamp"].min().isoformat(),
        "calibration_end": calibration["timestamp"].max().isoformat(),
        "calibration_rows": len(calibration),
        "scope": "Individual hourly counts; not simultaneous daily coverage.",
    }
    artifact["selected_artifact_sha256"] = hashlib.sha256(
        SELECTED.read_bytes()
    ).hexdigest()
    artifact["note"] = (
        "Point forecast selected using validation MAE. Interval margin "
        "estimated using calibration only. Test period not scored. "
        "Temporal dependence and distribution changes can affect coverage."
    )

    point, lower, upper = predict_with_intervals(calibration, artifact)
    covered = (actual >= lower) & (actual <= upper)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "split": "calibration",
        "rows": len(calibration),
        "nominal_coverage": 0.90,
        "absolute_error_margin": margin,
        "calibration_mae": float(np.mean(np.abs(actual - point))),
        "calibration_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "calibrated_artifact_sha256": hashlib.sha256(
            OUTPUT.read_bytes()
        ).hexdigest(),
        "note": (
            "Coverage measured on the same data used to choose the margin. "
            "This is a calibration diagnostic, not independent evidence "
            "of future coverage. Test period not scored."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("\nSaved models/calibrated_forecaster.json")


if __name__ == "__main__":
    main()
