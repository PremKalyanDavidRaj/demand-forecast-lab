"""Prepare an hourly timeline without filling missing rental counts."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hour.csv"
DESTINATION = ROOT / "data" / "processed" / "hourly.csv"
REPORT = ROOT / "reports" / "preparation_report.json"


def main():
    if not SOURCE.exists():
        raise SystemExit("Run python scripts/download_data.py first.")

    raw = pd.read_csv(SOURCE)
    required = {"dteday", "hr", "cnt"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    hours = pd.to_numeric(raw["hr"], errors="raise")
    target = pd.to_numeric(raw["cnt"], errors="raise")

    if hours.isna().any() or not (
        hours.between(0, 23) & hours.eq(hours.round())
    ).all():
        raise ValueError("Hours must be integers from 0 to 23.")

    if target.isna().any() or not (
        target.ge(0) & target.eq(target.round())
    ).all():
        raise ValueError("Observed counts must be nonnegative integers.")

    timestamps = (
        pd.to_datetime(raw["dteday"], format="%Y-%m-%d", errors="raise")
        + pd.to_timedelta(hours, unit="h")
    )
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Missing or duplicate timestamps require review.")

    observed = pd.DataFrame({
        "timestamp": timestamps,
        "cnt": target,
    }).set_index("timestamp").sort_index()

    # Calendar-hour grid, using the dataset's original date/hour labels.
    timeline = pd.date_range(
        observed.index.min(),
        observed.index.max(),
        freq="h",
    )
    hourly = observed.reindex(timeline)
    hourly.index.name = "timestamp"
    hourly["target_observed"] = hourly["cnt"].notna()
    hourly["split"] = "train"

    hourly.loc[hourly.index >= "2012-07-01", "split"] = "validation"
    hourly.loc[hourly.index >= "2012-09-01", "split"] = "calibration"
    hourly.loc[hourly.index >= "2012-11-01", "split"] = "test"

    if set(hourly["split"]) != {
        "train", "validation", "calibration", "test"
    }:
        raise ValueError("Dataset does not cover all planned splits.")

    split_summary = {}
    for name in ("train", "validation", "calibration", "test"):
        part = hourly.loc[hourly["split"] == name]
        split_summary[name] = {
            "first_timestamp": part.index.min().isoformat(),
            "last_timestamp": part.index.max().isoformat(),
            "grid_hours": len(part),
            "observed_targets": int(part["target_observed"].sum()),
            "missing_targets": int(part["cnt"].isna().sum()),
        }

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(DESTINATION, na_rep="")

    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(
            DESTINATION.read_bytes()
        ).hexdigest(),
        "original_rows": len(raw),
        "grid_rows": len(hourly),
        "inserted_missing_hours": int(hourly["cnt"].isna().sum()),
        "splits": split_summary,
        "forecast_design": (
            "Predict a calendar day's 24 hourly counts at its start, "
            "assuming all recorded prior-day counts are available."
        ),
        "missing_data_policy": (
            "Missing rental counts remain missing. No zero filling "
            "or interpolation. Missing targets cannot be scored."
        ),
        "feature_policy": (
            "Only timestamp and target history retained. Same-hour casual, "
            "registered, and observed weather are excluded."
        ),
        "note": (
            "Split assignment only; no model fitting or evaluation. "
            "Timestamp labels are not converted to UTC."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("\nSaved data/processed/hourly.csv")


if __name__ == "__main__":
    main()
