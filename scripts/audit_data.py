"""Audit hourly timestamps and rental counts without changing raw data."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "hour.csv"
REPORT = ROOT / "reports" / "data_audit.json"


def main():
    if not SOURCE.exists():
        raise SystemExit("Run python scripts/download_data.py first.")

    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dteday", "hr", "cnt", "casual", "registered"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}")
        rows = list(reader)

    if not rows:
        raise ValueError("The hourly dataset is empty.")

    timestamps = []
    counts = []
    negative_count_rows = 0
    component_mismatch_rows = 0

    for row_number, row in enumerate(rows, start=2):
        hour = int(row["hr"])
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour at CSV row {row_number}.")

        timestamp = (
            datetime.strptime(row["dteday"], "%Y-%m-%d")
            + timedelta(hours=hour)
        )
        total = int(row["cnt"])
        casual = int(row["casual"])
        registered = int(row["registered"])

        timestamps.append(timestamp)
        counts.append(total)
        negative_count_rows += int(min(total, casual, registered) < 0)
        component_mismatch_rows += int(total != casual + registered)

    frequency = Counter(timestamps)
    observed = set(timestamps)
    start, end = min(observed), max(observed)

    expected_hours = int((end - start).total_seconds() / 3600) + 1
    missing = [
        start + timedelta(hours=offset)
        for offset in range(expected_hours)
        if start + timedelta(hours=offset) not in observed
    ]

    # Count distinct recorded hours per calendar date.
    hours_by_day = Counter(timestamp.date() for timestamp in observed)
    duplicate_rows = sum(number - 1 for number in frequency.values())
    null_cells = sum(
        value is None or value.strip() == ""
        for row in rows
        for value in row.values()
    )

    report = {
        "source": "data/raw/hour.csv",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rows": len(rows),
        "first_timestamp": start.isoformat(),
        "last_timestamp": end.isoformat(),
        "timestamps_in_nondecreasing_order": timestamps == sorted(timestamps),
        "duplicate_timestamp_rows_beyond_first": duplicate_rows,
        "expected_hours_between_first_and_last": expected_hours,
        "missing_hours_between_first_and_last": len(missing),
        "first_10_missing_timestamps": [
            timestamp.isoformat() for timestamp in missing[:10]
        ],
        "observed_calendar_days": len(hours_by_day),
        "observed_days_with_fewer_than_24_hours": sum(
            number < 24 for number in hours_by_day.values()
        ),
        "empty_cells": null_cells,
        "minimum_rental_count": min(counts),
        "maximum_rental_count": max(counts),
        "negative_count_rows": negative_count_rows,
        "target_component_mismatch_rows": component_mismatch_rows,
        "note": (
            "Structural audit only; no forecasting model or evaluation. "
            "Raw files unchanged. Timestamps use dataset date/hour labels "
            "without timezone conversion. Missing hours are not assumed "
            "to represent zero rentals."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

    if duplicate_rows or negative_count_rows or component_mismatch_rows or null_cells:
        raise SystemExit(
            "Audit found issues requiring review. See reports/data_audit.json."
        )

    print("\nAudit saved to reports/data_audit.json")


if __name__ == "__main__":
    main()
