"""Create feature data without fitting models or scoring predictions."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS, LAG_HOURS, build_features

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "hourly.csv"
OUTPUT = ROOT / "data" / "processed" / "features.csv"
REPORT = ROOT / "reports" / "feature_report.json"


def main():
    if not SOURCE.exists():
        raise SystemExit("Run python scripts/prepare_data.py first.")

    hourly = pd.read_csv(
        SOURCE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    features = build_features(hourly)
    features["target"] = hourly["cnt"]
    features["split"] = hourly["split"]

    # Remove the initial period before a full 14-day history can exist.
    first_eligible = hourly.index.min() + pd.Timedelta(
        hours=max(LAG_HOURS)
    )
    features = features.loc[features.index >= first_eligible]

    # Missing targets cannot train a model or be used to score forecasts.
    missing_targets = int(features["target"].isna().sum())
    features = features.loc[features["target"].notna()].copy()

    # Keep missing lag values for a model supporting missing features.
    features.to_csv(OUTPUT, na_rep="")

    report = {
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "feature_columns": FEATURE_COLUMNS,
        "lag_hours": list(LAG_HOURS),
        "rows": len(features),
        "rows_by_split": {
            name: int(number)
            for name, number in features["split"].value_counts().items()
        },
        "missing_target_rows_excluded_after_warmup": missing_targets,
        "missing_values_by_feature": {
            name: int(features[name].isna().sum())
            for name in FEATURE_COLUMNS
        },
        "evaluation_design": (
            "Daily rolling forecast origins. Earlier observed validation "
            "or test counts may become history for later forecast days. "
            "No actual counts from the forecast day enter its features."
        ),
        "note": (
            "No model fitting or scoring performed. Missing lag values "
            "are retained. This is not a single forecast of the entire "
            "validation or test period."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
