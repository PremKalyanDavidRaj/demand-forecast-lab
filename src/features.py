"""Features for forecasts issued at the start of each calendar day."""

import numpy as np
import pandas as pd

LAG_HOURS = (24, 48, 168, 336)

FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "trend_days",
    "lag_24",
    "lag_48",
    "lag_168",
    "lag_336",
]


def build_features(hourly):
    """Return features on a complete hourly grid.

    Input must have a DatetimeIndex and a cnt column.
    Missing historical counts remain NaN.
    """
    if "cnt" not in hourly.columns:
        raise ValueError("Input must contain a cnt column.")

    index = hourly.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Input must use a DatetimeIndex.")
    if len(index) == 0 or index.hasnans:
        raise ValueError("Timeline must be nonempty with valid timestamps.")
    if index.tz is not None:
        raise ValueError("Use the dataset's original timezone-naive labels.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Timeline must be sorted and contain no duplicates.")
    if not index.equals(index.floor("h")):
        raise ValueError("Timestamps must fall on exact hours.")

    expected = pd.date_range(index.min(), index.max(), freq="h")
    if not index.equals(expected):
        raise ValueError(
            "Timeline has gaps. Run scripts/prepare_data.py first."
        )

    counts = pd.to_numeric(hourly["cnt"], errors="raise")
    observed = counts.dropna()
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Observed counts must be finite and nonnegative.")

    features = pd.DataFrame(index=index)

    features["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)

    # Fixed epoch keeps the feature consistent across different input windows.
    features["trend_days"] = (
        index.normalize() - pd.Timestamp("2011-01-01")
    ).days

    forecast_origins = index.normalize()

    for lag in LAG_HOURS:
        source_times = index - pd.Timedelta(hours=lag)

        # Every lag must precede midnight on the forecast date.
        if not (source_times < forecast_origins).all():
            raise ValueError(f"Lag {lag} uses unavailable same-day data.")

        # Row shifting represents hours only because we checked the grid.
        features[f"lag_{lag}"] = counts.shift(lag)

    return features[FEATURE_COLUMNS]
