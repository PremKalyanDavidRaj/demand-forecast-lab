import numpy as np
import pandas as pd
import pytest

from src.features import LAG_HOURS, build_features


@pytest.fixture
def hourly():
    index = pd.date_range("2011-01-01", periods=24 * 21, freq="h")
    return pd.DataFrame(
        {"cnt": np.arange(len(index), dtype=float)},
        index=index,
    )


def test_current_and_future_targets_cannot_change_daily_features(hourly):
    original = build_features(hourly)
    forecast_day = pd.Timestamp("2011-01-18")

    changed = hourly.copy()
    changed.loc[changed.index >= forecast_day, "cnt"] = 999999
    recomputed = build_features(changed)

    day_mask = (
        (hourly.index >= forecast_day)
        & (hourly.index < forecast_day + pd.Timedelta(days=1))
    )
    pd.testing.assert_frame_equal(
        original.loc[day_mask],
        recomputed.loc[day_mask],
    )


@pytest.mark.parametrize("lag", LAG_HOURS)
def test_lags_reference_exact_historical_hours(hourly, lag):
    features = build_features(hourly)
    timestamp = pd.Timestamp("2011-01-18 23:00")
    source_time = timestamp - pd.Timedelta(hours=lag)

    assert source_time < timestamp.normalize()
    assert features.loc[timestamp, f"lag_{lag}"] == hourly.loc[
        source_time, "cnt"
    ]


def test_missing_history_stays_missing(hourly):
    source_time = pd.Timestamp("2011-01-17 12:00")
    hourly.loc[source_time, "cnt"] = np.nan

    features = build_features(hourly)

    assert pd.isna(features.loc["2011-01-18 12:00", "lag_24"])


def test_gap_in_hourly_grid_is_rejected(hourly):
    broken = hourly.drop(hourly.index[30])

    with pytest.raises(ValueError, match="Timeline has gaps"):
        build_features(broken)


def test_duplicate_timestamps_are_rejected(hourly):
    duplicated = pd.concat([hourly, hourly.iloc[[-1]]])

    with pytest.raises(ValueError, match="no duplicates"):
        build_features(duplicated)


def test_reversed_timeline_is_rejected(hourly):
    with pytest.raises(ValueError, match="sorted"):
        build_features(hourly.iloc[::-1])


def test_negative_demand_is_rejected(hourly):
    hourly.iloc[10, 0] = -1

    with pytest.raises(ValueError, match="nonnegative"):
        build_features(hourly)


def test_shorter_history_preserves_features_when_lags_available(hourly):
    full = build_features(hourly)
    shorter = build_features(hourly.loc["2011-01-02":])
    timestamp = pd.Timestamp("2011-01-18 12:00")

    pd.testing.assert_series_equal(
        full.loc[timestamp],
        shorter.loc[timestamp],
    )
