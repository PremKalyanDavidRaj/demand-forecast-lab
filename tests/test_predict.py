import numpy as np
import pandas as pd
import pytest

from src.intervals import predict_with_intervals
from src.predict import predict


@pytest.fixture
def artifact():
    return {
        "model_type": "weekly_baseline",
        "training_median": 50.0,
        "interval": {
            "absolute_error_margin": 20.0,
            "nominal_coverage": 0.90,
        },
    }


def test_fallback_order(artifact):
    features = pd.DataFrame({
        "lag_168": [100.0, np.nan, np.nan],
        "lag_24": [80.0, 70.0, np.nan],
    })
    np.testing.assert_array_equal(
        predict(features, artifact), [100.0, 70.0, 50.0]
    )


def test_zero_is_not_missing(artifact):
    features = pd.DataFrame({
        "lag_168": [0.0],
        "lag_24": [70.0],
    })
    np.testing.assert_array_equal(predict(features, artifact), [0.0])


def test_interval_bounds(artifact):
    features = pd.DataFrame({
        "lag_168": [5.0, 100.0],
        "lag_24": [10.0, 90.0],
    })
    point, lower, upper = predict_with_intervals(features, artifact)
    np.testing.assert_array_equal(point, [5.0, 100.0])
    np.testing.assert_array_equal(lower, [0.0, 80.0])
    np.testing.assert_array_equal(upper, [25.0, 120.0])


def test_missing_column(artifact):
    with pytest.raises(ValueError, match="Required features"):
        predict(pd.DataFrame({"lag_24": [10.0]}), artifact)


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf])
def test_invalid_history(artifact, value):
    features = pd.DataFrame({
        "lag_168": [value],
        "lag_24": [10.0],
    })
    with pytest.raises(ValueError, match="finite and nonnegative"):
        predict(features, artifact)


@pytest.mark.parametrize("margin", [-1.0, np.inf, np.nan])
def test_invalid_margin(artifact, margin):
    artifact["interval"]["absolute_error_margin"] = margin
    features = pd.DataFrame({
        "lag_168": [20.0],
        "lag_24": [10.0],
    })
    with pytest.raises(ValueError, match="Interval margin"):
        predict_with_intervals(features, artifact)
