"""Empirical prediction intervals for hourly rental counts."""

import numpy as np

from src.predict import predict


def predict_with_intervals(features, artifact):
    point = predict(features, artifact)
    margin = float(artifact["interval"]["absolute_error_margin"])

    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Interval margin must be finite and nonnegative.")

    lower = np.maximum(point - margin, 0.0)
    upper = point + margin
    return point, lower, upper
