"""Prediction logic for the selected seasonal baseline."""

import numpy as np
import pandas as pd


def predict(features, artifact):
    if artifact["model_type"] != "weekly_baseline":
        raise ValueError("Unsupported model type.")

    required = {"lag_168", "lag_24"}
    if not required.issubset(features.columns):
        raise ValueError(f"Required features: {sorted(required)}")

    fallback = float(artifact["training_median"])
    if not np.isfinite(fallback) or fallback < 0:
        raise ValueError("Invalid training fallback.")

    weekly = pd.to_numeric(features["lag_168"], errors="raise")
    daily = pd.to_numeric(features["lag_24"], errors="raise")

    for values in (weekly, daily):
        observed = values.dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError("Historical counts must be finite and nonnegative.")

    return weekly.fillna(daily).fillna(fallback).to_numpy(dtype=float)
