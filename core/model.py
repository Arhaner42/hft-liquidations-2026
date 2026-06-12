"""
Model training and prediction interface.

Decoupled from feature computation. Accepts a feature matrix and targets,
returns a trained model that can predict on new features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_PARAMS = {
    "objective":       "huber",
    "alpha":           0.9,
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_child_samples": 50,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "n_jobs":          -1,
    "random_state":    42,
}


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series | None = None,
    model_params: dict | None = None,
) -> Any:
    """
    Train a model on features → target.

    Default: LGBMRegressor with Huber loss and sample_weight.
    Returns the fitted model object (supports .predict(features) → np.ndarray).

    Parameters
    ----------
    features      : feature matrix (NaN rows should be pre-dropped by caller)
    target        : regression target (e.g. pnl_{tau})
    sample_weight : per-sample weight (e.g. clipped notional w_i)
    model_params  : override default LightGBM hyperparameters
    """
    from lightgbm import LGBMRegressor

    params = {**_DEFAULT_PARAMS, **(model_params or {})}
    model  = LGBMRegressor(**params)
    model.fit(
        features,
        target,
        sample_weight=sample_weight,
    )
    return model


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Run model prediction. Returns float64 array, higher = better predicted trade."""
    return model.predict(features).astype(np.float64)
