"""
Normalization transforms: zscore, clamp, winsorize, NaN cleanup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series, window: int = 1000) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    roll  = series.rolling(window, min_periods=2)
    return (series - roll.mean()) / roll.std()


def clamp(series: pd.Series, lo: float = -5.0, hi: float = 5.0) -> pd.Series:
    """Clip values to [lo, hi]."""
    return series.clip(lo, hi)


def winsorize(series: pd.Series, quantile: float = 0.01) -> pd.Series:
    """Clip values to [q, 1-q] quantiles."""
    lo = series.quantile(quantile)
    hi = series.quantile(1 - quantile)
    return series.clip(lo, hi)


def fill_nan_inf(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Replace NaN and inf with fill_value. Applied as the last transform step."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
