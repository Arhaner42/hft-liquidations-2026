"""
Heuristic benchmark: EWMA liquidation pressure → raw score.

The non-ML baseline that the ML model must beat.
Score = -(same-side liquidation pressure), so higher = safer trade.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def strategy_raw_score(features: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    """
    Heuristic raw score: higher = better trade (higher expected pnl, keep it).

    Simplest version: score = -(same-side liq pressure at preferred halflife).
    Uses direction-relative features (same_side_liq_ewm_*), not absolute buy/sell.

    Returns float64 array of length len(features).
    """
    same_cols = [c for c in features.columns if re.match(r"liq_ewma_\w+_same_[\d.]+s$", c)]
    if not same_cols:
        raise ValueError(
            "No same-side liq EWMA columns found. "
            "Ensure direction_relativize has been applied before calling strategy_raw_score."
        )

    hl_map: dict[float, list[str]] = {}
    for col in same_cols:
        m = re.search(r"_([\d.]+)s$", col)
        if m:
            hl_map.setdefault(float(m.group(1)), []).append(col)

    cols = hl_map[max(hl_map)]
    pressure = features[cols].sum(axis=1).to_numpy(dtype=np.float64)
    return -pressure
