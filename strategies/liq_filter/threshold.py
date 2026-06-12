"""
Threshold calibration under the turnover constraint, and filter application.

fit_threshold is label-free (uses only w_i), so it works on the hidden test.
"""

from __future__ import annotations

import numpy as np

from strategies.liq_filter.config import TURNOVER_FLOOR_PER_DAY


def fit_threshold(
    raw_score: np.ndarray,
    w: np.ndarray,
    num_days: float,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
) -> float:
    """
    Pick threshold t* so kept trades (raw_score >= t*) have daily turnover >= target.

    Algorithm:
      1. Sort by raw_score descending.
      2. Accumulate w from top.
      3. t* = score of the marginal trade where cumsum/num_days >= target.
      4. If all trades together don't meet target, return -inf (keep everything).

    Label-free: requires only w_i, not pnl. Valid on hidden test.
    """
    if num_days <= 0:
        return float("-inf")

    order = np.argsort(raw_score)[::-1]
    daily_cumsum = np.cumsum(w[order]) / num_days

    idx = int(np.searchsorted(daily_cumsum, target_turnover_per_day, side="left"))
    if idx >= len(order):
        return float("-inf")

    return float(raw_score[order[idx]])


def apply_filter(
    raw_score: np.ndarray,
    threshold: float,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    f_i = 1 (filter) if raw_score < threshold, else 0 (keep).
    Edge trades (edge_mask=True) forced to f_i = 0 (free turnover, out of score).
    Returns int array of 0/1.
    """
    f = (raw_score < threshold).astype(np.int8)
    if edge_mask is not None:
        f[edge_mask] = 0
    return f
