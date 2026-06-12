"""
Forward markout computation.

For each trade i and horizon tau, compute m_i(tau) = the forward-fill mid
at time t_i + tau (the last BBO mid with timestamp <= t_i + tau).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND: int = 1_000_000
TAUS: tuple[int, ...] = (30, 120, 300)


def add_mid(bbo: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns to BBO frame:
        mid        = (bid_price + ask_price) / 2
        microprice = (bid_price * ask_amount + ask_price * bid_amount) / (bid_amount + ask_amount)
    """
    bbo = bbo.copy()
    bbo["mid"] = (bbo["bid_price"] + bbo["ask_price"]) / 2.0
    total_amt = bbo["bid_amount"] + bbo["ask_amount"]
    bbo["microprice"] = (
        bbo["bid_price"] * bbo["ask_amount"] + bbo["ask_price"] * bbo["bid_amount"]
    ) / total_amt
    return bbo


def compute_markout(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    taus: tuple[int, ...] = TAUS,
) -> pd.DataFrame:
    """
    For each trade and each tau, compute the forward-fill mid at t_i + tau.

    Implementation:
        For each tau:
          1. lookup_ts = trades.timestamp + tau * US_PER_SECOND
          2. asof-join (backward) lookup_ts against bbo.timestamp → mid value
          3. Where lookup_ts > max(bbo.timestamp), mid = NaN, edge_{tau} = True

    Adds columns to trades:
        mid_{tau}  : float64 (NaN on edge trades)
        edge_{tau} : bool (True = trade falls off BBO range, excluded from scoring)
    """
    if "mid" not in bbo.columns:
        bbo = add_mid(bbo)

    trades = trades.copy()
    bbo_ts  = bbo["timestamp"].to_numpy()
    bbo_mid = bbo["mid"].to_numpy()
    max_bbo_ts = bbo_ts[-1]  # bbo is sorted

    for tau in taus:
        lookup_ts = trades["timestamp"].to_numpy() + tau * US_PER_SECOND
        # searchsorted gives insertion point; -1 gives the last bbo entry <= lookup_ts
        idx = np.searchsorted(bbo_ts, lookup_ts, side="right") - 1
        edge = lookup_ts > max_bbo_ts
        mid_vals = np.where((idx >= 0) & ~edge, bbo_mid[np.clip(idx, 0, len(bbo_mid) - 1)], np.nan)
        trades[f"mid_{tau}"]  = mid_vals
        trades[f"edge_{tau}"] = edge

    return trades
