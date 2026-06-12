"""
Book / L1 features from Binance BBO stream.

Computes features from the last BBO update strictly before each trade:
  - spread (in bps)
  - top-of-book imbalance
  - microprice deviation from mid
  - depth dynamics (bid/ask amount changes over rolling windows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND = 1_000_000


def _last_asof(
    query_ts: np.ndarray,
    ref_ts: np.ndarray,
    ref_vals: np.ndarray,
) -> np.ndarray:
    """
    For each query_ts, return the value from ref_vals at the last ref_ts < query_ts.
    Strictly causal (allow_exact_matches=False equivalent).
    Returns NaN where no prior reference exists.
    """
    idx = np.searchsorted(ref_ts, query_ts, side="left") - 1
    result = np.full(len(query_ts), np.nan, dtype=np.float64)
    valid = idx >= 0
    result[valid] = ref_vals[idx[valid]]
    return result


def compute_book_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    windows_s: tuple[float, ...] = (1.0, 10.0),
) -> pd.DataFrame:
    """
    Compute book features for each trade.

    Output columns:
        spread_bps      — (ask - bid) / mid * 10000
        imbalance       — (bid_amt - ask_amt) / (bid_amt + ask_amt)
        microprice_dev  — (microprice - mid) / mid * 10000
        bid_amount      — last bid size before trade (absolute, pre-direction-transform)
        ask_amount      — last ask size before trade (absolute, pre-direction-transform)
        bid_amount_delta_{w}s — change in bid size over window
        ask_amount_delta_{w}s — change in ask size over window

    Absolute BID/ASK columns will be converted to direction-relative form
    by core.transforms.direction before reaching the model.
    """
    bbo = bbo.sort_values("timestamp")
    bbo_ts      = bbo["timestamp"].to_numpy(dtype=np.int64)
    bid_price   = bbo["bid_price"].to_numpy(dtype=np.float64)
    ask_price   = bbo["ask_price"].to_numpy(dtype=np.float64)
    bid_amt     = bbo["bid_amount"].to_numpy(dtype=np.float64)
    ask_amt     = bbo["ask_amount"].to_numpy(dtype=np.float64)

    query_ts = trades["timestamp"].to_numpy(dtype=np.int64)

    bid_now = _last_asof(query_ts, bbo_ts, bid_price)
    ask_now = _last_asof(query_ts, bbo_ts, ask_price)
    ba_now  = _last_asof(query_ts, bbo_ts, bid_amt)
    aa_now  = _last_asof(query_ts, bbo_ts, ask_amt)

    mid        = (bid_now + ask_now) / 2.0
    total_amt  = ba_now + aa_now
    microprice = np.where(
        total_amt > 0,
        (bid_now * aa_now + ask_now * ba_now) / total_amt,
        mid,
    )

    cols: dict[str, np.ndarray] = {
        "spread_bps":     (ask_now - bid_now) / mid * 10_000,
        "imbalance":      np.where(total_amt > 0, (ba_now - aa_now) / total_amt, 0.0),
        "microprice_dev": np.where(mid > 0, (microprice - mid) / mid * 10_000, 0.0),
        "bid_amount":     ba_now,
        "ask_amount":     aa_now,
    }

    for w_s in windows_s:
        w_us = int(w_s * US_PER_SECOND)
        lag_ts   = query_ts - w_us
        ba_lag   = _last_asof(lag_ts, bbo_ts, bid_amt)
        aa_lag   = _last_asof(lag_ts, bbo_ts, ask_amt)
        cols[f"bid_amount_delta_{w_s:g}s"] = ba_now - ba_lag
        cols[f"ask_amount_delta_{w_s:g}s"] = aa_now - aa_lag

    return pd.DataFrame(cols, index=trades.index)
