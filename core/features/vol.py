"""
Volatility and regime-context features.

Conditioning layer: modulates how dangerous a given level of liq pressure is
given the current market regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND = 1_000_000


def compute_vol_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    windows_s: tuple[float, ...] = (60.0, 300.0),
    rank_window: int = 1000,
) -> pd.DataFrame:
    """
    Compute volatility and regime features for each trade.

    Output columns:
        rolling_vol_{w}s       — std of log-returns over window (annualized or raw)
        spread_bps_rank_{n}    — rolling rank percentile of spread (0-1)
        depth_rank_{n}         — rolling rank percentile of top-of-book size (0-1)
        vol_rank_{n}           — rolling rank percentile of volatility (0-1)
    """
    bbo = bbo.sort_values("timestamp")
    bbo_ts  = bbo["timestamp"].to_numpy(dtype=np.int64)
    bbo_mid = ((bbo["bid_price"] + bbo["ask_price"]) / 2.0).to_numpy(dtype=np.float64)
    bbo_spread_bps = (
        (bbo["ask_price"] - bbo["bid_price"]) / bbo_mid * 10_000
    ).to_numpy(dtype=np.float64)
    bbo_depth = (bbo["bid_amount"] + bbo["ask_amount"]).to_numpy(dtype=np.float64)

    query_ts = trades["timestamp"].to_numpy(dtype=np.int64)

    # Asof-join: for each trade, get index of last BBO update strictly before it
    idx_now = np.searchsorted(bbo_ts, query_ts, side="left") - 1

    # Build BBO-aligned mid series for log-return computation
    # We compute log-returns on the BBO stream, then asof-join std to each trade.
    log_ret = np.full(len(bbo_ts), np.nan, dtype=np.float64)
    log_ret[1:] = np.log(bbo_mid[1:] / bbo_mid[:-1])

    cols: dict[str, np.ndarray] = {}

    for w_s in windows_s:
        w_us   = int(w_s * US_PER_SECOND)
        vol_at = np.full(len(query_ts), np.nan, dtype=np.float64)

        for q in range(len(query_ts)):
            i = idx_now[q]
            if i < 1:
                continue
            # Find the left boundary of the BBO window
            t_start = bbo_ts[i] - w_us
            j = np.searchsorted(bbo_ts, t_start, side="left")
            window_ret = log_ret[j : i + 1]
            if len(window_ret) >= 2:
                vol_at[q] = np.nanstd(window_ret)

        cols[f"rolling_vol_{w_s:g}s"] = vol_at

    # Rolling rank percentiles (over last `rank_window` trades)
    # Use vectorised approach: compute the feature at each bbo tick, asof-join to trades
    spread_at_trade = np.where(idx_now >= 0, bbo_spread_bps[np.clip(idx_now, 0, len(bbo_ts)-1)], np.nan)
    depth_at_trade  = np.where(idx_now >= 0, bbo_depth[np.clip(idx_now, 0, len(bbo_ts)-1)], np.nan)

    s_spread = pd.Series(spread_at_trade)
    s_depth  = pd.Series(depth_at_trade)

    # vol_rank uses the first vol window
    first_vol_key = f"rolling_vol_{windows_s[0]:g}s"
    s_vol = pd.Series(cols.get(first_vol_key, np.full(len(query_ts), np.nan)))

    cols[f"spread_bps_rank_{rank_window}"] = (
        s_spread.rolling(rank_window, min_periods=1).rank(pct=True).to_numpy()
    )
    cols[f"depth_rank_{rank_window}"] = (
        s_depth.rolling(rank_window, min_periods=1).rank(pct=True).to_numpy()
    )
    cols[f"vol_rank_{rank_window}"] = (
        s_vol.rolling(rank_window, min_periods=1).rank(pct=True).to_numpy()
    )

    return pd.DataFrame(cols, index=trades.index)
