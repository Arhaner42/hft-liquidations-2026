"""
Signed trade flow features.

Leading indicator of cascade onset: aggressive directional flow building
before liquidations fire. Computed from the Binance trades stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND = 1_000_000
NOTIONAL_CAP = 100_000.0


def _rolling_sum_count(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    event_vals: np.ndarray,
    window_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each query_ts, compute the sum and count of event_vals
    in the window [query_ts - window_us, query_ts).

    Uses searchsorted(side='left') to ensure strict causality:
    events at exactly query_ts are excluded.
    """
    right = np.searchsorted(event_ts, query_ts,            side="left")
    left  = np.searchsorted(event_ts, query_ts - window_us, side="left")

    # cumsum trick for O(n) window sums
    cum_vals  = np.concatenate([[0.0], np.cumsum(event_vals)])
    cum_count = np.arange(len(event_ts) + 1, dtype=np.float64)

    sums   = cum_vals[right]  - cum_vals[left]
    counts = cum_count[right] - cum_count[left]
    return sums, counts


def compute_flow_features(
    trades: pd.DataFrame,
    windows_s: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    """
    Compute signed taker flow features for each trade.

    Output columns (for each window):
        signed_volume_{w}s     — Σ s_j * min(notional_j, 100k) over window
        total_volume_{w}s      — Σ min(notional_j, 100k) over window
        taker_imbalance_{w}s   — signed_volume / total_volume (in [-1, 1])
        trade_count_{w}s       — number of trades in window

    These are in absolute signed terms. Direction-relative conversion
    (same_side_flow = s_i * signed_volume) is in core.transforms.direction.
    """
    trades = trades.sort_values("timestamp")
    ev_ts  = trades["timestamp"].to_numpy(dtype=np.int64)

    notional = (trades["price"] * trades["amount"]).clip(upper=NOTIONAL_CAP).to_numpy(dtype=np.float64)
    s        = np.where(trades["side"] == "buy", 1.0, -1.0)
    signed   = s * notional

    query_ts = ev_ts  # features are computed for every trade against itself (causal window)

    cols: dict[str, np.ndarray] = {}
    for w_s in windows_s:
        w_us = int(w_s * US_PER_SECOND)
        sv, cnt = _rolling_sum_count(query_ts, ev_ts, signed,  w_us)
        tv, _   = _rolling_sum_count(query_ts, ev_ts, notional, w_us)

        safe_tv = np.where(tv > 0, tv, 1.0)  # avoid 0/0; np.where evaluates both branches eagerly
        imb = np.where(tv > 0, sv / safe_tv, 0.0)
        cols[f"signed_volume_{w_s:g}s"]   = sv
        cols[f"total_volume_{w_s:g}s"]    = tv
        cols[f"taker_imbalance_{w_s:g}s"] = np.clip(imb, -1.0, 1.0)  # clip for fp cancellation
        cols[f"trade_count_{w_s:g}s"]     = cnt

    return pd.DataFrame(cols, index=trades.index)
