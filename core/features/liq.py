"""
Liquidation pressure features.

Primary signal for cascade detection. Computes EWMA of liquidation notional
across venues (Binance, Bybit) and sides (buy, sell), plus intensity metrics
(event count, time since last liq).

Bybit timestamps must be pre-shifted (+200ms) before calling these functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)
US_PER_SECOND = 1_000_000


def _ewma_event_state_at(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    event_vals: np.ndarray,
    halflife_us: float,
) -> np.ndarray:
    """
    For each query timestamp, compute the EWMA state of an event stream
    using only events strictly before the query time.

    The EWMA decays as exp(-(t_query - t_event) / tau) where tau = halflife_us / ln(2).

    Parameters
    ----------
    query_ts   : sorted int64 array of trade timestamps (microseconds)
    event_ts   : sorted int64 array of event timestamps (microseconds, shifted if Bybit)
    event_vals : float64 array of event values (e.g. notional = price * amount)
    halflife_us: EWMA half-life in microseconds

    Returns
    -------
    float64 array of length len(query_ts), the EWMA state at each query time.
    """
    tau = halflife_us / _LN2
    result = np.zeros(len(query_ts), dtype=np.float64)

    n_events = len(event_ts)
    if n_events == 0:
        return result

    # Two-pointer: for each query, sum contributions of all prior events.
    # Use cumulative representation to avoid O(n*m): maintain a running
    # weighted sum and decay it to the current query time.
    #
    # At each step the accumulated state S is the EWMA sum at reference time t_ref.
    # Moving to a new query time t_q: S_new = S * exp(-(t_q - t_ref) / tau) + new_events
    #
    # Process queries and events together in timestamp order.

    e = 0  # event pointer
    S = 0.0
    t_ref = 0  # reference time for current accumulated state

    for q in range(len(query_ts)):
        t_q = query_ts[q]

        # Decay existing state to current query time
        if S != 0.0 and t_ref < t_q:
            S *= np.exp(-(t_q - t_ref) / tau)
            t_ref = t_q

        # Absorb all events strictly before t_q
        while e < n_events and event_ts[e] < t_q:
            t_e = event_ts[e]
            # Decay S from t_ref to t_e, add event, then S stays at t_e
            if S != 0.0:
                S *= np.exp(-(t_e - t_ref) / tau)
            S += event_vals[e]
            t_ref = t_e
            e += 1

        # Decay to t_q if we moved t_ref to some event time < t_q
        if t_ref < t_q and S != 0.0:
            S *= np.exp(-(t_q - t_ref) / tau)
            t_ref = t_q

        result[q] = S

    return result


def _liq_event_count(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    window_us: int,
) -> np.ndarray:
    """
    Count liquidation events in [query_ts - window_us, query_ts) for each query.
    Uses searchsorted for O(n log n) performance.
    """
    right = np.searchsorted(event_ts, query_ts, side="left")
    left  = np.searchsorted(event_ts, query_ts - window_us, side="left")
    return (right - left).astype(np.float64)


def _time_since_last_liq(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
) -> np.ndarray:
    """
    Microseconds since the last liquidation event strictly before each query time.
    Returns np.inf where no prior event exists.
    """
    idx = np.searchsorted(event_ts, query_ts, side="left") - 1
    result = np.full(len(query_ts), np.inf, dtype=np.float64)
    valid = idx >= 0
    result[valid] = (query_ts[valid] - event_ts[idx[valid]]).astype(np.float64)
    return result


def compute_liq_features(
    trades: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
    halflives_s: tuple[float, ...] = (1.0, 5.0, 30.0),
) -> pd.DataFrame:
    """
    Compute all liquidation pressure features for each trade.

    Output columns (for each venue, side, halflife):
        liq_ewma_{venue}_{side}_{hl}s     — EWMA of liq notional
        liq_count_{venue}_{side}_{window}s — event count in window
        liq_time_since_{venue}_{side}      — seconds since last liq

    All values are in absolute buy/sell terms. Direction-relative conversion
    (same_side / opp_side) is handled by core.transforms.direction.
    """
    query_ts = trades["timestamp"].to_numpy(dtype=np.int64)
    cols: dict[str, np.ndarray] = {}

    venues = [
        ("binance", liq_binance),
        ("bybit",   liq_bybit),
    ]

    for venue_name, liq_df in venues:
        for side in ("buy", "sell"):
            mask     = liq_df["side"] == side
            ev_df    = liq_df[mask].sort_values("timestamp")
            ev_ts    = ev_df["timestamp"].to_numpy(dtype=np.int64)
            ev_vals  = (ev_df["price"] * ev_df["amount"]).to_numpy(dtype=np.float64)

            for hl_s in halflives_s:
                hl_us = hl_s * US_PER_SECOND
                col   = f"liq_ewma_{venue_name}_{side}_{hl_s:g}s"
                cols[col] = _ewma_event_state_at(query_ts, ev_ts, ev_vals, hl_us)

            # Count windows match the halflife windows for consistency
            for w_s in halflives_s:
                w_us = int(w_s * US_PER_SECOND)
                col  = f"liq_count_{venue_name}_{side}_{w_s:g}s"
                cols[col] = _liq_event_count(query_ts, ev_ts, w_us)

            col = f"liq_time_since_{venue_name}_{side}"
            cols[col] = _time_since_last_liq(query_ts, ev_ts) / US_PER_SECOND  # → seconds

    return pd.DataFrame(cols, index=trades.index)
