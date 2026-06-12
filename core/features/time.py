"""
Time features.

Time-of-day, day-of-week, and time-to-known-event encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

US_PER_SECOND = 1_000_000
FUNDING_HOURS = (0, 8, 16)  # UTC hours when Binance perpetual funding fires


def compute_time_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time features for each trade.

    Output columns:
        hour_utc          — hour of day (0-23) from timestamp
        minute_utc        — minute of hour (0-59)
        time_to_funding_s — seconds until next Binance funding (00:00, 08:00, 16:00 UTC)
    """
    dt = pd.to_datetime(trades["timestamp"], unit="us", utc=True)

    hour   = dt.dt.hour.to_numpy(dtype=np.float64)
    minute = dt.dt.minute.to_numpy(dtype=np.float64)
    second = dt.dt.second.to_numpy(dtype=np.float64)

    # Seconds elapsed since midnight UTC
    seconds_in_day = hour * 3600.0 + minute * 60.0 + second

    # Funding fires at 0, 8, 16 hours → 0, 28800, 57600 seconds
    funding_seconds = np.array([h * 3600.0 for h in FUNDING_HOURS])
    DAY_S = 86_400.0

    # For each trade, find seconds until the next funding time (wraps past midnight)
    time_to_funding = np.full(len(trades), np.inf, dtype=np.float64)
    for f_s in funding_seconds:
        delta = (f_s - seconds_in_day) % DAY_S
        time_to_funding = np.minimum(time_to_funding, delta)

    return pd.DataFrame(
        {
            "hour_utc":          hour,
            "minute_utc":        minute,
            "time_to_funding_s": time_to_funding,
        },
        index=trades.index,
    )
