"""
Data loading and mandatory preprocessing.

Responsible for:
  - reading parquet files per symbol
  - casting timestamps to int64 microseconds
  - shifting Bybit liquidation timestamps by +200ms
  - sorting all frames by timestamp
  - optional date-range filtering for train/val splits
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

Symbol = Literal["btcusdt", "ethusdt"]

US_PER_SECOND: int = 1_000_000
BYBIT_LAG_US: int = 200_000  # +200 ms
LIQ_LOOKBACK_US: int = 5 * 60 * US_PER_SECOND  # 5 min cold-start buffer

SPLIT_RANGES: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train":      (pd.Timestamp("2025-12-01", tz="UTC"), pd.Timestamp("2026-02-01", tz="UTC")),
    "validation": (pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
}


def _read_parquet(path: str, start_us: int | None = None, end_us: int | None = None) -> pd.DataFrame:
    """
    Read a parquet file with optional timestamp predicate pushdown.
    Avoids loading the entire file when only a date range is needed.
    Relies on row-group min/max statistics (files are sorted by timestamp).
    """
    filters = []
    if start_us is not None:
        filters.append(("timestamp", ">=", start_us))
    if end_us is not None:
        filters.append(("timestamp", "<", end_us))
    return pq.read_table(path, filters=filters or None).to_pandas()


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw parquet frame:
      - ensure timestamp is int64
      - lowercase string columns (side, ticker)
      - sort by timestamp ascending
    Idempotent.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("side", "ticker"):
        if col in df.columns:
            df[col] = df[col].str.lower()
    return df.sort_values("timestamp", ignore_index=True)


def load_data_with_required_preprocess(
    data_dir: str,
    symbol: Symbol,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 4 frames for one symbol and run mandatory preprocessing.

    Steps:
      1. Read parquet files for the given symbol.
      2. If split is given, filter to the date range BEFORE shifting Bybit.
      3. Shift liq_bybit.timestamp += BYBIT_LAG_US.
      4. Sort each frame by timestamp ascending.
      5. For liq data, include a lookback buffer (~5 min before split start)
         to avoid cold-start on EWMA features.

    Returns (trades, bbo, liq_binance, liq_bybit).
    liq_bybit timestamps are ALREADY shifted.
    """
    start_us: int | None = None
    end_us:   int | None = None
    liq_start_us: int | None = None

    if split is not None:
        start_ts, end_ts = SPLIT_RANGES[split]
        start_us     = int(start_ts.timestamp() * US_PER_SECOND)
        end_us       = int(end_ts.timestamp()   * US_PER_SECOND)
        liq_start_us = start_us - LIQ_LOOKBACK_US  # 5-min cold-start buffer for liq EWMA

    # Predicate pushdown: only rows in the split range are read from disk
    trades    = _prepare_frame(_read_parquet(f"{data_dir}/binance_trades/perp_{symbol}.parquet",        start_us,     end_us))
    bbo       = _prepare_frame(_read_parquet(f"{data_dir}/binance_booktickers/perp_{symbol}.parquet",   start_us,     end_us))
    liq_bin   = _prepare_frame(_read_parquet(f"{data_dir}/binance_liquidations/perp_{symbol}.parquet",  liq_start_us, end_us))
    liq_bybit = _prepare_frame(_read_parquet(f"{data_dir}/bybit_liquidations/{symbol}.parquet",         liq_start_us, end_us))

    # Shift Bybit timestamps AFTER date filtering
    liq_bybit["timestamp"] = liq_bybit["timestamp"] + BYBIT_LAG_US

    return trades, bbo, liq_bin, liq_bybit


def compute_num_days(trades: pd.DataFrame) -> float:
    """
    Number of distinct calendar dates (UTC) spanned by the trades frame.
    Uses trades["timestamp"] (int64 microseconds).
    """
    return pd.to_datetime(trades["timestamp"], unit="us", utc=True).dt.date.nunique()


def detect_symbol(trades: pd.DataFrame) -> Symbol:
    """
    Infer symbol from the ticker column.
    'perp:btcusdt' -> 'btcusdt', 'perp:ethusdt' -> 'ethusdt'.
    Raises ValueError on mixed or unknown tickers.
    """
    tickers = trades["ticker"].unique()
    if len(tickers) != 1:
        raise ValueError(f"Mixed tickers in trades frame: {tickers}")
    raw = tickers[0]
    symbol = raw.removeprefix("perp:")
    if symbol not in ("btcusdt", "ethusdt"):
        raise ValueError(f"Unknown ticker: {raw}")
    return symbol  # type: ignore[return-value]
