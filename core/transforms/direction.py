"""
Direction-relative feature transform.

Converts absolute BID/ASK and signed features into same-side / opposite-side
relative to the trade's direction (s_i).

Rule: Dir (s_i) is allowed as its own feature. Every other feature must be
expressed relative to the trade's direction, not in absolute terms.
This is enforced as a pipeline contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def direction_relativize(features: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """
    Convert absolute features to direction-relative form.

    Transformations:
        bid_amount, ask_amount → same_side_depth, opp_side_depth
            same_side = ask_amount where s=+1 (taker bought, hit our ask)
            same_side = bid_amount where s=-1 (taker sold, hit our bid)
        bid_amount_delta_*, ask_amount_delta_* → same_side_depth_delta_*, opp_side_depth_delta_*
        signed_volume_* → same_side_flow_* (= s * signed_volume)
        taker_imbalance_* → same_side_taker_imbalance_* (= s * taker_imbalance)
        liq_ewma_{venue}_buy_*, liq_ewma_{venue}_sell_* → liq_ewma_{venue}_same_*, liq_ewma_{venue}_opp_*

    Parameters
    ----------
    features : DataFrame with absolute feature columns
    s        : int array, +1 (taker buy / maker sell) or -1 (taker sell / maker buy)

    Returns
    -------
    DataFrame with absolute columns replaced by direction-relative columns.
    Columns that are already direction-agnostic (spread_bps, imbalance, vol, etc.)
    pass through unchanged.
    """
    df   = features.copy()
    s    = np.asarray(s, dtype=np.float64)
    buy  = s > 0  # boolean mask: True where taker bought

    # --- depth: bid_amount / ask_amount → same_side_depth / opp_side_depth ---
    if "bid_amount" in df.columns and "ask_amount" in df.columns:
        bid = df["bid_amount"].to_numpy(dtype=np.float64)
        ask = df["ask_amount"].to_numpy(dtype=np.float64)
        # s=+1 (taker buy, maker sell): maker liquidity is on ask side → same = ask
        # s=-1 (taker sell, maker buy): maker liquidity is on bid side → same = bid
        df["same_side_depth"] = np.where(buy, ask, bid)
        df["opp_side_depth"]  = np.where(buy, bid, ask)
        df.drop(columns=["bid_amount", "ask_amount"], inplace=True)

    # --- depth deltas ---
    for col in list(df.columns):
        if col.startswith("bid_amount_delta_"):
            suffix  = col[len("bid_amount_delta_"):]
            ask_col = f"ask_amount_delta_{suffix}"
            if ask_col in df.columns:
                bid_d = df[col].to_numpy(dtype=np.float64)
                ask_d = df[ask_col].to_numpy(dtype=np.float64)
                df[f"same_side_depth_delta_{suffix}"] = np.where(buy, ask_d, bid_d)
                df[f"opp_side_depth_delta_{suffix}"]  = np.where(buy, bid_d, ask_d)
                df.drop(columns=[col, ask_col], inplace=True)

    # --- signed flow: signed_volume_* → same_side_flow_* (= s * signed_volume) ---
    for col in list(df.columns):
        if col.startswith("signed_volume_"):
            suffix = col[len("signed_volume_"):]
            df[f"same_side_flow_{suffix}"] = s * df[col].to_numpy(dtype=np.float64)
            df.drop(columns=[col], inplace=True)

    # --- taker imbalance: already relative to market, multiply by s for maker perspective ---
    for col in list(df.columns):
        if col.startswith("taker_imbalance_"):
            suffix = col[len("taker_imbalance_"):]
            df[f"same_side_taker_imbalance_{suffix}"] = s * df[col].to_numpy(dtype=np.float64)
            df.drop(columns=[col], inplace=True)

    # --- all liq buy/sell columns → same/opp (covers ewma, count, time_since) ---
    for col in list(df.columns):
        if not col.startswith("liq_"):
            continue
        for venue in ("binance", "bybit"):
            buy_marker = f"_{venue}_buy"
            if buy_marker in col:
                sell_col = col.replace(f"_{venue}_buy", f"_{venue}_sell")
                same_col = col.replace(f"_{venue}_buy", f"_{venue}_same")
                opp_col  = col.replace(f"_{venue}_buy", f"_{venue}_opp")
                if sell_col in df.columns:
                    buy_vals  = df[col].to_numpy(dtype=np.float64)
                    sell_vals = df[sell_col].to_numpy(dtype=np.float64)
                    # s=+1 (taker buy): same-side liq = buy liquidations (shorts squeezed)
                    df[same_col] = np.where(buy, buy_vals, sell_vals)
                    df[opp_col]  = np.where(buy, sell_vals, buy_vals)
                    df.drop(columns=[col, sell_col], inplace=True)
                break

    return df
