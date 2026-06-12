"""
Final submission function — called on the hidden test.

make_filter() is self-contained: it embeds pre-trained model/params
and calibrates the threshold on the test data via turnover (label-free).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.liq_filter.config import TAUS, FittedPipeline


# Populated by run_train() before submission. Baked into module scope.
FITTED: FittedPipeline | None = None


def make_filter(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """
    FINAL submission function.

    Accepts 4 frames (same schemas as public files; liq_bybit arrives UNSHIFTED).
    Returns { 30: arr_30, 120: arr_120, 300: arr_300 },
    each arr is np.ndarray of length len(trades) with values 0 or 1.

    Internal flow:
      1. Detect symbol, shift Bybit +200ms, sort.
      2. Compute features via DatasetBuilder (causal).
      3. For each tau: raw_score from FITTED model/strategy.
      4. fit_threshold on test data's w_i (label-free).
      5. apply_filter → 0/1 array.
    """
    if FITTED is None:
        raise RuntimeError("FITTED pipeline not set. Call run_train() first.")

    from core.data import detect_symbol, compute_num_days, BYBIT_LAG_US
    from core.targets.markout import add_mid
    from core.model import predict
    from strategies.liq_filter.train import _build_features
    from strategies.liq_filter.threshold import fit_threshold, apply_filter
    from strategies.liq_filter.strategy import strategy_raw_score

    # Defensive copies so we don't mutate caller's frames.
    trades = trades.copy().sort_values("timestamp").reset_index(drop=True)
    bbo = bbo.copy().sort_values("timestamp").reset_index(drop=True)
    liq_binance = liq_binance.copy().sort_values("timestamp").reset_index(drop=True)
    liq_bybit = liq_bybit.copy()
    liq_bybit["timestamp"] += BYBIT_LAG_US
    liq_bybit = liq_bybit.sort_values("timestamp").reset_index(drop=True)

    symbol = detect_symbol(trades)
    # s is needed for direction_relativize; no compute_pnl on hidden test.
    trades["s"] = np.where(trades["side"] == "buy", 1, -1)

    bbo = add_mid(bbo)
    num_days = compute_num_days(trades)
    features = _build_features(trades, bbo, liq_binance, liq_bybit, FITTED.feature_config)

    # Clipped notional weight — label-free, used only for threshold calibration.
    w = (trades["price"] * trades["amount"]).clip(upper=100_000.0).to_numpy(dtype=np.float64)

    result: dict[int, np.ndarray] = {}
    for tau in TAUS:
        if FITTED.use_ml:
            raw_score = predict(FITTED.models[(symbol, tau)], features)
        else:
            raw_score = strategy_raw_score(features, trades)

        threshold = fit_threshold(raw_score, w, num_days, FITTED.target_turnover_per_day)
        result[tau] = apply_filter(raw_score, threshold)

    return result
