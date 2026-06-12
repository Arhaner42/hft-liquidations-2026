"""
Training and evaluation orchestration.

Wires together: data loading → targets → features → model/strategy → threshold → scoring.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from strategies.liq_filter.config import (
    TAUS, SYMBOLS, TURNOVER_FLOOR_PER_DAY, FeatureConfig, FittedPipeline,
)
from strategies.liq_filter.scoring import ScoreReport


Symbol = Literal["btcusdt", "ethusdt"]


def _build_features(
    trades: pd.DataFrame,
    bbo: pd.DataFrame,
    liq_binance: pd.DataFrame,
    liq_bybit: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    # Imports deferred so this module loads even before core/ is complete.
    from core.features.liq import LiqPressureFeatures
    from core.features.book import BookFeatures
    from core.features.flow import FlowFeatures
    from core.transforms.direction import direction_relativize
    from core.transforms.normalize import fill_nan_inf
    from core.dataset import DatasetBuilder
    from core.sampling.samplers import EveryTrade

    # s may already exist (added by compute_pnl); recompute from side if absent.
    s = (
        trades["s"].to_numpy()
        if "s" in trades.columns
        else np.where(trades["side"] == "buy", 1, -1)
    )

    builder = DatasetBuilder(
        features=[
            LiqPressureFeatures(halflives_s=cfg.liq_halflives_s),
            BookFeatures(windows_s=cfg.book_windows_s),
            FlowFeatures(windows_s=cfg.flow_windows_s),
        ],
        transforms=[
            lambda df, **_: direction_relativize(df, s),
            lambda df, **_: fill_nan_inf(df),
        ],
        sampler=EveryTrade(),
    )
    return builder.build(trades, bbo, liq_binance=liq_binance, liq_bybit=liq_bybit)


def run_train(
    data_dir: str,
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    feature_config: FeatureConfig | None = None,
    model_params: dict | None = None,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    verbose: bool = False,
) -> FittedPipeline:
    """
    Train on the train split for the given symbols. Produces a FittedPipeline.

    For each symbol:
      1. load_data_with_required_preprocess(data_dir, symbol, split='train')
      2. add_mid → compute_markout → compute_pnl
      3. make_features via DatasetBuilder
      4. For each tau: train_model or store strategy params

    Threshold is NOT fitted here — calibrated at inference time via fit_threshold.
    """
    from core.data import load_data_with_required_preprocess
    from core.targets.markout import add_mid, compute_markout
    from core.targets.pnl import compute_pnl
    from core.model import train_model

    cfg = feature_config or FeatureConfig()
    models: dict = {}

    for symbol in symbols:
        if verbose:
            print(f"[run_train] {symbol}")

        trades, bbo, liq_binance, liq_bybit = load_data_with_required_preprocess(
            data_dir, symbol, split="train"
        )
        bbo = add_mid(bbo)
        trades = compute_markout(trades, bbo)
        trades = compute_pnl(trades)
        features = _build_features(trades, bbo, liq_binance, liq_bybit, cfg)

        if use_ml:
            for tau in TAUS:
                valid = ~trades[f"pnl_{tau}"].isna()
                models[(symbol, tau)] = train_model(
                    features[valid],
                    trades.loc[valid, f"pnl_{tau}"],
                    sample_weight=trades.loc[valid, "w"],
                    model_params=model_params,
                )

    return FittedPipeline(
        feature_config=cfg,
        models=models,
        use_ml=use_ml,
        target_turnover_per_day=target_turnover_per_day,
    )


def run_eval(
    data_dir: str,
    split: str,
    fitted: FittedPipeline,
    symbols: tuple[str, ...] = SYMBOLS,
    verbose: bool = False,
) -> dict[str, dict[int, ScoreReport]]:
    """
    Evaluate on the given split using a pre-trained FittedPipeline.

    Returns {symbol: {tau: ScoreReport}}.
    ONE-SHOT evaluation — do NOT iterate on the threshold to improve val Score.
    """
    from core.data import load_data_with_required_preprocess, compute_num_days
    from core.targets.markout import add_mid, compute_markout
    from core.targets.pnl import compute_pnl
    from core.model import predict
    from strategies.liq_filter.scoring import score_all
    from strategies.liq_filter.threshold import fit_threshold, apply_filter
    from strategies.liq_filter.strategy import strategy_raw_score

    all_reports: dict[str, dict[int, ScoreReport]] = {}

    for symbol in symbols:
        if verbose:
            print(f"[run_eval] {symbol}  split={split}")

        trades, bbo, liq_binance, liq_bybit = load_data_with_required_preprocess(
            data_dir, symbol, split=split
        )
        bbo = add_mid(bbo)
        trades = compute_markout(trades, bbo)
        trades = compute_pnl(trades)
        features = _build_features(trades, bbo, liq_binance, liq_bybit, fitted.feature_config)
        num_days = compute_num_days(trades)
        w = trades["w"].to_numpy(dtype=np.float64)

        f_by_tau: dict[int, np.ndarray] = {}
        for tau in TAUS:
            if fitted.use_ml:
                raw_score = predict(fitted.models[(symbol, tau)], features)
            else:
                raw_score = strategy_raw_score(features, trades)

            edge_mask = trades[f"edge_{tau}"].to_numpy()
            threshold = fit_threshold(raw_score, w, num_days, fitted.target_turnover_per_day)
            f_by_tau[tau] = apply_filter(raw_score, threshold, edge_mask)

        all_reports[symbol] = score_all(trades, f_by_tau, num_days)

    return all_reports


def run_experiment(
    data_dir: str,
    name: str = "experiment",
    use_ml: bool = False,
    symbols: tuple[str, ...] = SYMBOLS,
    target_turnover_per_day: float = TURNOVER_FLOOR_PER_DAY,
    model_params: dict | None = None,
    verbose: bool = False,
) -> tuple[FittedPipeline, dict, pd.DataFrame]:
    """
    Full experiment: train on train split, evaluate on both train and validation.
    Returns (fitted_pipeline, raw_reports_dict, summary_dataframe).
    """
    from strategies.liq_filter.scoring import reports_to_frame

    fitted = run_train(
        data_dir,
        use_ml=use_ml,
        symbols=symbols,
        target_turnover_per_day=target_turnover_per_day,
        model_params=model_params,
        verbose=verbose,
    )

    all_reports: dict = {}
    dfs = []

    for split in ("train", "validation"):
        split_reports = run_eval(data_dir, split, fitted, symbols=symbols, verbose=verbose)
        all_reports[split] = split_reports
        for sym, tau_reports in split_reports.items():
            dfs.append(reports_to_frame(tau_reports, symbol=sym, experiment=f"{name}/{split}"))

    return fitted, all_reports, pd.concat(dfs, ignore_index=True)
