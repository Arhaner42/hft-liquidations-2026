"""
Scoring harness for the liquidation filter task.

Computes Score(τ) = PnL_kept(τ) - PnL_all(τ) and related diagnostics.
This is application-specific — core/ does not know about ScoreReport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.liq_filter.config import TAUS


@dataclass
class ScoreReport:
    """Unified report for one (symbol, tau)."""
    tau: int
    score: float             # PnL_kept - PnL_all (main metric)
    pnl_all: float           # baseline, constant
    pnl_kept: float          # NaN if no valid kept trades
    pnl_filtered: float      # NaN if no valid filtered trades
    kept_turnover_per_day: float
    filtered_turnover_per_day: float
    constraint_ok: bool
    n_trades: int
    n_valid: int
    n_kept: int
    n_filtered: int
    n_edge: int


def score_one(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    num_days: float,
    tau: int,
) -> ScoreReport:
    """
    Compute ScoreReport for one tau.

    Formulas (over valid trades only, where pnl is not NaN):
        PnL_all      = Σ w·pnl         / Σ w
        PnL_kept     = Σ (1-f)·w·pnl   / Σ (1-f)·w
        PnL_filtered = Σ f·w·pnl       / Σ f·w
        Score        = PnL_kept - PnL_all

    Division-by-zero: returns NaN for the affected metric.
    """
    from strategies.liq_filter.config import TURNOVER_FLOOR_PER_DAY

    n_trades = len(pnl)
    valid = ~np.isnan(pnl)
    n_valid = int(valid.sum())
    n_edge = n_trades - n_valid
    n_kept = int((f == 0).sum())
    n_filtered = int((f == 1).sum())

    vw, vp, vf = w[valid], pnl[valid], f[valid]

    def _wavg(vals, weights):
        denom = weights.sum()
        return float((weights * vals).sum() / denom) if denom > 0 else float("nan")

    pnl_all = _wavg(vp, vw)
    pnl_kept = _wavg(vp[vf == 0], vw[vf == 0])
    pnl_filtered = _wavg(vp[vf == 1], vw[vf == 1])

    if np.isnan(pnl_kept) or np.isnan(pnl_all):
        score = float("nan")
    else:
        score = pnl_kept - pnl_all

    kept_turnover = float((w * (1 - f)).sum() / num_days)
    filtered_turnover = float((w * f).sum() / num_days)

    return ScoreReport(
        tau=tau,
        score=score,
        pnl_all=pnl_all,
        pnl_kept=pnl_kept,
        pnl_filtered=pnl_filtered,
        kept_turnover_per_day=kept_turnover,
        filtered_turnover_per_day=filtered_turnover,
        constraint_ok=kept_turnover >= TURNOVER_FLOOR_PER_DAY,
        n_trades=n_trades,
        n_valid=n_valid,
        n_kept=n_kept,
        n_filtered=n_filtered,
        n_edge=n_edge,
    )


def score_all(
    trades_with_pnl: pd.DataFrame,
    f_by_tau: dict[int, np.ndarray],
    num_days: float,
) -> dict[int, ScoreReport]:
    """Run score_one for all taus. Unified output format."""
    w = trades_with_pnl["w"].to_numpy(dtype=np.float64)
    return {
        tau: score_one(
            trades_with_pnl[f"pnl_{tau}"].to_numpy(dtype=np.float64),
            w,
            f,
            num_days,
            tau,
        )
        for tau, f in f_by_tau.items()
    }


def reports_to_frame(
    reports: dict[int, ScoreReport],
    symbol: str = "",
    experiment: str = "",
) -> pd.DataFrame:
    """Convert ScoreReport dict to a summary DataFrame for display."""
    rows = []
    for tau, r in reports.items():
        rows.append({
            "symbol": symbol,
            "experiment": experiment,
            "tau": r.tau,
            "score": r.score,
            "pnl_all": r.pnl_all,
            "pnl_kept": r.pnl_kept,
            "pnl_filtered": r.pnl_filtered,
            "kept_turnover_per_day": r.kept_turnover_per_day,
            "filtered_turnover_per_day": r.filtered_turnover_per_day,
            "constraint_ok": r.constraint_ok,
            "n_trades": r.n_trades,
            "n_valid": r.n_valid,
            "n_kept": r.n_kept,
            "n_filtered": r.n_filtered,
            "n_edge": r.n_edge,
        })
    return pd.DataFrame(rows)
