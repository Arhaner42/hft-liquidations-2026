"""
End-to-end pipeline smoke test.
Loads one day of data via predicate pushdown — no full-file reads.
"""

import numpy as np
import pandas as pd

DATA_DIR = "data"
SYMBOL   = "btcusdt"
US = 1_000_000

# Load just one day to keep memory usage small
TEST_START = pd.Timestamp("2025-12-01", tz="UTC")
TEST_END   = pd.Timestamp("2025-12-02", tz="UTC")
START_US   = int(TEST_START.timestamp() * US)
END_US     = int(TEST_END.timestamp()   * US)

print("=== 1. Load data (1-day window via predicate pushdown) ===")
from core.data import _read_parquet, _prepare_frame, BYBIT_LAG_US, LIQ_LOOKBACK_US
from core.data import compute_num_days, detect_symbol

trades    = _prepare_frame(_read_parquet(f"{DATA_DIR}/binance_trades/perp_{SYMBOL}.parquet",        START_US,                   END_US))
bbo       = _prepare_frame(_read_parquet(f"{DATA_DIR}/binance_booktickers/perp_{SYMBOL}.parquet",   START_US,                   END_US))
liq_bin   = _prepare_frame(_read_parquet(f"{DATA_DIR}/binance_liquidations/perp_{SYMBOL}.parquet",  START_US - LIQ_LOOKBACK_US, END_US))
liq_bybit = _prepare_frame(_read_parquet(f"{DATA_DIR}/bybit_liquidations/{SYMBOL}.parquet",         START_US - LIQ_LOOKBACK_US, END_US))

print(f"  trades:    {len(trades):,} rows")
print(f"  bbo:       {len(bbo):,} rows")
print(f"  liq_bin:   {len(liq_bin):,} rows")
print(f"  liq_bybit: {len(liq_bybit):,} rows  (pre-shift)")
print(f"  num_days:  {compute_num_days(trades)}")
print(f"  symbol:    {detect_symbol(trades)}")

# Bybit shift check before applying
raw_ts0 = int(liq_bybit["timestamp"].iloc[0])
liq_bybit = liq_bybit.copy()
liq_bybit["timestamp"] = liq_bybit["timestamp"] + BYBIT_LAG_US
shifted_ts0 = int(liq_bybit["timestamp"].iloc[0])
print(f"  Bybit shift: diff={shifted_ts0 - raw_ts0} (expect {BYBIT_LAG_US})")
assert shifted_ts0 - raw_ts0 == BYBIT_LAG_US, "Bybit shift mismatch!"

print("\n=== 2. Targets (markout + PnL) ===")
from core.targets.markout import add_mid, compute_markout
from core.targets.pnl import compute_pnl

bbo_with_mid = add_mid(bbo)
trades = compute_markout(trades, bbo_with_mid)
trades = compute_pnl(trades)

for tau in (30, 120, 300):
    pnl_col  = f"pnl_{tau}"
    edge_col = f"edge_{tau}"
    valid    = trades[pnl_col].notna()
    print(f"  tau={tau:3d}s  valid={valid.sum():,}  edge={trades[edge_col].sum():,}  "
          f"mean_pnl={trades.loc[valid, pnl_col].mean():.4f} bps")

print("\n=== 3. PnL formula spot-check ===")
from core.targets.pnl import compute_pnl as _cpnl

row1 = pd.DataFrame({"timestamp": [0], "ticker": ["perp:btcusdt"],
                     "side": ["buy"], "price": [100000.0], "amount": [1.0],
                     "mid_30": [100050.0], "edge_30": [False]})
got1 = _cpnl(row1, taus=(30,))["pnl_30"].iloc[0]
print(f"  buy, mid↑50: {got1:.2f} bps  (expect -4.50)")
assert abs(got1 - (-4.5)) < 1e-9, f"Wrong: {got1}"

row2 = pd.DataFrame({"timestamp": [0], "ticker": ["perp:btcusdt"],
                     "side": ["buy"], "price": [100000.0], "amount": [1.0],
                     "mid_30": [99950.0], "edge_30": [False]})
got2 = _cpnl(row2, taus=(30,))["pnl_30"].iloc[0]
print(f"  buy, mid↓50: {got2:.2f} bps  (expect +5.50)")
assert abs(got2 - 5.5) < 1e-9, f"Wrong: {got2}"

print("\n=== 4. Liq features ===")
from core.features.liq import compute_liq_features

liq_feats = compute_liq_features(trades, liq_bin, liq_bybit)
print(f"  shape: {liq_feats.shape}")
nan_frac = liq_feats.isna().mean().mean()
print(f"  NaN fraction: {nan_frac:.4f}  (expect 0)")
# Causality: EWMA at first trade must not see any events at or after its timestamp
t0 = trades["timestamp"].iloc[0]
events_before_t0 = (liq_bin["timestamp"] < t0).sum()
print(f"  liq events strictly before t0: {events_before_t0}")

print("\n=== 5. Book features ===")
from core.features.book import compute_book_features

book_feats = compute_book_features(trades, bbo)
print(f"  shape: {book_feats.shape}")
print(f"  spread_bps mean={book_feats['spread_bps'].mean():.4f}  min={book_feats['spread_bps'].min():.6f}")
print(f"  imbalance  in [-1,1]: {((book_feats['imbalance'] >= -1) & (book_feats['imbalance'] <= 1)).all()}")
nan_frac = book_feats.isna().mean().mean()
print(f"  NaN fraction: {nan_frac:.4f}")

print("\n=== 6. Flow features ===")
from core.features.flow import compute_flow_features

flow_feats = compute_flow_features(trades)
print(f"  shape: {flow_feats.shape}")
imb_ok = ((flow_feats["taker_imbalance_5s"] >= -1) & (flow_feats["taker_imbalance_5s"] <= 1)).all()
print(f"  taker_imbalance_5s in [-1,1]: {imb_ok}")

print("\n=== 7. Direction transform ===")
from core.transforms.direction import direction_relativize

all_feats = pd.concat([liq_feats, book_feats, flow_feats], axis=1)
s = np.where(trades["side"] == "buy", 1, -1)
dir_feats = direction_relativize(all_feats, s)

print(f"  bid_amount removed: {'bid_amount' not in dir_feats.columns}")
print(f"  ask_amount removed: {'ask_amount' not in dir_feats.columns}")
print(f"  same_side_depth present: {'same_side_depth' in dir_feats.columns}")
liq_abs_cols = [c for c in dir_feats.columns if "_buy_" in c or "_sell_" in c]
print(f"  absolute liq buy/sell cols remaining: {liq_abs_cols}  (expect [])")

print("\n=== 8. fill_nan_inf ===")
from core.transforms.normalize import fill_nan_inf

clean = fill_nan_inf(dir_feats)
print(f"  NaN count: {clean.isna().sum().sum()}  (expect 0)")
print(f"  Inf count: {np.isinf(clean.values).sum()}  (expect 0)")

print("\n=== 9. Samplers ===")
from core.sampling.samplers import EveryTrade, VolumeThreshold, TimeInterval

m1 = EveryTrade().sample_mask(trades)
m2 = VolumeThreshold(100_000).sample_mask(trades)
m3 = TimeInterval(10.0).sample_mask(trades)
print(f"  EveryTrade:          {m1.sum():,} / {len(trades):,}")
print(f"  VolumeThreshold(100k): {m2.sum():,} / {len(trades):,}")
print(f"  TimeInterval(10s):   {m3.sum():,} / {len(trades):,}")
assert m2.sum() < m1.sum(), "VolumeThreshold should select fewer than all trades"
assert m3.sum() < m1.sum(), "TimeInterval should select fewer than all trades"

print("\n=== ALL CHECKS PASSED ===")
