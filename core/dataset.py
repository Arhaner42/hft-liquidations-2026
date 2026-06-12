"""
DatasetBuilder — Orchestrates feature computation, transforms, sampling, and labeling.

This is the central composition point. It wires together:
    raw market data → feature blocks → transforms → sampling → labeling → dataset

Each block is independently testable and reusable across strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.features.base import FeatureBlock
from core.sampling.samplers import Sampler, EveryTrade


@dataclass
class DatasetBuilder:
    """
    Declarative dataset construction pipeline.

    Usage:
        builder = DatasetBuilder(
            features=[LiqPressureFeatures(...), BookFeatures(...), FlowFeatures(...)],
            transforms=[direction_relativize, fill_nan_inf],
            sampler=EveryTrade(),
        )
        dataset = builder.build(trades, bbo, liq_binance, liq_bybit)
    """

    features: list[FeatureBlock] = field(default_factory=list)
    transforms: list[Any] = field(default_factory=list)  # callable(df, **ctx) -> df
    sampler: Sampler = field(default_factory=EveryTrade)

    def build(
        self,
        trades: pd.DataFrame,
        bbo: pd.DataFrame,
        liq_binance: pd.DataFrame | None = None,
        liq_bybit: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the feature matrix.

        Steps:
          1. Compute features from each block → concatenate columns.
          2. Apply transforms in order.
          3. Apply sampler mask → subset rows.
          4. Return the feature DataFrame (same index as sampled trades).

        Target columns (markout, pnl) are NOT computed here — they are added
        by the strategy layer using core.targets.
        """
        kwargs = dict(bbo=bbo, liq_binance=liq_binance, liq_bybit=liq_bybit)

        blocks = [block.compute(trades, **kwargs) for block in self.features]
        if blocks:
            feature_df = pd.concat(blocks, axis=1)
        else:
            feature_df = pd.DataFrame(index=trades.index)

        for transform in self.transforms:
            feature_df = transform(feature_df)

        mask = self.sampler.sample_mask(trades)
        return feature_df.iloc[mask]
