"""
Sampling strategies: every trade, volume-triggered, time-triggered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

US_PER_SECOND = 1_000_000


class Sampler(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        """
        Return a boolean mask of length len(trades).
        True = this trade is a datapoint in the supervised dataset.
        """
        ...


class EveryTrade(Sampler):
    """Use every trade as a datapoint. Default for the liquidation filter task."""

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        return np.ones(len(trades), dtype=bool)


class VolumeThreshold(Sampler):
    """
    Emit a datapoint every time cumulative traded notional exceeds a threshold.
    The emitted row is the trade that crossed the threshold.
    """

    def __init__(self, notional_threshold: float = 100_000.0):
        self.notional_threshold = notional_threshold

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        notional = (trades["price"] * trades["amount"]).to_numpy(dtype=np.float64)
        mask = np.zeros(len(trades), dtype=bool)
        cumsum = 0.0
        for i, n in enumerate(notional):
            cumsum += n
            if cumsum >= self.notional_threshold:
                mask[i] = True
                cumsum = 0.0
        return mask


class TimeInterval(Sampler):
    """Emit a datapoint every N seconds (the last trade in each interval)."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s

    def sample_mask(self, trades: pd.DataFrame) -> np.ndarray:
        ts = trades["timestamp"].to_numpy(dtype=np.int64)
        interval_us = int(self.interval_s * US_PER_SECOND)
        buckets = ts // interval_us
        mask = np.zeros(len(trades), dtype=bool)
        # Mark the last trade in each bucket
        # A trade is "last in its bucket" if it's the final row with that bucket value
        # Efficient: compare each element's bucket to the next element's bucket
        if len(buckets) == 0:
            return mask
        # Last trade overall is always emitted
        mask[-1] = True
        # Emit where bucket changes (last in bucket is where next row has different bucket)
        mask[:-1] = buckets[:-1] != buckets[1:]
        return mask
