"""Delayed labeling logic: generates training labels from future price data."""

from collections import deque
from typing import Any

import numpy as np

import config
from agents.regime_agent import heuristic_label


class DelayedLabeler:
    """Buffers features and generates labels once enough future bars are available."""

    def __init__(self):
        self._pending: deque[dict[str, Any]] = deque()
        # Price lookup by bar_index for accurate forward returns
        self._closes: dict[int, float] = {}
        self._highs: dict[int, float] = {}
        self._lows: dict[int, float] = {}

    def record_prices(self, bar_index: int, close: float, high: float, low: float):
        """Record bar prices for forward-looking label generation."""
        self._closes[bar_index] = close
        self._highs[bar_index] = high
        self._lows[bar_index] = low

    def cleanup_prices(self, current_bar: int):
        """Remove old price entries no longer needed."""
        cutoff = current_bar - config.LOOKBACK_WINDOW
        for d in (self._closes, self._highs, self._lows):
            stale = [k for k in d if k < cutoff]
            for k in stale:
                del d[k]

    def enqueue(self, bar_index: int, features: dict, regime: str, valuation_score: float):
        self._pending.append({
            "bar_index": bar_index,
            "features": features,
            "regime": regime,
            "valuation_score": valuation_score,
            "close": features["close"],
            "high": features["high"],
            "low": features["low"],
            "atr": features["atr"],
        })

    def generate_labels(self) -> list[dict[str, Any]]:
        """Check pending items and generate labels for those with enough future data."""
        results = []
        max_delay = max(config.SIGNAL_FORWARD_BARS, config.VALUATION_REVERT_BARS)

        while self._pending:
            item = self._pending[0]
            entry_idx = item["bar_index"]
            forward_idx = entry_idx + config.SIGNAL_FORWARD_BARS
            val_idx = entry_idx + config.VALUATION_REVERT_BARS

            # Need forward close prices to exist
            if forward_idx not in self._closes:
                break

            self._pending.popleft()
            entry_close = item["close"]
            entry_atr = item["atr"]

            # --- Regime label (heuristic from the features at that time) ---
            regime_label = heuristic_label(item["features"])

            # --- Signal label (from forward returns using exact bar) ---
            forward_close = self._closes[forward_idx]
            forward_return_pct = ((forward_close - entry_close) / entry_close) * 100
            if forward_return_pct > config.SIGNAL_THRESHOLD_PCT:
                signal_label = "buy"
            elif forward_return_pct < -config.SIGNAL_THRESHOLD_PCT:
                signal_label = "sell"
            else:
                signal_label = "hold"

            # --- Valuation label (mean-reversion outcome) ---
            val_close = self._closes.get(val_idx, forward_close)
            val_return_pct = ((val_close - entry_close) / entry_close) * 100
            valuation_label = -np.tanh(val_return_pct / 0.15)
            valuation_label = max(-1.0, min(1.0, valuation_label))

            # --- Risk label (optimal stop from actual adverse excursion) ---
            adverse_low = entry_close
            adverse_high = entry_close
            for i in range(entry_idx, entry_idx + max_delay + 1):
                if i in self._lows:
                    adverse_low = min(adverse_low, self._lows[i])
                if i in self._highs:
                    adverse_high = max(adverse_high, self._highs[i])
            adverse_long = max(0, entry_close - adverse_low)
            adverse_short = max(0, adverse_high - entry_close)
            optimal_stop = max(adverse_long, adverse_short)

            results.append({
                "features": item["features"],
                "regime": item["regime"],
                "valuation_score": item["valuation_score"],
                "regime_label": regime_label,
                "signal_label": signal_label,
                "valuation_label": valuation_label,
                "risk_label": optimal_stop,
            })

        return results
