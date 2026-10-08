"""Volatility Agent: determines adaptive stop-loss distance based on current volatility."""

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

import config
from agents.base_agent import BaseAgent

FEATURE_KEYS = [
    "atr", "bb_width", "vol_ratio", "vol_relative", "adx", "rsi", "close",
]


class VolatilityAgent(BaseAgent):
    """Predicts optimal stop-loss distance using volatility features.

    Before warm-up, uses ATR * multiplier as a heuristic.
    Once warm, the MLP learns the relationship between volatility
    conditions and actual adverse price excursion (from the risk label).
    """

    def __init__(self):
        super().__init__("VolatilityAgent")
        self._feature_keys = FEATURE_KEYS
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1,
            batch_size=1,
        )

    def predict(self, features: dict) -> dict[str, Any]:
        atr_val = features.get("atr", 0)
        bb_width = features.get("bb_width", 0)
        close = features.get("close", 0)

        # Heuristic: scale ATR multiplier by volatility regime
        # High bb_width or vol_ratio -> widen stop; low -> tighten
        vol_ratio = features.get("vol_ratio", 1.0)
        vol_scale = np.clip(vol_ratio, 0.5, 2.0)
        heuristic_stop = atr_val * config.RISK_ATR_MULTIPLIER * vol_scale

        if not self.is_warm:
            return {
                "stop_distance": heuristic_stop,
                "volatility_regime": self._classify_vol(vol_ratio, bb_width),
                "confidence": 0.3,
            }

        x = self._to_array(features)
        x_scaled = self._scale(x).reshape(1, -1)
        ml_stop = max(0, float(self.model.predict(x_scaled)[0]))

        # Blend: 70% ML, 30% heuristic for stability
        stop_distance = 0.7 * ml_stop + 0.3 * heuristic_stop

        # Floor: never less than 0.5 * ATR
        stop_distance = max(stop_distance, atr_val * 0.5)

        return {
            "stop_distance": stop_distance,
            "volatility_regime": self._classify_vol(vol_ratio, bb_width),
            "confidence": 0.5 + min(0.5, self.samples_seen / 200),
        }

    def learn(self, features: dict, label: float) -> None:
        x = self._to_array(features)
        self.samples_seen += 1
        self._update_scaler(x)
        x_scaled = self._scale(x).reshape(1, -1)
        self.model.partial_fit(x_scaled, [label])

    @staticmethod
    def _classify_vol(vol_ratio: float, bb_width: float) -> str:
        """Simple volatility regime classification for logging."""
        if vol_ratio > 1.5 or bb_width > 0.03:
            return "high"
        elif vol_ratio < 0.7 and bb_width < 0.015:
            return "low"
        return "normal"
