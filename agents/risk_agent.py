"""Risk Agent: MLP neural network that predicts optimal stop-loss distance."""

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

import config
from agents.base_agent import BaseAgent

FEATURE_KEYS = [
    "atr", "adx", "vol_ratio", "bb_width", "close", "rsi",
]


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskAgent")
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
        x = self._to_array(features)
        close = features.get("close", 0)
        atr_val = features.get("atr", 0)

        if not self.is_warm:
            stop_distance = atr_val * config.RISK_ATR_MULTIPLIER
            return {
                "stop_distance": stop_distance,
                "stop_price_long": close - stop_distance,
                "stop_price_short": close + stop_distance,
                "confidence": 0.3,
            }

        x_scaled = self._scale(x).reshape(1, -1)
        stop_distance = max(0, float(self.model.predict(x_scaled)[0]))
        return {
            "stop_distance": stop_distance,
            "stop_price_long": close - stop_distance,
            "stop_price_short": close + stop_distance,
            "confidence": 0.5 + min(0.5, self.samples_seen / 200),
        }

    def learn(self, features: dict, label: float) -> None:
        x = self._to_array(features)
        self.samples_seen += 1
        self._update_scaler(x)
        x_scaled = self._scale(x).reshape(1, -1)
        self.model.partial_fit(x_scaled, [label])
