"""Valuation Agent: MLP neural network that scores overbought (+1) / oversold (-1)."""

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from agents.base_agent import BaseAgent

FEATURE_KEYS = [
    "rsi", "bb_pct", "roc", "vol_relative", "vol_ratio",
]


class ValuationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ValuationAgent")
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

        if not self.is_warm:
            rsi = features.get("rsi", 50)
            bb_pct = features.get("bb_pct", 0.5)
            # Combine RSI and Bollinger %B for stronger signal
            rsi_score = (rsi - 50) / 30       # ±0.67 at RSI 30/70
            bb_score = (bb_pct - 0.5) / 0.3   # ±0.67 at %B 0.2/0.8
            score = (rsi_score + bb_score) / 2
            return {"valuation_score": max(-1.0, min(1.0, score)), "confidence": 0.3}

        x_scaled = self._scale(x).reshape(1, -1)
        raw = float(self.model.predict(x_scaled)[0])
        score = max(-1.0, min(1.0, raw))
        confidence = 0.5 + 0.5 * abs(score)
        return {"valuation_score": score, "confidence": confidence}

    def learn(self, features: dict, label: float) -> None:
        x = self._to_array(features)
        self.samples_seen += 1
        self._update_scaler(x)
        x_scaled = self._scale(x).reshape(1, -1)
        self.model.partial_fit(x_scaled, [label])
