"""Regime Agent: MLP neural network that classifies market regime."""

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

import config
from agents.base_agent import BaseAgent

REGIMES = ["trending_up", "trending_down", "ranging", "chop"]

FEATURE_KEYS = [
    "adx", "plus_di", "minus_di",
    "sma_short_slope", "sma_long_slope",
    "vol_ratio", "bb_width", "roc",
]


def heuristic_label(features: dict) -> str:
    """Rule-based regime label used for bootstrapping training labels."""
    adx = features.get("adx", 0)
    slope = features.get("sma_short_slope", 0)
    bb_width = features.get("bb_width", 0)
    vol_ratio = features.get("vol_ratio", 1)

    trending = adx > config.ADX_TREND_THRESHOLD
    if trending and slope > 0.3:
        return "trending_up"
    if trending and slope < -0.3:
        return "trending_down"
    if bb_width < 0.01 or vol_ratio < 0.6:
        return "chop"
    return "ranging"


class RegimeAgent(BaseAgent):
    def __init__(self):
        super().__init__("RegimeAgent")
        self._feature_keys = FEATURE_KEYS
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
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
            regime = heuristic_label(features)
            return {"regime": regime, "confidence": 0.4}

        x_scaled = self._scale(x).reshape(1, -1)
        proba = self.model.predict_proba(x_scaled)[0]
        classes = list(self.model.classes_)
        best_idx = int(np.argmax(proba))
        regime = classes[best_idx]
        confidence = float(proba[best_idx])
        return {"regime": regime, "confidence": confidence}

    def learn(self, features: dict, label: Any) -> None:
        x = self._to_array(features)
        self.samples_seen += 1
        self._update_scaler(x)
        x_scaled = self._scale(x).reshape(1, -1)
        self.model.partial_fit(x_scaled, [label], classes=REGIMES)
