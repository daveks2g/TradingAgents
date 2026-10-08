"""Signal Agent: MLP neural network that determines entry timing (buy/sell/hold)."""

from typing import Any

from sklearn.neural_network import MLPClassifier

from agents.base_agent import BaseAgent

REGIMES = ["trending_up", "trending_down", "ranging", "chop"]
SIGNALS = ["buy", "sell", "hold"]

BASE_FEATURE_KEYS = [
    "rsi", "bb_pct", "roc", "sma_short_slope", "sma_long_slope",
    "adx", "vol_ratio", "atr", "vol_relative",
]

# Full feature list includes one-hot regime + valuation score
FEATURE_KEYS = BASE_FEATURE_KEYS + [f"regime_{r}" for r in REGIMES] + ["valuation_score"]

# Buy/sell samples get boosted weight to counteract hold-dominated labels
SAMPLE_WEIGHTS = {"buy": 3.0, "sell": 3.0, "hold": 1.0}


class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
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

    def _build_features(self, features: dict, regime: str, valuation_score: float) -> dict:
        x = {k: features.get(k, 0.0) for k in BASE_FEATURE_KEYS}
        for r in REGIMES:
            x[f"regime_{r}"] = 1.0 if regime == r else 0.0
        x["valuation_score"] = valuation_score
        return x

    def predict(self, features: dict, regime: str = "ranging",
                valuation_score: float = 0.0) -> dict[str, Any]:
        feat = self._build_features(features, regime, valuation_score)
        x = self._to_array(feat)

        if not self.is_warm:
            return {"signal": "hold", "confidence": 0.0}

        x_scaled = self._scale(x).reshape(1, -1)
        proba = self.model.predict_proba(x_scaled)[0]
        classes = list(self.model.classes_)

        # Get individual probabilities
        p = {c: float(proba[i]) for i, c in enumerate(classes)}
        p_buy = p.get("buy", 0)
        p_sell = p.get("sell", 0)
        p_hold = p.get("hold", 0)

        # Directional confidence: margin of winning class over the next best
        if p_buy > p_sell and p_buy > p_hold:
            return {"signal": "buy", "confidence": p_buy - max(p_sell, p_hold)}
        elif p_sell > p_buy and p_sell > p_hold:
            return {"signal": "sell", "confidence": p_sell - max(p_buy, p_hold)}
        else:
            return {"signal": "hold", "confidence": p_hold}

    def learn(self, features: dict, label: str,
              regime: str = "ranging", valuation_score: float = 0.0) -> None:
        feat = self._build_features(features, regime, valuation_score)
        x = self._to_array(feat)
        self.samples_seen += 1
        self._update_scaler(x)
        x_scaled = self._scale(x).reshape(1, -1)
        repeats = int(SAMPLE_WEIGHTS.get(label, 1.0))
        for _ in range(repeats):
            self.model.partial_fit(x_scaled, [label], classes=SIGNALS)
