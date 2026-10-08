"""Orchestrator: coordinates agents, data pipeline, and delayed learning loop."""

import csv

import config
from agents.regime_agent import RegimeAgent
from agents.signal_agent import SignalAgent
from agents.valuation_agent import ValuationAgent
from agents.risk_agent import RiskAgent
from agents.volatility_agent import VolatilityAgent
from data.features import compute_features
from data.pipeline import BarBuffer
from utils.labels import DelayedLabeler
from utils.logging import DecisionLogger, setup_logger
from utils.performance import PredictionTracker


class Orchestrator:
    def __init__(self, csv_path=None):
        self.buffer = BarBuffer(csv_path or config.CSV_PATH)
        self.regime_agent = RegimeAgent()
        self.signal_agent = SignalAgent()
        self.valuation_agent = ValuationAgent()
        self.risk_agent = RiskAgent()
        self.volatility_agent = VolatilityAgent()
        self.labeler = DelayedLabeler()
        self.tracker = PredictionTracker()
        self.logger = setup_logger()
        self.decision_log = DecisionLogger()
        self.bar_count = 0
        self._scaled_params: dict | None = None
        self._learn_count = 0
        self._reward_risk_ratio = 1.0  # neutral default; overridden by NinjaTrader via bars.csv
        self._target_mode = "UseRR"  # "UseRR" or "UseML", read from bar CSV
        self._agents = [self.regime_agent, self.signal_agent, self.valuation_agent,
                        self.risk_agent, self.volatility_agent]
        self._load_models()

    def _detect_and_scale(self, bar: dict):
        """Auto-detect bar period from CSV and scale config parameters once."""
        # Read RR ratio and target mode from every bar (NT can change them)
        rr = bar.get("reward_risk_ratio", 0)
        if rr > 0:
            self._reward_risk_ratio = rr
        tm = bar.get("target_mode", "")
        if isinstance(tm, str) and tm in ("UseRR", "UseML"):
            self._target_mode = tm

        if self.bar_count <= 2:
            self.logger.info(
                f"Bar config: rr_raw={bar.get('reward_risk_ratio', 'MISSING')} "
                f"tm_raw={bar.get('target_mode', 'MISSING')} -> "
                f"RR={self._reward_risk_ratio:.4f} mode={self._target_mode}"
            )

        if self._scaled_params is not None:
            return
        bar_minutes = bar.get("bar_period_minutes", config.BASE_BAR_MINUTES)
        # Non-time bar types (Renko, Range, Tick) report 0 — use defaults
        if bar_minutes <= 0:
            bar_minutes = config.BASE_BAR_MINUTES
        if bar_minutes != config.BASE_BAR_MINUTES:
            self._scaled_params = config.scale_for_bar_period(bar_minutes)
            config.SMA_SHORT = self._scaled_params["sma_short"]
            config.SMA_LONG = self._scaled_params["sma_long"]
            config.RSI_PERIOD = self._scaled_params["rsi_period"]
            config.ATR_PERIOD = self._scaled_params["atr_period"]
            config.BBANDS_PERIOD = self._scaled_params["bbands_period"]
            config.ADX_PERIOD = self._scaled_params["adx_period"]
            config.REGIME_WARMUP_BARS = self._scaled_params["regime_warmup_bars"]
            config.SIGNAL_FORWARD_BARS = self._scaled_params["signal_forward_bars"]
            config.VALUATION_REVERT_BARS = self._scaled_params["valuation_revert_bars"]
            config.SIGNAL_THRESHOLD_PCT = self._scaled_params["signal_threshold_pct"]
            self.logger.info(f"Detected {bar_minutes}-min bars. Scaled params: {self._scaled_params}")
        else:
            self._scaled_params = {}
            self.logger.info(f"Detected {bar_minutes}-min bars. Using default parameters.")
        self.logger.info(f"Target mode: {self._target_mode}, RR ratio: {self._reward_risk_ratio:.2f}")

    def _process_bar(self, bar: dict) -> dict | None:
        """Process a single new bar through all agents."""
        self.bar_count += 1
        # The bar's own timestamp, echoed back as bar_time so NinjaTrader can match the signal to the
        # exact bar it was computed from.
        self._current_bar_time = str(bar.get("datetime", "")).strip()
        self._detect_and_scale(bar)

        close = bar.get("close", 0)
        high = bar.get("high", close)
        low = bar.get("low", close)

        # Record prices for labeler and tracker
        self.labeler.record_prices(self.bar_count, close, high, low)

        # Compute features from full buffer
        df = self.buffer.to_dataframe()
        features = compute_features(df)
        if features is None:
            self.logger.info(f"Bar {self.bar_count}: warming up ({len(df)}/{config.REGIME_WARMUP_BARS} bars)")
            return None

        # --- Agent predictions ---
        regime_out = self.regime_agent.predict(features)
        valuation_out = self.valuation_agent.predict(features)
        risk_out = self.risk_agent.predict(features)
        vol_out = self.volatility_agent.predict(features)
        signal_out = self.signal_agent.predict(
            features,
            regime=regime_out["regime"],
            valuation_score=valuation_out["valuation_score"],
        )

        close = features["close"]
        # Stop distance from volatility agent
        stop_dist = vol_out.get("stop_distance", 0)
        # Profit target: RR mode uses fixed ratio, ML mode uses risk agent prediction
        if self._target_mode == "UseML":
            # ML target: risk agent's stop * learned ratio (asymmetric)
            ml_target = risk_out.get("stop_distance", 0) * self._reward_risk_ratio
            target_dist = max(ml_target, stop_dist * 0.5)  # floor at 0.5x stop
        else:
            target_dist = stop_dist * self._reward_risk_ratio

        if self.bar_count <= 5:
            self.logger.info(
                f"SL/PT calc: stop_dist={stop_dist:.4f} * RR={self._reward_risk_ratio:.4f} "
                f"= target_dist={target_dist:.4f} | mode={self._target_mode} | "
                f"SL_long={close - stop_dist:.2f} PT_long={close + target_dist:.2f}"
            )

        # --- Track prediction for performance evaluation ---
        self.tracker.record(
            bar_index=self.bar_count,
            prediction=signal_out["signal"],
            confidence=signal_out["confidence"],
            entry_close=close,
            stop_distance=stop_dist,
            target_distance=target_dist,
        )

        # --- Determine action (confidence scaled by accuracy) ---
        action = self._decide(signal_out, regime_out, valuation_out, risk_out)

        # --- Log ---
        acc_factor = self.tracker.accuracy_factor()
        self.decision_log.log(self.bar_count, close, regime_out, signal_out, valuation_out, risk_out, action)
        self.logger.info(
            f"Bar {self.bar_count} | close={close:.2f} | "
            f"regime={regime_out['regime']}({regime_out['confidence']:.2f}) | "
            f"signal={signal_out['signal']}({signal_out['confidence']:.2f}) | "
            f"val={valuation_out['valuation_score']:.2f} | "
            f"vol={vol_out.get('volatility_regime', '?')} | "
            f"stop={stop_dist:.2f} target={target_dist:.2f}({self._target_mode}) | "
            f"acc={self.tracker.accuracy():.0%}x{acc_factor:.2f} | "
            f"action={action}"
        )

        # --- Write signal CSV for NinjaTrader ---
        self._write_signal_csv(close, regime_out, signal_out, valuation_out, risk_out,
                               vol_out, stop_dist, target_dist, action)

        # --- Enqueue for delayed labeling ---
        self.labeler.enqueue(self.bar_count, features, regime_out["regime"], valuation_out["valuation_score"])

        # --- Train agents with delayed labels ---
        self._train_agents()

        # --- Evaluate pending predictions ---
        self.tracker.evaluate(
            self.bar_count,
            lambda idx: self.labeler._closes.get(idx),
        )

        # --- Cleanup old price data ---
        self.labeler.cleanup_prices(self.bar_count)

        # --- Periodic accuracy summary ---
        if self.bar_count % config.ACCURACY_LOG_INTERVAL == 0:
            s = self.tracker.summary()
            self.logger.info(
                f"Performance | accuracy={s['accuracy']:.1%} | "
                f"factor={s['accuracy_factor']:.2f} | "
                f"pnl={s['cumulative_pnl']:.2f} | "
                f"resolved={s['total_resolved']} | pending={s['pending']}"
            )

        return {
            "bar_index": self.bar_count,
            "close": close,
            "regime": regime_out,
            "signal": signal_out,
            "valuation": valuation_out,
            "risk": risk_out,
            "action": action,
        }

    def _decide(self, signal_out: dict, regime_out: dict,
                valuation_out: dict, risk_out: dict) -> str:
        """Combine agent outputs into a final action, scaled by accuracy."""
        signal = signal_out.get("signal", "hold")
        confidence = signal_out.get("confidence", 0)
        regime = regime_out.get("regime", "ranging")
        val_score = valuation_out.get("valuation_score", 0)

        # Scale confidence by tracking accuracy
        effective_confidence = confidence * self.tracker.accuracy_factor()

        # Chop = directionless noise, block trading entirely
        if regime == "chop":
            return "hold"
        if signal == "buy" and val_score > 0.6:
            effective_confidence *= 0.7
        if signal == "sell" and val_score < -0.6:
            effective_confidence *= 0.7

        if effective_confidence < config.MIN_CONFIDENCE:
            return "hold"

        if signal == "buy":
            return "buy"
        if signal == "sell":
            return "sell"

        return "hold"

    def _write_signal_csv(self, close: float, regime_out: dict, signal_out: dict,
                           valuation_out: dict, risk_out: dict, vol_out: dict,
                           stop_dist: float, target_dist: float, action: str):
        """Write latest agent outputs to CSV for NinjaTrader to read."""
        from datetime import datetime
        row = {
            "bar_index": self.bar_count,
            "bar_time": getattr(self, "_current_bar_time", ""),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "regime": regime_out.get("regime", "unknown"),
            "regime_conf": f"{regime_out.get('confidence', 0):.4f}",
            "signal": action,
            "signal_conf": f"{signal_out.get('confidence', 0):.4f}",
            "valuation_score": f"{valuation_out.get('valuation_score', 0):.4f}",
            "valuation_conf": f"{valuation_out.get('confidence', 0):.4f}",
            "stop_distance": f"{stop_dist:.4f}",
            "stop_price_long": f"{close - stop_dist:.4f}",
            "stop_price_short": f"{close + stop_dist:.4f}",
            "target_price_long": f"{close + target_dist:.4f}",
            "target_price_short": f"{close - target_dist:.4f}",
            "volatility_regime": vol_out.get("volatility_regime", "normal"),
            "signal_accuracy": f"{self.tracker.accuracy():.4f}",
            "accuracy_factor": f"{self.tracker.accuracy_factor():.4f}",
            "simulated_pnl": f"{self.tracker.cumulative_pnl:.4f}",
        }
        try:
            with open(config.SIGNAL_CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
        except OSError:
            pass  # File may be locked by NinjaTrader

    def _train_agents(self):
        """Feed delayed labels back to agents for learning."""
        labeled = self.labeler.generate_labels()
        for item in labeled:
            self.regime_agent.learn(item["features"], item["regime_label"])
            self.valuation_agent.learn(item["features"], item["valuation_label"])
            self.risk_agent.learn(item["features"], item["risk_label"])
            self.volatility_agent.learn(item["features"], item["risk_label"])
            self.signal_agent.learn(
                item["features"],
                item["signal_label"],
                regime=item["regime"],
                valuation_score=item["valuation_score"],
            )
            self._learn_count += 1
            if self._learn_count % config.SAVE_INTERVAL == 0:
                self._save_models()

    def _load_models(self):
        """Load persisted models from disk if available."""
        for agent in self._agents:
            if agent.load(config.MODELS_DIR):
                self.logger.info(f"Loaded {agent.name} ({agent.samples_seen} samples)")
            else:
                self.logger.info(f"No saved model for {agent.name}, starting fresh")
        if self.tracker.load(config.MODELS_DIR):
            s = self.tracker.summary()
            self.logger.info(f"Loaded tracker: accuracy={s['accuracy']:.1%}, pnl={s['cumulative_pnl']:.2f}")
        else:
            self.logger.info("No saved tracker, starting fresh")

    def _save_models(self):
        """Persist all agent models and tracker to disk."""
        for agent in self._agents:
            agent.save(config.MODELS_DIR)
        self.tracker.save(config.MODELS_DIR)

    def run(self):
        """Main loop: poll for new bars and process them."""
        import time
        preloaded = len(self.buffer)
        self.logger.info(f"Orchestrator started. Preloaded {preloaded} historical bars. Waiting for new bars...")
        try:
            while True:
                new_bars = self.buffer.poll()
                for bar in new_bars:
                    self.buffer.add(bar)
                    self._process_bar(bar)
                if not new_bars:
                    time.sleep(config.POLL_INTERVAL_SEC)
        finally:
            self._save_models()
            self.logger.info("Models saved on shutdown.")
