"""Performance tracking: monitors prediction accuracy and simulated P&L."""

import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import config


@dataclass
class PredictionRecord:
    bar_index: int
    prediction: str          # "buy", "sell", "hold"
    confidence: float
    entry_close: float
    stop_distance: float
    target_distance: float
    outcome: str | None = None      # "correct", "incorrect", "neutral"
    pnl: float = 0.0
    resolved: bool = False


class PredictionTracker:
    """Tracks signal prediction accuracy and simulated P&L."""

    def __init__(self, window: int = config.TRACKER_WINDOW):
        self._pending: deque[PredictionRecord] = deque()
        self._history: deque[PredictionRecord] = deque(maxlen=window)
        self._cumulative_pnl: float = 0.0
        self._total_resolved: int = 0

    def record(self, bar_index: int, prediction: str, confidence: float,
               entry_close: float, stop_distance: float, target_distance: float):
        """Record a new signal prediction for later evaluation."""
        if prediction == "hold":
            return  # only track actionable signals
        self._pending.append(PredictionRecord(
            bar_index=bar_index,
            prediction=prediction,
            confidence=confidence,
            entry_close=entry_close,
            stop_distance=stop_distance,
            target_distance=target_distance,
        ))

    def evaluate(self, current_bar: int, get_close: Callable[[int], float | None]):
        """Evaluate pending predictions that have enough future data."""
        remaining = deque()
        for rec in self._pending:
            exit_idx = rec.bar_index + config.SIGNAL_FORWARD_BARS
            exit_close = get_close(exit_idx)

            if exit_close is None:
                # Check for stale predictions (stuck too long)
                if current_bar - rec.bar_index > config.SIGNAL_FORWARD_BARS * 3:
                    continue  # discard stale
                remaining.append(rec)
                continue

            # Evaluate outcome
            if rec.prediction == "buy":
                pnl = exit_close - rec.entry_close
                # Check if stop or target was hit (simplified — uses exit close)
                if pnl >= rec.target_distance:
                    pnl = rec.target_distance
                elif pnl <= -rec.stop_distance:
                    pnl = -rec.stop_distance
                rec.outcome = "correct" if pnl > 0 else ("incorrect" if pnl < 0 else "neutral")

            elif rec.prediction == "sell":
                pnl = rec.entry_close - exit_close
                if pnl >= rec.target_distance:
                    pnl = rec.target_distance
                elif pnl <= -rec.stop_distance:
                    pnl = -rec.stop_distance
                rec.outcome = "correct" if pnl > 0 else ("incorrect" if pnl < 0 else "neutral")
            else:
                pnl = 0.0
                rec.outcome = "neutral"

            rec.pnl = pnl
            rec.resolved = True
            self._cumulative_pnl += pnl
            self._total_resolved += 1
            self._history.append(rec)

        self._pending = remaining

    def accuracy(self) -> float:
        """Rolling accuracy of signal predictions."""
        if len(self._history) < config.TRACKER_MIN_SAMPLES:
            return 0.5  # neutral default
        correct = sum(1 for r in self._history if r.outcome == "correct")
        total = sum(1 for r in self._history if r.outcome in ("correct", "incorrect"))
        return correct / max(1, total)

    def accuracy_factor(self) -> float:
        """Scale factor for confidence based on rolling accuracy.
        40% accuracy -> 0.3x, 55% -> 0.5x, 70% -> 1.0x, 75%+ -> 1.2x
        """
        if len(self._history) < config.TRACKER_MIN_SAMPLES:
            return 1.0  # no adjustment until enough data
        acc = self.accuracy()
        return max(0.3, min(1.2, (acc - 0.4) / 0.3))

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    @property
    def total_resolved(self) -> int:
        return self._total_resolved

    def summary(self) -> dict:
        """Summary stats for logging."""
        return {
            "accuracy": self.accuracy(),
            "accuracy_factor": self.accuracy_factor(),
            "cumulative_pnl": self._cumulative_pnl,
            "total_resolved": self._total_resolved,
            "pending": len(self._pending),
            "history_size": len(self._history),
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "history": self._history,
            "cumulative_pnl": self._cumulative_pnl,
            "total_resolved": self._total_resolved,
        }
        with open(directory / "tracker.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, directory: Path) -> bool:
        path = directory / "tracker.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._history = state["history"]
            self._cumulative_pnl = state["cumulative_pnl"]
            self._total_resolved = state["total_resolved"]
            return True
        except Exception:
            return False
