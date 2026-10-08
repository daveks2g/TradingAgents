"""Abstract base class for all trading agents."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Base class providing the predict/learn interface for online-learning MLP agents."""

    def __init__(self, name: str):
        self.name = name
        self.samples_seen = 0
        self._feature_keys: list[str] = []
        # Running stats for online standardization
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None

    def _to_array(self, feature_dict: dict) -> np.ndarray:
        return np.array([feature_dict.get(k, 0.0) for k in self._feature_keys], dtype=np.float64)

    def _update_scaler(self, x: np.ndarray):
        """Welford's online algorithm for running mean/variance."""
        if self._mean is None:
            self._mean = x.copy()
            self._var = np.zeros_like(x)
            return
        n = self.samples_seen  # already incremented before calling this
        delta = x - self._mean
        self._mean += delta / n
        delta2 = x - self._mean
        self._var += delta * delta2

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """Standardize using running statistics."""
        if self._mean is None or self.samples_seen < 2:
            return x
        std = np.sqrt(self._var / max(1, self.samples_seen - 1))
        std = np.where(std < 1e-8, 1.0, std)
        return (x - self._mean) / std

    @abstractmethod
    def predict(self, features: dict) -> dict[str, Any]:
        """Make a prediction given current features."""

    @abstractmethod
    def learn(self, features: dict, label: Any) -> None:
        """Update the model with a new labeled example."""

    @property
    def is_warm(self) -> bool:
        """Whether the agent has seen enough samples to make meaningful predictions."""
        return self.samples_seen >= 15

    def save(self, directory: Path) -> None:
        """Persist model, scaler stats, and state to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "samples_seen": self.samples_seen,
            "mean": self._mean,
            "var": self._var,
        }
        with open(directory / f"{self.name}.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, directory: Path) -> bool:
        """Load model and state from disk. Returns True if loaded successfully."""
        path = directory / f"{self.name}.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state["model"]
            self.samples_seen = state["samples_seen"]
            self._mean = state["mean"]
            self._var = state["var"]
            return True
        except Exception:
            return False
