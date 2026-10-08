"""Logging utility for agent decisions and trade signals."""

import csv
import logging
from datetime import datetime
from pathlib import Path

import config


def setup_logger(name: str = "TradingAgents") -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(config.LOG_DIR / "trading.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        import sys, io
        ch = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        )
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


class DecisionLogger:
    """Logs every agent decision to a CSV for later analysis."""

    def __init__(self, path: Path | None = None):
        self.path = path or config.LOG_DIR / "decisions.csv"
        config.LOG_DIR.mkdir(exist_ok=True)
        self._initialized = False

    def _init_file(self):
        if not self._initialized:
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "bar_index", "close",
                    "regime", "regime_conf",
                    "signal", "signal_conf",
                    "valuation_score", "valuation_conf",
                    "stop_distance", "risk_conf",
                    "action",
                ])
            self._initialized = True

    def log(self, bar_index: int, close: float, regime_out: dict,
            signal_out: dict, valuation_out: dict, risk_out: dict, action: str):
        self._init_file()
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                bar_index,
                f"{close:.4f}",
                regime_out.get("regime", ""),
                f"{regime_out.get('confidence', 0):.3f}",
                signal_out.get("signal", ""),
                f"{signal_out.get('confidence', 0):.3f}",
                f"{valuation_out.get('valuation_score', 0):.4f}",
                f"{valuation_out.get('confidence', 0):.3f}",
                f"{risk_out.get('stop_distance', 0):.4f}",
                f"{risk_out.get('confidence', 0):.3f}",
                action,
            ])
