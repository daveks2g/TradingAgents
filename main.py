"""Entry point for the multi-agent trading system.

Usage:
    .venv\\Scripts\\python main.py

Auto-detects the active instrument/timeframe folder from data_files/.
Waits for NinjaTrader to start writing bars, then begins processing.
"""

import time
import sys
from pathlib import Path

import config
from orchestrator import Orchestrator


def _find_active_bars_csv() -> Path | None:
    """Find the most recently modified bars.csv across all instrument/timeframe folders."""
    if not config.DATA_DIR.exists():
        return None
    candidates = list(config.DATA_DIR.glob("*/*/bars.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _wait_for_bars() -> Path:
    """Wait until NinjaTrader starts writing a bars.csv somewhere."""
    print("Waiting for NinjaTrader to write bars...")
    last_path = None
    while True:
        csv_path = _find_active_bars_csv()
        if csv_path and csv_path.stat().st_size > 0:
            if csv_path != last_path:
                print(f"  Found: {csv_path}")
                last_path = csv_path
            return csv_path
        time.sleep(1.0)


def _setup_config(csv_path: Path):
    """Set config paths based on the detected bars.csv location."""
    tf_dir = csv_path.parent
    config.SIGNAL_CSV_PATH = tf_dir / "agent_signals.csv"
    config.MODELS_DIR = tf_dir / "models"


def main():
    print("Starting Trading Agents system")
    print(f"  Scanning: {config.DATA_DIR}")

    while True:
        csv_path = _wait_for_bars()
        _setup_config(csv_path)

        instrument = csv_path.parent.parent.name
        timeframe = csv_path.parent.name

        print(f"\n  Instrument: {instrument}")
        print(f"  Timeframe:  {timeframe}")
        print(f"  Bars CSV:   {csv_path}")
        print(f"  Signal CSV: {config.SIGNAL_CSV_PATH}")
        print(f"  Models:     {config.MODELS_DIR}")
        print(f"  Lookback: {config.LOOKBACK_WINDOW} bars")
        print(f"  Poll interval: {config.POLL_INTERVAL_SEC}s")
        print()

        orchestrator = Orchestrator(csv_path)
        try:
            orchestrator.run()
        except KeyboardInterrupt:
            print("\nShutting down.")
            sys.exit(0)


if __name__ == "__main__":
    main()
