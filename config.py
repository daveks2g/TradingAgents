from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_files"
CSV_PATH = DATA_DIR / "bars.csv"              # overridden by main.py with suffix
SIGNAL_CSV_PATH = DATA_DIR / "agent_signals.csv"  # overridden by main.py with suffix
LOG_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = DATA_DIR / "models"
SAVE_INTERVAL = 10             # save models every N learning cycles

# --- Data Pipeline ---
LOOKBACK_WINDOW = 200          # bars kept in rolling buffer
POLL_INTERVAL_SEC = 1.0        # how often to check CSV for new rows

# --- Base bar period (all parameters below are calibrated for this) ---
BASE_BAR_MINUTES = 5.0

# --- Feature Engineering (calibrated for BASE_BAR_MINUTES) ---
SMA_SHORT = 10
SMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2.0
ADX_PERIOD = 14

# --- Regime Agent ---
REGIME_WARMUP_BARS = 50        # bars before regime agent starts predicting
ADX_TREND_THRESHOLD = 25.0     # ADX above this = trending

# --- Signal Agent ---
SIGNAL_FORWARD_BARS = 6        # bars to look ahead for labeling
SIGNAL_THRESHOLD_PCT = 0.15    # min % move to label as buy/sell (vs hold)

# --- Valuation Agent ---
VALUATION_REVERT_BARS = 4      # bars to measure mean-reversion outcome

# --- Risk Agent ---
RISK_ATR_MULTIPLIER = 1.5      # default ATR multiplier for stop-loss

# --- Orchestrator ---
MIN_CONFIDENCE = 0.25          # minimum margin (winner - runner-up) to act on signals

# --- Performance Tracking ---
TRACKER_WINDOW = 100           # rolling window of predictions for accuracy
TRACKER_MIN_SAMPLES = 10       # min predictions before accuracy adjusts confidence
ACCURACY_LOG_INTERVAL = 50     # log accuracy summary every N bars


def scale_for_bar_period(bar_minutes: float) -> dict:
    """Return scaled parameters based on actual bar period vs the baseline.
    Periods that represent a fixed time window (e.g. 50-bar SMA = 250 min)
    get scaled so they cover the same wall-clock duration on any timeframe.
    """
    ratio = BASE_BAR_MINUTES / bar_minutes  # e.g. 5/1 = 5 for 1-min bars

    def _scale(val):
        return max(2, round(val * ratio))

    return {
        "sma_short": _scale(SMA_SHORT),
        "sma_long": _scale(SMA_LONG),
        "rsi_period": _scale(RSI_PERIOD),
        "atr_period": _scale(ATR_PERIOD),
        "bbands_period": _scale(BBANDS_PERIOD),
        "adx_period": _scale(ADX_PERIOD),
        "regime_warmup_bars": _scale(REGIME_WARMUP_BARS),
        "signal_forward_bars": _scale(SIGNAL_FORWARD_BARS),
        "valuation_revert_bars": _scale(VALUATION_REVERT_BARS),
        "signal_threshold_pct": SIGNAL_THRESHOLD_PCT / ratio,  # smaller bars = smaller moves
    }
