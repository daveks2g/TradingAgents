"""Feature engineering from OHLCV bar data using pure pandas/numpy."""

import numpy as np
import pandas as pd

import config


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = config.ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = config.BBANDS_PERIOD, num_std: float = config.BBANDS_STD):
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, pct_b


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = config.ADX_PERIOD) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # zero out when the other is larger
    mask = plus_dm > minus_dm
    minus_dm = minus_dm.where(~mask, 0)
    plus_dm = plus_dm.where(mask, 0)

    atr_vals = atr(high, low, close, period)
    plus_di = 100 * ema(plus_dm, period) / atr_vals.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_vals.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period), plus_di, minus_di


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period) * 100


def compute_features(df: pd.DataFrame) -> dict | None:
    """Compute all features from an OHLCV DataFrame. Returns latest bar's features or None if not enough data."""
    if len(df) < config.REGIME_WARMUP_BARS:
        return None

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

    sma_short = sma(c, config.SMA_SHORT)
    sma_long = sma(c, config.SMA_LONG)
    rsi_val = rsi(c)
    atr_val = atr(h, l, c)
    adx_val, plus_di, minus_di = adx(h, l, c)
    bb_upper, bb_mid, bb_lower, bb_pct = bollinger_bands(c)
    roc = rate_of_change(c)

    # Slope of SMA (normalized by ATR)
    sma_short_slope = sma_short.diff(3) / atr_val.replace(0, np.nan)
    sma_long_slope = sma_long.diff(3) / atr_val.replace(0, np.nan)

    # Volatility ratio: short-term vol / long-term vol
    vol_short = c.pct_change().rolling(10).std()
    vol_long = c.pct_change().rolling(50).std()
    vol_ratio = vol_short / vol_long.replace(0, np.nan)

    # Price range compression (Bollinger Band width / price)
    bb_width = (bb_upper - bb_lower) / c.replace(0, np.nan)

    # Volume relative to moving average
    vol_ma = sma(v, 20)
    vol_relative = v / vol_ma.replace(0, np.nan)

    idx = len(df) - 1
    features = {
        "close": c.iloc[idx],
        "sma_short": sma_short.iloc[idx],
        "sma_long": sma_long.iloc[idx],
        "sma_short_slope": sma_short_slope.iloc[idx],
        "sma_long_slope": sma_long_slope.iloc[idx],
        "rsi": rsi_val.iloc[idx],
        "atr": atr_val.iloc[idx],
        "adx": adx_val.iloc[idx],
        "plus_di": plus_di.iloc[idx],
        "minus_di": minus_di.iloc[idx],
        "bb_pct": bb_pct.iloc[idx],
        "bb_width": bb_width.iloc[idx],
        "roc": roc.iloc[idx],
        "vol_ratio": vol_ratio.iloc[idx],
        "vol_relative": vol_relative.iloc[idx],
        "high": h.iloc[idx],
        "low": l.iloc[idx],
    }

    # Replace NaN/inf with 0 for model safety
    for k, val in features.items():
        if not np.isfinite(val):
            features[k] = 0.0

    return features
