"""
Functional helper wrappers for common indicator calculations.

These thin wrappers provide a one-liner API for the most common indicator
computations, suitable for use in analysis modules.

Usage:
    from lib.indicators.helpers import ema, sma, rsi, atr, macd, bollinger, vwap

All functions accept pandas Series / DataFrames and return Series or dicts of Series.
For lower-level numpy-array helpers, see lib.core.utils.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average
# ---------------------------------------------------------------------------


def ema(series: pd.Series | np.ndarray, period: int) -> pd.Series:
    """Exponential Moving Average using pandas ewm (alpha = 2/(period+1)).

    Args:
        series: Price series (pd.Series or array-like).
        period: EMA span.

    Returns:
        pd.Series of EMA values.
    """
    s = pd.Series(series)
    return pd.Series(s.ewm(span=period, adjust=False).mean(), index=s.index)


def ema_numpy(values: np.ndarray, span: int) -> np.ndarray:
    """Vectorised EMA for raw numpy arrays.

    Handles NaN seeds by forward-filling from the first finite value.

    Args:
        values: 1-D numpy array of floats.
        span: EMA span.

    Returns:
        numpy array of the same length with EMA values.
    """
    alpha = 2.0 / (span + 1)
    out = np.empty(len(values), dtype=float)
    # Seed with first finite value
    seed_idx = 0
    for i, v in enumerate(values):
        if np.isfinite(v):
            seed_idx = i
            break
    out[:seed_idx] = np.nan
    if len(values) == 0:
        return out
    out[seed_idx] = values[seed_idx]
    for i in range(seed_idx + 1, len(values)):
        v = values[i]
        if np.isfinite(v):
            out[i] = alpha * v + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


# ---------------------------------------------------------------------------
# SMA — Simple Moving Average
# ---------------------------------------------------------------------------


def sma(series: pd.Series | np.ndarray, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series.
        period: Window size.

    Returns:
        pd.Series of SMA values.
    """
    s = pd.Series(series)
    return pd.Series(s.rolling(window=period).mean(), index=s.index)


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index
# ---------------------------------------------------------------------------


def rsi(series: pd.Series | np.ndarray, period: int = 14) -> pd.Series:
    """Relative Strength Index using EWM smoothing.

    Args:
        series: Close price series.
        period: RSI look-back period (default 14).

    Returns:
        pd.Series of RSI values in [0, 100].
    """
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return pd.Series(100.0 - (100.0 / (1.0 + rs)), index=s.index)


def rsi_scalar(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI and return the last value as a float.

    Uses Wilder smoothing (seed with SMA, then recursive).

    Args:
        close: Array of close prices.
        period: RSI period.

    Returns:
        Latest RSI value as a float, or 50.0 if insufficient data.
    """
    n = len(close)
    if n < period + 1:
        return 50.0
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# ---------------------------------------------------------------------------
# ATR — Average True Range
# ---------------------------------------------------------------------------


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using EWM smoothing.

    Args:
        data: DataFrame with 'high', 'low', 'close' columns
              (case-insensitive — also accepts 'High', 'Low', 'Close').
        period: ATR look-back period (default 14).

    Returns:
        pd.Series of ATR values.
    """
    # Handle both lowercase and capitalized column names
    col_map = {c.lower(): c for c in data.columns}
    h = data[col_map.get("high", "high")].astype(float)
    lo = data[col_map.get("low", "low")].astype(float)
    c = data[col_map.get("close", "close")].astype(float)

    tr1 = h - lo
    tr2 = (h - c.shift(1)).abs()
    tr3 = (lo - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pd.Series(tr.ewm(span=period, adjust=False).mean(), index=data.index)


def atr_scalar(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """ATR returning the last value as a scalar float.

    Uses Wilder smoothing (RMA, alpha = 1/period). Matches Pine Script ta.atr().

    Args:
        highs: Array of high prices.
        lows: Array of low prices.
        closes: Array of close prices.
        period: ATR period.

    Returns:
        Latest ATR value as a float, or 0.0 if insufficient data.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0
    alpha = 1.0 / period
    prev_c = closes[:-1]
    curr_h = highs[1:]
    curr_l = lows[1:]
    tr = np.maximum(
        np.maximum(curr_h - curr_l, np.abs(curr_h - prev_c)),
        np.abs(curr_l - prev_c),
    )
    # Seed ATR with simple average of first `period` TRs
    if len(tr) < period:
        return 0.0
    atr_val = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_val = alpha * tr[i] + (1.0 - alpha) * atr_val
    return float(atr_val)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    series: pd.Series | np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """MACD line, signal line, and histogram.

    Args:
        series: Close price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Dict with keys 'macd_line', 'signal_line', 'histogram'.
    """
    s = pd.Series(series)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    idx = s.index
    return {
        "macd_line": pd.Series(macd_line, index=idx),
        "signal_line": pd.Series(signal_line, index=idx),
        "histogram": pd.Series(histogram, index=idx),
    }


def macd_numpy(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD using numpy arrays (for performance-sensitive paths).

    Args:
        closes: Array of close prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram) as numpy arrays.
    """
    ema_f = ema_numpy(closes, fast)
    ema_s = ema_numpy(closes, slow)
    macd_line = ema_f - ema_s
    signal_line = ema_numpy(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger(
    series: pd.Series | np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands: middle (SMA), upper, lower, bandwidth, %B.

    Args:
        series: Close price series.
        period: SMA period (default 20).
        std_dev: Standard deviation multiplier (default 2.0).

    Returns:
        Dict with keys 'middle', 'upper', 'lower', 'bandwidth', 'percent_b'.
    """
    s = pd.Series(series)
    middle = s.rolling(window=period).mean()
    std = s.rolling(window=period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    bandwidth = (upper - lower) / (middle + 1e-10)
    percent_b = (s - lower) / (upper - lower + 1e-10)
    idx = s.index
    return {
        "middle": pd.Series(middle, index=idx),
        "upper": pd.Series(upper, index=idx),
        "lower": pd.Series(lower, index=idx),
        "bandwidth": pd.Series(bandwidth, index=idx),
        "percent_b": pd.Series(percent_b, index=idx),
    }


# ---------------------------------------------------------------------------
# VWAP — Volume-Weighted Average Price
# ---------------------------------------------------------------------------


def vwap(data: pd.DataFrame) -> pd.Series:
    """Session VWAP: cumsum(TypicalPrice * Volume) / cumsum(Volume).

    Args:
        data: DataFrame with high, low, close, volume columns
              (case-insensitive — also accepts High, Low, Close, Volume).

    Returns:
        pd.Series of VWAP values.
    """
    col_map = {c.lower(): c for c in data.columns}
    h = data[col_map.get("high", "high")].astype(float)
    lo = data[col_map.get("low", "low")].astype(float)
    c = data[col_map.get("close", "close")].astype(float)
    v = data[col_map.get("volume", "volume")].astype(float)

    typical = (h + lo + c) / 3.0
    cum_tp_vol = (typical * v).cumsum()
    cum_vol = v.cumsum().replace(0, np.nan)
    result = cum_tp_vol / cum_vol
    return pd.Series(result, index=data.index, name="VWAP")


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------


def awesome_oscillator(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    fast: int = 5,
    slow: int = 34,
) -> float:
    """Awesome Oscillator: SMA(hl2, fast) - SMA(hl2, slow). Returns last value.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        fast: Fast SMA period (default 5).
        slow: Slow SMA period (default 34).

    Returns:
        Latest AO value as float, or 0.0 if insufficient data.
    """
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    if len(h) < slow:
        return 0.0
    hl2 = (h + lo) / 2.0
    fast_sma = float(np.mean(hl2[-fast:]))
    slow_sma = float(np.mean(hl2[-slow:]))
    return fast_sma - slow_sma
