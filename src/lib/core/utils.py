"""
Shared utility functions for the futures trading system.

This module consolidates small helper functions that were previously duplicated
across multiple modules (safe_float, ema, atr, rsi).

Usage:
    from lib.utils import safe_float, ema, atr, rsi
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# A1 — safe_float
# ---------------------------------------------------------------------------


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning *default* on failure.

    Handles None, NaN, Inf, and non-numeric types gracefully.

    Args:
        val: Value to convert.
        default: Fallback value when conversion fails or result is non-finite.

    Returns:
        Float representation of *val*, or *default*.
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# A2 — ema
# ---------------------------------------------------------------------------


def ema(series: pd.Series | np.ndarray, span: int) -> pd.Series:
    """Exponential Moving Average.

    Accepts both a pandas Series and a raw numpy array. Always returns a
    ``pd.Series`` aligned with the input index (or a RangeIndex for arrays).

    Args:
        series: Price (or any numeric) series.
        span: EMA span (decay α = 2/(span+1)).

    Returns:
        ``pd.Series`` of EMA values.
    """
    s = pd.Series(series)
    return pd.Series(s.ewm(span=span, adjust=False).mean(), index=s.index)


def ema_numpy(values: np.ndarray, span: int) -> np.ndarray:
    """Vectorised EMA for raw numpy arrays (no pandas overhead).

    Args:
        values: 1-D numpy array of floats.
        span: EMA span.

    Returns:
        numpy array of the same length with EMA values.
    """
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# A3 — atr
# ---------------------------------------------------------------------------


def atr(
    high: Any,
    low: Any,
    close: Any,
    length: int = 14,
) -> pd.Series:
    """Average True Range (EWM / Wilder-style smoothing).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: ATR look-back period (default 14).

    Returns:
        ``pd.Series`` of ATR values aligned with the input index.

    See Also:
        - ``lib.trading.strategies.rb.range_builders.compute_atr``:
          DataFrame-based version (accepts a ``pd.DataFrame`` with OHLCV
          columns and returns a ``pd.Series``).
        - ``lib.analysis.crypto_momentum.compute_atr``:
          Numpy-array version that returns a single ``float`` (last ATR
          value), useful for scalar computations on raw arrays.
    """
    h = pd.Series(high)
    lo = pd.Series(low)
    c = pd.Series(close)
    tr1 = h - lo
    tr2 = (h - c.shift(1)).abs()
    tr3 = (lo - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pd.Series(tr.ewm(span=length, adjust=False).mean(), index=h.index)


# ---------------------------------------------------------------------------
# A4 — rsi
# ---------------------------------------------------------------------------


def rsi(series: pd.Series | np.ndarray, length: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        series: Price (typically close) series.
        length: RSI look-back period (default 14).

    Returns:
        ``pd.Series`` of RSI values in the range [0, 100].
    """
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=length, adjust=False).mean()
    avg_loss = loss.ewm(span=length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    return pd.Series(100 - (100 / (1 + rs)), index=s.index)
