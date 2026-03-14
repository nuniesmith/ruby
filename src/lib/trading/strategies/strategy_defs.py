"""
Strategy module for futures backtesting.

Provides eleven strategy classes with trend filters and ATR-based risk management:
  1. TrendEMACross         — EMA crossover filtered by a longer trend EMA, ATR SL/TP
  2. RSIReversal           — RSI mean-reversion entries with trend filter, ATR SL/TP
  3. BreakoutStrategy      — Breakout of recent high/low with volume filter, ATR SL/TP
  4. VWAPReversion         — Mean-reversion around daily VWAP with trend filter
  5. ORBStrategy           — Opening Range Breakout of first N bars each session
  6. MACDMomentum          — MACD crossover with histogram acceleration filter
  7. PullbackEMA           — Pullback-to-EMA with candlestick confirmation
  8. EventReaction         — Post-event continuation/fade with volume confirmation
  9. ICTTrendEMA           — TrendEMA + ICT Smart Money confluence (FVG/OB filters)
  10. PlainEMACross        — Plain EMA crossover (legacy)
  11. VolumeProfileStrategy — Volume Profile POC reversion + Value Area rejection

All strategies are compatible with the `backtesting.py` library and expose
class-level parameters that Optuna can tune.

VolumeProfileStrategy uses rolling POC/VAH/VAL indicators computed via helper
functions imported from lib.analysis.volume_profile.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

from lib.analysis.volume_profile import _rolling_poc, _rolling_vah_val
from lib.core.utils import atr as _atr
from lib.core.utils import ema as _ema
from lib.core.utils import rsi as _rsi
from lib.core.utils import safe_float as _safe_float

logger = logging.getLogger("strategies")

# ---------------------------------------------------------------------------
# Indicator helper functions (compatible with backtesting.py's self.I())
# ---------------------------------------------------------------------------


def _passthrough(arr):
    """Identity function for pre-computed indicator arrays."""
    return arr


def _sma(series, length: int):
    """Simple Moving Average."""
    return pd.Series(series).rolling(length).mean()


def _rolling_max(series, length: int):
    """Rolling maximum (for breakout high detection)."""
    return pd.Series(series).rolling(length).max()


def _rolling_min(series, length: int):
    """Rolling minimum (for breakout low detection)."""
    return pd.Series(series).rolling(length).min()


def _macd_line(series, fast: int = 12, slow: int = 26):
    """MACD line (fast EMA - slow EMA)."""
    s = pd.Series(series)
    return s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()


def _macd_signal(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD signal line."""
    macd = _macd_line(series, fast, slow)
    return macd.ewm(span=signal, adjust=False).mean()


def _macd_histogram(series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD histogram (MACD line - signal line)."""
    macd = _macd_line(series, fast, slow)
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


# ---------------------------------------------------------------------------
# Strategy 1 — Trend-Filtered EMA Cross
# ---------------------------------------------------------------------------


class TrendEMACross(Strategy):
    """EMA crossover with a longer-term trend filter and ATR-based SL/TP.

    Only takes LONG trades when price is above the trend EMA, and SHORT
    trades when price is below it.  Exits on the opposite crossover OR
    when the ATR stop-loss / take-profit is hit — whichever comes first.

    Optimisable parameters
    ----------------------
    n1           : int    fast EMA period           (5 – 20)
    n2           : int    slow EMA period            (15 – 50)
    trend_period : int    trend-direction EMA period (40 – 120)
    atr_period   : int    ATR look-back              (10 – 20)
    atr_sl_mult  : float  SL distance = ATR × this   (1.0 – 3.0)
    atr_tp_mult  : float  TP distance = ATR × this   (1.5 – 5.0)
    trade_size   : float  fraction of equity per trade (0.05 – 0.30)
    """

    # Defaults (will be overridden by the optimizer)
    n1: int = 9
    n2: int = 21
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(_ema, close, self.n1, name=f"EMA{self.n1}")
        self.ema_slow = self.I(_ema, close, self.n2, name=f"EMA{self.n2}")
        self.ema_trend = self.I(_ema, close, self.trend_period, name=f"Trend{self.trend_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        if _is_nan(trend):
            return

        # --- Exit logic (signal-based; SL/TP handled by broker) ---
        if self.position:
            if (
                self.position.is_long
                and crossover(list(self.ema_slow), list(self.ema_fast))
                or self.position.is_short
                and crossover(list(self.ema_fast), list(self.ema_slow))
            ):
                self.position.close()
            return  # no new entries while in a position

        # --- Entry logic ---
        if crossover(list(self.ema_fast), list(self.ema_slow)) and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossover(list(self.ema_slow), list(self.ema_fast)) and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 2 — RSI Mean-Reversion
# ---------------------------------------------------------------------------


class RSIReversal(Strategy):
    """RSI mean-reversion entries with trend filter and ATR-based SL/TP.

    Enters LONG when RSI crosses up from oversold (and price > trend EMA).
    Enters SHORT when RSI crosses down from overbought (and price < trend EMA).
    Exits when RSI reaches the opposite extreme, or SL/TP is hit.

    Optimisable parameters
    ----------------------
    rsi_period     : int    RSI look-back               (7 – 21)
    rsi_oversold   : int    oversold threshold           (20 – 40)
    rsi_overbought : int    overbought threshold         (60 – 80)
    trend_period   : int    trend-direction EMA period   (40 – 120)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.0
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.rsi = self.I(_rsi, close, self.rsi_period, name=f"RSI{self.rsi_period}")
        self.ema_trend = self.I(_ema, close, self.trend_period, name=f"Trend{self.trend_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Need at least 2 RSI values for crossover detection
        if len(self.rsi) < 2 or _is_nan(self.rsi[-1]) or _is_nan(self.rsi[-2]):
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        if _is_nan(trend):
            return

        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]

        # --- Exit: RSI reaches opposite extreme ---
        if self.position:
            if (
                self.position.is_long
                and rsi_now >= self.rsi_overbought
                or self.position.is_short
                and rsi_now <= self.rsi_oversold
            ):
                self.position.close()
            return

        # --- Entry ---
        # Long: RSI crosses UP through oversold threshold, price above trend
        if rsi_prev <= self.rsi_oversold and rsi_now > self.rsi_oversold and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Short: RSI crosses DOWN through overbought threshold, price below trend
        elif rsi_prev >= self.rsi_overbought and rsi_now < self.rsi_overbought and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 3 — Breakout
# ---------------------------------------------------------------------------


class BreakoutStrategy(Strategy):
    """Price breakout of recent high/low with volume filter and ATR SL/TP.

    Enters LONG when Close breaks above the rolling highest-high of the
    prior `lookback` bars AND volume exceeds its moving average × vol_mult.
    Mirror logic for shorts.  Exits purely on SL/TP — no signal-based exit.

    Optimisable parameters
    ----------------------
    lookback       : int    high/low look-back bars      (10 – 50)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (2.0 – 6.0)
    vol_sma_period : int    volume SMA look-back          (10 – 30)
    vol_mult       : float  volume filter multiplier      (1.0 – 2.0)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    lookback: int = 20
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0
    vol_sma_period: int = 20
    vol_mult: float = 1.2
    trade_size: float = 0.10

    def init(self):
        self.highest = self.I(_rolling_max, self.data.High, self.lookback, name="HH")
        self.lowest = self.I(_rolling_min, self.data.Low, self.lookback, name="LL")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(_sma, self.data.Volume, self.vol_sma_period, name="VolSMA")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Need prior-bar rolling extremes (avoid look-ahead)
        if len(self.highest) < 2 or _is_nan(self.highest[-2]):
            return
        if len(self.lowest) < 2 or _is_nan(self.lowest[-2]):
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]

        # Volume gate — skip if below-average volume
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # Let SL/TP manage exits — no signal-based close
        if self.position:
            return

        prior_high = self.highest[-2]  # rolling max as of the previous bar
        prior_low = self.lowest[-2]

        # Breakout long
        if price > prior_high:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Breakout short
        elif price < prior_low:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 4 — VWAP Mean-Reversion
# ---------------------------------------------------------------------------


class VWAPReversion(Strategy):
    """Mean-reversion around daily VWAP with trend filter and ATR SL/TP.

    Enters LONG when price crosses back above VWAP after being below it
    (pullback buy in an uptrend).  Enters SHORT when price crosses below
    VWAP after being above it (rally sell in a downtrend).

    Designed for intraday futures where VWAP acts as a magnet / fair-value
    anchor.  Requires a trend EMA to filter direction and volume confirmation.

    Optimisable parameters
    ----------------------
    trend_period   : int    trend-direction EMA period   (40 – 120)
    atr_period     : int    ATR look-back                (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this     (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this     (1.5 – 5.0)
    vol_sma_period : int    volume SMA look-back          (10 – 30)
    vol_mult       : float  volume filter multiplier      (0.8 – 1.5)
    trade_size     : float  fraction of equity per trade  (0.05 – 0.30)
    """

    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.0
    vol_sma_period: int = 20
    vol_mult: float = 1.0
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_trend = self.I(_ema, close, self.trend_period, name=f"Trend{self.trend_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(_sma, self.data.Volume, self.vol_sma_period, name="VolSMA")

        # Pre-compute daily-resetting VWAP
        df = self.data.df
        idx = df.index
        high = df["High"]
        low = df["Low"]
        close_s = df["Close"]
        volume = df["Volume"]
        typical = (high + low + close_s) / 3
        tpv = typical * volume

        try:
            dates = idx.to_series().dt.date
            cum_tpv = tpv.groupby(dates.values).cumsum()
            cum_vol = volume.groupby(dates.values).cumsum()
        except AttributeError:
            # Non-datetime index — running cumulative VWAP
            cum_tpv = tpv.cumsum()
            cum_vol = volume.cumsum()

        vwap_arr = (cum_tpv / (cum_vol + 1e-10)).values
        self.vwap = self.I(_passthrough, vwap_arr, name="VWAP")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        if len(self.vwap) < 2 or _is_nan(self.vwap[-1]) or _is_nan(self.vwap[-2]):
            return

        price = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        vwap_now = self.vwap[-1]
        vwap_prev = self.vwap[-2]
        trend = self.ema_trend[-1]

        if _is_nan(trend) or _is_nan(vwap_now):
            return

        # Volume filter
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # --- Exit: price reverts back through VWAP ---
        if self.position:
            if self.position.is_long and price < vwap_now or self.position.is_short and price > vwap_now:
                self.position.close()
            return

        # --- Entry: VWAP crossover with trend filter ---
        crossed_above = prev_close <= vwap_prev and price > vwap_now
        crossed_below = prev_close >= vwap_prev and price < vwap_now

        if crossed_above and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossed_below and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 5 — Opening Range Breakout
# ---------------------------------------------------------------------------


class ORBStrategy(Strategy):
    """Opening Range Breakout — classic morning intraday strategy.

    Computes the high/low of the first `orb_bars` bars each session day.
    After the opening range is established:
      - LONG when price breaks above the opening range high
      - SHORT when price breaks below the opening range low

    Ideal for the first 2-3 hours of futures trading where morning
    momentum drives directional moves off the opening range.

    Optimisable parameters
    ----------------------
    orb_bars       : int    bars forming the opening range (3 – 12)
    atr_period     : int    ATR look-back                  (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this       (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this       (1.5 – 5.0)
    vol_sma_period : int    volume SMA look-back            (10 – 30)
    vol_mult       : float  volume filter multiplier        (0.8 – 1.5)
    trade_size     : float  fraction of equity per trade    (0.05 – 0.30)
    """

    orb_bars: int = 6  # 6 × 5min = 30 minute opening range
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    vol_sma_period: int = 20
    vol_mult: float = 1.0
    trade_size: float = 0.10

    def init(self):
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(_sma, self.data.Volume, self.vol_sma_period, name="VolSMA")

        # Pre-compute ORB levels per session day
        df = self.data.df
        idx = df.index
        h = df["High"]
        low_s = df["Low"]
        orb_bars = int(self.orb_bars)

        orb_h = np.full(len(idx), np.nan)
        orb_l = np.full(len(idx), np.nan)

        try:
            dates = idx.to_series().dt.date.values
            unique_dates = sorted(set(dates))
            for date_val in unique_dates:
                day_positions = np.where(dates == date_val)[0]
                if len(day_positions) < orb_bars + 1:
                    continue
                range_positions = day_positions[:orb_bars]
                trade_positions = day_positions[orb_bars:]
                range_high = h.iloc[range_positions].max()
                range_low = low_s.iloc[range_positions].min()
                orb_h[trade_positions] = range_high
                orb_l[trade_positions] = range_low
        except AttributeError:
            # Non-datetime index — use rolling lookback as fallback
            orb_h = pd.Series(h).rolling(orb_bars).max().shift(1).values
            orb_l = pd.Series(low_s).rolling(orb_bars).min().shift(1).values

        self.orb_high = self.I(_passthrough, orb_h, name="ORB_H")
        self.orb_low = self.I(_passthrough, orb_l, name="ORB_L")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        orb_h = self.orb_high[-1]
        orb_l = self.orb_low[-1]
        if _is_nan(orb_h) or _is_nan(orb_l):
            return  # Still in opening range formation

        price = self.data.Close[-1]

        # Volume gate
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        if _is_nan(vol_avg) or vol_avg <= 0 or vol < vol_avg * self.vol_mult:
            return

        # SL/TP manage exits — no signal-based close
        if self.position:
            return

        # Breakout long
        if price > orb_h:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # Breakout short
        elif price < orb_l:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 6 — MACD Momentum
# ---------------------------------------------------------------------------


class MACDMomentum(Strategy):
    """MACD crossover with histogram acceleration and trend filter.

    Enters LONG when MACD crosses above its signal line, histogram is
    accelerating (building momentum), and price is above the trend EMA.
    Mirror logic for shorts.  Exits on the opposite MACD crossover.

    Good for catching medium-momentum intraday moves where the initial
    impulse is confirmed by accelerating MACD histogram.

    Optimisable parameters
    ----------------------
    macd_fast      : int    fast EMA period for MACD      (8 – 16)
    macd_slow      : int    slow EMA period for MACD      (20 – 34)
    macd_signal    : int    signal line EMA period         (6 – 12)
    trend_period   : int    trend-direction EMA period     (40 – 120)
    atr_period     : int    ATR look-back                  (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this       (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this       (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade    (0.05 – 0.30)
    """

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.macd = self.I(_macd_line, close, self.macd_fast, self.macd_slow, name="MACD")
        self.signal = self.I(
            _macd_signal,
            close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            name="Signal",
        )
        self.histogram = self.I(
            _macd_histogram,
            close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            name="Hist",
        )
        self.ema_trend = self.I(_ema, close, self.trend_period, name=f"Trend{self.trend_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        if len(self.histogram) < 2:
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        hist_now = self.histogram[-1]
        hist_prev = self.histogram[-2]

        if _is_nan(trend) or _is_nan(hist_now) or _is_nan(hist_prev):
            return

        # Histogram acceleration: momentum is building, not fading
        hist_growing = hist_now > hist_prev

        # --- Exit: opposite MACD crossover ---
        if self.position:
            if (
                self.position.is_long
                and crossover(list(self.signal), list(self.macd))
                or self.position.is_short
                and crossover(list(self.macd), list(self.signal))
            ):
                self.position.close()
            return

        # --- Entry: MACD crossover + histogram acceleration + trend filter ---
        if crossover(list(self.macd), list(self.signal)) and hist_growing and price > trend:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossover(list(self.signal), list(self.macd)) and not hist_growing and price < trend:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 7 — Pullback-to-EMA with Candlestick Confirmation
# ---------------------------------------------------------------------------


def _is_bullish_engulfing(o_prev, c_prev, o_now, c_now):
    """Detect a bullish engulfing pattern (two-bar)."""
    prev_bearish = c_prev < o_prev
    curr_bullish = c_now > o_now
    engulfs = c_now > o_prev and o_now < c_prev
    return prev_bearish and curr_bullish and engulfs


def _is_bearish_engulfing(o_prev, c_prev, o_now, c_now):
    """Detect a bearish engulfing pattern (two-bar)."""
    prev_bullish = c_prev > o_prev
    curr_bearish = c_now < o_now
    engulfs = c_now < o_prev and o_now > c_prev
    return prev_bullish and curr_bearish and engulfs


def _is_hammer(o, h, lo, c, atr_val):
    """Detect a hammer / pin-bar (bullish reversal)."""
    body = abs(c - o)
    full_range = h - lo
    if full_range < 1e-10:
        return False
    lower_wick = min(o, c) - lo
    upper_wick = h - max(o, c)
    # Lower wick >= 2× body, upper wick small, close in upper half
    return lower_wick >= 2 * body and upper_wick <= body * 0.5 and c >= (h + lo) / 2 and full_range >= atr_val * 0.3


def _is_shooting_star(o, h, lo, c, atr_val):
    """Detect a shooting star / inverted pin (bearish reversal)."""
    body = abs(c - o)
    full_range = h - lo
    if full_range < 1e-10:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - lo
    return upper_wick >= 2 * body and lower_wick <= body * 0.5 and c <= (h + lo) / 2 and full_range >= atr_val * 0.3


class PullbackEMA(Strategy):
    """Pullback-to-EMA with candlestick confirmation and ATR-based SL/TP.

    Waits for EMAs to be stacked in trend order (fast > mid > slow for
    bullish), then enters when price retraces to the pullback EMA and
    prints a confirming candlestick pattern (bullish engulfing or hammer
    for longs; bearish engulfing or shooting star for shorts).

    An RSI filter prevents entries on exhausted pullbacks (e.g., RSI too
    high in an uptrend pullback means it hasn't really pulled back).

    Recommended EMA presets by instrument:
      - ES/GC: 9/21/50  (5-min)
      - NQ:   10/20/50  (5-min)
      - CL:    8/21/50  (3-min, faster due to volatility)

    Optimisable parameters
    ----------------------
    ema_fast       : int    fast EMA period (trend direction)  (5 – 15)
    ema_mid        : int    pullback target EMA period          (15 – 30)
    ema_slow       : int    slow EMA (trend backbone)           (40 – 80)
    rsi_period     : int    RSI look-back for exhaustion filter (7 – 21)
    rsi_limit      : int    RSI must be below this for longs    (30 – 60)
    atr_period     : int    ATR look-back                       (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this            (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this            (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade         (0.05 – 0.30)
    """

    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_limit: int = 45
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_f = self.I(_ema, close, self.ema_fast, name=f"EMA{self.ema_fast}")
        self.ema_m = self.I(_ema, close, self.ema_mid, name=f"EMA{self.ema_mid}")
        self.ema_s = self.I(_ema, close, self.ema_slow, name=f"EMA{self.ema_slow}")
        self.rsi = self.I(_rsi, close, self.rsi_period, name=f"RSI{self.rsi_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Need at least 2 bars for candle patterns
        if len(self.data.Close) < 3:
            return

        rsi_now = self.rsi[-1]
        if _is_nan(rsi_now):
            return

        ef = self.ema_f[-1]
        em = self.ema_m[-1]
        es = self.ema_s[-1]
        if _is_nan(ef) or _is_nan(em) or _is_nan(es):
            return

        # Let SL/TP manage exits
        if self.position:
            return

        price = self.data.Close[-1]
        o_now = self.data.Open[-1]
        h_now = self.data.High[-1]
        l_now = self.data.Low[-1]
        c_now = self.data.Close[-1]
        o_prev = self.data.Open[-2]
        c_prev = self.data.Close[-2]

        # Check EMA stacking
        bullish_stack = ef > em > es
        bearish_stack = ef < em < es

        # Pullback detection: price touched / crossed below the mid EMA
        pulled_to_mid_bull = l_now <= em * 1.002  # within 0.2% of mid EMA
        pulled_to_mid_bear = h_now >= em * 0.998

        # Candlestick confirmation
        bull_candle = _is_bullish_engulfing(o_prev, c_prev, o_now, c_now) or _is_hammer(
            o_now, h_now, l_now, c_now, atr_val
        )
        bear_candle = _is_bearish_engulfing(o_prev, c_prev, o_now, c_now) or _is_shooting_star(
            o_now, h_now, l_now, c_now, atr_val
        )

        # --- Long: stacked bull + pullback to mid EMA + bull candle + RSI filter ---
        if bullish_stack and pulled_to_mid_bull and bull_candle and c_now > em and rsi_now < self.rsi_limit:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        # --- Short: stacked bear + pullback to mid EMA + bear candle + RSI filter ---
        elif bearish_stack and pulled_to_mid_bear and bear_candle and c_now < em and rsi_now > (100 - self.rsi_limit):
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy 8 — Event Reaction (Post-News Continuation/Fade)
# ---------------------------------------------------------------------------


class EventReaction(Strategy):
    """Post-event continuation or fade strategy with volume confirmation.

    This strategy is designed for high-impact economic events (CPI, NFP,
    EIA, FOMC).  It detects "event bars" by looking for sudden volume
    spikes (volume > vol_spike_mult × average) combined with large price
    moves (move > move_atr_mult × ATR).  After the initial spike bar:

      1. Wait `wait_bars` for the dust to settle
      2. Enter in the direction of the spike if volume remains elevated
         (continuation) — the most common profitable pattern
      3. ATR-based SL/TP manage exits

    The strategy naturally fires on any high-volatility catalyst bar,
    making it useful even without an explicit event calendar.

    Optimisable parameters
    ----------------------
    vol_spike_mult : float  volume must exceed avg × this to detect event  (1.5 – 4.0)
    move_atr_mult  : float  price move must exceed ATR × this              (0.5 – 2.0)
    wait_bars      : int    bars to wait after spike before entry           (1 – 5)
    vol_confirm    : float  post-wait volume must exceed avg × this         (1.0 – 3.0)
    vol_sma_period : int    volume SMA look-back                            (10 – 30)
    atr_period     : int    ATR look-back                                   (10 – 20)
    atr_sl_mult    : float  SL distance = ATR × this                        (1.0 – 3.0)
    atr_tp_mult    : float  TP distance = ATR × this                        (1.5 – 5.0)
    trade_size     : float  fraction of equity per trade                    (0.05 – 0.30)
    """

    vol_spike_mult: float = 2.0
    move_atr_mult: float = 1.0
    wait_bars: int = 2
    vol_confirm: float = 1.5
    vol_sma_period: int = 20
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0
    trade_size: float = 0.10

    def init(self):
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )
        self.vol_avg = self.I(_sma, self.data.Volume, self.vol_sma_period, name="VolSMA")
        # Track detected spike bars: direction (+1 long, -1 short) and countdown
        # We use pre-computed arrays to avoid look-ahead
        df = self.data.df
        n = len(df)
        close = df["Close"].values
        volume = df["Volume"].values
        high = df["High"].values
        low = df["Low"].values

        # Pre-compute ATR and volume SMA for spike detection
        atr_arr = np.asarray(_atr(high, low, close, self.atr_period))
        vol_sma_arr = np.asarray(pd.Series(volume).rolling(self.vol_sma_period).mean())

        # Detect spike bars and build a signal array:
        #   spike_signal[i] = direction of a spike that occurred `wait_bars` ago
        # This avoids look-ahead: we only act on spikes from the past
        spike_dir = np.zeros(n, dtype=float)
        for i in range(1, n):
            if np.isnan(atr_arr[i]) or np.isnan(vol_sma_arr[i]) or vol_sma_arr[i] <= 0:
                continue
            if atr_arr[i] <= 0:
                continue
            vol_ratio = volume[i] / vol_sma_arr[i]
            price_move = close[i] - close[i - 1]
            move_magnitude = abs(price_move) / atr_arr[i]
            if vol_ratio >= self.vol_spike_mult and move_magnitude >= self.move_atr_mult:
                spike_dir[i] = 1.0 if price_move > 0 else -1.0

        # Shift spike signal forward by wait_bars so we act after waiting
        signal = np.zeros(n, dtype=float)
        wait = int(self.wait_bars)
        for i in range(wait, n):
            if spike_dir[i - wait] != 0:
                signal[i] = spike_dir[i - wait]

        self.event_signal = self.I(_passthrough, signal, name="EventSignal")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        # Let SL/TP manage exits
        if self.position:
            return

        signal = self.event_signal[-1]
        if signal == 0:
            return

        # Volume confirmation: current volume must still be elevated
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        if _is_nan(vol_avg) or vol_avg <= 0:
            return
        if vol < vol_avg * self.vol_confirm:
            return  # Volume has died down — skip

        price = self.data.Close[-1]

        # Continuation entry in the direction of the spike
        if signal > 0:
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)
        elif signal < 0:
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Legacy compatibility: plain EMA Cross (no filters, no stops)
# ---------------------------------------------------------------------------


class PlainEMACross(Strategy):
    """Original bare EMA crossover — kept for A/B comparison only."""

    n1: int = 9
    n2: int = 21
    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema1 = self.I(_ema, close, self.n1, name=f"EMA{self.n1}")
        self.ema2 = self.I(_ema, close, self.n2, name=f"EMA{self.n2}")

    def next(self):
        if self.position:
            if (
                self.position.is_long
                and crossover(list(self.ema2), list(self.ema1))
                or self.position.is_short
                and crossover(list(self.ema1), list(self.ema2))
            ):
                self.position.close()
            return
        if crossover(list(self.ema1), list(self.ema2)):
            self.buy(size=self.trade_size)
        elif crossover(list(self.ema2), list(self.ema1)):
            self.sell(size=self.trade_size)


# ---------------------------------------------------------------------------
# Strategy 10 — ICT-Enhanced Trend EMA (Smart Money Confluence)
# ---------------------------------------------------------------------------


def _compute_ict_confluence(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    ob_proximity_atr: float = 1.5,
    fvg_proximity_atr: float = 2.0,
    atr_period: int = 14,
) -> dict:
    """Pre-compute ICT confluence signals for a single bar.

    Checks whether the current price is near an active OB or unfilled FVG
    that aligns with the proposed trade direction.

    Returns a dict with boolean flags and optional SL/TP refinements.
    """
    result: dict[str, Any] = {
        "ob_aligned": False,
        "fvg_aligned": False,
        "ob_sl": None,
        "fvg_tp": None,
        "score": 0,
    }

    try:
        from lib.analysis.ict import get_active_order_blocks, get_unfilled_fvgs
    except ImportError:
        return result

    if df.empty or len(df) < atr_period + 10 or bar_index < atr_period:
        return result

    # Use data up to (and including) the current bar — no look-ahead
    df_slice = df.iloc[: bar_index + 1]
    if len(df_slice) < atr_period + 10:
        return result

    price = float(df_slice["Close"].iloc[-1])
    high_s = df_slice["High"].astype(float)
    low_s = df_slice["Low"].astype(float)
    close_s = df_slice["Close"].astype(float)

    # Compute ATR at current bar
    from lib.analysis.ict import _atr as ict_atr

    atr_series = ict_atr(high_s, low_s, close_s, atr_period)
    atr_val = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
    if atr_val <= 0:
        return result

    # --- Order Block alignment ---
    try:
        active_obs = get_active_order_blocks(df_slice, impulse_atr_mult=1.2, atr_period=atr_period)
        for ob in active_obs:
            dist = abs(price - ob["midpoint"])
            if dist > ob_proximity_atr * atr_val:
                continue
            # Bullish OB aligns with long entries (price near/above OB zone)
            if direction == "long" and ob["type"] == "bullish":
                if ob["low"] <= price <= ob["high"] + atr_val * 0.5:
                    result["ob_aligned"] = True
                    result["score"] += 1
                    # Use OB low as a tighter SL
                    result["ob_sl"] = ob["low"]
                    break
            # Bearish OB aligns with short entries (price near/below OB zone)
            elif direction == "short" and ob["type"] == "bearish" and ob["low"] - atr_val * 0.5 <= price <= ob["high"]:
                result["ob_aligned"] = True
                result["score"] += 1
                result["ob_sl"] = ob["high"]
                break
    except Exception:
        pass

    # --- FVG alignment ---
    try:
        unfilled = get_unfilled_fvgs(df_slice, min_gap_atr=0.2, atr_period=atr_period)
        for fvg in unfilled:
            dist = abs(price - fvg["midpoint"])
            if dist > fvg_proximity_atr * atr_val:
                continue
            # Bullish FVG above price → target for long TP
            if direction == "long" and fvg["type"] == "bullish":
                if fvg["midpoint"] > price:
                    result["fvg_aligned"] = True
                    result["score"] += 1
                    result["fvg_tp"] = fvg["midpoint"]
                    break
            # Bearish FVG below price → target for short TP
            elif direction == "short" and fvg["type"] == "bearish" and fvg["midpoint"] < price:
                result["fvg_aligned"] = True
                result["score"] += 1
                result["fvg_tp"] = fvg["midpoint"]
                break
    except Exception:
        pass

    return result


def _ict_confluence_array(
    df: pd.DataFrame,
    ema_fast_arr,
    ema_slow_arr,
    ema_trend_arr,
    atr_arr,
    ob_proximity_atr: float = 1.5,
    fvg_proximity_atr: float = 2.0,
    atr_period: int = 14,
) -> np.ndarray:
    """Pre-compute ICT confluence scores for the entire DataFrame.

    Returns an array of floats where each element is:
      +score  if long confluence detected (OB/FVG aligned with bullish)
      -score  if short confluence detected (OB/FVG aligned with bearish)
       0      if no ICT confluence at this bar

    The score ranges from 0 to 2 (1 for OB match, 1 for FVG match).

    This is done once during init() to avoid per-bar ICT detection overhead.
    """
    n = len(df)
    scores = np.zeros(n, dtype=float)

    try:
        from lib.analysis.ict import get_active_order_blocks, get_unfilled_fvgs
    except ImportError:
        return scores

    if n < atr_period + 20:
        return scores

    # We sample ICT every `stride` bars to keep init() fast.  Between
    # samples the score carries forward (ICT levels don't change bar-to-bar).
    stride = max(5, n // 100)

    for i in range(atr_period + 10, n, stride):
        atr_val = float(atr_arr[i]) if not _is_nan(atr_arr[i]) else 0.0
        if atr_val <= 0:
            continue

        price = float(df["Close"].iloc[i])
        ef = float(ema_fast_arr[i]) if not _is_nan(ema_fast_arr[i]) else price
        es = float(ema_slow_arr[i]) if not _is_nan(ema_slow_arr[i]) else price
        et = float(ema_trend_arr[i]) if not _is_nan(ema_trend_arr[i]) else price

        # Determine directional bias from EMAs
        if ef > es and price > et:
            direction = "long"
        elif ef < es and price < et:
            direction = "short"
        else:
            continue  # no clear bias → skip ICT check

        df_slice = df.iloc[: i + 1]
        score = 0.0

        # Check OBs
        try:
            active_obs = get_active_order_blocks(df_slice, impulse_atr_mult=1.2, atr_period=atr_period)
            for ob in active_obs[:5]:  # check nearest 5
                dist = abs(price - ob["midpoint"])
                if dist > ob_proximity_atr * atr_val:
                    continue
                if direction == "long" and ob["type"] == "bullish":
                    if ob["low"] <= price <= ob["high"] + atr_val * 0.5:
                        score += 1.0
                        break
                elif (
                    direction == "short"
                    and ob["type"] == "bearish"
                    and ob["low"] - atr_val * 0.5 <= price <= ob["high"]
                ):
                    score += 1.0
                    break
        except Exception:
            pass

        # Check FVGs
        try:
            unfilled = get_unfilled_fvgs(df_slice, min_gap_atr=0.2, atr_period=atr_period)
            for fvg in unfilled[:5]:
                dist = abs(price - fvg["midpoint"])
                if dist > fvg_proximity_atr * atr_val:
                    continue
                if (direction == "long" and fvg["type"] == "bullish" and fvg["midpoint"] > price) or (
                    direction == "short" and fvg["type"] == "bearish" and fvg["midpoint"] < price
                ):
                    score += 1.0
                    break
        except Exception:
            pass

        if score > 0:
            signed = score if direction == "long" else -score
            # Fill forward until next sample
            end = min(i + stride, n)
            scores[i:end] = signed

    return scores


class ICTTrendEMA(Strategy):
    """EMA crossover + ICT Smart Money Concepts confluence filter.

    Extends TrendEMACross by requiring ICT alignment before entry:
    price must be near an active Order Block and/or have an unfilled
    Fair Value Gap as a target in the trade direction.

    ICT confluence is pre-computed during init() to keep per-bar
    ``next()`` calls fast.

    Confluence modes (``ict_mode``):
      - 0 = ICT is optional bonus (standard EMA entry, tighter SL/TP if ICT)
      - 1 = Require at least 1 ICT signal (OB *or* FVG aligned)
      - 2 = Require both OB and FVG alignment (highest conviction only)

    Optimisable parameters
    ----------------------
    n1              : int    fast EMA period           (5 – 20)
    n2              : int    slow EMA period            (15 – 55)
    trend_period    : int    trend-direction EMA period (40 – 120)
    atr_period      : int    ATR look-back              (10 – 20)
    atr_sl_mult     : float  SL distance = ATR × this   (1.0 – 3.0)
    atr_tp_mult     : float  TP distance = ATR × this   (1.5 – 5.0)
    ob_proximity    : float  OB proximity in ATR multiples (0.5 – 2.5)
    fvg_proximity   : float  FVG proximity in ATR multiples (1.0 – 3.0)
    ict_mode        : int    confluence strictness (0, 1, 2)
    trade_size      : float  fraction of equity per trade (0.05 – 0.30)
    """

    # EMA parameters (same as TrendEMACross)
    n1: int = 9
    n2: int = 21
    trend_period: int = 50
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5

    # ICT-specific parameters
    ob_proximity: float = 1.5
    fvg_proximity: float = 2.0
    ict_mode: int = 1  # require at least 1 ICT signal

    trade_size: float = 0.10

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(_ema, close, self.n1, name=f"EMA{self.n1}")
        self.ema_slow = self.I(_ema, close, self.n2, name=f"EMA{self.n2}")
        self.ema_trend = self.I(_ema, close, self.trend_period, name=f"Trend{self.trend_period}")
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

        # Pre-compute ICT confluence scores across entire dataset
        ema_f_arr = np.asarray(_ema(close, self.n1))
        ema_s_arr = np.asarray(_ema(close, self.n2))
        ema_t_arr = np.asarray(_ema(close, self.trend_period))
        atr_arr = np.asarray(_atr(self.data.High, self.data.Low, self.data.Close, self.atr_period))

        ict_scores = _ict_confluence_array(
            self.data.df,
            ema_f_arr,
            ema_s_arr,
            ema_t_arr,
            atr_arr,
            ob_proximity_atr=self.ob_proximity,
            fvg_proximity_atr=self.fvg_proximity,
            atr_period=self.atr_period,
        )
        self.ict_score = self.I(_passthrough, ict_scores, name="ICT_Score")

    def next(self):
        atr_val = self.atr[-1]
        if _is_nan(atr_val) or atr_val <= 0:
            return

        price = self.data.Close[-1]
        trend = self.ema_trend[-1]
        if _is_nan(trend):
            return

        # --- Exit logic (same as TrendEMACross) ---
        if self.position:
            if (
                self.position.is_long
                and crossover(list(self.ema_slow), list(self.ema_fast))
                or self.position.is_short
                and crossover(list(self.ema_fast), list(self.ema_slow))
            ):
                self.position.close()
            return

        # --- ICT confluence gate ---
        ict_val = self.ict_score[-1]
        ict_score_abs = abs(ict_val) if not _is_nan(ict_val) else 0.0

        if self.ict_mode >= 1 and ict_score_abs < 1:
            return  # need at least 1 ICT signal
        if self.ict_mode >= 2 and ict_score_abs < 2:
            return  # need both OB + FVG

        # --- Entry logic ---
        if crossover(list(self.ema_fast), list(self.ema_slow)) and price > trend:
            if self.ict_mode > 0 and ict_val < 0:
                return  # ICT says short, skip long entry
            sl = price - atr_val * self.atr_sl_mult
            tp = price + atr_val * self.atr_tp_mult
            self.buy(size=self.trade_size, sl=sl, tp=tp)

        elif crossover(list(self.ema_slow), list(self.ema_fast)) and price < trend:
            if self.ict_mode > 0 and ict_val > 0:
                return  # ICT says long, skip short entry
            sl = price + atr_val * self.atr_sl_mult
            tp = price - atr_val * self.atr_tp_mult
            self.sell(size=self.trade_size, sl=sl, tp=tp)


# ---------------------------------------------------------------------------
# Strategy registry — used by the optimizer / engine
# (VolumeProfileStrategy is defined later in this file; registry is populated
# after its definition at module bottom)
# ---------------------------------------------------------------------------

STRATEGY_CLASSES: dict[str, type] = {
    "TrendEMA": TrendEMACross,
    "RSI": RSIReversal,
    "Breakout": BreakoutStrategy,
    "VWAP": VWAPReversion,
    "ORB": ORBStrategy,
    "MACD": MACDMomentum,
    "PullbackEMA": PullbackEMA,
    "EventReaction": EventReaction,
    "ICTTrendEMA": ICTTrendEMA,
    "PlainEMA": PlainEMACross,
}

# Human-readable labels
STRATEGY_LABELS = {
    "TrendEMA": "Trend-Filtered EMA Cross",
    "RSI": "RSI Mean-Reversion",
    "Breakout": "Breakout + Volume",
    "VWAP": "VWAP Reversion",
    "ORB": "Opening Range Breakout",
    "MACD": "MACD Momentum",
    "PullbackEMA": "Pullback-to-EMA + Candle Confirm",
    "EventReaction": "Event Reaction (Post-News)",
    "ICTTrendEMA": "ICT Smart Money + EMA Cross",
    "VolumeProfile": "Volume Profile (POC/VA)",
    "PlainEMA": "Plain EMA Cross (legacy)",
}


def suggest_params(trial, strategy_key: str) -> dict:
    """Ask Optuna to suggest hyper-parameters for the given strategy.

    Returns a dict that can be unpacked into the Strategy's class attributes.
    """
    params: dict = {}

    if strategy_key == "VolumeProfile":
        return suggest_volume_profile_params(trial)

    elif strategy_key == "TrendEMA":
        params["n1"] = trial.suggest_int("n1", 5, 20)
        params["n2"] = trial.suggest_int("n2", max(params["n1"] + 5, 15), 55)
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "RSI":
        params["rsi_period"] = trial.suggest_int("rsi_period", 7, 21)
        params["rsi_oversold"] = trial.suggest_int("rsi_oversold", 20, 40)
        params["rsi_overbought"] = trial.suggest_int(
            "rsi_overbought",
            max(params["rsi_oversold"] + 20, 60),
            80,
        )
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "Breakout":
        params["lookback"] = trial.suggest_int("lookback", 10, 50)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 2.0, 6.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 1.0, 2.0, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "VWAP":
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 0.8, 1.5, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "ORB":
        params["orb_bars"] = trial.suggest_int("orb_bars", 3, 12)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["vol_mult"] = trial.suggest_float("vol_mult", 0.8, 1.5, step=0.1)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "MACD":
        params["macd_fast"] = trial.suggest_int("macd_fast", 8, 16)
        params["macd_slow"] = trial.suggest_int("macd_slow", max(params["macd_fast"] + 8, 20), 34)
        params["macd_signal"] = trial.suggest_int("macd_signal", 6, 12)
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "PullbackEMA":
        params["ema_fast"] = trial.suggest_int("ema_fast", 5, 15)
        params["ema_mid"] = trial.suggest_int("ema_mid", max(params["ema_fast"] + 5, 15), 30)
        params["ema_slow"] = trial.suggest_int("ema_slow", max(params["ema_mid"] + 10, 40), 80)
        params["rsi_period"] = trial.suggest_int("rsi_period", 7, 21)
        params["rsi_limit"] = trial.suggest_int("rsi_limit", 30, 60)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "EventReaction":
        params["vol_spike_mult"] = trial.suggest_float("vol_spike_mult", 1.5, 4.0, step=0.25)
        params["move_atr_mult"] = trial.suggest_float("move_atr_mult", 0.5, 2.0, step=0.25)
        params["wait_bars"] = trial.suggest_int("wait_bars", 1, 5)
        params["vol_confirm"] = trial.suggest_float("vol_confirm", 1.0, 3.0, step=0.25)
        params["vol_sma_period"] = trial.suggest_int("vol_sma_period", 10, 30)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "ICTTrendEMA":
        params["n1"] = trial.suggest_int("n1", 5, 20)
        params["n2"] = trial.suggest_int("n2", max(params["n1"] + 5, 15), 55)
        params["trend_period"] = trial.suggest_int("trend_period", 40, 120)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        params["atr_sl_mult"] = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        params["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25)
        params["ob_proximity"] = trial.suggest_float("ob_proximity", 0.5, 2.5, step=0.25)
        params["fvg_proximity"] = trial.suggest_float("fvg_proximity", 1.0, 3.0, step=0.25)
        params["ict_mode"] = trial.suggest_int("ict_mode", 0, 2)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    elif strategy_key == "PlainEMA":
        params["n1"] = trial.suggest_int("n1", 5, 20)
        params["n2"] = trial.suggest_int("n2", max(params["n1"] + 5, 15), 55)
        params["trade_size"] = trial.suggest_float("trade_size", 0.05, 0.30, step=0.05)

    return params


def make_strategy(strategy_key: str, params: dict) -> type:
    """Create a new Strategy *subclass* with the given params baked in.

    We create a fresh subclass each time so that class-attribute mutations
    in one backtest don't leak into another.
    """
    base_cls = STRATEGY_CLASSES[strategy_key]

    # Build a new class dynamically with params as class attributes
    attrs = dict(params)  # copy
    new_cls = type(f"{base_cls.__name__}_Configured", (base_cls,), attrs)
    return new_cls


# ---------------------------------------------------------------------------
# Scoring helper — used by the optimizer
# ---------------------------------------------------------------------------

_PENALTY = -100.0  # returned for invalid / degenerate runs


def score_backtest(stats, min_trades: int = 3) -> float:
    """Compute a risk-adjusted score from backtest stats.

    Designed for funded-account (TPT) trading where drawdown control is
    paramount and consistent win rates matter more than occasional big wins.

    Scoring components:
      - Base: Sharpe (40%) + Sortino (30%) + normalised Profit Factor (30%)
      - Drawdown penalty: progressive, severe above 6%
      - Win rate bonus: rewards consistency above 45%
      - Expectancy bonus: rewards positive per-trade edge
      - Trade count bonus: prefers statistically significant sample sizes
    """
    n_trades = int(stats["# Trades"])
    if n_trades < min_trades:
        return _PENALTY

    sharpe = float(stats["Sharpe Ratio"])
    if _is_nan(sharpe):
        return _PENALTY

    max_dd = abs(float(stats["Max. Drawdown [%]"]))
    wr = _safe_float(stats["Win Rate [%]"])
    pf = _safe_float(stats.get("Profit Factor", 0))
    sortino = _safe_float(stats.get("Sortino Ratio", sharpe))
    expectancy = _safe_float(stats.get("Expectancy [%]", 0))

    # Base: weighted combination of risk-adjusted metrics
    pf_norm = min(pf / 3.0, 1.0) * 2.0 if pf > 0 else 0.0
    score = 0.4 * sharpe + 0.3 * sortino + 0.3 * pf_norm

    # Drawdown penalty — progressive and severe for funded accounts
    if max_dd > 3:
        score -= (max_dd - 3) * 0.08
    if max_dd > 6:
        score -= (max_dd - 6) * 0.15
    if max_dd > 10:
        score -= (max_dd - 10) * 0.30

    # Win rate bonus (consistent winners)
    if wr > 45:
        score += (wr - 45) * 0.015
    if wr > 60:
        score += (wr - 60) * 0.01  # diminishing returns above 60%

    # Expectancy bonus (per-trade edge)
    if expectancy > 0:
        score += min(expectancy * 0.1, 0.5)

    # Trade count: prefer statistical significance
    if n_trades >= 8:
        score += 0.1
    if n_trades >= 15:
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _is_nan(x) -> bool:
    """Robust NaN check for floats and numpy scalars."""
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Volume Profile Strategy
# ---------------------------------------------------------------------------


class VolumeProfileStrategy(Strategy):
    """Volume Profile strategy combining POC reversion and Value Area rejection.

    Setup 1 — POC Mean Reversion:
        When price moves significantly away from the rolling POC (> atr_dist_mult × ATR),
        enter toward the POC. Price reverts to POC ~75% of the time in ranging markets.

    Setup 2 — Value Area Rejection:
        When price dips below VAL and closes back above it (bullish),
        or pushes above VAH and closes back below it (bearish),
        enter in the rejection direction targeting POC then opposite VA boundary.

    Parameters (Optuna-tunable):
        vp_lookback: int — bars for rolling volume profile (50–200)
        vp_bins: int — number of price bins (20–60)
        atr_period: int — ATR period for distance/SL/TP
        atr_dist_mult: float — ATR multiple for POC distance threshold
        atr_sl_mult: float — ATR multiple for stop loss
        atr_tp_mult: float — ATR multiple for take profit
        trade_size: float — position size as fraction of equity
    """

    # Tunable parameters with defaults
    vp_lookback: int = 100
    vp_bins: int = 30
    atr_period: int = 14
    atr_dist_mult: float = 1.5  # enter when price is this many ATRs from POC
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trade_size: float = 0.10

    def init(self):
        # ATR for distance thresholds and SL/TP
        self.atr = self.I(
            _atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name="ATR",
        )

        # Rolling POC
        self.poc = self.I(
            _rolling_poc,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.vp_lookback,
            self.vp_bins,
            name="POC",
        )

        # Rolling VAH/VAL — compute together, store as separate indicators
        vah_arr, val_arr = _rolling_vah_val(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            self.vp_lookback,
            self.vp_bins,
        )
        self.vah = self.I(_passthrough, vah_arr.values, name="VAH")
        self.val_line = self.I(_passthrough, val_arr.values, name="VAL")

        # Trend filter: 50-period EMA
        self.trend_ema = self.I(_ema, pd.Series(self.data.Close), 50, name="EMA50")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        poc = self.poc[-1]
        vah = self.vah[-1]
        val = self.val_line[-1]
        trend = self.trend_ema[-1]

        # Skip if indicators aren't ready
        if (
            math.isnan(atr_val)
            or math.isnan(poc)
            or math.isnan(vah)
            or math.isnan(val)
            or math.isnan(trend)
            or atr_val <= 0
        ):
            return

        # If already in a position, manage it (SL/TP handled by order placement)
        if self.position:
            return

        dist_from_poc = price - poc
        atr_threshold = atr_val * self.atr_dist_mult

        sl_dist = atr_val * self.atr_sl_mult
        tp_dist = atr_val * self.atr_tp_mult

        # --- Setup 1: POC Mean Reversion ---
        # Price is far above POC → short toward POC (if below trend = bearish bias)
        if dist_from_poc > atr_threshold and price < trend:
            sl = price + sl_dist
            tp = price - tp_dist
            self.sell(size=self.trade_size, sl=sl, tp=tp)
            return

        # Price is far below POC → long toward POC (if above trend = bullish bias)
        if dist_from_poc < -atr_threshold and price > trend:
            sl = price - sl_dist
            tp = price + tp_dist
            self.buy(size=self.trade_size, sl=sl, tp=tp)
            return

        # --- Setup 2: Value Area Rejection ---
        if len(self.data.Close) < 3:
            return

        prev_close = self.data.Close[-2]
        prev_low = self.data.Low[-2]
        prev_high = self.data.High[-2]
        prev_open = self.data.Open[-2]

        # Bullish rejection: previous bar dipped below VAL, current close above VAL
        # and previous bar shows bullish character (close > open = bullish candle)
        if prev_low < val and price > val and prev_close > prev_open:
            sl = price - sl_dist
            tp = poc  # target POC first
            # Ensure TP is actually above entry
            if tp <= price:
                tp = price + tp_dist
            self.buy(size=self.trade_size, sl=sl, tp=tp)
            return

        # Bearish rejection: previous bar pushed above VAH, current close below VAH
        # and previous bar shows bearish character (close < open = bearish candle)
        if prev_high > vah and price < vah and prev_close < prev_open:
            sl = price + sl_dist
            tp = poc  # target POC first
            # Ensure TP is actually below entry
            if tp >= price:
                tp = price - tp_dist
            self.sell(size=self.trade_size, sl=sl, tp=tp)
            return


# ---------------------------------------------------------------------------
# Optuna parameter suggestion for VolumeProfileStrategy
# ---------------------------------------------------------------------------


def suggest_volume_profile_params(trial) -> dict:
    """Ask Optuna to suggest parameters for VolumeProfileStrategy.

    Compatible with the optimizer in engine.py.
    """
    return {
        "vp_lookback": trial.suggest_int("vp_lookback", 50, 200, step=10),
        "vp_bins": trial.suggest_int("vp_bins", 20, 60, step=5),
        "atr_period": trial.suggest_int("atr_period", 10, 20),
        "atr_dist_mult": trial.suggest_float("atr_dist_mult", 1.0, 3.0, step=0.25),
        "atr_sl_mult": trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25),
        "atr_tp_mult": trial.suggest_float("atr_tp_mult", 1.5, 5.0, step=0.25),
        "trade_size": trial.suggest_float("trade_size", 0.05, 0.30, step=0.05),
    }


# Register now that the class is defined
STRATEGY_CLASSES["VolumeProfile"] = VolumeProfileStrategy
