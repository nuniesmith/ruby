"""
Ruby Signal Engine — Pine Script v6 Port
=========================================
Complete Python port of the Ruby indicator (ruby.pine) including:

  1. Top G Channel  (rolling highest / lowest + HMA mid, strong / simple / potential signals)
  2. Wave Analysis  (bull/bear wave amplitude tracking via EMA crossover, wave ratio, momentum)
  3. Market Regime  (SMA-200 slope normalised by avg-change + volatility normalised)
  4. Market Phase   (UPTREND / DOWNTREND / DISTRIB / ACCUM / NEUTRAL)
  5. Volatility Percentile  (ATR percentile over rolling 200-bar window)
  6. Session Bias + ORB  (previous-day H/L, initial balance, ORB range, bullBias vote)
  7. Squeeze Detection  (Bollinger inside Keltner)
  8. Quality Score  (0–100 from 5 sub-scores)
  9. Main Signal generation  (LONG / SHORT with 5-bar cooldown)
 10. Level computation  (entry, SL ±1 ATR, TP1/2/3 at configurable R multiples)

``RubySignalEngine.update(bar)`` is the public API.  Call it once per
confirmed 1-minute bar and it returns a ``RubySignal`` that is compatible
with ``PositionManager.process_signal()`` — the relevant attributes
(``symbol``, ``direction``, ``trigger_price``, ``breakout_detected``,
``cnn_prob``, ``cnn_signal``, ``filter_passed``, ``mtf_score``,
``atr_value``, ``range_high``, ``range_low``, ``regime``, ``wave_ratio``)
are all present and typed correctly.

Usage::

    from lib.services.engine.ruby_signal_engine import RubySignalEngine

    engine = RubySignalEngine(symbol="MNQ", config=RubyConfig())
    engine.load_state()   # optional — restore from Redis

    for bar in bar_stream:
        sig = engine.update(bar)
        if sig.breakout_detected:
            orders = position_manager.process_signal(sig, bars_1m)

Environment Variables:
    RUBY_TOP_G_LEN          — Top G channel length (default 50)
    RUBY_SIG_SENS           — Signal sensitivity multiplier (default 0.5)
    RUBY_ORB_MINUTES        — ORB formation window in minutes (default 5)
    RUBY_VOL_MULT           — Volume spike multiplier (default 1.2)
    RUBY_MIN_QUALITY        — Minimum quality score to emit signal (default 45)
    RUBY_HTF_EMA_PERIOD     — HTF EMA period for bias vote (default 9)
    RUBY_TP1_R              — TP1 R multiple (default 1.5)
    RUBY_TP2_R              — TP2 R multiple (default 2.5)
    RUBY_TP3_R              — TP3 R multiple (default 4.0)
    RUBY_REQUIRE_VWAP       — Require VWAP alignment (default 1)
    RUBY_IB_MINUTES         — Initial Balance window in minutes (default 60)
    RUBY_BIAS_MODE          — Session bias mode: Auto | Long Only | Short Only (default Auto)
    RUBY_STATE_TTL          — Redis state key TTL in seconds (default 86400)
    RUBY_MAX_HISTORY        — Max bars to retain in internal history (default 500)
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Any

import numpy as np

logger = logging.getLogger("engine.ruby_signal_engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REDIS_KEY_PREFIX = "engine:ruby:"


@dataclass
class RubyConfig:
    """Configuration for the Ruby Signal Engine.

    All defaults mirror the Pine Script defaults so back-test results
    are directly comparable.
    """

    # Top G Channel
    top_g_len: int = int(os.getenv("RUBY_TOP_G_LEN", "50"))
    sig_sens: float = float(os.getenv("RUBY_SIG_SENS", "0.5"))

    # ORB / IB
    orb_minutes: int = int(os.getenv("RUBY_ORB_MINUTES", "5"))
    ib_minutes: int = int(os.getenv("RUBY_IB_MINUTES", "60"))

    # Volume / Quality
    vol_mult: float = float(os.getenv("RUBY_VOL_MULT", "1.2"))
    min_quality_pct: int = int(os.getenv("RUBY_MIN_QUALITY", "45"))

    # HTF EMA (used for bull-bias vote)
    htf_ema_period: int = int(os.getenv("RUBY_HTF_EMA_PERIOD", "9"))

    # R multiples for TP levels
    tp1_r: float = float(os.getenv("RUBY_TP1_R", "1.5"))
    tp2_r: float = float(os.getenv("RUBY_TP2_R", "2.5"))
    tp3_r: float = float(os.getenv("RUBY_TP3_R", "4.0"))

    # VWAP alignment requirement
    require_vwap: bool = os.getenv("RUBY_REQUIRE_VWAP", "1") == "1"

    # Bias mode: "Auto" | "Long Only" | "Short Only"
    bias_mode: str = os.getenv("RUBY_BIAS_MODE", "Auto")

    # Redis / memory
    state_ttl: int = int(os.getenv("RUBY_STATE_TTL", "86400"))
    max_history: int = int(os.getenv("RUBY_MAX_HISTORY", "500"))


# ---------------------------------------------------------------------------
# RubySignal — output dataclass, compatible with PositionManager
# ---------------------------------------------------------------------------


@dataclass
class RubySignal:
    """Output of ``RubySignalEngine.update()``.

    Attributes mirror the ``BreakoutResult`` interface consumed by
    ``PositionManager.process_signal()`` so the Ruby engine is a
    drop-in signal source alongside the ORB/breakout detectors.
    """

    # ------------------------------------------------------------------
    # PositionManager-compatible fields
    # ------------------------------------------------------------------
    symbol: str = ""
    direction: str = ""  # "LONG" | "SHORT" | ""
    trigger_price: float = 0.0
    breakout_detected: bool = False
    cnn_prob: float = 0.0  # quality score mapped to [0, 1] for PM gates
    cnn_signal: bool = False
    filter_passed: bool = True
    mtf_score: float = 0.0
    atr_value: float = 0.0
    range_high: float = 0.0  # Top G upper bound
    range_low: float = 0.0  # Top G lower bound

    # ------------------------------------------------------------------
    # Ruby-specific computed fields
    # ------------------------------------------------------------------
    quality: float = 0.0  # 0–100 quality score
    regime: str = "NEUTRAL"  # TRENDING ↑ / TRENDING ↓ / VOLATILE / RANGING / NEUTRAL
    phase: str = "NEUTRAL"  # UPTREND / DOWNTREND / DISTRIB / ACCUM / NEUTRAL
    wave_ratio: float = 1.0  # bull avg / bear avg amplitude ratio
    cur_ratio: float = 0.0  # current bar momentum vs historical average
    mkt_bias: str = "Neutral"  # Bullish / Bearish / Neutral
    bull_bias: bool = False  # composite bias vote
    vol_pct: float = 0.5  # ATR volatility percentile [0, 1]
    vol_regime: str = "MED"  # VERY HIGH / HIGH / MED / LOW / VERY LOW
    ao: float = 0.0  # Awesome Oscillator value
    vwap: float = 0.0
    ema9: float = 0.0
    rsi14: float = 50.0
    tg_hi: float = 0.0  # Top G resistance
    tg_lo: float = 0.0  # Top G support
    tg_mid: float = 0.0  # Top G midline (HMA)
    tg_range: float = 0.0  # tg_hi - tg_lo
    orb_high: float = 0.0
    orb_low: float = 0.0
    orb_ready: bool = False
    pd_high: float = 0.0  # previous-day high
    pd_low: float = 0.0  # previous-day low
    ib_high: float = 0.0
    ib_low: float = 0.0
    ib_done: bool = False
    sqz_on: bool = False
    sqz_fired: bool = False

    # Entry / SL / TP levels (populated when breakout_detected = True)
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    risk: float = 0.0

    # Signal class
    signal_class: str = ""  # "TG_BOUNCE" | "TG_REJECT" | "ORB_UP" | "ORB_DN" | "RB_UP" | "RB_DN"
    is_orb_window: bool = False  # True if within 30 min of NY / London / Asia open

    # Timestamps
    bar_time: datetime | None = None
    computed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d = {
            "symbol": self.symbol,
            "direction": self.direction,
            "trigger_price": round(self.trigger_price, 6),
            "breakout_detected": self.breakout_detected,
            "cnn_prob": round(self.cnn_prob, 4),
            "cnn_signal": self.cnn_signal,
            "filter_passed": self.filter_passed,
            "mtf_score": round(self.mtf_score, 4),
            "atr_value": round(self.atr_value, 6),
            "range_high": round(self.range_high, 6),
            "range_low": round(self.range_low, 6),
            "quality": round(self.quality, 1),
            "regime": self.regime,
            "phase": self.phase,
            "wave_ratio": round(self.wave_ratio, 4),
            "cur_ratio": round(self.cur_ratio, 4),
            "mkt_bias": self.mkt_bias,
            "bull_bias": self.bull_bias,
            "vol_pct": round(self.vol_pct, 4),
            "vol_regime": self.vol_regime,
            "ao": round(self.ao, 6),
            "vwap": round(self.vwap, 6),
            "ema9": round(self.ema9, 6),
            "rsi14": round(self.rsi14, 2),
            "tg_hi": round(self.tg_hi, 6),
            "tg_lo": round(self.tg_lo, 6),
            "tg_mid": round(self.tg_mid, 6),
            "tg_range": round(self.tg_range, 6),
            "orb_high": round(self.orb_high, 6),
            "orb_low": round(self.orb_low, 6),
            "orb_ready": self.orb_ready,
            "pd_high": round(self.pd_high, 6),
            "pd_low": round(self.pd_low, 6),
            "ib_high": round(self.ib_high, 6),
            "ib_low": round(self.ib_low, 6),
            "ib_done": self.ib_done,
            "sqz_on": self.sqz_on,
            "sqz_fired": self.sqz_fired,
            "entry": round(self.entry, 6),
            "sl": round(self.sl, 6),
            "tp1": round(self.tp1, 6),
            "tp2": round(self.tp2, 6),
            "tp3": round(self.tp3, 6),
            "risk": round(self.risk, 6),
            "signal_class": self.signal_class,
            "is_orb_window": self.is_orb_window,
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
        }
        return d


# ---------------------------------------------------------------------------
# Internal per-bar record
# ---------------------------------------------------------------------------


@dataclass
class _Bar:
    """Normalised OHLCV bar used internally."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Low-level maths helpers (Pine-equivalent)
# ---------------------------------------------------------------------------


def _hma(values: np.ndarray, period: int) -> float:
    """Hull Moving Average — last value only.

    HMA(n) = WMA(2 * WMA(n/2) - WMA(n), floor(sqrt(n)))
    """
    if len(values) < period:
        return float(np.mean(values)) if len(values) else 0.0
    half = max(1, period // 2)
    sqrt_n = max(1, int(math.isqrt(period)))

    def _wma(v: np.ndarray, n: int) -> np.ndarray:
        n = min(n, len(v))
        weights = np.arange(1, n + 1, dtype=float)
        result = np.empty(len(v), dtype=float)
        result[:] = np.nan
        for i in range(n - 1, len(v)):
            result[i] = np.dot(v[i - n + 1 : i + 1], weights) / weights.sum()
        return result

    wma_half = _wma(values, half)
    wma_full = _wma(values, period)
    raw = 2.0 * wma_half - wma_full
    # Re-apply WMA of sqrt(n) on the raw series
    raw_valid = raw[~np.isnan(raw)]
    if len(raw_valid) < sqrt_n:
        return float(raw_valid[-1]) if len(raw_valid) else 0.0
    result = _wma(raw_valid, sqrt_n)
    valid = result[~np.isnan(result)]
    return float(valid[-1]) if len(valid) else 0.0


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Vectorised EMA (alpha = 2/(period+1)).  Matches Pine ta.ema()."""
    if len(values) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (period + 1)
    out = np.empty(len(values), dtype=float)
    # Seed: first finite value
    out[0] = values[0]
    for i in range(1, len(values)):
        if np.isfinite(values[i]):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def _rma(values: np.ndarray, period: int) -> np.ndarray:
    """Running Moving Average (Wilder smoothing).  Matches Pine ta.rma()."""
    if len(values) == 0:
        return np.array([], dtype=float)
    alpha = 1.0 / period
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        if np.isfinite(values[i]):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def _sma_last(values: np.ndarray, period: int) -> float:
    """Last SMA value."""
    n = min(period, len(values))
    if n == 0:
        return 0.0
    return float(np.mean(values[-n:]))


def _ema_last(values: np.ndarray, period: int) -> float:
    """Last EMA value."""
    if len(values) == 0:
        return 0.0
    return float(_ema(values, period)[-1])


def _rma_last(values: np.ndarray, period: int) -> float:
    """Last RMA value."""
    if len(values) == 0:
        return 0.0
    return float(_rma(values, period)[-1])


def _atr14_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """ATR (Wilder/RMA) — last value. Matches Pine ta.atr(14)."""
    n = len(closes)
    if n < 2:
        return 0.0
    tr = np.maximum(
        np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1])),
        np.abs(lows[1:] - closes[:-1]),
    )
    if len(tr) == 0:
        return 0.0
    return float(_rma(tr, period)[-1])


def _rsi_last(closes: np.ndarray, period: int = 14) -> float:
    """RSI (Wilder smoothing) — last value. Matches Pine ta.rsi()."""
    n = len(closes)
    if n < period + 1:
        return 50.0
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _rma_last(gains, period)
    avg_loss = _rma_last(losses, period)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _stdev_last(values: np.ndarray, period: int) -> float:
    n = min(period, len(values))
    if n < 2:
        return 0.0
    return float(np.std(values[-n:], ddof=0))


def _linreg_last(values: np.ndarray, period: int) -> float:
    """Last value of a linear regression line. Matches Pine ta.linreg()."""
    n = min(period, len(values))
    if n < 2:
        return float(values[-1]) if len(values) else 0.0
    y = values[-n:]
    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, y, 1)
    return float(m * (n - 1) + b)


def _roc(values: np.ndarray, period: int) -> np.ndarray:
    """Rate of change — matches Pine ta.roc()."""
    out = np.full(len(values), np.nan)
    for i in range(period, len(values)):
        prev = values[i - period]
        if prev != 0:
            out[i] = (values[i] - prev) / prev * 100.0
    return out


def _ao_last(highs: np.ndarray, lows: np.ndarray) -> float:
    """Awesome Oscillator: SMA(hl2, 5) - SMA(hl2, 34)."""
    if len(highs) < 34:
        return 0.0
    hl2 = (highs + lows) / 2.0
    return float(np.mean(hl2[-5:]) - np.mean(hl2[-34:]))


def _vwap_session(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray
) -> float:
    """Session VWAP using all bars provided. Matches Pine ta.vwap(hlc3)."""
    if len(closes) == 0:
        return float(closes[-1]) if len(closes) else 0.0
    typical = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_tpv = np.cumsum(typical * volumes)
    if cum_vol[-1] == 0:
        return float(typical[-1])
    return float(cum_tpv[-1] / cum_vol[-1])


# ---------------------------------------------------------------------------
# ORB window helper (mirrors Pine's 30-min session check for NY/London/Asia)
# ---------------------------------------------------------------------------
# These are the same windows as in ruby.pine §7, converted to ET equivalents.
# True-ORB windows run for 30 min after each major session open:
#   NY RTH  09:30–10:00 ET
#   London  03:00–03:30 ET  (08:00 GMT expressed on US-exchange CT/ET)
#   Asia    19:00–19:30 ET  (approximate CME opening window)

_ORB_WINDOWS_ET: list[tuple[dt_time, dt_time]] = [
    (dt_time(9, 30), dt_time(10, 0)),  # NY RTH open
    (dt_time(3, 0), dt_time(3, 30)),  # London open
    (dt_time(19, 0), dt_time(19, 30)),  # Asia / CME Globex open
]


def _in_orb_window(t: datetime) -> bool:
    """Return True if the given ET time is within a recognised ORB window."""
    try:
        from zoneinfo import ZoneInfo

        et = t.astimezone(ZoneInfo("America/New_York"))
        current = et.time()
        for start, end in _ORB_WINDOWS_ET:
            if start <= current < end:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# RubySignalEngine
# ---------------------------------------------------------------------------


class RubySignalEngine:
    """Stateful, bar-by-bar Ruby indicator engine.

    Internal history is stored as parallel ``deque`` structures (one per
    field) bounded at ``config.max_history`` bars so memory stays flat
    regardless of runtime duration.

    Persistent state (wave arrays, PD levels, ORB levels, bar counts,
    cooldown) is saved to and restored from Redis so the engine survives
    process restarts without losing in-progress wave tracking.
    """

    def __init__(self, symbol: str, config: RubyConfig | None = None) -> None:
        self.symbol = symbol
        self.cfg = config or RubyConfig()
        self._redis_key = f"{_REDIS_KEY_PREFIX}{symbol}"

        N = self.cfg.max_history

        # Raw OHLCV history
        self._opens: deque[float] = deque(maxlen=N)
        self._highs: deque[float] = deque(maxlen=N)
        self._lows: deque[float] = deque(maxlen=N)
        self._closes: deque[float] = deque(maxlen=N)
        self._volumes: deque[float] = deque(maxlen=N)
        self._times: deque[datetime] = deque(maxlen=N)

        # Wave analysis state (Pine var — persisted across bars)
        self._bull_waves: list[float] = []  # amplitude of completed bull waves (up to 200)
        self._bear_waves: list[float] = []  # amplitude of completed bear waves (absolute values)
        self._wave_speed: float = 0.0  # running c_rma - o_rma accumulator
        self._last_wave_bar: int = 0  # bar index at last wave flip
        self._wave_speed_lo: float = 0.0  # lowest speed seen in current wave
        self._wave_speed_hi: float = 0.0  # highest speed seen in current wave
        self._prev_ema20_cross: int = 0  # 0 = unknown, 1 = bull, -1 = bear
        self._bar_index: int = 0

        # Cooldown (Pine var lastLong / lastShort)
        self._last_long_bar: int = -999
        self._last_short_bar: int = -999

        # ORB state (resets on new day)
        self._orb_high: float | None = None
        self._orb_low: float | None = None
        self._orb_ready: bool = False
        self._orb_bars_today: int = 0  # bars elapsed in session today
        self._orb_new_day_seen: bool = False

        # Previous day H/L
        self._pd_high: float | None = None
        self._pd_low: float | None = None
        self._today_high: float | None = None
        self._today_low: float | None = None
        self._last_day: int | None = None  # day-of-year

        # Initial Balance
        self._ib_high: float | None = None
        self._ib_low: float | None = None
        self._ib_bars: int = 0
        self._ib_done: bool = False

        # Volatility percentile window (200 bars of ATR14)
        self._vol_arr: deque[float] = deque(maxlen=200)

        # Market phase (Pine var — sticky)
        self._market_phase: str = "NEUTRAL"

        # Wave ratio (Pine var — sticky)
        self._wave_ratio: float = 1.0
        self._mkt_bias: str = "Neutral"
        self._bull_avg: float = 0.0001
        self._bear_avg_v: float = 0.0001

        # Last emitted signal (for deduplication / status)
        self._last_signal: RubySignal | None = None

        logger.debug("RubySignalEngine(%s) initialised — top_g_len=%d", symbol, self.cfg.top_g_len)

    # ------------------------------------------------------------------
    # State persistence (Redis)
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        """Restore persisted wave / ORB / PD state from Redis.

        Non-fatal — if Redis is unavailable or key missing, engine starts
        fresh and will re-learn wave state from incoming bars.
        """
        try:
            import json as _json

            from lib.core.cache import cache_get

            raw = cache_get(self._redis_key)
            if not raw:
                return
            data = _json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            self._bull_waves = data.get("bull_waves", [])[:200]
            self._bear_waves = data.get("bear_waves", [])[:200]
            self._wave_speed = data.get("wave_speed", 0.0)
            self._last_wave_bar = data.get("last_wave_bar", 0)
            self._wave_speed_lo = data.get("wave_speed_lo", 0.0)
            self._wave_speed_hi = data.get("wave_speed_hi", 0.0)
            self._prev_ema20_cross = data.get("prev_ema20_cross", 0)
            self._bar_index = data.get("bar_index", 0)
            self._last_long_bar = data.get("last_long_bar", -999)
            self._last_short_bar = data.get("last_short_bar", -999)
            self._pd_high = data.get("pd_high")
            self._pd_low = data.get("pd_low")
            self._today_high = data.get("today_high")
            self._today_low = data.get("today_low")
            self._last_day = data.get("last_day")
            self._orb_high = data.get("orb_high")
            self._orb_low = data.get("orb_low")
            self._orb_ready = data.get("orb_ready", False)
            self._orb_bars_today = data.get("orb_bars_today", 0)
            self._ib_high = data.get("ib_high")
            self._ib_low = data.get("ib_low")
            self._ib_bars = data.get("ib_bars", 0)
            self._ib_done = data.get("ib_done", False)
            self._market_phase = data.get("market_phase", "NEUTRAL")
            self._wave_ratio = data.get("wave_ratio", 1.0)
            self._mkt_bias = data.get("mkt_bias", "Neutral")
            self._bull_avg = data.get("bull_avg", 0.0001)
            self._bear_avg_v = data.get("bear_avg_v", 0.0001)
            logger.debug("RubySignalEngine(%s) state restored from Redis", self.symbol)
        except Exception as exc:
            logger.debug("RubySignalEngine.load_state(%s) error (non-fatal): %s", self.symbol, exc)

    def save_state(self) -> None:
        """Persist current wave / ORB / PD state to Redis."""
        try:
            import json as _json

            from lib.core.cache import cache_set

            data = {
                "bull_waves": self._bull_waves[:200],
                "bear_waves": self._bear_waves[:200],
                "wave_speed": self._wave_speed,
                "last_wave_bar": self._last_wave_bar,
                "wave_speed_lo": self._wave_speed_lo,
                "wave_speed_hi": self._wave_speed_hi,
                "prev_ema20_cross": self._prev_ema20_cross,
                "bar_index": self._bar_index,
                "last_long_bar": self._last_long_bar,
                "last_short_bar": self._last_short_bar,
                "pd_high": self._pd_high,
                "pd_low": self._pd_low,
                "today_high": self._today_high,
                "today_low": self._today_low,
                "last_day": self._last_day,
                "orb_high": self._orb_high,
                "orb_low": self._orb_low,
                "orb_ready": self._orb_ready,
                "orb_bars_today": self._orb_bars_today,
                "ib_high": self._ib_high,
                "ib_low": self._ib_low,
                "ib_bars": self._ib_bars,
                "ib_done": self._ib_done,
                "market_phase": self._market_phase,
                "wave_ratio": self._wave_ratio,
                "mkt_bias": self._mkt_bias,
                "bull_avg": self._bull_avg,
                "bear_avg_v": self._bear_avg_v,
            }
            cache_set(self._redis_key, _json.dumps(data).encode("utf-8"), ttl=self.cfg.state_ttl)
        except Exception as exc:
            logger.debug("RubySignalEngine.save_state(%s) error (non-fatal): %s", self.symbol, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bar: dict[str, Any] | _Bar) -> RubySignal:
        """Process a single confirmed bar and return a ``RubySignal``.

        ``bar`` may be a ``_Bar`` instance or a plain dict with keys:
        ``open``, ``high``, ``low``, ``close``, ``volume``, and an
        optional ``time`` (ISO-8601 string or datetime).

        This method matches the Pine Script execution model — it processes
        **one bar at a time** and maintains rolling state internally.  For
        live trading, call once per confirmed 1-minute bar.  For
        backtesting, iterate through a historical bar list.
        """
        b = self._normalise_bar(bar)
        if b is None:
            return RubySignal(symbol=self.symbol)

        # Append to history
        self._opens.append(b.open)
        self._highs.append(b.high)
        self._lows.append(b.low)
        self._closes.append(b.close)
        self._volumes.append(b.volume)
        self._times.append(b.time)
        self._bar_index += 1

        # Convert deques to arrays once (cheap — all elements are float)
        opens = np.array(self._opens, dtype=float)
        highs = np.array(self._highs, dtype=float)
        lows = np.array(self._lows, dtype=float)
        closes = np.array(self._closes, dtype=float)
        volumes = np.array(self._volumes, dtype=float)

        sig = self._compute(b, opens, highs, lows, closes, volumes)
        self._last_signal = sig

        # Persist state periodically (every 10 bars) to avoid Redis spam
        if self._bar_index % 10 == 0:
            self.save_state()

        return sig

    def last_signal(self) -> RubySignal | None:
        """Return the most recently computed signal (may be None if no bars yet)."""
        return self._last_signal

    def status(self) -> dict[str, Any]:
        """Return a human-readable status dict for dashboard / API use."""
        sig = self._last_signal
        if sig is None:
            return {
                "symbol": self.symbol,
                "bar_count": self._bar_index,
                "last_signal": None,
            }
        return {
            "symbol": self.symbol,
            "bar_count": self._bar_index,
            "last_signal": sig.to_dict(),
            "bull_waves_count": len(self._bull_waves),
            "bear_waves_count": len(self._bear_waves),
            "orb_ready": self._orb_ready,
            "ib_done": self._ib_done,
            "pd_high": self._pd_high,
            "pd_low": self._pd_low,
        }

    # ------------------------------------------------------------------
    # Core computation — runs once per bar
    # ------------------------------------------------------------------

    def _compute(
        self,
        bar: _Bar,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> RubySignal:
        cfg = self.cfg
        n = len(closes)

        # ──────────────────────────────────────────────────────────────
        # §2  Core indicators
        # ──────────────────────────────────────────────────────────────
        ema9_arr = _ema(closes, 9)
        ema9_val = float(ema9_arr[-1])
        vwap_val = _vwap_session(opens, highs, lows, closes, volumes)
        atr14_val = _atr14_last(highs, lows, closes, 14)
        vol_avg = _sma_last(volumes, 20)
        ao_val = _ao_last(highs, lows)
        rsi14_val = _rsi_last(closes, 14)

        # ──────────────────────────────────────────────────────────────
        # §3  Top G Channel
        # tg_lo = lowest(low, topGLen), tg_hi = highest(high, topGLen)
        # tg_mid = HMA( avg(tg_lo, tg_hi), 15 )
        # roc8_norm for momentum confirmation
        # ──────────────────────────────────────────────────────────────
        tg_len = cfg.top_g_len
        lo_window = lows[-tg_len:] if n >= tg_len else lows
        hi_window = highs[-tg_len:] if n >= tg_len else highs
        tg_lo = float(np.min(lo_window))
        tg_hi = float(np.max(hi_window))
        tg_range = tg_hi - tg_lo

        # HMA(15) mid-line — requires a series of avg(tg_lo_i, tg_hi_i)
        if n >= 2:
            lo_series = np.array(
                [float(np.min(lows[max(0, i - tg_len + 1) : i + 1])) for i in range(max(0, n - 30), n)]
            )
            hi_series = np.array(
                [float(np.max(highs[max(0, i - tg_len + 1) : i + 1])) for i in range(max(0, n - 30), n)]
            )
            mid_series = (lo_series + hi_series) / 2.0
            tg_mid = _hma(mid_series, 15)
        else:
            tg_mid = (tg_lo + tg_hi) / 2.0

        # Normalised ROC for momentum confirmation
        roc8_arr = _roc(closes, 8)
        valid_roc = roc8_arr[~np.isnan(roc8_arr)]
        roc_std = float(np.std(valid_roc[-200:])) if len(valid_roc) >= 10 else 1.0
        _roc_norm_cur = float(roc8_arr[-1] / roc_std) if (roc_std > 0 and not np.isnan(roc8_arr[-1])) else 0.0

        # roc_norm[2] — value 2 bars ago
        roc_norm_2 = (
            float(roc8_arr[-3] / roc_std)
            if (len(roc8_arr) >= 3 and roc_std > 0 and not np.isnan(roc8_arr[-3]))
            else 0.0
        )

        # ── Strong TG signals (entry-grade) ──────────────────────────
        # strong_bot: lows[-2] == tg_lo[-2] and low > tg_lo and lows[-3] == tg_lo[-3]
        #             and roc_norm[-3] < -2*sens
        strong_bot = False
        strong_top = False
        _simple_bot = False
        _simple_top = False

        if n >= 3:
            prev2_lo = float(np.min(lows[max(0, n - 3 - tg_len + 1) : n - 2 + 1]))  # tg_lo[1]
            prev3_lo = float(np.min(lows[max(0, n - 4 - tg_len + 1) : n - 3 + 1]))  # tg_lo[2]
            prev2_hi = float(np.max(highs[max(0, n - 3 - tg_len + 1) : n - 2 + 1]))  # tg_hi[1]

            # strong_bot
            if (
                lows[-2] == prev2_lo  # low[1] == tg_lo[1]
                and bar.low > tg_lo  # low > tg_lo (current)
                and lows[-3] == prev3_lo  # low[2] == tg_lo[2]
                and roc_norm_2 < -2.0 * cfg.sig_sens
            ):
                strong_bot = True

            # strong_top
            tg_lo_now_5 = float(np.min(lows[max(0, n - 6 - tg_len + 1) : n - 5 + 1])) if n >= 6 else tg_lo
            if (
                bar.high < tg_hi
                and highs[-2] == prev2_hi  # high[1] == tg_hi[1]
                and roc_norm_2 > 2.0 * cfg.sig_sens
                and tg_lo == tg_lo_now_5  # tg_lo == tg_lo[5]
            ):
                strong_top = True

            # simple signals (not entry-grade)
            if lows[-2] == prev2_lo and bar.low > tg_lo and not strong_bot:
                _simple_bot = True
            if highs[-2] == prev2_hi and bar.high < tg_hi and not strong_top:
                _simple_top = True

        # ──────────────────────────────────────────────────────────────
        # §4  Wave Analysis
        # Pine: c_rma = ta.rma(close, 10), o_rma = ta.rma(open, 10)
        #       dyn_ema = ta.ema(close, 20)
        #       speed += (c_rma - o_rma)  each bar
        #       On EMA20 crossover → record wave amplitude, reset tracking
        # ──────────────────────────────────────────────────────────────
        c_rma_val = _rma_last(closes, 10)
        o_rma_val = _rma_last(opens, 10)
        ema20_arr = _ema(closes, 20)
        ema20_val = float(ema20_arr[-1])
        ema20_prev = float(ema20_arr[-2]) if n >= 2 else ema20_val

        speed_delta = c_rma_val - o_rma_val
        self._wave_speed += speed_delta

        # Track extremes within current wave
        self._wave_speed_lo = min(self._wave_speed_lo, self._wave_speed)
        self._wave_speed_hi = max(self._wave_speed_hi, self._wave_speed)

        # Detect EMA20 crossovers (crossover = bull flip, crossunder = bear flip)
        prev_close = float(closes[-2]) if n >= 2 else bar.close

        bull_cross = (prev_close <= ema20_prev) and (bar.close > ema20_val)
        bear_cross = (prev_close >= ema20_prev) and (bar.close < ema20_val)

        if bull_cross:
            # Record completed bear wave amplitude
            amp = self._wave_speed_lo  # negative value
            self._bear_waves.insert(0, amp)
            if len(self._bear_waves) > 200:
                self._bear_waves.pop()
            self._last_wave_bar = self._bar_index
            # Reset speed tracking
            self._wave_speed = speed_delta
            self._wave_speed_lo = self._wave_speed
            self._wave_speed_hi = self._wave_speed
            self._prev_ema20_cross = 1

        elif bear_cross:
            # Record completed bull wave amplitude
            amp = self._wave_speed_hi  # positive value
            self._bull_waves.insert(0, amp)
            if len(self._bull_waves) > 200:
                self._bull_waves.pop()
            self._last_wave_bar = self._bar_index
            self._wave_speed = speed_delta
            self._wave_speed_lo = self._wave_speed
            self._wave_speed_hi = self._wave_speed
            self._prev_ema20_cross = -1

        # Update wave ratio & bias
        if self._bull_waves:
            self._bull_avg = float(np.mean(self._bull_waves))
        if self._bear_waves:
            self._bear_avg_v = abs(float(np.mean(self._bear_waves)))
        if self._bear_avg_v > 0:
            self._wave_ratio = self._bull_avg / self._bear_avg_v
        dom_val = self._bull_avg - self._bear_avg_v
        self._mkt_bias = "Bullish" if dom_val > 0 else ("Bearish" if dom_val < 0 else "Neutral")

        # trendspeed = HMA(speed, 5) — approximate using last speed deltas
        # cur_ratio: current bar momentum vs historical average
        if self._wave_speed > 0 and self._bull_avg > 0:
            cur_ratio = self._wave_speed / self._bull_avg
        elif self._wave_speed < 0 and self._bear_avg_v > 0:
            cur_ratio = abs(self._wave_speed) / self._bear_avg_v * -1.0
        else:
            cur_ratio = 0.0

        # ──────────────────────────────────────────────────────────────
        # §5  Market Regime + Phase
        # ──────────────────────────────────────────────────────────────
        sma200_arr = np.array([_sma_last(closes[: i + 1], 200) for i in range(max(0, n - 25), n)])
        sma200_val = float(_sma_last(closes, 200))

        sma200_20ago = float(_sma_last(closes[:-20], 200)) if n >= 20 else sma200_val

        # avg_chg = SMA(|change(ma_200)|, 100)
        avg_chg = float(np.mean(np.abs(np.diff(sma200_arr)))) if len(sma200_arr) >= 2 else 0.0

        # slope_n: normalised slope over 20 bars
        slope_n = 0.0
        if avg_chg > 0:
            slope_n = (sma200_val - sma200_20ago) / (avg_chg * 20.0)

        # vol_n: normalised return volatility
        if n >= 2:
            ret_changes = np.diff(closes) / np.maximum(np.abs(closes[:-1]), 1e-10)
            ret_s_val = float(np.std(ret_changes[-min(100, len(ret_changes)) :], ddof=0))
            ret_sma_val = (
                _sma_last(
                    np.array(
                        [
                            float(np.std(ret_changes[max(0, i - 100) : i + 1], ddof=0))
                            for i in range(max(0, len(ret_changes) - 50), len(ret_changes))
                        ]
                    ),
                    50,
                )
                if len(ret_changes) >= 10
                else ret_s_val
            )
        else:
            ret_s_val = 0.0
            ret_sma_val = 1.0

        vol_n = ret_s_val / ret_sma_val if ret_sma_val > 0 else 1.0

        if slope_n > 1.0:
            market_regime = "TRENDING ↑"
        elif slope_n < -1.0:
            market_regime = "TRENDING ↓"
        elif vol_n > 1.5:
            market_regime = "VOLATILE"
        elif vol_n < 0.8:
            market_regime = "RANGING"
        else:
            market_regime = "NEUTRAL"

        # Market phase — sticky, updated on conditions
        ao_prev = _ao_last(highs[:-1], lows[:-1]) if n >= 35 else ao_val

        # tg_hi[10] / tg_lo[10] — channel bounds 10 bars ago
        tg_hi_10 = float(np.max(highs[max(0, n - 11 - tg_len + 1) : n - 10 + 1])) if n >= 10 else tg_hi
        tg_lo_10 = float(np.min(lows[max(0, n - 11 - tg_len + 1) : n - 10 + 1])) if n >= 10 else tg_lo
        tg_hi_5 = float(np.max(highs[max(0, n - 6 - tg_len + 1) : n - 5 + 1])) if n >= 5 else tg_hi
        tg_lo_5 = float(np.min(lows[max(0, n - 6 - tg_len + 1) : n - 5 + 1])) if n >= 5 else tg_lo

        # Pine crossover(close, tg_hi[10]) — close crossed above tg_hi 10 bars ago
        if bar.close > tg_hi_10 and ao_val > 0:
            self._market_phase = "UPTREND"
        elif bar.close < tg_lo_10 and ao_val < 0:
            self._market_phase = "DOWNTREND"
        elif bar.close > tg_hi_5 and bar.close < tg_hi and ao_val > 0 and ao_val < ao_prev:
            self._market_phase = "DISTRIB"
        elif bar.close < tg_lo_5 and bar.close > tg_lo and ao_val < 0 and ao_val > ao_prev:
            self._market_phase = "ACCUM"

        # ──────────────────────────────────────────────────────────────
        # §6  Volatility Percentile
        # ──────────────────────────────────────────────────────────────
        self._vol_arr.append(atr14_val)
        vol_arr_np = np.array(self._vol_arr, dtype=float)
        vol_pct = float(np.sum(vol_arr_np < atr14_val) / len(vol_arr_np)) if len(vol_arr_np) > 1 else 0.5

        if vol_pct >= 0.8:
            vol_regime = "VERY HIGH"
        elif vol_pct >= 0.6:
            vol_regime = "HIGH"
        elif vol_pct <= 0.2:
            vol_regime = "VERY LOW"
        elif vol_pct <= 0.4:
            vol_regime = "LOW"
        else:
            vol_regime = "MED"

        # ──────────────────────────────────────────────────────────────
        # §7  Session Bias + ORB
        # ──────────────────────────────────────────────────────────────
        is_new_day = self._detect_new_day(bar.time)

        if is_new_day:
            # Roll PD levels
            if self._today_high is not None:
                self._pd_high = self._today_high
                self._pd_low = self._today_low
            else:
                # Seed from first bar on a new day if no session was seen
                self._pd_high = bar.high
                self._pd_low = bar.low
            # Reset today's levels
            self._today_high = bar.high
            self._today_low = bar.low
            # Reset ORB
            self._orb_high = None
            self._orb_low = None
            self._orb_ready = False
            self._orb_bars_today = 0
            # Reset IB
            self._ib_high = bar.high
            self._ib_low = bar.low
            self._ib_bars = 0
            self._ib_done = False
        else:
            # Update today's running H/L
            if self._today_high is None:
                self._today_high = bar.high
                self._today_low = bar.low
            else:
                self._today_high = max(self._today_high or bar.high, bar.high)
                self._today_low = min(self._today_low or bar.low, bar.low)

        # ORB formation
        minutes_per_bar = 1  # engine runs on 1-minute bars
        self._orb_bars_today += 1
        elapsed_min = self._orb_bars_today * minutes_per_bar

        if elapsed_min <= cfg.orb_minutes:
            # Still building the ORB
            if self._orb_high is None:
                self._orb_high = bar.high
                self._orb_low = bar.low
            else:
                self._orb_high = max(self._orb_high or bar.high, bar.high)
                self._orb_low = min(self._orb_low or bar.low, bar.low)
        elif not self._orb_ready and self._orb_high is not None:
            self._orb_ready = True

        # IB formation
        if not self._ib_done:
            if self._ib_high is None:
                self._ib_high = bar.high
                self._ib_low = bar.low
            else:
                self._ib_high = max(self._ib_high or bar.high, bar.high)
                self._ib_low = min(self._ib_low or bar.low, bar.low)
            self._ib_bars += 1
            if self._ib_bars * minutes_per_bar >= cfg.ib_minutes:
                self._ib_done = True

        # ── Bull bias vote (3-component, matches Pine bullBias) ───────
        vwap_vote = 1 if bar.close > vwap_val else 0
        ao_vote = 1 if ao_val > 0 else 0
        # HTF trend: approximate with EMA9 on current bars (no request.security())
        htf_vote = 1 if bar.close > ema9_val else 0

        if cfg.bias_mode == "Long Only":
            bull_bias = True
        elif cfg.bias_mode == "Short Only":
            bull_bias = False
        else:  # Auto — majority vote
            bull_bias = (vwap_vote + ao_vote + htf_vote) >= 2

        # ── ORB breakout detection ─────────────────────────────────────
        orb_high = self._orb_high or 0.0
        orb_low = self._orb_low or 0.0

        prev_close = float(closes[-2]) if n >= 2 else bar.close
        cross_hi = (prev_close <= orb_high) and (bar.close > orb_high) if orb_high > 0 else False
        cross_lo = (prev_close >= orb_low) and (bar.close < orb_low) if orb_low > 0 else False

        orb_break_up = self._orb_ready and cross_hi and bull_bias
        orb_break_dn = self._orb_ready and cross_lo and (not bull_bias)

        # ── Squeeze Detection ──────────────────────────────────────────
        bb_u, bb_l, kc_u, kc_l = self._compute_squeeze_bands(closes, highs, lows)
        sqz_on = (bb_l > kc_l) and (bb_u < kc_u)
        # sqz_fired: sqz_on[1] and not sqz_on
        if n >= 2:
            prev_sqz_closes = closes[:-1]
            prev_sqz_highs = highs[:-1]
            prev_sqz_lows = lows[:-1]
            pb_u, pb_l, pk_u, pk_l = self._compute_squeeze_bands(prev_sqz_closes, prev_sqz_highs, prev_sqz_lows)
            sqz_on_prev = (pb_l > pk_l) and (pb_u < pk_u)
        else:
            sqz_on_prev = False
        sqz_fired = sqz_on_prev and (not sqz_on)

        # ──────────────────────────────────────────────────────────────
        # §8  Quality Score (0–100)
        # ──────────────────────────────────────────────────────────────
        quality = 0.0

        # +20: AO momentum aligned with bias
        ao_prev_val = _ao_last(highs[:-1], lows[:-1]) if n >= 35 else 0.0
        if (bull_bias and ao_val > ao_prev_val) or (not bull_bias and ao_val < ao_prev_val):
            quality += 20.0

        # +15: close vs EMA9 aligned with bias
        if (bar.close > ema9_val and bull_bias) or (bar.close < ema9_val and not bull_bias):
            quality += 15.0

        # +20: close vs VWAP aligned with bias
        if (bull_bias and bar.close > vwap_val) or (not bull_bias and bar.close < vwap_val):
            quality += 20.0

        # +25: volume spike
        if vol_avg > 0 and bar.volume > vol_avg * cfg.vol_mult:
            quality += 25.0

        # +20: price beyond ORB range aligned with bias
        if (
            self._orb_ready
            and orb_high > 0
            and ((bull_bias and bar.close > orb_high) or (not bull_bias and bar.close < orb_low))
        ):
            quality += 20.0

        quality = min(100.0, max(0.0, quality))

        # ──────────────────────────────────────────────────────────────
        # §9  Main Signals (LONG / SHORT)
        # Pine: longSignal = strong_bot + bullBias + ao > 0 + volOK + vwapL + qualOK
        #       5-bar cooldown
        # ──────────────────────────────────────────────────────────────
        vol_ok = vol_avg > 0 and bar.volume > vol_avg * cfg.vol_mult
        vwap_l = (not cfg.require_vwap) or (bar.close > vwap_val)
        vwap_s = (not cfg.require_vwap) or (bar.close < vwap_val)
        qual_ok = quality >= cfg.min_quality_pct

        long_raw = strong_bot and bull_bias and ao_val > 0 and vol_ok and vwap_l and qual_ok
        short_raw = strong_top and (not bull_bias) and ao_val < 0 and vol_ok and vwap_s and qual_ok

        long_cooldown_ok = (self._bar_index - self._last_long_bar) > 5
        short_cooldown_ok = (self._bar_index - self._last_short_bar) > 5

        long_signal = long_raw and long_cooldown_ok
        short_signal = short_raw and short_cooldown_ok

        if long_signal:
            self._last_long_bar = self._bar_index
        if short_signal:
            self._last_short_bar = self._bar_index

        # Also detect ORB breakouts as signals
        # These are lower-priority: emitted when no TG signal fires
        orb_long = orb_break_up and (not long_signal)
        orb_short = orb_break_dn and (not short_signal)

        # ──────────────────────────────────────────────────────────────
        # §10  Entry / SL / TP Levels
        # ──────────────────────────────────────────────────────────────
        entry_price = bar.close
        direction = ""
        sl = 0.0
        risk_r = 0.0
        tp1 = tp2 = tp3 = 0.0
        signal_class = ""

        if long_signal:
            direction = "LONG"
            sl = bar.low - atr14_val
            risk_r = entry_price - sl
            tp1 = entry_price + risk_r * cfg.tp1_r
            tp2 = entry_price + risk_r * cfg.tp2_r
            tp3 = entry_price + risk_r * cfg.tp3_r
            signal_class = "TG_BOUNCE"

        elif short_signal:
            direction = "SHORT"
            sl = bar.high + atr14_val
            risk_r = sl - entry_price
            tp1 = entry_price - risk_r * cfg.tp1_r
            tp2 = entry_price - risk_r * cfg.tp2_r
            tp3 = entry_price - risk_r * cfg.tp3_r
            signal_class = "TG_REJECT"

        elif orb_long:
            direction = "LONG"
            sl = (orb_low - atr14_val) if orb_low > 0 else (entry_price - atr14_val)
            risk_r = entry_price - sl
            tp1 = entry_price + risk_r * cfg.tp1_r
            tp2 = entry_price + risk_r * cfg.tp2_r
            tp3 = entry_price + risk_r * cfg.tp3_r
            signal_class = "ORB_UP" if _in_orb_window(bar.time) else "RB_UP"

        elif orb_short:
            direction = "SHORT"
            sl = (orb_high + atr14_val) if orb_high > 0 else (entry_price + atr14_val)
            risk_r = sl - entry_price
            tp1 = entry_price - risk_r * cfg.tp1_r
            tp2 = entry_price - risk_r * cfg.tp2_r
            tp3 = entry_price - risk_r * cfg.tp3_r
            signal_class = "ORB_DN" if _in_orb_window(bar.time) else "RB_DN"

        breakout_detected = direction != ""

        # ── Map quality score to cnn_prob equivalent [0, 1] ──────────
        # The PositionManager uses cnn_prob for pyramid gates (min 0.65).
        # We normalise quality: 45 → 0.45, 80 → 0.80, 100 → 1.0
        cnn_prob_equiv = quality / 100.0

        # ── mtf_score: use wave_ratio as a proxy ─────────────────────
        # PM reversal gates require mtf_score ≥ 0.60; wave_ratio > 1 → bullish
        mtf_score_equiv = min(1.0, max(0.0, self._wave_ratio / 2.0))

        # ── is_orb_window ──────────────────────────────────────────────
        in_orb_win = _in_orb_window(bar.time)

        # ──────────────────────────────────────────────────────────────
        # Build output signal
        # ──────────────────────────────────────────────────────────────
        return RubySignal(
            # PositionManager-compatible
            symbol=self.symbol,
            direction=direction,
            trigger_price=entry_price,
            breakout_detected=breakout_detected,
            cnn_prob=cnn_prob_equiv,
            cnn_signal=cnn_prob_equiv >= 0.65,
            filter_passed=qual_ok or not breakout_detected,
            mtf_score=mtf_score_equiv,
            atr_value=atr14_val,
            range_high=tg_hi,
            range_low=tg_lo,
            # Ruby-specific
            quality=quality,
            regime=market_regime,
            phase=self._market_phase,
            wave_ratio=self._wave_ratio,
            cur_ratio=cur_ratio,
            mkt_bias=self._mkt_bias,
            bull_bias=bull_bias,
            vol_pct=vol_pct,
            vol_regime=vol_regime,
            ao=ao_val,
            vwap=vwap_val,
            ema9=ema9_val,
            rsi14=rsi14_val,
            tg_hi=tg_hi,
            tg_lo=tg_lo,
            tg_mid=tg_mid,
            tg_range=tg_range,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_ready=self._orb_ready,
            pd_high=self._pd_high or 0.0,
            pd_low=self._pd_low or 0.0,
            ib_high=self._ib_high or 0.0,
            ib_low=self._ib_low or 0.0,
            ib_done=self._ib_done,
            sqz_on=sqz_on,
            sqz_fired=sqz_fired,
            # Levels
            entry=entry_price if breakout_detected else 0.0,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk=risk_r,
            signal_class=signal_class,
            is_orb_window=in_orb_win,
            bar_time=bar.time,
            computed_at=datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Squeeze band helper
    # ------------------------------------------------------------------

    def _compute_squeeze_bands(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Return (bb_u, bb_l, kc_u, kc_l) — all last-bar values.

        BB: SMA(20) ± 2σ
        KC: SMA(20) ± ATR(20) × 1.5
        Matches Pine bb_u / bb_l / kc_u / kc_l in §7.
        """
        period = 20
        sma20 = _sma_last(closes, period)
        std20 = _stdev_last(closes, period)
        atr20 = _atr14_last(highs, lows, closes, period)

        bb_u = sma20 + 2.0 * std20
        bb_l = sma20 - 2.0 * std20
        kc_u = sma20 + atr20 * 1.5
        kc_l = sma20 - atr20 * 1.5
        return bb_u, bb_l, kc_u, kc_l

    # ------------------------------------------------------------------
    # New-day detection
    # ------------------------------------------------------------------

    def _detect_new_day(self, t: datetime) -> bool:
        """Return True on the first bar of a new calendar day (UTC)."""
        day = t.timetuple().tm_yday
        if self._last_day is None:
            self._last_day = day
            return True
        if day != self._last_day:
            self._last_day = day
            return True
        return False

    # ------------------------------------------------------------------
    # Bar normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_bar(bar: dict[str, Any] | _Bar) -> _Bar | None:
        """Convert a raw dict or _Bar into a normalised _Bar.

        Accepts both Title-case (High, Low, Close, Volume) and lower-case
        (high, low, close, volume) column names as well as ``open_price``,
        ``close_price``, etc. variants.
        """
        if isinstance(bar, _Bar):
            return bar

        if not isinstance(bar, dict):
            logger.warning("RubySignalEngine: invalid bar type %s — skipping", type(bar))
            return None

        def _get(keys: list[str], default: float = 0.0) -> float:
            for k in keys:
                v = bar.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return default

        def _get_time() -> datetime:
            for k in ("time", "timestamp", "datetime", "date", "t"):
                v = bar.get(k)
                if v is None:
                    continue
                if isinstance(v, datetime):
                    return v if v.tzinfo else v.replace(tzinfo=UTC)
                if isinstance(v, (int, float)):
                    return datetime.fromtimestamp(v, tz=UTC)
                if isinstance(v, str):
                    try:
                        dt = datetime.fromisoformat(v)
                        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                    except ValueError:
                        pass
            return datetime.now(UTC)

        o = _get(["open", "Open", "open_price"])
        h = _get(["high", "High", "high_price"])
        lo = _get(["low", "Low", "low_price"])
        c = _get(["close", "Close", "close_price"])
        v = _get(["volume", "Volume", "vol"])

        if c == 0.0:
            return None

        # If open not provided, fall back to close
        if o == 0.0:
            o = c
        if h == 0.0:
            h = max(o, c)
        if lo == 0.0:
            lo = min(o, c)

        return _Bar(time=_get_time(), open=o, high=h, low=lo, close=c, volume=v)


# ---------------------------------------------------------------------------
# Module-level registry — one engine per symbol
# ---------------------------------------------------------------------------

_engines: dict[str, RubySignalEngine] = {}


def get_ruby_engine(symbol: str, config: RubyConfig | None = None) -> RubySignalEngine:
    """Return (or create) the singleton RubySignalEngine for *symbol*.

    The engine is lazily instantiated on first call and state is loaded
    from Redis at that point.  Subsequent calls return the same instance.
    """
    if symbol not in _engines:
        eng = RubySignalEngine(symbol=symbol, config=config)
        eng.load_state()
        _engines[symbol] = eng
        logger.info("RubySignalEngine(%s) registered", symbol)
    return _engines[symbol]


def reset_ruby_engines() -> None:
    """Clear all registered engines (used in tests)."""
    _engines.clear()
