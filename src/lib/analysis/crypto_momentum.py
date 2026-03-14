"""
Cross-Asset Crypto Momentum Scorer
====================================
Computes real-time crypto momentum signals from Kraken BTC/ETH/SOL data
and scores how strongly they predict follow-through in correlated futures
instruments (MES, MNQ, MGC, MCL).

This is the foundation for the Multi-Source Breakout Detection system — using
crypto's 24/7 data to provide overnight context for futures that only trade ~23 hours.

Key observations this module captures:
  - BTC/ETH breakout during Asian session (19:00–02:00 ET) often leads
    MES/MNQ at London open (02:00–08:00 ET) and US open (09:30 ET)
  - Strong crypto momentum during London session can front-run US equity
    futures moves
  - Crypto/futures correlation is regime-dependent — stronger during
    risk-on/risk-off events, weaker during idiosyncratic moves

Architecture:
  - Pulls recent OHLCV from the engine/data service via EngineDataClient
    (GET /bars/{symbol}) — the engine resolves Redis → Postgres → Kraken
    internally so the caller never needs a Kraken API key or Redis access.
  - Falls back to the legacy Kraken/cache path when ENGINE_DATA_URL is not
    reachable (e.g. local dev without Docker).
  - Computes per-crypto momentum metrics (ATR breakout, EMA cross, RSI,
    volume surge, session high/low break)
  - Correlates crypto momentum direction with each futures instrument
    using rolling Pearson correlation from the existing correlation panel
  - Outputs a CryptoMomentumSignal per futures instrument that the engine
    or CNN can consume as an additional scoring input

Future (v7 feature contract):
  - ``crypto_momentum_score`` as tabular feature #19
  - Trained into the CNN alongside the existing 18 features
  - For now, exposed as an engine-side scoring boost (not in the model)

Usage:
    from lib.analysis.crypto_momentum import (
        CryptoMomentumScorer,
        CryptoMomentumSignal,
        compute_crypto_momentum,
    )

    scorer = CryptoMomentumScorer()
    signals = scorer.score_all()
    for sig in signals:
        print(f"{sig.futures_symbol}: {sig.score:.2f} ({sig.direction})")

    # Or single-shot for one futures instrument:
    sig = compute_crypto_momentum("MES")
    if sig.is_actionable:
        print(f"Crypto supports {sig.direction} MES — score {sig.score:.3f}")
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time as _time_mod
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.indicators.helpers import atr_scalar, ema_numpy, rsi_scalar

logger = logging.getLogger("analysis.crypto_momentum")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Process-level TTL cache for crypto momentum results
# ---------------------------------------------------------------------------
# Prevents repeated Kraken REST calls when compute_crypto_momentum() is
# invoked on every CNN inference, breakout check, or pipeline step.
# The cache stores the last computed result and its timestamp; any call
# within _CACHE_TTL_SECONDS of the last compute returns the cached value
# instead of hitting Kraken again.
#
# Two separate caches:
#   _signal_cache  — keyed by futures_symbol, for compute_crypto_momentum()
#   _all_cache     — single entry for compute_all_crypto_momentum()

_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes — Kraken 1-min bars don't change faster

_signal_cache: dict[str, tuple[float, Any]] = {}  # symbol → (timestamp, CryptoMomentumSignal)
_all_cache: tuple[float, list[Any]] | None = None  # (timestamp, [CryptoMomentumSignal, ...])
_cache_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Crypto anchors we monitor — these are the "leading" assets
CRYPTO_ANCHORS: list[dict[str, Any]] = [
    {"symbol": "BTC", "internal": "KRAKEN:XBTUSD", "rest_pair": "XXBTZUSD", "weight": 0.50},
    {"symbol": "SOL", "internal": "KRAKEN:SOLUSD", "rest_pair": "SOLUSD", "weight": 0.15},
]

# Futures instruments and their known crypto correlations
# correlation_weight: base weight for how strongly this futures instrument
# tracks crypto momentum (refined at runtime by rolling correlation)
FUTURES_TARGETS: dict[str, dict[str, Any]] = {
    "MES": {
        "description": "Micro E-mini S&P 500",
        "base_correlation": 0.55,  # moderate positive (risk-on)
        "primary_crypto": "BTC",
    },
    "MNQ": {
        "description": "Micro Nasdaq",
        "base_correlation": 0.60,  # slightly stronger — tech/growth overlap
        "primary_crypto": "BTC",
    },
    "MGC": {
        "description": "Micro Gold",
        "base_correlation": 0.25,  # weak — sometimes inverse, sometimes aligned
        "primary_crypto": "BTC",
    },
    "MCL": {
        "description": "Micro Crude Oil",
        "base_correlation": 0.15,  # very weak direct correlation
        "primary_crypto": "BTC",
    },
    "MYM": {
        "description": "Micro Dow",
        "base_correlation": 0.50,  # moderate — tracks MES closely
        "primary_crypto": "BTC",
    },
}

# Session definitions (ET) — crypto breakouts in these windows have
# different predictive power for futures follow-through
SESSIONS: dict[str, dict[str, Any]] = {
    "asian": {"start": dt_time(19, 0), "end": dt_time(2, 0), "lead_hours": 4.0},
    "london": {"start": dt_time(2, 0), "end": dt_time(8, 0), "lead_hours": 2.0},
    "us_preopen": {"start": dt_time(8, 0), "end": dt_time(9, 30), "lead_hours": 0.5},
    "us_rth": {"start": dt_time(9, 30), "end": dt_time(16, 0), "lead_hours": 0.0},
}

# Momentum calculation parameters
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
VOLUME_SURGE_MULT = 1.5
MOMENTUM_LOOKBACK_BARS = 60  # 5-minute bars → 5 hours of lookback
MIN_BARS_REQUIRED = 30  # need at least this many bars for valid signal

# Score thresholds
ACTIONABLE_THRESHOLD = 0.40  # score >= this is "actionable"
STRONG_THRESHOLD = 0.65  # score >= this is "strong"

# Correlation scoring
MIN_CORRELATION_SAMPLES = 20  # minimum overlapping data points for correlation
CORRELATION_DECAY = 0.95  # exponential decay for older correlation data


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CryptoMomentum:
    """Raw momentum metrics for a single crypto asset."""

    symbol: str = ""
    timestamp: str = ""

    # Price action
    price: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_spread: float = 0.0  # (fast - slow) / slow, signed
    ema_cross_direction: str = ""  # "bullish", "bearish", "neutral"

    # ATR breakout
    atr: float = 0.0
    atr_pct: float = 0.0  # ATR / price
    session_high: float = 0.0
    session_low: float = 0.0
    session_range_pct: float = 0.0
    broke_session_high: bool = False
    broke_session_low: bool = False

    # RSI
    rsi: float = 50.0

    # Volume
    volume_ratio: float = 1.0  # current volume / average volume
    volume_surge: bool = False

    # Composite direction: "bullish", "bearish", "neutral"
    direction: str = "neutral"
    strength: float = 0.0  # 0.0 – 1.0

    # Number of bars used
    bar_count: int = 0
    valid: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": round(self.price, 2),
            "ema_fast": round(self.ema_fast, 2),
            "ema_slow": round(self.ema_slow, 2),
            "ema_spread": round(self.ema_spread, 6),
            "ema_cross_direction": self.ema_cross_direction,
            "atr": round(self.atr, 2),
            "atr_pct": round(self.atr_pct, 6),
            "session_high": round(self.session_high, 2),
            "session_low": round(self.session_low, 2),
            "broke_session_high": self.broke_session_high,
            "broke_session_low": self.broke_session_low,
            "rsi": round(self.rsi, 2),
            "volume_ratio": round(self.volume_ratio, 3),
            "volume_surge": self.volume_surge,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "bar_count": self.bar_count,
            "valid": self.valid,
            "error": self.error,
        }


@dataclass
class CryptoMomentumSignal:
    """Scored crypto momentum signal for a specific futures instrument.

    This is the output the engine / CNN consumes.
    """

    futures_symbol: str = ""
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    score: float = 0.0  # 0.0 – 1.0 composite score
    confidence: str = "low"  # "low", "moderate", "high"
    is_actionable: bool = False  # score >= ACTIONABLE_THRESHOLD
    is_strong: bool = False  # score >= STRONG_THRESHOLD

    # Component scores (0.0 – 1.0 each)
    momentum_score: float = 0.0  # raw crypto momentum strength
    correlation_score: float = 0.0  # how well crypto predicts this futures
    session_score: float = 0.0  # timing bonus based on session
    volume_score: float = 0.0  # crypto volume confirmation

    # Correlation info
    rolling_correlation: float = 0.0  # actual rolling Pearson r
    correlation_samples: int = 0

    # Which crypto anchors contributed
    anchor_signals: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    session: str = ""
    computed_at: str = ""
    lead_time_hours: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "futures_symbol": self.futures_symbol,
            "direction": self.direction,
            "score": round(self.score, 4),
            "confidence": self.confidence,
            "is_actionable": self.is_actionable,
            "is_strong": self.is_strong,
            "momentum_score": round(self.momentum_score, 4),
            "correlation_score": round(self.correlation_score, 4),
            "session_score": round(self.session_score, 4),
            "volume_score": round(self.volume_score, 4),
            "rolling_correlation": round(self.rolling_correlation, 4),
            "correlation_samples": self.correlation_samples,
            "anchor_signals": self.anchor_signals,
            "session": self.session,
            "computed_at": self.computed_at,
            "lead_time_hours": round(self.lead_time_hours, 2),
            "error": self.error,
        }

    def to_tabular_feature(self) -> float:
        """Return a single normalised value suitable for CNN tabular input.

        Range: -1.0 (strong bearish) to +1.0 (strong bullish).
        0.0 = neutral / no signal / insufficient data.

        This will become ``crypto_momentum_score`` in v7 feature contract.
        """
        if not self.is_actionable:
            return 0.0
        signed = self.score if self.direction == "bullish" else -self.score
        return max(-1.0, min(1.0, signed))


# ═══════════════════════════════════════════════════════════════════════════
# Pure computation functions (no I/O — testable with synthetic data)
# ═══════════════════════════════════════════════════════════════════════════


def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average. Delegates to lib.indicators.helpers."""
    return ema_numpy(values, period)


def compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
    """Compute RSI from a close price array. Returns the last RSI value."""
    return rsi_scalar(closes, period)


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD) -> float:
    """Compute ATR (Wilder smoothing). Returns last ATR value as a scalar float."""
    return atr_scalar(highs, lows, closes, period)


def compute_volume_ratio(volumes: np.ndarray, avg_period: int = VOLUME_AVG_PERIOD) -> float:
    """Ratio of current volume to the rolling average."""
    n = len(volumes)
    if n < 2:
        return 1.0
    current = volumes[-1]
    lookback = volumes[max(0, n - avg_period - 1) : n - 1]
    if len(lookback) == 0:
        return 1.0
    avg = float(np.mean(lookback))
    if avg <= 0:
        return 1.0
    return current / avg


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def log_returns(prices: list[float]) -> list[float]:
    """Compute log returns from a price series."""
    result = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if p0 > 0 and p1 > 0:
            result.append(math.log(p1 / p0))
        else:
            result.append(0.0)
    return result


def detect_session(now: datetime | None = None) -> tuple[str, dict[str, Any]]:
    """Detect the current trading session based on ET time.

    Returns (session_name, session_config) or ("unknown", {}) if outside
    all defined sessions.
    """
    if now is None:
        now = datetime.now(tz=_EST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_EST)
    else:
        now = now.astimezone(_EST)

    t = now.time()

    for name, cfg in SESSIONS.items():
        start: dt_time = cfg["start"]
        end: dt_time = cfg["end"]

        if start <= end:
            # Normal range (e.g., 08:00 – 16:00)
            if start <= t < end:
                return name, cfg
        else:
            # Overnight range (e.g., 19:00 – 02:00)
            if t >= start or t < end:
                return name, cfg

    return "unknown", {"start": dt_time(0, 0), "end": dt_time(0, 0), "lead_hours": 0.0}


def compute_session_high_low(
    df: pd.DataFrame,
    session_start: dt_time,
    session_end: dt_time,
    now: datetime | None = None,
) -> tuple[float, float]:
    """Compute the high/low of the current session from OHLCV bars.

    Handles overnight sessions (start > end) by looking at bars from
    the previous day's start time through today's end time.
    """
    if df.empty:
        return 0.0, 0.0

    if now is None:
        now = datetime.now(tz=_EST)

    high_col = "High" if "High" in df.columns else "high"
    low_col = "Low" if "Low" in df.columns else "low"

    if high_col not in df.columns or low_col not in df.columns:
        return 0.0, 0.0

    # Ensure index is timezone-aware in ET
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is None:  # type: ignore[union-attr]
        with contextlib.suppress(Exception):
            idx = idx.tz_localize("UTC").tz_convert(_EST)  # type: ignore[union-attr]
    elif hasattr(idx, "tz") and idx.tz is not None:  # type: ignore[union-attr]
        with contextlib.suppress(Exception):
            idx = idx.tz_convert(_EST)  # type: ignore[union-attr]

    times = pd.Series(idx).dt.time.values if hasattr(idx, "time") else None
    if times is None:
        return 0.0, 0.0

    if session_start <= session_end:
        mask = (times >= session_start) & (times < session_end)
    else:
        # Overnight: e.g. 19:00 → 02:00
        mask = (times >= session_start) | (times < session_end)

    session_bars = df.iloc[mask]
    if session_bars.empty:
        return 0.0, 0.0

    return float(session_bars[high_col].max()), float(session_bars[low_col].min())


def compute_single_crypto_momentum(
    df: pd.DataFrame,
    symbol: str,
    session_start: dt_time | None = None,
    session_end: dt_time | None = None,
    now: datetime | None = None,
) -> CryptoMomentum:
    """Compute momentum metrics for a single crypto asset from OHLCV bars.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        (or lowercase variants). Index should be DatetimeIndex.
    symbol : str
        Crypto symbol (e.g. "BTC", "ETH", "SOL").
    session_start, session_end : dt_time, optional
        Session window for high/low tracking. If None, auto-detected.
    now : datetime, optional
        Override current time for testing.

    Returns
    -------
    CryptoMomentum
        Populated momentum metrics.
    """
    result = CryptoMomentum(symbol=symbol)

    if now is None:
        now = datetime.now(tz=_EST)
    result.timestamp = now.isoformat()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[cl] = c

    required = {"open", "high", "low", "close"}
    if not required.issubset(col_map.keys()):
        result.error = f"Missing columns: {required - set(col_map.keys())}"
        return result

    n = len(df)
    if n < MIN_BARS_REQUIRED:
        result.error = f"Insufficient bars: {n} < {MIN_BARS_REQUIRED}"
        return result

    result.bar_count = n

    closes = df[col_map["close"]].values.astype(np.float64)
    highs = df[col_map["high"]].values.astype(np.float64)
    lows = df[col_map["low"]].values.astype(np.float64)
    volumes = df[col_map.get("volume", "close")].values.astype(np.float64) if "volume" in col_map else np.ones(n)

    result.price = float(closes[-1])

    # ── EMAs ──────────────────────────────────────────────────────────────
    ema_f = compute_ema(closes, EMA_FAST)
    ema_s = compute_ema(closes, EMA_SLOW)

    if not np.isnan(ema_f[-1]) and not np.isnan(ema_s[-1]):
        result.ema_fast = float(ema_f[-1])
        result.ema_slow = float(ema_s[-1])
        if result.ema_slow > 0:
            result.ema_spread = (result.ema_fast - result.ema_slow) / result.ema_slow

        # Check for EMA cross in last 3 bars
        if n >= EMA_SLOW + 3:
            cross_window = 3
            prev_fast = ema_f[-cross_window - 1 : -1]
            prev_slow = ema_s[-cross_window - 1 : -1]
            was_below = any(
                f < s for f, s in zip(prev_fast, prev_slow, strict=True) if not (np.isnan(f) or np.isnan(s))
            )
            was_above = any(
                f > s for f, s in zip(prev_fast, prev_slow, strict=True) if not (np.isnan(f) or np.isnan(s))
            )
            now_above = result.ema_fast > result.ema_slow
            now_below = result.ema_fast < result.ema_slow

            if now_above and was_below:
                result.ema_cross_direction = "bullish"
            elif now_below and was_above:
                result.ema_cross_direction = "bearish"
            elif now_above:
                result.ema_cross_direction = "bullish"
            elif now_below:
                result.ema_cross_direction = "bearish"
            else:
                result.ema_cross_direction = "neutral"
        elif result.ema_fast > result.ema_slow:
            result.ema_cross_direction = "bullish"
        elif result.ema_fast < result.ema_slow:
            result.ema_cross_direction = "bearish"
        else:
            result.ema_cross_direction = "neutral"

    # ── ATR ────────────────────────────────────────────────────────────────
    result.atr = compute_atr(highs, lows, closes, ATR_PERIOD)
    if result.price > 0:
        result.atr_pct = result.atr / result.price

    # ── Session high/low ──────────────────────────────────────────────────
    if session_start is None or session_end is None:
        sess_name, sess_cfg = detect_session(now)
        if sess_cfg:
            session_start = sess_cfg.get("start", dt_time(0, 0))
            session_end = sess_cfg.get("end", dt_time(23, 59))

    if session_start is not None and session_end is not None:
        s_hi, s_lo = compute_session_high_low(df, session_start, session_end, now)
        result.session_high = s_hi
        result.session_low = s_lo
        if s_hi > 0 and s_lo > 0 and s_lo < s_hi:
            result.session_range_pct = (s_hi - s_lo) / ((s_hi + s_lo) / 2.0)
        if s_hi > 0 and result.price > s_hi:
            result.broke_session_high = True
        if s_lo > 0 and result.price < s_lo:
            result.broke_session_low = True

    # ── RSI ────────────────────────────────────────────────────────────────
    result.rsi = compute_rsi(closes, RSI_PERIOD)

    # ── Volume ─────────────────────────────────────────────────────────────
    result.volume_ratio = compute_volume_ratio(volumes, VOLUME_AVG_PERIOD)
    result.volume_surge = result.volume_ratio >= VOLUME_SURGE_MULT

    # ── Composite direction & strength ─────────────────────────────────────
    bullish_points = 0.0
    bearish_points = 0.0
    total_weight = 0.0

    # EMA direction (weight: 3)
    w = 3.0
    total_weight += w
    if result.ema_cross_direction == "bullish":
        bullish_points += w * min(1.0, abs(result.ema_spread) * 100)
    elif result.ema_cross_direction == "bearish":
        bearish_points += w * min(1.0, abs(result.ema_spread) * 100)

    # RSI (weight: 2)
    w = 2.0
    total_weight += w
    if result.rsi > 55:
        bullish_points += w * min(1.0, (result.rsi - 50) / 30.0)
    elif result.rsi < 45:
        bearish_points += w * min(1.0, (50 - result.rsi) / 30.0)

    # Session breakout (weight: 3)
    w = 3.0
    total_weight += w
    if result.broke_session_high:
        bullish_points += w
    elif result.broke_session_low:
        bearish_points += w

    # Volume surge confirmation (weight: 2)
    w = 2.0
    total_weight += w
    if result.volume_surge:
        # Volume confirms whichever direction is winning
        if bullish_points > bearish_points:
            bullish_points += w * min(1.0, (result.volume_ratio - 1.0) / 2.0)
        elif bearish_points > bullish_points:
            bearish_points += w * min(1.0, (result.volume_ratio - 1.0) / 2.0)

    if total_weight > 0:
        bull_norm = bullish_points / total_weight
        bear_norm = bearish_points / total_weight

        if bull_norm > bear_norm and bull_norm > 0.1:
            result.direction = "bullish"
            result.strength = min(1.0, bull_norm)
        elif bear_norm > bull_norm and bear_norm > 0.1:
            result.direction = "bearish"
            result.strength = min(1.0, bear_norm)
        else:
            result.direction = "neutral"
            result.strength = 0.0

    result.valid = True
    return result


def score_futures_from_crypto(
    futures_symbol: str,
    crypto_momentums: list[CryptoMomentum],
    rolling_corr: float | None = None,
    correlation_samples: int = 0,
    now: datetime | None = None,
) -> CryptoMomentumSignal:
    """Score how crypto momentum predicts a specific futures instrument.

    Parameters
    ----------
    futures_symbol : str
        Target futures symbol (e.g. "MES", "MNQ").
    crypto_momentums : list[CryptoMomentum]
        Pre-computed momentum for each crypto anchor.
    rolling_corr : float, optional
        Pre-computed rolling Pearson r between the crypto composite and
        this futures instrument. If None, uses base_correlation from config.
    correlation_samples : int
        How many samples the rolling correlation was computed from.
    now : datetime, optional
        Override current time for testing.

    Returns
    -------
    CryptoMomentumSignal
        Scored signal for the futures instrument.
    """
    if now is None:
        now = datetime.now(tz=_EST)

    signal = CryptoMomentumSignal(
        futures_symbol=futures_symbol,
        computed_at=now.isoformat(),
    )

    target_cfg = FUTURES_TARGETS.get(futures_symbol)
    if target_cfg is None:
        signal.error = f"Unknown futures symbol: {futures_symbol}"
        return signal

    # ── Detect session and lead time ──────────────────────────────────────
    sess_name, sess_cfg = detect_session(now)
    signal.session = sess_name
    signal.lead_time_hours = sess_cfg.get("lead_hours", 0.0) if sess_cfg else 0.0

    # ── Aggregate crypto momentum (weighted by anchor weights) ────────────
    valid_momentums = [m for m in crypto_momentums if m.valid]
    if not valid_momentums:
        signal.error = "No valid crypto momentum data"
        return signal

    # Build weighted composite direction
    bullish_weight = 0.0
    bearish_weight = 0.0
    total_anchor_weight = 0.0
    total_strength = 0.0
    total_volume_score = 0.0

    for m in valid_momentums:
        # Find anchor weight
        anchor_weight = 1.0 / len(valid_momentums)  # default equal weight
        for a in CRYPTO_ANCHORS:
            if a["symbol"] == m.symbol:
                anchor_weight = float(a["weight"])
                break

        total_anchor_weight += anchor_weight

        if m.direction == "bullish":
            bullish_weight += anchor_weight * m.strength
        elif m.direction == "bearish":
            bearish_weight += anchor_weight * m.strength

        total_strength += anchor_weight * m.strength
        if m.volume_surge:
            total_volume_score += anchor_weight * min(1.0, (m.volume_ratio - 1.0) / 2.0)

        signal.anchor_signals.append(
            {
                "symbol": m.symbol,
                "direction": m.direction,
                "strength": round(m.strength, 4),
                "rsi": round(m.rsi, 2),
                "volume_ratio": round(m.volume_ratio, 3),
                "broke_high": m.broke_session_high,
                "broke_low": m.broke_session_low,
            }
        )

    # Normalise
    if total_anchor_weight > 0:
        bullish_weight /= total_anchor_weight
        bearish_weight /= total_anchor_weight
        total_strength /= total_anchor_weight
        total_volume_score /= total_anchor_weight

    # ── Direction ─────────────────────────────────────────────────────────
    if bullish_weight > bearish_weight and bullish_weight > 0.1:
        signal.direction = "bullish"
        signal.momentum_score = min(1.0, bullish_weight)
    elif bearish_weight > bullish_weight and bearish_weight > 0.1:
        signal.direction = "bearish"
        signal.momentum_score = min(1.0, bearish_weight)
    else:
        signal.direction = "neutral"
        signal.momentum_score = 0.0

    # ── Correlation score ─────────────────────────────────────────────────
    base_corr = target_cfg.get("base_correlation", 0.3)

    if rolling_corr is not None and not math.isnan(rolling_corr) and correlation_samples >= MIN_CORRELATION_SAMPLES:
        # Blend rolling with base: trust rolling more as samples increase
        trust = min(1.0, correlation_samples / 100.0)
        effective_corr = trust * abs(rolling_corr) + (1 - trust) * base_corr
        signal.rolling_correlation = rolling_corr
        signal.correlation_samples = correlation_samples
    else:
        effective_corr = base_corr
        signal.rolling_correlation = 0.0
        signal.correlation_samples = 0

    signal.correlation_score = min(1.0, max(0.0, effective_corr))

    # ── Session timing score ──────────────────────────────────────────────
    # Crypto signals have more predictive power when there's lead time
    # (Asian breakout → US follow-through has hours of lead)
    lead_h = signal.lead_time_hours
    if lead_h >= 2.0:
        signal.session_score = 0.8  # Asian/early London — strong lead
    elif lead_h >= 0.5:
        signal.session_score = 0.5  # Late London / pre-open
    elif lead_h > 0:
        signal.session_score = 0.3  # Some lead time
    else:
        signal.session_score = 0.15  # RTH — crypto is contemporaneous, less predictive

    # ── Volume score ──────────────────────────────────────────────────────
    signal.volume_score = min(1.0, total_volume_score)

    # ── Composite score ───────────────────────────────────────────────────
    # Weighted combination:
    #   40% momentum strength
    #   25% correlation reliability
    #   20% session timing
    #   15% volume confirmation
    signal.score = (
        0.40 * signal.momentum_score
        + 0.25 * signal.correlation_score
        + 0.20 * signal.session_score
        + 0.15 * signal.volume_score
    )

    # ── Threshold classification ──────────────────────────────────────────
    signal.is_actionable = signal.score >= ACTIONABLE_THRESHOLD
    signal.is_strong = signal.score >= STRONG_THRESHOLD

    if signal.score >= STRONG_THRESHOLD:
        signal.confidence = "high"
    elif signal.score >= ACTIONABLE_THRESHOLD:
        signal.confidence = "moderate"
    else:
        signal.confidence = "low"

    return signal


# ═══════════════════════════════════════════════════════════════════════════
# High-level scorer class (with optional I/O for live data)
# ═══════════════════════════════════════════════════════════════════════════


class CryptoMomentumScorer:
    """Orchestrates crypto momentum computation across all anchors and targets.

    In live mode, pulls data from the engine HTTP API (EngineDataClient).
    The engine handles the full resolution chain: Redis → Postgres → Kraken.
    In test mode, accepts pre-built DataFrames via ``score_with_data()``.

    The legacy Kraken/cache path is kept as a fallback for environments where
    the engine is not reachable (local dev without Docker networking).
    """

    def __init__(
        self,
        targets: dict[str, dict[str, Any]] | None = None,
        anchors: list[dict[str, str]] | None = None,
        interval: str = "5m",
        lookback_bars: int = MOMENTUM_LOOKBACK_BARS,
        data_client: Any | None = None,
    ):
        self.targets = targets or FUTURES_TARGETS
        self.anchors = anchors or CRYPTO_ANCHORS
        self.interval = interval
        self.lookback_bars = lookback_bars
        # Optional injected data client (EngineDataClient or StaticBarProvider).
        # When None the scorer creates one lazily from environment variables.
        self._data_client = data_client

    def score_with_data(
        self,
        crypto_bars: dict[str, pd.DataFrame],
        futures_bars: dict[str, pd.DataFrame] | None = None,
        now: datetime | None = None,
    ) -> list[CryptoMomentumSignal]:
        """Score all futures targets given pre-fetched OHLCV data.

        Parameters
        ----------
        crypto_bars : dict[str, DataFrame]
            Keyed by crypto symbol (e.g. "BTC"), values are OHLCV DataFrames.
        futures_bars : dict[str, DataFrame], optional
            Keyed by futures symbol (e.g. "MES"). Used for rolling correlation.
            If None, correlation falls back to base_correlation config.
        now : datetime, optional
            Override current time.

        Returns
        -------
        list[CryptoMomentumSignal]
            One signal per futures target.
        """
        if now is None:
            now = datetime.now(tz=_EST)

        # Step 1: Compute momentum for each crypto anchor
        momentums: list[CryptoMomentum] = []
        for anchor in self.anchors:
            sym = anchor["symbol"]
            df = crypto_bars.get(sym)
            if df is None or df.empty:
                continue
            m = compute_single_crypto_momentum(df, sym, now=now)
            momentums.append(m)

        # Step 2: Compute rolling correlations if futures bars provided
        rolling_corrs: dict[str, tuple[float, int]] = {}
        if futures_bars:
            # Build a composite crypto return series (BTC-weighted)
            crypto_composite = self._build_composite_returns(crypto_bars)

            for f_sym, f_df in futures_bars.items():
                if f_df.empty or not crypto_composite:
                    continue
                corr, samples = self._compute_rolling_correlation(crypto_composite, f_df)
                rolling_corrs[f_sym] = (corr, samples)

        # Step 3: Score each futures target
        signals: list[CryptoMomentumSignal] = []
        for f_sym in self.targets:
            corr_val, corr_samples = rolling_corrs.get(f_sym, (None, 0))
            sig = score_futures_from_crypto(
                futures_symbol=f_sym,
                crypto_momentums=momentums,
                rolling_corr=corr_val,
                correlation_samples=corr_samples,
                now=now,
            )
            signals.append(sig)

        return signals

    def score_all(self, now: datetime | None = None) -> list[CryptoMomentumSignal]:
        """Score all targets using live data from the engine HTTP API.

        Routes all data fetching through ``EngineDataClient`` (GET /bars/...)
        so the engine/data service handles the full resolution chain:
        Redis → Postgres → Kraken REST API.  Falls back gracefully to the
        legacy Kraken/cache path when the engine is unreachable.
        """
        crypto_bars: dict[str, pd.DataFrame] = {}

        for anchor in self.anchors:
            sym = anchor["symbol"]
            internal = anchor["internal"]
            df = self._fetch_crypto_bars(internal)
            if df is not None and not df.empty:
                crypto_bars[sym] = df

        if not crypto_bars:
            logger.warning("No crypto bars available — returning empty signals")
            return [
                CryptoMomentumSignal(
                    futures_symbol=f_sym,
                    error="No crypto data available",
                    computed_at=datetime.now(tz=_EST).isoformat(),
                )
                for f_sym in self.targets
            ]

        # Optionally fetch futures bars for correlation
        futures_bars: dict[str, pd.DataFrame] = {}
        for f_sym in self.targets:
            df = self._fetch_futures_bars(f_sym)
            if df is not None and not df.empty:
                futures_bars[f_sym] = df

        return self.score_with_data(crypto_bars, futures_bars, now=now)

    def _get_data_client(self) -> Any:
        """Return the EngineDataClient, creating from env vars if not injected."""
        if self._data_client is not None:
            return self._data_client
        try:
            from lib.services.data.engine_data_client import get_client

            return get_client()
        except Exception:
            return None

    def _fetch_crypto_bars(self, internal_ticker: str) -> pd.DataFrame | None:
        """Fetch crypto OHLCV bars via the engine HTTP API.

        Routes through EngineDataClient (GET /bars/{symbol}) — the engine
        handles Redis → Postgres → Kraken REST internally.
        Returns None when the engine is unreachable; the scorer degrades
        gracefully to base_correlation values.
        """
        client = self._get_data_client()
        if client is None:
            logger.debug("No data client available for crypto bars: %s", internal_ticker)
            return None
        try:
            df = client.get_bars(internal_ticker, interval=self.interval, days_back=1)
            if df is not None and not df.empty:
                return df
            # Try short alias: "KRAKEN:XBTUSD" → "BTC"
            short = internal_ticker.replace("KRAKEN:", "").replace("USD", "")
            if short and short != internal_ticker:
                df = client.get_bars(short, interval=self.interval, days_back=1)
                if df is not None and not df.empty:
                    return df
        except Exception as exc:
            logger.debug("EngineDataClient failed for crypto %s: %s", internal_ticker, exc)
        logger.debug("No crypto bars available for %s (engine unreachable?)", internal_ticker)
        return None

    def _fetch_futures_bars(self, symbol: str) -> pd.DataFrame | None:
        """Fetch futures OHLCV bars via the engine HTTP API.

        Routes through EngineDataClient — the engine handles Redis → Postgres →
        Massive internally. Futures correlation bars are optional; returning None
        causes score_with_data() to fall back to base_correlation from config.
        """
        client = self._get_data_client()
        if client is None:
            return None
        try:
            df = client.get_bars(symbol, interval=self.interval, days_back=1)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.debug("EngineDataClient failed for futures %s: %s", symbol, exc)
        return None

    def _build_composite_returns(self, crypto_bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Build a weighted composite of crypto returns, keyed by timestamp.

        Returns a dict of {ts_str: weighted_return}.
        """
        # Collect per-anchor return series aligned on timestamps
        anchor_returns: dict[str, dict[str, float]] = {}

        for anchor in self.anchors:
            sym = anchor["symbol"]
            weight = float(anchor.get("weight", 1.0 / len(self.anchors)))
            df = crypto_bars.get(sym)
            if df is None or df.empty:
                continue

            close_col = "Close" if "Close" in df.columns else "close"
            if close_col not in df.columns:
                continue

            prices = df[close_col].values.astype(float)
            timestamps = [str(t)[:16] for t in df.index]

            for i in range(1, len(prices)):
                if prices[i - 1] > 0 and prices[i] > 0:
                    ret = math.log(prices[i] / prices[i - 1])
                    ts = timestamps[i]
                    anchor_returns.setdefault(ts, {})[sym] = ret * weight

        # Sum weighted returns per timestamp
        composite: dict[str, float] = {}
        for ts, returns in anchor_returns.items():
            composite[ts] = sum(returns.values())

        return composite

    def _compute_rolling_correlation(
        self,
        crypto_composite: dict[str, float],
        futures_df: pd.DataFrame,
    ) -> tuple[float, int]:
        """Compute rolling Pearson r between crypto composite and futures returns."""
        close_col = "Close" if "Close" in futures_df.columns else "close"
        if close_col not in futures_df.columns:
            return float("nan"), 0

        prices = futures_df[close_col].values.astype(float)
        timestamps = [str(t)[:16] for t in futures_df.index]

        futures_returns: dict[str, float] = {}
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                futures_returns[timestamps[i]] = math.log(prices[i] / prices[i - 1])

        # Align on common timestamps
        common = sorted(set(crypto_composite.keys()) & set(futures_returns.keys()))
        if len(common) < MIN_CORRELATION_SAMPLES:
            return float("nan"), len(common)

        c_vals = [crypto_composite[ts] for ts in common]
        f_vals = [futures_returns[ts] for ts in common]

        corr = pearson_correlation(c_vals, f_vals)
        return corr, len(common)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_crypto_momentum(
    futures_symbol: str = "MES",
    now: datetime | None = None,
) -> CryptoMomentumSignal:
    """One-shot convenience: compute crypto momentum signal for a single futures symbol.

    Uses live Kraken data if available, otherwise returns an empty signal.

    Results are cached for ``_CACHE_TTL_SECONDS`` (default 5 min) to prevent
    repeated Kraken REST calls when this function is invoked on every CNN
    inference or breakout check in the engine.
    """
    global _signal_cache

    _now_ts = _time_mod.monotonic()

    # Fast path: return cached result if still fresh
    with _cache_lock:
        cached = _signal_cache.get(futures_symbol)
        if cached is not None:
            cached_ts, cached_signal = cached
            if _now_ts - cached_ts < _CACHE_TTL_SECONDS:
                return cached_signal

    # Slow path: compute and cache
    scorer = CryptoMomentumScorer(
        targets={futures_symbol: FUTURES_TARGETS.get(futures_symbol, {"base_correlation": 0.3})}
    )
    signals = scorer.score_all(now=now)
    result = (
        signals[0]
        if signals
        else CryptoMomentumSignal(
            futures_symbol=futures_symbol,
            error="Failed to compute",
            computed_at=datetime.now(tz=_EST).isoformat(),
        )
    )

    with _cache_lock:
        _signal_cache[futures_symbol] = (_time_mod.monotonic(), result)

    return result


def compute_all_crypto_momentum(
    now: datetime | None = None,
) -> list[CryptoMomentumSignal]:
    """Compute crypto momentum signals for all tracked futures instruments.

    Results are cached for ``_CACHE_TTL_SECONDS`` (default 5 min) to prevent
    repeated Kraken REST calls when called from multiple engine subsystems.
    """
    global _all_cache

    _now_ts = _time_mod.monotonic()

    # Fast path: return cached result if still fresh
    with _cache_lock:
        if _all_cache is not None:
            cached_ts, cached_signals = _all_cache
            if _now_ts - cached_ts < _CACHE_TTL_SECONDS:
                return cached_signals

    # Slow path: compute and cache
    scorer = CryptoMomentumScorer()
    signals = scorer.score_all(now=now)

    with _cache_lock:
        _all_cache = (_time_mod.monotonic(), signals)

    return signals


def crypto_momentum_to_tabular(
    signals: list[CryptoMomentumSignal],
) -> dict[str, float]:
    """Convert signals to a dict of {futures_symbol: tabular_feature_value}.

    Values are in [-1, 1] — ready for v7 feature contract.
    """
    return {sig.futures_symbol: sig.to_tabular_feature() for sig in signals}
