"""
Wave Analysis Module.

This module implements the core wave dominance tracking:
  - Dynamic accelerated EMA (adaptive alpha based on momentum change)
  - Bull/bear wave tracking (magnitude + duration per wave)
  - Wave ratio: bull_avg / |bear_avg| — directional strength
  - Current ratio: current speed / avg wave — how extended the current move is
  - Dominance: net bull vs bear strength — overall market bias
  - Trend speed: smoothed cumulative momentum from wave transitions

All calculations are stateless (pure functions on DataFrames), designed to
be called once per refresh cycle and cached. No new dependencies — pure
NumPy/pandas.

Note: This module was renamed from ``wave.py`` to ``wave_analysis.py`` to
avoid shadowing Python's built-in ``wave`` (audio) module.

Usage:
    from lib.wave_analysis import calculate_wave_analysis

    result = calculate_wave_analysis(df)
    # result = {
    #     "wave_ratio": 1.85,
    #     "current_ratio": 0.73,
    #     "dominance": 0.312,
    #     "trend_speed": 2.45,
    #     "bias": "BULLISH",
    #     "trend_direction": "BULLISH ↗️",
    #     "trend_strength": "Strong",
    #     "bull_avg": 3.21,
    #     "bear_avg": -1.74,
    #     "bull_max": 8.50,
    #     "bear_max": -4.20,
    #     "market_phase": "UPTREND",
    #     "momentum_state": "ACCELERATING",
    # }
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("wave_analysis")

# ---------------------------------------------------------------------------
# Asset-specific tuning (ported from fks.pine barstate.isfirst block)
# ---------------------------------------------------------------------------
# Covers all assets in models.ASSETS (CME micro futures + Kraken spot crypto).
#
# Tuning guide:
#   max_length        — controls EMA responsiveness (lower = faster)
#   accel_multiplier  — how aggressively the EMA accelerates on big moves
#   lookback_period   — number of completed waves to include in ratio stats
#
# Asset class defaults:
#   Metals             — slow EMA (20), moderate accel, long lookback
#   Energy             — medium EMA (20), higher accel (volatile)
#   Equity index       — fast EMA (15–18), low accel (mean-reverting)
#   FX futures         — medium EMA (18), low accel (range-bound)
#   Rates (T-Note/Bond)— slow EMA (22), very low accel (macro-driven)
#   Ags (corn/wheat)   — medium EMA (20), moderate accel (seasonal)
#   CME crypto futures — fast EMA (14), high accel (crypto-like vol)
#   Kraken spot crypto — very fast EMA (12), highest accel (24/7 moves)
# ---------------------------------------------------------------------------

ASSET_PARAMS: dict[str, dict[str, float | int]] = {
    # ── Metals ──────────────────────────────────────────────────────────────
    "Gold": {"max_length": 20, "accel_multiplier": 0.015, "lookback_period": 200},
    "Silver": {"max_length": 20, "accel_multiplier": 0.015, "lookback_period": 200},
    "Copper": {"max_length": 20, "accel_multiplier": 0.015, "lookback_period": 200},
    # ── Energy ──────────────────────────────────────────────────────────────
    "Crude Oil": {"max_length": 20, "accel_multiplier": 0.02, "lookback_period": 150},
    "Natural Gas": {"max_length": 18, "accel_multiplier": 0.025, "lookback_period": 150},
    # ── Equity index ────────────────────────────────────────────────────────
    "S&P": {"max_length": 18, "accel_multiplier": 0.01, "lookback_period": 150},
    "Nasdaq": {"max_length": 15, "accel_multiplier": 0.01, "lookback_period": 150},
    "Russell 2000": {"max_length": 18, "accel_multiplier": 0.012, "lookback_period": 150},
    "Dow Jones": {"max_length": 18, "accel_multiplier": 0.01, "lookback_period": 150},
    # ── FX futures ──────────────────────────────────────────────────────────
    "Euro FX": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    "British Pound": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    "Japanese Yen": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    "Australian Dollar": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    "Canadian Dollar": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    "Swiss Franc": {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180},
    # ── Interest rate futures ────────────────────────────────────────────────
    "10Y T-Note": {"max_length": 22, "accel_multiplier": 0.006, "lookback_period": 200},
    "30Y T-Bond": {"max_length": 22, "accel_multiplier": 0.006, "lookback_period": 200},
    # ── Agricultural futures ─────────────────────────────────────────────────
    "Corn": {"max_length": 20, "accel_multiplier": 0.018, "lookback_period": 180},
    "Soybeans": {"max_length": 20, "accel_multiplier": 0.018, "lookback_period": 180},
    "Wheat": {"max_length": 20, "accel_multiplier": 0.020, "lookback_period": 180},
    # ── CME crypto futures ───────────────────────────────────────────────────
    "Micro Bitcoin": {"max_length": 14, "accel_multiplier": 0.030, "lookback_period": 120},
    "Micro Ether": {"max_length": 14, "accel_multiplier": 0.030, "lookback_period": 120},
    # ── Kraken spot crypto (24/7) ────────────────────────────────────────────
    "BTC/USD": {"max_length": 12, "accel_multiplier": 0.035, "lookback_period": 120},
    "ETH/USD": {"max_length": 12, "accel_multiplier": 0.035, "lookback_period": 120},
    "SOL/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "LINK/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "AVAX/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "DOT/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "ADA/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "POL/USD": {"max_length": 12, "accel_multiplier": 0.040, "lookback_period": 100},
    "XRP/USD": {"max_length": 12, "accel_multiplier": 0.038, "lookback_period": 100},
}

DEFAULT_PARAMS: dict[str, float | int] = {"max_length": 20, "accel_multiplier": 0.02, "lookback_period": 200}

# ---------------------------------------------------------------------------
# Ticker → asset-name lookup (built lazily from models.ASSETS / KRAKEN_CONTRACT_SPECS)
# ---------------------------------------------------------------------------
# Allows callers that only have a ticker (e.g. "MGC=F", "KRAKEN:XBTUSD") to
# resolve the human-readable asset name used as the key in ASSET_PARAMS.

_ticker_to_asset_name: dict[str, str] | None = None


def _build_ticker_to_asset_name() -> dict[str, str]:
    """Build a reverse map from Yahoo/Kraken ticker → asset name string.

    Reads from ``models.CONTRACT_SPECS`` and ``models.KRAKEN_CONTRACT_SPECS``
    so the map always reflects the current set of tracked assets without
    hardcoding tickers here.
    """
    result: dict[str, str] = {}
    try:
        from lib.core.models import KRAKEN_CONTRACT_SPECS, MICRO_CONTRACT_SPECS

        for name, spec in MICRO_CONTRACT_SPECS.items():
            for field in ("ticker", "data_ticker"):
                t = spec.get(field, "")
                if t:
                    result[str(t)] = name
        for name, spec in KRAKEN_CONTRACT_SPECS.items():
            for field in ("ticker", "data_ticker"):
                t = spec.get(field, "")
                if t:
                    result[str(t)] = name
    except Exception:
        pass
    return result


def resolve_asset_params(
    asset_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, float | int]:
    """Return the wave-analysis tuning parameters for an asset.

    Accepts either an asset name (``"Gold"``, ``"BTC/USD"``) or a ticker
    (``"MGC=F"``, ``"KRAKEN:XBTUSD"``).  Falls back to ``DEFAULT_PARAMS``
    when neither resolves to a known entry.

    This is the preferred way for callers to get params instead of
    accessing ``ASSET_PARAMS`` directly, since it handles ticker →
    asset-name resolution automatically.

    Parameters
    ----------
    asset_name:
        Human-readable asset name as used in ``models.MICRO_CONTRACT_SPECS``
        (e.g. ``"Gold"``, ``"S&P"``, ``"Micro Bitcoin"``).
    ticker:
        Yahoo-style or Kraken ticker (e.g. ``"MGC=F"``, ``"KRAKEN:XBTUSD"``).
        Only used when ``asset_name`` is ``None`` or not found.

    Returns
    -------
    dict with keys ``max_length``, ``accel_multiplier``, ``lookback_period``.
    """
    global _ticker_to_asset_name

    # 1. Direct lookup by asset name
    if asset_name and asset_name in ASSET_PARAMS:
        return ASSET_PARAMS[asset_name]

    # 2. Resolve ticker → asset name → params
    if ticker:
        if _ticker_to_asset_name is None:
            _ticker_to_asset_name = _build_ticker_to_asset_name()
        resolved_name = _ticker_to_asset_name.get(ticker)
        if resolved_name and resolved_name in ASSET_PARAMS:
            return ASSET_PARAMS[resolved_name]

    # 3. Heuristic fallback: classify by ticker/name prefix
    probe = (asset_name or ticker or "").upper()
    if any(probe.startswith(p) for p in ("KRAKEN:", "BTC", "ETH", "SOL", "AVAX", "LINK", "DOT", "ADA", "XRP")):
        # Spot crypto — fast/aggressive
        return {"max_length": 12, "accel_multiplier": 0.035, "lookback_period": 100}
    if any(probe.startswith(p) for p in ("MBT", "MET", "BTC=F", "ETH=F")):
        # CME crypto futures
        return {"max_length": 14, "accel_multiplier": 0.030, "lookback_period": 120}
    if any(probe.startswith(p) for p in ("6E", "6B", "6J", "6A", "6C", "6S", "M6")):
        # FX futures
        return {"max_length": 18, "accel_multiplier": 0.008, "lookback_period": 180}
    if any(probe.startswith(p) for p in ("ZN", "ZB", "ZF", "ZT")):
        # Rates
        return {"max_length": 22, "accel_multiplier": 0.006, "lookback_period": 200}
    if any(probe.startswith(p) for p in ("ZC", "ZS", "ZW", "ZL", "ZM")):
        # Ags
        return {"max_length": 20, "accel_multiplier": 0.018, "lookback_period": 180}
    if any(probe.startswith(p) for p in ("MES", "MNQ", "M2K", "MYM", "ES", "NQ", "RTY", "YM")):
        # Equity index
        return {"max_length": 18, "accel_multiplier": 0.010, "lookback_period": 150}
    if any(probe.startswith(p) for p in ("MCL", "MNG", "CL", "NG")):
        # Energy
        return {"max_length": 20, "accel_multiplier": 0.022, "lookback_period": 150}

    return dict(DEFAULT_PARAMS)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero guard — mirrors Pine safeDivide()."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Core: Dynamic Accelerated EMA (ported from fks.pine)
# ---------------------------------------------------------------------------


def _compute_dynamic_ema(
    close: np.ndarray,
    max_length: int = 20,
    accel_multiplier: float = 0.02,
) -> np.ndarray:
    """Compute the dynamic EMA with adaptive alpha.

    Port of fks.pine's dyn_ema logic:
      1. Normalize close into [0, 1] range over rolling 200-bar window
      2. Map to dynamic length between 5 and max_length
      3. Compute acceleration factor from bar-to-bar change magnitude
      4. Adjust alpha = base_alpha * (1 + accel * multiplier * 0.8), capped at 0.9
      5. Apply recursive EMA: ema[i] = alpha * close[i] + (1-alpha) * ema[i-1]
    """
    n = len(close)
    if n < 2:
        return close.copy()

    dyn_ema = np.empty(n)
    dyn_ema[0] = close[0]

    for i in range(1, n):
        # Rolling 200-bar window for normalization
        lookback = min(i + 1, 200)
        window = close[max(0, i - lookback + 1) : i + 1]
        max_abs = np.max(np.abs(window))
        max_abs = max(max_abs, 1e-10)

        # Normalize close to [0, 1]
        counts_diff_norm = (close[i] + max_abs) / (2 * max_abs)

        # Dynamic length
        dyn_length = 5 + counts_diff_norm * (max_length - 5)

        # Acceleration factor: normalized magnitude of bar-to-bar change
        delta = abs(close[i] - close[i - 1])
        lookback_accel = min(i + 1, 200)
        delta_window = np.abs(np.diff(close[max(0, i - lookback_accel) : i + 1]))
        max_delta = np.max(delta_window) if len(delta_window) > 0 else 1.0
        max_delta = max(max_delta, 1e-10)
        accel_factor = delta / max_delta

        # Adjusted alpha (capped at 0.9 per Pine logic)
        alpha_base = 2.0 / (dyn_length + 1.0)
        alpha = alpha_base * (1.0 + accel_factor * accel_multiplier * 0.8)
        alpha = min(alpha, 0.9)

        dyn_ema[i] = alpha * close[i] + (1.0 - alpha) * dyn_ema[i - 1]

    return dyn_ema


# ---------------------------------------------------------------------------
# Core: Wave detection and tracking
# ---------------------------------------------------------------------------


def _detect_waves(
    close: np.ndarray,
    dyn_ema: np.ndarray,
) -> tuple[list[float], list[float], list[int], list[int]]:
    """Detect bull and bear waves from price crossing the dynamic EMA.

    Port of fks.pine's wave transition logic:
      - When close crosses above dyn_ema → end of bear wave, start bull wave
      - When close crosses below dyn_ema → end of bull wave, start bear wave
      - Track cumulative speed (smoothed close - smoothed open) per wave
      - Record magnitude of each completed wave

    Returns:
        bull_changes: list of completed bull wave magnitudes (positive)
        bear_changes: list of completed bear wave magnitudes (negative)
        bull_durations: list of bull wave durations in bars
        bear_durations: list of bear wave durations in bars
    """
    n = len(close)
    if n < 20:
        return [0.0001], [-0.0001], [1], [1]

    # Smoothed close and open (RMA-10 approximation via EWM)
    close_s = pd.Series(close)
    c_smooth = np.asarray(close_s.ewm(alpha=1.0 / 10.0, adjust=False).mean())
    # For 'open', approximate with shifted close (open ≈ prior close in OHLCV)
    o_smooth = np.roll(c_smooth, 1)
    o_smooth[0] = c_smooth[0]

    bull_changes: list[float] = []
    bear_changes: list[float] = []
    bull_durations: list[int] = []
    bear_durations: list[int] = []

    # State tracking
    wave_start_idx = 0
    speed = 0.0
    pos = 0  # 1 = in bull wave, -1 = in bear wave, 0 = undetermined

    # Track lowest/highest speed within each wave for magnitude recording
    lowest_speed = 0.0
    highest_speed = 0.0

    for i in range(1, n):
        # Accumulate speed
        speed += c_smooth[i] - o_smooth[i]

        # Track extremes within current wave
        lowest_speed = min(lowest_speed, speed)
        highest_speed = max(highest_speed, speed)

        # Bull-to-bear transition: close crosses below EMA
        if close[i] < dyn_ema[i] and close[i - 1] >= dyn_ema[i - 1]:
            if pos == 1:
                # Ending a bull wave — record the highest speed achieved
                bull_changes.append(highest_speed)
                bull_durations.append(i - wave_start_idx)
            wave_start_idx = i
            pos = -1
            speed = c_smooth[i] - o_smooth[i]
            lowest_speed = speed
            highest_speed = speed

        # Bear-to-bull transition: close crosses above EMA
        elif close[i] > dyn_ema[i] and close[i - 1] <= dyn_ema[i - 1]:
            if pos == -1:
                # Ending a bear wave — record the lowest speed achieved
                bear_changes.append(lowest_speed)
                bear_durations.append(i - wave_start_idx)
            wave_start_idx = i
            pos = 1
            speed = c_smooth[i] - o_smooth[i]
            lowest_speed = speed
            highest_speed = speed

    # Ensure we always have at least one entry
    if not bull_changes:
        bull_changes = [0.0001]
        bull_durations = [1]
    if not bear_changes:
        bear_changes = [-0.0001]
        bear_durations = [1]

    return bull_changes, bear_changes, bull_durations, bear_durations


# ---------------------------------------------------------------------------
# Core: Trend speed (HMA-smoothed cumulative momentum)
# ---------------------------------------------------------------------------


def _compute_trend_speed(close: np.ndarray, dyn_ema: np.ndarray) -> np.ndarray:
    """Compute bar-by-bar trend speed, reset at wave transitions.

    Port of fks.pine's speed accumulation + HMA(5) smoothing.
    """
    n = len(close)
    if n < 5:
        return np.zeros(n)

    close_s = pd.Series(close)
    c_smooth = np.asarray(close_s.ewm(alpha=1.0 / 10.0, adjust=False).mean())
    o_smooth = np.roll(c_smooth, 1)
    o_smooth[0] = c_smooth[0]

    raw_speed = np.zeros(n)
    speed = 0.0

    for i in range(1, n):
        # Reset speed on wave transitions
        if (close[i] > dyn_ema[i] and close[i - 1] <= dyn_ema[i - 1]) or (
            close[i] < dyn_ema[i] and close[i - 1] >= dyn_ema[i - 1]
        ):
            speed = c_smooth[i] - o_smooth[i]
        else:
            speed += c_smooth[i] - o_smooth[i]
        raw_speed[i] = speed

    # HMA(5) smoothing — approximate with WMA(WMA(n/2), sqrt(n))
    # For period 5: WMA of WMA(2) and WMA(3) combined, then WMA(2) on result
    # Simplified: use pandas WMA approximation
    speed_series = pd.Series(raw_speed)

    # Hull Moving Average period 5
    half_period = max(int(5 / 2), 1)
    sqrt_period = max(int(np.sqrt(5)), 1)

    wma_half = speed_series.rolling(half_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = speed_series.rolling(5).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
    hull_input = 2 * wma_half - wma_full
    trend_speed = hull_input.rolling(sqrt_period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )

    result = trend_speed.fillna(0).values
    return result


# ---------------------------------------------------------------------------
# Market phase detection (ported from fks.pine)
# ---------------------------------------------------------------------------


def _detect_market_phase(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ao: np.ndarray,
) -> str:
    """Detect market phase: UPTREND, DOWNTREND, ACCUMULATION, DISTRIBUTION.

    Simplified port of fks.pine's market phase logic using Awesome Oscillator
    and price relative to recent highs/lows.
    """
    if len(close) < 20:
        return "ACCUMULATION"

    # Recent highest high and lowest low (10-bar lookback)
    recent_high = np.max(high[-10:])
    recent_low = np.min(low[-10:])
    _wider_high = np.max(high[-20:]) if len(high) >= 20 else recent_high  # noqa: F841
    _wider_low = np.min(low[-20:]) if len(low) >= 20 else recent_low  # noqa: F841

    current_close = close[-1]
    current_ao = ao[-1] if len(ao) > 0 else 0.0
    ao_10_ago = ao[-11] if len(ao) > 11 else 0.0

    # Uptrend: price breaking recent highs with positive AO
    if current_close > recent_high and current_ao > 0:
        return "UPTREND"

    # Downtrend: price breaking recent lows with negative AO
    if current_close < recent_low and current_ao < 0:
        return "DOWNTREND"

    # Distribution: near highs but AO weakening
    near_high = current_close > np.percentile(high[-10:], 75)
    if near_high and current_ao > 0 and current_ao < ao_10_ago:
        return "DISTRIBUTION"

    # Accumulation: near lows but AO strengthening
    near_low = current_close < np.percentile(low[-10:], 25)
    if near_low and current_ao < 0 and current_ao > ao_10_ago:
        return "ACCUMULATION"

    # Default based on AO direction
    if current_ao > 0:
        return "UPTREND" if current_close > np.mean(close[-20:]) else "ACCUMULATION"
    elif current_ao < 0:
        return "DOWNTREND" if current_close < np.mean(close[-20:]) else "DISTRIBUTION"

    return "ACCUMULATION"


# ---------------------------------------------------------------------------
# Trend strength category (ported from fks.pine trendStrengthCategory)
# ---------------------------------------------------------------------------


def _trend_strength_category(ratio: float) -> str:
    """Categorize trend strength from absolute current ratio."""
    if ratio > 2.0:
        return "Very Strong"
    elif ratio > 1.5:
        return "Strong"
    elif ratio > 1.0:
        return "Moderate"
    elif ratio > 0.5:
        return "Weak"
    else:
        return "Very Weak"


# ---------------------------------------------------------------------------
# Momentum state detection
# ---------------------------------------------------------------------------


def _detect_momentum_state(trend_speed: np.ndarray) -> str:
    """Detect whether momentum is accelerating, decelerating, or transitioning."""
    if len(trend_speed) < 4:
        return "NEUTRAL"

    ts = trend_speed[-3:]  # last 3 values

    if ts[-1] > 0:
        if ts[-1] > ts[-2] and ts[-2] > ts[-3]:
            return "ACCELERATING"
        elif ts[-1] < ts[-2] and ts[-2] < ts[-3]:
            return "DECELERATING"
        else:
            return "BULLISH"
    elif ts[-1] < 0:
        if ts[-1] < ts[-2] and ts[-2] < ts[-3]:
            return "ACCELERATING"  # accelerating bearish
        elif ts[-1] > ts[-2] and ts[-2] > ts[-3]:
            return "DECELERATING"  # bear losing steam
        else:
            return "BEARISH"

    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_wave_analysis(
    df: pd.DataFrame,
    asset_name: str | None = None,
    ticker: str | None = None,
    lookback_waves: int | None = None,
) -> dict[str, Any]:
    """Full wave analysis on OHLCV DataFrame — main entry point.

    This is the Python port of fks.pine's complete wave tracking system:
      1. Compute dynamic accelerated EMA
      2. Detect bull/bear waves at EMA crossovers
      3. Calculate wave ratios, dominance, current ratio
      4. Compute trend speed with HMA smoothing
      5. Detect market phase and momentum state

    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        asset_name: Optional asset name for tuned parameters (e.g. "Gold", "S&P",
                    "BTC/USD"). Any name from ``models.MICRO_CONTRACT_SPECS`` or
                    ``models.KRAKEN_CONTRACT_SPECS`` is accepted.
        ticker: Optional ticker string (e.g. "MGC=F", "KRAKEN:XBTUSD"). Used to
                look up params when ``asset_name`` is not provided or not found.
        lookback_waves: Max number of recent waves to consider (default from asset params)

    Returns:
        Dict with all wave analysis metrics. Returns safe defaults on insufficient data.
    """
    default_result: dict[str, Any] = {
        "wave_ratio": 1.0,
        "current_ratio": 0.0,
        "dominance": 0.0,
        "trend_speed": 0.0,
        "bias": "NEUTRAL",
        "trend_direction": "NEUTRAL ↔️",
        "trend_strength": "Very Weak",
        "bull_avg": 0.0,
        "bear_avg": 0.0,
        "bull_max": 0.0,
        "bear_max": 0.0,
        "bull_waves_count": 0,
        "bear_waves_count": 0,
        "market_phase": "ACCUMULATION",
        "momentum_state": "NEUTRAL",
        "speed_normalized": 0.0,
    }

    if df is None or df.empty or len(df) < 30:
        return default_result

    # Get asset-specific parameters — resolve_asset_params handles ticker
    # aliases (e.g. "MGC=F" → "Gold") and heuristic fallbacks so all 25+
    # tracked assets get appropriate tuning without hardcoding tickers here.
    params = resolve_asset_params(asset_name=asset_name, ticker=ticker)
    max_length = int(params["max_length"])
    accel_mult = float(params["accel_multiplier"])
    lookback_period = lookback_waves or int(params["lookback_period"])

    try:
        close = np.asarray(df["Close"].astype(float))
        high = np.asarray(df["High"].astype(float))
        low = np.asarray(df["Low"].astype(float))
    except (KeyError, ValueError) as exc:
        logger.warning("Wave analysis failed — missing OHLC columns: %s", exc)
        return default_result

    n = len(close)
    if n < 30:
        return default_result

    # Step 1: Dynamic accelerated EMA
    dyn_ema = _compute_dynamic_ema(close, max_length, accel_mult)

    # Step 2: Detect waves
    bull_changes, bear_changes, _bull_durations, _bear_durations = _detect_waves(close, dyn_ema)

    # Limit to recent waves (lookback_period controls how many waves to consider)
    bull_recent = bull_changes[-lookback_period:]
    bear_recent = bear_changes[-lookback_period:]

    # Step 3: Wave statistics
    bull_avg = float(np.mean(bull_recent)) if bull_recent else 0.0001
    bear_avg = float(np.mean(bear_recent)) if bear_recent else -0.0001
    bull_max = float(np.max(bull_recent)) if bull_recent else 0.0001
    bear_max = float(np.min(bear_recent)) if bear_recent else -0.0001

    # Ensure non-zero denominators
    if bull_avg == 0:
        bull_avg = 0.0001
    if bear_avg == 0:
        bear_avg = -0.0001

    # Wave size ratio (avg): how much larger bull waves are vs bear waves
    wave_ratio = _safe_divide(bull_avg, abs(bear_avg), 1.0)

    # Wave size ratio (max)
    wave_ratio_max = _safe_divide(bull_max, abs(bear_max), 1.0)

    # Step 4: Trend speed
    trend_speed_arr = _compute_trend_speed(close, dyn_ema)
    current_speed = float(trend_speed_arr[-1])

    # Normalize speed for collection period (last 100 bars)
    collection = min(100, n)
    speed_window = trend_speed_arr[-collection:]
    min_speed = float(np.min(speed_window))
    max_speed = float(np.max(speed_window))
    speed_range = max_speed - min_speed
    speed_normalized = (current_speed - min_speed) / speed_range if speed_range > 0 else 0.5

    # Step 5: Current ratio — how extended is the current wave vs historical avg
    if current_speed > 0:
        current_ratio = _safe_divide(current_speed, bull_avg, 1.0)
    else:
        current_ratio = _safe_divide(current_speed, abs(bear_avg), -1.0)

    # Step 6: Dominance — net bull vs bear strength
    dominance = _safe_divide(
        bull_avg - abs(bear_avg),
        bull_avg + abs(bear_avg),
        0.0,
    )

    # Step 7: Trend direction & strength
    if current_speed > 0:
        trend_direction = "BULLISH ↗️"
    elif current_speed < 0:
        trend_direction = "BEARISH ↘️"
    else:
        trend_direction = "NEUTRAL ↔️"

    trend_strength = _trend_strength_category(abs(current_ratio))

    # Step 8: Overall bias from dominance
    if dominance > 0.05:
        bias = "BULLISH"
    elif dominance < -0.05:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Step 9: Market phase (needs Awesome Oscillator)
    # AO = SMA(hl2, 5) - SMA(hl2, 34)
    hl2 = (high + low) / 2.0
    hl2_series = pd.Series(hl2)
    ao_fast = hl2_series.rolling(5).mean()
    ao_slow = hl2_series.rolling(34).mean()
    ao = np.asarray(pd.Series(ao_fast - ao_slow).fillna(0).values)
    market_phase = _detect_market_phase(close, high, low, ao)

    # Step 10: Momentum state
    momentum_state = _detect_momentum_state(trend_speed_arr)

    # Build dominance text (same format as Pine dashboard)
    if dominance > 0:
        dominance_text = f"Bullish +{wave_ratio:.2f}x"
    elif dominance < 0:
        inv_ratio = _safe_divide(1.0, wave_ratio, 0.0)
        dominance_text = f"Bearish -{inv_ratio:.2f}x"
    else:
        dominance_text = "Neutral"

    return {
        # Core metrics
        "wave_ratio": round(float(wave_ratio), 2),
        "wave_ratio_max": round(float(wave_ratio_max), 2),
        "current_ratio": round(float(current_ratio), 2),
        "dominance": round(float(dominance), 3),
        "dominance_text": dominance_text,
        "trend_speed": round(float(current_speed), 4),
        "bias": bias,
        # Display helpers
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "market_phase": market_phase,
        "momentum_state": momentum_state,
        "speed_normalized": round(float(speed_normalized), 3),
        # Raw wave stats
        "bull_avg": round(float(bull_avg), 4),
        "bear_avg": round(float(bear_avg), 4),
        "bull_max": round(float(bull_max), 4),
        "bear_max": round(float(bear_max), 4),
        "bull_waves_count": len(bull_recent),
        "bear_waves_count": len(bear_recent),
        # Formatted text for display/Grok
        "wave_ratio_text": f"{wave_ratio:.2f}x",
        "current_ratio_text": f"{abs(current_ratio):.2f}x",
    }


def wave_summary_text(result: dict[str, Any]) -> str:
    """One-line summary suitable for Grok prompts or dashboard captions."""
    return (
        f"Wave {result['bias']} — "
        f"ratio={result['wave_ratio_text']}, "
        f"current={result['current_ratio_text']}, "
        f"dominance={result['dominance_text']}, "
        f"phase={result['market_phase']}, "
        f"momentum={result['momentum_state']}"
    )
