"""
Daily Bias Analyzer
===================
Computes directional conviction ("what direction for today?") per asset
based on prior-day, weekly, and monthly price action analysis.

This is the foundation of the Daily Strategy Layer — it answers the question
"should I be looking for longs or shorts on Gold today?" before you ever
open a chart.

Inputs:
  - Prior day's OHLCV
  - Prior week's OHLCV
  - Monthly trend (20-day EMA slope on daily bars)
  - ATR regime (expanding or contracting)
  - Volume confirmation (above/below 20-day average)
  - Overnight gap context (Globex open vs prior close)

Output:
  DailyBias dataclass per asset — direction, confidence, reasoning,
  key levels (support/resistance from prior day + weekly)

Usage:
    from lib.trading.strategies.daily.bias_analyzer import compute_daily_bias, DailyBias

    bias = compute_daily_bias(daily_bars_df, weekly_bars_df)
    print(bias.direction)    # "LONG" | "SHORT" | "NEUTRAL"
    print(bias.confidence)   # 0.0 – 1.0
    print(bias.reasoning)    # Human-readable explanation
    print(bias.key_levels)   # {"prior_day_high": ..., "prior_day_low": ..., ...}

Pure computation — no side effects, no Redis, no network calls, fully testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from lib.core.utils import safe_float as _safe_float

logger = logging.getLogger("strategies.daily.bias_analyzer")


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------
class BiasDirection(StrEnum):
    """Directional bias for the trading day."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class CandlePattern(StrEnum):
    """Prior day candle classification."""

    INSIDE_DAY = "inside_day"
    OUTSIDE_DAY = "outside_day"
    DOJI = "doji"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    STRONG_CLOSE_UP = "strong_close_up"
    STRONG_CLOSE_DOWN = "strong_close_down"
    NEUTRAL_CANDLE = "neutral"


# Ordinal encoding for CNN feature (Phase 4A, Feature #21)
CANDLE_PATTERN_ORDINAL: dict[CandlePattern, int] = {
    CandlePattern.INSIDE_DAY: 0,
    CandlePattern.DOJI: 1,
    CandlePattern.BULLISH_ENGULFING: 2,
    CandlePattern.BEARISH_ENGULFING: 3,
    CandlePattern.HAMMER: 4,
    CandlePattern.SHOOTING_STAR: 5,
    CandlePattern.STRONG_CLOSE_UP: 6,
    CandlePattern.STRONG_CLOSE_DOWN: 7,
    CandlePattern.OUTSIDE_DAY: 8,
    CandlePattern.NEUTRAL_CANDLE: 9,
}

# Number of distinct patterns for normalization
NUM_CANDLE_PATTERNS = len(CANDLE_PATTERN_ORDINAL)


@dataclass
class KeyLevels:
    """Key support/resistance levels derived from recent price action."""

    prior_day_high: float = 0.0
    prior_day_low: float = 0.0
    prior_day_mid: float = 0.0
    prior_day_close: float = 0.0
    weekly_high: float = 0.0
    weekly_low: float = 0.0
    weekly_mid: float = 0.0
    monthly_ema20: float = 0.0
    overnight_high: float = 0.0
    overnight_low: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "prior_day_high": self.prior_day_high,
            "prior_day_low": self.prior_day_low,
            "prior_day_mid": self.prior_day_mid,
            "prior_day_close": self.prior_day_close,
            "weekly_high": self.weekly_high,
            "weekly_low": self.weekly_low,
            "weekly_mid": self.weekly_mid,
            "monthly_ema20": self.monthly_ema20,
            "overnight_high": self.overnight_high,
            "overnight_low": self.overnight_low,
        }


@dataclass
class DailyBias:
    """Complete daily bias analysis result for one asset."""

    asset_name: str = ""
    direction: BiasDirection = BiasDirection.NEUTRAL
    confidence: float = 0.0  # 0.0 – 1.0
    reasoning: str = ""
    key_levels: KeyLevels = field(default_factory=KeyLevels)

    # Component scores (for debugging / CNN feature extraction)
    candle_pattern: CandlePattern = CandlePattern.NEUTRAL_CANDLE
    weekly_range_position: float = 0.5  # 0.0 = at low, 1.0 = at high
    monthly_trend_score: float = 0.0  # -1.0 to +1.0
    volume_confirmation: bool = False
    overnight_gap_direction: float = 0.0  # -1.0 to +1.0
    overnight_gap_atr_ratio: float = 0.0  # gap size relative to ATR
    atr_expanding: bool = False

    # Raw component weights (for transparency in dashboard)
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for Redis/SSE/dashboard consumption."""
        return {
            "asset_name": self.asset_name,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "key_levels": self.key_levels.to_dict(),
            "candle_pattern": self.candle_pattern.value,
            "weekly_range_position": round(self.weekly_range_position, 4),
            "monthly_trend_score": round(self.monthly_trend_score, 4),
            "volume_confirmation": self.volume_confirmation,
            "overnight_gap_direction": round(self.overnight_gap_direction, 4),
            "overnight_gap_atr_ratio": round(self.overnight_gap_atr_ratio, 4),
            "atr_expanding": self.atr_expanding,
            "component_scores": {k: round(v, 4) for k, v in self.component_scores.items()},
        }

    # ── CNN feature helpers (for v7 feature contract) ───────────────────
    @property
    def direction_feature(self) -> float:
        """Normalized direction for CNN: -1 (short) → 0 (neutral) → +1 (long), mapped to [0, 1]."""
        if self.direction == BiasDirection.LONG:
            return (1.0 + 1.0) / 2.0  # = 1.0
        if self.direction == BiasDirection.SHORT:
            return (-1.0 + 1.0) / 2.0  # = 0.0
        return 0.5  # NEUTRAL

    @property
    def confidence_feature(self) -> float:
        """Confidence as [0, 1] scalar — already normalized."""
        return self.confidence

    @property
    def candle_pattern_feature(self) -> float:
        """Normalized candle pattern ordinal for CNN: ordinal / (num_patterns - 1) → [0, 1]."""
        ordinal = CANDLE_PATTERN_ORDINAL.get(self.candle_pattern, 9)
        return ordinal / max(NUM_CANDLE_PATTERNS - 1, 1)

    @property
    def weekly_range_feature(self) -> float:
        """Weekly range position — already [0, 1]."""
        return self.weekly_range_position

    @property
    def monthly_trend_feature(self) -> float:
        """Monthly trend score mapped from [-1, +1] to [0, 1]."""
        return (self.monthly_trend_score + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Internal analysis functions
# ---------------------------------------------------------------------------


def _classify_candle(
    o: float,
    h: float,
    l: float,  # noqa: E741
    c: float,
    prev_o: float,
    prev_h: float,
    prev_l: float,
    prev_c: float,
) -> CandlePattern:
    """Classify a daily candle relative to the previous day.

    Args:
        o, h, l, c: Current day OHLC
        prev_o, prev_h, prev_l, prev_c: Prior day OHLC

    Returns:
        CandlePattern classification
    """
    body = abs(c - o)
    candle_range = h - l
    if candle_range <= 0:
        return CandlePattern.DOJI

    body_ratio = body / candle_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    _ = prev_h - prev_l  # noqa: B018

    # Inside day: today's range entirely within yesterday's range
    if h <= prev_h and l >= prev_l:
        return CandlePattern.INSIDE_DAY

    # Outside day: today's range engulfs yesterday's range
    if h > prev_h and l < prev_l:
        return CandlePattern.OUTSIDE_DAY

    # Doji: body < 10% of range
    if body_ratio < 0.10:
        return CandlePattern.DOJI

    # Bullish engulfing: current closes above prior open, prior was bearish
    if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
        return CandlePattern.BULLISH_ENGULFING

    # Bearish engulfing: current closes below prior open, prior was bullish
    if c < o and prev_c > prev_o and c < prev_o and o > prev_c:
        return CandlePattern.BEARISH_ENGULFING

    # Hammer: small body, long lower wick (> 2x body), small upper wick
    if body_ratio < 0.35 and lower_wick > body * 2 and upper_wick < body:
        return CandlePattern.HAMMER

    # Shooting star: small body, long upper wick (> 2x body), small lower wick
    if body_ratio < 0.35 and upper_wick > body * 2 and lower_wick < body:
        return CandlePattern.SHOOTING_STAR

    # Strong close up: closed in upper 25% of range, bullish body
    if c > o and (c - l) / candle_range >= 0.75:
        return CandlePattern.STRONG_CLOSE_UP

    # Strong close down: closed in lower 25% of range, bearish body
    if c < o and (h - c) / candle_range >= 0.75:
        return CandlePattern.STRONG_CLOSE_DOWN

    return CandlePattern.NEUTRAL_CANDLE


def _candle_pattern_bias(pattern: CandlePattern) -> float:
    """Return a directional score [-1, +1] for a candle pattern."""
    return {
        CandlePattern.INSIDE_DAY: 0.0,  # Neutral — waiting for breakout
        CandlePattern.OUTSIDE_DAY: 0.0,  # Mixed — needs close context
        CandlePattern.DOJI: 0.0,  # Indecision
        CandlePattern.BULLISH_ENGULFING: 0.8,
        CandlePattern.BEARISH_ENGULFING: -0.8,
        CandlePattern.HAMMER: 0.6,
        CandlePattern.SHOOTING_STAR: -0.6,
        CandlePattern.STRONG_CLOSE_UP: 0.7,
        CandlePattern.STRONG_CLOSE_DOWN: -0.7,
        CandlePattern.NEUTRAL_CANDLE: 0.0,
    }.get(pattern, 0.0)


def _compute_weekly_range_position(close: float, weekly_high: float, weekly_low: float) -> float:
    """Where price closed relative to the prior week's high/low.

    Returns:
        0.0 = at week low, 0.5 = mid-range, 1.0 = at week high
    """
    weekly_range = weekly_high - weekly_low
    if weekly_range <= 0:
        return 0.5
    return max(0.0, min(1.0, (close - weekly_low) / weekly_range))


def _compute_monthly_trend(daily_closes: pd.Series, ema_period: int = 20) -> float:
    """Compute normalized slope of the 20-day EMA on daily bars.

    Returns:
        Score in [-1, +1]: positive = uptrend, negative = downtrend
    """
    if daily_closes is None or len(daily_closes) < ema_period + 5:
        return 0.0

    ema = daily_closes.ewm(span=ema_period, adjust=False).mean()
    if len(ema) < 5:
        return 0.0

    # Slope over last 5 days, normalized by ATR
    recent_ema = ema.iloc[-5:]
    slope = float(recent_ema.iloc[-1] - recent_ema.iloc[0])

    # Normalize by average daily range for scale-independence
    daily_ranges = daily_closes.diff().abs().iloc[-ema_period:]
    avg_range = float(daily_ranges.mean()) if len(daily_ranges) > 0 else 1.0
    if avg_range <= 0:
        avg_range = 1.0

    normalized = slope / (avg_range * 5)  # 5-day slope / (avg_range * 5)
    return max(-1.0, min(1.0, normalized))


def _compute_volume_confirmation(volumes: pd.Series, period: int = 20) -> bool:
    """Check if yesterday's volume was above the 20-day average."""
    if volumes is None or len(volumes) < period + 1:
        return False
    avg_vol = float(volumes.iloc[-(period + 1) : -1].mean())
    if avg_vol <= 0:
        return False
    return float(volumes.iloc[-1]) > avg_vol


def _compute_overnight_gap(prior_close: float, current_open: float, atr: float) -> tuple[float, float]:
    """Compute overnight gap direction and size relative to ATR.

    Returns:
        (gap_direction, gap_atr_ratio)
        gap_direction: -1.0 to +1.0 (clamped), positive = gap up
        gap_atr_ratio: absolute gap size as fraction of ATR
    """
    if prior_close <= 0 or current_open <= 0 or atr <= 0:
        return 0.0, 0.0

    gap = current_open - prior_close
    gap_ratio = gap / atr
    direction = max(-1.0, min(1.0, gap_ratio))
    return direction, abs(gap_ratio)


def _compute_atr_trend(daily_df: pd.DataFrame, atr_period: int = 14, lookback: int = 10) -> bool:
    """Check if ATR is expanding (True) or contracting (False) over recent bars."""
    if daily_df is None or len(daily_df) < atr_period + lookback:
        return False

    highs = daily_df["High"].values
    lows = daily_df["Low"].values
    closes = daily_df["Close"].values

    # Compute true range
    tr = np.maximum(
        highs[1:] - lows[1:],  # type: ignore[operator]
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),  # type: ignore[operator]
            np.abs(lows[1:] - closes[:-1]),  # type: ignore[operator]
        ),
    )

    if len(tr) < atr_period + lookback:
        return False

    # Simple moving average ATR
    atr_series = pd.Series(tr).rolling(atr_period).mean().dropna()  # type: ignore[union-attr]
    if len(atr_series) < lookback:
        return False

    recent = atr_series.iloc[-lookback:]
    # ATR expanding if the recent slope is positive
    return float(recent.iloc[-1]) > float(recent.iloc[0])


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def compute_daily_bias(
    daily_bars: pd.DataFrame,
    weekly_bars: pd.DataFrame | None = None,
    asset_name: str = "",
    current_open: float | None = None,
) -> DailyBias:
    """Compute directional bias for the upcoming trading day.

    Args:
        daily_bars: DataFrame with OHLCV columns, at least 25 rows of daily data,
                    most recent bar is yesterday (completed day).
        weekly_bars: Optional DataFrame with OHLCV weekly data.
                     If None, weekly analysis is skipped.
        asset_name: Human-readable asset name for labeling.
        current_open: Today's opening price (Globex open / overnight).
                      If None, gap analysis is skipped.

    Returns:
        DailyBias with direction, confidence, reasoning, and key levels.

    Column expectations:
        daily_bars must have: Open, High, Low, Close, Volume (case-sensitive)
        weekly_bars must have: High, Low, Close (case-sensitive)
    """
    result = DailyBias(asset_name=asset_name)

    # Validate input
    if daily_bars is None or len(daily_bars) < 3:
        result.reasoning = "Insufficient daily data (need >= 3 bars)"
        return result

    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(daily_bars.columns):
        result.reasoning = f"Missing required columns: {required_cols - set(daily_bars.columns)}"
        return result

    # ── Extract prior day and day-before-that data ──────────────────────
    prev = daily_bars.iloc[-1]  # Yesterday (most recent completed bar)
    prev2 = daily_bars.iloc[-2]  # Day before yesterday

    prev_o = _safe_float(prev["Open"])
    prev_h = _safe_float(prev["High"])
    prev_l = _safe_float(prev["Low"])
    prev_c = _safe_float(prev["Close"])

    prev2_o = _safe_float(prev2["Open"])
    prev2_h = _safe_float(prev2["High"])
    prev2_l = _safe_float(prev2["Low"])
    prev2_c = _safe_float(prev2["Close"])

    if prev_h <= 0 or prev_l <= 0:
        result.reasoning = "Invalid price data in prior day bar"
        return result

    # ── Key levels ──────────────────────────────────────────────────────
    levels = KeyLevels(
        prior_day_high=prev_h,
        prior_day_low=prev_l,
        prior_day_mid=round((prev_h + prev_l) / 2, 8),
        prior_day_close=prev_c,
    )

    # Weekly levels
    if weekly_bars is not None and len(weekly_bars) >= 2:
        prev_week = weekly_bars.iloc[-1]
        levels.weekly_high = _safe_float(prev_week.get("High", 0))
        levels.weekly_low = _safe_float(prev_week.get("Low", 0))
        levels.weekly_mid = round((levels.weekly_high + levels.weekly_low) / 2, 8)

    # Monthly EMA
    if len(daily_bars) >= 25:
        ema20 = daily_bars["Close"].ewm(span=20, adjust=False).mean()
        levels.monthly_ema20 = _safe_float(ema20.iloc[-1])

    result.key_levels = levels

    # ══════════════════════════════════════════════════════════════════════
    # Component 1: Candle Pattern (weight: 25%)
    # ══════════════════════════════════════════════════════════════════════
    pattern = _classify_candle(prev_o, prev_h, prev_l, prev_c, prev2_o, prev2_h, prev2_l, prev2_c)
    result.candle_pattern = pattern
    candle_bias = _candle_pattern_bias(pattern)
    result.component_scores["candle_pattern"] = candle_bias

    # ══════════════════════════════════════════════════════════════════════
    # Component 2: Weekly Range Position (weight: 20%)
    # ══════════════════════════════════════════════════════════════════════
    weekly_pos = 0.5
    if weekly_bars is not None and len(weekly_bars) >= 2:
        weekly_pos = _compute_weekly_range_position(prev_c, levels.weekly_high, levels.weekly_low)
    result.weekly_range_position = weekly_pos
    # Convert to [-1, +1]: 0.0 → -1, 0.5 → 0, 1.0 → +1
    weekly_bias = (weekly_pos - 0.5) * 2.0
    result.component_scores["weekly_position"] = weekly_bias

    # ══════════════════════════════════════════════════════════════════════
    # Component 3: Monthly Trend Score (weight: 25%)
    # ══════════════════════════════════════════════════════════════════════
    monthly_trend = _compute_monthly_trend(daily_bars["Close"])  # type: ignore[arg-type]
    result.monthly_trend_score = monthly_trend
    result.component_scores["monthly_trend"] = monthly_trend

    # ══════════════════════════════════════════════════════════════════════
    # Component 4: Volume Confirmation (weight: 10%)
    # ══════════════════════════════════════════════════════════════════════
    has_volume = "Volume" in daily_bars.columns
    vol_confirm = _compute_volume_confirmation(daily_bars["Volume"]) if has_volume else False  # type: ignore[arg-type]
    result.volume_confirmation = vol_confirm
    # Volume confirms the candle direction — if no volume, neutral
    vol_bias = 0.0
    if vol_confirm:
        vol_bias = 0.5 if candle_bias > 0 else (-0.5 if candle_bias < 0 else 0.0)
    result.component_scores["volume"] = vol_bias

    # ══════════════════════════════════════════════════════════════════════
    # Component 5: Overnight Gap (weight: 10%)
    # ══════════════════════════════════════════════════════════════════════
    gap_dir = 0.0
    gap_ratio = 0.0
    if current_open is not None and current_open > 0:
        # Compute ATR for gap normalization
        if len(daily_bars) >= 15:
            tr_vals = []
            for i in range(1, min(15, len(daily_bars))):
                row = daily_bars.iloc[-i]
                prev_row = daily_bars.iloc[-(i + 1)]
                h_val = _safe_float(row["High"])
                l_val = _safe_float(row["Low"])
                pc = _safe_float(prev_row["Close"])
                tr = max(h_val - l_val, abs(h_val - pc), abs(l_val - pc))
                tr_vals.append(tr)
            atr = sum(tr_vals) / len(tr_vals) if tr_vals else 1.0
        else:
            atr = abs(prev_h - prev_l) if prev_h > prev_l else 1.0

        gap_dir, gap_ratio = _compute_overnight_gap(prev_c, current_open, atr)
        levels.overnight_high = max(current_open, prev_c)
        levels.overnight_low = min(current_open, prev_c)

    result.overnight_gap_direction = gap_dir
    result.overnight_gap_atr_ratio = gap_ratio
    result.component_scores["overnight_gap"] = gap_dir

    # ══════════════════════════════════════════════════════════════════════
    # Component 6: ATR Trend — expanding = momentum, contracting = fading
    # (weight: 10%)
    # ══════════════════════════════════════════════════════════════════════
    atr_expanding = _compute_atr_trend(daily_bars)
    result.atr_expanding = atr_expanding
    # ATR expanding amplifies conviction in trend direction, doesn't set direction itself
    atr_multiplier = 1.15 if atr_expanding else 0.85
    result.component_scores["atr_trend"] = 1.0 if atr_expanding else -0.3

    # ══════════════════════════════════════════════════════════════════════
    # Composite scoring — weighted combination
    # ══════════════════════════════════════════════════════════════════════
    # Weights sum to 1.0
    W_CANDLE = 0.25
    W_WEEKLY = 0.20
    W_MONTHLY = 0.25
    W_VOLUME = 0.10
    W_GAP = 0.10
    W_ATR = 0.10

    raw_score = (
        candle_bias * W_CANDLE
        + weekly_bias * W_WEEKLY
        + monthly_trend * W_MONTHLY
        + vol_bias * W_VOLUME
        + gap_dir * W_GAP
        + result.component_scores["atr_trend"] * W_ATR
    )

    # Apply ATR expansion multiplier to amplify/dampen
    raw_score *= atr_multiplier

    # Clamp to [-1, +1]
    raw_score = max(-1.0, min(1.0, raw_score))

    # ══════════════════════════════════════════════════════════════════════
    # Determine direction and confidence
    # ══════════════════════════════════════════════════════════════════════
    DIRECTION_THRESHOLD = 0.15  # Must exceed this for directional call

    if raw_score > DIRECTION_THRESHOLD:
        result.direction = BiasDirection.LONG
    elif raw_score < -DIRECTION_THRESHOLD:
        result.direction = BiasDirection.SHORT
    else:
        result.direction = BiasDirection.NEUTRAL

    # Confidence is the absolute magnitude, scaled to [0, 1]
    result.confidence = min(1.0, abs(raw_score))

    # ══════════════════════════════════════════════════════════════════════
    # Build human-readable reasoning
    # ══════════════════════════════════════════════════════════════════════
    reasons: list[str] = []

    # Candle pattern
    if pattern != CandlePattern.NEUTRAL_CANDLE:
        emoji = "🟢" if candle_bias > 0 else ("🔴" if candle_bias < 0 else "⚪")
        reasons.append(f"{emoji} Prior day: {pattern.value.replace('_', ' ')} ({candle_bias:+.1f})")

    # Weekly position
    if abs(weekly_bias) > 0.3:
        if weekly_bias > 0:
            reasons.append(f"🟢 Near weekly high ({weekly_pos:.0%} of range)")
        else:
            reasons.append(f"🔴 Near weekly low ({weekly_pos:.0%} of range)")

    # Monthly trend
    if abs(monthly_trend) > 0.2:
        emoji = "🟢" if monthly_trend > 0 else "🔴"
        trend_word = "up" if monthly_trend > 0 else "down"
        reasons.append(f"{emoji} Monthly trend: {trend_word} ({monthly_trend:+.2f})")

    # Volume
    if vol_confirm:
        emoji = "🟢" if candle_bias > 0 else "🔴" if candle_bias < 0 else "⚪"
        reasons.append(f"{emoji} Volume above 20-day average confirms prior day")

    # Gap
    if gap_ratio > 0.3:
        gap_word = "up" if gap_dir > 0 else "down"
        reasons.append(f"{'🟢' if gap_dir > 0 else '🔴'} Overnight gap {gap_word} ({gap_ratio:.1f}× ATR)")

    # ATR
    if atr_expanding:
        reasons.append("⚡ ATR expanding — momentum increasing")
    elif not atr_expanding and len(daily_bars) >= 24:
        reasons.append("🧘 ATR contracting — momentum fading")

    if not reasons:
        reasons.append("⚪ No strong signals — stay neutral")

    result.reasoning = " | ".join(reasons)

    logger.debug(
        "Daily bias for %s: %s (%.1f%% confidence) — %s",
        asset_name,
        result.direction.value,
        result.confidence * 100,
        result.reasoning,
    )

    return result


# ---------------------------------------------------------------------------
# Batch analysis — run for all assets at once
# ---------------------------------------------------------------------------
def compute_all_daily_biases(
    daily_data: dict[str, pd.DataFrame],
    weekly_data: dict[str, pd.DataFrame] | None = None,
    current_opens: dict[str, float] | None = None,
) -> dict[str, DailyBias]:
    """Compute daily bias for multiple assets.

    Args:
        daily_data: {asset_name: daily_bars_df}
        weekly_data: {asset_name: weekly_bars_df} or None
        current_opens: {asset_name: today's open price} or None

    Returns:
        {asset_name: DailyBias}
    """
    results: dict[str, DailyBias] = {}

    for name, daily_df in daily_data.items():
        weekly_df = weekly_data.get(name) if weekly_data else None
        open_price = current_opens.get(name) if current_opens else None

        try:
            bias = compute_daily_bias(
                daily_bars=daily_df,
                weekly_bars=weekly_df,
                asset_name=name,
                current_open=open_price,
            )
            results[name] = bias
        except Exception as exc:
            logger.warning("Daily bias failed for %s: %s", name, exc)
            results[name] = DailyBias(
                asset_name=name,
                reasoning=f"Analysis error: {exc}",
            )

    return results


# ---------------------------------------------------------------------------
# Quick scoring for asset selection (Phase 3A integration)
# ---------------------------------------------------------------------------
def rank_assets_by_conviction(biases: dict[str, DailyBias]) -> list[tuple[str, float, str]]:
    """Rank assets by directional conviction strength.

    Returns:
        List of (asset_name, confidence, direction) sorted by confidence descending.
        Only directional (non-NEUTRAL) assets are included.
    """
    ranked = []
    for name, bias in biases.items():
        if bias.direction != BiasDirection.NEUTRAL:
            ranked.append((name, bias.confidence, bias.direction.value))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
