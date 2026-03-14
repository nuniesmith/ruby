"""
Swing Detector — Phase 2C
=========================
Intraday swing entry/exit detection engine that operates alongside the
always-running RB scalping system with separate position tracking and
risk budget.

Three entry styles:
  1. **Pullback Entry** — price pulls back to a key level (prior day H/L,
     VWAP, EMA-21) in the direction of the daily bias, then confirms with
     a reversal bar.
  2. **Breakout Entry** — price breaks the prior day high (long bias) or
     low (short bias) with volume confirmation.
  3. **Gap Continuation** — overnight gap aligns with daily bias and doesn't
     fill in first 30 minutes; enter on first pullback.

Exit logic:
  - TP1 at 2× ATR (scale 50%), TP2 at 3.5× ATR (remaining), or trail
    with EMA-21 on 15-min bars.
  - SL at 1.5× ATR from entry — wider than scalp trades.
  - Time stop: close by 15:30 ET if neither TP nor SL hit (no overnight holds).

Risk:
  - Separate risk allocation: 0.5% of account per swing trade
    (vs 0.75% for scalps).

Usage:
    from lib.trading.strategies.daily.swing_detector import (
        detect_swing_entries,
        evaluate_swing_exits,
        SwingSignal,
        SwingExitSignal,
        SwingEntryStyle,
    )

    signals = detect_swing_entries(
        bars_15m=df_15m,
        bias=daily_bias,
        current_price=2735.0,
        atr=12.5,
    )
    for sig in signals:
        print(sig.entry_style, sig.entry_price, sig.stop_loss, sig.tp1)

    exit_signals = evaluate_swing_exits(
        bars_15m=df_15m,
        entry_price=2720.0,
        direction="LONG",
        stop_loss=2698.0,
        tp1=2745.0,
        tp2=2763.0,
        current_price=2748.0,
    )

Pure computation — no side effects, no Redis, no network calls, fully testable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from lib.core.utils import safe_float as _safe_float
from lib.trading.strategies.daily.bias_analyzer import (
    BiasDirection,
    DailyBias,
)

logger = logging.getLogger("strategies.daily.swing_detector")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants — Swing Trade Parameters
# ---------------------------------------------------------------------------

# ATR multipliers for stop and target levels
SWING_SL_ATR_MULT = 1.5  # Stop loss at 1.5× ATR from entry
SWING_TP1_ATR_MULT = 2.0  # TP1 at 2× ATR — scale 50%
SWING_TP2_ATR_MULT = 3.5  # TP2 at 3.5× ATR — scale remaining
SWING_TRAIL_ATR_MULT = 1.0  # Trailing stop distance (1× ATR from high/low)

# Risk allocation (separate from scalp's 0.75%)
SWING_RISK_PCT = 0.005  # 0.5% of account per swing trade

# Time stop — no overnight holds
TIME_STOP_HOUR = 15
TIME_STOP_MINUTE = 30  # Close by 15:30 ET

# Pullback detection parameters
PULLBACK_TOLERANCE_ATR = 0.3  # Price within 0.3× ATR of key level = "at level"
PULLBACK_MIN_RETRACE_PCT = 0.25  # Must retrace at least 25% of prior impulse
PULLBACK_MAX_RETRACE_PCT = 0.75  # Must not retrace more than 75% (trend broken)
EMA_PULLBACK_PERIOD = 21  # EMA-21 for pullback detection

# Breakout detection parameters
BREAKOUT_VOLUME_MULT = 1.3  # Volume must be ≥ 1.3× average for confirmation
BREAKOUT_CLOSE_PCT = 0.6  # Bar must close in top/bottom 60% of range

# Gap continuation parameters
GAP_MIN_ATR_RATIO = 0.3  # Minimum gap size: 0.3× ATR
GAP_FILL_THRESHOLD = 0.5  # Gap is "filled" if price retraces 50%+ of gap
GAP_SETTLE_BARS = 6  # Wait 6 bars (30 min on 5m) before looking for entry

# Confirmation bar parameters
CONFIRM_BAR_BODY_PCT = 0.5  # Body must be ≥ 50% of total range

# Position sizing
MAX_SWING_CONTRACTS = 3  # Cap at 3 micro contracts per swing
MIN_SWING_CONTRACTS = 1

# TP1 scale-out fraction
TP1_SCALE_FRACTION = 0.5  # Scale 50% at TP1

# Trailing stop: EMA-21 on 15-min bars
TRAIL_EMA_PERIOD = 21


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SwingEntryStyle(StrEnum):
    """Available swing entry styles."""

    PULLBACK = "pullback_entry"
    BREAKOUT = "breakout_entry"
    GAP_CONTINUATION = "gap_continuation"


class SwingExitReason(StrEnum):
    """Reason a swing exit was triggered."""

    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    EMA_TRAIL = "ema_trail"
    INVALIDATED = "invalidated"  # Bias reversal or structure break


class SwingPhase(StrEnum):
    """Current phase of a swing trade lifecycle."""

    WATCHING = "watching"  # Signal detected, waiting for confirmation
    ENTRY_READY = "entry_ready"  # Confirmed, ready to execute
    ACTIVE = "active"  # Position is open
    TP1_HIT = "tp1_hit"  # TP1 scaled out, trailing remainder
    TRAILING = "trailing"  # Trailing stop active
    CLOSED = "closed"  # Position closed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SwingSignal:
    """A detected swing entry signal with full trade plan."""

    asset_name: str = ""
    entry_style: SwingEntryStyle = SwingEntryStyle.PULLBACK
    direction: str = "LONG"  # "LONG" or "SHORT"
    confidence: float = 0.0  # 0.0–1.0 — from bias + confirmation quality
    entry_price: float = 0.0  # Suggested entry price
    entry_zone_low: float = 0.0  # Entry zone lower bound
    entry_zone_high: float = 0.0  # Entry zone upper bound
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    atr: float = 0.0
    risk_reward_tp1: float = 0.0  # R:R to TP1
    risk_reward_tp2: float = 0.0  # R:R to TP2
    risk_dollars: float = 0.0
    position_size: int = 1
    reasoning: str = ""
    key_level_used: str = ""  # Which key level triggered the signal
    key_level_price: float = 0.0  # The actual price of that level
    confirmation_bar_idx: int = -1  # Index of the confirmation bar (-1 = pending)
    detected_at: str = ""  # ISO timestamp of detection
    phase: SwingPhase = SwingPhase.WATCHING

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "entry_style": self.entry_style.value,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "entry_zone_low": self.entry_zone_low,
            "entry_zone_high": self.entry_zone_high,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "atr": round(self.atr, 6),
            "risk_reward_tp1": round(self.risk_reward_tp1, 2),
            "risk_reward_tp2": round(self.risk_reward_tp2, 2),
            "risk_dollars": round(self.risk_dollars, 2),
            "position_size": self.position_size,
            "reasoning": self.reasoning,
            "key_level_used": self.key_level_used,
            "key_level_price": self.key_level_price,
            "confirmation_bar_idx": self.confirmation_bar_idx,
            "detected_at": self.detected_at,
            "phase": self.phase.value,
        }


@dataclass
class SwingExitSignal:
    """A swing exit trigger with reason and details."""

    reason: SwingExitReason = SwingExitReason.STOP_LOSS
    exit_price: float = 0.0
    pnl_estimate: float = 0.0  # Estimated P&L in dollars
    r_multiple: float = 0.0  # P&L as multiple of risk
    scale_fraction: float = 1.0  # Fraction of position to close (0.5 for TP1)
    trailing_stop_price: float = 0.0  # Updated trailing stop (for EMA trail)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "exit_price": self.exit_price,
            "pnl_estimate": round(self.pnl_estimate, 2),
            "r_multiple": round(self.r_multiple, 2),
            "scale_fraction": self.scale_fraction,
            "trailing_stop_price": self.trailing_stop_price,
            "reasoning": self.reasoning,
        }


@dataclass
class SwingState:
    """Tracks the state of an active or pending swing trade.

    This is used by the engine to persist swing trade state across ticks.
    """

    asset_name: str = ""
    signal: SwingSignal | None = None
    phase: SwingPhase = SwingPhase.WATCHING
    entry_price: float = 0.0
    current_stop: float = 0.0  # May move (trailing)
    tp1: float = 0.0
    tp2: float = 0.0
    direction: str = "LONG"
    position_size: int = 1
    remaining_size: int = 1  # After TP1 scale-out
    highest_price: float = 0.0  # Highest price since entry (for trailing)
    lowest_price: float = float("inf")  # Lowest price since entry (for trailing)
    entry_time: str = ""
    last_update: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "signal": self.signal.to_dict() if self.signal else None,
            "phase": self.phase.value,
            "entry_price": self.entry_price,
            "current_stop": self.current_stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "direction": self.direction,
            "position_size": self.position_size,
            "remaining_size": self.remaining_size,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "entry_time": self.entry_time,
            "last_update": self.last_update,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _price_decimals_from_tick(tick_size: float) -> int:
    """Determine number of decimal places from tick size."""
    if tick_size <= 0:
        return 2
    if tick_size >= 1.0:
        return 0
    s = f"{tick_size:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 2


def _get_tick_size(asset_name: str) -> float:
    """Get tick size for an asset. Falls back to 0.01."""
    try:
        from lib.core.asset_registry import get_asset

        asset_obj = get_asset(asset_name)
        if asset_obj and asset_obj.micro:
            return asset_obj.micro.tick_size
    except ImportError:
        pass
    return 0.01


def _get_point_value(asset_name: str) -> float:
    """Get point value for an asset. Falls back to 1.0."""
    try:
        from lib.core.asset_registry import get_asset

        asset_obj = get_asset(asset_name)
        if asset_obj and asset_obj.micro:
            return asset_obj.micro.point_value
    except ImportError:
        pass
    return 1.0


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute EMA on a price series."""
    return pd.Series(series.ewm(span=period, adjust=False).mean())


def _bar_body_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Ratio of candle body to total range. 1.0 = full body, 0.0 = doji."""
    total_range = high - low
    if total_range <= 0:
        return 0.0
    body = abs(close - open_)
    return body / total_range


def _bar_is_bullish(open_: float, close: float) -> bool:
    """True if bar closed above its open."""
    return close > open_


def _bar_close_position(open_: float, high: float, low: float, close: float) -> float:
    """Where the close sits in the bar's range. 1.0 = at high, 0.0 = at low."""
    total_range = high - low
    if total_range <= 0:
        return 0.5
    return (close - low) / total_range


def _is_time_stop_due(now: datetime | None = None) -> bool:
    """Check if we are past the time-stop cutoff (15:30 ET)."""
    if now is None:
        now = datetime.now(tz=_EST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_EST)
    cutoff = time(TIME_STOP_HOUR, TIME_STOP_MINUTE)
    return now.time() >= cutoff


def _compute_position_size(
    entry_price: float,
    stop_price: float,
    account_size: int,
    asset_name: str = "",
    risk_pct: float = SWING_RISK_PCT,
) -> tuple[int, float]:
    """Compute position size and risk in dollars.

    Returns:
        (position_size, risk_dollars)
    """
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return MIN_SWING_CONTRACTS, account_size * risk_pct

    max_risk = account_size * risk_pct
    point_value = _get_point_value(asset_name)

    risk_per_contract = stop_distance * point_value
    if risk_per_contract <= 0:
        return MIN_SWING_CONTRACTS, max_risk

    size = max(MIN_SWING_CONTRACTS, int(max_risk / risk_per_contract))
    size = min(size, MAX_SWING_CONTRACTS)
    risk_dollars = size * risk_per_contract

    return size, risk_dollars


def _compute_risk_reward(
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> float:
    """Compute risk:reward ratio. Returns 0.0 if stop distance is zero."""
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return 0.0
    reward = abs(target_price - entry_price)
    return reward / risk


# ---------------------------------------------------------------------------
# Pullback Entry Detection
# ---------------------------------------------------------------------------


def detect_pullback_entry(
    bars: pd.DataFrame,
    bias: DailyBias,
    current_price: float,
    atr: float,
    asset_name: str = "",
    account_size: int = 50_000,
) -> SwingSignal | None:
    """Detect a pullback entry to a key level in the direction of the daily bias.

    Logic:
    1. Identify key levels from bias analyzer (prior day H/L, EMA-20, weekly mid).
    2. Check if current price has pulled back to within tolerance of a key level.
    3. Look for a confirmation bar (body ≥ 50% of range, closing in bias direction).
    4. Verify the pullback is 25–75% of the prior impulse move.

    Args:
        bars: Intraday DataFrame (5m or 15m) with OHLCV columns. Most recent bar is last.
        bias: Daily bias from bias_analyzer.
        current_price: Latest price.
        atr: Current ATR value.
        asset_name: Human-readable name.
        account_size: For position sizing.

    Returns:
        SwingSignal if pullback entry is detected, None otherwise.
    """
    if bias.direction == BiasDirection.NEUTRAL:
        return None

    if bars is None or len(bars) < 10:
        return None

    if atr <= 0:
        return None

    direction = bias.direction.value  # "LONG" or "SHORT"
    kl = bias.key_levels
    if kl is None:
        return None

    # Build list of key levels to check
    levels: list[tuple[str, float]] = []
    if kl.prior_day_high > 0:
        levels.append(("prior_day_high", kl.prior_day_high))
    if kl.prior_day_low > 0:
        levels.append(("prior_day_low", kl.prior_day_low))
    if kl.prior_day_mid > 0:
        levels.append(("prior_day_mid", kl.prior_day_mid))
    if kl.weekly_mid > 0:
        levels.append(("weekly_mid", kl.weekly_mid))
    if kl.monthly_ema20 > 0:
        levels.append(("monthly_ema20", kl.monthly_ema20))

    # Add intraday EMA-21 if we have enough bars
    if len(bars) >= EMA_PULLBACK_PERIOD + 5:
        ema21 = _compute_ema(bars["Close"], EMA_PULLBACK_PERIOD)  # type: ignore[arg-type]
        ema_val = _safe_float(ema21.iloc[-1])
        if ema_val > 0:
            levels.append(("ema_21", ema_val))

    if not levels:
        return None

    # For LONG pullback: price should be pulling back DOWN toward support
    # For SHORT pullback: price should be pulling back UP toward resistance
    tolerance = atr * PULLBACK_TOLERANCE_ATR

    # Find recent swing high/low for impulse measurement
    recent = bars.tail(20)
    recent_high = _safe_float(recent["High"].max())
    recent_low = _safe_float(recent["Low"].min())
    impulse_range = recent_high - recent_low

    if impulse_range <= 0:
        return None

    # Check each level for pullback proximity
    best_level: tuple[str, float] | None = None
    best_distance = float("inf")

    for level_name, level_price in levels:
        if level_price <= 0:
            continue

        distance = abs(current_price - level_price)

        # Must be within tolerance
        if distance > tolerance:
            continue

        # For LONG: price should be near support (level below or near price)
        # We want levels that act as support (price pulling back down to them)
        if direction == "LONG":
            # Level should be at or below current price (support)
            if level_price > current_price + tolerance * 0.5:
                continue
            # Check retrace depth: price should have come down from recent high
            retrace = (recent_high - current_price) / impulse_range if impulse_range > 0 else 0
            if retrace < PULLBACK_MIN_RETRACE_PCT or retrace > PULLBACK_MAX_RETRACE_PCT:
                continue
        else:  # SHORT
            # Level should be at or above current price (resistance)
            if level_price < current_price - tolerance * 0.5:
                continue
            retrace = (current_price - recent_low) / impulse_range if impulse_range > 0 else 0
            if retrace < PULLBACK_MIN_RETRACE_PCT or retrace > PULLBACK_MAX_RETRACE_PCT:
                continue

        if distance < best_distance:
            best_distance = distance
            best_level = (level_name, level_price)

    if best_level is None:
        return None

    level_name, level_price = best_level

    # Check for confirmation bar (most recent completed bar)
    last_bar = bars.iloc[-1]
    lb_open = _safe_float(last_bar.get("Open", 0))
    lb_high = _safe_float(last_bar.get("High", 0))
    lb_low = _safe_float(last_bar.get("Low", 0))
    lb_close = _safe_float(last_bar.get("Close", 0))

    if lb_high <= 0 or lb_low <= 0:
        return None

    body_ratio = _bar_body_ratio(lb_open, lb_high, lb_low, lb_close)
    close_pos = _bar_close_position(lb_open, lb_high, lb_low, lb_close)

    # Confirmation: body ≥ 50% of range AND closing in bias direction
    confirmed = False
    if body_ratio >= CONFIRM_BAR_BODY_PCT and (
        direction == "LONG"
        and close_pos >= BREAKOUT_CLOSE_PCT
        or direction == "SHORT"
        and close_pos <= (1.0 - BREAKOUT_CLOSE_PCT)
    ):
        confirmed = True

    tick_size = _get_tick_size(asset_name)
    decimals = _price_decimals_from_tick(tick_size)

    # Compute entry, stop, targets
    entry_price = current_price
    if direction == "LONG":
        stop = round(entry_price - atr * SWING_SL_ATR_MULT, decimals)
        tp1 = round(entry_price + atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price + atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(level_price - tolerance * 0.5, decimals)
        entry_zone_high = round(level_price + tolerance * 0.5, decimals)
    else:
        stop = round(entry_price + atr * SWING_SL_ATR_MULT, decimals)
        tp1 = round(entry_price - atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price - atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(level_price - tolerance * 0.5, decimals)
        entry_zone_high = round(level_price + tolerance * 0.5, decimals)

    size, risk_dollars = _compute_position_size(entry_price, stop, account_size, asset_name)

    rr_tp1 = _compute_risk_reward(entry_price, stop, tp1)
    rr_tp2 = _compute_risk_reward(entry_price, stop, tp2)

    # Build confidence from bias confidence + confirmation quality
    conf = bias.confidence * 0.6
    if confirmed:
        conf += 0.25
    if body_ratio > 0.7:
        conf += 0.1
    if bias.volume_confirmation:
        conf += 0.05
    conf = min(1.0, conf)

    reasoning_parts = [
        f"Pullback to {level_name.replace('_', ' ')} ({level_price:.{decimals}f})",
        f"in {direction} bias ({bias.confidence:.0%} conviction)",
    ]
    if confirmed:
        reasoning_parts.append("confirmation bar present")
    else:
        reasoning_parts.append("awaiting confirmation bar")

    signal = SwingSignal(
        asset_name=asset_name,
        entry_style=SwingEntryStyle.PULLBACK,
        direction=direction,
        confidence=conf,
        entry_price=round(entry_price, decimals),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop,
        tp1=tp1,
        tp2=tp2,
        atr=atr,
        risk_reward_tp1=rr_tp1,
        risk_reward_tp2=rr_tp2,
        risk_dollars=risk_dollars,
        position_size=size,
        reasoning=" | ".join(reasoning_parts),
        key_level_used=level_name,
        key_level_price=level_price,
        confirmation_bar_idx=len(bars) - 1 if confirmed else -1,
        detected_at=datetime.now(tz=_EST).isoformat(),
        phase=SwingPhase.ENTRY_READY if confirmed else SwingPhase.WATCHING,
    )

    logger.info(
        "Pullback signal: %s %s at %s (level=%s @ %.{dec}f, conf=%.0f%%, RR=%.1f/%.1f)".replace("{dec}", str(decimals)),
        direction,
        asset_name,
        entry_price,
        level_name,
        level_price,
        conf * 100,
        rr_tp1,
        rr_tp2,
    )

    return signal


# ---------------------------------------------------------------------------
# Breakout Entry Detection
# ---------------------------------------------------------------------------


def detect_breakout_entry(
    bars: pd.DataFrame,
    bias: DailyBias,
    current_price: float,
    atr: float,
    asset_name: str = "",
    account_size: int = 50_000,
) -> SwingSignal | None:
    """Detect a breakout entry through prior day high (long) or low (short).

    Logic:
    1. For LONG bias: check if current price has broken above prior day high.
    2. For SHORT bias: check if current price has broken below prior day low.
    3. Require volume confirmation (current bar volume ≥ 1.3× average).
    4. Require the breakout bar to close in the top/bottom 60% of its range.

    Args:
        bars: Intraday DataFrame with OHLCV columns.
        bias: Daily bias from bias_analyzer.
        current_price: Latest price.
        atr: Current ATR value.
        asset_name: Human-readable name.
        account_size: For position sizing.

    Returns:
        SwingSignal if breakout entry is detected, None otherwise.
    """
    if bias.direction == BiasDirection.NEUTRAL:
        return None

    if bars is None or len(bars) < 10:
        return None

    if atr <= 0:
        return None

    direction = bias.direction.value
    kl = bias.key_levels
    if kl is None:
        return None

    # Determine breakout level
    if direction == "LONG":
        breakout_level = kl.prior_day_high
        level_name = "prior_day_high"
    else:
        breakout_level = kl.prior_day_low
        level_name = "prior_day_low"

    if breakout_level <= 0:
        return None

    # Check if price has broken through
    broken = False
    if (
        direction == "LONG"
        and current_price > breakout_level
        or direction == "SHORT"
        and current_price < breakout_level
    ):
        broken = True

    if not broken:
        return None

    # Check breakout bar quality
    last_bar = bars.iloc[-1]
    lb_open = _safe_float(last_bar.get("Open", 0))
    lb_high = _safe_float(last_bar.get("High", 0))
    lb_low = _safe_float(last_bar.get("Low", 0))
    lb_close = _safe_float(last_bar.get("Close", 0))

    if lb_high <= 0 or lb_low <= 0:
        return None

    close_pos = _bar_close_position(lb_open, lb_high, lb_low, lb_close)
    body_ratio = _bar_body_ratio(lb_open, lb_high, lb_low, lb_close)

    # Breakout bar quality check
    bar_quality_ok = False
    if (
        direction == "LONG"
        and close_pos >= BREAKOUT_CLOSE_PCT
        or direction == "SHORT"
        and close_pos <= (1.0 - BREAKOUT_CLOSE_PCT)
    ):
        bar_quality_ok = True

    if not bar_quality_ok:
        return None

    # Volume confirmation
    vol_confirmed = False
    if "Volume" in bars.columns:
        avg_volume = _safe_float(bars["Volume"].tail(20).mean())
        current_volume = _safe_float(last_bar.get("Volume", 0))
        if avg_volume > 0 and current_volume >= avg_volume * BREAKOUT_VOLUME_MULT:
            vol_confirmed = True
    else:
        # No volume data — allow breakout but with lower confidence
        vol_confirmed = False

    tick_size = _get_tick_size(asset_name)
    decimals = _price_decimals_from_tick(tick_size)

    # Entry at current price (just broke out)
    entry_price = current_price

    if direction == "LONG":
        stop = round(breakout_level - atr * (SWING_SL_ATR_MULT - 0.5), decimals)
        tp1 = round(entry_price + atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price + atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(breakout_level, decimals)
        entry_zone_high = round(breakout_level + atr * 0.5, decimals)
    else:
        stop = round(breakout_level + atr * (SWING_SL_ATR_MULT - 0.5), decimals)
        tp1 = round(entry_price - atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price - atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(breakout_level - atr * 0.5, decimals)
        entry_zone_high = round(breakout_level, decimals)

    size, risk_dollars = _compute_position_size(entry_price, stop, account_size, asset_name)

    rr_tp1 = _compute_risk_reward(entry_price, stop, tp1)
    rr_tp2 = _compute_risk_reward(entry_price, stop, tp2)

    # Confidence: bias + volume + bar quality + ATR expanding
    conf = bias.confidence * 0.5
    if vol_confirmed:
        conf += 0.2
    if body_ratio >= 0.6:
        conf += 0.1
    if bias.atr_expanding:
        conf += 0.1
    conf += 0.1  # Base credit for price actually breaking
    conf = min(1.0, conf)

    reasoning_parts = [
        f"Breakout through {level_name.replace('_', ' ')} ({breakout_level:.{decimals}f})",
        f"{direction} bias ({bias.confidence:.0%})",
    ]
    if vol_confirmed:
        reasoning_parts.append("volume confirmed")
    else:
        reasoning_parts.append("no volume confirmation")
    if bias.atr_expanding:
        reasoning_parts.append("ATR expanding")

    signal = SwingSignal(
        asset_name=asset_name,
        entry_style=SwingEntryStyle.BREAKOUT,
        direction=direction,
        confidence=conf,
        entry_price=round(entry_price, decimals),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop,
        tp1=tp1,
        tp2=tp2,
        atr=atr,
        risk_reward_tp1=rr_tp1,
        risk_reward_tp2=rr_tp2,
        risk_dollars=risk_dollars,
        position_size=size,
        reasoning=" | ".join(reasoning_parts),
        key_level_used=level_name,
        key_level_price=breakout_level,
        confirmation_bar_idx=len(bars) - 1,
        detected_at=datetime.now(tz=_EST).isoformat(),
        phase=SwingPhase.ENTRY_READY,
    )

    logger.info(
        "Breakout signal: %s %s through %.{dec}f (vol=%s, conf=%.0f%%, RR=%.1f/%.1f)".replace("{dec}", str(decimals)),
        direction,
        asset_name,
        breakout_level,
        "yes" if vol_confirmed else "no",
        conf * 100,
        rr_tp1,
        rr_tp2,
    )

    return signal


# ---------------------------------------------------------------------------
# Gap Continuation Entry Detection
# ---------------------------------------------------------------------------


def detect_gap_continuation(
    bars: pd.DataFrame,
    bias: DailyBias,
    current_price: float,
    atr: float,
    session_open_price: float | None = None,
    asset_name: str = "",
    account_size: int = 50_000,
) -> SwingSignal | None:
    """Detect a gap continuation entry.

    Logic:
    1. Overnight gap must align with daily bias direction.
    2. Gap must be at least 0.3× ATR in size.
    3. Gap must NOT have been filled (price hasn't retraced 50%+ of gap).
    4. Must have at least 6 bars (30 min on 5m) of settled price action.
    5. Enter on first pullback within the gap zone.

    Args:
        bars: Intraday DataFrame with OHLCV columns. At least 6 bars.
        bias: Daily bias from bias_analyzer.
        current_price: Latest price.
        atr: Current ATR value.
        session_open_price: Today's session open (Globex open). If None,
                            uses bias.key_levels.overnight_high/low or
                            the first bar's Open.
        asset_name: Human-readable name.
        account_size: For position sizing.

    Returns:
        SwingSignal if gap continuation entry is detected, None otherwise.
    """
    if bias.direction == BiasDirection.NEUTRAL:
        return None

    if bars is None or len(bars) < GAP_SETTLE_BARS:
        return None

    if atr <= 0:
        return None

    direction = bias.direction.value
    kl = bias.key_levels
    if kl is None:
        return None

    # Determine the gap: session open vs prior day close
    prior_close = kl.prior_day_close if kl else 0.0
    if prior_close <= 0:
        return None

    # Resolve session open
    open_price = session_open_price
    if open_price is None or open_price <= 0:
        # Try from bars
        open_price = _safe_float(bars.iloc[0].get("Open", 0))
    if open_price <= 0:
        return None

    gap_size = open_price - prior_close  # Positive = gap up, negative = gap down
    gap_abs = abs(gap_size)

    # Minimum gap size
    if gap_abs < atr * GAP_MIN_ATR_RATIO:
        return None

    # Gap must align with bias
    gap_is_up = gap_size > 0
    if direction == "LONG" and not gap_is_up:
        return None
    if direction == "SHORT" and gap_is_up:
        return None

    # Check gap fill: has price retraced 50%+ of the gap?
    fill_level = prior_close + gap_size * (1.0 - GAP_FILL_THRESHOLD)
    if direction == "LONG":
        # Gap up: fill means price went below fill_level
        session_low = _safe_float(bars["Low"].min())
        if session_low <= fill_level:
            return None  # Gap was filled
    else:
        # Gap down: fill means price went above fill_level
        session_high = _safe_float(bars["High"].max())
        if session_high >= fill_level:
            return None  # Gap was filled

    # Ensure enough bars have passed (30 min settle period)
    if len(bars) < GAP_SETTLE_BARS:
        return None

    # Look for a pullback within the gap zone
    # For LONG: price pulling back within the gap (between prior_close and open)
    # For SHORT: price pulling back within the gap (between open and prior_close)
    settle_bars = bars.iloc[GAP_SETTLE_BARS:]
    if len(settle_bars) == 0:
        return None

    # Check if current price is in a pullback zone (within the gap or just above/below it)
    pullback_detected = False
    if direction == "LONG":
        # Price should have pulled back toward the bottom of the gap
        gap_bottom = prior_close
        gap_top = open_price
        pullback_zone_low = gap_bottom
        pullback_zone_high = gap_top + atr * 0.3
        if pullback_zone_low <= current_price <= pullback_zone_high:
            pullback_detected = True
    else:
        gap_bottom = open_price
        gap_top = prior_close
        pullback_zone_low = gap_bottom - atr * 0.3
        pullback_zone_high = gap_top
        if pullback_zone_low <= current_price <= pullback_zone_high:
            pullback_detected = True

    if not pullback_detected:
        return None

    # Confirmation: recent bar should be showing resumption in bias direction
    last_bar = bars.iloc[-1]
    lb_open = _safe_float(last_bar.get("Open", 0))
    lb_close = _safe_float(last_bar.get("Close", 0))

    confirmed = False
    if direction == "LONG" and lb_close > lb_open or direction == "SHORT" and lb_close < lb_open:
        confirmed = True

    tick_size = _get_tick_size(asset_name)
    decimals = _price_decimals_from_tick(tick_size)

    entry_price = current_price
    if direction == "LONG":
        stop = round(prior_close - atr * 0.5, decimals)  # Below gap origin
        tp1 = round(entry_price + atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price + atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(prior_close, decimals)
        entry_zone_high = round(open_price, decimals)
    else:
        stop = round(prior_close + atr * 0.5, decimals)  # Above gap origin
        tp1 = round(entry_price - atr * SWING_TP1_ATR_MULT, decimals)
        tp2 = round(entry_price - atr * SWING_TP2_ATR_MULT, decimals)
        entry_zone_low = round(open_price, decimals)
        entry_zone_high = round(prior_close, decimals)

    size, risk_dollars = _compute_position_size(entry_price, stop, account_size, asset_name)

    rr_tp1 = _compute_risk_reward(entry_price, stop, tp1)
    rr_tp2 = _compute_risk_reward(entry_price, stop, tp2)

    gap_atr_ratio = gap_abs / atr if atr > 0 else 0

    conf = bias.confidence * 0.5
    if confirmed:
        conf += 0.2
    if gap_atr_ratio >= 0.6:
        conf += 0.15  # Bigger gap = stronger conviction
    elif gap_atr_ratio >= 0.3:
        conf += 0.1
    if bias.atr_expanding:
        conf += 0.05
    conf = min(1.0, conf)

    reasoning_parts = [
        f"Gap {'up' if gap_is_up else 'down'} ({gap_abs:.{decimals}f}, {gap_atr_ratio:.1f}× ATR)",
        f"aligns with {direction} bias ({bias.confidence:.0%})",
        "gap unfilled",
    ]
    if confirmed:
        reasoning_parts.append("pullback + resumption confirmed")
    else:
        reasoning_parts.append("awaiting resumption bar")

    signal = SwingSignal(
        asset_name=asset_name,
        entry_style=SwingEntryStyle.GAP_CONTINUATION,
        direction=direction,
        confidence=conf,
        entry_price=round(entry_price, decimals),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop,
        tp1=tp1,
        tp2=tp2,
        atr=atr,
        risk_reward_tp1=rr_tp1,
        risk_reward_tp2=rr_tp2,
        risk_dollars=risk_dollars,
        position_size=size,
        reasoning=" | ".join(reasoning_parts),
        key_level_used="gap_zone",
        key_level_price=open_price,
        confirmation_bar_idx=len(bars) - 1 if confirmed else -1,
        detected_at=datetime.now(tz=_EST).isoformat(),
        phase=SwingPhase.ENTRY_READY if confirmed else SwingPhase.WATCHING,
    )

    logger.info(
        "Gap continuation signal: %s %s (gap=%.{dec}f, %.1f× ATR, conf=%.0f%%)".replace("{dec}", str(decimals)),
        direction,
        asset_name,
        gap_abs,
        gap_atr_ratio,
        conf * 100,
    )

    return signal


# ---------------------------------------------------------------------------
# Combined Entry Detection — Main API
# ---------------------------------------------------------------------------


def detect_swing_entries(
    bars: pd.DataFrame,
    bias: DailyBias,
    current_price: float,
    atr: float,
    asset_name: str = "",
    account_size: int = 50_000,
    session_open_price: float | None = None,
    enabled_styles: list[SwingEntryStyle] | None = None,
) -> list[SwingSignal]:
    """Run all swing entry detectors and return any signals found.

    This is the primary entry point. It runs pullback, breakout, and gap
    continuation detectors and returns all signals sorted by confidence.

    Args:
        bars: Intraday DataFrame with OHLCV columns.
        bias: Daily bias from bias_analyzer.
        current_price: Latest price.
        atr: Current ATR value.
        asset_name: Human-readable name.
        account_size: For position sizing.
        session_open_price: Today's session open for gap analysis.
        enabled_styles: Which entry styles to check. None = all.

    Returns:
        List of SwingSignal objects, sorted by confidence descending.
        May be empty if no entry conditions are met.
    """
    if bias.direction == BiasDirection.NEUTRAL:
        logger.debug("No swing entries for %s — neutral bias", asset_name)
        return []

    if bars is None or len(bars) < 5:
        return []

    if atr <= 0:
        return []

    # Check time stop — don't open new positions after 15:30 ET
    if _is_time_stop_due():
        logger.debug("No swing entries for %s — past time stop", asset_name)
        return []

    if enabled_styles is None:
        enabled_styles = list(SwingEntryStyle)

    signals: list[SwingSignal] = []

    # 1. Pullback entry
    if SwingEntryStyle.PULLBACK in enabled_styles:
        sig = detect_pullback_entry(
            bars=bars,
            bias=bias,
            current_price=current_price,
            atr=atr,
            asset_name=asset_name,
            account_size=account_size,
        )
        if sig is not None:
            signals.append(sig)

    # 2. Breakout entry
    if SwingEntryStyle.BREAKOUT in enabled_styles:
        sig = detect_breakout_entry(
            bars=bars,
            bias=bias,
            current_price=current_price,
            atr=atr,
            asset_name=asset_name,
            account_size=account_size,
        )
        if sig is not None:
            signals.append(sig)

    # 3. Gap continuation
    if SwingEntryStyle.GAP_CONTINUATION in enabled_styles:
        sig = detect_gap_continuation(
            bars=bars,
            bias=bias,
            current_price=current_price,
            atr=atr,
            session_open_price=session_open_price,
            asset_name=asset_name,
            account_size=account_size,
        )
        if sig is not None:
            signals.append(sig)

    # Sort by confidence descending
    signals.sort(key=lambda s: s.confidence, reverse=True)

    if signals:
        logger.info(
            "Swing entries for %s: %d signal(s) — %s",
            asset_name,
            len(signals),
            ", ".join(f"{s.entry_style.value}({s.confidence:.0%})" for s in signals),
        )

    return signals


# ---------------------------------------------------------------------------
# Exit Evaluation
# ---------------------------------------------------------------------------


def evaluate_swing_exits(
    bars: pd.DataFrame | None,
    entry_price: float,
    direction: str,
    stop_loss: float,
    tp1: float,
    tp2: float,
    current_price: float,
    atr: float = 0.0,
    phase: SwingPhase = SwingPhase.ACTIVE,
    highest_since_entry: float = 0.0,
    lowest_since_entry: float = float("inf"),
    risk_dollars: float = 0.0,
    point_value: float = 1.0,
    position_size: int = 1,
    now: datetime | None = None,
) -> list[SwingExitSignal]:
    """Evaluate whether any exit conditions are met for an active swing trade.

    Checks (in priority order):
    1. Stop loss hit
    2. TP1 hit (scale 50%)
    3. TP2 hit (close remaining)
    4. EMA-21 trailing stop (after TP1)
    5. Time stop (15:30 ET)

    Args:
        bars: Recent intraday bars for EMA computation.
        entry_price: Original entry price.
        direction: "LONG" or "SHORT".
        stop_loss: Current stop loss level.
        tp1: Take-profit level 1.
        tp2: Take-profit level 2.
        current_price: Current market price.
        atr: Current ATR (for trailing calculations).
        phase: Current swing phase (ACTIVE, TP1_HIT, TRAILING).
        highest_since_entry: Highest price since entry.
        lowest_since_entry: Lowest price since entry.
        risk_dollars: Risk in dollars per contract.
        point_value: Dollar value per point of price movement.
        position_size: Current position size.
        now: Current datetime (for time stop). None = use wall clock.

    Returns:
        List of SwingExitSignal objects. Empty if no exit triggered.
        Multiple signals possible (e.g., TP1 + trailing stop update).
    """
    exits: list[SwingExitSignal] = []

    if entry_price <= 0:
        return exits

    # Track extremes
    if direction == "LONG":
        highest_since_entry = max(highest_since_entry, current_price)
    else:
        lowest_since_entry = min(lowest_since_entry, current_price)

    # Compute P&L
    pnl_points = current_price - entry_price if direction == "LONG" else entry_price - current_price

    pnl_dollars = pnl_points * point_value * position_size
    risk_per_r = risk_dollars if risk_dollars > 0 else abs(entry_price - stop_loss) * point_value
    r_multiple = pnl_dollars / risk_per_r if risk_per_r > 0 else 0.0

    # ── 1. Stop Loss ────────────────────────────────────────────────────
    stop_hit = False
    if direction == "LONG" and current_price <= stop_loss or direction == "SHORT" and current_price >= stop_loss:
        stop_hit = True

    if stop_hit:
        exits.append(
            SwingExitSignal(
                reason=SwingExitReason.STOP_LOSS,
                exit_price=stop_loss,
                pnl_estimate=pnl_dollars,
                r_multiple=r_multiple,
                scale_fraction=1.0,
                reasoning=f"Stop loss hit at {stop_loss}",
            )
        )
        return exits  # Stop loss is terminal — no other exits matter

    # ── 2. TP1 Hit (scale out 50%) ─────────────────────────────────────
    if phase == SwingPhase.ACTIVE:
        tp1_hit = False
        if direction == "LONG" and current_price >= tp1 or direction == "SHORT" and current_price <= tp1:
            tp1_hit = True

        if tp1_hit:
            tp1_pnl = abs(tp1 - entry_price) * point_value * position_size * TP1_SCALE_FRACTION
            exits.append(
                SwingExitSignal(
                    reason=SwingExitReason.TP1_HIT,
                    exit_price=tp1,
                    pnl_estimate=tp1_pnl,
                    r_multiple=_compute_risk_reward(entry_price, stop_loss, tp1),
                    scale_fraction=TP1_SCALE_FRACTION,
                    reasoning=f"TP1 hit at {tp1} — scaling out {TP1_SCALE_FRACTION:.0%}",
                )
            )
            # Don't return — also check for trailing stop update

    # ── 3. TP2 Hit (close remaining) ───────────────────────────────────
    if phase in (SwingPhase.TP1_HIT, SwingPhase.TRAILING):
        tp2_hit = False
        if direction == "LONG" and current_price >= tp2 or direction == "SHORT" and current_price <= tp2:
            tp2_hit = True

        if tp2_hit:
            tp2_pnl = abs(tp2 - entry_price) * point_value * position_size * (1.0 - TP1_SCALE_FRACTION)
            exits.append(
                SwingExitSignal(
                    reason=SwingExitReason.TP2_HIT,
                    exit_price=tp2,
                    pnl_estimate=tp2_pnl,
                    r_multiple=_compute_risk_reward(entry_price, stop_loss, tp2),
                    scale_fraction=1.0 - TP1_SCALE_FRACTION,
                    reasoning=f"TP2 hit at {tp2} — closing remaining position",
                )
            )
            return exits  # TP2 is terminal for the remainder

    # ── 4. EMA-21 Trailing Stop (after TP1 hit) ────────────────────────
    if phase in (SwingPhase.TP1_HIT, SwingPhase.TRAILING) and bars is not None and len(bars) >= TRAIL_EMA_PERIOD + 3:
        ema = _compute_ema(bars["Close"], TRAIL_EMA_PERIOD)  # type: ignore[arg-type]
        ema_val = _safe_float(ema.iloc[-1])

        if ema_val > 0:
            ema_trail_hit = False
            if direction == "LONG" and current_price < ema_val or direction == "SHORT" and current_price > ema_val:
                ema_trail_hit = True

            if ema_trail_hit:
                remaining_frac = 1.0 - TP1_SCALE_FRACTION
                trail_pnl = pnl_points * point_value * position_size * remaining_frac
                exits.append(
                    SwingExitSignal(
                        reason=SwingExitReason.EMA_TRAIL,
                        exit_price=current_price,
                        pnl_estimate=trail_pnl,
                        r_multiple=r_multiple,
                        scale_fraction=remaining_frac,
                        trailing_stop_price=ema_val,
                        reasoning=f"EMA-{TRAIL_EMA_PERIOD} trail exit at {ema_val:.4f} (price={current_price})",
                    )
                )
                return exits
            else:
                # Not hit yet — report updated trailing level
                exits.append(
                    SwingExitSignal(
                        reason=SwingExitReason.TRAILING_STOP,
                        exit_price=0.0,  # Not exiting
                        pnl_estimate=pnl_dollars,
                        r_multiple=r_multiple,
                        scale_fraction=0.0,  # No exit action
                        trailing_stop_price=ema_val,
                        reasoning=f"EMA-{TRAIL_EMA_PERIOD} trailing stop at {ema_val:.4f}",
                    )
                )

    # ── 5. ATR-based Trailing Stop (fallback if EMA unavailable) ───────
    if (
        phase in (SwingPhase.TP1_HIT, SwingPhase.TRAILING)
        and atr > 0
        and (bars is None or len(bars) < TRAIL_EMA_PERIOD + 3)
    ):
        trail_distance = atr * SWING_TRAIL_ATR_MULT
        if direction == "LONG":
            atr_trail_level = highest_since_entry - trail_distance
            if current_price < atr_trail_level:
                remaining_frac = 1.0 - TP1_SCALE_FRACTION
                trail_pnl = pnl_points * point_value * position_size * remaining_frac
                exits.append(
                    SwingExitSignal(
                        reason=SwingExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        pnl_estimate=trail_pnl,
                        r_multiple=r_multiple,
                        scale_fraction=remaining_frac,
                        trailing_stop_price=atr_trail_level,
                        reasoning=f"ATR trail stop at {atr_trail_level:.4f} (high={highest_since_entry})",
                    )
                )
                return exits
        else:
            atr_trail_level = lowest_since_entry + trail_distance
            if current_price > atr_trail_level:
                remaining_frac = 1.0 - TP1_SCALE_FRACTION
                trail_pnl = pnl_points * point_value * position_size * remaining_frac
                exits.append(
                    SwingExitSignal(
                        reason=SwingExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        pnl_estimate=trail_pnl,
                        r_multiple=r_multiple,
                        scale_fraction=remaining_frac,
                        trailing_stop_price=atr_trail_level,
                        reasoning=f"ATR trail stop at {atr_trail_level:.4f} (low={lowest_since_entry})",
                    )
                )
                return exits

    # ── 6. Time Stop ───────────────────────────────────────────────────
    if _is_time_stop_due(now):
        remaining_frac = 1.0 if phase == SwingPhase.ACTIVE else (1.0 - TP1_SCALE_FRACTION)
        time_pnl = pnl_points * point_value * position_size * remaining_frac
        exits.append(
            SwingExitSignal(
                reason=SwingExitReason.TIME_STOP,
                exit_price=current_price,
                pnl_estimate=time_pnl,
                r_multiple=r_multiple,
                scale_fraction=remaining_frac,
                reasoning=f"Time stop at {TIME_STOP_HOUR}:{TIME_STOP_MINUTE:02d} ET — closing to avoid overnight",
            )
        )

    return exits


# ---------------------------------------------------------------------------
# Swing State Management Helpers
# ---------------------------------------------------------------------------


def create_swing_state(signal: SwingSignal) -> SwingState:
    """Create a new SwingState from a confirmed entry signal.

    Call this when a swing signal transitions to ENTRY_READY and you
    decide to execute it.
    """
    now = datetime.now(tz=_EST).isoformat()
    return SwingState(
        asset_name=signal.asset_name,
        signal=signal,
        phase=SwingPhase.ACTIVE,
        entry_price=signal.entry_price,
        current_stop=signal.stop_loss,
        tp1=signal.tp1,
        tp2=signal.tp2,
        direction=signal.direction,
        position_size=signal.position_size,
        remaining_size=signal.position_size,
        highest_price=signal.entry_price,
        lowest_price=signal.entry_price,
        entry_time=now,
        last_update=now,
    )


def update_swing_state(
    state: SwingState,
    current_price: float,
    bars: pd.DataFrame | None = None,
    atr: float = 0.0,
    point_value: float = 1.0,
    now: datetime | None = None,
) -> tuple[SwingState, list[SwingExitSignal]]:
    """Tick a swing state: update extremes, check exits, advance phase.

    Call this on every price update for an active swing trade.

    Args:
        state: Current swing state.
        current_price: Latest price.
        bars: Recent intraday bars (for EMA trailing).
        atr: Current ATR.
        point_value: Dollar value per price point.
        now: Current datetime (None = wall clock).

    Returns:
        (updated_state, exit_signals)
    """
    if state.phase == SwingPhase.CLOSED:
        return state, []

    # Update extremes
    if state.direction == "LONG":
        state.highest_price = max(state.highest_price, current_price)
    else:
        state.lowest_price = min(state.lowest_price, current_price)

    state.last_update = (now or datetime.now(tz=_EST)).isoformat()

    # Evaluate exits
    risk_dollars = state.signal.risk_dollars if state.signal else 0.0

    exit_signals = evaluate_swing_exits(
        bars=bars,
        entry_price=state.entry_price,
        direction=state.direction,
        stop_loss=state.current_stop,
        tp1=state.tp1,
        tp2=state.tp2,
        current_price=current_price,
        atr=atr,
        phase=state.phase,
        highest_since_entry=state.highest_price,
        lowest_since_entry=state.lowest_price,
        risk_dollars=risk_dollars,
        point_value=point_value,
        position_size=state.remaining_size,
        now=now,
    )

    # Process exit signals to update state
    for ex in exit_signals:
        if ex.reason == SwingExitReason.STOP_LOSS:
            state.phase = SwingPhase.CLOSED
            state.remaining_size = 0
            break

        elif ex.reason == SwingExitReason.TP1_HIT:
            state.phase = SwingPhase.TP1_HIT
            # Scale out: reduce remaining size
            scaled = int(math.ceil(state.position_size * TP1_SCALE_FRACTION))
            state.remaining_size = max(0, state.position_size - scaled)
            # Move stop to breakeven
            state.current_stop = state.entry_price

        elif ex.reason == SwingExitReason.TP2_HIT:
            state.phase = SwingPhase.CLOSED
            state.remaining_size = 0
            break

        elif ex.reason in (SwingExitReason.EMA_TRAIL, SwingExitReason.TRAILING_STOP):
            if ex.scale_fraction > 0 and ex.exit_price > 0:
                # Actual exit
                state.phase = SwingPhase.CLOSED
                state.remaining_size = 0
                break
            elif ex.trailing_stop_price > 0:
                # Update trailing stop level, don't close
                state.phase = SwingPhase.TRAILING
                # Only move stop in favorable direction
                if state.direction == "LONG":
                    state.current_stop = max(state.current_stop, ex.trailing_stop_price)
                else:
                    state.current_stop = min(state.current_stop, ex.trailing_stop_price)

        elif ex.reason == SwingExitReason.TIME_STOP:
            state.phase = SwingPhase.CLOSED
            state.remaining_size = 0
            break

    return state, exit_signals


# ---------------------------------------------------------------------------
# Batch / Multi-Asset Detection
# ---------------------------------------------------------------------------


def scan_swing_entries_all_assets(
    asset_bars: dict[str, pd.DataFrame],
    biases: dict[str, DailyBias],
    current_prices: dict[str, float],
    atrs: dict[str, float],
    account_size: int = 50_000,
    session_opens: dict[str, float] | None = None,
    enabled_styles: list[SwingEntryStyle] | None = None,
    max_signals: int = 5,
) -> list[SwingSignal]:
    """Scan all assets for swing entry signals.

    Convenience wrapper that calls detect_swing_entries() for each asset
    and returns the top signals across all assets.

    Args:
        asset_bars: {asset_name: intraday_df}
        biases: {asset_name: DailyBias}
        current_prices: {asset_name: float}
        atrs: {asset_name: float}
        account_size: Account size for sizing.
        session_opens: {asset_name: float} for gap analysis.
        enabled_styles: Which entry styles to check.
        max_signals: Maximum signals to return.

    Returns:
        Top signals across all assets, sorted by confidence descending.
    """
    all_signals: list[SwingSignal] = []

    for asset_name, bars in asset_bars.items():
        bias = biases.get(asset_name)
        if bias is None:
            continue

        price = current_prices.get(asset_name, 0.0)
        atr = atrs.get(asset_name, 0.0)
        session_open = session_opens.get(asset_name) if session_opens else None

        if price <= 0 or atr <= 0:
            continue

        signals = detect_swing_entries(
            bars=bars,
            bias=bias,
            current_price=price,
            atr=atr,
            asset_name=asset_name,
            account_size=account_size,
            session_open_price=session_open,
            enabled_styles=enabled_styles,
        )

        all_signals.extend(signals)

    # Sort all by confidence and return top N
    all_signals.sort(key=lambda s: s.confidence, reverse=True)

    if all_signals:
        logger.info(
            "Swing scan: %d signals across %d assets (top: %s %s %.0f%%)",
            len(all_signals),
            len(asset_bars),
            all_signals[0].direction,
            all_signals[0].asset_name,
            all_signals[0].confidence * 100,
        )

    return all_signals[:max_signals]


# ---------------------------------------------------------------------------
# Integration with DailyPlan swing candidates
# ---------------------------------------------------------------------------


def enrich_swing_candidates(
    swing_candidates: list[Any],
    asset_bars: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    atrs: dict[str, float],
    biases: dict[str, DailyBias],
    account_size: int = 50_000,
    session_opens: dict[str, float] | None = None,
) -> dict[str, list[SwingSignal]]:
    """Enrich DailyPlan swing candidates with live entry signals.

    Takes the swing candidates from generate_daily_plan() and runs
    the swing detector on each to find actual entry signals.

    Args:
        swing_candidates: List of SwingCandidate objects from DailyPlan.
        asset_bars: {asset_name: intraday_df}
        current_prices: {asset_name: float}
        atrs: {asset_name: float}
        biases: {asset_name: DailyBias}
        account_size: Account size for sizing.
        session_opens: Session opens for gap detection.

    Returns:
        {asset_name: [SwingSignal, ...]} — entry signals for each candidate.
    """
    result: dict[str, list[SwingSignal]] = {}

    for candidate in swing_candidates:
        name = candidate.asset_name if hasattr(candidate, "asset_name") else str(candidate)
        bias = biases.get(name)
        if bias is None:
            continue

        bars = asset_bars.get(name)
        price = current_prices.get(name, 0.0)
        atr = atrs.get(name, 0.0)
        session_open = session_opens.get(name) if session_opens else None

        if bars is None or price <= 0 or atr <= 0:
            continue

        signals = detect_swing_entries(
            bars=bars,
            bias=bias,
            current_price=price,
            atr=atr,
            asset_name=name,
            account_size=account_size,
            session_open_price=session_open,
        )

        if signals:
            result[name] = signals

    return result


# ---------------------------------------------------------------------------
# Redis publish / load for swing state
# ---------------------------------------------------------------------------

REDIS_KEY_SWING_SIGNALS = "engine:swing_signals"
REDIS_KEY_SWING_STATES = "engine:swing_states"
REDIS_PUBSUB_SWING = "dashboard:swing_update"


def publish_swing_signals(
    signals: list[SwingSignal],
    redis_client: Any,
) -> bool:
    """Publish detected swing signals to Redis for dashboard consumption.

    Args:
        signals: List of SwingSignal objects.
        redis_client: Redis client.

    Returns:
        True on success.
    """
    try:
        import json as _json

        payload = _json.dumps(
            [s.to_dict() for s in signals],
            default=str,
        )
        redis_client.set(REDIS_KEY_SWING_SIGNALS, payload)
        redis_client.expire(REDIS_KEY_SWING_SIGNALS, 18 * 3600)
        redis_client.publish(REDIS_PUBSUB_SWING, payload)

        logger.info("Published %d swing signals to Redis", len(signals))
        return True
    except Exception as exc:
        logger.error("Failed to publish swing signals: %s", exc)
        return False


def publish_swing_states(
    states: dict[str, SwingState],
    redis_client: Any,
) -> bool:
    """Publish active swing states to Redis.

    Args:
        states: {asset_name: SwingState}
        redis_client: Redis client.

    Returns:
        True on success.
    """
    try:
        import json as _json

        payload = _json.dumps(
            {name: st.to_dict() for name, st in states.items()},
            default=str,
        )
        redis_client.set(REDIS_KEY_SWING_STATES, payload)
        redis_client.expire(REDIS_KEY_SWING_STATES, 18 * 3600)

        logger.info("Published %d swing states to Redis", len(states))
        return True
    except Exception as exc:
        logger.error("Failed to publish swing states: %s", exc)
        return False


def load_swing_signals(redis_client: Any) -> list[SwingSignal]:
    """Load swing signals from Redis.

    Returns:
        List of SwingSignal objects, or empty list on failure.
    """
    try:
        import json as _json

        raw = redis_client.get(REDIS_KEY_SWING_SIGNALS)
        if raw is None:
            return []

        data = _json.loads(raw)
        signals = []
        for d in data:
            sig = SwingSignal(
                asset_name=d.get("asset_name", ""),
                entry_style=SwingEntryStyle(d.get("entry_style", "pullback_entry")),
                direction=d.get("direction", "LONG"),
                confidence=d.get("confidence", 0.0),
                entry_price=d.get("entry_price", 0.0),
                entry_zone_low=d.get("entry_zone_low", 0.0),
                entry_zone_high=d.get("entry_zone_high", 0.0),
                stop_loss=d.get("stop_loss", 0.0),
                tp1=d.get("tp1", 0.0),
                tp2=d.get("tp2", 0.0),
                atr=d.get("atr", 0.0),
                risk_reward_tp1=d.get("risk_reward_tp1", 0.0),
                risk_reward_tp2=d.get("risk_reward_tp2", 0.0),
                risk_dollars=d.get("risk_dollars", 0.0),
                position_size=d.get("position_size", 1),
                reasoning=d.get("reasoning", ""),
                key_level_used=d.get("key_level_used", ""),
                key_level_price=d.get("key_level_price", 0.0),
                confirmation_bar_idx=d.get("confirmation_bar_idx", -1),
                detected_at=d.get("detected_at", ""),
                phase=SwingPhase(d.get("phase", "watching")),
            )
            signals.append(sig)

        return signals
    except Exception as exc:
        logger.error("Failed to load swing signals from Redis: %s", exc)
        return []
