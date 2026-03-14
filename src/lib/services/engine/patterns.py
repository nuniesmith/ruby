"""
"Should Not Trade" Detector
========================================
Rules-based filter that flags low-conviction days to prevent overtrading.

This module expands the basic `should_not_trade()` in focus.py into a
comprehensive detector that checks multiple conditions and returns
structured results for the dashboard NO TRADE banner.

Conditions for NO TRADE:
  1. All focus assets have quality < 55%
  2. Any focus asset has volatility percentile > 88% (extreme vol)
  3. Daily loss already exceeds -$250
  4. More than 2 consecutive losing trades today
  5. It's after 10:00 AM ET and no setups have triggered

Each condition is checked independently and returns a typed result so
the dashboard can display the specific reason(s) and the engine can
publish granular `no-trade-alert` SSE events.

Public API:
    result = evaluate_no_trade(focus_assets, risk_status=None, now=None)
    # result.should_skip  -> bool
    # result.reasons      -> list[str]
    # result.checks       -> list[NoTradeCheck]  (per-condition detail)

Usage:
    from lib.services.engine.patterns import evaluate_no_trade, publish_no_trade_alert

    result = evaluate_no_trade(focus_assets, risk_status=risk_mgr.get_status())
    if result.should_skip:
        publish_no_trade_alert(result)
"""

import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

from lib.core.utils import safe_float as _safe_float

logger = logging.getLogger("engine.patterns")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Thresholds (configurable via environment or constructor)
# ---------------------------------------------------------------------------
DEFAULT_MIN_QUALITY = 0.55  # 55% — below this, asset is low conviction
DEFAULT_EXTREME_VOL = 0.88  # 88th percentile — too volatile
DEFAULT_MAX_DAILY_LOSS = -250.0  # -$250 daily loss threshold
DEFAULT_MAX_CONSECUTIVE_LOSSES = 2  # after 2 consecutive losses, stop
DEFAULT_LATE_SESSION_HOUR = 10  # 10:00 AM ET cutoff
DEFAULT_SESSION_END_HOUR = 12  # 12:00 PM ET session end


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class NoTradeCondition(StrEnum):
    """Enumeration of all no-trade conditions."""

    ALL_LOW_QUALITY = "all_low_quality"
    EXTREME_VOLATILITY = "extreme_volatility"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    LATE_SESSION_NO_SETUPS = "late_session_no_setups"
    NO_MARKET_DATA = "no_market_data"
    SESSION_ENDED = "session_ended"


@dataclass
class NoTradeCheck:
    """Result of a single no-trade condition check."""

    condition: NoTradeCondition
    triggered: bool
    reason: str
    severity: str = "warning"  # "warning", "critical", "info"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class NoTradeResult:
    """Aggregated result of all no-trade checks."""

    should_skip: bool = False
    reasons: list[str] = field(default_factory=list)
    checks: list[NoTradeCheck] = field(default_factory=list)
    checked_at: str = ""
    primary_reason: str = ""
    severity: str = "info"  # highest severity across all triggered checks

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON / Redis publishing."""
        return {
            "should_skip": self.should_skip,
            "reasons": self.reasons,
            "primary_reason": self.primary_reason,
            "severity": self.severity,
            "checked_at": self.checked_at,
            "checks": [
                {
                    "condition": c.condition.value,
                    "triggered": c.triggered,
                    "reason": c.reason,
                    "severity": c.severity,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Individual condition checkers
# ---------------------------------------------------------------------------


def _check_no_market_data(
    focus_assets: list[dict[str, Any]],
) -> NoTradeCheck:
    """Check: Do we have any market data at all?"""
    if not focus_assets:
        return NoTradeCheck(
            condition=NoTradeCondition.NO_MARKET_DATA,
            triggered=True,
            reason="No market data available — cannot assess any assets",
            severity="critical",
        )
    return NoTradeCheck(
        condition=NoTradeCondition.NO_MARKET_DATA,
        triggered=False,
        reason="",
        details={"asset_count": len(focus_assets)},
    )


def _check_all_low_quality(
    focus_assets: list[dict[str, Any]],
    min_quality: float = DEFAULT_MIN_QUALITY,
) -> NoTradeCheck:
    """Check: Are ALL focus assets below the quality threshold?"""
    if not focus_assets:
        return NoTradeCheck(
            condition=NoTradeCondition.ALL_LOW_QUALITY,
            triggered=False,
            reason="",
        )

    qualities = [_safe_float(a.get("quality", 0)) for a in focus_assets]
    all_below = all(q < min_quality for q in qualities)
    best_q = max(qualities) if qualities else 0
    best_asset = ""
    for a in focus_assets:
        if _safe_float(a.get("quality", 0)) == best_q:
            best_asset = a.get("symbol", "?")
            break

    if all_below:
        return NoTradeCheck(
            condition=NoTradeCondition.ALL_LOW_QUALITY,
            triggered=True,
            reason=(f"All assets below {min_quality * 100:.0f}% quality (best: {best_asset} at {best_q * 100:.0f}%)"),
            severity="critical",
            details={
                "threshold": min_quality,
                "best_quality": round(best_q, 3),
                "best_asset": best_asset,
                "qualities": {a.get("symbol", "?"): round(_safe_float(a.get("quality", 0)), 3) for a in focus_assets},
            },
        )

    return NoTradeCheck(
        condition=NoTradeCondition.ALL_LOW_QUALITY,
        triggered=False,
        reason="",
        details={
            "best_quality": round(best_q, 3),
            "best_asset": best_asset,
            "above_threshold": sum(1 for q in qualities if q >= min_quality),
        },
    )


def _check_extreme_volatility(
    focus_assets: list[dict[str, Any]],
    extreme_vol: float = DEFAULT_EXTREME_VOL,
) -> NoTradeCheck:
    """Check: Does ANY focus asset have extreme volatility?"""
    if not focus_assets:
        return NoTradeCheck(
            condition=NoTradeCondition.EXTREME_VOLATILITY,
            triggered=False,
            reason="",
        )

    extreme_assets: list[str] = []
    vol_details: dict[str, float] = {}

    for a in focus_assets:
        vol_pct = _safe_float(a.get("vol_percentile", 0))
        sym = a.get("symbol", "?")
        vol_details[sym] = round(vol_pct, 4)
        if vol_pct > extreme_vol:
            extreme_assets.append(sym)

    if extreme_assets:
        max_vol = max(vol_details.values())
        return NoTradeCheck(
            condition=NoTradeCondition.EXTREME_VOLATILITY,
            triggered=True,
            reason=(
                f"Extreme volatility on {', '.join(extreme_assets)} "
                f"({max_vol:.0%} percentile) — high risk of stop hunts"
            ),
            severity="critical",
            details={
                "threshold": extreme_vol,
                "extreme_assets": extreme_assets,
                "vol_percentiles": vol_details,
            },
        )

    return NoTradeCheck(
        condition=NoTradeCondition.EXTREME_VOLATILITY,
        triggered=False,
        reason="",
        details={"vol_percentiles": vol_details},
    )


def _check_daily_loss(
    risk_status: dict[str, Any] | None,
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
) -> NoTradeCheck:
    """Check: Has the daily loss exceeded the threshold?"""
    if risk_status is None:
        return NoTradeCheck(
            condition=NoTradeCondition.DAILY_LOSS_EXCEEDED,
            triggered=False,
            reason="",
            details={"note": "No risk status available — skipping daily loss check"},
        )

    daily_pnl = _safe_float(risk_status.get("daily_pnl", 0))

    if daily_pnl <= max_daily_loss:
        return NoTradeCheck(
            condition=NoTradeCondition.DAILY_LOSS_EXCEEDED,
            triggered=True,
            reason=(f"Daily loss ${daily_pnl:,.2f} exceeds limit ${max_daily_loss:,.2f} — stop trading"),
            severity="critical",
            details={
                "daily_pnl": round(daily_pnl, 2),
                "threshold": max_daily_loss,
                "trade_count": risk_status.get("daily_trade_count", 0),
            },
        )

    return NoTradeCheck(
        condition=NoTradeCondition.DAILY_LOSS_EXCEEDED,
        triggered=False,
        reason="",
        details={
            "daily_pnl": round(daily_pnl, 2),
            "headroom": round(daily_pnl - max_daily_loss, 2),
        },
    )


def _check_consecutive_losses(
    risk_status: dict[str, Any] | None,
    max_consecutive: int = DEFAULT_MAX_CONSECUTIVE_LOSSES,
) -> NoTradeCheck:
    """Check: Have there been too many consecutive losing trades?"""
    if risk_status is None:
        return NoTradeCheck(
            condition=NoTradeCondition.CONSECUTIVE_LOSSES,
            triggered=False,
            reason="",
            details={"note": "No risk status available — skipping loss streak check"},
        )

    streak = int(risk_status.get("consecutive_losses", 0))

    if streak > max_consecutive:
        return NoTradeCheck(
            condition=NoTradeCondition.CONSECUTIVE_LOSSES,
            triggered=True,
            reason=(f"{streak} consecutive losing trades (max allowed: {max_consecutive}) — take a break"),
            severity="warning",
            details={
                "consecutive_losses": streak,
                "threshold": max_consecutive,
            },
        )

    return NoTradeCheck(
        condition=NoTradeCondition.CONSECUTIVE_LOSSES,
        triggered=False,
        reason="",
        details={"consecutive_losses": streak},
    )


def _check_late_session_no_setups(
    focus_assets: list[dict[str, Any]],
    now: datetime | None = None,
    late_hour: int = DEFAULT_LATE_SESSION_HOUR,
    session_end_hour: int = DEFAULT_SESSION_END_HOUR,
) -> NoTradeCheck:
    """Check: Is it after 10 AM ET with no quality setups triggered?"""
    if now is None:
        now = datetime.now(tz=_EST)

    hour = now.hour

    # Only applies during the late-session window
    if hour < late_hour or hour >= session_end_hour:
        return NoTradeCheck(
            condition=NoTradeCondition.LATE_SESSION_NO_SETUPS,
            triggered=False,
            reason="",
            details={"current_hour": hour, "note": "Outside late-session window"},
        )

    # Check if any tradeable (non-skip) assets exist
    tradeable = [a for a in focus_assets if not a.get("skip", False)]

    if not tradeable:
        return NoTradeCheck(
            condition=NoTradeCondition.LATE_SESSION_NO_SETUPS,
            triggered=True,
            reason=(f"After {late_hour}:00 AM ET with no quality setups — session winding down"),
            severity="warning",
            details={
                "current_hour": hour,
                "total_assets": len(focus_assets),
                "tradeable_assets": 0,
            },
        )

    return NoTradeCheck(
        condition=NoTradeCondition.LATE_SESSION_NO_SETUPS,
        triggered=False,
        reason="",
        details={
            "current_hour": hour,
            "tradeable_assets": len(tradeable),
            "tradeable_symbols": [a.get("symbol", "?") for a in tradeable],
        },
    )


def _check_session_ended(
    now: datetime | None = None,
    session_end_hour: int = DEFAULT_SESSION_END_HOUR,
) -> NoTradeCheck:
    """Check: Has the trading session already ended?"""
    if now is None:
        now = datetime.now(tz=_EST)

    hour = now.hour

    if hour >= session_end_hour:
        return NoTradeCheck(
            condition=NoTradeCondition.SESSION_ENDED,
            triggered=True,
            reason="Trading session has ended for today",
            severity="info",
            details={"current_hour": hour, "session_end": session_end_hour},
        )

    return NoTradeCheck(
        condition=NoTradeCondition.SESSION_ENDED,
        triggered=False,
        reason="",
        details={"current_hour": hour, "session_end": session_end_hour},
    )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_no_trade(
    focus_assets: list[dict[str, Any]],
    risk_status: dict[str, Any] | None = None,
    now: datetime | None = None,
    min_quality: float = DEFAULT_MIN_QUALITY,
    extreme_vol: float = DEFAULT_EXTREME_VOL,
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
    max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES,
    late_hour: int = DEFAULT_LATE_SESSION_HOUR,
    session_end_hour: int = DEFAULT_SESSION_END_HOUR,
) -> NoTradeResult:
    """Evaluate all no-trade conditions and return a comprehensive result.

    This is the main entry point. Called every engine cycle during active
    hours and the result is published to Redis → SSE → dashboard.

    Args:
        focus_assets: List of asset focus dicts from compute_daily_focus().
        risk_status: Risk manager status dict (from RiskManager.get_status()).
        now: Override current time for testing.
        min_quality: Minimum quality threshold (0-1 scale).
        extreme_vol: Extreme volatility percentile threshold (0-1 scale).
        max_daily_loss: Max daily loss in dollars (negative number).
        max_consecutive_losses: Max consecutive losing trades allowed.
        late_hour: Hour (ET) after which late-session check applies.
        session_end_hour: Hour (ET) when session ends.

    Returns:
        NoTradeResult with should_skip, reasons, and detailed checks.
    """
    if now is None:
        now = datetime.now(tz=_EST)

    # Run all checks
    checks: list[NoTradeCheck] = [
        _check_no_market_data(focus_assets),
        _check_all_low_quality(focus_assets, min_quality=min_quality),
        _check_extreme_volatility(focus_assets, extreme_vol=extreme_vol),
        _check_daily_loss(risk_status, max_daily_loss=max_daily_loss),
        _check_consecutive_losses(risk_status, max_consecutive=max_consecutive_losses),
        _check_late_session_no_setups(
            focus_assets,
            now=now,
            late_hour=late_hour,
            session_end_hour=session_end_hour,
        ),
        _check_session_ended(now=now, session_end_hour=session_end_hour),
    ]

    # Collect triggered checks
    triggered = [c for c in checks if c.triggered]
    reasons = [c.reason for c in triggered]

    # Determine highest severity
    severity_order = {"info": 0, "warning": 1, "critical": 2}
    max_severity = "info"
    for c in triggered:
        if severity_order.get(c.severity, 0) > severity_order.get(max_severity, 0):
            max_severity = c.severity

    result = NoTradeResult(
        should_skip=len(triggered) > 0,
        reasons=reasons,
        checks=checks,
        checked_at=now.isoformat(),
        primary_reason=reasons[0] if reasons else "",
        severity=max_severity,
    )

    if result.should_skip:
        logger.warning(
            "⛔ NO TRADE: %d condition(s) triggered — %s",
            len(triggered),
            "; ".join(reasons),
        )
    else:
        logger.debug("✅ Trade conditions OK — all checks passed")

    return result


# ---------------------------------------------------------------------------
# Redis publishing
# ---------------------------------------------------------------------------


def publish_no_trade_alert(result: NoTradeResult) -> bool:
    """Publish a no-trade alert to Redis for SSE consumption.

    Writes to:
      - `engine:no_trade` — full result JSON (TTL 300s)
      - Redis PubSub `dashboard:no_trade` — trigger for SSE no-trade-alert event

    Returns True on success.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module")
        return False

    try:
        payload = result.to_dict()
        payload_json = json.dumps(payload, default=str, allow_nan=False)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize no-trade result: %s", exc)
        return False

    try:
        cache_set("engine:no_trade", payload_json.encode(), ttl=300)

        if REDIS_AVAILABLE and _r is not None:
            try:
                _r.publish("dashboard:no_trade", payload_json)
                logger.debug("No-trade alert published to Redis PubSub")
            except Exception as exc:
                logger.debug("Redis PubSub publish failed (non-fatal): %s", exc)

        return True

    except Exception as exc:
        logger.error("Failed to publish no-trade alert: %s", exc)
        return False


def clear_no_trade_alert() -> bool:
    """Clear any active no-trade alert from Redis.

    Called when conditions improve and trading is allowed again.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        return False

    try:
        cleared = json.dumps(
            {
                "should_skip": False,
                "reasons": [],
                "primary_reason": "",
                "severity": "info",
                "checked_at": datetime.now(tz=_EST).isoformat(),
                "checks": [],
            }
        )
        cache_set("engine:no_trade", cleared.encode(), ttl=300)

        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish(
                    "dashboard:no_trade",
                    json.dumps({"should_skip": False, "cleared": True}),
                )

        return True
    except Exception:
        return False
