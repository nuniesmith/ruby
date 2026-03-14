"""
RB Publisher — Redis Publishing and Alert Dispatch for Range Breakouts
=======================================================================
Extracted from ``lib.services.engine.handlers.py`` (Phase 1G).

Provides a clean public API for publishing breakout results to Redis,
dispatching to the PositionManager, and sending push notifications.
These functions are **orchestration** — they call into Redis, the alert
system, and the position manager but contain no detection logic.

Public API::

    from lib.trading.strategies.rb.publisher import (
        publish_breakout_result,
        persist_breakout_result,
        dispatch_to_position_manager,
        send_breakout_alert,
        get_alert_template,
    )

    # After detection:
    persist_breakout_result(result, session_key="london_ny")
    if result.breakout_detected:
        publish_breakout_result(result, session_key="london_ny")
        dispatch_to_position_manager(result, bars_1m=bars, session_key="london_ny")
        send_breakout_alert(result, result.breakout_type, session_key="london_ny")

Design:
  - All errors are caught and logged — never raises into the caller.
  - Thread-safe: no shared mutable state.
  - Delegates to ``lib.services.engine.handlers`` for the actual
    implementation (avoiding duplication during the migration period).
  - Will become the canonical home for publishing logic once the
    engine handlers are fully migrated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from lib.core.breakout_types import BreakoutType

logger = logging.getLogger("strategies.rb.publisher")


# ---------------------------------------------------------------------------
# Alert message templates per breakout type
# ---------------------------------------------------------------------------

_ALERT_TEMPLATES: dict[BreakoutType, dict[str, str]] = {
    BreakoutType.ORB: {
        "emoji": "📊",
        "short": "ORB",
        "title": "Opening Range Breakout",
    },
    BreakoutType.PrevDay: {
        "emoji": "📊",
        "short": "PDR",
        "title": "Previous Day Range Breakout",
    },
    BreakoutType.InitialBalance: {
        "emoji": "📊",
        "short": "IB",
        "title": "Initial Balance Breakout",
    },
    BreakoutType.Consolidation: {
        "emoji": "📊",
        "short": "SQUEEZE",
        "title": "Consolidation Squeeze Breakout",
    },
    BreakoutType.Weekly: {
        "emoji": "📈",
        "short": "WEEKLY",
        "title": "Weekly Range Breakout",
    },
    BreakoutType.Monthly: {
        "emoji": "📈",
        "short": "MONTHLY",
        "title": "Monthly Range Breakout",
    },
    BreakoutType.Asian: {
        "emoji": "🌏",
        "short": "ASIAN",
        "title": "Asian Session Range Breakout",
    },
    BreakoutType.BollingerSqueeze: {
        "emoji": "💥",
        "short": "BBSQUEEZE",
        "title": "Bollinger Squeeze Breakout",
    },
    BreakoutType.ValueArea: {
        "emoji": "📊",
        "short": "VA",
        "title": "Value Area Breakout",
    },
    BreakoutType.InsideDay: {
        "emoji": "📦",
        "short": "INSIDE",
        "title": "Inside Day Breakout",
    },
    BreakoutType.GapRejection: {
        "emoji": "🔲",
        "short": "GAP",
        "title": "Gap Rejection Breakout",
    },
    BreakoutType.PivotPoints: {
        "emoji": "📍",
        "short": "PIVOT",
        "title": "Pivot Points Breakout",
    },
    BreakoutType.Fibonacci: {
        "emoji": "🔢",
        "short": "FIB",
        "title": "Fibonacci Retracement Breakout",
    },
}


def get_alert_template(bt: BreakoutType) -> dict[str, str]:
    """Return alert template for a breakout type, with sensible fallback.

    Args:
        bt: The ``BreakoutType`` to look up.

    Returns:
        Dict with keys ``"emoji"``, ``"short"``, ``"title"``.
    """
    return _ALERT_TEMPLATES.get(
        bt,
        {
            "emoji": "📊",
            "short": bt.name.upper(),
            "title": f"{bt.name} Breakout",
        },
    )


# ---------------------------------------------------------------------------
# Publishing functions
# ---------------------------------------------------------------------------


def publish_breakout_result(result: Any, session_key: str = "us") -> None:
    """Publish a breakout result to Redis for SSE / dashboard consumption.

    Delegates to ``lib.services.engine.handlers.publish_breakout_result()``
    during the migration period.  All errors are caught and logged.

    Args:
        result:      A ``BreakoutResult`` (or compatible) object.
        session_key: Session key for Redis key namespacing.
    """
    try:
        from lib.services.engine.handlers import (
            publish_breakout_result as _handler_publish,
        )

        _handler_publish(result, session_key=session_key)
    except ImportError:
        # Minimal fallback: direct Redis publish
        _minimal_redis_publish(result, session_key)
    except Exception as exc:
        logger.debug("publish_breakout_result error (non-fatal): %s", exc)


def persist_breakout_result(result: Any, session_key: str = "") -> int | None:
    """Persist a BreakoutResult to the database (orb_events table).

    Delegates to ``lib.services.engine.handlers.persist_breakout_result()``
    during the migration period.

    Args:
        result:      A ``BreakoutResult`` (or compatible) object.
        session_key: Session key for metadata.

    Returns:
        Row ID if persisted successfully, or ``None`` on failure.
    """
    try:
        from lib.services.engine.handlers import (
            persist_breakout_result as _handler_persist,
        )

        return _handler_persist(result, session_key=session_key)
    except ImportError:
        logger.debug("persist_breakout_result: handlers module not available")
    except Exception as exc:
        logger.debug("persist_breakout_result error (non-fatal): %s", exc)
    return None


def dispatch_to_position_manager(
    result: Any,
    bars_1m: pd.DataFrame | None = None,
    session_key: str = "us",
) -> None:
    """Forward a breakout result to the PositionManager for order execution.

    Delegates to ``lib.services.engine.handlers.dispatch_to_position_manager()``
    during the migration period.

    Args:
        result:      A ``BreakoutResult`` (or compatible) object.
        bars_1m:     1-minute bars for additional context.
        session_key: Session key for metadata.
    """
    try:
        from lib.services.engine.handlers import (
            dispatch_to_position_manager as _handler_dispatch,
        )

        _handler_dispatch(result, bars_1m=bars_1m, session_key=session_key)
    except ImportError:
        logger.debug("dispatch_to_position_manager: handlers module not available")
    except Exception as exc:
        logger.debug("dispatch_to_position_manager error (non-fatal): %s", exc)


def send_breakout_alert(
    result: Any,
    breakout_type: BreakoutType,
    session_key: str = "",
) -> None:
    """Send a push notification / alert for a detected breakout.

    Uses the alert template for the given breakout type to format a
    human-readable message.  Delegates to
    ``lib.services.engine.handlers.send_breakout_alert()`` during the
    migration period.

    Args:
        result:         A ``BreakoutResult`` (or compatible) object.
        breakout_type:  Which ``BreakoutType`` triggered.
        session_key:    Session key for alert deduplication.
    """
    try:
        from lib.services.engine.handlers import (
            send_breakout_alert as _handler_alert,
        )

        _handler_alert(result, breakout_type, session_key=session_key)
    except ImportError:
        # Fallback: try to send directly
        _fallback_send_alert(result, breakout_type, session_key)
    except Exception as exc:
        logger.debug("send_breakout_alert error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Convenience: full pipeline (detect → persist → publish → alert)
# ---------------------------------------------------------------------------


def publish_pipeline(
    result: Any,
    breakout_type: BreakoutType,
    session_key: str = "us",
    bars_1m: pd.DataFrame | None = None,
) -> None:
    """Run the full post-detection pipeline: persist → publish → dispatch → alert.

    Convenience function that combines all four steps.  Each step is
    independent and failure in one does not block the others.

    Args:
        result:         A ``BreakoutResult`` (or compatible) object.
        breakout_type:  Which ``BreakoutType`` triggered.
        session_key:    Session key for Redis key namespacing.
        bars_1m:        1-minute bars for position manager context.
    """
    # 1. Always persist (even non-detections for audit trail)
    persist_breakout_result(result, session_key=session_key)

    # 2-4. Only for confirmed breakouts
    if getattr(result, "breakout_detected", False):
        publish_breakout_result(result, session_key=session_key)
        dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
        send_breakout_alert(result, breakout_type, session_key=session_key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _minimal_redis_publish(result: Any, session_key: str) -> None:
    """Minimal Redis publish fallback when the handler module is unavailable."""
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    _EST = ZoneInfo("America/New_York")

    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        payload = result.to_dict() if hasattr(result, "to_dict") else {"symbol": str(result)}
        payload["published_at"] = datetime.now(tz=_EST).isoformat()
        payload["orb_session"] = session_key

        bt_name = getattr(result, "breakout_type", BreakoutType.ORB)
        if hasattr(bt_name, "name"):
            bt_name = bt_name.name
        symbol = getattr(result, "symbol", "unknown")

        key = f"engine:breakout:{str(bt_name).lower()}:{symbol}"
        cache_set(key, json.dumps(payload).encode(), ttl=300)

        if REDIS_AVAILABLE and _r is not None:
            _r.publish("dashboard:breakout", json.dumps(payload))
    except Exception as exc:
        logger.debug("_minimal_redis_publish fallback error: %s", exc)


def _fallback_send_alert(
    result: Any,
    breakout_type: BreakoutType,
    session_key: str,
) -> None:
    """Fallback alert sender when the handler module is unavailable."""
    try:
        from lib.core.alerts import send_signal

        tmpl = get_alert_template(breakout_type)
        symbol = getattr(result, "symbol", "unknown")
        direction = getattr(result, "direction", "")
        trigger_price = getattr(result, "trigger_price", 0.0)

        lines = [
            f"{tmpl['title']}!",
            f"Direction: {direction}",
            f"Trigger: {trigger_price:,.4f}",
        ]

        range_low = getattr(result, "range_low", 0.0)
        range_high = getattr(result, "range_high", 0.0)
        if range_low > 0 and range_high > 0:
            lines.append(f"Range: {range_low:,.4f} – {range_high:,.4f}")

        atr_value = getattr(result, "atr_value", 0.0)
        if atr_value > 0:
            lines.append(f"ATR: {atr_value:,.4f}")

        mtf_score = getattr(result, "mtf_score", None)
        if mtf_score is not None:
            lines.append(f"MTF Score: {mtf_score:.3f}")

        signal_key = f"{tmpl['short'].lower()}_{symbol}_{direction}"
        if session_key:
            signal_key = f"{signal_key}_{session_key}"

        send_signal(
            signal_key=signal_key,
            title=f"{tmpl['emoji']} {tmpl['short']} {direction}: {symbol}",
            message="\n".join(lines),
            asset=symbol,
            direction=direction,
        )
    except Exception as exc:
        logger.debug("_fallback_send_alert error: %s", exc)


__all__ = [
    "get_alert_template",
    "publish_breakout_result",
    "persist_breakout_result",
    "dispatch_to_position_manager",
    "send_breakout_alert",
    "publish_pipeline",
]
