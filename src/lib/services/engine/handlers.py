"""
Generic Breakout Handler Pipeline — Phase 1C/1D + Ruby Signal Engine
================================================
One handler function for **all** 13 breakout types — including ORB — with
optional quality-filter and CNN-inference stages that replace the ~800-line
``_handle_check_orb()`` in ``main.py``.

Public API::

    from lib.services.engine.handlers import (
        handle_breakout_check,
        handle_orb_check,
    )

    # Non-ORB types — simple:
    handle_breakout_check(engine, BreakoutType.PrevDay, session_key="london_ny")

    # ORB — full pipeline with filters + CNN:
    handle_orb_check(engine, session_key="us")
    handle_orb_check(engine, session_key="london")

    # Any type with filters + CNN enabled:
    handle_breakout_check(
        engine, BreakoutType.ORB, session_key="us",
        enable_filters=True, enable_cnn=True,
    )

Design:
  - Pure orchestration — no detection logic.  Delegates to
    ``detect_range_breakout()`` for detection.
  - Shared helpers (``fetch_bars_1m``, ``get_htf_bars``, ``run_mtf_on_result``,
    ``persist_breakout_result``, ``publish_breakout_result``,
    ``send_breakout_alert``) extracted from ``main.py``.
  - Optional quality-filter gate (``enable_filters=True``): runs
    ``apply_all_filters()`` with session-aware windows before publishing.
  - Optional CNN inference gate (``enable_cnn=True``): renders a chart
    snapshot and runs the hybrid CNN for breakout probability.
  - ``handle_orb_check()`` is a convenience wrapper that translates an
    ``ORBSession`` into the correct ``session_key``, ``config_override``,
    and ``enable_filters=True`` / ``enable_cnn=True`` flags.
  - Thread-safe, no shared mutable state.
  - All errors are caught and logged — never raises into the scheduler.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from datetime import datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

from lib.core.breakout_types import BreakoutType

logger = logging.getLogger("engine.handlers")

# Redis key for published Ruby signals (one per symbol, TTL 15 min)
_RUBY_SIGNAL_KEY_PREFIX = "engine:ruby_signal:"
_RUBY_SIGNAL_TTL = 15 * 60  # 15 minutes
# Aggregate map key: engine:ruby_signals → {symbol: signal_dict}
_RUBY_SIGNALS_MAP_KEY = "engine:ruby_signals"

_EST = ZoneInfo("America/New_York")


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


def _get_alert_template(bt: BreakoutType) -> dict[str, str]:
    """Return alert template for a breakout type, with sensible fallback."""
    return _ALERT_TEMPLATES.get(
        bt,
        {
            "emoji": "📊",
            "short": bt.name.upper(),
            "title": f"{bt.name} Breakout",
        },
    )


# ===========================================================================
# Shared helper functions (extracted from main.py)
# ===========================================================================


def get_assets_for_session_key(session_key: str) -> list[dict[str, Any]]:
    """Return the focus asset list filtered to the given session's asset set.

    Falls back to the full daily focus if no session-specific list is found.
    """
    try:
        from lib.core.cache import cache_get
        from lib.trading.strategies.rb.open import SESSION_ASSETS

        raw_focus = cache_get("engine:daily_focus")
        if not raw_focus:
            return []

        focus_data = json.loads(raw_focus)
        all_assets: list[dict[str, Any]] = focus_data.get("assets", [])

        session_tickers = set(SESSION_ASSETS.get(session_key, []))
        if not session_tickers:
            return all_assets

        return [
            a for a in all_assets if a.get("ticker", "") in session_tickers or a.get("symbol", "") in session_tickers
        ]
    except Exception as exc:
        logger.debug("get_assets_for_session_key(%s) error: %s", session_key, exc)
        return []


def fetch_bars_1m(engine: Any, ticker: str, symbol: str) -> pd.DataFrame | None:
    """Fetch 1-minute bars from cache or engine data service (best-effort)."""
    try:
        import pandas as pd

        from lib.core.cache import cache_get

        bars_key = f"engine:bars_1m:{ticker or symbol}"
        raw_bars = cache_get(bars_key)
        if raw_bars:
            raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
            return pd.read_json(io.StringIO(raw_str))

        with contextlib.suppress(Exception):
            return engine._fetch_tf_safe(ticker or symbol, interval="1m", period="1d")
    except Exception as exc:
        logger.debug("fetch_bars_1m(%s) error: %s", symbol, exc)
    return None


def get_htf_bars(
    bars_1m: pd.DataFrame | None,
    ticker: str,
) -> pd.DataFrame | None:
    """Get 15-minute bars for MTF enrichment — from cache or resampled from 1m.

    Returns ``None`` if no usable data is available.
    """
    import pandas as pd

    # Try cached 15-min bars first
    try:
        from lib.core.cache import cache_get

        htf_raw = cache_get(f"engine:bars_15m:{ticker}")
        if htf_raw:
            raw_str = htf_raw.decode("utf-8") if isinstance(htf_raw, bytes) else htf_raw
            return pd.read_json(io.StringIO(raw_str))
    except Exception:
        pass

    # Fall back to resampling 1-minute bars
    if bars_1m is not None and not bars_1m.empty:
        with contextlib.suppress(Exception):
            import pandas as pd

            resampled = (
                bars_1m.resample("15min")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
            if isinstance(resampled, pd.DataFrame):
                return resampled
    return None


def get_prev_day_levels(ticker: str, symbol: str) -> tuple[float | None, float | None]:
    """Try to get pre-computed previous-day high/low from the daily bars cache.

    Returns (prev_high, prev_low) or (None, None) if unavailable.
    """
    try:
        import pandas as pd

        from lib.core.cache import cache_get

        daily_key = f"engine:bars_daily:{ticker or symbol}"
        raw_daily = cache_get(daily_key)
        if raw_daily:
            raw_str = raw_daily.decode("utf-8") if isinstance(raw_daily, bytes) else raw_daily
            bars_daily = pd.read_json(io.StringIO(raw_str))
            if len(bars_daily) >= 2:
                return (
                    float(bars_daily["High"].iloc[-2]),
                    float(bars_daily["Low"].iloc[-2]),
                )
    except Exception:
        pass
    return None, None


def run_mtf_on_result(
    result: Any,
    bars_htf: pd.DataFrame | None,
) -> Any:
    """Run the MTF analyzer on a BreakoutResult and enrich it in-place.

    Returns the (possibly enriched) result.
    """
    if not getattr(result, "breakout_detected", False):
        return result
    if bars_htf is None or bars_htf.empty:
        return result
    try:
        from lib.analysis.mtf_analyzer import analyze_mtf

        mtf = analyze_mtf(bars_htf, direction=result.direction)
        result.mtf_score = mtf.mtf_score
        result.mtf_direction = mtf.ema_slope_direction
        result.macd_slope = mtf.macd_histogram_slope
        result.macd_divergence = mtf.divergence_detected
        result.extra["mtf"] = mtf.to_dict()
    except Exception as exc:
        logger.debug("run_mtf_on_result error (non-fatal): %s", exc)
    return result


def persist_breakout_result(result: Any, session_key: str = "") -> int | None:
    """Persist a BreakoutResult to orb_events table."""
    try:
        from lib.core.models import record_orb_event

        row_id = record_orb_event(
            symbol=result.symbol,
            or_high=result.range_high,
            or_low=result.range_low,
            or_range=result.range_size,
            atr_value=result.atr_value,
            breakout_detected=result.breakout_detected,
            direction=result.direction,
            trigger_price=result.trigger_price,
            long_trigger=result.long_trigger,
            short_trigger=result.short_trigger,
            bar_count=result.range_bar_count,
            session=session_key,
            metadata=result.extra or {},
            breakout_type=result.breakout_type.name,
            mtf_score=result.mtf_score,
            macd_slope=result.macd_slope,
            divergence=getattr(result, "divergence_type", "") or "",
        )
        return row_id
    except Exception as exc:
        logger.debug("persist_breakout_result error (non-fatal): %s", exc)
        return None


def publish_breakout_result(result: Any, session_key: str = "us") -> None:
    """Publish a breakout result to Redis for SSE / dashboard consumption.

    Delegates to ``main._publish_breakout_result()`` for TradingView signal
    integration.  If that function is unavailable, does a minimal Redis publish.
    """
    try:
        # Prefer the main.py publisher which handles TV signals + full pipeline
        from lib.services.engine.main import _publish_breakout_result

        _publish_breakout_result(result, orb_session_key=session_key)
    except ImportError:
        # Fallback: minimal Redis publish
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

            payload = result.to_dict()
            payload["published_at"] = datetime.now(tz=_EST).isoformat()
            payload["orb_session"] = session_key

            key = f"engine:breakout:{result.breakout_type.name.lower()}:{result.symbol}"
            cache_set(key, json.dumps(payload).encode(), ttl=300)

            if REDIS_AVAILABLE and _r is not None:
                _r.publish("dashboard:breakout", json.dumps(payload))
        except Exception:
            pass
    except Exception as exc:
        logger.debug("publish_breakout_result error (non-fatal): %s", exc)


def dispatch_to_position_manager(
    result: Any,
    bars_1m: pd.DataFrame | None = None,
    session_key: str = "us",
) -> None:
    """Forward a breakout result to the PositionManager for order execution.

    Delegates to ``main._dispatch_to_position_manager()`` so we don't
    duplicate the ORBResult compatibility shim logic.
    """
    try:
        from lib.services.engine.main import _dispatch_to_position_manager

        _dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)
    except ImportError:
        logger.debug("dispatch_to_position_manager: main module not available")
    except Exception as exc:
        logger.debug("dispatch_to_position_manager error (non-fatal): %s", exc)


def send_breakout_alert(
    result: Any,
    breakout_type: BreakoutType,
    session_key: str = "",
) -> None:
    """Send a push notification / alert for a detected breakout.

    Uses the alert template for the given breakout type to format a
    human-readable message.
    """
    try:
        from lib.core.alerts import send_signal

        tmpl = _get_alert_template(breakout_type)
        symbol = result.symbol

        # Build message body
        lines = [
            f"{tmpl['title']}!",
            f"Direction: {result.direction}",
            f"Trigger: {result.trigger_price:,.4f}",
            f"Range: {result.range_low:,.4f} – {result.range_high:,.4f}",
        ]

        # Type-specific extra lines
        if breakout_type == BreakoutType.PrevDay and getattr(result, "prev_day_high", 0) > 0:
            lines.append(f"PDR: {result.prev_day_low:,.4f} – {result.prev_day_high:,.4f}")
        elif breakout_type == BreakoutType.InitialBalance and getattr(result, "ib_high", 0) > 0:
            lines.append(f"IB: {getattr(result, 'ib_low', 0):,.4f} – {result.ib_high:,.4f}")
        elif breakout_type in (BreakoutType.Consolidation, BreakoutType.BollingerSqueeze) and getattr(
            result, "squeeze_detected", False
        ):
            lines.append(f"Squeeze: {result.squeeze_bar_count} bars, BB width {result.squeeze_bb_width:.4f}")

        lines.append(f"ATR: {result.atr_value:,.4f}")

        mtf_score = getattr(result, "mtf_score", None)
        if mtf_score is not None:
            lines.append(f"MTF Score: {mtf_score:.3f}")

        signal_key = f"{tmpl['short'].lower()}_{symbol}_{result.direction}"
        if session_key:
            signal_key = f"{signal_key}_{session_key}"

        send_signal(
            signal_key=signal_key,
            title=f"{tmpl['emoji']} {tmpl['short']} {result.direction}: {symbol}",
            message="\n".join(lines),
            asset=symbol,
            direction=result.direction,
        )
    except Exception as exc:
        logger.debug("send_breakout_alert error (non-fatal): %s", exc)


# ===========================================================================
# Ruby Signal Engine handler
# ===========================================================================


def handle_ruby_recompute(engine: Any, session_key: str = "us") -> None:
    """Run the RubySignalEngine over all focus assets and publish results.

    For each focus asset:
      1. Fetch the latest 1-minute bars from Redis / engine data cache.
      2. Feed each bar to the ``RubySignalEngine`` (stateful, incremental).
      3. Publish the latest ``RubySignal`` to Redis for the dashboard and
         WebUI to consume.
      4. If the signal has ``breakout_detected=True`` and ``filter_passed``,
         forward it to ``dispatch_to_position_manager()`` so the
         PositionManager can open / pyramid / reverse positions.

    The function is intentionally side-effect-free with respect to the
    engine object itself — it only reads bar data and writes to Redis.

    All errors are caught per-asset so one bad asset never blocks the rest.
    """
    try:
        import json as _json

        import pandas as pd

        from lib.core.cache import cache_get, cache_set
        from lib.services.engine.ruby_signal_engine import RubySignal, get_ruby_engine

        assets = get_assets_for_session_key(session_key)
        if not assets:
            # Fall back to all focus assets when session key is too restrictive
            assets = get_assets_for_session_key("us") or get_assets_for_session_key("london_ny") or []

        if not assets:
            logger.debug("handle_ruby_recompute: no assets found for session=%s", session_key)
            return

        signals_map: dict[str, dict] = {}
        published = 0
        dispatched = 0

        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "") or symbol
            if not symbol:
                continue

            try:
                # 1. Fetch 1-minute bars
                bars_1m = fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    logger.debug("handle_ruby_recompute: no bars for %s", symbol)
                    continue

                # 2. Get (or create) the engine for this symbol
                ruby_eng = get_ruby_engine(symbol)

                # Feed new bars to the engine.
                # We track the last bar timestamp so we only process genuinely
                # new bars — avoids re-processing the entire history on every
                # 5-minute tick.
                last_ts_key = f"engine:ruby_last_ts:{symbol}"
                last_ts_raw = cache_get(last_ts_key)
                last_ts: Any = None
                if last_ts_raw:
                    with contextlib.suppress(Exception):
                        last_ts = pd.Timestamp(last_ts_raw.decode() if isinstance(last_ts_raw, bytes) else last_ts_raw)

                # Normalise index to a datetime index
                if not isinstance(bars_1m.index, pd.DatetimeIndex):
                    with contextlib.suppress(Exception):
                        bars_1m.index = pd.to_datetime(bars_1m.index, utc=True)

                # Filter to only new bars
                bars_new = bars_1m.tail(200)  # default: most recent 200 bars
                if last_ts is not None:
                    with contextlib.suppress(Exception):
                        bars_new = bars_1m[bars_1m.index > last_ts]

                if bars_new.empty:
                    # No new bars — use cached last signal for the map
                    last_sig_raw = cache_get(f"{_RUBY_SIGNAL_KEY_PREFIX}{symbol}")
                    if last_sig_raw:
                        with contextlib.suppress(Exception):
                            signals_map[symbol] = _json.loads(
                                last_sig_raw.decode() if isinstance(last_sig_raw, bytes) else last_sig_raw
                            )
                    continue

                # Feed bars one-by-one (incremental, stateful)
                latest_signal: RubySignal | None = None
                col_map = {c.lower(): c for c in bars_new.columns}

                for ts, row in bars_new.iterrows():
                    bar_dict = {
                        "time": ts,
                        "open": float(row.get(col_map.get("open", "Open"), row.get("Open", 0.0)) or 0.0),
                        "high": float(row.get(col_map.get("high", "High"), row.get("High", 0.0)) or 0.0),
                        "low": float(row.get(col_map.get("low", "Low"), row.get("Low", 0.0)) or 0.0),
                        "close": float(row.get(col_map.get("close", "Close"), row.get("Close", 0.0)) or 0.0),
                        "volume": float(row.get(col_map.get("volume", "Volume"), row.get("Volume", 0.0)) or 0.0),
                    }
                    latest_signal = ruby_eng.update(bar_dict)

                if latest_signal is None:
                    continue

                # Update last-processed timestamp
                if isinstance(bars_new.index, pd.DatetimeIndex) and len(bars_new) > 0:
                    with contextlib.suppress(Exception):
                        cache_set(last_ts_key, str(bars_new.index[-1]).encode("utf-8"), ttl=_RUBY_SIGNAL_TTL * 2)

                # 3. Publish signal to Redis
                sig_dict = latest_signal.to_dict()
                sig_json = _json.dumps(sig_dict).encode("utf-8")
                cache_set(f"{_RUBY_SIGNAL_KEY_PREFIX}{symbol}", sig_json, ttl=_RUBY_SIGNAL_TTL)
                signals_map[symbol] = sig_dict
                published += 1

                # Log non-trivial signals
                if latest_signal.breakout_detected:
                    logger.info(
                        "🔔 Ruby %s: %s %s @ %.4f  quality=%.0f%%  regime=%s  wave=%.2fx",
                        latest_signal.signal_class or "SIGNAL",
                        latest_signal.direction,
                        symbol,
                        latest_signal.trigger_price,
                        latest_signal.quality,
                        latest_signal.regime,
                        latest_signal.wave_ratio,
                    )

                    # 4. Forward to PositionManager if quality gate passes
                    if latest_signal.filter_passed and latest_signal.cnn_prob >= 0.45:
                        dispatch_to_position_manager(
                            latest_signal,
                            bars_1m=bars_1m,
                            session_key=session_key,
                        )
                        dispatched += 1

            except Exception as exc:
                logger.debug("handle_ruby_recompute error for %s: %s", symbol, exc)

        # Publish the aggregate signals map so the dashboard can render all
        # symbols in one Redis read.
        if signals_map:
            with contextlib.suppress(Exception):
                cache_set(_RUBY_SIGNALS_MAP_KEY, _json.dumps(signals_map).encode("utf-8"), ttl=_RUBY_SIGNAL_TTL)

        if published:
            logger.info(
                "✅ Ruby recompute [%s]: %d signal(s) published, %d dispatched to PM",
                session_key,
                published,
                dispatched,
            )
        else:
            logger.debug("Ruby recompute [%s]: no new bars processed", session_key)

    except Exception as exc:
        logger.warning("handle_ruby_recompute top-level error (non-fatal): %s", exc)


# ===========================================================================
# Main generic handler
# ===========================================================================


# ===========================================================================
# Session-aware filter windows
# ===========================================================================


def get_filter_windows_for_session(
    session_key: str,
) -> tuple[list[tuple[dt_time, dt_time]], dt_time, bool]:
    """Return (allowed_windows, premarket_end, enable_lunch) for a session.

    These drive the ``apply_all_filters()`` call with correct time windows
    per session, matching the logic previously inlined in ``_handle_check_orb``.

    Returns:
        Tuple of (allowed_windows, pm_end, enable_lunch_filter).
    """
    if session_key == "cme":
        return [(dt_time(18, 0), dt_time(20, 0))], dt_time(18, 0), False
    if session_key == "sydney":
        return [(dt_time(18, 30), dt_time(20, 30))], dt_time(18, 30), False
    if session_key == "tokyo":
        return [(dt_time(19, 0), dt_time(21, 0))], dt_time(19, 0), False
    if session_key == "shanghai":
        return [(dt_time(21, 0), dt_time(23, 0))], dt_time(21, 0), False
    if session_key == "frankfurt":
        return [(dt_time(3, 0), dt_time(4, 30))], dt_time(3, 0), False
    if session_key == "london":
        return [(dt_time(3, 0), dt_time(5, 0))], dt_time(3, 0), False
    if session_key == "london_ny":
        return [(dt_time(8, 0), dt_time(10, 0))], dt_time(8, 0), False
    if session_key == "cme_settle":
        return [(dt_time(14, 0), dt_time(15, 30))], dt_time(8, 20), False
    if session_key in ("crypto_utc0", "crypto_utc12"):
        return [(dt_time(0, 0), dt_time(23, 59))], dt_time(0, 0), False
    # US session (default)
    return [(dt_time(8, 20), dt_time(10, 30))], dt_time(8, 20), True


# ===========================================================================
# Quality filter pipeline
# ===========================================================================


def run_quality_filters(
    result: Any,
    bars_1m: pd.DataFrame | None,
    bars_daily: pd.DataFrame | None,
    bars_htf: pd.DataFrame | None,
    session_key: str = "us",
) -> tuple[bool, str]:
    """Run breakout quality filters (NR7, session window, MTF, etc.).

    Returns ``(passed, summary_string)``.  On error, returns ``(True, "")``
    so the breakout is allowed through (fail-open).
    """
    try:
        from lib.analysis.breakout_filters import (
            apply_all_filters,
            extract_premarket_range,
        )
    except ImportError:
        logger.debug("Quality-filter module not available — breakout allowed")
        return True, ""

    try:
        allowed_windows, pm_end, enable_lunch = get_filter_windows_for_session(session_key)

        pm_high, pm_low = extract_premarket_range(bars_1m, pm_end=pm_end)

        signal_time = datetime.now(tz=_EST)
        gate_mode = os.environ.get("ORB_FILTER_GATE", "majority")
        mtf_min_score = float(os.environ.get("ORB_MTF_MIN_SCORE", "0.55"))

        filter_result = apply_all_filters(
            direction=result.direction,
            trigger_price=result.trigger_price,
            signal_time=signal_time,
            bars_daily=bars_daily,
            bars_1m=bars_1m,
            bars_htf=bars_htf,
            premarket_high=pm_high,
            premarket_low=pm_low,
            orb_high=getattr(result, "range_high", 0) or getattr(result, "or_high", 0),
            orb_low=getattr(result, "range_low", 0) or getattr(result, "or_low", 0),
            gate_mode=gate_mode,
            allowed_windows=allowed_windows,
            enable_lunch_filter=enable_lunch,
            enable_mtf_analyzer=True,
            mtf_min_pass_score=mtf_min_score,
        )
        return filter_result.passed, filter_result.summary

    except Exception as exc:
        logger.warning("Quality-filter error (allowing breakout): %s", exc)
        return True, ""


# ===========================================================================
# CNN inference pipeline
# ===========================================================================


def _get_daily_bars(ticker: str, symbol: str) -> pd.DataFrame | None:
    """Load cached daily bars for NR7 check and CNN features."""
    try:
        import pandas as pd

        from lib.core.cache import cache_get

        daily_key = f"engine:bars_daily:{ticker or symbol}"
        raw_daily = cache_get(daily_key)
        if raw_daily:
            raw_str = raw_daily.decode("utf-8") if isinstance(raw_daily, bytes) else raw_daily
            return pd.read_json(io.StringIO(raw_str))
    except Exception:
        pass
    return None


def build_cnn_tabular_features(
    result: Any,
    bars_1m: pd.DataFrame | None,
    bars_daily: pd.DataFrame | None,
    session_key: str = "us",
    ticker: str = "",
) -> list[float]:
    """Build the 18-element tabular feature vector for CNN inference.

    Matches the exact v6 feature order used by ``predict_breakout()``.
    Extracted from the inline code in ``_handle_check_orb()`` to be
    reusable for all breakout types.

    Returns:
        List of 18 floats in the canonical TABULAR_FEATURES order.
    """
    # Lazy-import CNN helpers — graceful fallback if not available
    _get_btype_ord: Callable[[str], float] = lambda t: 0.0  # noqa: E731
    _get_vol_class: Callable[[str], float] = lambda t: 0.5  # noqa: E731
    _get_asset_cls: Callable[[str], float] = lambda t: 0.0  # noqa: E731
    _session_enc: float = 0.875
    try:
        from lib.analysis.ml.breakout_cnn import (
            get_asset_class_id as _get_asset_cls,
        )
        from lib.analysis.ml.breakout_cnn import (
            get_asset_volatility_class as _get_vol_class,
        )
        from lib.analysis.ml.breakout_cnn import (
            get_breakout_type_ordinal as _get_btype_ord,
        )
        from lib.analysis.ml.breakout_cnn import (
            get_session_ordinal as _get_session_ordinal,
        )

        _session_enc = _get_session_ordinal(session_key)
    except ImportError:
        pass

    # --- [0] quality_pct_norm ---
    _quality_norm = getattr(result, "quality_pct", 0) / 100.0

    # --- [1] volume_ratio ---
    _vol_ratio = 1.0
    _vol_ratio_raw = getattr(result, "volume_ratio", None)
    if _vol_ratio_raw is not None and float(_vol_ratio_raw) > 0:
        _vol_ratio = float(_vol_ratio_raw)

    # --- [2] atr_pct ---
    _atr_pct = 0.0
    if hasattr(result, "atr_value") and result.atr_value > 0:
        _atr_pct = result.atr_value / result.trigger_price

    # --- [3] cvd_delta ---
    _cvd_delta = 0.0
    try:
        if bars_1m is not None and len(bars_1m) > 30:
            _closes = bars_1m["Close"].values.astype(float)
            _opens = bars_1m["Open"].values.astype(float) if "Open" in bars_1m.columns else _closes
            _vols = bars_1m["Volume"].values.astype(float) if "Volume" in bars_1m.columns else None
            if _vols is not None:
                _total_v = float(_vols.sum())
                if _total_v > 0:
                    _buy_v = float(_vols[_closes > _opens].sum())
                    _sell_v = float(_vols[_closes <= _opens].sum())
                    _cvd_delta = (_buy_v - _sell_v) / _total_v
    except Exception:
        pass

    # --- [4] nr7_flag ---
    _nr7_flag = 0.0
    try:
        if bars_daily is not None and len(bars_daily) >= 7:
            _d_highs = bars_daily["High"].values[-7:].astype(float)
            _d_lows = bars_daily["Low"].values[-7:].astype(float)
            _d_ranges = _d_highs - _d_lows
            _nr7_flag = 1.0 if _d_ranges[-1] <= _d_ranges.min() else 0.0
    except Exception:
        pass

    # --- [5] direction_flag ---
    _direction_flag = 1.0 if getattr(result, "direction", "") == "LONG" else 0.0

    # --- [7] london_overlap_flag ---
    _now_hour = 10
    with contextlib.suppress(Exception):
        _now_hour = datetime.now(tz=_EST).hour
    _london_overlap = 1.0 if 8 <= _now_hour <= 9 else 0.0

    # --- [8] or_range_atr_ratio ---
    _or_range = getattr(result, "or_range", 0.0) or getattr(result, "range_size", 0.0)
    _or_range_atr = 0.0
    if result.atr_value > 0 and _or_range > 0:
        _or_range_atr = _or_range / result.atr_value

    # --- [9] premarket_range_ratio ---
    _pm_range_ratio = 0.0
    try:
        _pm_high = getattr(result, "pm_high", None)
        _pm_low = getattr(result, "pm_low", None)
        if _pm_high is not None and _pm_low is not None and _or_range > 0:
            _pm_range = float(_pm_high) - float(_pm_low)
            if _pm_range > 0:
                _pm_range_ratio = _pm_range / _or_range
    except Exception:
        pass

    # --- [10] bar_of_day ---
    _bar_of_day_min = (_now_hour - 18) * 60 if _now_hour >= 18 else (_now_hour + 6) * 60
    _bar_of_day = max(0.0, min(1.0, _bar_of_day_min / 1380.0))

    # --- [11] day_of_week ---
    _dow = 0.5
    try:
        _dow_raw = datetime.now().weekday()
        if 0 <= _dow_raw <= 4:
            _dow = _dow_raw / 4.0
    except Exception:
        pass

    # --- [12] vwap_distance ---
    _vwap_dist = 0.0
    try:
        _range_high = getattr(result, "range_high", 0) or getattr(result, "or_high", 0)
        _range_low = getattr(result, "range_low", 0) or getattr(result, "or_low", 0)
        _vwap = getattr(result, "vwap", None)
        if _vwap is None and _range_high > 0:
            _vwap = (_range_high + _range_low) / 2.0
        if _vwap is not None and result.atr_value > 0:
            _vwap_dist = (result.trigger_price - float(_vwap)) / result.atr_value
    except Exception:
        pass

    # --- [13] asset_class_id ---
    _asset_cls = _get_asset_cls(ticker or getattr(result, "symbol", ""))

    # --- [14] breakout_type_ord ---
    _btype_raw = getattr(result, "breakout_type", "ORB")
    _btype_name = _btype_raw.value if hasattr(_btype_raw, "value") else str(_btype_raw)  # type: ignore[union-attr]
    _btype_ord_val = _get_btype_ord(_btype_name)

    # --- [15] asset_volatility_class ---
    _vol_class_val = _get_vol_class(ticker or getattr(result, "symbol", ""))

    # --- [16] hour_of_day ---
    _hour_of_day = max(0.0, min(1.0, _now_hour / 23.0))

    # --- [17] tp3_atr_mult_norm ---
    _tp3_norm = 0.0
    try:
        from lib.core.breakout_types import (
            BreakoutType as _BT,
        )
        from lib.core.breakout_types import (
            breakout_type_from_name as _bt_from_name,
        )
        from lib.core.breakout_types import (
            get_range_config as _get_rc,
        )

        try:
            _bt_enum = _bt_from_name(str(_btype_name))
        except ValueError:
            _bt_enum = _BT.ORB
        _rc = _get_rc(_bt_enum)
        _tp3_norm = max(0.0, min(1.0, _rc.tp3_atr_mult / 5.0))
    except Exception:
        pass

    return [
        _quality_norm,  # [0]  quality_pct_norm
        _vol_ratio,  # [1]  volume_ratio
        _atr_pct,  # [2]  atr_pct
        _cvd_delta,  # [3]  cvd_delta
        _nr7_flag,  # [4]  nr7_flag
        _direction_flag,  # [5]  direction_flag
        _session_enc,  # [6]  session_ordinal
        _london_overlap,  # [7]  london_overlap_flag
        _or_range_atr,  # [8]  or_range_atr_ratio
        _pm_range_ratio,  # [9]  premarket_range_ratio
        _bar_of_day,  # [10] bar_of_day
        _dow,  # [11] day_of_week
        _vwap_dist,  # [12] vwap_distance
        _asset_cls,  # [13] asset_class_id
        _btype_ord_val,  # [14] breakout_type_ord
        _vol_class_val,  # [15] asset_volatility_class
        _hour_of_day,  # [16] hour_of_day
        _tp3_norm,  # [17] tp3_atr_mult_norm
    ]


def run_cnn_inference(
    result: Any,
    bars_1m: pd.DataFrame | None,
    bars_daily: pd.DataFrame | None,
    session_key: str = "us",
    ticker: str = "",
) -> tuple[float | None, str, bool]:
    """Run CNN inference on a detected breakout and return verdict.

    Returns ``(cnn_prob, cnn_confidence, cnn_signal)``.
    On error or when the CNN module is unavailable, returns
    ``(None, "", True)`` so the breakout passes through.
    """
    cnn_prob: float | None = None
    cnn_confidence: str = ""
    cnn_signal: bool = True  # default: pass through

    try:
        from lib.analysis.ml.breakout_cnn import _find_latest_model, predict_breakout
        from lib.analysis.rendering.chart_renderer import (
            cleanup_inference_images,
            render_snapshot_for_inference,
        )

        _cnn_model = _find_latest_model()
        if not _cnn_model or bars_1m is None:
            return None, "", True

        # Render a chart snapshot for the CNN
        _range_high = getattr(result, "range_high", 0) or getattr(result, "or_high", 0)
        _range_low = getattr(result, "range_low", 0) or getattr(result, "or_low", 0)

        snap_path = render_snapshot_for_inference(
            bars=bars_1m,
            symbol=getattr(result, "symbol", ""),
            orb_high=_range_high,
            orb_low=_range_low,
            direction=result.direction,
            quality_pct=int(getattr(result, "quality_pct", 0)),
        )

        if snap_path:
            tab_features = build_cnn_tabular_features(
                result,
                bars_1m,
                bars_daily,
                session_key,
                ticker,
            )

            cnn_result = predict_breakout(
                image_path=snap_path,
                tabular_features=tab_features,
                model_path=_cnn_model,
            )

            if cnn_result:
                cnn_prob = cnn_result["prob"]
                cnn_confidence = cnn_result["confidence"]
                cnn_signal = cnn_result["signal"]
                logger.info(
                    "🧠 CNN: %s %s P(good)=%.3f (%s) %s",
                    result.direction,
                    getattr(result, "symbol", ""),
                    cnn_prob,
                    cnn_confidence,
                    "SIGNAL" if cnn_signal else "NO SIGNAL",
                )
                # Prometheus
                try:
                    from lib.services.data.api.metrics import (
                        record_orb_cnn_prob,
                        record_orb_cnn_signal,
                    )

                    if cnn_prob is not None:
                        record_orb_cnn_prob(cnn_prob)
                    record_orb_cnn_signal("signal" if cnn_signal else "no_signal")
                except Exception:
                    pass

        # Periodic cleanup
        cleanup_inference_images(max_age_seconds=1800)

    except ImportError:
        logger.debug("CNN module not available — skipping inference")
        try:
            from lib.services.data.api.metrics import record_orb_cnn_signal

            record_orb_cnn_signal("skipped")
        except Exception:
            pass
    except Exception as cnn_exc:
        logger.debug("CNN inference error (non-fatal): %s", cnn_exc)
        try:
            from lib.services.data.api.metrics import record_orb_cnn_signal

            record_orb_cnn_signal("skipped")
        except Exception:
            pass

    return cnn_prob, cnn_confidence, cnn_signal


def _is_cnn_gate_active(session_key: str) -> bool:
    """Check whether the CNN hard gate is active for this session.

    Resolution order:
      1. Redis key ``engine:config:cnn_gate:{session_key}``
      2. ``ORB_CNN_GATE`` env var
      3. Default: disabled
    """
    try:
        from lib.core.redis_helpers import get_cnn_gate

        return get_cnn_gate(session_key)
    except Exception:
        return os.environ.get("ORB_CNN_GATE", "0") == "1"


def _persist_enrichment(row_id: int | None, metadata: dict[str, Any]) -> None:
    """Update an existing orb_events row with filter/CNN enrichment metadata."""
    if row_id is None:
        return
    try:
        from lib.core.models import _get_conn, _is_using_postgres

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        meta_json = json.dumps(metadata, default=str)
        conn = _get_conn()
        conn.execute(
            f"UPDATE orb_events SET metadata_json = {ph} WHERE id = {ph}",
            (meta_json, row_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Failed to enrich event %s (non-fatal): %s", row_id, exc)


def _persist_mtf_columns(
    row_id: int | None,
    breakout_type_name: str,
    mtf_score: float | None,
    macd_slope: float | None,
    divergence: str,
) -> None:
    """Update the dedicated breakout_type/mtf_score/macd_slope/divergence columns."""
    if row_id is None:
        return
    if mtf_score is None and macd_slope is None and not divergence:
        return
    try:
        from lib.core.models import _get_conn, _is_using_postgres

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        conn = _get_conn()
        conn.execute(
            f"UPDATE orb_events SET breakout_type={ph}, mtf_score={ph}, macd_slope={ph}, divergence={ph} WHERE id={ph}",
            (breakout_type_name, mtf_score, macd_slope, divergence, row_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _publish_orb_result(result: Any, session_key: str = "us") -> None:
    """Publish an ORB breakout result via the ORB-specific Redis keys.

    Uses ``publish_orb_alert()`` from ``orb.py`` for the ORB-specific key
    layout (``engine:orb:{session}``, PubSub ``dashboard:orb``).
    Falls back to the generic publisher if the ORB module is not available.
    """
    try:
        from lib.trading.strategies.rb.open import ORB_SESSIONS, publish_orb_alert

        # Build a lightweight ORB-compat shim so publish_orb_alert can consume it
        class _ORBShim:
            """Minimal duck-typed shim that exposes ORBResult attributes from a BreakoutResult."""

            def __init__(self, br: Any, sk: str) -> None:
                self.symbol = br.symbol
                self.session_name = ""
                self.session_key = sk
                self.or_high = getattr(br, "range_high", 0) or getattr(br, "or_high", 0)
                self.or_low = getattr(br, "range_low", 0) or getattr(br, "or_low", 0)
                self.or_range = getattr(br, "range_size", 0) or getattr(br, "or_range", 0)
                self.atr_value = br.atr_value
                self.breakout_threshold = getattr(br, "breakout_threshold", 0)
                self.breakout_detected = br.breakout_detected
                self.direction = br.direction
                self.trigger_price = br.trigger_price
                self.breakout_bar_time = getattr(br, "breakout_bar_time", "")
                self.long_trigger = br.long_trigger
                self.short_trigger = br.short_trigger
                self.or_complete = getattr(br, "range_complete", False) or getattr(br, "or_complete", False)
                self.or_bar_count = getattr(br, "range_bar_count", 0) or getattr(br, "or_bar_count", 0)
                self.evaluated_at = getattr(br, "evaluated_at", "")
                self.error = getattr(br, "error", "")
                self.cnn_prob = getattr(br, "cnn_prob", None)
                self.cnn_confidence = getattr(br, "cnn_confidence", "")
                self.cnn_signal = getattr(br, "cnn_signal", None)
                self.filter_passed = getattr(br, "filter_passed", None)
                self.filter_summary = getattr(br, "filter_summary", "")
                self.depth_ok = getattr(br, "depth_ok", None)
                self.body_ratio_ok = getattr(br, "body_ratio_ok", None)
                self.or_size_ok = getattr(br, "range_size_ok", None)
                self.breakout_bar_depth = getattr(br, "breakout_bar_depth", 0)
                self.breakout_bar_body_ratio = getattr(br, "breakout_bar_body_ratio", 0)

            def to_dict(self) -> dict[str, Any]:
                d = {
                    "type": "ORB",
                    "symbol": self.symbol,
                    "session_name": self.session_name,
                    "session_key": self.session_key,
                    "or_high": round(self.or_high, 4),
                    "or_low": round(self.or_low, 4),
                    "or_range": round(self.or_range, 4),
                    "atr_value": round(self.atr_value, 4),
                    "breakout_threshold": round(self.breakout_threshold, 4),
                    "breakout_detected": self.breakout_detected,
                    "direction": self.direction,
                    "trigger_price": round(self.trigger_price, 4),
                    "breakout_bar_time": self.breakout_bar_time,
                    "long_trigger": round(self.long_trigger, 4),
                    "short_trigger": round(self.short_trigger, 4),
                    "or_complete": self.or_complete,
                    "or_bar_count": self.or_bar_count,
                    "evaluated_at": self.evaluated_at,
                    "error": self.error,
                    "depth_ok": self.depth_ok,
                    "body_ratio_ok": self.body_ratio_ok,
                    "or_size_ok": self.or_size_ok,
                    "breakout_bar_depth": round(self.breakout_bar_depth, 6),
                    "breakout_bar_body_ratio": round(self.breakout_bar_body_ratio, 4),
                }
                if self.cnn_prob is not None:
                    d["cnn_prob"] = round(self.cnn_prob, 4)
                    d["cnn_confidence"] = self.cnn_confidence
                    d["cnn_signal"] = bool(self.cnn_signal) if self.cnn_signal is not None else False
                if self.filter_passed is not None:
                    d["filter_passed"] = bool(self.filter_passed)
                    d["filter_summary"] = self.filter_summary
                return d

        # Find matching ORBSession for session_key
        _orb_session = None
        for s in ORB_SESSIONS:
            if s.key == session_key:
                _orb_session = s
                break

        shim = _ORBShim(result, session_key)
        if _orb_session:
            shim.session_name = _orb_session.name
        publish_orb_alert(shim, session=_orb_session)  # type: ignore[arg-type]  # duck-typed shim

    except ImportError:
        # ORB module not available — fall back to generic publisher
        publish_breakout_result(result, session_key=session_key)
    except Exception as exc:
        logger.debug("_publish_orb_result error (non-fatal): %s", exc)
        # Fall back to generic publisher
        with contextlib.suppress(Exception):
            publish_breakout_result(result, session_key=session_key)


# ===========================================================================
# Main generic handler
# ===========================================================================


def handle_breakout_check(
    engine: Any,
    breakout_type: BreakoutType,
    session_key: str = "us",
    *,
    # Optional overrides for specific types
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: Any | None = None,
    orb_session_end: Any | None = None,
    orb_scan_start: Any | None = None,
    config_override: Any | None = None,
    # --- Phase 1C additions: filter + CNN ---
    enable_filters: bool = False,
    enable_cnn: bool = False,
) -> None:
    """Universal handler for any of the 13 breakout types.

    Replaces ``_handle_check_pdr``, ``_handle_check_ib``,
    ``_handle_check_consolidation``, and (with ``enable_filters``/``enable_cnn``)
    also replaces the ~800-line ``_handle_check_orb()`` in ``main.py``.

    Pipeline:
        1. Get asset list for the session
        2. For each asset:
           a. Fetch 1m bars → detect breakout → MTF enrich
           b. Persist every evaluation (breakout or not) to the audit trail
           c. If breakout detected:
              - (optional) run quality filters → reject if failed
              - (optional) run CNN inference → gate if cnn_gate active
              - publish → dispatch to PositionManager → send alert
        3. Log summary

    All errors are caught per-asset so one failure never blocks the rest.

    Args:
        engine: The engine singleton (for bar fetching fallback).
        breakout_type: Which ``BreakoutType`` to detect.
        session_key: Session key whose asset list to use.
        prev_day_high: Override PDR high (PrevDay type only).
        prev_day_low: Override PDR low (PrevDay type only).
        ib_high: Override IB high (InitialBalance type only).
        ib_low: Override IB low (InitialBalance type only).
        orb_session_start: ORB session start time (ORB type only).
        orb_session_end: ORB session end time (ORB type only).
        orb_scan_start: ORB scan start time (ORB type only).
        config_override: Optional ``RangeConfig`` override.
        enable_filters: If True, run ``apply_all_filters()`` before publishing.
        enable_cnn: If True, run CNN inference and optionally gate by threshold.
    """
    from lib.trading.strategies.rb.breakout import (
        DEFAULT_CONFIGS,
        detect_range_breakout,
    )

    short_name = _get_alert_template(breakout_type)["short"]
    logger.debug("▶ %s breakout check [session=%s]...", short_name, session_key)

    try:
        assets = get_assets_for_session_key(session_key)
        if not assets:
            logger.debug("No assets for %s check (session=%s)", short_name, session_key)
            return

        # Resolve config
        config = config_override or DEFAULT_CONFIGS.get(breakout_type)
        if config is None:
            logger.warning("No config for breakout type %s", breakout_type.name)
            return

        found = 0
        filtered_out = 0

        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "")
            if not symbol:
                continue

            try:
                # 1. Fetch 1-minute bars
                bars_1m = fetch_bars_1m(engine, ticker, symbol)
                if bars_1m is None or bars_1m.empty:
                    continue

                # 2. For PDR: try to get pre-computed daily levels
                _pdh, _pdl = prev_day_high, prev_day_low
                if breakout_type == BreakoutType.PrevDay and _pdh is None:
                    _pdh, _pdl = get_prev_day_levels(ticker, symbol)

                # 3. Detect breakout
                result = detect_range_breakout(
                    bars_1m,
                    symbol=symbol,
                    config=config,
                    prev_day_high=_pdh,
                    prev_day_low=_pdl,
                    ib_high=ib_high,
                    ib_low=ib_low,
                    orb_session_start=orb_session_start,
                    orb_session_end=orb_session_end,
                    orb_scan_start=orb_scan_start,
                )

                # 4. MTF enrichment (always, even if no breakout — captures range levels)
                bars_htf = get_htf_bars(bars_1m, ticker or symbol)
                result = run_mtf_on_result(result, bars_htf)

                # 4b. Capture MTF values for enrichment
                _mtf_score = getattr(result, "mtf_score", None)
                _macd_slope = getattr(result, "macd_slope", None)
                _divergence = getattr(result, "divergence_type", "") or ""

                # 5. Persist every evaluation to the audit trail
                row_id = persist_breakout_result(result, session_key=session_key)

                # 6. On detection: optional filter → optional CNN → publish
                if result.breakout_detected:
                    found += 1

                    # --- Daily bars (needed by filters + CNN) ---
                    bars_daily = _get_daily_bars(ticker, symbol) if (enable_filters or enable_cnn) else None

                    # --- Quality filter gate ---
                    filter_passed = True
                    filter_summary = ""

                    if enable_filters:
                        filter_passed, filter_summary = run_quality_filters(
                            result,
                            bars_1m,
                            bars_daily,
                            bars_htf,
                            session_key,
                        )

                        if not filter_passed:
                            filtered_out += 1
                            logger.info(
                                "🚫 %s FILTERED: %s %s @ %.4f — %s",
                                short_name,
                                result.direction,
                                symbol,
                                result.trigger_price,
                                filter_summary,
                            )
                            _persist_enrichment(
                                row_id,
                                {
                                    "orb_session": session_key,
                                    "filter_passed": False,
                                    "filter_summary": filter_summary,
                                    "published": False,
                                    "mtf_score": _mtf_score,
                                    "macd_slope": _macd_slope,
                                    "divergence": _divergence,
                                },
                            )
                            _persist_mtf_columns(
                                row_id,
                                breakout_type.name,
                                _mtf_score,
                                _macd_slope,
                                _divergence,
                            )
                            # Prometheus
                            try:
                                from lib.services.data.api.metrics import record_orb_filter_result

                                record_orb_filter_result("rejected")
                            except Exception:
                                pass
                            continue  # next asset

                        logger.info(
                            "✅ %s PASSED filters: %s %s — %s",
                            short_name,
                            result.direction,
                            symbol,
                            filter_summary,
                        )
                        try:
                            from lib.services.data.api.metrics import record_orb_filter_result

                            record_orb_filter_result("passed")
                        except Exception:
                            pass

                    # --- CNN inference gate ---
                    cnn_prob: float | None = None
                    cnn_confidence: str = ""
                    cnn_signal: bool = True

                    if enable_cnn:
                        cnn_prob, cnn_confidence, cnn_signal = run_cnn_inference(
                            result,
                            bars_1m,
                            bars_daily,
                            session_key,
                            ticker,
                        )

                        # Attach CNN results to the result object
                        with contextlib.suppress(AttributeError):
                            result.cnn_prob = cnn_prob
                            result.cnn_confidence = cnn_confidence
                            result.cnn_signal = cnn_signal

                        # CNN hard gate (per-session or global)
                        if _is_cnn_gate_active(session_key) and not cnn_signal:
                            filtered_out += 1
                            logger.info(
                                "🚫 %s CNN-GATED [%s]: %s %s — P(good)=%.3f < threshold",
                                short_name,
                                session_key,
                                result.direction,
                                symbol,
                                cnn_prob or 0.0,
                            )
                            _persist_enrichment(
                                row_id,
                                {
                                    "orb_session": session_key,
                                    "filter_passed": True,
                                    "filter_summary": filter_summary,
                                    "cnn_prob": cnn_prob,
                                    "cnn_confidence": cnn_confidence,
                                    "cnn_signal": cnn_signal,
                                    "cnn_gated": True,
                                    "published": False,
                                },
                            )
                            continue  # next asset

                    # --- Publish ---
                    # For ORB type, use ORB-specific Redis keys for backward compat
                    if breakout_type == BreakoutType.ORB:
                        _publish_orb_result(result, session_key=session_key)
                    else:
                        publish_breakout_result(result, session_key=session_key)

                    # Forward to PositionManager
                    dispatch_to_position_manager(result, bars_1m=bars_1m, session_key=session_key)

                    # Build enriched log message
                    cnn_line = ""
                    if cnn_prob is not None:
                        cnn_line = f" | CNN P(good)={cnn_prob:.3f} ({cnn_confidence})"

                    logger.info(
                        "🔔 %s BREAKOUT: %s %s @ %.4f (range %.4f–%.4f)%s%s",
                        short_name,
                        result.direction,
                        symbol,
                        result.trigger_price,
                        result.range_low,
                        result.range_high,
                        f" [{filter_summary}]" if filter_summary else "",
                        cnn_line,
                    )

                    # Enrich audit row — published
                    _persist_enrichment(
                        row_id,
                        {
                            "orb_session": session_key,
                            "filter_passed": True,
                            "filter_summary": filter_summary,
                            "cnn_prob": cnn_prob,
                            "cnn_confidence": cnn_confidence,
                            "cnn_signal": cnn_signal,
                            "cnn_gated": False,
                            "published": True,
                            "mtf_score": _mtf_score,
                            "macd_slope": _macd_slope,
                            "divergence": _divergence,
                        },
                    )
                    _persist_mtf_columns(
                        row_id,
                        breakout_type.name,
                        _mtf_score,
                        _macd_slope,
                        _divergence,
                    )

                    # Send push alert
                    send_breakout_alert(result, breakout_type, session_key)

            except Exception as exc:
                logger.debug("%s check failed for %s: %s", short_name, symbol, exc)

        if found:
            logger.info(
                "✅ %s [%s] check complete: %d breakout(s) found, %d filtered out, %d published",
                short_name,
                session_key,
                found,
                filtered_out,
                found - filtered_out,
            )
        else:
            logger.debug("%s check [%s] complete: no breakouts", short_name, session_key)

    except Exception as exc:
        logger.debug("%s handler error (non-fatal): %s", short_name, exc)


def handle_breakout_multi(
    engine: Any,
    session_key: str = "us",
    types: list[BreakoutType] | None = None,
) -> None:
    """Run multiple breakout type detectors for a session's assets.

    Dispatches each type sequentially via ``handle_breakout_check()``.
    For parallel execution, wrap in a ThreadPoolExecutor at the call site.

    Args:
        engine: The engine singleton.
        session_key: Session key whose asset list to use.
        types: List of ``BreakoutType`` to check.
               Defaults to ``[PrevDay, Consolidation]``.
    """
    if types is None:
        types = [BreakoutType.PrevDay, BreakoutType.Consolidation]

    logger.debug(
        "▶ Multi-type breakout sweep [session=%s types=%s]...",
        session_key,
        [bt.name for bt in types],
    )

    import concurrent.futures

    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(types), 4),
        thread_name_prefix="breakout",
    ) as executor:
        for btype in types:
            fut = executor.submit(
                handle_breakout_check,
                engine,
                btype,
                session_key=session_key,
            )
            futures_map[fut] = btype

        for future in concurrent.futures.as_completed(futures_map, timeout=60):
            btype = futures_map[future]
            try:
                future.result()
            except Exception as exc:
                logger.warning(
                    "Multi-type sweep [%s/%s] error: %s",
                    session_key,
                    btype.name,
                    exc,
                )

    logger.debug("Multi-type breakout sweep [%s] complete", session_key)


# ===========================================================================
# ORB convenience wrapper — Phase 1C
# ===========================================================================


def handle_orb_check(
    engine: Any,
    session_key: str = "us",
    *,
    orb_session: Any | None = None,
) -> None:
    """Check for Opening Range Breakouts using the unified pipeline.

    This is the Phase 1C replacement for the ~800-line ``_handle_check_orb()``
    in ``main.py``.  It translates the ORB session into the correct config
    and delegates to ``handle_breakout_check()`` with filters and CNN enabled.

    If an ``orb_session`` (``ORBSession`` from ``orb.py``) is provided, its
    ``key`` is used as the ``session_key`` and its time windows are passed
    through as ``orb_session_start``/``orb_session_end``/``orb_scan_start``.

    Args:
        engine: The engine singleton.
        session_key: Session key (e.g. ``"us"``, ``"london"``).
                     Overridden by ``orb_session.key`` if provided.
        orb_session: Optional ``ORBSession`` instance from ``orb.py``.
    """
    _session_start = None
    _session_end = None
    _scan_start = None

    if orb_session is not None:
        session_key = getattr(orb_session, "key", session_key)
        _session_start = getattr(orb_session, "or_start", None)
        _session_end = getattr(orb_session, "or_end", None)
        _scan_start = getattr(orb_session, "scan_end", None)

    logger.debug("▶ Opening Range Breakout check [%s] (unified pipeline)...", session_key)

    handle_breakout_check(
        engine,
        BreakoutType.ORB,
        session_key=session_key,
        orb_session_start=_session_start,
        orb_session_end=_session_end,
        orb_scan_start=_scan_start,
        enable_filters=True,
        enable_cnn=True,
    )
