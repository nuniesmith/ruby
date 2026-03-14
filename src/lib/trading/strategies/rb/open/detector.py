"""
Opening Range Breakout — Detection Engine
==========================================
Pure detection logic for all ORB variants.  No Redis, no publishing — only
bar math and quality gates.

Public API
----------
compute_atr(highs, lows, closes, period)
    Wilder ATR on raw numpy arrays.

compute_opening_range(bars_1m, session)
    Extract (or_high, or_low, bar_count, is_complete) from 1-minute bars.

detect_opening_range_breakout(bars_1m, symbol, session, ...)
    Full single-session ORB detector with all quality gates.

detect_all_sessions(bars_1m, symbol, sessions, now_fn)
    Run detection across every session for one symbol.

scan_orb_all_assets(bars_by_symbol, session, ...)
    Single-session scan across multiple symbols.

scan_orb_all_sessions_all_assets(bars_by_symbol, sessions)
    Full cross-product scan.

Private helpers
---------------
_localize_index(df)          — ensure DatetimeIndex is ET-aware
_check_or_size(...)          — OR-size cap/floor gate
_check_breakout_bar_quality(...) — depth + body-ratio gate
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.trading.strategies.rb.open.assets import get_symbol_session_overrides
from lib.trading.strategies.rb.open.models import MultiSessionORBResult, ORBResult
from lib.trading.strategies.rb.open.sessions import (
    ATR_PERIOD,
    ORB_SESSIONS,
    US_SESSION,
    ORBSession,
)
from lib.trading.strategies.rb.range_builders import compute_atr as _rb_compute_atr

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("engine.orb")

_EST = ZoneInfo("America/New_York")


# ===========================================================================
# ATR
# ===========================================================================


def compute_atr(
    highs: np.ndarray | Any,
    lows: np.ndarray | Any,
    closes: np.ndarray | Any,
    period: int = ATR_PERIOD,
) -> float:
    """Compute the Average True Range (Wilder method) from raw bar arrays.

    This is a thin wrapper around the **canonical** implementation in
    :func:`lib.trading.strategies.rb.range_builders.compute_atr`.  All ATR
    calculations in the RB system are routed through that single function to
    guarantee identical values regardless of call site.

    The canonical implementation uses Wilder's EMA (alpha = 1/period)::

        TR_0  = High_0 − Low_0
        TR_i  = max(High_i − Low_i,
                    |High_i − Close_{i-1}|,
                    |Low_i  − Close_{i-1}|)
        ATR   = SMA(TR[:period])                          # seed
        ATR_i = (1/period) * TR_i + (1 − 1/period) * ATR_{i-1}

    Falls back to a simple H-L mean when fewer than ``period + 1`` bars are
    available.  Returns ``0.0`` for fewer than 2 bars.

    Args:
        highs:  1-D array of bar high prices.
        lows:   1-D array of bar low prices.
        closes: 1-D array of bar close prices.
        period: ATR look-back window (default 14).

    Returns:
        ATR as a float, or ``0.0`` on insufficient data.
    """
    # range_builders.compute_atr expects a DataFrame with High/Low/Close columns.
    # Build a minimal one from the raw arrays so we can delegate cleanly.
    df = pd.DataFrame(
        {
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": np.asarray(closes, dtype=float),
        }
    )
    return _rb_compute_atr(df, period=period)


# ===========================================================================
# Index localisation
# ===========================================================================


def _localize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure *df* has a tz-aware DatetimeIndex in Eastern Time.

    Handles three cases:

    * Already localised to US/Eastern or America/New_York → no-op.
    * Localised to another tz → ``tz_convert`` to ET.
    * Naive (no tz) → ``tz_localize`` as ET.

    The original DataFrame is not mutated — a copy is returned when
    timezone conversion is needed.
    """
    idx: Any = df.index
    tz = getattr(idx, "tz", None)
    if tz is not None:
        return df.tz_convert(_EST) if str(tz) != str(_EST) else df
    # Naive index — assume ET
    with contextlib.suppress(Exception):
        df = df.copy()
        df.index = idx.tz_localize(_EST)
    return df


# ===========================================================================
# Opening range computation
# ===========================================================================


def compute_opening_range(
    bars_1m: pd.DataFrame | None,
    session: ORBSession | None = None,
) -> tuple[float, float, int, bool]:
    """Extract the opening range (high, low) from 1-minute bars.

    Handles ``wraps_midnight`` sessions (e.g. CME Open, Sydney, Tokyo):
    those sessions start in the *previous* ET calendar day, so bars are
    filtered purely by wall-clock time across any date.  The most recent
    contiguous block of OR bars is used.

    Args:
        bars_1m: DataFrame with a DatetimeIndex and at least ``High``,
                 ``Low``, ``Close`` columns.  For overnight sessions pass
                 at least 2 calendar days of 1-minute bars.
        session: ORB session definition.  Defaults to :data:`US_SESSION`.

    Returns:
        ``(or_high, or_low, bar_count, is_complete)`` where *is_complete*
        is ``True`` once at least one bar *past* ``session.or_end`` is
        present in the data.

    .. note:: ``complete`` semantics vs ``build_orb_range``

        This function uses a **data-presence** check: ``is_complete`` is
        only ``True`` when a bar timestamped after ``or_end`` actually
        exists in *bars_1m*.  This is intentionally stricter than
        :func:`~lib.trading.strategies.rb.range_builders.build_orb_range`,
        which sets ``complete`` as soon as the last bar's wall-clock time
        reaches ``session_end`` (a pure time comparison).

        The stricter check is safer for live feeds where the most recent
        bar may land exactly on the ``or_end`` boundary tick.  Use
        ``build_orb_range`` for back-testing pipelines where the bar
        window is guaranteed to be fully formed.
    """
    if session is None:
        session = US_SESSION

    if bars_1m is None or bars_1m.empty:
        return 0.0, 0.0, 0, False

    df = _localize_index(bars_1m.copy())
    idx: Any = df.index
    times = idx.time

    if session.wraps_midnight:
        # Overnight session: OR window lives in the previous calendar day.
        # Filter by wall-clock time only, then take the most recent block.
        or_mask = (times >= session.or_start) & (times < session.or_end)
        or_bars_all: pd.DataFrame = df.loc[or_mask]
        if or_bars_all.empty:
            return 0.0, 0.0, 0, False

        last_or_date = pd.Timestamp(or_bars_all.index[-1]).date()  # type: ignore[arg-type]
        or_bars: pd.DataFrame = or_bars_all[pd.DatetimeIndex(or_bars_all.index).date == last_or_date]  # type: ignore[attr-defined]

        # is_complete: any bar past or_end on that date, or any bar on the
        # *next* calendar day (i.e. "today" if the OR was yesterday).
        next_date = last_or_date + _dt.timedelta(days=1)
        _dates = pd.DatetimeIndex(df.index).date  # type: ignore[attr-defined]
        post_or = df[((_dates == last_or_date) & (times >= session.or_end)) | (_dates == next_date)]
        is_complete = not post_or.empty
    else:
        # Normal intraday session
        or_mask = (times >= session.or_start) & (times < session.or_end)
        or_bars = df.loc[or_mask]
        if or_bars.empty:
            return 0.0, 0.0, 0, False
        is_complete = bool(np.any(times >= session.or_end))

    if or_bars.empty:
        return 0.0, 0.0, 0, False

    or_high = float(or_bars["High"].max())  # type: ignore[arg-type]
    or_low = float(or_bars["Low"].min())  # type: ignore[arg-type]
    bar_count: int = len(or_bars)
    return or_high, or_low, bar_count, is_complete


# ===========================================================================
# Quality gates
# ===========================================================================


def _check_or_size(
    or_range: float,
    atr: float,
    session: ORBSession,
) -> tuple[bool, str]:
    """Check that the opening range is neither too wide nor too narrow.

    Args:
        or_range: Height of the OR (``or_high − or_low``).
        atr:      Current ATR value.
        session:  Session whose cap/floor thresholds to apply.

    Returns:
        ``(passed, reason)`` — *reason* is an empty string when *passed*
        is ``True``.
    """
    if atr <= 0:
        return True, ""  # cannot evaluate without ATR; defer to caller

    ratio = or_range / atr

    if session.max_or_atr_ratio > 0 and ratio > session.max_or_atr_ratio:
        return False, (f"OR too wide: {or_range:.4f} = {ratio:.2f}× ATR (cap {session.max_or_atr_ratio:.2f}×)")
    if session.min_or_atr_ratio > 0 and ratio < session.min_or_atr_ratio:
        return False, (f"OR too narrow: {or_range:.4f} = {ratio:.2f}× ATR (floor {session.min_or_atr_ratio:.2f}×)")
    return True, ""


def _check_breakout_bar_quality(
    row: pd.Series,
    direction: str,
    or_high: float,
    or_low: float,
    atr: float,
    session: ORBSession,
) -> tuple[bool, bool, float, float]:
    """Evaluate depth and body-ratio quality of a candidate breakout bar.

    * **Depth** — the close must penetrate beyond the OR level by at least
      ``session.min_depth_atr_pct × ATR``.
    * **Body ratio** — ``|close − open| / (high − low)`` must reach
      ``session.min_body_ratio``, filtering out doji / shooting-star bars.

    Args:
        row:       The breakout bar Series (must have Open, High, Low, Close).
        direction: ``"LONG"`` or ``"SHORT"``.
        or_high:   Opening range high.
        or_low:    Opening range low.
        atr:       Current ATR value.
        session:   Session whose quality thresholds to apply.

    Returns:
        ``(depth_ok, body_ratio_ok, depth_value, body_ratio_value)``

        * *depth_value*      — absolute penetration beyond the OR level (≥ 0).
        * *body_ratio_value* — clamped to ``[0.0, 1.0]``.
    """
    bar_open: float = float(row.get("Open") if row.get("Open") is not None else row.get("open") or 0.0)  # type: ignore[arg-type]
    bar_high: float = float(row.get("High") if row.get("High") is not None else row.get("high") or 0.0)  # type: ignore[arg-type]
    bar_low: float = float(row.get("Low") if row.get("Low") is not None else row.get("low") or 0.0)  # type: ignore[arg-type]
    bar_close: float = float(row.get("Close") if row.get("Close") is not None else row.get("close") or 0.0)  # type: ignore[arg-type]

    # Depth: distance the close moved *past* the OR boundary
    depth = max(bar_close - or_high, 0.0) if direction == "LONG" else max(or_low - bar_close, 0.0)

    # Body ratio: fraction of bar range covered by the candle body
    bar_range = bar_high - bar_low
    body_ratio = min(abs(bar_close - bar_open) / bar_range, 1.0) if bar_range > 0 else 0.0

    min_depth = atr * session.min_depth_atr_pct if atr > 0 else 0.0
    depth_ok = (session.min_depth_atr_pct <= 0) or (depth >= min_depth)
    body_ok = (session.min_body_ratio <= 0) or (body_ratio >= session.min_body_ratio)

    return depth_ok, body_ok, depth, body_ratio


# ===========================================================================
# Core single-session detector
# ===========================================================================


def detect_opening_range_breakout(
    bars_1m: pd.DataFrame | None,
    symbol: str = "",
    session: ORBSession | None = None,
    atr_period: int | None = None,
    breakout_multiplier: float | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> ORBResult:
    """Detect an Opening Range Breakout from 1-minute bar data.

    Algorithm
    ~~~~~~~~~
    1. Compute the OR (high/low) from bars within the session window.
    2. Compute ATR from all available bars.
    3. Apply OR-size quality gate (too wide or too narrow → return early).
    4. Compute breakout trigger levels::

           long_trigger  = or_high + ATR × multiplier
           short_trigger = or_low  − ATR × multiplier

    5. Scan bars between ``or_end`` and ``scan_end`` for a close beyond
       either trigger.
    6. For each candidate bar apply depth and body-ratio quality gates.
       On failure continue scanning for the next candidate.
    7. Return the first bar passing all gates, or a no-breakout result.

    Per-symbol overrides (crypto assets) are applied before evaluation:
    caller-supplied *breakout_multiplier* takes highest precedence, then
    symbol-level overrides, then session defaults.

    Args:
        bars_1m:              1-minute OHLCV DataFrame with DatetimeIndex.
                              Pass at least 2 days for overnight sessions.
        symbol:               Instrument ticker for logging/result labelling.
        session:              ORB session definition.  Defaults to
                              :data:`~sessions.US_SESSION`.
        atr_period:           Override session ATR look-back period.
        breakout_multiplier:  Override ATR multiplier for trigger levels.
        now_fn:               Optional clock callable for unit-test
                              determinism (returns a tz-aware datetime).

    Returns:
        :class:`~models.ORBResult` with all fields populated.
        ``breakout_detected=True`` only when every quality gate passes.
    """
    if session is None:
        session = US_SESSION

    _atr_period = atr_period if atr_period is not None else session.atr_period

    # --- Per-symbol overrides (crypto needs wider thresholds) ---
    _sym_overrides = get_symbol_session_overrides(symbol, session)
    _breakout_mult = (
        breakout_multiplier
        if breakout_multiplier is not None
        else _sym_overrides.get("breakout_multiplier", session.breakout_multiplier)
    )

    # Merge symbol-level OR-size / quality-gate overrides into a patched
    # session object so _check_or_size() and _check_breakout_bar_quality()
    # automatically see the right thresholds.
    if _sym_overrides:
        session = dataclasses.replace(
            session,
            max_or_atr_ratio=_sym_overrides.get("max_or_atr_ratio", session.max_or_atr_ratio),
            min_or_atr_ratio=_sym_overrides.get("min_or_atr_ratio", session.min_or_atr_ratio),
            min_depth_atr_pct=_sym_overrides.get("min_depth_atr_pct", session.min_depth_atr_pct),
            min_body_ratio=_sym_overrides.get("min_body_ratio", session.min_body_ratio),
        )

    now = (now_fn or (lambda: datetime.now(tz=_EST)))()
    result = ORBResult(
        symbol=symbol,
        session_name=session.name,
        session_key=session.key,
        evaluated_at=now.isoformat(),
    )

    # --- Validate input ---
    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data provided"
        return result

    missing_cols = {"High", "Low", "Close"} - set(bars_1m.columns)
    if missing_cols:
        result.error = f"Missing columns: {missing_cols}"
        return result

    if len(bars_1m) < session.min_bars:
        result.error = f"Insufficient bars ({len(bars_1m)} < {session.min_bars})"
        return result

    # --- Opening range ---
    or_high, or_low, or_bar_count, or_complete = compute_opening_range(bars_1m, session=session)

    result.or_high = or_high
    result.or_low = or_low
    result.or_range = or_high - or_low if or_high > or_low else 0.0
    result.or_bar_count = or_bar_count
    result.or_complete = or_complete

    if or_bar_count < session.min_bars:
        result.error = f"{session.name} opening range has only {or_bar_count} bars (need >= {session.min_bars})"
        return result

    if or_high <= 0 or or_low <= 0 or or_high <= or_low:
        result.error = f"Invalid opening range: high={or_high}, low={or_low}"
        return result

    # --- ATR ---
    highs = bars_1m["High"].values.astype(float)
    lows = bars_1m["Low"].values.astype(float)
    closes = bars_1m["Close"].values.astype(float)

    atr = compute_atr(highs, lows, closes, period=_atr_period)
    result.atr_value = atr

    if atr <= 0:
        result.error = "ATR is zero — cannot compute breakout thresholds"
        return result

    # --- Trigger levels (always populated so callers can read them) ---
    threshold = atr * _breakout_mult
    result.breakout_threshold = threshold
    result.long_trigger = or_high + threshold
    result.short_trigger = or_low - threshold

    # --- OR-size quality gate ---
    or_size_ok, or_size_reason = _check_or_size(result.or_range, atr, session)
    result.or_size_ok = or_size_ok
    if not or_size_ok:
        result.error = f"OR-size gate: {or_size_reason}"
        logger.debug("ORB [%s] %s skipped — %s", session.name, symbol, or_size_reason)
        return result

    # --- Wait for OR to close before scanning ---
    if not or_complete:
        return result

    # --- Locate post-OR bars ---
    df = _localize_index(bars_1m.copy())
    _idx: Any = df.index
    times = _idx.time

    if session.wraps_midnight:
        # Post-OR bars may fall on the *next* calendar day
        _or_mask = (times >= session.or_start) & (times < session.or_end)
        _or_dates = df.loc[_or_mask].index
        if _or_dates.empty:
            return result
        _last_or_date = pd.Timestamp(_or_dates[-1]).date()  # type: ignore[arg-type]
        _next_date = _last_or_date + _dt.timedelta(days=1)

        _df_dates = pd.DatetimeIndex(df.index).date  # type: ignore[attr-defined]
        _post_or_mask = ((_df_dates == _last_or_date) & (times >= session.or_end) & (times <= session.scan_end)) | (
            (_df_dates == _next_date) & (times <= session.scan_end)
        )
        post_or_bars: pd.DataFrame = df.loc[_post_or_mask]
    else:
        post_or_mask = (times >= session.or_end) & (times <= session.scan_end)
        post_or_bars = df.loc[post_or_mask]

    if post_or_bars.empty:
        return result

    # --- Scan for first qualifying breakout ---
    # Track the best (deepest penetration) rejected candidate so that when
    # no bar passes all gates the stored quality metrics reflect the most
    # significant attempt rather than the first chronological one.
    _best_candidate_depth: float = -1.0
    _best_depth_ok: bool | None = None
    _best_body_ok: bool | None = None
    _best_depth_val: float = 0.0
    _best_body_val: float = 0.0

    for idx_label, row in post_or_bars.iterrows():
        close: float = float(row["Close"])  # type: ignore[arg-type]

        candidate_direction: str = ""
        if close > result.long_trigger:
            candidate_direction = "LONG"
        elif close < result.short_trigger:
            candidate_direction = "SHORT"
        else:
            continue

        depth_ok, body_ok, depth_val, body_val = _check_breakout_bar_quality(
            row, candidate_direction, or_high, or_low, atr, session
        )

        # Keep track of the deepest penetrating candidate seen so far so
        # that if no bar ultimately passes all gates, the stored metrics
        # describe the closest attempt (most useful for post-hoc review).
        if depth_val > _best_candidate_depth:
            _best_candidate_depth = depth_val
            _best_depth_ok = depth_ok
            _best_body_ok = body_ok
            _best_depth_val = depth_val
            _best_body_val = body_val

        if not depth_ok:
            logger.debug(
                "ORB [%s] %s %s candidate rejected: depth %.4f < min %.4f (%.2f× ATR)",
                session.name,
                candidate_direction,
                symbol,
                depth_val,
                atr * session.min_depth_atr_pct,
                session.min_depth_atr_pct,
            )
            continue

        if not body_ok:
            logger.debug(
                "ORB [%s] %s %s candidate rejected: body ratio %.2f < min %.2f",
                session.name,
                candidate_direction,
                symbol,
                body_val,
                session.min_body_ratio,
            )
            continue

        # All gates passed — confirmed breakout
        result.breakout_detected = True
        result.direction = candidate_direction
        result.trigger_price = close
        result.breakout_bar_time = str(idx_label)
        result.depth_ok = depth_ok
        result.body_ratio_ok = body_ok
        result.breakout_bar_depth = depth_val
        result.breakout_bar_body_ratio = body_val
        break

    # If no confirmed breakout, populate quality metrics from the best
    # (deepest) candidate that was evaluated so callers can inspect why
    # detection failed.  Only written when at least one candidate was seen.
    if not result.breakout_detected and _best_depth_ok is not None:
        result.depth_ok = _best_depth_ok
        result.body_ratio_ok = _best_body_ok
        result.breakout_bar_depth = _best_depth_val
        result.breakout_bar_body_ratio = _best_body_val

    if result.breakout_detected:
        logger.info(
            "ORB [%s] detected: %s %s @ %.4f (OR %.4f–%.4f, ATR %.4f, threshold %.4f, depth %.4f, body_ratio %.2f)",
            session.name,
            result.direction,
            symbol,
            result.trigger_price,
            result.or_low,
            result.or_high,
            atr,
            threshold,
            result.breakout_bar_depth,
            result.breakout_bar_body_ratio,
        )

    return result


# ===========================================================================
# Multi-session and multi-asset convenience functions
# ===========================================================================


def detect_all_sessions(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    sessions: list[ORBSession] | None = None,
    now_fn: Callable[..., Any] | None = None,
) -> MultiSessionORBResult:
    """Run ORB detection across all sessions for a single symbol.

    Args:
        bars_1m:  1-minute OHLCV DataFrame with DatetimeIndex.
        symbol:   Instrument ticker.
        sessions: Sessions to evaluate.  Defaults to :data:`~sessions.ORB_SESSIONS`.
        now_fn:   Optional clock callable for deterministic tests.

    Returns:
        :class:`~models.MultiSessionORBResult` with a result per session.
    """
    if sessions is None:
        sessions = ORB_SESSIONS

    now = (now_fn or (lambda: datetime.now(tz=_EST)))()
    multi = MultiSessionORBResult(
        symbol=symbol,
        evaluated_at=now.isoformat(),
    )

    for session in sessions:
        try:
            result = detect_opening_range_breakout(
                bars_1m,
                symbol=symbol,
                session=session,
                now_fn=now_fn,
            )
            multi.sessions[session.key] = result
        except Exception as exc:
            logger.error("ORB [%s] detection failed for %s: %s", session.name, symbol, exc)
            multi.sessions[session.key] = ORBResult(
                symbol=symbol,
                session_name=session.name,
                session_key=session.key,
                error=str(exc),
                evaluated_at=now.isoformat(),
            )

    return multi


def scan_orb_all_assets(
    bars_by_symbol: dict[str, pd.DataFrame],
    session: ORBSession | None = None,
    atr_period: int | None = None,
    breakout_multiplier: float | None = None,
) -> list[ORBResult]:
    """Run ORB detection across multiple symbols for a single session.

    Args:
        bars_by_symbol:       Dict mapping symbol → 1-minute DataFrame.
        session:              ORB session.  Defaults to :data:`~sessions.US_SESSION`.
        atr_period:           Override ATR period.
        breakout_multiplier:  Override breakout multiplier.

    Returns:
        List of :class:`~models.ORBResult` for each symbol (including
        non-breakout results).
    """
    results: list[ORBResult] = []
    for symbol, bars in bars_by_symbol.items():
        try:
            result = detect_opening_range_breakout(
                bars,
                symbol=symbol,
                session=session,
                atr_period=atr_period,
                breakout_multiplier=breakout_multiplier,
            )
            results.append(result)
        except Exception as exc:
            logger.error("ORB scan failed for %s: %s", symbol, exc)
            results.append(ORBResult(symbol=symbol, error=str(exc)))
    return results


def scan_orb_all_sessions_all_assets(
    bars_by_symbol: dict[str, pd.DataFrame],
    sessions: list[ORBSession] | None = None,
) -> dict[str, MultiSessionORBResult]:
    """Run ORB detection across multiple symbols and all sessions.

    Args:
        bars_by_symbol: Dict mapping symbol → 1-minute DataFrame.
        sessions:       Sessions to evaluate.  Defaults to
                        :data:`~sessions.ORB_SESSIONS`.

    Returns:
        Dict mapping symbol → :class:`~models.MultiSessionORBResult`.
    """
    results: dict[str, MultiSessionORBResult] = {}
    for symbol, bars in bars_by_symbol.items():
        try:
            results[symbol] = detect_all_sessions(bars, symbol=symbol, sessions=sessions)
        except Exception as exc:
            logger.error("Multi-session ORB scan failed for %s: %s", symbol, exc)
    return results
