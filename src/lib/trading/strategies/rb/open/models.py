"""
Opening Range Breakout — Result Dataclasses
============================================
Canonical home for the two result types produced by the ORB detector.

``ORBResult``
    Single-session evaluation result for one symbol.  Carries the opening
    range levels, breakout detection outcome, trigger price, quality-gate
    verdicts, and optional CNN / filter enrichment fields.

``MultiSessionORBResult``
    Aggregated results across all sessions for a single symbol.  Provides
    convenience properties to query the aggregate state without iterating
    over the sessions dict manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ORBResult:
    """Result of an Opening Range Breakout evaluation for a specific session.

    Lifecycle
    ~~~~~~~~~
    1. Detector populates core fields (``or_high``, ``or_low``, ``or_range``,
       ``atr_value``, ``breakout_threshold``, ``long_trigger``,
       ``short_trigger``, quality-gate booleans).
    2. Engine main optionally enriches with CNN inference fields
       (``cnn_prob``, ``cnn_confidence``, ``cnn_signal``).
    3. Engine main optionally enriches with ORB-filter verdict fields
       (``filter_passed``, ``filter_summary``).
    4. :func:`~publisher.publish_orb_alert` serialises via :meth:`to_dict`
       and writes to Redis / pub-sub for dashboard consumption.

    Quality gate fields
    ~~~~~~~~~~~~~~~~~~~
    All three gate fields are ``None`` until the detector evaluates them,
    making it possible to distinguish "not yet evaluated" from ``False``.

    * ``depth_ok``       — breakout bar closed far enough past the OR level.
    * ``body_ratio_ok``  — breakout candle body ≥ ``min_body_ratio`` of range.
    * ``or_size_ok``     — opening range is neither too wide nor too narrow.
    """

    # --- Identity ---
    symbol: str = ""
    session_name: str = ""  # e.g. "London Open", "US Equity Open"
    session_key: str = ""  # e.g. "london", "us"

    # --- Opening range levels ---
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr_value: float = 0.0
    breakout_threshold: float = 0.0

    # --- Breakout detection ---
    breakout_detected: bool = False
    direction: str = ""  # "LONG", "SHORT", or ""
    trigger_price: float = 0.0
    breakout_bar_time: str = ""

    # --- Trigger levels (always populated once ATR is known) ---
    long_trigger: float = 0.0
    short_trigger: float = 0.0

    # --- OR status ---
    or_complete: bool = False  # True once OR window has closed
    or_bar_count: int = 0
    evaluated_at: str = ""
    error: str = ""

    # --- CNN enrichment (populated by engine after inference) ---
    cnn_prob: float | None = None
    cnn_confidence: str = ""
    cnn_signal: bool | None = None

    # --- Filter enrichment (populated by engine after filter pass) ---
    filter_passed: bool | None = None
    filter_summary: str = ""

    # --- Quality gate verdicts (populated by detector) ---
    depth_ok: bool | None = None  # True if depth filter passed
    body_ratio_ok: bool | None = None  # True if body-ratio filter passed
    or_size_ok: bool | None = None  # True if OR-size cap/floor passed
    breakout_bar_depth: float = 0.0  # Actual penetration beyond OR level
    breakout_bar_body_ratio: float = 0.0  # Actual body/range ratio of breakout bar

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this result.

        Optional fields (CNN, filter) are only included when they have been
        populated, keeping the payload lean for dashboard consumption.
        """
        d: dict[str, Any] = {
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
            # Quality gate verdicts
            "depth_ok": self.depth_ok,
            "body_ratio_ok": self.body_ratio_ok,
            "or_size_ok": self.or_size_ok,
            "breakout_bar_depth": round(self.breakout_bar_depth, 6),
            "breakout_bar_body_ratio": round(self.breakout_bar_body_ratio, 4),
        }

        # CNN fields — only when inference has run
        if self.cnn_prob is not None:
            d["cnn_prob"] = round(self.cnn_prob, 4)
            d["cnn_confidence"] = self.cnn_confidence
            d["cnn_signal"] = bool(self.cnn_signal) if self.cnn_signal is not None else False

        # Filter fields — only when the filter pass has run
        if self.filter_passed is not None:
            d["filter_passed"] = bool(self.filter_passed)
            d["filter_summary"] = self.filter_summary

        return d


@dataclass
class MultiSessionORBResult:
    """Aggregated ORB results across all sessions for a single symbol.

    The ``sessions`` dict maps session key → :class:`ORBResult`.  The
    convenience properties let callers query the aggregate state without
    manually iterating over the dict.

    Example::

        multi = detect_all_sessions(bars_1m, symbol="MGC=F")
        if multi.has_any_breakout:
            best = multi.best_breakout
            print(best.direction, best.session_name)
    """

    symbol: str = ""
    sessions: dict[str, ORBResult] = field(default_factory=dict)
    evaluated_at: str = ""

    @property
    def has_any_breakout(self) -> bool:
        """``True`` if at least one session recorded a confirmed breakout."""
        return any(r.breakout_detected for r in self.sessions.values())

    @property
    def active_breakouts(self) -> list[ORBResult]:
        """All session results where ``breakout_detected`` is ``True``."""
        return [r for r in self.sessions.values() if r.breakout_detected]

    @property
    def best_breakout(self) -> ORBResult | None:
        """The breakout with the largest OR range (most significant move).

        Returns ``None`` when no sessions have a confirmed breakout.
        """
        breakouts = self.active_breakouts
        if not breakouts:
            return None
        return max(breakouts, key=lambda r: r.or_range)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all session results."""
        return {
            "symbol": self.symbol,
            "evaluated_at": self.evaluated_at,
            "has_any_breakout": self.has_any_breakout,
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
        }
