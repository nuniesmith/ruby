"""
Multi-Session Support — All 9 Globex Sessions with Bracket Parameters
=======================================================================
Single source of truth for every session used by the ``rb`` platform.
Both the Python dataset generator / CNN pipeline **and** the
trading strategy read from this contract.

.. note::

   ``RBSession`` is the preferred alias for ``ORBSession`` (Phase 1G rename).
   Both names are exported and functionally identical.  New code should use
   ``RBSession``; ``ORBSession`` is kept for backward compatibility.

Sessions defined (chronological Globex-day order, 18:00 ET start):
  1. ``cme``        — CME Globex Re-open      18:00–18:30 ET
  2. ``sydney``     — Sydney / ASX            18:30–19:00 ET
  3. ``tokyo``      — Tokyo / TSE             19:00–19:30 ET
  4. ``shanghai``   — Shanghai / HK           21:00–21:30 ET
  5. ``frankfurt``  — Frankfurt / Xetra       03:00–03:30 ET
  6. ``london``     — London Open             03:00–03:30 ET
  7. ``london_ny``  — London-NY Crossover     08:00–08:30 ET
  8. ``us``         — US Equity Open          09:30–10:00 ET
  9. ``cme_settle`` — CME Settlement          14:00–14:30 ET

Each session carries:
  - Open/close time for the opening range window (ET)
  - Pre-market end time (used to compute premarket_range_ratio)
  - Default bracket parameters (SL/TP/max-hold)
  - CNN inference threshold (per ``breakout_cnn.SESSION_THRESHOLDS``)
  - Session ordinal [0, 1] for the CNN tabular feature
  - Metadata (display name, applies_to asset classes, overnight flag)

Usage::

    from lib.core.multi_session import (
        get_session,
        all_sessions,
        session_keys,
        RBSession,       # preferred (Phase 1G)
        ORBSession,      # backward compat alias
        SESSION_BY_KEY,
    )

    sess = get_session("london")
    print(sess.or_start)           # datetime.time(3, 0)
    print(sess.cnn_threshold)      # 0.82
    print(sess.session_ordinal)    # 0.625

    for sess in all_sessions():
        print(sess.key, sess.display_name)

Design:
  - ``ORBSession`` is a frozen dataclass so instances are hashable and safe
    to use as dict keys.
  - ``SESSION_BY_KEY`` is the module-level registry — callers should use
    ``get_session()`` for safe lookup with a clear error message.
  - All times are ``datetime.time`` objects in US/Eastern (ET) — callers are
    responsible for attaching the correct tzinfo when building full datetimes.
  - The ordering in ``ALL_SESSION_KEYS`` mirrors the canonical session list
    in ``orb.py`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Ordered session key list — chronological Globex-day (18:00 ET start)
# ---------------------------------------------------------------------------

ALL_SESSION_KEYS: list[str] = [
    "cme",  # 18:00–18:30 ET
    "sydney",  # 18:30–19:00 ET
    "tokyo",  # 19:00–19:30 ET
    "shanghai",  # 21:00–21:30 ET
    "frankfurt",  # 03:00–03:30 ET
    "london",  # 03:00–03:30 ET  (primary)
    "london_ny",  # 08:00–08:30 ET
    "us",  # 09:30–10:00 ET  (primary)
    "cme_settle",  # 14:00–14:30 ET
]


# ---------------------------------------------------------------------------
# ORBSession dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ORBSession:
    """All parameters for one named Globex session.

    .. note:: ``RBSession`` is the preferred name (Phase 1G rename).
       ``ORBSession`` is kept as a backward-compatible alias.

    Attributes:
        key:                 Short identifier used everywhere in code
                             (e.g. ``"london"``, ``"us"``).
        display_name:        Human-readable label for logs and the dashboard.
        or_start:            Opening-range window start (ET).
        or_end:              Opening-range window end (ET).
        pm_end:              Pre-market window end (ET) — used to compute the
                             ``premarket_range_ratio`` tabular feature.
                             For overnight sessions this equals ``or_start``.
        wraps_midnight:      True if the session straddles the ET midnight
                             boundary (i.e. 18:00–23:59 → 00:00–…).  The
                             dataset generator uses this to correctly slice
                             bar data across date boundaries.
        is_overnight:        True for sessions that run during thin overnight
                             markets (CME, Sydney, Tokyo, Shanghai).  These
                             use lower CNN thresholds.
        sl_atr_mult:         Default stop-loss ATR multiplier.
        tp1_atr_mult:        Default TP1 ATR multiplier.
        tp2_atr_mult:        Default TP2 ATR multiplier.
        max_hold_bars:       Maximum bars to stay in trade before forced exit.
        cnn_threshold:       CNN inference probability threshold for this
                             session.  Overnight sessions use lower values to
                             avoid over-filtering a thinner signal pool.
        session_ordinal:     Normalised position in the 24-h Globex day
                             [0.0, 1.0].  Matches ``SESSION_ORDINAL`` in
                             ``breakout_cnn.py`` and ``feature_contract.json``.
        applies_to:          Asset class tags this session is most relevant
                             for.  ``"all"`` means no filtering.
        description:         One-line summary for documentation.
        extra:               Reserved dict for future per-session parameters.
    """

    key: str
    display_name: str
    or_start: dt_time
    or_end: dt_time
    pm_end: dt_time
    wraps_midnight: bool
    is_overnight: bool
    sl_atr_mult: float
    tp1_atr_mult: float
    tp2_atr_mult: float
    max_hold_bars: int
    cnn_threshold: float
    session_ordinal: float
    applies_to: list[str]
    description: str
    extra: dict[str, Any] = field(default_factory=dict)


# Phase 1G rename: RBSession is the preferred name going forward.
# ORBSession is kept as a backward-compatible alias (they are the same class).
RBSession = ORBSession


# ---------------------------------------------------------------------------
# Session definitions
# ---------------------------------------------------------------------------
#
# Bracket parameters (SL/TP/max_hold) are the *default* values used by
# ``_bracket_configs_for_session()`` in ``dataset_generator.py`` and by the
# trading strategy when no per-symbol override is configured.
#
# CNN thresholds match ``breakout_cnn.SESSION_THRESHOLDS`` exactly — keep
# them in sync if you tune one side.

_CME_SESSION = ORBSession(
    key="cme",
    display_name="CME Globex Re-open",
    or_start=dt_time(18, 0),
    or_end=dt_time(18, 30),
    pm_end=dt_time(18, 0),  # no pre-market concept at open of Globex day
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.75,
    session_ordinal=0.0 / 8.0,  # 0.000
    applies_to=["all"],
    description="CME Globex session re-opens at 18:00 ET — first bars of the new trading day.",
    extra={
        "globex_day_start": True,
        "typical_spread_wide": True,
    },
)

_SYDNEY_SESSION = ORBSession(
    key="sydney",
    display_name="Sydney / ASX",
    or_start=dt_time(18, 30),
    or_end=dt_time(19, 0),
    pm_end=dt_time(18, 30),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.72,  # thinnest session — lowest threshold
    session_ordinal=1.0 / 8.0,  # 0.125
    applies_to=["fx", "metals"],
    description="Sydney / ASX session open at 18:30 ET — thinnest overnight session.",
    extra={
        "active_pairs": ["6A=F", "6J=F"],
    },
)

_TOKYO_SESSION = ORBSession(
    key="tokyo",
    display_name="Tokyo / TSE",
    or_start=dt_time(19, 0),
    or_end=dt_time(19, 30),
    pm_end=dt_time(19, 0),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.74,
    session_ordinal=2.0 / 8.0,  # 0.250
    applies_to=["fx", "metals"],
    description="Tokyo / TSE session at 19:00 ET — narrow ranges, metals and JPY driver.",
    extra={
        "active_pairs": ["6J=F", "MGC=F", "SIL=F"],
    },
)

_SHANGHAI_SESSION = ORBSession(
    key="shanghai",
    display_name="Shanghai / HK",
    or_start=dt_time(21, 0),
    or_end=dt_time(21, 30),
    pm_end=dt_time(21, 0),
    wraps_midnight=True,
    is_overnight=True,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=90,
    cnn_threshold=0.74,
    session_ordinal=3.0 / 8.0,  # 0.375
    applies_to=["metals", "energy"],
    description="Shanghai / Hong Kong session at 21:00 ET — copper and gold driver.",
    extra={
        "active_pairs": ["MHG=F", "MGC=F", "MCL=F"],
    },
)

_FRANKFURT_SESSION = ORBSession(
    key="frankfurt",
    display_name="Frankfurt / Xetra",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    pm_end=dt_time(3, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.80,
    session_ordinal=4.0 / 8.0,  # 0.500
    applies_to=["fx", "equity_index"],
    description="Frankfurt / Xetra open at 03:00 ET — pre-London, good volume on EUR pairs.",
    extra={
        "active_pairs": ["6E=F", "6B=F", "MES=F"],
        "pre_london": True,
    },
)

_LONDON_SESSION = ORBSession(
    key="london",
    display_name="London Open",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    pm_end=dt_time(3, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.82,  # primary session — highest conviction bar
    session_ordinal=5.0 / 8.0,  # 0.625
    applies_to=["all"],
    description="London Open at 03:00 ET — PRIMARY session, highest volume and conviction.",
    extra={
        "primary_session": True,
        "overlaps_frankfurt": True,
    },
)

_LONDON_NY_SESSION = ORBSession(
    key="london_ny",
    display_name="London-NY Crossover",
    or_start=dt_time(8, 0),
    or_end=dt_time(8, 30),
    pm_end=dt_time(8, 0),
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.5,  # highest volume window — wider TP target
    max_hold_bars=120,
    cnn_threshold=0.82,
    session_ordinal=6.0 / 8.0,  # 0.750
    applies_to=["all"],
    description="London-NY Crossover at 08:00 ET — highest total volume of the Globex day.",
    extra={
        "highest_volume": True,
        "overlap_window_start": dt_time(8, 0),
        "overlap_window_end": dt_time(10, 0),
    },
)

_US_SESSION = ORBSession(
    key="us",
    display_name="US Equity Open",
    or_start=dt_time(9, 30),
    or_end=dt_time(10, 0),
    pm_end=dt_time(8, 20),  # pre-market ends at 08:20 ET (gap-up/down reference)
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    max_hold_bars=120,
    cnn_threshold=0.82,  # primary session
    session_ordinal=7.0 / 8.0,  # 0.875
    applies_to=["all"],
    description="US Equity Open at 09:30 ET — classic ORB session for equity indices.",
    extra={
        "primary_session": True,
        "rth_open": True,
        "premarket_ref_start": dt_time(4, 0),  # used for pm_high / pm_low computation
    },
)

_CME_SETTLE_SESSION = ORBSession(
    key="cme_settle",
    display_name="CME Settlement",
    or_start=dt_time(14, 0),
    or_end=dt_time(14, 30),
    pm_end=dt_time(8, 20),  # same pre-market reference as US session
    wraps_midnight=False,
    is_overnight=False,
    sl_atr_mult=1.5,
    tp1_atr_mult=1.5,  # tighter targets near settlement
    tp2_atr_mult=2.5,
    max_hold_bars=60,  # shorter hold — settlement can reverse quickly
    cnn_threshold=0.78,
    session_ordinal=8.0 / 8.0,  # 1.000
    applies_to=["metals", "energy"],
    description="CME Settlement at 14:00 ET — metals and energy settlement window.",
    extra={
        "settlement_window": True,
        "active_pairs": ["MGC=F", "MCL=F", "MHG=F", "MNG=F"],
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SESSION_BY_KEY: dict[str, ORBSession] = {
    "cme": _CME_SESSION,
    "sydney": _SYDNEY_SESSION,
    "tokyo": _TOKYO_SESSION,
    "shanghai": _SHANGHAI_SESSION,
    "frankfurt": _FRANKFURT_SESSION,
    "london": _LONDON_SESSION,
    "london_ny": _LONDON_NY_SESSION,
    "us": _US_SESSION,
    "cme_settle": _CME_SETTLE_SESSION,
}


# ---------------------------------------------------------------------------
# UTC exchange hours — single source of truth for all global session times
# ---------------------------------------------------------------------------
#
# Every exchange opens and closes at a fixed UTC time (no DST — the exchange
# doesn't move, only our local clock does).  We store open/close as fractional
# UTC hours so that ``exchange_hours_in_et()`` can convert them to the correct
# ET wall-clock hour automatically, handling both EST (UTC-5) and EDT (UTC-4).
#
# Format: (label, utc_open, utc_close, fg_hex, bg_hex, row)
#   utc_close > 24 means the session closes the following UTC calendar day.
#   row: 0 = background span (CME Globex), 1 = overnight/Asian/EU, 2 = primary
#
# UTC open/close derivations:
#   CME Globex re-open : 22:00 UTC (18:00 EST / 17:00 EDT — but CME actually
#                        re-opens at 18:00 ET, so UTC = 18:00+5=23:00 in winter,
#                        18:00+4=22:00 in summer.  We store as ET wall-clock
#                        via the ORBSession or_start already; see note below.)
#
# NOTE: CME, Sydney, Tokyo, Shanghai times are all defined in ET wall-clock in
# the ORBSession objects above (they follow US DST, since CME is Chicago-based).
# Frankfurt and London are fixed UTC — they DO shift relative to ET when DST
# transitions happen on different dates in the US vs Europe.
#
# Full exchange hours (UTC, no DST):
#   ASX / Sydney    :  23:00 – 06:00 UTC  (session open 10:00–16:00 AEST/AEDT;
#                       AEDT=UTC+11 → 23:00–05:00 UTC Oct–Apr,
#                       AEST=UTC+10 → 00:00–06:00 UTC Apr–Oct)
#                      We use AEDT (active until ~early Apr): open=23:00, close=05:00
#   Tokyo / TSE     :  00:00 – 06:25 UTC  (09:00–15:25 JST, UTC+9, no DST)
#                      Lunch break 02:30–03:30 UTC (11:30–12:30 JST)
#   Shanghai / SSE  :  01:30 – 07:00 UTC  (09:30–15:00 CST, UTC+8, no DST)
#   Frankfurt/Xetra :  08:00 – 16:30 UTC  (09:00–17:30 CET=UTC+1 / CEST=UTC+2)
#                      During CET  (winter): open=08:00, close=16:30 UTC
#                      During CEST (summer): open=07:00, close=15:30 UTC
#                      Europe switches ~last Sun Mar → last Sun Oct
#   London / LSE    :  08:00 – 16:30 UTC  (08:00–16:30 GMT / 07:00–15:30 BST)
#                      During GMT  (winter): open=08:00, close=16:30 UTC
#                      During BST  (summer): open=07:00, close=15:30 UTC
#   NYSE / CME RTH  :  14:30 – 21:00 UTC  (09:30–16:00 ET, DST-aware via ET)
#   CME Settlement  :  19:00 – 19:30 UTC  (14:00–14:30 EST / 15:00–15:30 EDT)
#                      We keep settlement as ET wall-clock (14:00 ET) in ORBSession.
#
# For the dashboard strip we store everything as UTC fractional hours and
# convert at render time so the displayed ET positions are always correct.

_UTC = UTC
_ET = ZoneInfo("America/New_York")


def _utc_frac_to_et_frac(utc_frac: float) -> float:
    """Convert a fractional UTC hour (e.g. 8.5 = 08:30 UTC) to a fractional
    ET wall-clock hour using the *current* UTC offset for America/New_York.

    This correctly handles both EST (UTC-5, offset=-5) and EDT (UTC-4,
    offset=-4) without any hardcoded offsets.

    Returns a value in [0, 24) for same-day hours, or a negative / >24
    value if the conversion crosses midnight — callers normalise with % 24
    or handle wrap-around in the strip renderer.
    """
    # Use today's date to get an accurate DST-aware offset
    today = datetime.now(tz=_UTC).date()
    utc_h = int(utc_frac)
    utc_m = int(round((utc_frac - utc_h) * 60))
    # Clamp hour into 0-23 for the datetime constructor (wrap-around handled below)
    utc_h_clamped = utc_h % 24
    extra_days = utc_h // 24
    from datetime import timedelta

    dt_utc = datetime(today.year, today.month, today.day, utc_h_clamped, utc_m, tzinfo=_UTC) + timedelta(
        days=extra_days
    )
    dt_et = dt_utc.astimezone(_ET)
    return dt_et.hour + dt_et.minute / 60.0


def _et_offset_hours() -> float:
    """Return the current UTC offset for America/New_York as a signed float.

    Examples: -5.0 during EST (winter), -4.0 during EDT (summer).
    """
    now_et = datetime.now(tz=_ET)
    offset = now_et.utcoffset()
    assert offset is not None
    return offset.total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Exchange hours in UTC — used by exchange_hours_in_et()
# ---------------------------------------------------------------------------
#
# Tuple fields:
#   key        : matches SESSION_BY_KEY keys where applicable
#   label      : display label for the session strip
#   utc_open   : fractional UTC hour of exchange open
#   utc_close  : fractional UTC hour of exchange close (>24 = next UTC day)
#   fg_hex     : foreground / label colour
#   bg_hex     : background bar colour
#   row        : strip row (0=background, 1=overnight/Asian/EU, 2=primary)
#   note       : human-readable note about DST behaviour

_EXCHANGE_HOURS_UTC: list[dict] = [
    # ── CME Globex background span (22:00 UTC → ~21:00 UTC next day, ~23 h) ──
    # CME re-opens at 18:00 ET regardless of DST, so the UTC time shifts with
    # the US DST transition.  We derive this from _ET offset rather than
    # hardcoding UTC, via the OR start time in _CME_SESSION.
    # The background bar is rendered separately using ET wall-clock from ORBSession.
    # ── Asian overnight sessions ──
    {
        "key": "sydney",
        "label": "SYD/ASX",
        "utc_open": 23.0,  # 10:00 AEDT (UTC+11); shifts to 00:00 UTC when AEST (UTC+10) active Apr–Oct
        "utc_close": 29.0,  # 06:00 UTC next day (16:00 AEDT) — stored as 24+5
        "fg_hex": "#94a3b8",
        "bg_hex": "#1e293b",
        "row": 1,
        "note": "AEDT=UTC+11 (Oct–Apr), AEST=UTC+10 (Apr–Oct). Open shifts 1h later in AEST.",
    },
    {
        "key": "tokyo",
        "label": "TYO/TSE",
        "utc_open": 0.0,  # 09:00 JST (UTC+9), no DST
        "utc_close": 6.4167,  # 15:25 JST = 06:25 UTC
        "fg_hex": "#a5b4fc",
        "bg_hex": "#1e1b4b",
        "row": 1,
        "note": "JST=UTC+9, no DST. Lunch 02:30–03:30 UTC (11:30–12:30 JST).",
    },
    {
        "key": "shanghai",
        "label": "SHA/SSE",
        "utc_open": 1.5,  # 09:30 CST (UTC+8), no DST
        "utc_close": 7.0,  # 15:00 CST = 07:00 UTC
        "fg_hex": "#fca5a5",
        "bg_hex": "#3b0a0a",
        "row": 1,
        "note": "CST=UTC+8, no DST.",
    },
    # ── European sessions ──
    # Frankfurt and London share the same UTC hours (both CET/BST align with GMT/CET).
    # During EU summer time (CEST/BST) the UTC open is 07:00 instead of 08:00.
    # We compute the correct UTC open dynamically in exchange_hours_in_et().
    {
        "key": "frankfurt",
        "label": "FRA/Xetra",
        "utc_open": 8.0,  # 09:00 CET (UTC+1); 07:00 UTC during CEST (UTC+2)
        "utc_close": 16.5,  # 17:30 CET = 16:30 UTC; 15:30 UTC during CEST
        "fg_hex": "#fde68a",
        "bg_hex": "#1c1a08",
        "row": 1,
        "note": "CET=UTC+1 (Oct–Mar), CEST=UTC+2 (Mar–Oct). Open shifts 1h earlier in CEST.",
    },
    {
        "key": "london",
        "label": "LON/LSE",
        "utc_open": 8.0,  # 08:00 GMT (UTC+0); 07:00 UTC during BST (UTC+1)
        "utc_close": 16.5,  # 16:30 GMT = 16:30 UTC; 15:30 UTC during BST
        "fg_hex": "#93c5fd",
        "bg_hex": "#1e3a5f",
        "row": 2,
        "note": "GMT=UTC+0 (Oct–Mar), BST=UTC+1 (Mar–Oct). Open shifts 1h earlier in BST.",
    },
    # ── US primary session ──
    {
        "key": "us",
        "label": "US Equity",
        "utc_open": 14.5,  # 09:30 ET — DST-aware: 14:30 UTC (EDT=UTC-4) or 14:30 UTC (EST=UTC-5=14:30... wait)
        # NYSE 09:30 ET: EDT(UTC-4) → 13:30 UTC; EST(UTC-5) → 14:30 UTC
        # We leave utc_open as 13.5 and recompute below via ET offset.
        # Placeholder overridden in exchange_hours_in_et() for ET-anchored sessions.
        "utc_close": 21.0,  # 16:00 ET → 20:00 UTC (EDT) or 21:00 UTC (EST)
        "fg_hex": "#6ee7b7",
        "bg_hex": "#052e16",
        "row": 2,
        "note": "NYSE RTH 09:30–16:00 ET. UTC time shifts with US DST.",
    },
]

# Sessions whose open/close times are anchored to ET wall-clock (not fixed UTC).
# These are derived from the ORBSession or_start and the known session length,
# so they automatically follow the US DST schedule.
_ET_ANCHORED_SESSION_KEYS = {"cme", "sydney", "tokyo", "shanghai", "us", "cme_settle"}


def _eu_utc_offset(dt_utc: datetime) -> int:
    """Return the UTC offset (hours) for Central European Time at a given UTC datetime.

    CET  (winter) = UTC+1: last Sunday Oct → last Sunday Mar
    CEST (summer) = UTC+2: last Sunday Mar → last Sunday Oct
    UK/BST follows the same schedule: GMT=UTC+0 winter, BST=UTC+1 summer.

    We determine this by asking ZoneInfo for Europe/Berlin (CET/CEST).
    """
    berlin = ZoneInfo("Europe/Berlin")
    dt_berlin = dt_utc.astimezone(berlin)
    offset = dt_berlin.utcoffset()
    assert offset is not None
    return int(offset.total_seconds() // 3600)


def exchange_hours_in_et() -> list[dict]:
    """Return all exchange session hours converted to ET wall-clock fractions.

    This is the single authoritative source consumed by the dashboard session
    strip renderer and the JS badge updater.  All times are expressed as
    fractional ET hours (e.g. 9.5 = 09:30 ET).  Sessions that wrap past
    midnight have ``et_close > 24`` (e.g. 26.0 = 02:00 ET next day).

    The conversion is done at call time using the live UTC↔ET offset
    (``ZoneInfo("America/New_York")``), so it is always correct for both
    EDT (UTC-4, summer) and EST (UTC-5, winter) without any hardcoded offsets.

    European sessions (Frankfurt, London) additionally account for the
    EU DST schedule (CET/CEST, GMT/BST) which transitions on different dates
    than the US schedule.

    ET-anchored sessions (CME, Sydney, Tokyo, Shanghai, US Equity, CME
    Settlement) are taken directly from the ORBSession ``or_start``/``or_end``
    ET wall-clock times plus their known full-session lengths, so they
    automatically follow the US DST calendar.

    Returns a list of dicts, one per session, with keys:
        key, label, et_open, et_close, fg_hex, bg_hex, row, note
    """

    now_utc = datetime.now(tz=_UTC)
    et_offset = _et_offset_hours()  # e.g. -4.0 (EDT) or -5.0 (EST)
    eu_offset = _eu_utc_offset(now_utc)  # e.g. +1 (CET) or +2 (CEST)

    result: list[dict] = []

    # ── ET-anchored sessions (follow US DST via ORBSession ET wall-clock) ──
    # CME Globex re-open: 18:00–18:30 ET (ORB window only; full session is 23 h)
    cme = SESSION_BY_KEY["cme"]
    cme_open_et = cme.or_start.hour + cme.or_start.minute / 60.0
    result.append(
        {
            "key": "cme",
            "label": "CME Re-open",
            "et_open": cme_open_et,  # 18.0
            "et_close": cme_open_et + 0.5,  # 18.5  (ORB window end)
            "fg_hex": "#2dd4bf",
            "bg_hex": "#042f2e",
            "row": 0,
            "note": "CME Globex re-opens 18:00 ET (ORB window 18:00–18:30).",
        }
    )
    # CME Globex full background bar: 18:00 ET → 17:00 ET next day (~23 h)
    result.append(
        {
            "key": "cme_background",
            "label": "CME Globex",
            "et_open": cme_open_et,  # 18.0
            "et_close": cme_open_et + 23.0,  # 41.0 → rendered as 18:00–41:00 (clips at 24h on strip)
            "fg_hex": "#5eead4",
            "bg_hex": "#042f2e",
            "row": 0,
            "note": "CME Globex full 23-hour background session.",
        }
    )

    # Sydney / ASX: full exchange 10:00–16:00 AEST/AEDT
    # AEDT (UTC+11, Oct–Apr): 23:00–05:00 UTC → ET via utc_frac_to_et
    # AEST (UTC+10, Apr–Oct): 00:00–06:00 UTC → ET via utc_frac_to_et
    # We determine which is active by asking Australia/Sydney ZoneInfo.
    sydney_tz = ZoneInfo("Australia/Sydney")
    dt_sydney = now_utc.astimezone(sydney_tz)
    aest_offset = int(dt_sydney.utcoffset().total_seconds() // 3600)  # type: ignore[union-attr]
    # ASX opens 10:00 local = 10:00 - aest_offset UTC
    asx_open_utc = 10.0 - aest_offset
    asx_close_utc = 16.0 - aest_offset
    # Convert to ET: add et_offset (negative)
    asx_open_et = asx_open_utc + et_offset
    asx_close_et = asx_close_utc + et_offset
    # Normalise into Globex-day frame (18:00 ET = hour 18; midnight = 24; etc.)
    if asx_open_et < 0:
        asx_open_et += 24
    if asx_close_et <= asx_open_et:
        asx_close_et += 24
    result.append(
        {
            "key": "sydney",
            "label": "SYD/ASX",
            "et_open": asx_open_et,
            "et_close": asx_close_et,
            "fg_hex": "#94a3b8",
            "bg_hex": "#1e293b",
            "row": 1,
            "note": f"ASX 10:00–16:00 local (AEDT/AEST=UTC+{aest_offset}). "
            f"Opens {asx_open_et:.4g}h ET, closes {asx_close_et:.4g}h ET.",
        }
    )

    # Tokyo / TSE: 09:00–15:25 JST (UTC+9, no DST)
    tyo_open_utc = 0.0  # 09:00 JST - 9 = 00:00 UTC
    tyo_close_utc = 6.4167  # 15:25 JST - 9 = 06:25 UTC
    tyo_open_et = tyo_open_utc + et_offset
    tyo_close_et = tyo_close_utc + et_offset
    if tyo_open_et < 0:
        tyo_open_et += 24
    if tyo_close_et <= 0:
        tyo_close_et += 24
    # Ensure close > open in Globex-day frame
    if tyo_close_et <= tyo_open_et:
        tyo_close_et += 24
    result.append(
        {
            "key": "tokyo",
            "label": "TYO/TSE",
            "et_open": tyo_open_et,
            "et_close": tyo_close_et,
            "fg_hex": "#a5b4fc",
            "bg_hex": "#1e1b4b",
            "row": 1,
            "note": f"TSE 09:00–15:25 JST (UTC+9, no DST). "
            f"Opens {tyo_open_et:.4g}h ET, closes {tyo_close_et:.4g}h ET. "
            "Lunch break 11:30–12:30 JST.",
        }
    )

    # Shanghai / SSE: 09:30–15:00 CST (UTC+8, no DST)
    sha_open_utc = 1.5  # 09:30 CST - 8 = 01:30 UTC
    sha_close_utc = 7.0  # 15:00 CST - 8 = 07:00 UTC
    sha_open_et = sha_open_utc + et_offset
    sha_close_et = sha_close_utc + et_offset
    if sha_open_et < 0:
        sha_open_et += 24
    if sha_close_et < 0:
        sha_close_et += 24
    if sha_close_et <= sha_open_et:
        sha_close_et += 24
    result.append(
        {
            "key": "shanghai",
            "label": "SHA/SSE",
            "et_open": sha_open_et,
            "et_close": sha_close_et,
            "fg_hex": "#fca5a5",
            "bg_hex": "#3b0a0a",
            "row": 1,
            "note": f"SSE 09:30–15:00 CST (UTC+8, no DST). Opens {sha_open_et:.4g}h ET, closes {sha_close_et:.4g}h ET.",
        }
    )

    # Frankfurt / Xetra: 09:00–17:30 local (CET=UTC+1 or CEST=UTC+2)
    fra_open_utc = 9.0 - eu_offset  # 08:00 UTC (CET) or 07:00 UTC (CEST)
    fra_close_utc = 17.5 - eu_offset  # 16:30 UTC (CET) or 15:30 UTC (CEST)
    fra_open_et = fra_open_utc + et_offset
    fra_close_et = fra_close_utc + et_offset
    result.append(
        {
            "key": "frankfurt",
            "label": "FRA/Xetra",
            "et_open": fra_open_et,
            "et_close": fra_close_et,
            "fg_hex": "#fde68a",
            "bg_hex": "#1c1a08",
            "row": 1,
            "note": f"Xetra 09:00–17:30 local (CET/CEST=UTC+{eu_offset}). "
            f"Opens {fra_open_et:.4g}h ET, closes {fra_close_et:.4g}h ET.",
        }
    )

    # London / LSE: 08:00–16:30 local (GMT=UTC+0 or BST=UTC+1)
    # BST offset = eu_offset - 1 (London is always 1h behind Frankfurt)
    bst_offset = eu_offset - 1  # 0 (GMT winter) or 1 (BST summer)
    lon_open_utc = 8.0 - bst_offset  # 08:00 UTC (GMT) or 07:00 UTC (BST)
    lon_close_utc = 16.5 - bst_offset  # 16:30 UTC (GMT) or 15:30 UTC (BST)
    lon_open_et = lon_open_utc + et_offset
    lon_close_et = lon_close_utc + et_offset
    result.append(
        {
            "key": "london",
            "label": "LON/LSE",
            "et_open": lon_open_et,
            "et_close": lon_close_et,
            "fg_hex": "#93c5fd",
            "bg_hex": "#1e3a5f",
            "row": 2,
            "note": f"LSE 08:00–16:30 local (GMT/BST=UTC+{bst_offset}). "
            f"Opens {lon_open_et:.4g}h ET, closes {lon_close_et:.4g}h ET.",
        }
    )

    # US Equity / NYSE: 09:30–16:00 ET (ET-anchored, follows US DST)
    us = SESSION_BY_KEY["us"]
    us_open_et = us.or_start.hour + us.or_start.minute / 60.0  # 9.5
    result.append(
        {
            "key": "us",
            "label": "US Equity",
            "et_open": us_open_et,  # 9.5
            "et_close": 16.0,
            "fg_hex": "#6ee7b7",
            "bg_hex": "#052e16",
            "row": 2,
            "note": "NYSE/CME RTH 09:30–16:00 ET. ET-anchored, follows US DST.",
        }
    )

    # CME Settlement: 14:00–14:30 ET (ET-anchored)
    settle = SESSION_BY_KEY["cme_settle"]
    settle_open_et = settle.or_start.hour + settle.or_start.minute / 60.0  # 14.0
    result.append(
        {
            "key": "cme_settle",
            "label": "CME Settle",
            "et_open": settle_open_et,
            "et_close": settle_open_et + 0.5,
            "fg_hex": "#fb923c",
            "bg_hex": "#1c0a00",
            "row": 1,
            "note": "CME metals/energy settlement window 14:00–14:30 ET.",
        }
    )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_session(key: str) -> ORBSession:
    """Return the ``ORBSession`` for *key*.

    Args:
        key: Session key string (e.g. ``"london"``, ``"us"``).
             Case-insensitive.

    Returns:
        ``ORBSession`` frozen dataclass instance.

    Raises:
        KeyError: If *key* is not a recognised session key, with a helpful
                  message listing all valid keys.

    Example::

        >>> from multi_session import get_session
        >>> sess = get_session("london")
        >>> sess.or_start
        datetime.time(3, 0)
        >>> sess.cnn_threshold
        0.82
    """
    _key = key.strip().lower()
    if _key not in SESSION_BY_KEY:
        raise KeyError(f"Unknown session key {key!r}. Valid keys: {ALL_SESSION_KEYS}")
    return SESSION_BY_KEY[_key]


def all_sessions() -> list[ORBSession]:
    """Return all ``ORBSession`` objects in Globex-day chronological order.

    Example::

        >>> from multi_session import all_sessions
        >>> for s in all_sessions():
        ...     print(s.key, s.session_ordinal)
        cme 0.0
        sydney 0.125
        ...
    """
    return [SESSION_BY_KEY[k] for k in ALL_SESSION_KEYS]


def session_keys() -> list[str]:
    """Return all session keys in Globex-day chronological order."""
    return list(ALL_SESSION_KEYS)


def overnight_sessions() -> list[ORBSession]:
    """Return only the overnight sessions (is_overnight=True).

    These sessions use lower CNN thresholds due to thin markets.
    """
    return [s for s in all_sessions() if s.is_overnight]


def daytime_sessions() -> list[ORBSession]:
    """Return only the daytime sessions (is_overnight=False)."""
    return [s for s in all_sessions() if not s.is_overnight]


def sessions_for_asset_class(asset_class: str) -> list[ORBSession]:
    """Return sessions relevant to *asset_class*.

    Args:
        asset_class: One of ``"fx"``, ``"metals"``, ``"energy"``,
                     ``"equity_index"``, ``"crypto"``, or ``"all"``.

    Returns:
        List of ``ORBSession`` objects whose ``applies_to`` list contains
        *asset_class* or ``"all"``.

    Example::

        >>> from multi_session import sessions_for_asset_class
        >>> [s.key for s in sessions_for_asset_class("metals")]
        ['sydney', 'tokyo', 'shanghai', 'frankfurt', 'london', 'london_ny', 'us', 'cme_settle']
    """
    _ac = asset_class.strip().lower()
    return [s for s in all_sessions() if "all" in s.applies_to or _ac in s.applies_to]


def to_bracket_params(session: ORBSession) -> dict[str, Any]:
    """Serialise a session's bracket parameters to a plain dict.

    This is the format consumed by ``rb_simulator.BracketConfig`` and the
    session bracket structure used by the trading strategy.

    Returns::

        {
            "key":           "london",
            "or_start":      "03:00",
            "or_end":        "03:30",
            "pm_end":        "03:00",
            "sl_atr_mult":   1.5,
            "tp1_atr_mult":  2.0,
            "tp2_atr_mult":  3.0,
            "max_hold_bars": 120,
        }
    """
    return {
        "key": session.key,
        "or_start": session.or_start.strftime("%H:%M"),
        "or_end": session.or_end.strftime("%H:%M"),
        "pm_end": session.pm_end.strftime("%H:%M"),
        "sl_atr_mult": session.sl_atr_mult,
        "tp1_atr_mult": session.tp1_atr_mult,
        "tp2_atr_mult": session.tp2_atr_mult,
        "max_hold_bars": session.max_hold_bars,
    }


def to_feature_contract_dict() -> dict[str, Any]:
    """Serialise all sessions to a dict for ``feature_contract.json``.

    Inserted under the ``"sessions"`` key so consumers can verify
    ordinals and thresholds at load time.

    Example output::

        {
          "us":        {"ordinal": 7, "session_ordinal": 0.875, "cnn_threshold": 0.82, ...},
          "london":    {"ordinal": 5, "session_ordinal": 0.625, "cnn_threshold": 0.82, ...},
          ...
        }
    """
    return {
        sess.key: {
            "ordinal": ALL_SESSION_KEYS.index(sess.key),
            "session_ordinal": round(sess.session_ordinal, 6),
            "cnn_threshold": sess.cnn_threshold,
            "or_start": sess.or_start.strftime("%H:%M"),
            "or_end": sess.or_end.strftime("%H:%M"),
            "pm_end": sess.pm_end.strftime("%H:%M"),
            "is_overnight": sess.is_overnight,
            "wraps_midnight": sess.wraps_midnight,
            "display_name": sess.display_name,
            "applies_to": sess.applies_to,
        }
        for sess in all_sessions()
    }
