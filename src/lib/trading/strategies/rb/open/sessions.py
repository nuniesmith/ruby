"""
Opening Range Breakout — Session Definitions
=============================================
Canonical home for ORBSession dataclass, all nine (+ two optional crypto)
session instances, lookup tables, and convenience groupings.

Session order (by ET wall-clock, Globex day starting 18:00 ET):

  1. CME Globex Open         18:00–18:30 ET  (wraps_midnight)
  2. Sydney / ASX Open       18:30–19:00 ET  (wraps_midnight)
  3. Tokyo / TSE Open        19:00–19:30 ET  (wraps_midnight)
  4. Shanghai / HK Open      21:00–21:30 ET  (wraps_midnight)
  5. Frankfurt / Xetra Open  03:00–03:30 ET
  6. London Open             03:00–03:30 ET  ← PRIMARY
  7. London–NY Crossover     08:00–08:30 ET
  8. US Equity Open          09:30–10:00 ET
  9. CME Settlement          14:00–14:30 ET

  Crypto (conditionally appended when ENABLE_KRAKEN_CRYPTO is set):
  10. Crypto UTC Midnight     19:00–19:30 ET  (wraps_midnight)
  11. Crypto UTC Noon         07:00–07:30 ET

DST Handling
------------
All ``or_start`` / ``or_end`` / ``scan_end`` values are **ET wall-clock**
times.  ``ZoneInfo("America/New_York")`` handles EST↔EDT transitions
automatically — no manual UTC offsets needed anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ORBSession:
    """Definition of an Opening Range Breakout session window.

    All time fields (``or_start``, ``or_end``, ``scan_end``) are
    **ET wall-clock** times (``datetime.time`` objects).  Instances are
    frozen and therefore hashable — they can be used as dict keys or in sets.

    Quality-gate thresholds
    ~~~~~~~~~~~~~~~~~~~~~~~
    * ``min_depth_atr_pct``  — close must clear the OR level by at least
      this fraction of ATR (0.0 = disabled).
    * ``min_body_ratio``     — breakout candle body / total range (0.0 = disabled).
    * ``max_or_atr_ratio``   — OR range cap expressed as × ATR (0.0 = disabled).
    * ``min_or_atr_ratio``   — OR range floor expressed as × ATR (0.0 = disabled).

    Midnight-wrap sessions
    ~~~~~~~~~~~~~~~~~~~~~~
    Sessions whose ``or_start`` is in the evening (≥ 17:00 ET) start during
    the *previous* calendar day in ET.  Set ``wraps_midnight=True`` on those
    sessions so bar-filtering logic looks back into the prior day's bars.
    """

    name: str
    key: str
    or_start: dt_time
    or_end: dt_time
    scan_end: dt_time
    atr_period: int = 14
    breakout_multiplier: float = 0.5
    min_bars: int = 5
    max_bars: int = 35
    description: str = ""

    # --- Quality gate thresholds ---
    min_depth_atr_pct: float = 0.15
    min_body_ratio: float = 0.55
    max_or_atr_ratio: float = 1.8
    min_or_atr_ratio: float = 0.05

    # Whether OR window straddles midnight in ET
    wraps_midnight: bool = False

    # Whether to include this session in CNN dataset generation
    include_in_dataset: bool = True


# ===========================================================================
# Session Instances — Full 24-Hour Globex Cycle (ET wall-clock, DST-aware)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. CME Globex Re-Open  18:00–18:30 ET  ← START OF GLOBEX DAY
# First bars after the 17:00–18:00 settlement break — clean overnight
# anchor for all CME micro products.  wraps_midnight=True.
# ---------------------------------------------------------------------------
CME_OPEN_SESSION = ORBSession(
    name="CME Globex Open",
    key="cme",
    or_start=dt_time(18, 0),
    or_end=dt_time(18, 30),
    scan_end=dt_time(20, 0),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=3,
    max_bars=35,
    description=(
        "CME Globex re-open after settlement break (18:00–18:30 ET / 23:00–23:30 UTC EST | 22:00–22:30 UTC EDT)"
    ),
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.6,
    min_or_atr_ratio=0.04,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 2. Sydney / ASX Open  18:30–19:00 ET  (ASX opens ~19:00 ET EST)
# Thin overnight; metals, energy, AUD-correlated FX, MBT.
# wraps_midnight=True.
# ---------------------------------------------------------------------------
SYDNEY_SESSION = ORBSession(
    name="Sydney Open",
    key="sydney",
    or_start=dt_time(18, 30),
    or_end=dt_time(19, 0),
    scan_end=dt_time(20, 30),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Sydney / ASX open (18:30–19:00 ET / 23:30–00:00 UTC EST | 22:30–23:00 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 3. Tokyo / TSE Open  19:00–19:30 ET  (09:00 JST = 19:00 ET EST / 18:00 ET EDT)
# Narrow-range; strongest for metals and JPY/AUD-correlated FX.
# wraps_midnight=True.
# ---------------------------------------------------------------------------
TOKYO_SESSION = ORBSession(
    name="Tokyo Open",
    key="tokyo",
    or_start=dt_time(19, 0),
    or_end=dt_time(19, 30),
    scan_end=dt_time(21, 0),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Tokyo / TSE open (19:00–19:30 ET / 00:00–00:30 UTC EST | 23:00–23:30 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.4,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 4. Shanghai / Hong Kong Open  21:00–21:30 ET
# CSI 300 / HKEX open (09:30 CST / HKT).  Copper (MHG) and gold (MGC)
# sentiment driver via SHFE open-price auction.  wraps_midnight=True.
# ---------------------------------------------------------------------------
SHANGHAI_SESSION = ORBSession(
    name="Shanghai/HK Open",
    key="shanghai",
    or_start=dt_time(21, 0),
    or_end=dt_time(21, 30),
    scan_end=dt_time(23, 0),
    atr_period=14,
    breakout_multiplier=0.4,
    min_bars=3,
    max_bars=35,
    description=("Shanghai/HK open — CSI 300 / HKEX (21:00–21:30 ET / 02:00–02:30 UTC EST | 01:00–01:30 UTC EDT)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.50,
    max_or_atr_ratio=1.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 5. Frankfurt / Xetra Open  03:00–03:30 ET  (08:00–08:30 CET / 09:00 CEST)
# Pre-London institutional flow; DAX-correlated index futures and EUR/USD.
# Same ET time as London open — separate session key for asset filtering.
# ---------------------------------------------------------------------------
FRANKFURT_SESSION = ORBSession(
    name="Frankfurt/Xetra Open",
    key="frankfurt",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    scan_end=dt_time(4, 30),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=4,
    max_bars=35,
    description=("Frankfurt / Xetra open (03:00–03:30 ET / 08:00–08:30 UTC EST | 07:00–07:30 UTC EDT)"),
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.7,
    min_or_atr_ratio=0.04,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 6. London Open  03:00–03:30 ET  (08:00–08:30 UTC)  ← PRIMARY SESSION
# Highest-conviction session.  Institutional flow drives the daily range
# for metals, energy, FX futures, and index futures.
# ---------------------------------------------------------------------------
LONDON_SESSION = ORBSession(
    name="London Open",
    key="london",
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    scan_end=dt_time(5, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description=("London open session (03:00–03:30 ET / 08:00–08:30 UTC EST | 07:00–07:30 UTC EDT)"),
    min_depth_atr_pct=0.15,
    min_body_ratio=0.55,
    max_or_atr_ratio=1.8,
    min_or_atr_ratio=0.05,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 7. London–NY Crossover  08:00–08:30 ET  (13:00–13:30 UTC)
# Both exchanges fully open.  Highest intraday volume and tightest spreads.
# Best assets: 6E, MES, MNQ, MGC.
# ---------------------------------------------------------------------------
LONDON_NY_SESSION = ORBSession(
    name="London-NY Crossover",
    key="london_ny",
    or_start=dt_time(8, 0),
    or_end=dt_time(8, 30),
    scan_end=dt_time(10, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description=("London-NY crossover (08:00–08:30 ET / 13:00–13:30 UTC EST | 12:00–12:30 UTC EDT)"),
    min_depth_atr_pct=0.18,
    min_body_ratio=0.58,
    max_or_atr_ratio=2.0,
    min_or_atr_ratio=0.06,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 8. US Equity Open  09:30–10:00 ET
# Classic Toby Crabel ORB for MES/MNQ.  Gold correlation window.
# ---------------------------------------------------------------------------
US_SESSION = ORBSession(
    name="US Equity Open",
    key="us",
    or_start=dt_time(9, 30),
    or_end=dt_time(10, 0),
    scan_end=dt_time(11, 0),
    atr_period=14,
    breakout_multiplier=0.5,
    min_bars=5,
    max_bars=35,
    description="US equity cash open (09:30–10:00 ET)",
    min_depth_atr_pct=0.15,
    min_body_ratio=0.55,
    max_or_atr_ratio=1.8,
    min_or_atr_ratio=0.05,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# 9. CME Settlement / Late Session  14:00–14:30 ET
# Metals and energy settlement window.  Gold (MGC) and crude (MCL) see
# directional resolution before the 17:00 ET close.
# ---------------------------------------------------------------------------
CME_SETTLEMENT_SESSION = ORBSession(
    name="CME Settlement",
    key="cme_settle",
    or_start=dt_time(14, 0),
    or_end=dt_time(14, 30),
    scan_end=dt_time(15, 30),
    atr_period=14,
    breakout_multiplier=0.45,
    min_bars=3,
    max_bars=35,
    description="CME metals/energy settlement (14:00–14:30 ET)",
    min_depth_atr_pct=0.12,
    min_body_ratio=0.52,
    max_or_atr_ratio=1.7,
    min_or_atr_ratio=0.04,
    wraps_midnight=False,
    include_in_dataset=True,
)

# ---------------------------------------------------------------------------
# Optional crypto sessions (appended at runtime when ENABLE_KRAKEN_CRYPTO)
# ---------------------------------------------------------------------------

# UTC 00:00 ≡ 19:00 ET EST / 20:00 ET EDT
# Reliable volume surge as the new UTC calendar day begins (Asia auction).
CRYPTO_UTC_MIDNIGHT_SESSION = ORBSession(
    name="Crypto UTC Midnight",
    key="crypto_utc0",
    or_start=dt_time(19, 0),  # 19:00 ET EST = 00:00 UTC
    or_end=dt_time(19, 30),
    scan_end=dt_time(21, 0),
    atr_period=14,
    breakout_multiplier=0.65,
    min_bars=3,
    max_bars=35,
    description=("Crypto UTC midnight session (19:00–19:30 ET EST / 20:00–20:30 ET EDT = 00:00–00:30 UTC)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.45,
    max_or_atr_ratio=2.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=True,
    include_in_dataset=True,
)

# UTC 12:00 ≡ 07:00 ET EST / 08:00 ET EDT
# London morning crypto session; European institutional pre-US-open flow.
CRYPTO_UTC_NOON_SESSION = ORBSession(
    name="Crypto UTC Noon",
    key="crypto_utc12",
    or_start=dt_time(7, 0),  # 07:00 ET EST = 12:00 UTC
    or_end=dt_time(7, 30),
    scan_end=dt_time(9, 0),
    atr_period=14,
    breakout_multiplier=0.65,
    min_bars=3,
    max_bars=35,
    description=("Crypto UTC noon session (07:00–07:30 ET EST / 08:00–08:30 ET EDT = 12:00–12:30 UTC)"),
    min_depth_atr_pct=0.10,
    min_body_ratio=0.45,
    max_or_atr_ratio=2.5,
    min_or_atr_ratio=0.03,
    wraps_midnight=False,
    include_in_dataset=True,
)


# ===========================================================================
# Master session list — chronological Globex-day order (18:00 ET start)
# Crypto sessions are NOT included here by default; they are appended at
# runtime by assets.py when ENABLE_KRAKEN_CRYPTO is True.
# ===========================================================================
ORB_SESSIONS: list[ORBSession] = [
    CME_OPEN_SESSION,  # 18:00 ET — start of Globex day
    SYDNEY_SESSION,  # 18:30 ET
    TOKYO_SESSION,  # 19:00 ET
    SHANGHAI_SESSION,  # 21:00 ET
    FRANKFURT_SESSION,  # 03:00 ET — pre-London
    LONDON_SESSION,  # 03:00 ET — primary
    LONDON_NY_SESSION,  # 08:00 ET
    US_SESSION,  # 09:30 ET
    CME_SETTLEMENT_SESSION,  # 14:00 ET
]

# Fast O(1) lookup: session key → ORBSession
SESSION_BY_KEY: dict[str, ORBSession] = {s.key: s for s in ORB_SESSIONS}


# ===========================================================================
# Convenience groupings
# ===========================================================================
# NOTE: These are populated via _rebuild_session_groups() so that any
# sessions appended to ORB_SESSIONS at runtime (e.g. crypto sessions
# injected by assets.py) are automatically reflected here.  Call
# _rebuild_session_groups() after mutating ORB_SESSIONS.
OVERNIGHT_SESSIONS: list[ORBSession] = []
DAYTIME_SESSIONS: list[ORBSession] = []
DATASET_SESSIONS: list[ORBSession] = []


def _rebuild_session_groups() -> None:
    """Rebuild the OVERNIGHT / DAYTIME / DATASET convenience lists in-place.

    Must be called whenever :data:`ORB_SESSIONS` is mutated (e.g. after
    crypto sessions are appended by ``assets.py`` at import time).  The
    lists are mutated in-place so that any code that already holds a
    reference to them sees the updated contents.
    """
    global OVERNIGHT_SESSIONS, DAYTIME_SESSIONS, DATASET_SESSIONS
    OVERNIGHT_SESSIONS[:] = [s for s in ORB_SESSIONS if s.wraps_midnight]
    DAYTIME_SESSIONS[:] = [s for s in ORB_SESSIONS if not s.wraps_midnight]
    DATASET_SESSIONS[:] = [s for s in ORB_SESSIONS if s.include_in_dataset]


# Populate on first import using the base (non-crypto) session list.
_rebuild_session_groups()


# ===========================================================================
# Legacy scalar aliases (kept for backward compatibility)
# ===========================================================================
OR_START = US_SESSION.or_start
OR_END = US_SESSION.or_end
ATR_PERIOD = 14
BREAKOUT_ATR_MULTIPLIER = 0.5
MIN_OR_BARS = 5
MAX_OR_BARS = 35


# ===========================================================================
# Session helper functions
# ===========================================================================


def get_session_for_utc(utc_dt: datetime) -> ORBSession | None:
    """Return the ORBSession currently active for the given UTC datetime.

    Converts *utc_dt* to ET wall-clock time and checks each session's
    ``or_start`` → ``scan_end`` window.  For ``wraps_midnight`` sessions
    the window may straddle 00:00 ET (e.g. CME open 18:00–20:00 ET);
    these are matched whenever the ET time falls within
    ``[or_start, scan_end]`` regardless of calendar date.

    Returns the *first* matching session in ``ORB_SESSIONS`` priority order,
    or ``None`` if no session is currently active.

    Args:
        utc_dt: A tz-aware datetime in UTC (or any tz — will be converted).

    Example::

        from datetime import datetime, timezone
        session = get_session_for_utc(datetime.now(timezone.utc))
        if session:
            print(session.name)
    """
    et_dt = utc_dt.astimezone(_EST)
    et_time = et_dt.time()

    for session in ORB_SESSIONS:
        start = session.or_start
        end = session.scan_end
        if start <= end:
            # Normal window — no midnight crossing
            if start <= et_time <= end:
                return session
        else:
            # Wraps midnight: e.g. 18:00–02:00 (scan_end < or_start)
            if et_time >= start or et_time <= end:
                return session

    return None


def get_active_session_keys(utc_dt: datetime | None = None) -> list[str]:
    """Return keys of ALL sessions whose windows overlap the given UTC time.

    Unlike :func:`get_session_for_utc` (first match only), this returns
    every session active simultaneously — important when Frankfurt and
    London both open at 03:00–03:30 ET.

    Args:
        utc_dt: UTC datetime to check.  Defaults to ``datetime.now(UTC)``.

    Returns:
        List of session key strings (may be empty).
    """
    if utc_dt is None:
        utc_dt = datetime.now(tz=_UTC)

    et_dt = utc_dt.astimezone(_EST)
    et_time = et_dt.time()

    active: list[str] = []
    for session in ORB_SESSIONS:
        start = session.or_start
        end = session.scan_end
        if start <= end:
            if start <= et_time <= end:
                active.append(session.key)
        else:
            if et_time >= start or et_time <= end:
                active.append(session.key)
    return active


def get_active_sessions(now: datetime | None = None) -> list[ORBSession]:
    """Return sessions currently in their OR formation or scan window.

    For overnight sessions (``wraps_midnight=True``) the OR start is in
    the evening ET; the scan window may extend past midnight.  The same
    midnight-wrap logic as :func:`get_active_session_keys` is used so
    that e.g. the CME session (18:00–20:00 ET) is correctly reported as
    active at 19:30 ET.

    Args:
        now: Current time (tz-aware).  Defaults to ``datetime.now(_EST)``.

    Returns:
        List of active :class:`ORBSession` objects (may be empty).
    """
    if now is None:
        now = datetime.now(tz=_EST)

    now_et = now.astimezone(_EST) if now.tzinfo else now.replace(tzinfo=_EST)
    t = now_et.time()

    active = []
    for session in ORB_SESSIONS:
        start = session.or_start
        end = session.scan_end
        if start <= end:
            # Normal (non-wrapping) window
            if start <= t <= end:
                active.append(session)
        else:
            # Wraps midnight: active when t >= start OR t <= end
            if t >= start or t <= end:
                active.append(session)
    return active


def is_any_session_active(now: datetime | None = None) -> bool:
    """Return ``True`` if any ORB session is currently active."""
    return len(get_active_sessions(now)) > 0


def get_session_status(now: datetime | None = None) -> dict[str, str]:
    """Return a status string for every session keyed by session key.

    Possible statuses:

    * ``"waiting"``  — before ``or_start``
    * ``"forming"``  — inside the OR window (``or_start`` ≤ t < ``or_end``)
    * ``"scanning"`` — OR complete, scanning for breakout (``or_end`` ≤ t ≤ ``scan_end``)
    * ``"complete"`` — past ``scan_end``

    For ``wraps_midnight=True`` sessions (e.g. CME Open 18:00–20:00 ET)
    the window straddles midnight, so the comparison is inverted: the
    session is active whenever ``t >= or_start`` *or* ``t <= scan_end``.

    Args:
        now: Current time (tz-aware).  Defaults to ``datetime.now(_EST)``.
    """
    if now is None:
        now = datetime.now(tz=_EST)

    now_et = now.astimezone(_EST) if now.tzinfo else now.replace(tzinfo=_EST)
    t = now_et.time()

    statuses: dict[str, str] = {}
    for session in ORB_SESSIONS:
        if not session.wraps_midnight:
            # Normal intraday window — straightforward comparisons
            if t < session.or_start:
                statuses[session.key] = "waiting"
            elif session.or_start <= t < session.or_end:
                statuses[session.key] = "forming"
            elif session.or_end <= t <= session.scan_end:
                statuses[session.key] = "scanning"
            else:
                statuses[session.key] = "complete"
        else:
            # Wraps-midnight session: or_start > scan_end in wall-clock terms.
            # Active span: [or_start, midnight) ∪ [midnight, scan_end]
            # "complete" means we are between scan_end and or_start on the
            # same afternoon/evening (i.e. outside both halves of the window).
            if session.scan_end < t < session.or_start:
                # Between end of scan window and start of next OR window
                statuses[session.key] = "waiting"
            elif t >= session.or_start and t < session.or_end:
                statuses[session.key] = "forming"
            elif (t >= session.or_end) or (t <= session.scan_end):
                # Either in the evening post-OR half or the early-morning half
                statuses[session.key] = "scanning"
            else:
                statuses[session.key] = "complete"

    return statuses


def get_session_by_key(key: str) -> ORBSession | None:
    """Return the :class:`ORBSession` matching *key*, or ``None``.

    Args:
        key: Session key string, e.g. ``"london"``, ``"tokyo"``, ``"us"``.
    """
    return SESSION_BY_KEY.get(key)
