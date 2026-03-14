"""
Breakout Types — BreakoutType enum + RangeConfig dataclass
===========================================================
Single source of truth for every range-breakout variant supported by the
platform.  Both the Python dataset generator / CNN pipeline **and**
external consumers read from this contract.

Canonical location: ``lib.core.breakout_types``

Breakout types
--------------
Exchange-based (original four):
- **ORB** (Opening Range Breakout)          — classic 30-min OR at session open
- **PrevDay** (Previous Day High/Low)        — prior session's H/L as the range
- **InitialBalance** (IB — first hour range) — first 60 min of RTH session
- **Consolidation** (detected tight range)   — auto-detected consolidation box

Researched non-exchange range breakouts:
- **Weekly**            — prior week's high/low as the range
- **Monthly**           — prior month's high/low as the range
- **Asian**             — Asian session (19:00–02:00 ET) high/low as the range
- **BollingerSqueeze**  — Bollinger Band compression → expansion breakout
- **ValueArea**         — Prior session's Value Area High/Low (volume profile)
- **InsideDay**         — Inside day (today's range inside yesterday's)
- **GapRejection**      — Gap fill / rejection of overnight gap
- **PivotPoints**       — Classic floor pivot S1/R1 as the range
- **Fibonacci**         — Fib retracement 38.2%–61.8% zone as the range

Design
------
- ``BreakoutType`` is an ``IntEnum`` so values survive JSON round-trips and
  map directly to the ``BreakoutType`` enum used by external consumers.
- **Do not reorder** existing values after training begins — they are embedded
  in saved CSV datasets.  New types are appended at the end.
- ``RangeConfig`` captures everything that varies per type: OR duration,
  look-back windows, box-style constants, TP/SL parameters including TP3
  and EMA9 trailing, and the ordinal used as a CNN tabular feature
  (``breakout_type_ord``).
- ``get_range_config()`` is the single authoritative lookup; callers should
  always use it rather than constructing ``RangeConfig`` by hand.

Usage::

    from lib.core.breakout_types import BreakoutType, get_range_config

    cfg = get_range_config(BreakoutType.ORB)
    print(cfg.or_duration_minutes)   # 30
    print(cfg.breakout_type_ord)     # 0.0
    print(cfg.box_style)             # "gold_dashed"
    print(cfg.tp3_atr_mult)          # 4.5
    print(cfg.enable_ema_trail_after_tp2)  # True

    # Iterate all types for dataset generation
    for bt in BreakoutType:
        cfg = get_range_config(bt)
        print(bt.name, cfg.breakout_type_ord)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dt_time
from enum import IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# BreakoutType enum
# ---------------------------------------------------------------------------

# Total number of breakout types — used to normalise ordinals to [0, 1].
_NUM_BREAKOUT_TYPES = 13  # 0..12 inclusive


class BreakoutType(IntEnum):
    """Range breakout variant.

    Values are stable integer ordinals — **do not reorder** after training
    begins, as they are embedded in saved CSV datasets.

    These mirror the canonical ``BreakoutType`` ordinals used across the
    platform::

        ORB = 0, PrevDay = 1, InitialBalance = 2, Consolidation = 3,
        Weekly = 4, Monthly = 5, Asian = 6, BollingerSqueeze = 7,
        ValueArea = 8, InsideDay = 9, GapRejection = 10,
        PivotPoints = 11, Fibonacci = 12
    """

    ORB = 0
    """Opening Range Breakout — standard 30-min OR at session open."""

    PrevDay = 1
    """Previous Day High/Low — prior session's high and low as the range."""

    InitialBalance = 2
    """Initial Balance — first 60 minutes of the RTH (or primary) session."""

    Consolidation = 3
    """Detected Consolidation — auto-identified tight range / inside day."""

    Weekly = 4
    """Weekly Range — prior week's high and low as the range."""

    Monthly = 5
    """Monthly Range — prior month's high and low as the range."""

    Asian = 6
    """Asian Session Range — 19:00–02:00 ET high/low (Tokyo/Sydney liquidity)."""

    BollingerSqueeze = 7
    """Bollinger Squeeze — BB compression followed by expansion bar."""

    ValueArea = 8
    """Value Area — prior session's VAH/VAL from volume profile."""

    InsideDay = 9
    """Inside Day — today's range entirely inside yesterday's range."""

    GapRejection = 10
    """Gap Rejection — overnight gap fill or rejection at gap edge."""

    PivotPoints = 11
    """Pivot Points — classic floor pivot R1/S1 as the range."""

    Fibonacci = 12
    """Fibonacci — 38.2%–61.8% retracement zone of prior swing."""


# ---------------------------------------------------------------------------
# RangeConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeConfig:
    """All parameters that vary by ``BreakoutType``.

    This is the **single source of truth** for both the training/dataset
    pipeline and the live detection engine.  Phase 1B merged the former
    engine-side ``RangeConfig`` (detection thresholds) into this dataclass
    so that ``get_range_config(BreakoutType.ORB)`` returns everything —
    TP/SL targets, visual styles, *and* detection parameters.

    Attributes — Identity & ordinal:
        breakout_type:        The ``BreakoutType`` this config describes.
        breakout_type_ord:    Normalised ordinal for the CNN tabular feature
                              ``breakout_type_ord`` = ``BreakoutType.value / (_NUM_BREAKOUT_TYPES - 1)``.
                              Range [0.0, 1.0].
        label:                Human-readable label for logging / alerts.

    Attributes — Range shape:
        or_duration_minutes:  How many minutes define the opening range.
                              ORB=30, IB=60, Asian=420, others=0 (pre-computed).
        lookback_days:        How many prior sessions to look back when computing
                              the range level (PrevDay=1, Weekly=5, Monthly=20,
                              IB=0 same day, etc.).

    Attributes — Range quality gates:
        min_range_atr_ratio:  Minimum (range / ATR) to be considered a valid
                              setup.  Tighter ranges have lower thresholds.
        max_range_atr_ratio:  Maximum (range / ATR).  Ranges that are enormous
                              relative to ATR tend to be gappy / low-probability.

    Attributes — TP / SL / trailing:
        tp1_atr_mult:         Default TP1 multiplier (in ATR units).
        tp2_atr_mult:         Default TP2 multiplier.
        tp3_atr_mult:         Default TP3 multiplier (extended target).
                              Set to 0.0 to disable TP3.
        sl_atr_mult:          Default stop-loss multiplier.
        enable_ema_trail_after_tp2:
                              If True, after TP2 is hit, trail remaining
                              contracts with EMA9 crossover instead of a
                              fixed stop.  Exit at TP3 or EMA9 stop.
        ema_trail_period:     EMA period for trailing after TP2 (default 9).

    Attributes — Visual / chart rendering:
        box_style:            Chart renderer token that controls the visual
                              style of the range box drawn on the chart.
        box_border_rgba:      RGBA tuple for the range box border line.
        box_fill_rgba:        RGBA tuple for the range box fill.
        description:          Human-readable description for logs / dashboard.

    Attributes — Detection thresholds (Phase 1B merge from engine):
        atr_period:           Look-back bars for ATR computation.
        atr_multiplier:       Breakout threshold = atr_multiplier × ATR.
        min_depth_atr_pct:    Close must clear level by ≥ N×ATR.
        min_body_ratio:       Body/range ≥ this on breakout bar.
        min_bars:             Minimum bars to form range.

    Attributes — IB-specific:
        ib_duration_minutes:  Initial Balance window length (minutes).
        ib_start_time:        RTH open time (ET) for IB computation.

    Attributes — PDR-specific:
        pdr_session_start:    Globex-day start (ET) for PDR range.
        pdr_session_end:      Globex-day end (ET) for PDR range.

    Attributes — Consolidation / Squeeze:
        squeeze_lookback:     Bars to measure contraction over.
        squeeze_atr_mult:     BB must be < mult×ATR to qualify as squeeze.
        squeeze_bb_period:    Bollinger Band period.
        squeeze_bb_std:       Bollinger Band std-dev multiplier.
        squeeze_min_bars:     Min consecutive squeeze bars needed.

    Attributes — Weekly-specific:
        weekly_lookback_days: Trading days to look back for prior week H/L.

    Attributes — Monthly-specific:
        monthly_lookback_days: Trading days to look back for prior month H/L.

    Attributes — Asian-specific:
        asian_start_time:     Asian range start (ET) — wraps midnight.
        asian_end_time:       Asian range end (ET).

    Attributes — BollingerSqueeze-specific (BB inside KC):
        bbsqueeze_bb_period:  BB period for BB-inside-KC squeeze.
        bbsqueeze_bb_std:     BB std-dev multiplier.
        bbsqueeze_kc_period:  Keltner Channel period.
        bbsqueeze_kc_atr_mult: KC ATR multiplier.
        bbsqueeze_min_squeeze_bars: Min consecutive bars where BB inside KC.

    Attributes — ValueArea-specific:
        va_value_area_pct:    Percentage of volume for value area.
        va_n_bins:            Price bins for volume profile computation.

    Attributes — InsideDay-specific:
        inside_min_compression: Today range / yesterday range floor.
        inside_max_compression: Today range / yesterday range cap.

    Attributes — GapRejection-specific:
        gap_min_atr_pct:      Minimum gap size as fraction of ATR.
        gap_fill_threshold_pct: How much of gap must fill to confirm rejection.
        gap_rejection_bars:   Bars to confirm rejection at gap edge.

    Attributes — PivotPoints-specific:
        pivot_formula:        ``"classic"``, ``"woodie"``, or ``"camarilla"``.

    Attributes — Fibonacci-specific:
        fib_upper:            Upper retracement level (0.618).
        fib_lower:            Lower retracement level (0.382).
        fib_swing_lookback:   Bars to search for prior swing H/L.
        fib_min_swing_atr_mult: Min swing size in ATR multiples.

    Attributes — Misc:
        extra:                Reserved dict for future per-type parameters.
    """

    # --- Identity ---
    breakout_type: BreakoutType
    breakout_type_ord: float
    label: str = ""

    # --- Range shape ---
    or_duration_minutes: int = 0
    lookback_days: int = 0

    # --- Range quality gates ---
    min_range_atr_ratio: float = 0.05
    max_range_atr_ratio: float = 2.0

    # --- TP / SL / trailing ---
    tp1_atr_mult: float = 2.0
    tp2_atr_mult: float = 3.0
    tp3_atr_mult: float = 4.5
    sl_atr_mult: float = 1.5
    enable_ema_trail_after_tp2: bool = True
    ema_trail_period: int = 9

    # --- Visual / chart rendering ---
    box_style: str = "gold_dashed"
    box_border_rgba: tuple[int, int, int, int] = (255, 215, 0, 100)
    box_fill_rgba: tuple[int, int, int, int] = (255, 215, 0, 30)
    description: str = ""

    # --- Detection thresholds (Phase 1B — merged from engine RangeConfig) ---
    atr_period: int = 14
    atr_multiplier: float = 0.5
    min_depth_atr_pct: float = 0.15
    min_body_ratio: float = 0.55
    min_bars: int = 3

    # --- IB-specific ---
    ib_duration_minutes: int = 60
    ib_start_time: Any = None  # dt_time(9, 30) — set per-config to avoid import

    # --- PDR-specific ---
    pdr_session_start: Any = None  # dt_time(18, 0)
    pdr_session_end: Any = None  # dt_time(17, 0)

    # --- Consolidation / Squeeze ---
    squeeze_lookback: int = 20
    squeeze_atr_mult: float = 1.5
    squeeze_bb_period: int = 20
    squeeze_bb_std: float = 2.0
    squeeze_min_bars: int = 5

    # --- Weekly-specific ---
    weekly_lookback_days: int = 5

    # --- Monthly-specific ---
    monthly_lookback_days: int = 20

    # --- Asian-specific ---
    asian_start_time: Any = None  # dt_time(19, 0)
    asian_end_time: Any = None  # dt_time(2, 0)

    # --- BollingerSqueeze-specific (BB inside KC) ---
    bbsqueeze_bb_period: int = 20
    bbsqueeze_bb_std: float = 2.0
    bbsqueeze_kc_period: int = 20
    bbsqueeze_kc_atr_mult: float = 1.5
    bbsqueeze_min_squeeze_bars: int = 6

    # --- ValueArea-specific ---
    va_value_area_pct: float = 0.70
    va_n_bins: int = 50

    # --- InsideDay-specific ---
    inside_min_compression: float = 0.30
    inside_max_compression: float = 0.85

    # --- GapRejection-specific ---
    gap_min_atr_pct: float = 0.15
    gap_fill_threshold_pct: float = 0.50
    gap_rejection_bars: int = 3

    # --- PivotPoints-specific ---
    pivot_formula: str = "classic"

    # --- Fibonacci-specific ---
    fib_upper: float = 0.618
    fib_lower: float = 0.382
    fib_swing_lookback: int = 100
    fib_min_swing_atr_mult: float = 1.5

    # --- Misc ---
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper to compute normalised ordinal
# ---------------------------------------------------------------------------


def _ord(bt: BreakoutType) -> float:
    """Normalise a BreakoutType value to [0.0, 1.0]."""
    return bt.value / (_NUM_BREAKOUT_TYPES - 1)


# ---------------------------------------------------------------------------
# Canonical configs — one per BreakoutType
# ---------------------------------------------------------------------------

_ORB_CONFIG = RangeConfig(
    breakout_type=BreakoutType.ORB,
    breakout_type_ord=_ord(BreakoutType.ORB),  # 0.000
    label="Opening Range Breakout",
    or_duration_minutes=30,
    lookback_days=0,
    min_range_atr_ratio=0.15,
    max_range_atr_ratio=2.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    tp3_atr_mult=4.5,
    sl_atr_mult=1.5,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    box_style="gold_dashed",
    box_border_rgba=(255, 215, 0, 100),
    box_fill_rgba=(255, 215, 0, 30),
    description="Opening Range Breakout — 30-min OR at session open",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.5,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.55,
    min_bars=5,
    extra={
        "sessions": ["us", "london", "london_ny", "frankfurt"],
        "min_or_bars": 20,
    },
)

_PREV_DAY_CONFIG = RangeConfig(
    breakout_type=BreakoutType.PrevDay,
    breakout_type_ord=_ord(BreakoutType.PrevDay),  # 0.0833
    label="Previous Day Range",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.20,
    max_range_atr_ratio=3.0,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.5,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    box_style="silver_solid",
    box_border_rgba=(192, 192, 192, 120),
    box_fill_rgba=(192, 192, 192, 20),
    description="Previous Day High/Low — prior session range as support/resistance",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.4,
    min_depth_atr_pct=0.20,
    min_body_ratio=0.50,
    min_bars=1,
    pdr_session_start=dt_time(18, 0),
    pdr_session_end=dt_time(17, 0),
    extra={
        "sessions": ["all"],
        "use_globex_high_low": True,
    },
)

_IB_CONFIG = RangeConfig(
    breakout_type=BreakoutType.InitialBalance,
    breakout_type_ord=_ord(BreakoutType.InitialBalance),  # 0.1667
    label="Initial Balance",
    or_duration_minutes=60,
    lookback_days=0,
    min_range_atr_ratio=0.10,
    max_range_atr_ratio=2.0,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    box_style="blue_dashed",
    box_border_rgba=(0, 229, 255, 110),
    box_fill_rgba=(0, 229, 255, 18),
    description="Initial Balance — first 60 min of the primary session (auction theory IB)",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.5,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.52,
    min_bars=10,
    ib_duration_minutes=60,
    ib_start_time=dt_time(9, 30),
    extra={
        "sessions": ["us"],
        "ib_start_hour": 9,
        "ib_start_minute": 30,
        "ib_end_hour": 10,
        "ib_end_minute": 30,
    },
)

_CONSOLIDATION_CONFIG = RangeConfig(
    breakout_type=BreakoutType.Consolidation,
    breakout_type_ord=_ord(BreakoutType.Consolidation),  # 0.25
    label="Consolidation/Squeeze",
    or_duration_minutes=0,
    lookback_days=0,
    min_range_atr_ratio=0.05,
    max_range_atr_ratio=0.80,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.5,
    tp3_atr_mult=5.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    box_style="purple_solid",
    box_border_rgba=(147, 0, 211, 130),
    box_fill_rgba=(147, 0, 211, 22),
    description="Detected Consolidation — auto-identified tight range / inside-bar cluster",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.6,
    min_depth_atr_pct=0.18,
    min_body_ratio=0.55,
    min_bars=5,
    squeeze_lookback=20,
    squeeze_atr_mult=1.5,
    squeeze_bb_period=20,
    squeeze_bb_std=2.0,
    squeeze_min_bars=5,
    extra={
        "sessions": ["all"],
        "min_consolidation_bars": 12,
        "max_range_pct": 0.003,
    },
)

# ---------------------------------------------------------------------------
# New researched range breakout types
# ---------------------------------------------------------------------------

_WEEKLY_CONFIG = RangeConfig(
    breakout_type=BreakoutType.Weekly,
    breakout_type_ord=_ord(BreakoutType.Weekly),  # 0.3333
    label="Weekly Range",
    or_duration_minutes=0,
    lookback_days=5,  # one trading week look-back
    min_range_atr_ratio=1.0,
    max_range_atr_ratio=8.0,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.5,
    tp3_atr_mult=5.0,
    sl_atr_mult=1.5,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Teal solid — wide HTF range, distinct from daily
    box_style="teal_solid",
    box_border_rgba=(0, 128, 128, 120),
    box_fill_rgba=(0, 128, 128, 15),
    description="Weekly Range — prior week's high/low as support/resistance",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.5,
    min_depth_atr_pct=0.20,
    min_body_ratio=0.50,
    min_bars=1,
    weekly_lookback_days=5,
    extra={
        "sessions": ["us", "london"],
        "timeframe": "W",
        "range_source": "prior_week_hl",
    },
)

_MONTHLY_CONFIG = RangeConfig(
    breakout_type=BreakoutType.Monthly,
    breakout_type_ord=_ord(BreakoutType.Monthly),  # 0.4167
    label="Monthly Range",
    or_duration_minutes=0,
    lookback_days=20,  # approximately one trading month
    min_range_atr_ratio=2.0,
    max_range_atr_ratio=15.0,
    tp1_atr_mult=2.5,
    tp2_atr_mult=4.0,
    tp3_atr_mult=6.0,
    sl_atr_mult=2.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Dark orange solid — monthly is the biggest HTF range
    box_style="orange_solid",
    box_border_rgba=(255, 140, 0, 100),
    box_fill_rgba=(255, 140, 0, 12),
    description="Monthly Range — prior month's high/low as macro support/resistance",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.5,
    min_depth_atr_pct=0.20,
    min_body_ratio=0.50,
    min_bars=1,
    monthly_lookback_days=20,
    extra={
        "sessions": ["us", "london"],
        "timeframe": "M",
        "range_source": "prior_month_hl",
    },
)

_ASIAN_CONFIG = RangeConfig(
    breakout_type=BreakoutType.Asian,
    breakout_type_ord=_ord(BreakoutType.Asian),  # 0.5
    label="Asian Session Range",
    or_duration_minutes=420,  # 19:00–02:00 ET = 7 hours
    lookback_days=0,  # same day (overnight into current day)
    min_range_atr_ratio=0.20,
    max_range_atr_ratio=2.5,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Red dashed — Asian session range, visible against overnight bars
    box_style="red_dashed",
    box_border_rgba=(220, 20, 60, 110),
    box_fill_rgba=(220, 20, 60, 18),
    description="Asian Session Range — 19:00–02:00 ET high/low (Tokyo/Sydney overnight)",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.45,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.50,
    min_bars=10,
    asian_start_time=dt_time(19, 0),
    asian_end_time=dt_time(2, 0),
    extra={
        "sessions": ["london", "london_ny", "us"],
        "range_start_hour": 19,
        "range_start_minute": 0,
        "range_end_hour": 2,
        "range_end_minute": 0,
        "wraps_midnight": True,
    },
)

_BOLLINGER_SQUEEZE_CONFIG = RangeConfig(
    breakout_type=BreakoutType.BollingerSqueeze,
    breakout_type_ord=_ord(BreakoutType.BollingerSqueeze),  # 0.5833
    label="Bollinger Squeeze",
    or_duration_minutes=0,  # detected algorithmically
    lookback_days=0,
    min_range_atr_ratio=0.05,
    max_range_atr_ratio=0.90,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.5,
    tp3_atr_mult=5.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Magenta dashed — squeeze/expansion visual
    box_style="magenta_dashed",
    box_border_rgba=(255, 0, 255, 110),
    box_fill_rgba=(255, 0, 255, 15),
    description="Bollinger Squeeze — BB compression + Keltner inside → expansion breakout",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.6,
    min_depth_atr_pct=0.18,
    min_body_ratio=0.55,
    min_bars=6,
    bbsqueeze_bb_period=20,
    bbsqueeze_bb_std=2.0,
    bbsqueeze_kc_period=20,
    bbsqueeze_kc_atr_mult=1.5,
    bbsqueeze_min_squeeze_bars=6,
    extra={
        "sessions": ["all"],
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_atr_mult": 1.5,
        "min_squeeze_bars": 6,
    },
)

_VALUE_AREA_CONFIG = RangeConfig(
    breakout_type=BreakoutType.ValueArea,
    breakout_type_ord=_ord(BreakoutType.ValueArea),  # 0.6667
    label="Value Area",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.30,
    max_range_atr_ratio=3.0,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Yellow-green solid — volume profile / auction theory colour
    box_style="olive_solid",
    box_border_rgba=(128, 128, 0, 120),
    box_fill_rgba=(128, 128, 0, 18),
    description="Value Area — prior session VAH/VAL from volume profile (70% rule)",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.4,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.50,
    min_bars=1,
    va_value_area_pct=0.70,
    va_n_bins=50,
    extra={
        "sessions": ["us", "london"],
        "value_area_pct": 0.70,
        "tick_size": 0.25,
        "range_source": "volume_profile",
    },
)

_INSIDE_DAY_CONFIG = RangeConfig(
    breakout_type=BreakoutType.InsideDay,
    breakout_type_ord=_ord(BreakoutType.InsideDay),  # 0.75
    label="Inside Day",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.30,
    max_range_atr_ratio=1.5,
    tp1_atr_mult=2.0,
    tp2_atr_mult=3.0,
    tp3_atr_mult=4.5,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Lime dashed — compressed day, spring-loaded
    box_style="lime_dashed",
    box_border_rgba=(50, 205, 50, 120),
    box_fill_rgba=(50, 205, 50, 18),
    description="Inside Day — today's range inside yesterday's → coiled energy breakout",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.45,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.52,
    min_bars=1,
    inside_min_compression=0.30,
    inside_max_compression=0.85,
    extra={
        "sessions": ["us", "london"],
        "require_full_containment": True,
        "min_compression_ratio": 0.30,
        "max_compression_ratio": 0.85,
    },
)

_GAP_REJECTION_CONFIG = RangeConfig(
    breakout_type=BreakoutType.GapRejection,
    breakout_type_ord=_ord(BreakoutType.GapRejection),  # 0.8333
    label="Gap Rejection",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.10,
    max_range_atr_ratio=2.0,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=3.5,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Coral solid — gap-fill visual
    box_style="coral_solid",
    box_border_rgba=(255, 127, 80, 120),
    box_fill_rgba=(255, 127, 80, 18),
    description="Gap Rejection — overnight gap fill or rejection at gap edge",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.4,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.50,
    min_bars=1,
    gap_min_atr_pct=0.15,
    gap_fill_threshold_pct=0.50,
    gap_rejection_bars=3,
    extra={
        "sessions": ["us", "london"],
        "min_gap_atr_pct": 0.15,
        "gap_fill_threshold_pct": 0.50,
        "rejection_confirmation_bars": 3,
    },
)

_PIVOT_POINTS_CONFIG = RangeConfig(
    breakout_type=BreakoutType.PivotPoints,
    breakout_type_ord=_ord(BreakoutType.PivotPoints),  # 0.9167
    label="Pivot Points",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.20,
    max_range_atr_ratio=3.5,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Steel blue dashed — classic pivot visual
    box_style="steel_dashed",
    box_border_rgba=(70, 130, 180, 120),
    box_fill_rgba=(70, 130, 180, 18),
    description="Pivot Points — classic floor pivot R1/S1 as the breakout range",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.4,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.50,
    min_bars=1,
    pivot_formula="classic",
    extra={
        "sessions": ["us", "london"],
        "pivot_formula": "classic",  # "classic", "woodie", "camarilla"
        "range_levels": ["S1", "R1"],
        "use_prior_day_hlc": True,
    },
)

_FIBONACCI_CONFIG = RangeConfig(
    breakout_type=BreakoutType.Fibonacci,
    breakout_type_ord=_ord(BreakoutType.Fibonacci),  # 1.0
    label="Fibonacci Retracement",
    or_duration_minutes=0,
    lookback_days=1,
    min_range_atr_ratio=0.15,
    max_range_atr_ratio=2.5,
    tp1_atr_mult=1.5,
    tp2_atr_mult=2.5,
    tp3_atr_mult=4.0,
    sl_atr_mult=1.0,
    enable_ema_trail_after_tp2=True,
    ema_trail_period=9,
    # Gold solid — Fibonacci golden ratio visual
    box_style="amber_solid",
    box_border_rgba=(255, 191, 0, 120),
    box_fill_rgba=(255, 191, 0, 18),
    description="Fibonacci — 38.2%–61.8% retracement zone of prior swing as breakout range",
    # Detection thresholds
    atr_period=14,
    atr_multiplier=0.4,
    min_depth_atr_pct=0.15,
    min_body_ratio=0.50,
    min_bars=1,
    fib_upper=0.618,
    fib_lower=0.382,
    fib_swing_lookback=100,
    fib_min_swing_atr_mult=1.5,
    extra={
        "sessions": ["us", "london"],
        "fib_upper": 0.618,
        "fib_lower": 0.382,
        "swing_lookback_bars": 100,
        "min_swing_atr_mult": 1.5,
    },
)


# ---------------------------------------------------------------------------
# Registry & lookup
# ---------------------------------------------------------------------------

_RANGE_CONFIG_REGISTRY: dict[BreakoutType, RangeConfig] = {
    BreakoutType.ORB: _ORB_CONFIG,
    BreakoutType.PrevDay: _PREV_DAY_CONFIG,
    BreakoutType.InitialBalance: _IB_CONFIG,
    BreakoutType.Consolidation: _CONSOLIDATION_CONFIG,
    BreakoutType.Weekly: _WEEKLY_CONFIG,
    BreakoutType.Monthly: _MONTHLY_CONFIG,
    BreakoutType.Asian: _ASIAN_CONFIG,
    BreakoutType.BollingerSqueeze: _BOLLINGER_SQUEEZE_CONFIG,
    BreakoutType.ValueArea: _VALUE_AREA_CONFIG,
    BreakoutType.InsideDay: _INSIDE_DAY_CONFIG,
    BreakoutType.GapRejection: _GAP_REJECTION_CONFIG,
    BreakoutType.PivotPoints: _PIVOT_POINTS_CONFIG,
    BreakoutType.Fibonacci: _FIBONACCI_CONFIG,
}


def get_range_config(breakout_type: BreakoutType) -> RangeConfig:
    """Return the canonical ``RangeConfig`` for *breakout_type*.

    This is the single authoritative lookup used by the dataset generator,
    CNN training loop, and external consumers.

    Args:
        breakout_type: A ``BreakoutType`` enum member.

    Returns:
        ``RangeConfig`` instance (frozen dataclass — do not mutate).

    Raises:
        KeyError: If *breakout_type* has no registered config (should never
                  happen for the canonical enum values).

    Example::

        >>> from lib.core.breakout_types import BreakoutType, get_range_config
        >>> cfg = get_range_config(BreakoutType.ORB)
        >>> cfg.box_style
        'gold_dashed'
        >>> cfg.breakout_type_ord
        0.0
        >>> cfg.tp3_atr_mult
        4.5
    """
    return _RANGE_CONFIG_REGISTRY[breakout_type]


def all_range_configs() -> list[RangeConfig]:
    """Return all ``RangeConfig`` objects in ``BreakoutType`` ordinal order.

    Useful for iterating over all types during dataset generation::

        for cfg in all_range_configs():
            print(cfg.breakout_type.name, cfg.breakout_type_ord)  # noqa: T201
    """
    return [_RANGE_CONFIG_REGISTRY[bt] for bt in BreakoutType]


def breakout_type_ord(breakout_type: BreakoutType) -> float:
    """Return the normalised ordinal ``[0, 1]`` for *breakout_type*.

    Convenience wrapper so callers don't need to import ``get_range_config``::

        >>> from lib.core.breakout_types import breakout_type_ord
        >>> breakout_type_ord(BreakoutType.PrevDay)
        0.08333333333333333
    """
    return get_range_config(breakout_type).breakout_type_ord


def breakout_type_from_ord(ord_value: float, tol: float = 0.01) -> BreakoutType:
    """Reverse-lookup a ``BreakoutType`` from its normalised ordinal.

    Useful when deserialising CSV rows where only the float is stored.

    Args:
        ord_value: Normalised ordinal value (e.g. ``0.0``, ``0.0833``, …).
        tol:       Floating-point tolerance for the comparison.

    Returns:
        Matching ``BreakoutType``.

    Raises:
        ValueError: If no type matches within tolerance.
    """
    for bt, cfg in _RANGE_CONFIG_REGISTRY.items():
        if abs(cfg.breakout_type_ord - ord_value) <= tol:
            return bt
    raise ValueError(
        f"No BreakoutType matches ord_value={ord_value} (tol={tol}). "
        f"Valid values: {[c.breakout_type_ord for c in _RANGE_CONFIG_REGISTRY.values()]}"
    )


def breakout_type_from_name(name: str) -> BreakoutType:
    """Look up a ``BreakoutType`` by case-insensitive name.

    Args:
        name: e.g. ``"orb"``, ``"ORB"``, ``"PrevDay"``, ``"initialbalance"``,
              ``"weekly"``, ``"bollingersqueeze"``.

    Returns:
        Matching ``BreakoutType``.

    Raises:
        ValueError: If *name* does not match any ``BreakoutType``.
    """
    _name = name.strip().lower()
    _map = {bt.name.lower(): bt for bt in BreakoutType}
    if _name not in _map:
        raise ValueError(f"Unknown BreakoutType name {name!r}. Valid names: {list(_map.keys())}")
    return _map[_name]


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

# Original four exchange-based types (backward compatible)
EXCHANGE_BREAKOUT_TYPES: list[BreakoutType] = [
    BreakoutType.ORB,
    BreakoutType.PrevDay,
    BreakoutType.InitialBalance,
    BreakoutType.Consolidation,
]

# New researched non-exchange types
RESEARCHED_BREAKOUT_TYPES: list[BreakoutType] = [
    BreakoutType.Weekly,
    BreakoutType.Monthly,
    BreakoutType.Asian,
    BreakoutType.BollingerSqueeze,
    BreakoutType.ValueArea,
    BreakoutType.InsideDay,
    BreakoutType.GapRejection,
    BreakoutType.PivotPoints,
    BreakoutType.Fibonacci,
]

# Types that use HTF (higher time-frame) data — weekly/monthly bars
HTF_BREAKOUT_TYPES: list[BreakoutType] = [
    BreakoutType.Weekly,
    BreakoutType.Monthly,
]

# Types that detect range algorithmically (no fixed time window)
DETECTED_BREAKOUT_TYPES: list[BreakoutType] = [
    BreakoutType.Consolidation,
    BreakoutType.BollingerSqueeze,
    BreakoutType.InsideDay,
]


def types_with_ema_trailing() -> list[BreakoutType]:
    """Return breakout types where EMA trailing after TP2 is enabled."""
    return [bt for bt, cfg in _RANGE_CONFIG_REGISTRY.items() if cfg.enable_ema_trail_after_tp2]


def types_with_tp3() -> list[BreakoutType]:
    """Return breakout types that have a non-zero TP3 target."""
    return [bt for bt, cfg in _RANGE_CONFIG_REGISTRY.items() if cfg.tp3_atr_mult > 0]


# ---------------------------------------------------------------------------
# Serialisation helpers (for feature_contract.json)
# ---------------------------------------------------------------------------


def to_feature_contract_dict() -> dict[str, Any]:
    """Serialise all ``RangeConfig`` objects to a dict for ``feature_contract.json``.

    The returned dict is inserted under the ``"breakout_types"`` key in the
    contract so consumers can verify the ordinal mapping at load time.

    Example output::

        {
          "ORB":              {"ordinal": 0,  "breakout_type_ord": 0.0,    "box_style": "gold_dashed", ...},
          "PrevDay":          {"ordinal": 1,  "breakout_type_ord": 0.083,  "box_style": "silver_solid", ...},
          ...
          "Fibonacci":        {"ordinal": 12, "breakout_type_ord": 1.0,    "box_style": "amber_solid", ...},
        }
    """
    return {
        bt.name: {
            "ordinal": int(bt),
            "breakout_type_ord": round(cfg.breakout_type_ord, 6),
            "label": cfg.label,
            "or_duration_minutes": cfg.or_duration_minutes,
            "lookback_days": cfg.lookback_days,
            "tp1_atr_mult": cfg.tp1_atr_mult,
            "tp2_atr_mult": cfg.tp2_atr_mult,
            "tp3_atr_mult": cfg.tp3_atr_mult,
            "sl_atr_mult": cfg.sl_atr_mult,
            "enable_ema_trail_after_tp2": cfg.enable_ema_trail_after_tp2,
            "ema_trail_period": cfg.ema_trail_period,
            "box_style": cfg.box_style,
            "box_border_rgba": list(cfg.box_border_rgba),
            "box_fill_rgba": list(cfg.box_fill_rgba),
            "description": cfg.description,
            # Detection thresholds (Phase 1B)
            "atr_period": cfg.atr_period,
            "atr_multiplier": cfg.atr_multiplier,
            "min_depth_atr_pct": cfg.min_depth_atr_pct,
            "min_body_ratio": cfg.min_body_ratio,
            "min_bars": cfg.min_bars,
        }
        for bt, cfg in _RANGE_CONFIG_REGISTRY.items()
    }
