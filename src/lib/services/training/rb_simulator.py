"""
RB Simulator — Auto-Labeling Engine for CNN Training
=====================================================
Replays historical 1-minute bar data through the exact same Range Breakout
detection + Bridge-style bracket logic used in live trading, then assigns
ground-truth labels suitable for supervised learning.

Covers all 13 breakout types: ORB, PrevDay, InitialBalance, Consolidation,
Weekly, Monthly, Asian, BollingerSqueeze, ValueArea, InsideDay,
GapRejection, PivotPoints, Fibonacci.

Labels produced:
  - ``good_long``   — Long breakout hit TP1 before SL within holding window.
  - ``good_short``  — Short breakout hit TP1 before SL within holding window.
  - ``bad_long``    — Long breakout hit SL first (or timed out without TP).
  - ``bad_short``   — Short breakout hit SL first (or timed out without TP).
  - ``no_trade``    — No valid breakout was detected in the window.

The simulator is intentionally conservative — it mirrors the original Ruby bridge
bracket sizing (ATR-based SL/TP) so that CNN training data reflects *real*
execution outcomes, not theoretical ones.

Public API:
    from lib.services.training.rb_simulator import (
        simulate_rb_outcome,
        simulate_batch,
        simulate_batch_prev_day,
        simulate_batch_ib,
        simulate_batch_consolidation,
        RBSimResult,
        BracketConfig,
    )

    result = simulate_rb_outcome(bars_1m, symbol="MGC")
    #  result.label        → "good_long"
    #  result.direction     → "LONG"
    #  result.entry         → 2345.60
    #  result.to_dict()     → JSON-friendly dict for labels.csv

    # PrevDay batch — one result per calendar day in bars_1m
    results = simulate_batch_prev_day(bars_1m, symbol="MGC")

    # InitialBalance batch — one result per RTH session day
    results = simulate_batch_ib(bars_1m, symbol="MGC")

    # Consolidation batch — multiple tight-range breakout candidates per day
    results = simulate_batch_consolidation(bars_1m, symbol="MGC")

Backward compatibility:
    ``ORBSimResult`` and ``simulate_orb_outcome`` are kept as aliases so
    existing callers (``dataset_generator.py``) continue to work unchanged
    until they are updated to use the new names.

Design:
  - Pure functions — no Redis, no side-effects, fully testable.
  - Parameterised via BracketConfig so you can sweep SL/TP ratios.
  - Uses the same ATR computation as orb.py for consistency.
  - Thread-safe: no shared mutable state.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("training.rb_simulator")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BracketConfig:
    """Bracket parameters matching the original Ruby bridge risk logic.

    These map directly to the bridge properties:
      - sl_atr_mult → stop-loss ATR multiplier  (default 1.5)
      - tp1_atr_mult → target 1 ATR multiplier   (default 2.0)
      - tp2_atr_mult → target 2 ATR multiplier   (default 3.0, optional)
      - tp3_atr_mult → target 3 ATR multiplier   (default 4.5, optional)
      - max_hold_bars → maximum bars to hold before labelling timeout
    """

    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 2.0
    tp2_atr_mult: float = 3.0
    tp3_atr_mult: float = 4.5  # extended target; 0 = disabled
    max_hold_bars: int = 120  # ~2 hours of 1-min bars
    atr_period: int = 14

    # EMA trailing after TP2
    enable_ema_trail_after_tp2: bool = True
    ema_trail_period: int = 9

    # Opening range parameters
    or_start: dt_time = dt_time(9, 30)
    or_end: dt_time = dt_time(10, 0)
    or_minutes: int = 30
    min_or_bars: int = 5

    # Pre-market window (for NR7 / pm range extraction)
    pm_start: dt_time = dt_time(0, 0)
    pm_end: dt_time = dt_time(8, 20)

    # Breakout confirmation: require close beyond ORB level (not just wick)
    require_close_break: bool = True

    # Minimum ORB range (in ATR fraction) to avoid tiny-range noise
    min_or_range_atr_frac: float = 0.3


DEFAULT_BRACKET = BracketConfig()

# London Open: OR 03:00–03:30 ET, premarket 00:00–03:00
LONDON_BRACKET = BracketConfig(
    or_start=dt_time(3, 0),
    or_end=dt_time(3, 30),
    pm_end=dt_time(3, 0),
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ORBSimResult:
    """Result of a single ORB simulation."""

    # Label
    label: str = "no_trade"  # good_long, bad_long, good_short, bad_short, no_trade

    # Trade details
    symbol: str = ""
    direction: str = ""  # "LONG", "SHORT", or ""
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0  # extended target (0 = disabled)

    # ORB details
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr: float = 0.0

    # Quality heuristic (0–100)
    quality_pct: int = 0

    # Outcome details
    outcome: str = ""  # "tp1_hit", "sl_hit", "timeout", "no_breakout", "tp3_hit", "ema_trail_exit", etc.
    pnl_r: float = 0.0  # P&L in R-multiples (1.0 = 1R win, -1.0 = 1R loss)
    hold_bars: int = 0  # how many bars the trade was held
    breakout_bar_idx: int = -1

    # TP3 / EMA trailing outcome
    hit_tp2: bool = False  # whether TP2 was reached (enables trailing)
    hit_tp3: bool = False  # whether TP3 was reached
    ema_trail_exit: bool = False  # whether EMA trail stopped out the trade
    trail_exit_price: float = 0.0  # price at which EMA trail exited

    # Timing
    or_start_time: str = ""
    breakout_time: str = ""
    exit_time: str = ""
    simulated_at: str = ""

    # Pre-market context
    pm_high: float = 0.0
    pm_low: float = 0.0

    # NR7 flag (narrow range day)
    nr7: bool = False

    # Volume context
    breakout_volume_ratio: float = 0.0  # breakout bar vol / avg vol

    # CVD & session context (for CNN tabular features)
    cvd_delta: float = 0.0  # cumulative volume delta from OR to breakout (-1 to 1)
    london_overlap_flag: float = 0.0  # 1.0 if breakout in 08:00–09:00 ET overlap

    # Dataset generation metadata — set by the dataset generator after
    # simulation so _build_row() can read breakout type and session context.
    _session_key: str = ""  # e.g. "london", "us", "cme" — set by dataset_generator
    _breakout_type: int = 0  # BreakoutType ordinal — set by dataset_generator

    # Window provenance — set by simulate_batch() so the dataset generator
    # can recover the exact slice of bars_1m used for this simulation.
    _window_offset: int = -1  # start index into the original bars_1m
    _window_size: int = 0  # number of bars in the window

    # Error
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON/CSV-friendly dict."""
        return {
            "label": self.label,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": round(self.entry, 6),
            "sl": round(self.sl, 6),
            "tp1": round(self.tp1, 6),
            "tp2": round(self.tp2, 6),
            "tp3": round(self.tp3, 6),
            "or_high": round(self.or_high, 6),
            "or_low": round(self.or_low, 6),
            "or_range": round(self.or_range, 6),
            "atr": round(self.atr, 6),
            "quality_pct": self.quality_pct,
            "outcome": self.outcome,
            "pnl_r": round(self.pnl_r, 3),
            "hold_bars": self.hold_bars,
            "hit_tp2": self.hit_tp2,
            "hit_tp3": self.hit_tp3,
            "ema_trail_exit": self.ema_trail_exit,
            "trail_exit_price": round(self.trail_exit_price, 6),
            "or_start_time": self.or_start_time,
            "breakout_time": self.breakout_time,
            "exit_time": self.exit_time,
            "pm_high": round(self.pm_high, 6),
            "pm_low": round(self.pm_low, 6),
            "nr7": self.nr7,
            "breakout_volume_ratio": round(self.breakout_volume_ratio, 3),
            "cvd_delta": round(self.cvd_delta, 4),
            "london_overlap_flag": self.london_overlap_flag,
            "error": self.error,
        }

    @property
    def is_winner(self) -> bool:
        return self.label.startswith("good_")

    @property
    def is_trade(self) -> bool:
        return self.label != "no_trade"


# ---------------------------------------------------------------------------
# ATR computation (matches orb.py and volatility.py)
# ---------------------------------------------------------------------------


def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """Compute ATR using simple moving average of True Range.

    Matches the implementation in orb.py for consistency.
    Returns 0.0 if insufficient data.
    """
    n = len(closes)
    if n < 2:
        return float(highs[0] - lows[0]) if n > 0 else 0.0

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    if n >= period:
        return float(np.mean(tr[-period:]))
    return float(np.mean(tr))


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _localize_to_est(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is tz-aware in Eastern Time."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    dti = pd.DatetimeIndex(df.index)
    if dti.tz is not None:
        if str(dti.tz) != str(_EST):
            df = df.tz_convert(_EST)
    else:
        with contextlib.suppress(Exception):
            df.index = dti.tz_localize(_EST)
    return df


def _safe_time(idx) -> dt_time | None:
    """Extract time from an index entry, returning None on failure."""
    try:
        return idx.time() if hasattr(idx, "time") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_orb_outcome(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
) -> ORBSimResult:
    """Simulate an ORB trade on a window of 1-minute bars.

    This replays the exact logic used by the Ruby indicator logic (ORB detection)
    and the original Ruby bridge (bracket sizing) to produce a ground-truth label.

    Algorithm:
      1. Identify the opening range (OR) from bars in [or_start, or_end).
      2. Compute ATR from all available bars.
      3. Scan post-OR bars for the first close beyond OR high/low.
      4. Set entry, SL, TP1, TP2 using bridge-style ATR multiples.
      5. Walk forward bar-by-bar to determine outcome.
      6. Assign label: good_long, bad_long, good_short, bad_short, no_trade.

    Args:
        bars_1m: 1-minute OHLCV DataFrame covering at least the OR period
                 and subsequent trading hours.  DatetimeIndex preferred.
        symbol: Instrument symbol for labelling.
        config: BracketConfig with SL/TP/holding parameters.
        bars_daily: Optional daily bars for NR7 detection (at least 7 rows).

    Returns:
        ORBSimResult with the label, trade details, and quality heuristic.
    """
    cfg = config or DEFAULT_BRACKET
    result = ORBSimResult(symbol=symbol, simulated_at=datetime.now(_EST).isoformat())

    # --- Validate input ---
    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data provided"
        return result

    required = {"High", "Low", "Close"}
    missing = required - set(bars_1m.columns)
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(bars_1m) < cfg.min_or_bars + 10:
        result.error = f"Insufficient bars ({len(bars_1m)})"
        return result

    # Localise index to Eastern Time
    df = _localize_to_est(bars_1m)
    df = df.sort_index()

    # Cast to float
    highs = np.asarray(df["High"].astype(float).values)
    lows = np.asarray(df["Low"].astype(float).values)
    closes = np.asarray(df["Close"].astype(float).values)
    has_volume = "Volume" in df.columns
    volumes = np.asarray(df["Volume"].astype(float).values) if has_volume else np.ones(len(df))
    has_open = "Open" in df.columns
    opens = np.asarray(df["Open"].astype(float).values) if has_open else closes.copy()

    # --- Step 1: Identify Opening Range ---
    try:
        times = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
    except Exception:
        times = None

    if times is None:
        result.error = "Cannot extract time from index"
        return result

    or_mask = (times >= cfg.or_start) & (times < cfg.or_end)
    or_indices = np.where(or_mask)[0]

    if len(or_indices) < cfg.min_or_bars:
        result.error = f"Only {len(or_indices)} bars in OR window (need {cfg.min_or_bars})"
        result.outcome = "insufficient_or_bars"
        return result

    or_high = float(np.max(highs[or_indices]))
    or_low = float(np.min(lows[or_indices]))
    or_range = or_high - or_low

    result.or_high = or_high
    result.or_low = or_low
    result.or_range = or_range

    with contextlib.suppress(Exception):
        result.or_start_time = str(df.index[or_indices[0]])

    if or_high <= 0 or or_low <= 0 or or_high <= or_low:
        result.error = f"Invalid OR: high={or_high}, low={or_low}"
        result.outcome = "invalid_or"
        return result

    # --- Step 2: Compute ATR ---
    atr = _compute_atr(np.asarray(highs), np.asarray(lows), np.asarray(closes), period=cfg.atr_period)
    result.atr = atr

    if atr <= 0:
        result.error = "ATR is zero"
        result.outcome = "zero_atr"
        return result

    # Check minimum OR range
    if or_range < atr * cfg.min_or_range_atr_frac:
        result.error = f"OR range {or_range:.4f} < {cfg.min_or_range_atr_frac}x ATR ({atr:.4f})"
        result.outcome = "or_too_narrow"
        return result

    # --- Pre-market range ---
    pm_mask = (times >= cfg.pm_start) & (times < cfg.pm_end)
    pm_indices = np.where(pm_mask)[0]
    if len(pm_indices) > 0:
        result.pm_high = float(np.max(highs[pm_indices]))
        result.pm_low = float(np.min(lows[pm_indices]))

    # --- NR7 detection ---
    if bars_daily is not None and len(bars_daily) >= 7:
        try:
            d_highs = np.asarray(bars_daily["High"].astype(float).values[-7:])
            d_lows = np.asarray(bars_daily["Low"].astype(float).values[-7:])
            daily_ranges = d_highs - d_lows
            today_range = daily_ranges[-1]
            result.nr7 = bool(today_range <= np.min(daily_ranges))
        except Exception:
            pass

    # --- Step 3: Scan for breakout ---
    # Post-OR bars: everything after or_end
    post_or_mask = times >= cfg.or_end
    post_or_indices = np.where(post_or_mask)[0]

    if len(post_or_indices) == 0:
        result.outcome = "no_post_or_bars"
        return result

    direction: str | None = None
    breakout_idx: int | None = None
    entry_price: float = 0.0
    bar_val: float = 0.0
    bar_val_high: float = 0.0
    bar_val_low: float = 0.0

    for idx in post_or_indices:
        if cfg.require_close_break:
            bar_val = float(closes[idx])
        else:
            bar_val_high = float(highs[idx])
            bar_val_low = float(lows[idx])

        # Long breakout: bar breaks above OR high
        if cfg.require_close_break:
            if bar_val > or_high:
                direction = "LONG"
                # Entry is the worse of OR high and bar open (simulate fill)
                entry_price = max(or_high, float(opens[idx]))
                breakout_idx = idx
                break
        else:
            if bar_val_high > or_high:
                direction = "LONG"
                entry_price = max(or_high, float(opens[idx]))
                breakout_idx = idx
                break

        # Short breakout: bar breaks below OR low
        if cfg.require_close_break:
            if bar_val < or_low:
                direction = "SHORT"
                entry_price = min(or_low, float(opens[idx]))
                breakout_idx = idx
                break
        else:
            if bar_val_low < or_low:
                direction = "SHORT"
                entry_price = min(or_low, float(opens[idx]))
                breakout_idx = idx
                break

    if direction is None or breakout_idx is None:
        result.outcome = "no_breakout"
        return result

    result.direction = direction
    result.entry = entry_price
    result.breakout_bar_idx = breakout_idx

    with contextlib.suppress(Exception):
        result.breakout_time = str(df.index[breakout_idx])

    # Volume ratio at breakout bar
    avg_vol = float(np.mean(np.asarray(volumes[max(0, breakout_idx - 20) : breakout_idx]))) if breakout_idx > 0 else 1.0
    if avg_vol > 0:
        result.breakout_volume_ratio = float(volumes[breakout_idx] / avg_vol)

    # --- Real CVD delta (cumulative volume delta from OR start to breakout) ---
    or_start_idx = int(or_indices[0])
    cvd = 0.0
    total_vol = 0.0
    for i in range(or_start_idx, breakout_idx + 1):
        vol = float(volumes[i])
        total_vol += vol
        if closes[i] > opens[i]:
            cvd += vol
        else:
            cvd -= vol
    result.cvd_delta = (cvd / total_vol) if total_vol > 0 else 0.0

    # --- London/NY overlap flag (08:00–09:00 ET is historically strongest) ---
    with contextlib.suppress(Exception):
        # Convert index element to string first so pd.Timestamp always gets
        # a well-typed scalar regardless of whether the index is DatetimeIndex,
        # Int64Index, or a plain object index.
        _ts_str: str = str(df.index[breakout_idx])
        breakout_et: pd.Timestamp = pd.Timestamp(_ts_str)  # type: ignore[assignment]
        _hour = int(getattr(breakout_et, "hour", 0))
        result.london_overlap_flag = 1.0 if 8 <= _hour <= 9 else 0.0

    # --- Step 4: Compute brackets (Bridge-style) ---
    sl_dist = atr * cfg.sl_atr_mult
    tp1_dist = atr * cfg.tp1_atr_mult
    tp2_dist = atr * cfg.tp2_atr_mult
    tp3_dist = atr * cfg.tp3_atr_mult if cfg.tp3_atr_mult > 0 else 0.0

    if direction == "LONG":
        result.sl = entry_price - sl_dist
        result.tp1 = entry_price + tp1_dist
        result.tp2 = entry_price + tp2_dist
        result.tp3 = entry_price + tp3_dist if tp3_dist > 0 else 0.0
    else:
        result.sl = entry_price + sl_dist
        result.tp1 = entry_price - tp1_dist
        result.tp2 = entry_price - tp2_dist
        result.tp3 = entry_price - tp3_dist if tp3_dist > 0 else 0.0

    # --- Step 5: Walk forward to determine outcome ---
    # Phase 1: SL vs TP1 (standard)
    # Phase 2: After TP2 hit, trail with EMA9 toward TP3
    max_exit_idx = min(breakout_idx + cfg.max_hold_bars, len(df))
    exit_bars = range(breakout_idx + 1, max_exit_idx)

    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False
    hit_sl = False
    ema_trail_exit = False
    exit_idx = breakout_idx
    trail_exit_price = 0.0

    for bar_idx in exit_bars:
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]

        if direction == "LONG":
            # Check SL first (conservative — if both hit in same bar, SL wins)
            if bar_low <= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_high >= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break
        else:  # SHORT
            if bar_high >= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_low <= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break

    # Phase 2: If TP1 hit, continue scanning for TP2 → TP3 / EMA trail
    if hit_tp1 and not hit_sl:
        # Walk forward from TP1 hit to find TP2
        tp2_scan_start = exit_idx + 1
        tp2_scan_end = min(tp2_scan_start + cfg.max_hold_bars, len(df))
        for bar_idx in range(tp2_scan_start, tp2_scan_end):
            bar_high = highs[bar_idx]
            bar_low = lows[bar_idx]
            if direction == "LONG":
                if bar_low <= result.sl:
                    break  # stopped out before TP2
                if bar_high >= result.tp2:
                    hit_tp2 = True
                    exit_idx = bar_idx
                    break
            else:
                if bar_high >= result.sl:
                    break
                if bar_low <= result.tp2:
                    hit_tp2 = True
                    exit_idx = bar_idx
                    break

        # Phase 3: After TP2, trail with EMA9 toward TP3
        if hit_tp2 and cfg.enable_ema_trail_after_tp2 and cfg.tp3_atr_mult > 0:
            ema_period = cfg.ema_trail_period
            alpha = 2.0 / (ema_period + 1)
            # Compute EMA on closes from breakout to current position
            ema_start = max(0, breakout_idx - ema_period * 2)
            ema_closes = closes[ema_start : exit_idx + 1]
            if len(ema_closes) >= ema_period:
                # Seed EMA with SMA
                ema = float(np.mean(ema_closes[:ema_period]))
                for ci in range(ema_period, len(ema_closes)):
                    ema = alpha * float(ema_closes[ci]) + (1.0 - alpha) * ema
            else:
                ema = float(closes[exit_idx])

            trail_scan_start = exit_idx + 1
            trail_scan_end = min(trail_scan_start + cfg.max_hold_bars, len(df))
            for bar_idx in range(trail_scan_start, trail_scan_end):
                bar_high = highs[bar_idx]
                bar_low = lows[bar_idx]
                bar_close = closes[bar_idx]

                # Update EMA
                ema = alpha * float(bar_close) + (1.0 - alpha) * ema

                # Check TP3 first
                if direction == "LONG":
                    if result.tp3 > 0 and bar_high >= result.tp3:
                        hit_tp3 = True
                        exit_idx = bar_idx
                        break
                    # EMA trail: exit if close crosses below EMA
                    if float(bar_close) < ema:
                        ema_trail_exit = True
                        trail_exit_price = float(bar_close)
                        exit_idx = bar_idx
                        break
                else:
                    if result.tp3 > 0 and bar_low <= result.tp3:
                        hit_tp3 = True
                        exit_idx = bar_idx
                        break
                    if float(bar_close) > ema:
                        ema_trail_exit = True
                        trail_exit_price = float(bar_close)
                        exit_idx = bar_idx
                        break

    result.hold_bars = exit_idx - breakout_idx
    result.hit_tp2 = hit_tp2
    result.hit_tp3 = hit_tp3
    result.ema_trail_exit = ema_trail_exit
    result.trail_exit_price = trail_exit_price

    with contextlib.suppress(Exception):
        result.exit_time = str(df.index[exit_idx])

    # --- Step 6: Assign label ---
    if hit_tp1:
        result.label = f"good_{direction.lower()}"
        if hit_tp3:
            result.outcome = "tp3_hit"
            result.pnl_r = cfg.tp3_atr_mult / cfg.sl_atr_mult
        elif ema_trail_exit:
            result.outcome = "ema_trail_exit"
            # R-multiple from trail exit
            if direction == "LONG":
                result.pnl_r = (trail_exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
            else:
                result.pnl_r = (entry_price - trail_exit_price) / sl_dist if sl_dist > 0 else 0.0
        elif hit_tp2:
            result.outcome = "tp2_hit"
            result.pnl_r = cfg.tp2_atr_mult / cfg.sl_atr_mult
        else:
            result.outcome = "tp1_hit"
            result.pnl_r = cfg.tp1_atr_mult / cfg.sl_atr_mult  # R-multiple
    elif hit_sl:
        result.label = f"bad_{direction.lower()}"
        result.outcome = "sl_hit"
        result.pnl_r = -1.0
    else:
        # Timeout — check if trade was in profit at expiry
        exit_close = closes[exit_idx] if exit_idx < len(closes) else entry_price
        if direction == "LONG":
            unrealised_r = (exit_close - entry_price) / sl_dist if sl_dist > 0 else 0.0
        else:
            unrealised_r = (entry_price - exit_close) / sl_dist if sl_dist > 0 else 0.0

        result.pnl_r = round(unrealised_r, 3)

        # Timeout with small gain is still "bad" for training purposes —
        # we only want clear TP1 hits as "good".
        result.label = f"bad_{direction.lower()}"
        result.outcome = "timeout"

    # --- Quality heuristic ---
    # Approximate the Ruby quality score: higher when OR is tight relative to
    # ATR, volume confirms, and NR7 is active.
    quality = 50.0  # base

    # OR range vs ATR: tighter OR + strong ATR = more coiled energy
    if or_range > 0 and atr > 0:
        range_ratio = atr / or_range
        quality += min(20.0, range_ratio * 10.0)

    # Volume confirmation
    if result.breakout_volume_ratio > 1.5:
        quality += 10.0
    elif result.breakout_volume_ratio > 1.2:
        quality += 5.0

    # NR7 bonus
    if result.nr7:
        quality += 15.0

    # Pre-market confluence
    if (
        direction == "LONG"
        and result.pm_high > 0
        and entry_price >= result.pm_high
        or direction == "SHORT"
        and result.pm_low > 0
        and entry_price <= result.pm_low
    ):
        quality += 5.0

    result.quality_pct = min(99, max(0, int(quality)))

    return result


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------


def simulate_batch(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    window_size: int = 240,
    step_size: int = 30,
    min_window_bars: int = 60,
) -> list[ORBSimResult]:
    """Run ORB simulation across sliding windows of bar data.

    This is the main entry point for the dataset generator.  It slides a
    window of ``window_size`` bars across the input data, stepping by
    ``step_size``, and simulates one ORB trade per window.

    This produces many training examples from a single day's data by
    varying the entry point within the session.

    Args:
        bars_1m: Full 1-minute OHLCV data (e.g. 60–120 trading days).
        symbol: Instrument symbol.
        config: BracketConfig for SL/TP/holding parameters.
        bars_daily: Daily bars for NR7 detection (optional).
        window_size: Number of 1-min bars per simulation window (default 240 = 4h).
        step_size: Step between windows (default 30 = 30 min).
        min_window_bars: Minimum bars in a window to attempt simulation.

    Returns:
        List of ORBSimResult (includes no_trade results for dataset balance).
    """
    cfg = config or DEFAULT_BRACKET
    results: list[ORBSimResult] = []

    if bars_1m is None or bars_1m.empty:
        return results

    n = len(bars_1m)

    for start in range(0, n - min_window_bars, step_size):
        end = min(start + window_size, n)
        window = bars_1m.iloc[start:end]

        if len(window) < min_window_bars:
            continue

        try:
            result = simulate_orb_outcome(
                window,
                symbol=symbol,
                config=cfg,
                bars_daily=bars_daily,
            )
            # Store window provenance so dataset generator can recover bars
            result._window_offset = start
            result._window_size = len(window)
            results.append(result)
        except Exception as exc:
            logger.debug("Simulation failed at offset %d: %s", start, exc)
            err_result = ORBSimResult(
                symbol=symbol,
                error=str(exc),
                simulated_at=datetime.now(_EST).isoformat(),
            )
            err_result._window_offset = start
            err_result._window_size = end - start
            results.append(err_result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Batch simulation for %s: %d windows → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )

    return results


def simulate_day(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
) -> ORBSimResult:
    """Simulate a single day's ORB trade.

    Unlike ``simulate_batch`` which slides windows, this function treats the
    entire ``bars_1m`` input as one day and produces exactly one ORBSimResult.
    Use this when you've already sliced your data to one trading session.

    Args:
        bars_1m: One day of 1-minute OHLCV bars.
        symbol: Instrument symbol.
        config: BracketConfig.
        bars_daily: Daily bars for NR7 (optional).

    Returns:
        A single ORBSimResult.
    """
    return simulate_orb_outcome(
        bars_1m=bars_1m,
        symbol=symbol,
        config=config,
        bars_daily=bars_daily,
    )


# ---------------------------------------------------------------------------
# PrevDay range simulation
# ---------------------------------------------------------------------------


def _simulate_range_outcome(
    bars_1m: pd.DataFrame,
    range_high: float,
    range_low: float,
    symbol: str,
    config: BracketConfig,
    bars_daily: pd.DataFrame | None,
    window_offset: int,
    range_start_time_str: str,
) -> ORBSimResult:
    """Shared bracket walk-forward used by PrevDay, IB, and Consolidation.

    Given a pre-computed *range_high* / *range_low*, this function:
      1. Scans *bars_1m* for the first close beyond the range.
      2. Sizes the bracket (SL/TP1/TP2) from ATR using *config*.
      3. Walks forward bar-by-bar to determine trade outcome.
      4. Populates and returns an :class:`ORBSimResult`.

    The function deliberately reuses ``config.or_end`` as the post-range
    scan start time — callers should set ``config.or_end`` to the moment
    after the range is fully formed (e.g. IB end = 10:30 ET).

    Args:
        bars_1m:             1-min OHLCV window (already localised to ET).
        range_high:          Pre-computed range high (H of the range period).
        range_low:           Pre-computed range low (L of the range period).
        symbol:              Instrument ticker for labelling.
        config:              BracketConfig — uses ``or_end``, ATR multiples,
                             ``max_hold_bars``, ``require_close_break``.
        bars_daily:          Optional daily bars for NR7 detection.
        window_offset:       Original index into the full bars_1m for provenance.
        range_start_time_str: ISO timestamp string of range start for the
                              ``or_start_time`` field.

    Returns:
        Populated :class:`ORBSimResult` (label may be ``"no_trade"``).
    """
    result = ORBSimResult(symbol=symbol, simulated_at=datetime.now(_EST).isoformat())
    result._window_offset = window_offset  # type: ignore[attr-defined]
    result._window_size = len(bars_1m)  # type: ignore[attr-defined]
    result.or_high = range_high
    result.or_low = range_low
    result.or_range = range_high - range_low
    result.or_start_time = range_start_time_str

    if bars_1m is None or bars_1m.empty:
        result.error = "No bar data"
        return result

    required = {"High", "Low", "Close"}
    if required - set(bars_1m.columns):
        result.error = f"Missing columns: {required - set(bars_1m.columns)}"
        return result

    df = _localize_to_est(bars_1m)
    df = df.sort_index()

    highs = np.asarray(df["High"].astype(float).values)
    lows = np.asarray(df["Low"].astype(float).values)
    closes = np.asarray(df["Close"].astype(float).values)
    has_volume = "Volume" in df.columns
    volumes = np.asarray(df["Volume"].astype(float).values) if has_volume else np.ones(len(df))
    has_open = "Open" in df.columns
    opens = np.asarray(df["Open"].astype(float).values) if has_open else closes.copy()

    try:
        times = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
    except Exception:
        result.error = "Cannot extract time from index"
        return result

    # ATR over all available bars
    atr = _compute_atr(highs, lows, closes, period=config.atr_period)
    result.atr = atr
    if atr <= 0:
        result.error = "ATR is zero"
        result.outcome = "zero_atr"
        return result

    or_range = range_high - range_low
    if or_range <= 0:
        result.error = f"Invalid range: high={range_high}, low={range_low}"
        result.outcome = "invalid_or"
        return result

    # Validate range/ATR ratio against config
    from lib.core.breakout_types import BreakoutType, get_range_config

    _bt = getattr(result, "_breakout_type", BreakoutType.ORB)
    _cfg_bt = get_range_config(_bt)
    range_atr_ratio = or_range / atr
    if range_atr_ratio < _cfg_bt.min_range_atr_ratio:
        result.outcome = "range_too_narrow"
        result.error = f"range/ATR={range_atr_ratio:.3f} < min={_cfg_bt.min_range_atr_ratio}"
        return result
    if range_atr_ratio > _cfg_bt.max_range_atr_ratio:
        result.outcome = "range_too_wide"
        result.error = f"range/ATR={range_atr_ratio:.3f} > max={_cfg_bt.max_range_atr_ratio}"
        return result

    # Pre-market range
    pm_mask = (times >= config.pm_start) & (times < config.pm_end)
    pm_indices = np.where(pm_mask)[0]
    if len(pm_indices) > 0:
        result.pm_high = float(np.max(highs[pm_indices]))
        result.pm_low = float(np.min(lows[pm_indices]))

    # NR7
    if bars_daily is not None and len(bars_daily) >= 7:
        with contextlib.suppress(Exception):
            d_highs = np.asarray(bars_daily["High"].astype(float).values[-7:])
            d_lows = np.asarray(bars_daily["Low"].astype(float).values[-7:])
            daily_ranges = d_highs - d_lows
            result.nr7 = bool(daily_ranges[-1] <= np.min(daily_ranges))

    # Scan for breakout after or_end (range formation complete)
    post_range_mask = times >= config.or_end
    post_range_indices = np.where(post_range_mask)[0]

    if len(post_range_indices) == 0:
        result.outcome = "no_post_or_bars"
        return result

    direction: str | None = None
    breakout_idx: int | None = None
    entry_price: float = 0.0

    for idx in post_range_indices:
        if config.require_close_break:
            val = float(closes[idx])
            if val > range_high:
                direction = "LONG"
                entry_price = max(range_high, float(opens[idx]))
                breakout_idx = idx
                break
            if val < range_low:
                direction = "SHORT"
                entry_price = min(range_low, float(opens[idx]))
                breakout_idx = idx
                break
        else:
            if float(highs[idx]) > range_high:
                direction = "LONG"
                entry_price = max(range_high, float(opens[idx]))
                breakout_idx = idx
                break
            if float(lows[idx]) < range_low:
                direction = "SHORT"
                entry_price = min(range_low, float(opens[idx]))
                breakout_idx = idx
                break

    if direction is None or breakout_idx is None:
        result.outcome = "no_breakout"
        return result

    result.direction = direction
    result.entry = entry_price
    result.breakout_bar_idx = breakout_idx

    with contextlib.suppress(Exception):
        result.breakout_time = str(df.index[breakout_idx])

    # Volume ratio
    avg_vol = float(np.mean(volumes[max(0, breakout_idx - 20) : breakout_idx])) if breakout_idx > 0 else 1.0
    if avg_vol > 0:
        result.breakout_volume_ratio = float(volumes[breakout_idx] / avg_vol)

    # CVD delta from bar 0 to breakout
    cvd = 0.0
    total_vol = 0.0
    for i in range(0, breakout_idx + 1):
        vol = float(volumes[i])
        total_vol += vol
        cvd += vol if closes[i] > opens[i] else -vol
    result.cvd_delta = (cvd / total_vol) if total_vol > 0 else 0.0

    # London/NY overlap flag
    with contextlib.suppress(Exception):
        _ts_str = str(df.index[breakout_idx])
        _ts = pd.Timestamp(_ts_str)
        _hour = int(getattr(_ts, "hour", 0))
        result.london_overlap_flag = 1.0 if 8 <= _hour <= 9 else 0.0

    # Bracket sizing
    sl_dist = atr * config.sl_atr_mult
    tp1_dist = atr * config.tp1_atr_mult
    tp2_dist = atr * config.tp2_atr_mult
    tp3_dist = atr * config.tp3_atr_mult if config.tp3_atr_mult > 0 else 0.0

    if direction == "LONG":
        result.sl = entry_price - sl_dist
        result.tp1 = entry_price + tp1_dist
        result.tp2 = entry_price + tp2_dist
        result.tp3 = entry_price + tp3_dist if tp3_dist > 0 else 0.0
    else:
        result.sl = entry_price + sl_dist
        result.tp1 = entry_price - tp1_dist
        result.tp2 = entry_price - tp2_dist
        result.tp3 = entry_price - tp3_dist if tp3_dist > 0 else 0.0

    # Walk forward — Phase 1: SL vs TP1
    max_exit_idx = min(breakout_idx + config.max_hold_bars, len(df))
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False
    hit_sl = False
    ema_trail_exit = False
    exit_idx = breakout_idx
    trail_exit_price = 0.0

    for bar_idx in range(breakout_idx + 1, max_exit_idx):
        bar_high = highs[bar_idx]
        bar_low = lows[bar_idx]
        if direction == "LONG":
            if bar_low <= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_high >= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break
        else:
            if bar_high >= result.sl:
                hit_sl = True
                exit_idx = bar_idx
                break
            if bar_low <= result.tp1:
                hit_tp1 = True
                exit_idx = bar_idx
                break

    # Phase 2: After TP1, scan for TP2
    if hit_tp1 and not hit_sl:
        tp2_end = min(exit_idx + 1 + config.max_hold_bars, len(df))
        for bar_idx in range(exit_idx + 1, tp2_end):
            bar_high = highs[bar_idx]
            bar_low = lows[bar_idx]
            if direction == "LONG":
                if bar_low <= result.sl:
                    break
                if bar_high >= result.tp2:
                    hit_tp2 = True
                    exit_idx = bar_idx
                    break
            else:
                if bar_high >= result.sl:
                    break
                if bar_low <= result.tp2:
                    hit_tp2 = True
                    exit_idx = bar_idx
                    break

        # Phase 3: After TP2, EMA9 trail toward TP3
        if hit_tp2 and config.enable_ema_trail_after_tp2 and config.tp3_atr_mult > 0:
            ema_period = config.ema_trail_period
            alpha = 2.0 / (ema_period + 1)
            ema_start = max(0, breakout_idx - ema_period * 2)
            ema_closes = closes[ema_start : exit_idx + 1]
            if len(ema_closes) >= ema_period:
                ema = float(np.mean(ema_closes[:ema_period]))
                for ci in range(ema_period, len(ema_closes)):
                    ema = alpha * float(ema_closes[ci]) + (1.0 - alpha) * ema
            else:
                ema = float(closes[exit_idx])

            trail_end = min(exit_idx + 1 + config.max_hold_bars, len(df))
            for bar_idx in range(exit_idx + 1, trail_end):
                bar_high = highs[bar_idx]
                bar_low = lows[bar_idx]
                bar_close = closes[bar_idx]
                ema = alpha * float(bar_close) + (1.0 - alpha) * ema

                if direction == "LONG":
                    if result.tp3 > 0 and bar_high >= result.tp3:
                        hit_tp3 = True
                        exit_idx = bar_idx
                        break
                    if float(bar_close) < ema:
                        ema_trail_exit = True
                        trail_exit_price = float(bar_close)
                        exit_idx = bar_idx
                        break
                else:
                    if result.tp3 > 0 and bar_low <= result.tp3:
                        hit_tp3 = True
                        exit_idx = bar_idx
                        break
                    if float(bar_close) > ema:
                        ema_trail_exit = True
                        trail_exit_price = float(bar_close)
                        exit_idx = bar_idx
                        break

    result.hold_bars = exit_idx - breakout_idx
    result.hit_tp2 = hit_tp2
    result.hit_tp3 = hit_tp3
    result.ema_trail_exit = ema_trail_exit
    result.trail_exit_price = trail_exit_price
    with contextlib.suppress(Exception):
        result.exit_time = str(df.index[exit_idx])

    if hit_tp1:
        result.label = f"good_{direction.lower()}"
        if hit_tp3:
            result.outcome = "tp3_hit"
            result.pnl_r = config.tp3_atr_mult / config.sl_atr_mult
        elif ema_trail_exit:
            result.outcome = "ema_trail_exit"
            if direction == "LONG":
                result.pnl_r = (trail_exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
            else:
                result.pnl_r = (entry_price - trail_exit_price) / sl_dist if sl_dist > 0 else 0.0
        elif hit_tp2:
            result.outcome = "tp2_hit"
            result.pnl_r = config.tp2_atr_mult / config.sl_atr_mult
        else:
            result.outcome = "tp1_hit"
            result.pnl_r = config.tp1_atr_mult / config.sl_atr_mult
    elif hit_sl:
        result.label = f"bad_{direction.lower()}"
        result.outcome = "sl_hit"
        result.pnl_r = -1.0
    else:
        exit_close = closes[exit_idx] if exit_idx < len(closes) else entry_price
        if direction == "LONG":
            unrealised_r = (exit_close - entry_price) / sl_dist if sl_dist > 0 else 0.0
        else:
            unrealised_r = (entry_price - exit_close) / sl_dist if sl_dist > 0 else 0.0
        result.pnl_r = round(unrealised_r, 3)
        result.label = f"bad_{direction.lower()}"
        result.outcome = "timeout"

    # Quality heuristic
    quality = 50.0
    if or_range > 0 and atr > 0:
        quality += min(20.0, (atr / or_range) * 10.0)
    if result.breakout_volume_ratio > 1.5:
        quality += 10.0
    elif result.breakout_volume_ratio > 1.2:
        quality += 5.0
    if result.nr7:
        quality += 15.0
    if (
        direction == "LONG"
        and result.pm_high > 0
        and entry_price >= result.pm_high
        or direction == "SHORT"
        and result.pm_low > 0
        and entry_price <= result.pm_low
    ):
        quality += 5.0
    result.quality_pct = min(99, max(0, int(quality)))

    return result


# ---------------------------------------------------------------------------
# PrevDay range simulation
# ---------------------------------------------------------------------------


def simulate_batch_prev_day(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
    session_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Previous Day High/Low breakouts across all days in *bars_1m*.

    For each calendar day in *bars_1m*, the prior session's high and low
    become the range boundaries.  The breakout scan begins at the RTH open
    (default 09:30 ET) and runs to the RTH close (default 16:00 ET).

    The *prior session* is defined as all bars from the previous day's RTH
    open up to (but not including) the current day's RTH open — this
    naturally includes any post-market/overnight bars that belong to the
    prior trading day without double-counting globex gaps.

    Tags every result with ``_breakout_type = BreakoutType.PrevDay``.

    Args:
        bars_1m:      Full 1-minute OHLCV history (multiple days).
        symbol:       Instrument ticker for labelling.
        config:       BracketConfig — ATR multiples, hold bars, etc.  The
                      ``or_start`` / ``or_end`` / ``pm_end`` fields are
                      overridden by *rth_start* / *rth_start* / *rth_start*.
        bars_daily:   Optional daily OHLCV bars for NR7 detection.
        rth_start:    Start of the breakout scan window (default 09:30 ET).
        rth_end:      End of the breakout scan window / session close
                      (default 16:00 ET).
        session_end:  Alias for *rth_end* — kept for API symmetry with IB/
                      Consolidation callers.

    Returns:
        One :class:`ORBSimResult` per day that has a valid prior-session
        range.  Days without sufficient prior-day data are silently skipped.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or session_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.PrevDay)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        logger.warning("simulate_batch_prev_day: missing columns %s", required - set(df.columns))
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    np.asarray(df["Close"].astype(float).values)

    try:
        times_all = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
        dates_all = pd.DatetimeIndex(df.index).date  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("simulate_batch_prev_day: cannot extract time/date — %s", exc)
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        # Need at least one prior day
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Prior session: all bars on prior_date
        prior_mask = dates_all == prior_date
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 5:
            logger.debug(
                "simulate_batch_prev_day: %s — not enough prior-day bars (%d)",
                trade_date,
                len(prior_indices),
            )
            continue

        prev_high = float(np.max(highs_all[prior_indices]))
        prev_low = float(np.min(lows_all[prior_indices]))

        if prev_high <= prev_low or prev_high <= 0:
            continue

        # Current day bars: from rth_start to rth_end
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            logger.debug(
                "simulate_batch_prev_day: %s — not enough today bars (%d)",
                trade_date,
                len(today_indices),
            )
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        # Build a BracketConfig where or_end == rth_start so the post-range
        # scan starts immediately at the open (no "formation" period needed).
        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            or_start=_rth_start,
            or_end=_rth_start,  # scan starts at rth_start — range is already set
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=prev_high,
            range_low=prev_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.PrevDay  # type: ignore[attr-defined]

        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "PrevDay batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


# ---------------------------------------------------------------------------
# Initial Balance range simulation
# ---------------------------------------------------------------------------


def simulate_batch_ib(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    ib_start: dt_time | None = None,
    ib_end: dt_time | None = None,
    session_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Initial Balance breakouts across all RTH sessions in *bars_1m*.

    The Initial Balance (IB) is the high/low of the first 60 minutes of
    the RTH session (default 09:30–10:30 ET), as defined in auction market
    theory.  A breakout beyond the IB is a high-conviction continuation or
    rotation signal.

    For each day with sufficient IB data the function:
      1. Computes IB high/low from bars in ``[ib_start, ib_end)``.
      2. Scans bars after ``ib_end`` for a close beyond the IB boundary.
      3. Applies ATR-based bracket sizing and walks forward to outcome.

    Tags every result with ``_breakout_type = BreakoutType.InitialBalance``.

    Args:
        bars_1m:     Full 1-minute OHLCV history (multiple days).
        symbol:      Instrument ticker.
        config:      BracketConfig — ATR multiples and hold parameters used
                     as defaults; ``or_start``/``or_end`` are overridden.
        bars_daily:  Optional daily bars for NR7.
        ib_start:    IB formation start (default 09:30 ET).
        ib_end:      IB formation end / breakout scan start (default 10:30 ET).
        session_end: End of the trading session (default 16:00 ET) — limits
                     the breakout scan window.

    Returns:
        One :class:`ORBSimResult` per day with a valid IB.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _ib_start: dt_time = ib_start or dt_time(9, 30)
    _ib_end: dt_time = ib_end or dt_time(10, 30)
    _sess_end: dt_time = session_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.InitialBalance)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        logger.warning("simulate_batch_ib: missing columns %s", required - set(df.columns))
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)

    try:
        times_all = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
        dates_all = pd.DatetimeIndex(df.index).date  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("simulate_batch_ib: cannot extract time/date — %s", exc)
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for trade_date in unique_dates:
        # IB formation bars
        ib_mask = (dates_all == trade_date) & (times_all >= _ib_start) & (times_all < _ib_end)
        ib_indices = np.where(ib_mask)[0]
        if len(ib_indices) < 5:
            logger.debug(
                "simulate_batch_ib: %s — only %d IB bars (need ≥5), skipping",
                trade_date,
                len(ib_indices),
            )
            continue

        ib_high = float(np.max(highs_all[ib_indices]))
        ib_low = float(np.min(lows_all[ib_indices]))

        if ib_high <= ib_low or ib_high <= 0:
            continue

        # Full session window: ib_start → session_end (includes IB bars for ATR)
        sess_mask = (dates_all == trade_date) & (times_all >= _ib_start) & (times_all < _sess_end)
        sess_indices = np.where(sess_mask)[0]
        if len(sess_indices) < 15:
            continue

        window_start = int(sess_indices[0])
        window_end = int(sess_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        # BracketConfig: or_start/or_end bound the formation period so that
        # _simulate_range_outcome scans after ib_end for the breakout.
        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            or_start=_ib_start,
            or_end=_ib_end,  # breakout scan starts here
            pm_start=dt_time(0, 0),
            pm_end=_ib_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[ib_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=ib_high,
            range_low=ib_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.InitialBalance  # type: ignore[attr-defined]

        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "IB batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


# ---------------------------------------------------------------------------
# Consolidation range simulation
# ---------------------------------------------------------------------------


def _detect_consolidation_boxes(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    times: np.ndarray,
    min_bars: int = 12,
    max_range_pct: float = 0.003,
    step: int = 1,
) -> list[tuple[int, int, float, float]]:
    """Scan a 1-min bar array for tight consolidation zones.

    Uses a sliding window to find the longest run of consecutive bars where
    ``(high - low) / close < max_range_pct`` for every bar, and the overall
    box ``(box_high - box_low) / mid_close < max_range_pct``.  Each found
    region is extended greedily until a bar violates the range constraint.

    Args:
        highs:         High prices array (float64).
        lows:          Low prices array (float64).
        closes:        Close prices array (float64).
        times:         Array of ``datetime.time`` objects (same length).
        min_bars:      Minimum consecutive bars to qualify (default 12).
        max_range_pct: Maximum ``(high - low) / close`` per bar AND for the
                       overall box (default 0.003 = 0.3 %).
        step:          Scan step size (default 1 bar = exhaustive scan).

    Returns:
        List of ``(start_idx, end_idx, box_high, box_low)`` tuples, one per
        non-overlapping consolidation region found (end_idx is exclusive).
        Regions are deduplicated so they do not overlap.
    """
    n = len(closes)
    boxes: list[tuple[int, int, float, float]] = []
    i = 0

    while i < n - min_bars:
        # Start a candidate box at bar i
        box_high = highs[i]
        box_low = lows[i]
        mid_close = closes[i]
        bar_range_pct = (box_high - box_low) / mid_close if mid_close > 0 else 1.0

        if bar_range_pct > max_range_pct:
            i += step
            continue

        # Greedily extend the box as long as the constraint holds
        j = i + 1
        while j < n:
            candidate_high = max(box_high, highs[j])
            candidate_low = min(box_low, lows[j])
            mid = closes[j]
            if mid <= 0:
                break
            bar_pct = (highs[j] - lows[j]) / mid
            box_pct = (candidate_high - candidate_low) / mid
            if bar_pct > max_range_pct or box_pct > max_range_pct:
                break
            box_high = candidate_high
            box_low = candidate_low
            j += 1

        run_len = j - i
        if run_len >= min_bars:
            boxes.append((i, j, box_high, box_low))
            i = j  # skip past this box — no overlapping boxes
        else:
            i += step

    return boxes


def simulate_batch_consolidation(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    min_consolidation_bars: int = 12,
    max_range_pct: float = 0.003,
    session_start: dt_time | None = None,
    session_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Consolidation breakouts detected algorithmically in *bars_1m*.

    For each calendar day this function:
      1. Extracts the intra-session bars (default 09:30–16:00 ET).
      2. Calls :func:`_detect_consolidation_boxes` to find tight N-bar
         zones where ``(high - low) / close < max_range_pct`` for at least
         *min_consolidation_bars* consecutive bars.
      3. For each box, scans the bars *immediately after* the box for a
         close beyond box high or box low.
      4. Applies ATR-based bracket sizing and walks forward to outcome.

    Multiple consolidation boxes may be found per day, yielding multiple
    training examples.  Each result is tagged with
    ``_breakout_type = BreakoutType.Consolidation``.

    Args:
        bars_1m:                 Full 1-min OHLCV history (multiple days).
        symbol:                  Instrument ticker.
        config:                  BracketConfig — ATR multiples and hold
                                 parameters; ``or_start``/``or_end`` are
                                 overridden per detected box.
        bars_daily:              Optional daily bars for NR7.
        min_consolidation_bars:  Minimum bars for a valid box (default 12).
        max_range_pct:           Maximum ``(H-L)/close`` per bar and for the
                                 box (default 0.003 = 0.3 %).
        session_start:           Start of the scan window (default 09:30 ET).
        session_end:             End of the scan window (default 16:00 ET).

    Returns:
        List of :class:`ORBSimResult` — potentially multiple per day if
        several tight boxes are found.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _sess_start: dt_time = session_start or dt_time(9, 30)
    _sess_end: dt_time = session_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.Consolidation)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        logger.warning(
            "simulate_batch_consolidation: missing columns %s",
            required - set(df.columns),
        )
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)

    try:
        times_all = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
        dates_all = pd.DatetimeIndex(df.index).date  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("simulate_batch_consolidation: cannot extract time/date — %s", exc)
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for trade_date in unique_dates:
        sess_mask = (dates_all == trade_date) & (times_all >= _sess_start) & (times_all < _sess_end)
        sess_indices = np.where(sess_mask)[0]
        if len(sess_indices) < min_consolidation_bars + 5:
            continue

        sess_highs = highs_all[sess_indices]
        sess_lows = lows_all[sess_indices]
        sess_closes = closes_all[sess_indices]
        sess_times = times_all[sess_indices]

        boxes = _detect_consolidation_boxes(
            highs=sess_highs,
            lows=sess_lows,
            closes=sess_closes,
            times=sess_times,
            min_bars=min_consolidation_bars,
            max_range_pct=max_range_pct,
        )

        for box_start_rel, box_end_rel, box_high, box_low in boxes:
            # Map relative box indices back to the full DataFrame
            abs_box_start = int(sess_indices[box_start_rel])
            abs_box_end = int(sess_indices[min(box_end_rel, len(sess_indices) - 1)])

            # The simulation window runs from the box start to session end
            # so ATR is computed over a meaningful range and the post-box
            # breakout scan has sufficient bars.
            abs_sess_end = int(sess_indices[-1]) + 1
            window_start = abs_box_start
            window_end = abs_sess_end
            window = df.iloc[window_start:window_end]

            if len(window) < min_consolidation_bars + 5:
                continue

            # or_end is set to the time immediately after the box ends so
            # _simulate_range_outcome scans post-box bars for the breakout.
            box_end_abs_idx = abs_box_end
            box_end_time: dt_time = dt_time(0, 0)
            with contextlib.suppress(Exception):
                box_end_time = pd.Timestamp(df.index[box_end_abs_idx]).time()  # type: ignore[arg-type]
            # Clamp to session bounds
            if box_end_time >= _sess_end:
                continue

            day_cfg = BracketConfig(
                sl_atr_mult=cfg_bt.sl_atr_mult,
                tp1_atr_mult=cfg_bt.tp1_atr_mult,
                tp2_atr_mult=cfg_bt.tp2_atr_mult,
                max_hold_bars=base_cfg.max_hold_bars,
                atr_period=base_cfg.atr_period,
                or_start=_sess_start,
                or_end=box_end_time,  # breakout scan starts right after box
                pm_start=dt_time(0, 0),
                pm_end=_sess_start,
                require_close_break=base_cfg.require_close_break,
                min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
            )

            range_start_str = ""
            with contextlib.suppress(Exception):
                range_start_str = str(df.index[abs_box_start])

            result = _simulate_range_outcome(
                bars_1m=window,
                range_high=box_high,
                range_low=box_low,
                symbol=symbol,
                config=day_cfg,
                bars_daily=bars_daily,
                window_offset=window_start,
                range_start_time_str=range_start_str,
            )
            result._breakout_type = BreakoutType.Consolidation  # type: ignore[attr-defined]

            results.append(result)
            logger.debug(
                "Consolidation box %s [%s – %s]: H=%.4f L=%.4f → %s",
                trade_date,
                range_start_str,
                box_end_time,
                box_high,
                box_low,
                result.label,
            )

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Consolidation batch for %s: %d boxes across %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        len(unique_dates),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarise_results(results: list[ORBSimResult]) -> dict[str, Any]:
    """Compute aggregate statistics from a batch of simulation results.

    Useful for evaluating filter effectiveness or bracket parameter sweeps.

    Returns:
        Dict with trade count, win rate, avg R, profit factor, etc.
    """
    trades = [r for r in results if r.is_trade]
    winners = [r for r in trades if r.is_winner]
    losers = [r for r in trades if not r.is_winner]

    if not trades:
        return {
            "total_windows": len(results),
            "total_trades": 0,
            "no_trade_count": len(results),
            "win_rate": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "avg_hold_bars": 0.0,
            "avg_quality": 0.0,
        }

    total_win_r = sum(r.pnl_r for r in winners)
    total_loss_r = abs(sum(r.pnl_r for r in losers))
    profit_factor = total_win_r / total_loss_r if total_loss_r > 0 else float("inf")

    label_counts: dict[str, int] = {}
    for r in results:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1

    return {
        "total_windows": len(results),
        "total_trades": len(trades),
        "no_trade_count": len(results) - len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(trades) * 100, 1),
        "avg_r": round(sum(r.pnl_r for r in trades) / len(trades), 3),
        "total_r": round(sum(r.pnl_r for r in trades), 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_bars": round(sum(r.hold_bars for r in trades) / len(trades), 1),
        "avg_quality": round(sum(r.quality_pct for r in trades) / len(trades), 1),
        "label_distribution": label_counts,
        "long_trades": sum(1 for r in trades if r.direction == "LONG"),
        "short_trades": sum(1 for r in trades if r.direction == "SHORT"),
        "nr7_trades": sum(1 for r in trades if r.nr7),
        "nr7_win_rate": round(
            sum(1 for r in trades if r.nr7 and r.is_winner) / max(1, sum(1 for r in trades if r.nr7)) * 100,
            1,
        ),
    }


def results_to_dataframe(results: list[ORBSimResult]) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame for analysis."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


# ===========================================================================
# New researched range breakout simulators
# ===========================================================================


def simulate_batch_weekly(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Weekly Range breakouts — prior week's high/low as the range.

    For each Monday (or first trading day of the week) in *bars_1m*, the
    prior week's high and low become the range boundaries.  The breakout
    scan runs during RTH on the first day of the new week.

    Tags results with ``_breakout_type = BreakoutType.Weekly``.

    Args:
        bars_1m:     Full 1-minute OHLCV history (multiple weeks).
        symbol:      Instrument ticker.
        config:      BracketConfig.
        bars_daily:  Optional daily bars for NR7.
        rth_start:   RTH scan start (default 09:30 ET).
        rth_end:     RTH scan end (default 16:00 ET).

    Returns:
        One :class:`ORBSimResult` per week with a valid prior-week range.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.Weekly)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    # Group dates by ISO week number
    from collections import defaultdict

    weeks: dict[tuple[int, int], list] = defaultdict(list)
    for d in unique_dates:
        iso = d.isocalendar()
        weeks[(iso[0], iso[1])].append(d)

    sorted_week_keys = sorted(weeks.keys())

    for wk_i, wk_key in enumerate(sorted_week_keys):
        if wk_i == 0:
            continue  # need prior week

        prev_wk_key = sorted_week_keys[wk_i - 1]
        prev_week_dates = weeks[prev_wk_key]
        this_week_dates = weeks[wk_key]

        # Prior week high/low
        prev_mask = np.isin(dates_all, prev_week_dates)
        prev_indices = np.where(prev_mask)[0]
        if len(prev_indices) < 10:
            continue

        prev_high = float(np.max(highs_all[prev_indices]))
        prev_low = float(np.min(lows_all[prev_indices]))
        if prev_high <= prev_low or prev_high <= 0:
            continue

        # First day of new week: scan RTH bars
        trade_date = this_week_dates[0]
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prev_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=prev_high,
            range_low=prev_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.Weekly  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Weekly batch for %s: %d weeks → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def simulate_batch_monthly(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Monthly Range breakouts — prior month's high/low as the range.

    For each first trading day of a new month, the prior month's high and
    low become the range boundaries.

    Tags results with ``_breakout_type = BreakoutType.Monthly``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.Monthly)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    # Group dates by (year, month)
    from collections import defaultdict

    months: dict[tuple[int, int], list] = defaultdict(list)
    for d in unique_dates:
        months[(d.year, d.month)].append(d)

    sorted_month_keys = sorted(months.keys())

    for mo_i, mo_key in enumerate(sorted_month_keys):
        if mo_i == 0:
            continue

        prev_mo_key = sorted_month_keys[mo_i - 1]
        prev_month_dates = months[prev_mo_key]
        this_month_dates = months[mo_key]

        prev_mask = np.isin(dates_all, prev_month_dates)
        prev_indices = np.where(prev_mask)[0]
        if len(prev_indices) < 20:
            continue

        prev_high = float(np.max(highs_all[prev_indices]))
        prev_low = float(np.min(lows_all[prev_indices]))
        if prev_high <= prev_low or prev_high <= 0:
            continue

        trade_date = this_month_dates[0]
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prev_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=prev_high,
            range_low=prev_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.Monthly  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Monthly batch for %s: %d months → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def simulate_batch_asian(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    asian_start: dt_time | None = None,
    asian_end: dt_time | None = None,
    scan_start: dt_time | None = None,
    scan_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Asian Session Range breakouts (19:00–02:00 ET).

    The Asian session range spans overnight — from 19:00 ET (Tokyo/Sydney
    open) to 02:00 ET (just before London pre-market).  Breakouts beyond
    this range during London or US sessions are high-conviction signals.

    Tags results with ``_breakout_type = BreakoutType.Asian``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _asian_start: dt_time = asian_start or dt_time(19, 0)
    _asian_end: dt_time = asian_end or dt_time(2, 0)
    _scan_start: dt_time = scan_start or dt_time(3, 0)  # London open
    _scan_end: dt_time = scan_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.Asian)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Asian range: prior_date 19:00–23:59 + trade_date 00:00–02:00
        # Evening portion on prior date
        evening_mask = (dates_all == prior_date) & (times_all >= _asian_start)
        # Early morning portion on trade date
        morning_mask = (dates_all == trade_date) & (times_all < _asian_end)

        asian_indices = np.where(evening_mask | morning_mask)[0]
        if len(asian_indices) < 10:
            continue

        asian_high = float(np.max(highs_all[asian_indices]))
        asian_low = float(np.min(lows_all[asian_indices]))
        if asian_high <= asian_low or asian_high <= 0:
            continue

        # Scan window: trade_date from scan_start to scan_end
        scan_mask = (dates_all == trade_date) & (times_all >= _scan_start) & (times_all < _scan_end)
        scan_indices = np.where(scan_mask)[0]
        if len(scan_indices) < 10:
            continue

        window_start = int(scan_indices[0])
        window_end = int(scan_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_scan_start,
            or_end=_scan_start,  # scan starts immediately
            pm_start=dt_time(0, 0),
            pm_end=_scan_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[asian_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=asian_high,
            range_low=asian_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.Asian  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Asian batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def _detect_bollinger_squeeze(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
    min_squeeze_bars: int = 6,
) -> list[tuple[int, int, float, float]]:
    """Detect Bollinger Band inside Keltner Channel squeeze zones.

    A squeeze occurs when the BB upper < KC upper AND BB lower > KC lower,
    indicating extreme compression.  The breakout is the first bar where
    BB expands outside KC.

    Returns:
        List of ``(squeeze_start_idx, squeeze_end_idx, range_high, range_low)``
        tuples for each non-overlapping squeeze found.
    """
    n = len(closes)
    if n < max(bb_period, kc_period) + 2:
        return []

    # Bollinger Bands
    bb_mid = np.asarray(pd.Series(closes).rolling(bb_period).mean())
    bb_std_vals = np.asarray(pd.Series(closes).rolling(bb_period).std(ddof=0))
    bb_upper = bb_mid + bb_std * bb_std_vals
    bb_lower = bb_mid - bb_std * bb_std_vals

    # Keltner Channel (ATR-based)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    kc_mid = np.asarray(pd.Series(closes).rolling(kc_period).mean())
    kc_atr = np.asarray(pd.Series(tr).rolling(kc_period).mean())
    kc_upper = kc_mid + kc_atr_mult * kc_atr
    kc_lower = kc_mid - kc_atr_mult * kc_atr

    # Squeeze: BB inside KC
    squeeze_flags = np.zeros(n, dtype=bool)
    for i in range(max(bb_period, kc_period), n):
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
        if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
            squeeze_flags[i] = True

    # Find runs of squeeze bars
    boxes: list[tuple[int, int, float, float]] = []
    i = 0
    while i < n:
        if squeeze_flags[i]:
            start = i
            while i < n and squeeze_flags[i]:
                i += 1
            run_len = i - start
            if run_len >= min_squeeze_bars:
                range_high = float(np.max(highs[start:i]))
                range_low = float(np.min(lows[start:i]))
                if range_high > range_low:
                    boxes.append((start, i, range_high, range_low))
        else:
            i += 1

    return boxes


def simulate_batch_bollinger_squeeze(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    session_start: dt_time | None = None,
    session_end: dt_time | None = None,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
    min_squeeze_bars: int = 6,
) -> list[ORBSimResult]:
    """Simulate Bollinger Squeeze breakouts — BB inside KC compression.

    Detects squeeze zones where BB contracts inside KC, then simulates
    breakout when BB expands.  Uses the squeeze range (high/low) as the
    breakout boundary.

    Tags results with ``_breakout_type = BreakoutType.BollingerSqueeze``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _sess_start: dt_time = session_start or dt_time(9, 30)
    _sess_end: dt_time = session_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.BollingerSqueeze)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for trade_date in unique_dates:
        sess_mask = (dates_all == trade_date) & (times_all >= _sess_start) & (times_all < _sess_end)
        sess_indices = np.where(sess_mask)[0]
        if len(sess_indices) < bb_period + min_squeeze_bars + 10:
            continue

        sess_highs = highs_all[sess_indices]
        sess_lows = lows_all[sess_indices]
        sess_closes = closes_all[sess_indices]

        boxes = _detect_bollinger_squeeze(
            closes=sess_closes,
            highs=sess_highs,
            lows=sess_lows,
            bb_period=bb_period,
            bb_std=bb_std,
            kc_period=kc_period,
            kc_atr_mult=kc_atr_mult,
            min_squeeze_bars=min_squeeze_bars,
        )

        for box_start_rel, box_end_rel, box_high, box_low in boxes:
            abs_box_start = int(sess_indices[box_start_rel])
            abs_box_end = int(sess_indices[min(box_end_rel, len(sess_indices) - 1)])
            abs_sess_end = int(sess_indices[-1]) + 1

            window = df.iloc[abs_box_start:abs_sess_end]
            if len(window) < min_squeeze_bars + 5:
                continue

            box_end_time: dt_time = dt_time(0, 0)
            with contextlib.suppress(Exception):
                box_end_time = pd.Timestamp(df.index[abs_box_end]).time()  # type: ignore[arg-type]
            if box_end_time >= _sess_end:
                continue

            day_cfg = BracketConfig(
                sl_atr_mult=cfg_bt.sl_atr_mult,
                tp1_atr_mult=cfg_bt.tp1_atr_mult,
                tp2_atr_mult=cfg_bt.tp2_atr_mult,
                tp3_atr_mult=cfg_bt.tp3_atr_mult,
                max_hold_bars=base_cfg.max_hold_bars,
                atr_period=base_cfg.atr_period,
                enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
                ema_trail_period=cfg_bt.ema_trail_period,
                or_start=_sess_start,
                or_end=box_end_time,
                pm_start=dt_time(0, 0),
                pm_end=_sess_start,
                require_close_break=base_cfg.require_close_break,
                min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
            )

            range_start_str = ""
            with contextlib.suppress(Exception):
                range_start_str = str(df.index[abs_box_start])

            result = _simulate_range_outcome(
                bars_1m=window,
                range_high=box_high,
                range_low=box_low,
                symbol=symbol,
                config=day_cfg,
                bars_daily=bars_daily,
                window_offset=abs_box_start,
                range_start_time_str=range_start_str,
            )
            result._breakout_type = BreakoutType.BollingerSqueeze  # type: ignore[attr-defined]
            results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "BollingerSqueeze batch for %s: %d squeezes → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def _compute_value_area(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    tick_size: float = 0.25,
    va_pct: float = 0.70,
) -> tuple[float, float, float]:
    """Compute Value Area High/Low/POC from OHLCV bar data.

    Uses a simplified TPO-like approach: distribute each bar's volume
    across its price range in tick-sized buckets, then find the 70%
    concentration zone around the POC (Point of Control).

    Returns:
        (vah, val, poc) — Value Area High, Value Area Low, Point of Control.
        Returns (0, 0, 0) if insufficient data.
    """
    n = len(closes)
    if n < 5 or tick_size <= 0:
        return 0.0, 0.0, 0.0

    all_high = float(np.max(highs))
    all_low = float(np.min(lows))
    if all_high <= all_low:
        return 0.0, 0.0, 0.0

    # Build volume-at-price histogram
    num_ticks = int((all_high - all_low) / tick_size) + 1
    if num_ticks < 2 or num_ticks > 50000:
        return 0.0, 0.0, 0.0

    vol_profile = np.zeros(num_ticks)
    for i in range(n):
        bar_low_tick = int((float(lows[i]) - all_low) / tick_size)
        bar_high_tick = int((float(highs[i]) - all_low) / tick_size)
        bar_low_tick = max(0, min(bar_low_tick, num_ticks - 1))
        bar_high_tick = max(0, min(bar_high_tick, num_ticks - 1))
        ticks_in_bar = bar_high_tick - bar_low_tick + 1
        if ticks_in_bar > 0:
            vol_per_tick = float(volumes[i]) / ticks_in_bar
            vol_profile[bar_low_tick : bar_high_tick + 1] += vol_per_tick

    # POC = tick with highest volume
    poc_tick = int(np.argmax(vol_profile))
    poc_price = all_low + poc_tick * tick_size
    total_vol = float(np.sum(vol_profile))
    if total_vol <= 0:
        return 0.0, 0.0, 0.0

    # Expand outward from POC until va_pct of total volume is included
    va_vol = float(vol_profile[poc_tick])
    lo_idx = poc_tick
    hi_idx = poc_tick
    target_vol = total_vol * va_pct

    while va_vol < target_vol:
        can_go_up = hi_idx < num_ticks - 1
        can_go_down = lo_idx > 0

        if not can_go_up and not can_go_down:
            break

        up_vol = float(vol_profile[hi_idx + 1]) if can_go_up else -1.0
        down_vol = float(vol_profile[lo_idx - 1]) if can_go_down else -1.0

        if up_vol >= down_vol:
            hi_idx += 1
            va_vol += up_vol
        else:
            lo_idx -= 1
            va_vol += down_vol

    vah = all_low + hi_idx * tick_size
    val = all_low + lo_idx * tick_size

    return vah, val, poc_price


def simulate_batch_value_area(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
    tick_size: float = 0.25,
    va_pct: float = 0.70,
) -> list[ORBSimResult]:
    """Simulate Value Area breakouts — prior session's VAH/VAL as range.

    For each day, computes the prior day's volume profile Value Area
    (High/Low), then scans the current day for a breakout beyond VAH/VAL.
    Based on the "70% rule" from market profile / auction market theory.

    Tags results with ``_breakout_type = BreakoutType.ValueArea``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.ValueArea)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close", "Volume"}
    if required - set(df.columns):
        logger.warning("simulate_batch_value_area: missing columns %s", required - set(df.columns))
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)
    volumes_all = np.asarray(df["Volume"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Prior day RTH bars for volume profile
        prior_mask = (dates_all == prior_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 20:
            continue

        vah, val, _poc = _compute_value_area(
            highs=highs_all[prior_indices],
            lows=lows_all[prior_indices],
            closes=closes_all[prior_indices],
            volumes=volumes_all[prior_indices],
            tick_size=tick_size,
            va_pct=va_pct,
        )
        if vah <= val or vah <= 0:
            continue

        # Current day scan window
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=vah,
            range_low=val,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.ValueArea  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "ValueArea batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def simulate_batch_inside_day(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
) -> list[ORBSimResult]:
    """Simulate Inside Day breakouts.

    An inside day occurs when the current day's high is below the prior
    day's high AND the current day's low is above the prior day's low.
    The prior day's high/low become the range; breakout is the first bar
    that closes beyond either level.

    Tags results with ``_breakout_type = BreakoutType.InsideDay``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.InsideDay)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i < 2:
            continue  # need two prior days

        prior_date = unique_dates[day_i - 1]
        prior_prior_date = unique_dates[day_i - 2]

        # Prior day's range (the "mother bar")
        prior_mask = (dates_all == prior_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 10:
            continue

        mother_high = float(np.max(highs_all[prior_indices]))
        mother_low = float(np.min(lows_all[prior_indices]))
        if mother_high <= mother_low:
            continue

        # Prior-prior day (to check the inside pattern on the prior day)
        pp_mask = (dates_all == prior_prior_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        pp_indices = np.where(pp_mask)[0]
        if len(pp_indices) < 10:
            continue

        pp_high = float(np.max(highs_all[pp_indices]))
        pp_low = float(np.min(lows_all[pp_indices]))

        # Check if prior day is inside the prior-prior day (inside day pattern)
        if mother_high >= pp_high or mother_low <= pp_low:
            continue  # not an inside day

        # Compression ratio check
        mother_range = mother_high - mother_low
        pp_range = pp_high - pp_low
        if pp_range <= 0:
            continue
        compression = mother_range / pp_range
        min_comp = cfg_bt.extra.get("min_compression_ratio", 0.30)
        max_comp = cfg_bt.extra.get("max_compression_ratio", 0.85)
        if compression < min_comp or compression > max_comp:
            continue

        # Use the mother bar's range as the breakout boundary
        # but trade on the NEXT day (trade_date) when the inside day resolves
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[0]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=mother_high,
            range_low=mother_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.InsideDay  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "InsideDay batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def simulate_batch_gap_rejection(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
    min_gap_atr_pct: float = 0.15,
) -> list[ORBSimResult]:
    """Simulate Gap Rejection breakouts.

    Detects overnight gaps (gap between prior session close and current
    session open) and trades the rejection at the gap edge.  The range is
    defined as the gap zone: [prior_close, current_open] (gap up) or
    [current_open, prior_close] (gap down).

    Tags results with ``_breakout_type = BreakoutType.GapRejection``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.GapRejection)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"Open", "High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)
    opens_all = np.asarray(df["Open"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Prior day's close (last bar of RTH)
        prior_mask = (dates_all == prior_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 10:
            continue

        prior_close = float(closes_all[prior_indices[-1]])

        # ATR for gap threshold
        atr = _compute_atr(
            highs_all[prior_indices],
            lows_all[prior_indices],
            closes_all[prior_indices],
            period=14,
        )
        if atr <= 0:
            continue

        # Current day open (first bar at rth_start)
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        current_open = float(opens_all[today_indices[0]])
        gap = current_open - prior_close

        # Check minimum gap size
        if abs(gap) < min_gap_atr_pct * atr:
            continue

        # Gap range: the zone between prior close and current open
        if gap > 0:  # gap up
            range_high = current_open
            range_low = prior_close
        else:  # gap down
            range_high = prior_close
            range_low = current_open

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[-1]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=range_high,
            range_low=range_low,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.GapRejection  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "GapRejection batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def _compute_pivots(
    prev_high: float,
    prev_low: float,
    prev_close: float,
    formula: str = "classic",
) -> tuple[float, float, float]:
    """Compute floor pivot levels from prior day's HLC.

    Args:
        prev_high:  Prior day's high.
        prev_low:   Prior day's low.
        prev_close: Prior day's close.
        formula:    ``"classic"`` (default), ``"woodie"``, or ``"camarilla"``.

    Returns:
        (pivot, r1, s1) — Pivot Point, R1, S1.
    """
    if formula == "woodie":
        pivot = (prev_high + prev_low + 2 * prev_close) / 4.0
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
    elif formula == "camarilla":
        pivot = (prev_high + prev_low + prev_close) / 3.0
        rng = prev_high - prev_low
        r1 = prev_close + rng * 1.1 / 12.0
        s1 = prev_close - rng * 1.1 / 12.0
    else:  # classic
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high

    return pivot, r1, s1


def simulate_batch_pivot_points(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
    pivot_formula: str = "classic",
) -> list[ORBSimResult]:
    """Simulate Pivot Point breakouts — classic floor pivot R1/S1 as range.

    For each day, computes the prior day's classic (or Woodie/Camarilla)
    pivot R1 and S1 levels.  The range is [S1, R1] and breakout is the
    first bar that closes beyond either level.

    Tags results with ``_breakout_type = BreakoutType.PivotPoints``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.PivotPoints)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Prior day RTH bars
        prior_mask = (dates_all == prior_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 10:
            continue

        prev_high = float(np.max(highs_all[prior_indices]))
        prev_low = float(np.min(lows_all[prior_indices]))
        prev_close = float(closes_all[prior_indices[-1]])

        _pivot, r1, s1 = _compute_pivots(prev_high, prev_low, prev_close, formula=pivot_formula)
        if r1 <= s1 or r1 <= 0:
            continue

        # Current day scan
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[-1]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=r1,
            range_low=s1,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.PivotPoints  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "PivotPoints batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


def _find_swing_hl(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 100,
    min_swing_atr: float = 1.5,
    atr: float = 1.0,
) -> tuple[float, float] | None:
    """Find the most recent significant swing high and swing low.

    Scans the last *lookback* bars for the highest high and lowest low
    that form a swing of at least *min_swing_atr × atr*.

    Returns:
        ``(swing_high, swing_low)`` or ``None`` if no valid swing found.
    """
    n = len(highs)
    if n < 10:
        return None

    start = max(0, n - lookback)
    seg_highs = highs[start:n]
    seg_lows = lows[start:n]

    swing_high = float(np.max(seg_highs))
    swing_low = float(np.min(seg_lows))
    swing_range = swing_high - swing_low

    if swing_range < min_swing_atr * atr or swing_high <= swing_low:
        return None

    return swing_high, swing_low


def simulate_batch_fibonacci(
    bars_1m: pd.DataFrame,
    symbol: str = "",
    config: BracketConfig | None = None,
    bars_daily: pd.DataFrame | None = None,
    rth_start: dt_time | None = None,
    rth_end: dt_time | None = None,
    fib_upper: float = 0.618,
    fib_lower: float = 0.382,
    swing_lookback_bars: int = 100,
    min_swing_atr_mult: float = 1.5,
) -> list[ORBSimResult]:
    """Simulate Fibonacci retracement zone breakouts.

    For each day, identifies the prior session's swing high/low, computes
    the 38.2%–61.8% Fibonacci retracement zone, and scans the current
    session for a breakout beyond the zone boundaries.

    The zone acts as a "decision zone" — price rejecting from within
    the zone triggers a breakout in the swing direction.

    Tags results with ``_breakout_type = BreakoutType.Fibonacci``.
    """
    from datetime import time as dt_time

    from lib.core.breakout_types import BreakoutType, get_range_config

    _rth_start: dt_time = rth_start or dt_time(9, 30)
    _rth_end: dt_time = rth_end or dt_time(16, 0)

    cfg_bt = get_range_config(BreakoutType.Fibonacci)
    base_cfg = config or DEFAULT_BRACKET

    if bars_1m is None or bars_1m.empty:
        return []

    df = _localize_to_est(bars_1m).sort_index()

    required = {"High", "Low", "Close"}
    if required - set(df.columns):
        return []

    highs_all = np.asarray(df["High"].astype(float).values)
    lows_all = np.asarray(df["Low"].astype(float).values)
    closes_all = np.asarray(df["Close"].astype(float).values)

    try:
        _dti = pd.DatetimeIndex(df.index)
        times_all = _dti.time  # type: ignore[attr-defined]
        dates_all = _dti.date  # type: ignore[attr-defined]
    except Exception:
        return []

    unique_dates = sorted(set(dates_all))
    results: list[ORBSimResult] = []

    for day_i, trade_date in enumerate(unique_dates):
        if day_i == 0:
            continue

        prior_date = unique_dates[day_i - 1]

        # Prior day bars for swing identification
        prior_mask = dates_all == prior_date
        prior_indices = np.where(prior_mask)[0]
        if len(prior_indices) < 20:
            continue

        # ATR from prior day
        atr = _compute_atr(
            highs_all[prior_indices],
            lows_all[prior_indices],
            closes_all[prior_indices],
            period=14,
        )
        if atr <= 0:
            continue

        # Find swing H/L from prior session bars
        swing_result = _find_swing_hl(
            highs=highs_all[prior_indices],
            lows=lows_all[prior_indices],
            lookback=min(swing_lookback_bars, len(prior_indices)),
            min_swing_atr=min_swing_atr_mult,
            atr=atr,
        )
        if swing_result is None:
            continue

        swing_high, swing_low = swing_result
        swing_range = swing_high - swing_low

        # Fibonacci retracement zone
        fib_high_level = swing_low + fib_upper * swing_range  # 61.8% level
        fib_low_level = swing_low + fib_lower * swing_range  # 38.2% level

        if fib_high_level <= fib_low_level or fib_high_level <= 0:
            continue

        # Current day scan
        today_mask = (dates_all == trade_date) & (times_all >= _rth_start) & (times_all < _rth_end)
        today_indices = np.where(today_mask)[0]
        if len(today_indices) < 10:
            continue

        window_start = int(today_indices[0])
        window_end = int(today_indices[-1]) + 1
        window = df.iloc[window_start:window_end]

        day_cfg = BracketConfig(
            sl_atr_mult=cfg_bt.sl_atr_mult,
            tp1_atr_mult=cfg_bt.tp1_atr_mult,
            tp2_atr_mult=cfg_bt.tp2_atr_mult,
            tp3_atr_mult=cfg_bt.tp3_atr_mult,
            max_hold_bars=base_cfg.max_hold_bars,
            atr_period=base_cfg.atr_period,
            enable_ema_trail_after_tp2=cfg_bt.enable_ema_trail_after_tp2,
            ema_trail_period=cfg_bt.ema_trail_period,
            or_start=_rth_start,
            or_end=_rth_start,
            pm_start=dt_time(0, 0),
            pm_end=_rth_start,
            require_close_break=base_cfg.require_close_break,
            min_or_range_atr_frac=cfg_bt.min_range_atr_ratio,
        )

        range_start_str = ""
        with contextlib.suppress(Exception):
            range_start_str = str(df.index[prior_indices[-1]])

        result = _simulate_range_outcome(
            bars_1m=window,
            range_high=fib_high_level,
            range_low=fib_low_level,
            symbol=symbol,
            config=day_cfg,
            bars_daily=bars_daily,
            window_offset=window_start,
            range_start_time_str=range_start_str,
        )
        result._breakout_type = BreakoutType.Fibonacci  # type: ignore[attr-defined]
        results.append(result)

    trades = sum(1 for r in results if r.is_trade)
    winners = sum(1 for r in results if r.is_winner)
    logger.info(
        "Fibonacci batch for %s: %d days → %d trades (%d winners, %.1f%% WR)",
        symbol,
        len(results),
        trades,
        winners,
        (winners / trades * 100) if trades > 0 else 0.0,
    )
    return results


# ---------------------------------------------------------------------------
# Backward-compatible aliases (Phase 1F rename: orb_simulator → rb_simulator)
# ---------------------------------------------------------------------------
# ``ORBSimResult`` and ``simulate_orb_outcome`` keep their original names so
# existing callers (``dataset_generator.py``, ``__init__.py``, tests) work
# unchanged until they are migrated to the new names.

#: New canonical name for the simulation result dataclass.
RBSimResult = ORBSimResult

#: New canonical name for the single-window simulation function.
simulate_rb_outcome = simulate_orb_outcome
