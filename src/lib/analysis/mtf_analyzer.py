"""
Multi-Timeframe Analyzer — EMA Alignment, MACD Momentum & Divergence
=====================================================================
Implements the full MTF analysis pipeline that serves two purposes:

  1. **Hard filter** — integrated into ``apply_all_filters()`` as an
     additional quality gate before publishing any breakout signal.
     Rejects breakouts where the HTF trend or momentum opposes the
     breakout direction.

  2. **CNN feature vector** — exports a fixed-length numeric vector
     that augments the tabular features fed to the breakout CNN.

Analysis layers
---------------
EMA Alignment / Slope
    Three EMAs (fast=9, mid=21, slow=50) on the higher-timeframe bars
    (default 15m).  A "stacked" alignment (fast>mid>slow for longs) gives
    the strongest trend signal.  Slope is measured as the % change in the
    slow EMA over the last N bars.

MACD Momentum
    Standard MACD(12,26,9) on HTF bars.  Histogram polarity must agree
    with breakout direction.  The histogram slope (change over last 3 bars)
    is exported as a CNN feature.  A flattening or reversing histogram is
    a soft warning even when the main signal agrees.

MACD Divergence
    Classic regular divergence: price makes a higher high (lower low) while
    MACD histogram makes a lower high (higher low).  Divergence opposing the
    breakout direction is a hard rejection.  Divergence *confirming* the
    breakout direction is a mild boost.

    Lookback window for divergence: ``divergence_lookback`` bars on HTF.

MTF Score
    A scalar 0.0–1.0 computed from:
      +0.30  EMA fully stacked in breakout direction
      +0.15  EMA slope positive in breakout direction (>= min_slope_pct)
      +0.25  MACD histogram agrees with direction
      +0.15  MACD histogram slope agrees
      +0.15  No opposing divergence detected
    Score >= ``min_pass_score`` (default 0.55) → hard filter passes.

CNN Feature Vector (10 values, always in [−1, 1])
    Index  Description
    ─────────────────────────────────────────────
    0      EMA fast vs slow normalised spread   (signed)
    1      EMA mid vs slow normalised spread    (signed)
    2      EMA slow slope (% / bar, signed)
    3      MACD line value (normalised by price)
    4      MACD signal value (normalised by price)
    5      MACD histogram (normalised by price)
    6      MACD histogram slope (normalised)
    7      MTF score (0–1)
    8      Divergence flag: +1 confirming, −1 opposing, 0 none
    9      EMA stacking flag: +1 aligned, 0 neutral, −1 counter

Public API
----------
    from lib.analysis.mtf_analyzer import (
        MTFAnalyzer,
        MTFResult,
        analyze_mtf,
        mtf_to_filter_verdict,
    )

    result = analyze_mtf(bars_htf, direction="LONG")
    verdict = mtf_to_filter_verdict(result, direction="LONG")
    cnn_features = result.cnn_features   # list[float], length 10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from lib.indicators.helpers import ema_numpy as _ema_series
from lib.indicators.helpers import macd_numpy as _macd_numpy

if TYPE_CHECKING:
    import pandas as pd

    pass

logger = logging.getLogger("analysis.mtf_analyzer")


# ===========================================================================
# Configuration
# ===========================================================================

# EMA periods
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50

# MACD periods
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Slope: measured over this many bars on the HTF
SLOPE_BARS = 5

# Divergence detection lookback (HTF bars)
DIV_LOOKBACK = 20

# Minimum MTF score to pass the hard filter
DEFAULT_MIN_PASS_SCORE = 0.55

# Minimum absolute EMA slow slope (% per bar) to count as "trending"
DEFAULT_MIN_SLOPE_PCT = 0.0002  # 0.02% per bar

# Score weights
_W_EMA_STACK = 0.30
_W_EMA_SLOPE = 0.15
_W_MACD_HIST = 0.25
_W_MACD_SLOPE = 0.15
_W_NO_DIV = 0.15


# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class MTFResult:
    """Full output of the MTF analysis for a single direction check."""

    # --- EMA ---
    ema_fast: float = 0.0
    ema_mid: float = 0.0
    ema_slow: float = 0.0
    ema_stacked: bool = False  # fast/mid/slow aligned in direction
    ema_slope: float = 0.0  # % change in slow EMA over SLOPE_BARS
    ema_slope_direction: str = ""  # "UP", "DOWN", "FLAT"

    # --- MACD ---
    macd_line: float = 0.0
    macd_signal_line: float = 0.0
    macd_histogram: float = 0.0
    macd_histogram_slope: float = 0.0  # change over last 3 bars
    macd_agrees: bool = False  # histogram polarity matches direction

    # --- Divergence ---
    divergence_type: str = ""  # "confirming", "opposing", or ""
    divergence_detected: bool = False

    # --- Score ---
    mtf_score: float = 0.0  # 0.0 – 1.0

    # --- CNN features ---
    cnn_features: list[float] = field(default_factory=lambda: [0.0] * 10)

    # --- Meta ---
    direction: str = ""  # "LONG" or "SHORT" that was analysed
    bars_used: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ema_fast": round(self.ema_fast, 4),
            "ema_mid": round(self.ema_mid, 4),
            "ema_slow": round(self.ema_slow, 4),
            "ema_stacked": self.ema_stacked,
            "ema_slope": round(self.ema_slope, 6),
            "ema_slope_direction": self.ema_slope_direction,
            "macd_line": round(self.macd_line, 6),
            "macd_signal_line": round(self.macd_signal_line, 6),
            "macd_histogram": round(self.macd_histogram, 6),
            "macd_histogram_slope": round(self.macd_histogram_slope, 6),
            "macd_agrees": self.macd_agrees,
            "divergence_type": self.divergence_type,
            "divergence_detected": self.divergence_detected,
            "mtf_score": round(self.mtf_score, 4),
            "cnn_features": [round(f, 6) for f in self.cnn_features],
            "direction": self.direction,
            "bars_used": self.bars_used,
            "error": self.error,
        }


# ===========================================================================
# Internal helpers
# ===========================================================================


# _ema_series and _macd_numpy are imported from lib.indicators.helpers above.
# _ema_series = ema_numpy, _macd_numpy = macd_numpy (aliased at import time).


def _detect_divergence(
    price_highs: np.ndarray,
    price_lows: np.ndarray,
    histogram: np.ndarray,
    direction: str,
    lookback: int = DIV_LOOKBACK,
) -> str:
    """Detect regular price/MACD divergence in the last ``lookback`` bars.

    Regular Bearish Divergence (opposing LONG):
        price makes higher high, MACD histogram makes lower high.
    Regular Bullish Divergence (opposing SHORT):
        price makes lower low, MACD histogram makes higher low.
    Confirming divergences (momentum agrees with direction) are also flagged.

    Returns:
        "opposing"    — divergence works against the breakout direction
        "confirming"  — divergence confirms the breakout direction
        ""            — no significant divergence detected
    """
    n = len(histogram)
    lb = min(lookback, n - 1)
    if lb < 4:
        return ""

    hist_window = histogram[n - lb :]
    h_window = price_highs[n - lb :]
    l_window = price_lows[n - lb :]

    dir_upper = direction.upper().strip()

    # Identify recent swing highs and lows in the price and histogram
    # Use a simple 3-bar pivot rule
    def _swing_highs(arr: np.ndarray) -> list[tuple[int, float]]:
        pivots = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                pivots.append((i, arr[i]))
        return pivots

    def _swing_lows(arr: np.ndarray) -> list[tuple[int, float]]:
        pivots = []
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                pivots.append((i, arr[i]))
        return pivots

    if dir_upper == "LONG":
        # Check for bearish divergence (opposing): price HH, histogram LH
        price_swings = _swing_highs(h_window)
        hist_swings = _swing_highs(hist_window)
        if len(price_swings) >= 2 and len(hist_swings) >= 2:
            p1_idx, p1_val = price_swings[-2]
            p2_idx, p2_val = price_swings[-1]
            h1_val = hist_window[p1_idx] if p1_idx < len(hist_window) else 0.0
            h2_val = hist_window[p2_idx] if p2_idx < len(hist_window) else 0.0
            if p2_val > p1_val and h2_val < h1_val:
                return "opposing"  # bearish divergence vs LONG

        # Check for bullish divergence (confirming): price HL, histogram HH
        price_swing_lows = _swing_lows(l_window)
        hist_swing_lows = _swing_lows(hist_window)
        if len(price_swing_lows) >= 2 and len(hist_swing_lows) >= 2:
            p1_idx, p1_val = price_swing_lows[-2]
            p2_idx, p2_val = price_swing_lows[-1]
            h1_val = hist_window[p1_idx] if p1_idx < len(hist_window) else 0.0
            h2_val = hist_window[p2_idx] if p2_idx < len(hist_window) else 0.0
            if p2_val > p1_val and h2_val > h1_val:
                return "confirming"  # bullish divergence confirms LONG

    elif dir_upper == "SHORT":
        # Check for bullish divergence (opposing): price LL, histogram HL
        price_swings = _swing_lows(l_window)
        hist_swings = _swing_lows(hist_window)
        if len(price_swings) >= 2 and len(hist_swings) >= 2:
            p1_idx, p1_val = price_swings[-2]
            p2_idx, p2_val = price_swings[-1]
            h1_val = hist_window[p1_idx] if p1_idx < len(hist_window) else 0.0
            h2_val = hist_window[p2_idx] if p2_idx < len(hist_window) else 0.0
            if p2_val < p1_val and h2_val > h1_val:
                return "opposing"  # bullish divergence vs SHORT

        # Check for bearish divergence (confirming): price HH, histogram LH
        price_swing_highs = _swing_highs(h_window)
        hist_swing_highs = _swing_highs(hist_window)
        if len(price_swing_highs) >= 2 and len(hist_swing_highs) >= 2:
            p1_idx, p1_val = price_swing_highs[-2]
            p2_idx, p2_val = price_swing_highs[-1]
            h1_val = hist_window[p1_idx] if p1_idx < len(hist_window) else 0.0
            h2_val = hist_window[p2_idx] if p2_idx < len(hist_window) else 0.0
            if p2_val > p1_val and h2_val < h1_val:
                return "confirming"  # bearish divergence confirms SHORT

    return ""


def _safe_norm(value: float, scale: float, clip: float = 1.0) -> float:
    """Normalise ``value`` by ``scale``, clamp to [-clip, +clip]."""
    if scale == 0 or not np.isfinite(scale):
        return 0.0
    return float(np.clip(value / scale, -clip, clip))


# ===========================================================================
# Core analyzer
# ===========================================================================


def analyze_mtf(
    bars_htf: pd.DataFrame | None,
    direction: str,
    ema_fast: int = EMA_FAST,
    ema_mid: int = EMA_MID,
    ema_slow: int = EMA_SLOW,
    macd_fast: int = MACD_FAST,
    macd_slow: int = MACD_SLOW,
    macd_signal: int = MACD_SIGNAL,
    slope_bars: int = SLOPE_BARS,
    divergence_lookback: int = DIV_LOOKBACK,
    min_slope_pct: float = DEFAULT_MIN_SLOPE_PCT,
) -> MTFResult:
    """Run the full MTF analysis on higher-timeframe bars.

    Args:
        bars_htf: Higher-timeframe OHLCV DataFrame (e.g. 15m bars).
                  Must have at minimum a ``Close`` column; ``High`` and
                  ``Low`` are used for divergence detection if present.
        direction: Breakout direction to evaluate against — "LONG" or "SHORT".
        ema_fast: Fast EMA period (default 9).
        ema_mid: Mid EMA period (default 21).
        ema_slow: Slow EMA period (default 50).
        macd_fast: MACD fast EMA period (default 12).
        macd_slow: MACD slow EMA period (default 26).
        macd_signal: MACD signal line period (default 9).
        slope_bars: Bars over which to measure EMA slow slope (default 5).
        divergence_lookback: HTF bars to search for divergence (default 20).
        min_slope_pct: Minimum |slope| to qualify as directional (default 0.02%).

    Returns:
        ``MTFResult`` with all indicators computed and a scalar ``mtf_score``.
        On insufficient data the result has ``error`` set and ``mtf_score=0``.
    """
    result = MTFResult(direction=direction.upper().strip())

    if bars_htf is None or bars_htf.empty:
        result.error = "No HTF bars supplied"
        result.cnn_features = [0.0] * 10
        return result

    min_required = max(ema_slow, macd_slow + macd_signal) + slope_bars + 2
    if len(bars_htf) < min_required:
        result.error = f"Insufficient HTF bars: {len(bars_htf)} < {min_required} required"
        result.cnn_features = [0.0] * 10
        return result

    try:
        closes = np.asarray(bars_htf["Close"].astype(float).values, dtype=float)
        highs = np.asarray(bars_htf["High"].astype(float).values, dtype=float) if "High" in bars_htf.columns else closes
        lows = np.asarray(bars_htf["Low"].astype(float).values, dtype=float) if "Low" in bars_htf.columns else closes
    except (KeyError, ValueError) as exc:
        result.error = f"Column error: {exc}"
        result.cnn_features = [0.0] * 10
        return result

    result.bars_used = len(closes)
    price_ref = float(closes[-1]) if closes[-1] > 0 else 1.0

    # ── EMA computation ────────────────────────────────────────────────────
    ema_f = _ema_series(np.asarray(closes, dtype=float), ema_fast)
    ema_m = _ema_series(np.asarray(closes, dtype=float), ema_mid)
    ema_s = _ema_series(np.asarray(closes, dtype=float), ema_slow)

    ef_val = float(ema_f[-1])
    em_val = float(ema_m[-1])
    es_val = float(ema_s[-1])

    result.ema_fast = round(ef_val, 4)
    result.ema_mid = round(em_val, 4)
    result.ema_slow = round(es_val, 4)

    dir_upper = result.direction
    if dir_upper == "LONG":
        result.ema_stacked = ef_val > em_val > es_val
    elif dir_upper == "SHORT":
        result.ema_stacked = ef_val < em_val < es_val
    else:
        result.ema_stacked = False

    # EMA slow slope over last slope_bars
    if len(ema_s) > slope_bars and es_val > 0:
        es_prev = float(ema_s[-(slope_bars + 1)])
        slope = (es_val - es_prev) / (es_prev + 1e-10)
    else:
        slope = 0.0
    result.ema_slope = round(slope, 8)

    if slope > min_slope_pct:
        result.ema_slope_direction = "UP"
    elif slope < -min_slope_pct:
        result.ema_slope_direction = "DOWN"
    else:
        result.ema_slope_direction = "FLAT"

    # ── MACD computation ───────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd_numpy(np.asarray(closes, dtype=float), macd_fast, macd_slow, macd_signal)

    macd_val = float(macd_line[-1])
    sig_val = float(signal_line[-1])
    hist_val = float(histogram[-1])

    result.macd_line = round(macd_val, 6)
    result.macd_signal_line = round(sig_val, 6)
    result.macd_histogram = round(hist_val, 6)

    # MACD histogram slope (change over last 3 bars)
    hist_slope_bars = 3
    if len(histogram) > hist_slope_bars:
        hist_prev = float(histogram[-(hist_slope_bars + 1)])
        hist_slope = hist_val - hist_prev
    else:
        hist_slope = 0.0
    result.macd_histogram_slope = round(hist_slope, 8)

    # MACD agrees with direction?
    if dir_upper == "LONG":
        result.macd_agrees = hist_val > 0
    elif dir_upper == "SHORT":
        result.macd_agrees = hist_val < 0
    else:
        result.macd_agrees = False

    # ── Divergence detection ───────────────────────────────────────────────
    div_type = _detect_divergence(
        np.asarray(highs, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(histogram, dtype=float),
        dir_upper,
        lookback=divergence_lookback,
    )
    result.divergence_type = div_type
    result.divergence_detected = div_type != ""

    # ── MTF Score ──────────────────────────────────────────────────────────
    score = 0.0

    # EMA stacking (30%)
    if result.ema_stacked:
        score += _W_EMA_STACK

    # EMA slope direction (15%)
    if (
        dir_upper == "LONG"
        and result.ema_slope_direction == "UP"
        or dir_upper == "SHORT"
        and result.ema_slope_direction == "DOWN"
    ):
        score += _W_EMA_SLOPE
    elif result.ema_slope_direction == "FLAT":
        score += _W_EMA_SLOPE * 0.5  # neutral slope is half credit

    # MACD histogram agrees (25%)
    if result.macd_agrees:
        score += _W_MACD_HIST

    # MACD histogram slope agrees (15%)
    if dir_upper == "LONG" and hist_slope > 0 or dir_upper == "SHORT" and hist_slope < 0:
        score += _W_MACD_SLOPE

    # No opposing divergence (15%)
    if div_type != "opposing":
        score += _W_NO_DIV
    if div_type == "confirming":
        score = min(1.0, score + 0.05)  # small bonus for confirming divergence

    result.mtf_score = round(min(score, 1.0), 4)

    # ── CNN Feature Vector (10 elements) ──────────────────────────────────
    # Normalise all values to approximately [-1, 1]
    # Feature 0: EMA fast spread vs slow  (signed, normalised by price)
    f0 = _safe_norm(ef_val - es_val, price_ref)

    # Feature 1: EMA mid spread vs slow  (signed, normalised by price)
    f1 = _safe_norm(em_val - es_val, price_ref)

    # Feature 2: EMA slow slope (clamp at ±0.01 = 1% per bar)
    f2 = float(np.clip(slope / 0.01, -1.0, 1.0)) if np.isfinite(slope) else 0.0

    # Feature 3: MACD line normalised by price
    f3 = _safe_norm(macd_val, price_ref * 0.01)

    # Feature 4: MACD signal line normalised by price
    f4 = _safe_norm(sig_val, price_ref * 0.01)

    # Feature 5: MACD histogram normalised by price
    f5 = _safe_norm(hist_val, price_ref * 0.01)

    # Feature 6: MACD histogram slope normalised (relative to histogram magnitude)
    hist_scale = max(abs(hist_val), price_ref * 0.0001)
    f6 = _safe_norm(hist_slope, hist_scale * 3)

    # Feature 7: MTF score (already 0–1)
    f7 = result.mtf_score

    # Feature 8: Divergence flag (+1 confirming, -1 opposing, 0 none)
    if div_type == "confirming":
        f8 = 1.0
    elif div_type == "opposing":
        f8 = -1.0
    else:
        f8 = 0.0

    # Feature 9: EMA stacking flag (+1 aligned, 0 neutral, -1 counter)
    if dir_upper == "LONG":
        if ef_val > em_val > es_val:
            f9 = 1.0
        elif ef_val < em_val or em_val < es_val:
            f9 = -1.0
        else:
            f9 = 0.0
    elif dir_upper == "SHORT":
        if ef_val < em_val < es_val:
            f9 = 1.0
        elif ef_val > em_val or em_val > es_val:
            f9 = -1.0
        else:
            f9 = 0.0
    else:
        f9 = 0.0

    result.cnn_features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

    logger.debug(
        "MTF [%s] score=%.3f ema_stacked=%s macd_agrees=%s div=%s slope=%s",
        dir_upper,
        result.mtf_score,
        result.ema_stacked,
        result.macd_agrees,
        div_type or "none",
        result.ema_slope_direction,
    )

    return result


# ===========================================================================
# Filter verdict adapter
# ===========================================================================


def mtf_to_filter_verdict(
    result: MTFResult,
    direction: str,
    min_pass_score: float = DEFAULT_MIN_PASS_SCORE,
) -> Any:
    """Convert an ``MTFResult`` into an ``orb_filters.FilterVerdict``.

    This bridges ``analyze_mtf()`` into the existing ``apply_all_filters()``
    pipeline without modifying that module's interface.

    The filter **passes** when ``result.mtf_score >= min_pass_score`` AND
    there is no opposing divergence detected.

    Opposing divergence is always a hard rejection regardless of score.

    Args:
        result: Output of ``analyze_mtf()``.
        direction: Breakout direction ("LONG" or "SHORT").
        min_pass_score: Minimum score threshold (default 0.55).

    Returns:
        ``FilterVerdict`` compatible with the ``orb_filters`` pipeline.
    """
    # Local import to avoid circular dependency — orb_filters → mtf_analyzer
    # is safe because orb_filters only imports mtf_analyzer inside functions.
    from lib.analysis.breakout_filters import FilterVerdict  # noqa: PLC0415

    name = "MTF Analyzer"

    # Insufficient data → skip (don't block the signal)
    if result.error:
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"MTF data unavailable ({result.error}) — skipped",
        )

    dir_upper = direction.upper().strip()

    # Hard rejection: opposing divergence
    if result.divergence_type == "opposing":
        reason = (
            f"{dir_upper} breakout vs opposing MACD divergence "
            f"(score={result.mtf_score:.3f}, "
            f"ema_stacked={result.ema_stacked}, "
            f"macd_agrees={result.macd_agrees})"
        )
        return FilterVerdict(name=name, passed=False, reason=reason)

    # Score gate
    passed = result.mtf_score >= min_pass_score

    # Build a readable summary
    parts = [
        f"score={result.mtf_score:.3f}",
        f"ema={'stacked' if result.ema_stacked else 'mixed'}",
        f"slope={result.ema_slope_direction}",
        f"macd={'✓' if result.macd_agrees else '✗'}",
    ]
    if result.divergence_type:
        parts.append(f"div={result.divergence_type}")

    reason = f"{dir_upper}: {', '.join(parts)}"

    if not passed:
        reason = f"MTF score {result.mtf_score:.3f} < threshold {min_pass_score:.2f} — {reason}"
        return FilterVerdict(name=name, passed=False, reason=reason)

    # Score boost for strong alignment
    boost = 0.0
    if result.ema_stacked and result.macd_agrees:
        boost = 0.08
    elif result.ema_stacked or result.macd_agrees:
        boost = 0.04
    if result.divergence_type == "confirming":
        boost += 0.03

    return FilterVerdict(
        name=name,
        passed=True,
        reason=f"MTF PASS — {reason}",
        score_boost=boost,
    )


# ===========================================================================
# Convenience wrapper: analyze + verdict in one call
# ===========================================================================


class MTFAnalyzer:
    """Stateless helper class that bundles ``analyze_mtf`` and
    ``mtf_to_filter_verdict`` into a single call.

    Instantiate with custom thresholds and reuse across many signals:

        analyzer = MTFAnalyzer(min_pass_score=0.60)
        result, verdict = analyzer.evaluate(bars_15m, direction="LONG")
    """

    def __init__(
        self,
        ema_fast: int = EMA_FAST,
        ema_mid: int = EMA_MID,
        ema_slow: int = EMA_SLOW,
        macd_fast: int = MACD_FAST,
        macd_slow: int = MACD_SLOW,
        macd_signal: int = MACD_SIGNAL,
        slope_bars: int = SLOPE_BARS,
        divergence_lookback: int = DIV_LOOKBACK,
        min_slope_pct: float = DEFAULT_MIN_SLOPE_PCT,
        min_pass_score: float = DEFAULT_MIN_PASS_SCORE,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.slope_bars = slope_bars
        self.divergence_lookback = divergence_lookback
        self.min_slope_pct = min_slope_pct
        self.min_pass_score = min_pass_score

    def evaluate(
        self,
        bars_htf: pd.DataFrame | None,
        direction: str,
    ) -> tuple[MTFResult, Any]:
        """Run MTF analysis and return ``(MTFResult, FilterVerdict)``.

        Args:
            bars_htf: Higher-timeframe bars (e.g. 15m).
            direction: "LONG" or "SHORT".

        Returns:
            Tuple of ``(MTFResult, FilterVerdict)``.  Never raises.
        """
        try:
            result = analyze_mtf(
                bars_htf,
                direction=direction,
                ema_fast=self.ema_fast,
                ema_mid=self.ema_mid,
                ema_slow=self.ema_slow,
                macd_fast=self.macd_fast,
                macd_slow=self.macd_slow,
                macd_signal=self.macd_signal,
                slope_bars=self.slope_bars,
                divergence_lookback=self.divergence_lookback,
                min_slope_pct=self.min_slope_pct,
            )
        except Exception as exc:
            logger.warning("MTFAnalyzer.evaluate error: %s", exc)
            from lib.analysis.breakout_filters import FilterVerdict  # noqa: PLC0415

            err_result = MTFResult(direction=direction, error=str(exc))
            err_result.cnn_features = [0.0] * 10
            return err_result, FilterVerdict(
                name="MTF Analyzer",
                passed=True,
                reason=f"MTF error — skipped: {exc}",
            )

        verdict = mtf_to_filter_verdict(result, direction, self.min_pass_score)
        return result, verdict
