"""
K-Means Adaptive Volatility Clustering.

This module implements the exact K-Means clustering logic:
  - Training window of ATR values (default 250 bars)
  - 3 clusters: LOW, MEDIUM, HIGH volatility
  - Iterative centroid convergence with configurable max iterations
  - Volatility percentile ranking
  - Position sizing multiplier per cluster
  - Volatility regime labels (VERY LOW → VERY HIGH)

The implementation is pure NumPy — no scikit-learn dependency required,
though scikit-learn is available in the environment if needed.

Design decisions:
  - Uses True Range (not just High-Low) for ATR, matching Pine's ta.atr()
  - Initial centroids placed at lowvol/midvol/highvol percentiles of the
    ATR range, exactly as in fks.pine
  - Convergence check uses relative threshold (0.1% of current volatility)
  - Recalculation frequency is handled by the caller (engine caches results)
  - Position multiplier: LOW=1.2x, MEDIUM=1.0x, HIGH=0.6x (conservative
    in high vol, slightly aggressive in low vol)
  - SL multiplier: LOW=1.2x, MEDIUM=1.0x, HIGH=0.8x (tighter stops in
    high vol to limit losses, wider in low vol for breathing room)

Usage:
    from lib.volatility import kmeans_volatility_clusters, volatility_summary_text

    result = kmeans_volatility_clusters(df)
    # result = {
    #     "cluster": "MEDIUM",
    #     "adaptive_atr": 2.45,
    #     "percentile": 0.62,
    #     "vol_status": "MEDIUM",
    #     "volatility_regime": "MEDIUM",
    #     "position_multiplier": 1.0,
    #     "sl_multiplier": 1.0,
    #     "centroids": {"LOW": 1.20, "MEDIUM": 2.30, "HIGH": 4.10},
    #     "raw_atr": 2.51,
    #     "strategy_hint": "NORMAL STOPS",
    # }
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("volatility")


# ---------------------------------------------------------------------------
# ATR computation (matches Pine's ta.atr — Wilder's smoothing / RMA)
# ---------------------------------------------------------------------------


def _compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Average True Range using Wilder's smoothing (RMA).

    This matches Pine Script's ta.atr(period) exactly:
      TR = max(high-low, |high-prev_close|, |low-prev_close|)
      ATR = RMA(TR, period)  where RMA uses alpha = 1/period
    """
    n = len(high)
    if n < 2:
        return np.full(n, high[0] - low[0] if n > 0 else 0.0)

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Wilder's smoothing (RMA) — equivalent to EMA with alpha = 1/period
    alpha = 1.0 / period
    atr = np.empty(n)
    # Seed with simple average of first `period` TRs
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
        # Backfill early values with expanding mean
        for i in range(period - 1):
            atr[i] = np.mean(tr[: i + 1])
    else:
        # Not enough bars for full ATR — use expanding mean
        for i in range(n):
            atr[i] = np.mean(tr[: i + 1])

    return atr


# ---------------------------------------------------------------------------
# Core: K-Means clustering (exact port of fks.pine)
# ---------------------------------------------------------------------------


def _kmeans_1d(
    data: np.ndarray,
    highvol: float = 0.8,
    midvol: float = 0.5,
    lowvol: float = 0.2,
    max_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """1D K-Means clustering with 3 clusters — exact port of Pine logic.

    Initial centroids are placed at fixed percentile positions within the
    data range, matching fks.pine's initialization:
      high_centroid = lower + range * highvol
      mid_centroid  = lower + range * midvol
      low_centroid  = lower + range * lowvol

    Args:
        data: 1D array of ATR values (training window)
        highvol: Position of high volatility centroid (0-1 of range)
        midvol: Position of medium volatility centroid (0-1 of range)
        lowvol: Position of low volatility centroid (0-1 of range)
        max_iterations: Maximum K-Means iterations

    Returns:
        centroids: array of 3 centroid values [high, medium, low]
        labels: array of cluster assignments for each data point (0=high, 1=med, 2=low)
    """
    if len(data) < 3:
        val = data[0] if len(data) > 0 else 1.0
        return np.array([val * 1.5, val, val * 0.5]), np.zeros(len(data), dtype=int)

    lower = float(np.min(data))
    upper = float(np.max(data))
    atr_range = upper - lower
    if atr_range <= 0:
        atr_range = max(abs(upper), 1e-8)

    # Initial centroids (same as Pine: lower + range * factor)
    centroids = np.array(
        [
            lower + atr_range * highvol,  # cluster 0: HIGH
            lower + atr_range * midvol,  # cluster 1: MEDIUM
            lower + atr_range * lowvol,  # cluster 2: LOW
        ]
    )

    # Convergence threshold (relative to mean volatility)
    convergence_threshold = np.mean(data) * 0.001

    labels = np.zeros(len(data), dtype=int)

    for _iteration in range(max_iterations):
        # Assignment step: assign each point to nearest centroid
        distances = np.abs(data[:, np.newaxis] - centroids[np.newaxis, :])
        labels = np.argmin(distances, axis=1)

        # Update step: recalculate centroids as cluster means
        new_centroids = np.empty(3)
        for k in range(3):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = np.mean(data[mask])
            else:
                # Empty cluster — keep old centroid
                new_centroids[k] = centroids[k]

        # Convergence check (same as Pine: all centroids stable)
        if np.all(np.abs(new_centroids - centroids) < convergence_threshold):
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids, labels


# ---------------------------------------------------------------------------
# Volatility percentile (ported from fks.pine)
# ---------------------------------------------------------------------------


def _compute_percentile(
    historical_volatility: np.ndarray,
    current_value: float,
) -> float:
    """Compute what percentile the current volatility falls at.

    Exact port of fks.pine's loop-based percentile calculation:
      count how many historical values are less than current, divide by total.
    """
    if len(historical_volatility) < 2:
        return 0.5
    count_less = int(np.sum(historical_volatility < current_value))
    return count_less / len(historical_volatility)


# ---------------------------------------------------------------------------
# Volatility regime labels (ported from fks.pine)
# ---------------------------------------------------------------------------


def _percentile_to_regime(percentile: float) -> str:
    """Map volatility percentile to regime label.

    Exact thresholds from fks.pine:
      < 0.2  → VERY LOW
      < 0.4  → LOW
      < 0.6  → MEDIUM (default)
      >= 0.6 → HIGH
      >= 0.8 → VERY HIGH
    """
    if percentile < 0.2:
        return "VERY LOW"
    elif percentile < 0.4:
        return "LOW"
    elif percentile >= 0.8:
        return "VERY HIGH"
    elif percentile >= 0.6:
        return "HIGH"
    return "MEDIUM"


def _regime_to_strategy_hint(regime: str, percentile: float) -> str:
    """Generate actionable strategy hint based on volatility regime.

    Ported from fks_info.pine's volatility analysis section.
    """
    if percentile > 0.8:
        return "AVOID TIGHT STOPS"
    elif percentile < 0.2:
        return "BREAKOUT WATCH"
    elif percentile > 0.6:
        return "WIDER STOPS"
    elif percentile < 0.4:
        return "TIGHTER STOPS"
    return "NORMAL STOPS"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kmeans_volatility_clusters(
    df: pd.DataFrame,
    atr_len: int = 14,
    training_period: int = 250,
    max_iterations: int = 10,
    highvol: float = 0.8,
    midvol: float = 0.5,
    lowvol: float = 0.2,
    volatility_history: int = 500,
) -> dict[str, Any]:
    """Full K-Means adaptive volatility analysis — main entry point.

    This is the Python port of fks.pine's complete volatility clustering:
      1. Compute ATR(atr_len) series using Wilder's smoothing
      2. Run K-Means on last `training_period` ATR values → 3 clusters
      3. Assign current volatility to nearest centroid
      4. Compute percentile over `volatility_history` bars
      5. Determine regime label and strategy hint
      6. Calculate position and stop-loss multipliers

    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        atr_len: ATR period (default 14, matching Pine)
        training_period: Number of bars for K-Means training (default 250)
        max_iterations: K-Means max iterations (default 10)
        highvol: High vol centroid position (default 0.8)
        midvol: Medium vol centroid position (default 0.5)
        lowvol: Low vol centroid position (default 0.2)
        volatility_history: Number of bars for percentile calculation

    Returns:
        Dict with cluster, adaptive_atr, percentile, multipliers, etc.
        Returns safe defaults on insufficient data.
    """
    default_result: dict[str, Any] = {
        "cluster": "MEDIUM",
        "adaptive_atr": 0.0,
        "raw_atr": 0.0,
        "percentile": 0.5,
        "vol_status": "MEDIUM",
        "volatility_regime": "MEDIUM",
        "position_multiplier": 1.0,
        "sl_multiplier": 1.0,
        "centroids": {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0},
        "strategy_hint": "NORMAL STOPS",
        "iterations_used": 0,
    }

    if df is None or df.empty or len(df) < 20:
        return default_result

    try:
        high = np.asarray(df["High"].astype(float).values)
        low = np.asarray(df["Low"].astype(float).values)
        close = np.asarray(df["Close"].astype(float).values)
    except (KeyError, ValueError) as exc:
        logger.warning("Volatility analysis failed — missing HLC columns: %s", exc)
        return default_result

    n = len(close)

    # Step 1: Compute ATR series
    atr_series = _compute_atr(high, low, close, period=atr_len)
    current_atr = float(atr_series[-1])

    # Guard against zero/negative ATR
    if current_atr <= 0:
        current_atr = float(np.mean(np.abs(high - low)[-14:])) or 0.0001  # type: ignore[operator]
        default_result["raw_atr"] = round(current_atr, 6)
        return default_result

    # Step 2: K-Means clustering on training window
    train_end = n
    train_start = max(0, n - training_period)
    training_data = atr_series[train_start:train_end]

    # Filter out any NaN/zero values
    training_data = training_data[~np.isnan(training_data)]
    training_data = training_data[training_data > 0]

    if len(training_data) < 10:
        # Not enough data for clustering — return percentile-based result
        result = default_result.copy()
        result["raw_atr"] = round(current_atr, 6)
        result["adaptive_atr"] = round(current_atr, 6)
        return result

    centroids, labels = _kmeans_1d(
        training_data,
        highvol=highvol,
        midvol=midvol,
        lowvol=lowvol,
        max_iterations=max_iterations,
    )

    # Step 3: Assign current ATR to nearest cluster
    distances = np.abs(current_atr - centroids)
    cluster_idx = int(np.argmin(distances))
    cluster_labels = {0: "HIGH", 1: "MEDIUM", 2: "LOW"}
    cluster = cluster_labels[cluster_idx]

    # Adaptive ATR = centroid value of assigned cluster
    adaptive_atr = float(centroids[cluster_idx])

    # Step 4: Volatility percentile over history window
    hist_start = max(0, n - volatility_history)
    vol_history = atr_series[hist_start:]
    vol_history = vol_history[~np.isnan(vol_history)]
    percentile = _compute_percentile(vol_history, current_atr)

    # Step 5: Regime label and strategy hint
    volatility_regime = _percentile_to_regime(percentile)
    strategy_hint = _regime_to_strategy_hint(volatility_regime, percentile)

    # Step 6: Multipliers (ported from fks.pine)
    # Position multiplier: conservative in high vol, slightly aggressive in low
    position_multipliers = {"HIGH": 0.6, "MEDIUM": 1.0, "LOW": 1.2}
    position_multiplier = position_multipliers[cluster]

    # SL multiplier: tighter stops in high vol (limit damage),
    # wider in low vol (avoid noise stops)
    # This matches fks.pine's vol_multiplier logic
    sl_multipliers = {"HIGH": 0.8, "MEDIUM": 1.0, "LOW": 1.2}
    sl_multiplier = sl_multipliers[cluster]

    # Sort centroids for display (LOW < MEDIUM < HIGH)
    sorted_centroids = sorted(centroids)

    return {
        "cluster": cluster,
        "adaptive_atr": round(float(adaptive_atr), 6),
        "raw_atr": round(float(current_atr), 6),
        "percentile": round(float(percentile), 4),
        "vol_status": cluster,
        "volatility_regime": volatility_regime,
        "position_multiplier": position_multiplier,
        "sl_multiplier": sl_multiplier,
        "centroids": {
            "LOW": round(float(sorted_centroids[0]), 6),
            "MEDIUM": round(float(sorted_centroids[1]), 6),
            "HIGH": round(float(sorted_centroids[2]), 6),
        },
        "strategy_hint": strategy_hint,
    }


def volatility_summary_text(result: dict[str, Any]) -> str:
    """One-line summary suitable for Grok prompts or dashboard captions."""
    return (
        f"Vol {result['cluster']} cluster "
        f"({result['percentile']:.0%} percentile) — "
        f"ATR={result['raw_atr']:.4f}, "
        f"adaptive={result['adaptive_atr']:.4f}, "
        f"size={result['position_multiplier']}x, "
        f"hint={result['strategy_hint']}"
    )


def should_filter_entry(
    result: dict[str, Any],
    min_percentile: float = 0.2,
    max_percentile: float = 0.9,
) -> tuple[bool, str]:
    """Check if current volatility conditions should filter out entries.

    Ported from fks.pine's volatility_within_range logic:
      - Too low volatility (< min_percentile): no movement, skip
      - Too high volatility (> max_percentile): dangerous, skip

    Returns:
        (should_skip, reason) — True if entry should be skipped.
    """
    pct = result.get("percentile", 0.5)

    if pct < min_percentile:
        return True, f"Volatility too low ({pct:.0%}) — no edge in dead markets"

    if pct > max_percentile:
        return True, f"Extreme volatility ({pct:.0%}) — risk of slippage/whipsaw"

    return False, ""
