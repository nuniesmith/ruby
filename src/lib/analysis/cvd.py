"""
Cumulative Volume Delta (CVD).

Two computation paths are supported and selected automatically:

  1. **Tick / L2 path** — when real bid/ask tick data is available (e.g. from
     Rithmic).  Each trade is classified as a buy (aggressor hit the ask) or
     sell (aggressor hit the bid) and delta is summed directly.  This is the
     accurate path and should be used whenever Rithmic credentials are
     configured and tick data has been pulled.

  2. **OHLCV heuristic fallback** — when only bar data is available.  Uses the
     standard approximation:

         buy_volume  = volume × (close − low) / (high − low)
         sell_volume = volume − buy_volume
         delta       = buy_volume − sell_volume

     Accuracy is ±15–25 % versus true bid/ask delta but adds useful confluence
     for confirming trends, detecting divergences, and identifying absorption.

The public ``compute_cvd`` function accepts an optional ``tick_df`` argument.
When supplied it takes the tick path; otherwise it falls back to the OHLCV
heuristic.  All downstream functions (divergence detection, absorption,
summaries, backtesting indicators) work identically regardless of which path
was used.

Features:
  - CVD calculation with intraday anchoring (reset at market open)
  - CVD divergence detection (price vs CVD direction mismatch)
  - Volume absorption candle identification (high volume, small body near S/R)
  - Rolling CVD slope for momentum confirmation
  - Dashboard-ready summary and indicator functions compatible with backtesting.py

Usage (OHLCV fallback — always available):
    from lib.analysis.cvd import compute_cvd, detect_cvd_divergences

    cvd_df = compute_cvd(bar_df)
    divergences = detect_cvd_divergences(cvd_df, lookback=20)

Usage (tick path — requires Rithmic tick data):
    from lib.analysis.cvd import compute_cvd, build_tick_df_from_rithmic

    tick_df = build_tick_df_from_rithmic(raw_ticks)   # shape helper
    cvd_df  = compute_cvd(bar_df, tick_df=tick_df)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cvd")


# ---------------------------------------------------------------------------
# Tick-data helpers
# ---------------------------------------------------------------------------


# Expected columns in a tick DataFrame (Rithmic or any L2 source).
# Column names are normalised inside compute_cvd so callers may use
# lower-case or any capitalisation variant.
#
#   price     — trade price
#   size      — trade size (number of contracts / shares / units)
#   side      — aggressor side: "buy", "sell", "B", "S", 1, -1
#               "buy"  / "B" / 1  → buyer-initiated (hit the ask)
#               "sell" / "S" / -1 → seller-initiated (hit the bid)
#   timestamp — trade timestamp (any pandas-parseable datetime)
#
TICK_REQUIRED_COLS = {"price", "size", "side", "timestamp"}


def build_tick_df_from_rithmic(raw_ticks: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of raw Rithmic tick dicts into a normalised tick DataFrame.

    Rithmic tick records typically look like::

        {
            "timestamp": "2025-06-01 09:31:00.123",
            "price": 2345.6,
            "volume": 3,
            "aggressor": "BUY",   # or "SELL"
        }

    This helper standardises the column names and ``side`` encoding so that
    ``compute_cvd`` can consume the result directly.

    Args:
        raw_ticks: List of tick dicts from the Rithmic provider.  Expected
                   keys (case-insensitive): ``timestamp``, ``price``,
                   ``volume`` or ``size``, ``aggressor`` or ``side``.

    Returns:
        DataFrame with columns: timestamp, price, size, side (str "buy"/"sell").
        Returns an empty DataFrame if ``raw_ticks`` is empty or malformed.
    """
    if not raw_ticks:
        return pd.DataFrame(columns=["timestamp", "price", "size", "side"])  # type: ignore[call-overload]

    try:
        df = pd.DataFrame(raw_ticks)
    except Exception as exc:
        logger.warning("build_tick_df_from_rithmic: failed to create DataFrame — %s", exc)
        return pd.DataFrame(columns=["timestamp", "price", "size", "side"])  # type: ignore[call-overload]

    # Normalise column names to lower-case
    df.columns = pd.Index([str(c).lower() for c in df.columns])

    # Rename volume → size if needed
    if "volume" in df.columns and "size" not in df.columns:
        df = df.rename(columns={"volume": "size"})

    # Rename aggressor → side if needed
    if "aggressor" in df.columns and "side" not in df.columns:
        df = df.rename(columns={"aggressor": "side"})

    # Normalise side to lowercase "buy" / "sell"
    if "side" in df.columns:
        df["side"] = (
            df["side"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(
                lambda s: (
                    "buy"
                    if s in ("buy", "b", "1", "bid_hit", "ask_taken")
                    else "sell"
                    if s in ("sell", "s", "-1", "ask_hit", "bid_taken")
                    else s
                )
            )
        )

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")

    return df


def _compute_cvd_from_ticks(
    bar_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    anchor_daily: bool = True,
) -> pd.Series:
    """Aggregate tick-level delta into per-bar CVD aligned to ``bar_df``.

    For each bar in ``bar_df`` (identified by its DatetimeIndex), all ticks
    whose timestamp falls within the bar's interval are summed:

        bar_delta = sum(size  for buy ticks) − sum(size for sell ticks)

    The per-bar deltas are then cumulatively summed to produce CVD, optionally
    anchored (reset) at each new trading day.

    Args:
        bar_df:      OHLCV DataFrame with a DatetimeIndex.
        tick_df:     Normalised tick DataFrame (output of
                     ``build_tick_df_from_rithmic``).
        anchor_daily: Reset cumulation at each calendar day boundary.

    Returns:
        pd.Series of per-bar CVD values, indexed like ``bar_df``.
    """
    # Validate tick_df columns
    tick_cols = set(tick_df.columns.str.lower())
    missing = {"price", "size", "side", "timestamp"} - tick_cols
    if missing:
        logger.warning(
            "_compute_cvd_from_ticks: tick_df missing columns %s — falling back to heuristic",
            missing,
        )
        return _compute_cvd_heuristic(bar_df, anchor_daily=anchor_daily)

    tick_df = tick_df.copy()
    tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"], errors="coerce")
    tick_df = tick_df.dropna(subset=["timestamp"])
    tick_df["size"] = pd.Series(pd.to_numeric(tick_df["size"], errors="coerce"), dtype=float).fillna(0.0).values

    # Signed size: +size for buys, -size for sells
    is_buy = tick_df["side"].str.lower().isin(["buy", "b"])
    tick_df["signed_size"] = np.where(is_buy, tick_df["size"], -tick_df["size"])

    # Build bar intervals from the DatetimeIndex
    bar_times = pd.DatetimeIndex(bar_df.index)
    if len(bar_times) < 2:
        # Not enough bars to infer interval
        return pd.Series(0.0, index=bar_df.index)

    # Infer bar duration from the most common gap between consecutive bars
    gaps = bar_times[1:] - bar_times[:-1]
    bar_duration = pd.Timedelta(gaps.value_counts().idxmax())

    # For each bar, find ticks in [bar_start, bar_start + bar_duration)
    # Cast to plain numpy arrays once outside the loop to avoid ExtensionArray issues
    ts_arr = np.asarray(tick_df["timestamp"].values, dtype="datetime64[ns]")
    signed = np.asarray(tick_df["signed_size"].values, dtype=float)

    bar_deltas = np.zeros(len(bar_times), dtype=float)

    for i, bar_start in enumerate(bar_times):
        bar_end = bar_start + bar_duration
        mask = (ts_arr >= np.datetime64(bar_start, "ns")) & (ts_arr < np.datetime64(bar_end, "ns"))
        bar_deltas[i] = float(signed[mask].sum())

    delta_series = pd.Series(bar_deltas, index=bar_df.index)

    # Cumulative sum with optional daily anchoring
    if anchor_daily and hasattr(bar_df.index, "date"):
        try:
            dates = pd.Series(bar_df.index).dt.date.values
            date_series = pd.Series(dates, index=bar_df.index)
            cvd = delta_series.groupby(date_series).cumsum()
        except Exception:
            cvd = delta_series.cumsum()
    else:
        cvd = delta_series.cumsum()

    return cvd


# ---------------------------------------------------------------------------
# OHLCV heuristic helpers
# ---------------------------------------------------------------------------


def _estimate_buy_volume(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Estimate buy volume using the OHLCV heuristic.

        buy_volume = volume × (close − low) / (high − low)

    When high == low (doji / zero-range bar), volume is split 50/50.
    """
    price_range = high - low
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (close - low) / safe_range
    buy_ratio = buy_ratio.fillna(0.5).clip(0.0, 1.0)
    return volume * buy_ratio


def _compute_cvd_heuristic(
    df: pd.DataFrame,
    anchor_daily: bool = True,
) -> pd.Series:
    """Compute per-bar CVD using the OHLCV heuristic.

    Returns a Series of CVD values indexed like ``df``.
    """
    high = pd.Series(df["High"].astype(float).values, index=df.index)
    low = pd.Series(df["Low"].astype(float).values, index=df.index)
    close = pd.Series(df["Close"].astype(float).values, index=df.index)
    volume = pd.Series(df["Volume"].astype(float).values, index=df.index)

    buy_vol = _estimate_buy_volume(high, low, close, volume)
    sell_vol = volume - buy_vol
    delta = buy_vol - sell_vol

    if anchor_daily and hasattr(df.index, "date"):
        try:
            dates = df.index.to_series().dt.date
            cvd = delta.groupby(dates).cumsum()
        except Exception:
            cvd = delta.cumsum()
    else:
        cvd = delta.cumsum()

    return cvd


# ---------------------------------------------------------------------------
# Public API: compute_cvd
# ---------------------------------------------------------------------------


def compute_cvd(
    df: pd.DataFrame,
    tick_df: pd.DataFrame | None = None,
    anchor_daily: bool = True,
) -> pd.DataFrame:
    """Compute Cumulative Volume Delta, using tick data when available.

    Automatically selects the best available computation path:

    * **Tick path** — used when ``tick_df`` is supplied and contains the
      required columns (timestamp, price, size, side).  Produces accurate
      delta values from real aggressor-classified trades.

    * **OHLCV heuristic** — used when ``tick_df`` is ``None`` or is missing
      required columns.  Approximates buy/sell split from bar OHLCV.

    The ``cvd_source`` column in the result indicates which path was used
    (``"tick"`` or ``"heuristic"``).

    Args:
        df: OHLCV DataFrame with columns High, Low, Close, Volume and a
            DatetimeIndex.
        tick_df: Optional tick DataFrame — normalised output of
                 ``build_tick_df_from_rithmic`` or any DataFrame with
                 columns: timestamp, price, size, side.  When supplied,
                 the tick path is attempted first.
        anchor_daily: Reset CVD accumulation at each new trading day
                      (intraday anchoring).  Set to False for continuous CVD.

    Returns:
        DataFrame with the original columns plus:
          - buy_volume   — estimated / actual buy volume per bar
          - sell_volume  — estimated / actual sell volume per bar
          - delta        — per-bar volume delta (buy − sell)
          - cvd          — cumulative volume delta
          - cvd_ema      — smoothed CVD (EMA-10) for cleaner signals
          - cvd_slope    — rolling normalised slope of CVD
          - cvd_source   — "tick" or "heuristic"
    """
    result = df.copy()

    use_tick_path = False

    if tick_df is not None and not tick_df.empty:
        # Check required columns (case-insensitive)
        tick_cols = set(tick_df.columns.str.lower())
        if {"timestamp", "size", "side"}.issubset(tick_cols):
            use_tick_path = True
        else:
            logger.info(
                "compute_cvd: tick_df present but missing columns %s — using heuristic",
                {"timestamp", "size", "side"} - tick_cols,
            )

    if use_tick_path:
        logger.debug("compute_cvd: using tick data path")
        cvd = _compute_cvd_from_ticks(df, tick_df, anchor_daily=anchor_daily)  # type: ignore[arg-type]

        # Reconstruct per-bar delta from tick cvd
        delta = cvd.diff().fillna(cvd.iloc[0] if len(cvd) > 0 else 0.0)

        # buy/sell breakdown from ticks
        tick_df_norm = tick_df.copy()  # type: ignore[union-attr]
        tick_df_norm.columns = pd.Index([str(c).lower() for c in tick_df_norm.columns])
        if "volume" in tick_df_norm.columns and "size" not in tick_df_norm.columns:
            tick_df_norm = tick_df_norm.rename(columns={"volume": "size"})
        tick_df_norm["size"] = (
            pd.Series(pd.to_numeric(tick_df_norm["size"], errors="coerce"), dtype=float).fillna(0.0).values
        )
        tick_df_norm["timestamp"] = pd.to_datetime(tick_df_norm["timestamp"], errors="coerce")
        tick_df_norm = tick_df_norm.dropna(subset=["timestamp"])
        is_buy = tick_df_norm["side"].astype(str).str.lower().isin(["buy", "b"])

        # Aggregate buy/sell volumes per bar (same interval logic as _compute_cvd_from_ticks)
        bar_times = pd.DatetimeIndex(df.index)
        buy_vols = np.zeros(len(bar_times))
        sell_vols = np.zeros(len(bar_times))

        if len(bar_times) >= 2:
            gaps = bar_times[1:] - bar_times[:-1]
            bar_duration = pd.Timedelta(gaps.value_counts().idxmax())
            ts_arr = np.asarray(tick_df_norm["timestamp"].values, dtype="datetime64[ns]")
            sizes = np.asarray(tick_df_norm["size"].values, dtype=float)
            is_buy_arr = np.asarray(is_buy.values, dtype=bool)

            for i, bar_start in enumerate(bar_times):
                bar_end = bar_start + bar_duration
                mask = (ts_arr >= np.datetime64(bar_start, "ns")) & (ts_arr < np.datetime64(bar_end, "ns"))
                buy_vols[i] = float(sizes[mask & is_buy_arr].sum())
                sell_vols[i] = float(sizes[mask & ~is_buy_arr].sum())

        result["buy_volume"] = buy_vols
        result["sell_volume"] = sell_vols
        result["delta"] = delta.values
        result["cvd"] = cvd.values
        result["cvd_source"] = "tick"
    else:
        logger.debug("compute_cvd: using OHLCV heuristic path")
        high = pd.Series(result["High"].astype(float).values, index=result.index)
        low = pd.Series(result["Low"].astype(float).values, index=result.index)
        close = pd.Series(result["Close"].astype(float).values, index=result.index)
        volume = pd.Series(result["Volume"].astype(float).values, index=result.index)

        buy_vol = _estimate_buy_volume(high, low, close, volume)
        sell_vol = volume - buy_vol
        delta = buy_vol - sell_vol

        result["buy_volume"] = buy_vol.values
        result["sell_volume"] = sell_vol.values
        result["delta"] = delta.values

        if anchor_daily and hasattr(result.index, "date"):
            try:
                dates = result.index.to_series().dt.date
                cvd = delta.groupby(dates).cumsum()
            except Exception:
                cvd = delta.cumsum()
        else:
            cvd = delta.cumsum()

        result["cvd"] = cvd.values
        result["cvd_source"] = "heuristic"

    # Smoothed CVD (EMA-10) for cleaner signals
    cvd_series = pd.Series(result["cvd"].values, index=result.index, dtype=float)
    result["cvd_ema"] = cvd_series.ewm(span=10, adjust=False).mean().values

    # CVD slope: rolling 5-bar rate of change, normalised by rolling std
    cvd_diff = cvd_series.diff(5)
    cvd_std = cvd_series.rolling(20).std()
    result["cvd_slope"] = (cvd_diff / (cvd_std + 1e-10)).values

    return result


# ---------------------------------------------------------------------------
# CVD Divergence detection
# ---------------------------------------------------------------------------


def _find_swing_points(
    series: pd.Series,
    lookback: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Find swing highs and swing lows in a series.

    A swing high is a value strictly greater than the ``lookback`` bars on
    each side.  A swing low is strictly less than.

    Returns:
        (swing_highs, swing_lows) — each a list of (positional_index, value).
    """
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []

    for i in range(lookback, n - lookback):
        if np.isnan(values[i]):
            continue
        window = values[i - lookback : i + lookback + 1]
        if np.any(np.isnan(window)):
            continue
        if values[i] == float(np.nanmax(window)):
            highs.append((i, float(values[i])))
        if values[i] == float(np.nanmin(window)):
            lows.append((i, float(values[i])))

    return highs, lows


def detect_cvd_divergences(
    df: pd.DataFrame,
    lookback: int = 20,
    swing_period: int = 5,
    min_bars_between: int = 5,
) -> list[dict[str, Any]]:
    """Detect divergences between price and CVD.

    Divergence types:
      - **Bullish**: price makes lower low, CVD makes higher low → hidden
        buying pressure, potential reversal upward.
      - **Bearish**: price makes higher high, CVD makes lower high → hidden
        selling pressure, potential reversal downward.

    Works with CVD computed from either the tick path or the heuristic path.

    Args:
        df: DataFrame with ``Close`` and ``cvd`` columns (output of
            ``compute_cvd``).
        lookback: Number of recent bars to scan for divergences.
        swing_period: Bars on each side required to qualify as a swing point.
        min_bars_between: Minimum bar separation between two swing points for a
                          valid divergence pair.

    Returns:
        List of dicts with keys: type, bar_index, datetime, price_1, price_2,
        cvd_1, cvd_2, strength, cvd_source.
    """
    if "cvd" not in df.columns or len(df) < lookback + swing_period * 2:
        return []

    cvd_source = str(df["cvd_source"].iloc[-1]) if "cvd_source" in df.columns else "unknown"

    # Only look at the recent portion
    recent = df.iloc[-lookback - swing_period * 2 :]
    close = recent["Close"].astype(float)
    cvd = recent["cvd"].astype(float)

    price_highs, price_lows = _find_swing_points(close, swing_period)
    cvd_highs, cvd_lows = _find_swing_points(cvd, swing_period)

    divergences: list[dict[str, Any]] = []

    # Bullish divergence: price lower low + CVD higher low
    for i in range(len(price_lows) - 1):
        p1_idx, p1_val = price_lows[i]
        p2_idx, p2_val = price_lows[i + 1]
        if p2_idx - p1_idx < min_bars_between:
            continue
        if p2_val >= p1_val:
            continue  # price must make lower low

        cvd_at_p1 = float(cvd.iloc[p1_idx]) if p1_idx < len(cvd) else np.nan
        cvd_at_p2 = float(cvd.iloc[p2_idx]) if p2_idx < len(cvd) else np.nan

        if np.isnan(cvd_at_p1) or np.isnan(cvd_at_p2):
            continue
        if cvd_at_p2 <= cvd_at_p1:
            continue  # CVD must make higher low

        price_drop = abs(p2_val - p1_val) / (abs(p1_val) + 1e-10) * 100
        cvd_rise = abs(cvd_at_p2 - cvd_at_p1) / (abs(cvd_at_p1) + 1e-10) * 100
        strength = min((price_drop + cvd_rise) / 2, 100)

        orig_idx = recent.index[p2_idx] if p2_idx < len(recent) else None
        divergences.append(
            {
                "type": "bullish",
                "bar_index": p2_idx,
                "datetime": orig_idx,
                "price_1": round(p1_val, 4),
                "price_2": round(p2_val, 4),
                "cvd_1": round(cvd_at_p1, 2),
                "cvd_2": round(cvd_at_p2, 2),
                "strength": round(strength, 1),
                "cvd_source": cvd_source,
            }
        )

    # Bearish divergence: price higher high + CVD lower high
    for i in range(len(price_highs) - 1):
        p1_idx, p1_val = price_highs[i]
        p2_idx, p2_val = price_highs[i + 1]
        if p2_idx - p1_idx < min_bars_between:
            continue
        if p2_val <= p1_val:
            continue  # price must make higher high

        cvd_at_p1 = float(cvd.iloc[p1_idx]) if p1_idx < len(cvd) else np.nan
        cvd_at_p2 = float(cvd.iloc[p2_idx]) if p2_idx < len(cvd) else np.nan

        if np.isnan(cvd_at_p1) or np.isnan(cvd_at_p2):
            continue
        if cvd_at_p2 >= cvd_at_p1:
            continue  # CVD must make lower high

        price_rise = abs(p2_val - p1_val) / (abs(p1_val) + 1e-10) * 100
        cvd_drop = abs(cvd_at_p1 - cvd_at_p2) / (abs(cvd_at_p1) + 1e-10) * 100
        strength = min((price_rise + cvd_drop) / 2, 100)

        orig_idx = recent.index[p2_idx] if p2_idx < len(recent) else None
        divergences.append(
            {
                "type": "bearish",
                "bar_index": p2_idx,
                "datetime": orig_idx,
                "price_1": round(p1_val, 4),
                "price_2": round(p2_val, 4),
                "cvd_1": round(cvd_at_p1, 2),
                "cvd_2": round(cvd_at_p2, 2),
                "strength": round(strength, 1),
                "cvd_source": cvd_source,
            }
        )

    return divergences


# ---------------------------------------------------------------------------
# Absorption candle detection
# ---------------------------------------------------------------------------


def detect_absorption_candles(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.3,
    volume_mult: float = 1.5,
    volume_lookback: int = 20,
) -> pd.Series:
    """Detect volume absorption candles.

    An absorption candle has:
      1. High volume (> ``volume_mult`` × rolling average over
         ``volume_lookback`` bars).
      2. Small body relative to range (body / range < ``body_ratio_threshold``).

    Interpretation:
      +1 = bullish absorption — close near high; buyers absorbing selling pressure.
      -1 = bearish absorption — close near low; sellers absorbing buying pressure.
       0 = no absorption.

    Works with bar data regardless of CVD computation path.

    Args:
        df: DataFrame with OHLCV columns.
        body_ratio_threshold: Maximum body/range ratio to qualify (default 0.3).
        volume_mult: Minimum volume multiplier vs rolling average (default 1.5).
        volume_lookback: Rolling window for volume average (default 20).

    Returns:
        pd.Series of absorption signals (+1, -1, 0) indexed like ``df``.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_price = df["Open"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    candle_range = high - low
    body = (close - open_price).abs()
    body_ratio = body / (candle_range + 1e-10)

    avg_volume = volume.rolling(volume_lookback).mean()
    high_volume = volume > (avg_volume * volume_mult)

    small_body = body_ratio < body_ratio_threshold
    is_absorption = high_volume & small_body & (candle_range > 0)

    # Close position within the bar range determines direction
    close_position = (close - low) / (candle_range + 1e-10)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[is_absorption & (close_position > 0.5)] = 1
    signal.loc[is_absorption & (close_position <= 0.5)] = -1

    return signal


# ---------------------------------------------------------------------------
# CVD trend / momentum helpers
# ---------------------------------------------------------------------------


def cvd_confirms_trend(
    df: pd.DataFrame,
    direction: str = "long",
    slope_threshold: float = 0.5,
) -> bool:
    """Check if the CVD slope confirms a directional bias.

    Args:
        df: DataFrame with ``cvd_slope`` column (output of ``compute_cvd``).
        direction: ``"long"`` or ``"short"``.
        slope_threshold: Minimum absolute slope value for confirmation.

    Returns:
        True if CVD slope confirms the given direction.
    """
    if "cvd_slope" not in df.columns or df.empty:
        return False

    current_slope = float(df["cvd_slope"].iloc[-1])
    if np.isnan(current_slope):
        return False

    if direction == "long":
        return current_slope > slope_threshold
    elif direction == "short":
        return current_slope < -slope_threshold
    return False


def cvd_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a dashboard-ready CVD summary.

    Includes the current CVD value, per-bar delta, slope, directional bias,
    latest absorption signal, and any recent divergences.  The ``cvd_source``
    field indicates whether values came from tick data or the heuristic.

    Args:
        df: DataFrame with CVD columns (output of ``compute_cvd``).

    Returns:
        Dict with keys: cvd_current, delta_current, cvd_slope, bias,
        bias_emoji, absorption, divergences, cvd_source.
    """
    if "cvd" not in df.columns or df.empty:
        return {
            "cvd_current": 0.0,
            "delta_current": 0.0,
            "cvd_slope": 0.0,
            "bias": "neutral",
            "bias_emoji": "⚪",
            "absorption": 0,
            "divergences": [],
            "cvd_source": "unavailable",
        }

    cvd_val = float(df["cvd"].iloc[-1]) if not np.isnan(df["cvd"].iloc[-1]) else 0.0
    delta_val = float(df["delta"].iloc[-1]) if not np.isnan(df["delta"].iloc[-1]) else 0.0
    slope_val = float(df["cvd_slope"].iloc[-1]) if not np.isnan(df["cvd_slope"].iloc[-1]) else 0.0
    cvd_source = str(df["cvd_source"].iloc[-1]) if "cvd_source" in df.columns else "unknown"

    if slope_val > 0.5:
        bias, emoji = "bullish", "🟢"
    elif slope_val < -0.5:
        bias, emoji = "bearish", "🔴"
    else:
        bias, emoji = "neutral", "🟡"

    absorption_signals = detect_absorption_candles(df)
    latest_absorption = int(absorption_signals.iloc[-1]) if len(absorption_signals) > 0 else 0

    divergences = detect_cvd_divergences(df, lookback=30)

    return {
        "cvd_current": round(cvd_val, 2),
        "delta_current": round(delta_val, 2),
        "cvd_slope": round(slope_val, 3),
        "bias": bias,
        "bias_emoji": emoji,
        "absorption": latest_absorption,
        "divergences": divergences,
        "cvd_source": cvd_source,
    }


# ---------------------------------------------------------------------------
# Indicator functions for backtesting.py compatibility
# ---------------------------------------------------------------------------


def _cvd_indicator(high, low, close, volume):  # type: ignore[no-untyped-def]
    """CVD indicator for use with backtesting.py's ``self.I()``.

    Uses the OHLCV heuristic (no tick data in backtesting context).
    Returns CVD as a numpy array.
    """
    h = pd.Series(high, dtype=float)
    lo = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)

    price_range = h - lo
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (c - lo) / safe_range
    buy_ratio = buy_ratio.fillna(0.5).clip(0.0, 1.0)

    delta = v * buy_ratio - v * (1 - buy_ratio)
    return delta.cumsum().values


def _delta_indicator(high, low, close, volume):  # type: ignore[no-untyped-def]
    """Per-bar volume delta indicator for backtesting.py's ``self.I()``.

    Returns delta as a numpy array.
    """
    h = pd.Series(high, dtype=float)
    lo = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)

    price_range = h - lo
    safe_range = price_range.replace(0, np.nan)
    buy_ratio = (c - lo) / safe_range
    buy_ratio = buy_ratio.fillna(0.5).clip(0.0, 1.0)

    buy_vol = v * buy_ratio
    return (buy_vol - (v - buy_vol)).values


def _cvd_ema_indicator(high, low, close, volume, span: int = 10):  # type: ignore[no-untyped-def]
    """Smoothed CVD (EMA) indicator for backtesting.py's ``self.I()``.

    Returns EMA of CVD as a numpy array.
    """
    cvd = pd.Series(_cvd_indicator(high, low, close, volume))
    return cvd.ewm(span=span, adjust=False).mean().values


# ---------------------------------------------------------------------------
# DataFrame-to-display helpers
# ---------------------------------------------------------------------------


def divergences_to_dataframe(divergences: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a divergence list to a display-ready DataFrame.

    Columns: Type, Datetime, Price Move, CVD Move, Strength, Source.
    """
    if not divergences:
        return pd.DataFrame(columns=pd.Index(["Type", "Datetime", "Price Move", "CVD Move", "Strength", "Source"]))

    rows = []
    for d in divergences:
        price_move = f"{d['price_1']:.2f} → {d['price_2']:.2f}"
        cvd_move = f"{d['cvd_1']:.0f} → {d['cvd_2']:.0f}"
        rows.append(
            {
                "Type": d["type"].capitalize(),
                "Datetime": d.get("datetime", ""),
                "Price Move": price_move,
                "CVD Move": cvd_move,
                "Strength": f"{d['strength']:.0f}%",
                "Source": d.get("cvd_source", "unknown"),
            }
        )

    return pd.DataFrame(rows)
