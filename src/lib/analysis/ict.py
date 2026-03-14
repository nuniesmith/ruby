"""
ICT / Smart Money Concepts (SMC) module for futures trading.

Implements four core institutional price-action concepts:

1. **Fair Value Gaps (FVGs)** — Three-candle imbalances where price moved
   so aggressively that the middle candle's range doesn't overlap with the
   combined wicks of candles 1 and 3.  These gaps act as magnets for price
   retracement (~70 % fill rate intraday on ES/NQ).

2. **Order Blocks (OBs)** — The last opposing candle before an impulsive
   move.  Institutional footprints: the final sell candle before a rally
   (bullish OB) or the final buy candle before a drop (bearish OB).
   Price revisiting an OB often triggers continuation.

3. **Liquidity Sweeps** — Engineered stop-hunts where price briefly
   breaches a swing high/low (taking out resting liquidity) and then
   reverses.  Detects both buy-side and sell-side sweeps.

4. **Breaker Blocks** — Former order blocks that failed and were broken
   through.  When price returns to a broken OB, it becomes a breaker —
   a powerful reversal zone because trapped traders need to exit.

All detectors return lists of plain dicts so they are easy to serialise,
cache, display in the dashboard, and overlay on Plotly charts.

Usage:
    from lib.ict import (
        detect_fvgs,
        detect_order_blocks,
        detect_liquidity_sweeps,
        detect_breaker_blocks,
        ict_summary,
    )

    df = get_data("ES=F", "5m", "5d")
    fvgs   = detect_fvgs(df)
    obs    = detect_order_blocks(df)
    sweeps = detect_liquidity_sweeps(df)
    breakers = detect_breaker_blocks(df)
    summary  = ict_summary(df)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from lib.core.utils import safe_float as _safe_float

logger = logging.getLogger("ict")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Extract a named column as a float Series (type-checker safe)."""
    s = df[name]
    assert isinstance(s, pd.Series), f"Expected Series for column {name}"
    return pd.Series(s.to_numpy(dtype=float), index=s.index, name=s.name)


def _swing_highs(high: pd.Series, lookback: int = 5, min_bars_between: int = 3) -> list[dict[str, Any]]:
    """Find swing highs — local maxima with *lookback* bars on each side.

    Returns list of ``{"index": int, "price": float, "timestamp": ...}``.
    """
    swings: list[dict[str, Any]] = []
    last_idx = -min_bars_between - 1
    for i in range(lookback, len(high) - lookback):
        window = high.iloc[i - lookback : i + lookback + 1]
        if float(high.iloc[i]) == float(window.max()) and (i - last_idx) >= min_bars_between:
            ts = high.index[i] if hasattr(high.index, "__getitem__") else i
            swings.append({"index": i, "price": float(high.iloc[i]), "timestamp": ts})
            last_idx = i
    return swings


def _swing_lows(low: pd.Series, lookback: int = 5, min_bars_between: int = 3) -> list[dict[str, Any]]:
    """Find swing lows — local minima with *lookback* bars on each side."""
    swings: list[dict[str, Any]] = []
    last_idx = -min_bars_between - 1
    for i in range(lookback, len(low) - lookback):
        window = low.iloc[i - lookback : i + lookback + 1]
        if float(low.iloc[i]) == float(window.min()) and (i - last_idx) >= min_bars_between:
            ts = low.index[i] if hasattr(low.index, "__getitem__") else i
            swings.append({"index": i, "price": float(low.iloc[i]), "timestamp": ts})
            last_idx = i
    return swings


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)  # noqa: E741
    c = close.to_numpy(dtype=float)
    c_prev = np.roll(c, 1)
    c_prev[0] = np.nan
    tr = np.maximum(np.maximum(h - l, np.abs(h - c_prev)), np.abs(l - c_prev))
    tr_series = pd.Series(tr, index=high.index)
    result = tr_series.ewm(span=length, adjust=False).mean()
    assert isinstance(result, pd.Series)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 1.  FAIR VALUE GAPS  (FVGs)
# ═══════════════════════════════════════════════════════════════════════════


def detect_fvgs(
    df: pd.DataFrame,
    min_gap_atr: float = 0.25,
    atr_period: int = 14,
    max_results: int = 50,
    check_fill: bool = True,
) -> list[dict[str, Any]]:
    """Detect Fair Value Gaps in OHLCV data.

    A **bullish FVG** exists when candle-3's low is above candle-1's high —
    price gapped up so fast that there is an unfilled zone between the
    two wicks.  A **bearish FVG** is the mirror.

    Args:
        df: OHLCV DataFrame (needs High, Low, Close, Open columns).
        min_gap_atr: Minimum gap size as a multiple of ATR to filter
            noise.  Smaller gaps are ignored.
        atr_period: ATR look-back for the filter.
        max_results: Cap on the number of FVGs returned (most recent first).
        check_fill: If True, mark each FVG as filled/unfilled by checking
            whether subsequent price action traded through the gap.

    Returns:
        List of dicts, each containing:
          - type: "bullish" | "bearish"
          - top: upper boundary of the gap
          - bottom: lower boundary of the gap
          - midpoint: (top + bottom) / 2  — optimal fill target
          - size: gap size in price units
          - size_atr: gap size as a fraction of ATR
          - bar_index: index of the middle candle (the imbalance candle)
          - timestamp: timestamp of the middle candle
          - filled: bool — whether price has since traded through the gap
          - fill_pct: 0.0–1.0 — how much of the gap has been filled
    """
    if df.empty or len(df) < max(atr_period + 3, 10):
        return []

    high = _col(df, "High")
    low = _col(df, "Low")
    close = _col(df, "Close")
    atr_series = _atr(high, low, close, atr_period)

    fvgs: list[dict[str, Any]] = []

    for i in range(2, len(df)):
        atr_val = _safe_float(atr_series.iloc[i])
        if atr_val <= 0:
            continue

        h1 = float(high.iloc[i - 2])  # candle 1 high
        l1 = float(low.iloc[i - 2])  # candle 1 low (unused but kept for clarity)
        h3 = float(high.iloc[i])  # candle 3 high
        l3 = float(low.iloc[i])  # candle 3 low

        # Bullish FVG: candle-3 low > candle-1 high  (gap up)
        if l3 > h1:
            gap_size = l3 - h1
            if gap_size / atr_val >= min_gap_atr:
                ts = df.index[i - 1]  # middle candle timestamp
                fvg: dict[str, Any] = {
                    "type": "bullish",
                    "top": l3,
                    "bottom": h1,
                    "midpoint": (l3 + h1) / 2,
                    "size": round(gap_size, 6),
                    "size_atr": round(gap_size / atr_val, 3),
                    "bar_index": i - 1,
                    "timestamp": ts,
                    "filled": False,
                    "fill_pct": 0.0,
                }
                fvgs.append(fvg)

        # Bearish FVG: candle-1 low > candle-3 high  (gap down)
        if l1 > h3:
            gap_size = l1 - h3
            if gap_size / atr_val >= min_gap_atr:
                ts = df.index[i - 1]
                fvg = {
                    "type": "bearish",
                    "top": l1,
                    "bottom": h3,
                    "midpoint": (l1 + h3) / 2,
                    "size": round(gap_size, 6),
                    "size_atr": round(gap_size / atr_val, 3),
                    "bar_index": i - 1,
                    "timestamp": ts,
                    "filled": False,
                    "fill_pct": 0.0,
                }
                fvgs.append(fvg)

    # Check fills  ─────────────────────────────────────────────────────────
    if check_fill and fvgs:
        for fvg in fvgs:
            idx = fvg["bar_index"] + 2  # start checking from candle after gap
            if idx >= len(df):
                continue
            subsequent_high = float(high.iloc[idx:].max())
            subsequent_low = float(low.iloc[idx:].min())

            if fvg["type"] == "bullish":
                # Filled when price drops through the gap (low ≤ bottom)
                penetration = max(0.0, fvg["top"] - subsequent_low)
                gap_range = fvg["top"] - fvg["bottom"]
                fill_pct = min(penetration / gap_range, 1.0) if gap_range > 0 else 0.0
                fvg["fill_pct"] = round(fill_pct, 3)
                fvg["filled"] = subsequent_low <= fvg["bottom"]
            else:
                # Filled when price rallies through the gap (high ≥ top)
                penetration = max(0.0, subsequent_high - fvg["bottom"])
                gap_range = fvg["top"] - fvg["bottom"]
                fill_pct = min(penetration / gap_range, 1.0) if gap_range > 0 else 0.0
                fvg["fill_pct"] = round(fill_pct, 3)
                fvg["filled"] = subsequent_high >= fvg["top"]

    # Return most recent first, capped
    fvgs.reverse()
    return fvgs[:max_results]


def get_unfilled_fvgs(
    df: pd.DataFrame,
    min_gap_atr: float = 0.25,
    atr_period: int = 14,
) -> list[dict[str, Any]]:
    """Convenience: return only unfilled FVGs (active trade targets)."""
    all_fvgs = detect_fvgs(df, min_gap_atr=min_gap_atr, atr_period=atr_period, check_fill=True)
    return [f for f in all_fvgs if not f["filled"]]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ORDER BLOCKS  (OBs)
# ═══════════════════════════════════════════════════════════════════════════


def detect_order_blocks(
    df: pd.DataFrame,
    impulse_atr_mult: float = 1.5,
    atr_period: int = 14,
    lookback: int = 3,
    max_results: int = 30,
    check_mitigated: bool = True,
) -> list[dict[str, Any]]:
    """Detect Order Blocks — the last opposing candle before an impulsive move.

    **Bullish OB**: the last bearish (red) candle before a strong up-move
    that displaces price by ≥ *impulse_atr_mult* × ATR.

    **Bearish OB**: the last bullish (green) candle before a strong down-move.

    Args:
        df: OHLCV DataFrame.
        impulse_atr_mult: The impulsive move after the OB candle must be
            at least this many ATRs to qualify.
        atr_period: ATR look-back.
        lookback: How many candles back to search for the opposing candle
            before the impulse.
        max_results: Cap returned results.
        check_mitigated: If True, mark OBs as mitigated when price
            later returns and trades through the OB zone.

    Returns:
        List of dicts with: type, high, low, midpoint, bar_index,
        timestamp, impulse_size, impulse_atr, mitigated, tested.
    """
    if df.empty or len(df) < atr_period + lookback + 5:
        return []

    high = _col(df, "High")
    low = _col(df, "Low")
    close = _col(df, "Close")
    open_price = _col(df, "Open")
    atr_series = _atr(high, low, close, atr_period)

    obs: list[dict[str, Any]] = []

    for i in range(lookback + 1, len(df) - 1):
        atr_val = _safe_float(atr_series.iloc[i])
        if atr_val <= 0:
            continue

        curr_close = float(close.iloc[i])
        curr_open = float(open_price.iloc[i])

        # Detect impulsive bullish candle (large green)
        if curr_close > curr_open:
            move = curr_close - float(low.iloc[i])
            if move >= impulse_atr_mult * atr_val:
                # Search backward for the last bearish candle
                for j in range(i - 1, max(i - lookback - 1, 0) - 1, -1):
                    if j < 0:
                        break
                    ob_close = float(close.iloc[j])
                    ob_open = float(open_price.iloc[j])
                    if ob_close < ob_open:  # bearish candle = bullish OB
                        ob_high = float(high.iloc[j])
                        ob_low = float(low.iloc[j])
                        ts = df.index[j]
                        ob: dict[str, Any] = {
                            "type": "bullish",
                            "high": ob_high,
                            "low": ob_low,
                            "midpoint": (ob_high + ob_low) / 2,
                            "bar_index": j,
                            "timestamp": ts,
                            "impulse_size": round(move, 6),
                            "impulse_atr": round(move / atr_val, 2),
                            "mitigated": False,
                            "tested": False,
                        }
                        obs.append(ob)
                        break

        # Detect impulsive bearish candle (large red)
        if curr_close < curr_open:
            move = float(high.iloc[i]) - curr_close
            if move >= impulse_atr_mult * atr_val:
                for j in range(i - 1, max(i - lookback - 1, 0) - 1, -1):
                    if j < 0:
                        break
                    ob_close = float(close.iloc[j])
                    ob_open = float(open_price.iloc[j])
                    if ob_close > ob_open:  # bullish candle = bearish OB
                        ob_high = float(high.iloc[j])
                        ob_low = float(low.iloc[j])
                        ts = df.index[j]
                        ob = {
                            "type": "bearish",
                            "high": ob_high,
                            "low": ob_low,
                            "midpoint": (ob_high + ob_low) / 2,
                            "bar_index": j,
                            "timestamp": ts,
                            "impulse_size": round(move, 6),
                            "impulse_atr": round(move / atr_val, 2),
                            "mitigated": False,
                            "tested": False,
                        }
                        obs.append(ob)
                        break

    # Deduplicate OBs at the same bar_index (keep the one with larger impulse)
    seen: dict[int, int] = {}
    deduped: list[dict[str, Any]] = []
    for _idx, ob in enumerate(obs):
        bi = ob["bar_index"]
        if bi in seen:
            existing = deduped[seen[bi]]
            if ob["impulse_atr"] > existing["impulse_atr"]:
                deduped[seen[bi]] = ob
        else:
            seen[bi] = len(deduped)
            deduped.append(ob)
    obs = deduped

    # Check mitigation / tested status
    if check_mitigated and obs:
        for ob in obs:
            start = ob["bar_index"] + 2
            if start >= len(df):
                continue
            subsequent_low = float(low.iloc[start:].min())
            subsequent_high = float(high.iloc[start:].max())

            if ob["type"] == "bullish":
                # Tested if price dipped into OB zone
                if subsequent_low <= ob["high"]:
                    ob["tested"] = True
                # Mitigated if price traded through the entire OB
                if subsequent_low <= ob["low"]:
                    ob["mitigated"] = True
            else:  # bearish
                if subsequent_high >= ob["low"]:
                    ob["tested"] = True
                if subsequent_high >= ob["high"]:
                    ob["mitigated"] = True

    obs.reverse()
    return obs[:max_results]


def get_active_order_blocks(
    df: pd.DataFrame,
    impulse_atr_mult: float = 1.5,
    atr_period: int = 14,
) -> list[dict[str, Any]]:
    """Return only non-mitigated (still active) order blocks."""
    all_obs = detect_order_blocks(
        df,
        impulse_atr_mult=impulse_atr_mult,
        atr_period=atr_period,
        check_mitigated=True,
    )
    return [ob for ob in all_obs if not ob["mitigated"]]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LIQUIDITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════════


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    min_reversal_atr: float = 0.5,
    atr_period: int = 14,
    max_results: int = 30,
) -> list[dict[str, Any]]:
    """Detect liquidity sweeps — engineered stop-hunts beyond swing points.

    A **buy-side sweep** occurs when price spikes above a prior swing
    high (taking out buy-stops) and then reverses back below.

    A **sell-side sweep** occurs when price drops below a prior swing
    low (taking out sell-stops) and then reverses back above.

    The reversal must be at least *min_reversal_atr* × ATR from the
    sweep extreme to qualify as a genuine rejection.

    Args:
        df: OHLCV DataFrame.
        swing_lookback: Bars on each side to define swing highs/lows.
        min_reversal_atr: Minimum reversal size (in ATRs) after the sweep.
        atr_period: ATR look-back.
        max_results: Cap returned results.

    Returns:
        List of dicts with: type ("buy_side" / "sell_side"), sweep_price,
        swing_price, reversal_size, bar_index, timestamp, swept_by.
    """
    if df.empty or len(df) < swing_lookback * 2 + atr_period + 5:
        return []

    high = _col(df, "High")
    low = _col(df, "Low")
    close = _col(df, "Close")
    atr_series = _atr(high, low, close, atr_period)

    swing_highs = _swing_highs(high, swing_lookback)
    swing_lows = _swing_lows(low, swing_lookback)

    sweeps: list[dict[str, Any]] = []

    # Buy-side sweeps (price exceeds a swing high then reverses)
    for sh in swing_highs:
        sh_idx = sh["index"]
        sh_price = sh["price"]

        # Look for bars after this swing high that breach it
        search_start = sh_idx + swing_lookback
        if search_start >= len(df):
            continue

        for i in range(search_start, min(search_start + 30, len(df))):
            bar_high = float(high.iloc[i])
            bar_close = float(close.iloc[i])
            atr_val = _safe_float(atr_series.iloc[i])

            if atr_val <= 0:
                continue

            # Price must exceed the swing high
            if bar_high > sh_price:
                swept_by = bar_high - sh_price
                # But close must be back below the swing high (rejection)
                if bar_close < sh_price:
                    reversal = bar_high - bar_close
                    if reversal >= min_reversal_atr * atr_val:
                        ts = df.index[i]
                        sweeps.append(
                            {
                                "type": "buy_side",
                                "sweep_price": round(bar_high, 6),
                                "swing_price": round(sh_price, 6),
                                "reversal_size": round(reversal, 6),
                                "reversal_atr": round(reversal / atr_val, 2),
                                "swept_by": round(swept_by, 6),
                                "bar_index": i,
                                "timestamp": ts,
                                "swing_timestamp": sh["timestamp"],
                            }
                        )
                        break  # only capture first sweep of this swing

    # Sell-side sweeps (price drops below a swing low then reverses)
    for sl in swing_lows:
        sl_idx = sl["index"]
        sl_price = sl["price"]

        search_start = sl_idx + swing_lookback
        if search_start >= len(df):
            continue

        for i in range(search_start, min(search_start + 30, len(df))):
            bar_low = float(low.iloc[i])
            bar_close = float(close.iloc[i])
            atr_val = _safe_float(atr_series.iloc[i])

            if atr_val <= 0:
                continue

            if bar_low < sl_price:
                swept_by = sl_price - bar_low
                if bar_close > sl_price:
                    reversal = bar_close - bar_low
                    if reversal >= min_reversal_atr * atr_val:
                        ts = df.index[i]
                        sweeps.append(
                            {
                                "type": "sell_side",
                                "sweep_price": round(bar_low, 6),
                                "swing_price": round(sl_price, 6),
                                "reversal_size": round(reversal, 6),
                                "reversal_atr": round(reversal / atr_val, 2),
                                "swept_by": round(swept_by, 6),
                                "bar_index": i,
                                "timestamp": ts,
                                "swing_timestamp": sl["timestamp"],
                            }
                        )
                        break

    # Sort by bar_index descending (most recent first)
    sweeps.sort(key=lambda s: s["bar_index"], reverse=True)
    return sweeps[:max_results]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  BREAKER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════


def detect_breaker_blocks(
    df: pd.DataFrame,
    impulse_atr_mult: float = 1.5,
    atr_period: int = 14,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Detect Breaker Blocks — failed order blocks that were traded through.

    A breaker forms when:
    1. An order block is identified.
    2. Price returns and *fully* trades through the OB (mitigating it).
    3. The broken OB zone now acts as a reversal level from the opposite
       direction (role reversal).

    A **bullish breaker** is a former bearish OB that was broken to the
    upside — it now acts as support.

    A **bearish breaker** is a former bullish OB that was broken to the
    downside — it now acts as resistance.

    Args:
        df: OHLCV DataFrame.
        impulse_atr_mult: Passed through to order-block detection.
        atr_period: ATR look-back.
        max_results: Cap returned results.

    Returns:
        List of dicts with: type ("bullish" / "bearish"), high, low,
        midpoint, ob_bar_index, break_bar_index, timestamps, retested.
    """
    if df.empty or len(df) < atr_period + 15:
        return []

    # Get all order blocks including mitigated ones
    all_obs = detect_order_blocks(
        df,
        impulse_atr_mult=impulse_atr_mult,
        atr_period=atr_period,
        check_mitigated=True,
        max_results=100,
    )

    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    breakers: list[dict[str, Any]] = []

    for ob in all_obs:
        if not ob["mitigated"]:
            continue  # only mitigated OBs become breakers

        start = ob["bar_index"] + 2
        if start >= len(df):
            continue

        # Find the bar that broke through the OB
        break_bar = None
        if ob["type"] == "bullish":
            # A bullish OB that was mitigated means price dropped through it.
            # The broken bullish OB becomes a *bearish* breaker (resistance).
            for k in range(start, len(df)):
                if float(low.iloc[k]) <= ob["low"]:
                    break_bar = k
                    break
            if break_bar is None:
                continue

            breaker_type = "bearish"

            # Check if breaker has been retested (price rallied back to zone)
            retested = False
            retest_start = break_bar + 1
            if retest_start < len(df):
                subsequent_high = float(high.iloc[retest_start:].max())
                if subsequent_high >= ob["low"]:
                    retested = True

        else:
            # A bearish OB that was mitigated means price rallied through it.
            # The broken bearish OB becomes a *bullish* breaker (support).
            for k in range(start, len(df)):
                if float(high.iloc[k]) >= ob["high"]:
                    break_bar = k
                    break
            if break_bar is None:
                continue

            breaker_type = "bullish"

            retested = False
            retest_start = break_bar + 1
            if retest_start < len(df):
                subsequent_low = float(low.iloc[retest_start:].min())
                if subsequent_low <= ob["high"]:
                    retested = True

        ts_break = df.index[break_bar] if break_bar < len(df) else None

        breakers.append(
            {
                "type": breaker_type,
                "high": ob["high"],
                "low": ob["low"],
                "midpoint": ob["midpoint"],
                "ob_bar_index": ob["bar_index"],
                "ob_timestamp": ob["timestamp"],
                "break_bar_index": break_bar,
                "break_timestamp": ts_break,
                "original_ob_type": ob["type"],
                "retested": retested,
            }
        )

    breakers.sort(key=lambda b: b["break_bar_index"], reverse=True)
    return breakers[:max_results]


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════


def ict_summary(
    df: pd.DataFrame,
    atr_period: int = 14,
) -> dict[str, Any]:
    """Run all ICT detectors and return a combined summary.

    Useful for quick dashboard display and alerting.

    Returns a dict with counts, nearest levels, and full result lists.
    """
    if df.empty or len(df) < 30:
        return {
            "fvgs": [],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "breaker_blocks": [],
            "stats": {
                "total_fvgs": 0,
                "unfilled_fvgs": 0,
                "active_obs": 0,
                "recent_sweeps": 0,
                "breakers": 0,
            },
            "nearest_levels": {},
        }

    fvgs = detect_fvgs(df, atr_period=atr_period)
    obs = detect_order_blocks(df, atr_period=atr_period)
    sweeps = detect_liquidity_sweeps(df, atr_period=atr_period)
    breakers = detect_breaker_blocks(df, atr_period=atr_period)

    unfilled = [f for f in fvgs if not f["filled"]]
    active_obs = [o for o in obs if not o["mitigated"]]

    current_price = float(df["Close"].iloc[-1])

    # Find nearest ICT levels above and below current price
    all_levels: list[dict[str, Any]] = []

    for f in unfilled:
        all_levels.append(
            {
                "price": f["midpoint"],
                "label": f"FVG {'▲' if f['type'] == 'bullish' else '▼'} midpoint",
                "type": f"fvg_{f['type']}",
            }
        )

    for o in active_obs:
        all_levels.append(
            {
                "price": o["midpoint"],
                "label": f"OB {'▲' if o['type'] == 'bullish' else '▼'} zone",
                "type": f"ob_{o['type']}",
            }
        )

    for b in breakers:
        all_levels.append(
            {
                "price": b["midpoint"],
                "label": f"Breaker {'▲' if b['type'] == 'bullish' else '▼'}",
                "type": f"breaker_{b['type']}",
            }
        )

    # Sort by distance from current price
    for lv in all_levels:
        lv["distance"] = lv["price"] - current_price

    above = sorted(
        [lv for lv in all_levels if lv["distance"] > 0],
        key=lambda x: x["distance"],
    )
    below = sorted(
        [lv for lv in all_levels if lv["distance"] <= 0],
        key=lambda x: abs(x["distance"]),
    )

    nearest: dict[str, Any] = {}
    if above:
        nearest["above"] = {
            "price": round(above[0]["price"], 4),
            "label": above[0]["label"],
            "distance": round(above[0]["distance"], 4),
        }
    if below:
        nearest["below"] = {
            "price": round(below[0]["price"], 4),
            "label": below[0]["label"],
            "distance": round(below[0]["distance"], 4),
        }

    return {
        "fvgs": fvgs,
        "order_blocks": obs,
        "liquidity_sweeps": sweeps,
        "breaker_blocks": breakers,
        "stats": {
            "total_fvgs": len(fvgs),
            "unfilled_fvgs": len(unfilled),
            "bullish_fvgs": len([f for f in unfilled if f["type"] == "bullish"]),
            "bearish_fvgs": len([f for f in unfilled if f["type"] == "bearish"]),
            "total_obs": len(obs),
            "active_obs": len(active_obs),
            "bullish_obs": len([o for o in active_obs if o["type"] == "bullish"]),
            "bearish_obs": len([o for o in active_obs if o["type"] == "bearish"]),
            "recent_sweeps": len(sweeps),
            "buy_side_sweeps": len([s for s in sweeps if s["type"] == "buy_side"]),
            "sell_side_sweeps": len([s for s in sweeps if s["type"] == "sell_side"]),
            "breakers": len(breakers),
        },
        "nearest_levels": nearest,
        "current_price": round(current_price, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATAFRAME CONVERTERS
# ═══════════════════════════════════════════════════════════════════════════


def fvgs_to_dataframe(fvgs: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert FVG list to a display-friendly DataFrame."""
    if not fvgs:
        return pd.DataFrame()
    rows = []
    for f in fvgs:
        emoji = "🟢" if f["type"] == "bullish" else "🔴"
        status = "Filled" if f["filled"] else f"Open ({f['fill_pct']:.0%} filled)"
        rows.append(
            {
                "Type": f"{emoji} {f['type'].title()}",
                "Top": round(f["top"], 4),
                "Bottom": round(f["bottom"], 4),
                "Midpoint": round(f["midpoint"], 4),
                "Size (ATR)": f"{f['size_atr']:.2f}×",
                "Status": status,
                "Time": f["timestamp"],
            }
        )
    return pd.DataFrame(rows)


def order_blocks_to_dataframe(obs: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert order block list to a display-friendly DataFrame."""
    if not obs:
        return pd.DataFrame()
    rows = []
    for o in obs:
        emoji = "🟢" if o["type"] == "bullish" else "🔴"
        status = "Mitigated" if o["mitigated"] else ("Tested" if o["tested"] else "Active")
        rows.append(
            {
                "Type": f"{emoji} {o['type'].title()}",
                "High": round(o["high"], 4),
                "Low": round(o["low"], 4),
                "Impulse": f"{o['impulse_atr']:.1f}× ATR",
                "Status": status,
                "Time": o["timestamp"],
            }
        )
    return pd.DataFrame(rows)


def sweeps_to_dataframe(sweeps: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert liquidity sweep list to a display-friendly DataFrame."""
    if not sweeps:
        return pd.DataFrame()
    rows = []
    for s in sweeps:
        emoji = "⬆️" if s["type"] == "buy_side" else "⬇️"
        rows.append(
            {
                "Type": f"{emoji} {s['type'].replace('_', ' ').title()}",
                "Sweep Price": round(s["sweep_price"], 4),
                "Swing Level": round(s["swing_price"], 4),
                "Swept By": round(s["swept_by"], 4),
                "Reversal": f"{s['reversal_atr']:.1f}× ATR",
                "Time": s["timestamp"],
            }
        )
    return pd.DataFrame(rows)


def breakers_to_dataframe(breakers: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert breaker block list to a display-friendly DataFrame."""
    if not breakers:
        return pd.DataFrame()
    rows = []
    for b in breakers:
        emoji = "🟢" if b["type"] == "bullish" else "🔴"
        status = "Retested ✅" if b["retested"] else "Pending"
        rows.append(
            {
                "Type": f"{emoji} {b['type'].title()}",
                "High": round(b["high"], 4),
                "Low": round(b["low"], 4),
                "Original OB": b["original_ob_type"].title(),
                "Status": status,
                "Break Time": b["break_timestamp"],
            }
        )
    return pd.DataFrame(rows)


def levels_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Build a combined levels table from an ict_summary() result.

    Merges unfilled FVGs, active OBs, and breakers into a single
    sorted-by-distance table useful for the Signals tab.
    """
    current = summary.get("current_price", 0)
    rows: list[dict[str, Any]] = []

    for f in summary.get("fvgs", []):
        if f["filled"]:
            continue
        dist = f["midpoint"] - current
        rows.append(
            {
                "Level": round(f["midpoint"], 4),
                "Zone": f"{round(f['bottom'], 4)} – {round(f['top'], 4)}",
                "Concept": f"FVG {'▲' if f['type'] == 'bullish' else '▼'}",
                "Distance": round(dist, 4),
                "Strength": f"{f['size_atr']:.2f}× ATR",
            }
        )

    for o in summary.get("order_blocks", []):
        if o["mitigated"]:
            continue
        dist = o["midpoint"] - current
        rows.append(
            {
                "Level": round(o["midpoint"], 4),
                "Zone": f"{round(o['low'], 4)} – {round(o['high'], 4)}",
                "Concept": f"OB {'▲' if o['type'] == 'bullish' else '▼'}",
                "Distance": round(dist, 4),
                "Strength": f"{o['impulse_atr']:.1f}× ATR impulse",
            }
        )

    for b in summary.get("breaker_blocks", []):
        dist = b["midpoint"] - current
        retest_str = " (retested)" if b["retested"] else ""
        rows.append(
            {
                "Level": round(b["midpoint"], 4),
                "Zone": f"{round(b['low'], 4)} – {round(b['high'], 4)}",
                "Concept": f"Breaker {'▲' if b['type'] == 'bullish' else '▼'}{retest_str}",
                "Distance": round(dist, 4),
                "Strength": "—",
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values("Distance", key=abs, ascending=True)
    return result.reset_index(drop=True)
