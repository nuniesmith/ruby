"""
Multi-timeframe confluence filter for futures trading.

Implements the 3-layer confluence system from the notes.md blueprint:
  1. Higher Timeframe (HTF) — directional bias (EMA alignment on 1H/15m)
  2. Setup Timeframe — pattern identification (15m/5m)
  3. Entry Timeframe — precise timing (5m/1m)

All three layers must agree on direction for a trade to qualify.
Confluence is scored 0–3; only trade on 3/3 alignment.

Recommended timeframe combinations (per notes.md):
  - ES/NQ (equity indices):  15m / 5m / 1m
  - GC (gold):               1H / 15m / 5m
  - CL (crude oil):          15m / 5m / 3m
  - SI/HG (silver/copper):   1H / 15m / 5m

The key rule is to maintain a 4–6× factor between timeframes (not 5/10/15,
which are too close together).

Usage:
    from lib.confluence import (
        MultiTimeframeFilter,
        check_confluence,
        get_recommended_timeframes,
        confluence_summary,
    )

    mtf = MultiTimeframeFilter()
    result = mtf.evaluate(
        htf_df=df_15m,
        setup_df=df_5m,
        entry_df=df_1m,
        asset_name="Gold",
    )
    if result["score"] == 3:
        print(f"Full confluence {result['direction']}!")
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from lib.core.utils import atr as _atr
from lib.core.utils import ema as _ema
from lib.core.utils import rsi as _rsi

logger = logging.getLogger("confluence")


# ---------------------------------------------------------------------------
# Recommended timeframe mappings per instrument
# ---------------------------------------------------------------------------

# Each tuple: (htf_interval, setup_interval, entry_interval)
TIMEFRAME_PRESETS = {
    "S&P": ("15m", "5m", "1m"),
    "Nasdaq": ("15m", "5m", "1m"),
    "Gold": ("1h", "15m", "5m"),
    "Silver": ("1h", "15m", "5m"),
    "Copper": ("1h", "15m", "5m"),
    "Crude Oil": ("15m", "5m", "1m"),
}

# EMA periods for each layer
DEFAULT_EMA_FAST = 10
DEFAULT_EMA_MID = 20
DEFAULT_EMA_SLOW = 50

# RSI period for entry-layer momentum confirmation
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_BULLISH_THRESHOLD = 45  # above this → bullish ok
DEFAULT_RSI_BEARISH_THRESHOLD = 55  # below this → bearish ok


# ---------------------------------------------------------------------------
# Layer 1: Higher Timeframe — Directional Bias
# ---------------------------------------------------------------------------


def evaluate_htf_bias(
    df: pd.DataFrame,
    ema_fast: int = DEFAULT_EMA_FAST,
    ema_mid: int = DEFAULT_EMA_MID,
    ema_slow: int = DEFAULT_EMA_SLOW,
) -> dict[str, Any]:
    """Determine directional bias from higher timeframe EMA alignment.

    Bullish bias: EMA_fast > EMA_mid > EMA_slow (stacked up)
    Bearish bias: EMA_fast < EMA_mid < EMA_slow (stacked down)
    Neutral: EMAs not aligned

    Also checks price position relative to EMAs for confirmation.

    Returns:
        Dict with direction ("bullish", "bearish", "neutral"),
        ema values, alignment quality, and confidence.
    """
    if df.empty or len(df) < ema_slow + 5:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "ema_fast": np.nan,
            "ema_mid": np.nan,
            "ema_slow": np.nan,
            "price_above_emas": False,
            "emas_stacked": False,
            "slope": 0.0,
        }

    close = pd.Series(df["Close"].astype(float))
    fast = _ema(close, ema_fast)
    mid = _ema(close, ema_mid)
    slow = _ema(close, ema_slow)

    fast_val = float(fast.iloc[-1])
    mid_val = float(mid.iloc[-1])
    slow_val = float(slow.iloc[-1])
    price = float(close.iloc[-1])

    # Check EMA stacking order
    bullish_stack = fast_val > mid_val > slow_val
    bearish_stack = fast_val < mid_val < slow_val

    # Check price position
    price_above_all = price > fast_val and price > mid_val and price > slow_val
    price_below_all = price < fast_val and price < mid_val and price < slow_val

    # Slope of slow EMA (momentum of the bias)
    slope = (float(slow.iloc[-1]) - float(slow.iloc[-5])) / (float(slow.iloc[-5]) + 1e-10) if len(slow) >= 5 else 0.0

    # Determine direction
    if bullish_stack:
        direction = "bullish"
        # Confidence based on how spread apart the EMAs are and price position
        spread = (fast_val - slow_val) / (slow_val + 1e-10)
        confidence = min(abs(spread) * 500 + (0.3 if price_above_all else 0.0), 1.0)
    elif bearish_stack:
        direction = "bearish"
        spread = (slow_val - fast_val) / (slow_val + 1e-10)
        confidence = min(abs(spread) * 500 + (0.3 if price_below_all else 0.0), 1.0)
    else:
        direction = "neutral"
        confidence = 0.0

    return {
        "direction": direction,
        "confidence": round(confidence, 3),
        "ema_fast": round(fast_val, 4),
        "ema_mid": round(mid_val, 4),
        "ema_slow": round(slow_val, 4),
        "price_above_emas": price_above_all,
        "emas_stacked": bullish_stack or bearish_stack,
        "slope": round(slope, 6),
    }


# ---------------------------------------------------------------------------
# Layer 2: Setup Timeframe — Pattern Identification
# ---------------------------------------------------------------------------


def evaluate_setup(
    df: pd.DataFrame,
    ema_fast: int = DEFAULT_EMA_FAST,
    ema_mid: int = DEFAULT_EMA_MID,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> dict[str, Any]:
    """Evaluate the setup timeframe for pattern quality.

    Checks for:
      - EMA alignment on setup TF (fast > mid for bullish)
      - Price interaction with EMAs (pullback to EMA = good setup)
      - RSI not in extreme (avoids exhaustion entries)
      - Recent higher highs / lower lows for trend confirmation

    Returns:
        Dict with direction, setup quality, pullback status, and details.
    """
    if df.empty or len(df) < ema_mid + 10:
        return {
            "direction": "neutral",
            "quality": 0.0,
            "has_pullback": False,
            "rsi": 50.0,
            "trend_bars": 0,
            "ema_fast": np.nan,
            "ema_mid": np.nan,
        }

    close = pd.Series(df["Close"].astype(float))
    high = pd.Series(df["High"].astype(float))
    low = pd.Series(df["Low"].astype(float))

    fast = _ema(close, ema_fast)
    mid = _ema(close, ema_mid)
    rsi = _rsi(close, rsi_period)

    fast_val = float(fast.iloc[-1])
    mid_val = float(mid.iloc[-1])
    rsi_val = float(rsi.iloc[-1])
    _ = float(close.iloc[-1])  # price (used implicitly via EMA comparison)

    # EMA alignment
    bullish_align = fast_val > mid_val
    bearish_align = fast_val < mid_val

    # Pullback detection: price recently touched or came within 0.1% of EMA
    atr_series = _atr(high, low, close, 14)
    atr_val = float(atr_series.iloc[-1])
    pullback_zone = atr_val * 0.5  # within half an ATR of the EMA

    # Check last 5 bars for pullback to fast EMA
    has_bullish_pullback = False
    has_bearish_pullback = False
    for i in range(-5, 0):
        if i < -len(df):
            continue
        bar_low = float(low.iloc[i])
        bar_high = float(high.iloc[i])
        ema_at_bar = float(fast.iloc[i])
        if abs(bar_low - ema_at_bar) < pullback_zone and bar_low <= ema_at_bar:
            has_bullish_pullback = True
        if abs(bar_high - ema_at_bar) < pullback_zone and bar_high >= ema_at_bar:
            has_bearish_pullback = True

    # Trend bars: count consecutive bars where close > fast EMA (bullish) or < (bearish)
    bullish_count = 0
    bearish_count = 0
    for i in range(-1, max(-21, -len(close)), -1):
        c = float(close.iloc[i])
        f = float(fast.iloc[i])
        if c > f:
            bullish_count += 1
        elif c < f:
            bearish_count += 1
        else:
            break

    # RSI filter: avoid exhaustion
    rsi_ok_bull = rsi_val < 70  # not overbought
    rsi_ok_bear = rsi_val > 30  # not oversold

    # Determine direction and quality
    if bullish_align:
        direction = "bullish"
        quality = 0.4  # base for alignment
        if has_bullish_pullback:
            quality += 0.3  # pullback adds setup quality
        if rsi_ok_bull and rsi_val > DEFAULT_RSI_BULLISH_THRESHOLD:
            quality += 0.15
        if bullish_count >= 3:
            quality += 0.15
        quality = min(quality, 1.0)
    elif bearish_align:
        direction = "bearish"
        quality = 0.4
        if has_bearish_pullback:
            quality += 0.3
        if rsi_ok_bear and rsi_val < DEFAULT_RSI_BEARISH_THRESHOLD:
            quality += 0.15
        if bearish_count >= 3:
            quality += 0.15
        quality = min(quality, 1.0)
    else:
        direction = "neutral"
        quality = 0.0

    return {
        "direction": direction,
        "quality": round(quality, 3),
        "has_pullback": has_bullish_pullback if direction == "bullish" else has_bearish_pullback,
        "rsi": round(rsi_val, 1),
        "trend_bars": bullish_count if direction == "bullish" else bearish_count,
        "ema_fast": round(fast_val, 4),
        "ema_mid": round(mid_val, 4),
    }


# ---------------------------------------------------------------------------
# Layer 3: Entry Timeframe — Timing & Trigger
# ---------------------------------------------------------------------------


def evaluate_entry(
    df: pd.DataFrame,
    ema_fast: int = 9,
    ema_mid: int = 21,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> dict[str, Any]:
    """Evaluate the entry timeframe for immediate timing signals.

    Checks for:
      - Short-term EMA alignment and crossover
      - RSI momentum confirmation
      - Candlestick patterns (bullish/bearish engulfing, hammer, shooting star)
      - Volume confirmation (above average)

    Returns:
        Dict with direction, trigger status, candle pattern, and details.
    """
    if df.empty or len(df) < ema_mid + 5:
        return {
            "direction": "neutral",
            "trigger": False,
            "candle_pattern": "none",
            "rsi": 50.0,
            "volume_confirmed": False,
            "ema_cross_recent": False,
        }

    close = pd.Series(df["Close"].astype(float))
    open_price = pd.Series(df["Open"].astype(float))
    high = pd.Series(df["High"].astype(float))
    low = pd.Series(df["Low"].astype(float))
    volume = pd.Series(df["Volume"].astype(float))

    fast = _ema(close, ema_fast)
    mid = _ema(close, ema_mid)
    rsi = _rsi(close, rsi_period)

    fast_val = float(fast.iloc[-1])
    mid_val = float(mid.iloc[-1])
    rsi_val = float(rsi.iloc[-1])

    # EMA alignment
    bullish_align = fast_val > mid_val
    bearish_align = fast_val < mid_val

    # Recent EMA crossover (within last 3 bars)
    ema_cross_bull = False
    ema_cross_bear = False
    for i in range(-3, 0):
        if i - 1 < -len(fast):
            continue
        prev_fast = float(fast.iloc[i - 1])
        prev_mid = float(mid.iloc[i - 1])
        curr_fast = float(fast.iloc[i])
        curr_mid = float(mid.iloc[i])
        if prev_fast <= prev_mid and curr_fast > curr_mid:
            ema_cross_bull = True
        if prev_fast >= prev_mid and curr_fast < curr_mid:
            ema_cross_bear = True

    # Candlestick patterns (last bar)
    candle_pattern = "none"
    if len(df) >= 2:
        curr_open = float(open_price.iloc[-1])
        curr_close = float(close.iloc[-1])
        curr_high = float(high.iloc[-1])
        curr_low = float(low.iloc[-1])
        prev_open = float(open_price.iloc[-2])
        prev_close = float(close.iloc[-2])
        _prev_high = float(high.iloc[-2])  # noqa: F841
        _prev_low = float(low.iloc[-2])  # noqa: F841

        curr_body = abs(curr_close - curr_open)
        curr_range = curr_high - curr_low
        prev_body = abs(prev_close - prev_open)

        if curr_range > 0:
            body_ratio = curr_body / curr_range

            # Bullish engulfing: prev bearish + current bullish + current body > prev body
            if (
                prev_close < prev_open
                and curr_close > curr_open
                and curr_body > prev_body
                and curr_close > prev_open
                and curr_open <= prev_close
            ):
                candle_pattern = "bullish_engulfing"

            # Bearish engulfing
            elif (
                prev_close > prev_open
                and curr_close < curr_open
                and curr_body > prev_body
                and curr_close < prev_open
                and curr_open >= prev_close
            ):
                candle_pattern = "bearish_engulfing"

            # Hammer (bullish): small body at top, long lower wick
            elif (
                body_ratio < 0.35
                and (curr_close - curr_low) > 2 * curr_body
                and (curr_high - max(curr_open, curr_close)) < curr_body * 0.5
            ):
                candle_pattern = "hammer"

            # Shooting star (bearish): small body at bottom, long upper wick
            elif (
                body_ratio < 0.35
                and (curr_high - curr_close) > 2 * curr_body
                and (min(curr_open, curr_close) - curr_low) < curr_body * 0.5
            ):
                candle_pattern = "shooting_star"

    # Volume confirmation: above 20-bar average
    avg_vol = float(pd.Series(volume.rolling(20).mean()).iloc[-1]) if len(volume) >= 20 else 0
    current_vol = float(volume.iloc[-1])
    volume_confirmed = current_vol > avg_vol * 1.0 if avg_vol > 0 else False

    # RSI momentum
    rsi_bull = rsi_val > DEFAULT_RSI_BULLISH_THRESHOLD
    rsi_bear = rsi_val < DEFAULT_RSI_BEARISH_THRESHOLD

    # Determine direction and trigger
    bull_candle = candle_pattern in ("bullish_engulfing", "hammer")
    bear_candle = candle_pattern in ("bearish_engulfing", "shooting_star")

    if bullish_align and rsi_bull:
        direction = "bullish"
        trigger = (ema_cross_bull or bull_candle) and volume_confirmed
    elif bearish_align and rsi_bear:
        direction = "bearish"
        trigger = (ema_cross_bear or bear_candle) and volume_confirmed
    else:
        direction = "neutral"
        trigger = False

    return {
        "direction": direction,
        "trigger": trigger,
        "candle_pattern": candle_pattern,
        "rsi": round(rsi_val, 1),
        "volume_confirmed": volume_confirmed,
        "ema_cross_recent": ema_cross_bull or ema_cross_bear,
        "ema_fast": round(fast_val, 4),
        "ema_mid": round(mid_val, 4),
    }


# ---------------------------------------------------------------------------
# Combined confluence scorer
# ---------------------------------------------------------------------------


class MultiTimeframeFilter:
    """Multi-timeframe confluence filter with configurable EMA periods.

    Evaluates three layers of analysis (HTF bias, setup quality, entry timing)
    and produces a confluence score of 0–3. Only 3/3 confluence qualifies
    as a tradeable signal.

    Usage:
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(df_htf, df_setup, df_entry, "Gold")
        if result["score"] == 3:
            take_trade(result["direction"])
    """

    def __init__(
        self,
        htf_ema_fast: int = DEFAULT_EMA_FAST,
        htf_ema_mid: int = DEFAULT_EMA_MID,
        htf_ema_slow: int = DEFAULT_EMA_SLOW,
        setup_ema_fast: int = DEFAULT_EMA_FAST,
        setup_ema_mid: int = DEFAULT_EMA_MID,
        entry_ema_fast: int = 9,
        entry_ema_mid: int = 21,
        rsi_period: int = DEFAULT_RSI_PERIOD,
    ):
        self.htf_ema_fast = htf_ema_fast
        self.htf_ema_mid = htf_ema_mid
        self.htf_ema_slow = htf_ema_slow
        self.setup_ema_fast = setup_ema_fast
        self.setup_ema_mid = setup_ema_mid
        self.entry_ema_fast = entry_ema_fast
        self.entry_ema_mid = entry_ema_mid
        self.rsi_period = rsi_period

    def evaluate(
        self,
        htf_df: pd.DataFrame,
        setup_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        asset_name: str = "",
    ) -> dict[str, Any]:
        """Run full 3-layer confluence analysis.

        Args:
            htf_df: Higher timeframe OHLCV DataFrame.
            setup_df: Setup timeframe OHLCV DataFrame.
            entry_df: Entry timeframe OHLCV DataFrame.
            asset_name: Asset name for logging.

        Returns:
            Dict with:
              - score: 0–3 (number of aligned layers)
              - direction: "bullish", "bearish", or "neutral"
              - tradeable: bool (True only if score == 3)
              - htf: HTF layer evaluation result
              - setup: Setup layer evaluation result
              - entry: Entry layer evaluation result
              - summary: human-readable summary string
        """
        htf = evaluate_htf_bias(
            htf_df,
            self.htf_ema_fast,
            self.htf_ema_mid,
            self.htf_ema_slow,
        )
        setup = evaluate_setup(
            setup_df,
            self.setup_ema_fast,
            self.setup_ema_mid,
            self.rsi_period,
        )
        entry = evaluate_entry(
            entry_df,
            self.entry_ema_fast,
            self.entry_ema_mid,
            self.rsi_period,
        )

        # Count aligned layers
        directions = [htf["direction"], setup["direction"], entry["direction"]]

        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")

        if bullish_count == 3:
            score = 3
            direction = "bullish"
        elif bearish_count == 3:
            score = 3
            direction = "bearish"
        elif bullish_count >= 2:
            score = bullish_count
            direction = "bullish"
        elif bearish_count >= 2:
            score = bearish_count
            direction = "bearish"
        elif bullish_count == 1 or bearish_count == 1:
            score = 1
            direction = "bullish" if bullish_count > bearish_count else "bearish"
        else:
            score = 0
            direction = "neutral"

        tradeable = score == 3 and entry.get("trigger", False)

        # Quality composite
        overall_quality = (
            htf["confidence"] * 0.4 + setup["quality"] * 0.35 + (1.0 if entry.get("trigger", False) else 0.0) * 0.25
        )

        # Summary
        emoji_map = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
        layers = [
            f"HTF={htf['direction']}{emoji_map.get(htf['direction'], '')}",
            f"Setup={setup['direction']}{emoji_map.get(setup['direction'], '')}",
            f"Entry={entry['direction']}{emoji_map.get(entry['direction'], '')}",
        ]
        asset_str = f"{asset_name} " if asset_name else ""
        summary = (
            f"{asset_str}Confluence {score}/3 {direction.upper()} "
            f"[{', '.join(layers)}] "
            f"{'→ TRADEABLE' if tradeable else '→ no trade'}"
        )

        return {
            "score": score,
            "direction": direction,
            "tradeable": tradeable,
            "quality": round(overall_quality, 3),
            "htf": htf,
            "setup": setup,
            "entry": entry,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def check_confluence(
    htf_df: pd.DataFrame,
    setup_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    asset_name: str = "",
) -> dict[str, Any]:
    """Quick confluence check using default settings.

    Shorthand for creating a MultiTimeframeFilter and calling evaluate().
    """
    mtf = MultiTimeframeFilter()
    return mtf.evaluate(htf_df, setup_df, entry_df, asset_name)


def get_recommended_timeframes(asset_name: str) -> tuple[str, str, str]:
    """Get the recommended HTF/setup/entry timeframe intervals for an asset.

    Returns:
        Tuple of (htf_interval, setup_interval, entry_interval).
    """
    return TIMEFRAME_PRESETS.get(asset_name, ("15m", "5m", "1m"))


def confluence_summary(result: dict[str, Any]) -> str:
    """Extract a clean one-line summary from a confluence result dict."""
    return result.get("summary", "No confluence data")


def confluence_to_dataframe(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert a dict of {asset_name: confluence_result} to a display DataFrame.

    Args:
        results: Dict mapping asset names to confluence evaluation results.

    Returns:
        DataFrame with columns: Asset, Score, Direction, HTF, Setup, Entry,
        Quality, Tradeable.
    """
    rows = []
    emoji_map = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}

    for asset, r in results.items():
        htf_dir = r.get("htf", {}).get("direction", "neutral")
        setup_dir = r.get("setup", {}).get("direction", "neutral")
        entry_dir = r.get("entry", {}).get("direction", "neutral")

        rows.append(
            {
                "Asset": asset,
                "Score": f"{r.get('score', 0)}/3",
                "Direction": f"{r.get('direction', 'neutral').upper()} {emoji_map.get(r.get('direction', 'neutral'), '')}",
                "HTF Bias": f"{htf_dir} {emoji_map.get(htf_dir, '')}",
                "Setup": f"{setup_dir} {emoji_map.get(setup_dir, '')}",
                "Entry": f"{entry_dir} {emoji_map.get(entry_dir, '')}",
                "Quality": f"{r.get('quality', 0):.0%}",
                "Tradeable": "✅" if r.get("tradeable", False) else "❌",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "Asset",
                    "Score",
                    "Direction",
                    "HTF Bias",
                    "Setup",
                    "Entry",
                    "Quality",
                    "Tradeable",
                ]
            )
        )

    return pd.DataFrame(rows)


def get_instrument_ema_presets(asset_name: str) -> dict[str, int]:
    """Get recommended EMA periods per instrument (from notes.md).

    Recommendations:
      - ES/NQ: 9/21/50 on 5-min
      - GC:    9/21/50 on 5-min
      - CL:    8/21/50 on 3-min (faster due to volatility)
      - SI/HG: 10/20/50 default

    Returns:
        Dict with ema_fast, ema_mid, ema_slow.
    """
    presets = {
        "S&P": {"ema_fast": 9, "ema_mid": 21, "ema_slow": 50},
        "Nasdaq": {"ema_fast": 10, "ema_mid": 20, "ema_slow": 50},
        "Gold": {"ema_fast": 9, "ema_mid": 21, "ema_slow": 50},
        "Silver": {"ema_fast": 10, "ema_mid": 20, "ema_slow": 50},
        "Copper": {"ema_fast": 10, "ema_mid": 20, "ema_slow": 50},
        "Crude Oil": {"ema_fast": 8, "ema_mid": 21, "ema_slow": 50},
    }
    return presets.get(asset_name, {"ema_fast": 10, "ema_mid": 20, "ema_slow": 50})
