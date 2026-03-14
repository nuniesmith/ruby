#!/usr/bin/env python
"""
Technical Indicators module for the BTC/USD Trading System.

This module implements various market indicators, candlestick pattern detectors,
and key price level identifiers optimized for cryptocurrency trading.
"""

from typing import Any

import numpy as np
import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


def identify_manipulation_candles(df: pd.DataFrame, enhanced_detection: bool = True) -> pd.DataFrame:
    """
    Identify manipulation candles in price data with enhanced detection
    specifically optimized for crypto markets.

    Args:
        df: DataFrame with OHLC data
        enhanced_detection: Whether to use enhanced detection parameters for crypto

    Returns:
        DataFrame with manipulation candle indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Calculate required metrics if not present
    if "candle_range" not in result.columns:
        result["candle_range"] = result["high"] - result["low"]

    if "candle_body" not in result.columns:
        result["candle_body"] = abs(result["close"] - result["open"])

    if "body_to_range" not in result.columns:
        # Avoid division by zero
        result["body_to_range"] = np.where(
            result["candle_range"] > 0, result["candle_body"] / result["candle_range"], 0
        )

    # Calculate upper and lower wicks
    result["upper_wick"] = result["high"] - result[["open", "close"]].max(axis=1)
    result["lower_wick"] = result[["open", "close"]].min(axis=1) - result["low"]

    # Calculate wick ratios (avoid division by zero)
    result["upper_wick_ratio"] = np.where(result["candle_range"] > 0, result["upper_wick"] / result["candle_range"], 0)

    result["lower_wick_ratio"] = np.where(result["candle_range"] > 0, result["lower_wick"] / result["candle_range"], 0)

    # Determine candle direction
    result["bullish"] = result["close"] > result["open"]
    result["bearish"] = result["close"] < result["open"]

    # Adjust thresholds for crypto if enhanced detection is enabled
    if enhanced_detection:
        # Crypto tends to have more volatility and fake-outs
        body_threshold = 0.2  # Body must be at least 20% of range
        wick_threshold = 0.6  # Wick must be at least 60% of range
        volatility_factor = 1.5  # Compare to 1.5x average volatility
    else:
        # Standard thresholds
        body_threshold = 0.3  # Body must be at least 30% of range
        wick_threshold = 0.5  # Wick must be at least 50% of range
        volatility_factor = 1.2  # Compare to 1.2x average volatility

    # Calculate average range for volatility comparison
    avg_range = result["candle_range"].rolling(window=20).mean()
    result["range_vs_avg"] = result["candle_range"] / avg_range

    # Detect bullish manipulation candle (strong lower wick, decent body)
    result["bullish_mc"] = (
        result["bullish"]  # Must be bullish (close > open)
        & (result["lower_wick_ratio"] > wick_threshold)  # Strong lower wick
        & (result["body_to_range"] > body_threshold)  # Decent body size
        & (result["range_vs_avg"] > volatility_factor)  # Above average volatility
    )

    # Detect bearish manipulation candle (strong upper wick, decent body)
    result["bearish_mc"] = (
        result["bearish"]  # Must be bearish (close < open)
        & (result["upper_wick_ratio"] > wick_threshold)  # Strong upper wick
        & (result["body_to_range"] > body_threshold)  # Decent body size
        & (result["range_vs_avg"] > volatility_factor)  # Above average volatility
    )

    # Add signal strength based on multiple factors (0-1 scale)
    result["bullish_mc_strength"] = np.where(
        result["bullish_mc"],
        (
            result["lower_wick_ratio"] * 0.5  # Stronger lower wick
            + result["body_to_range"] * 0.3  # Better body
            + result["range_vs_avg"] / 5 * 0.2
        ),  # Higher volatility (capped at 1)
        0,
    )

    result["bearish_mc_strength"] = np.where(
        result["bearish_mc"],
        (
            result["upper_wick_ratio"] * 0.5  # Stronger upper wick
            + result["body_to_range"] * 0.3  # Better body
            + result["range_vs_avg"] / 5 * 0.2
        ),  # Higher volatility (capped at 1)
        0,
    )

    # Enhanced detection additions
    if enhanced_detection:
        # Check for volume confirmation if volume data is available
        if "volume" in result.columns:
            # Calculate average volume
            avg_volume = result["volume"].rolling(window=20).mean()
            result["volume_vs_avg"] = result["volume"] / avg_volume

            # Stronger signals have above-average volume
            volume_factor = 0.2  # Weight for volume in strength calculation

            # Adjust strength based on volume (only increase, don't decrease)
            result["bullish_mc_strength"] = np.where(
                (result["bullish_mc"]) & (result["volume_vs_avg"] > 1.5),
                result["bullish_mc_strength"] + volume_factor,
                result["bullish_mc_strength"],
            )

            result["bearish_mc_strength"] = np.where(
                (result["bearish_mc"]) & (result["volume_vs_avg"] > 1.5),
                result["bearish_mc_strength"] + volume_factor,
                result["bearish_mc_strength"],
            )

        # Look for failed breakout patterns (rejection of key levels)
        if "at_key_level" in result.columns:
            # Signals near key levels are stronger
            level_factor = 0.15  # Weight for key level in strength calculation

            result["bullish_mc_strength"] = np.where(
                (result["bullish_mc"]) & (result["at_key_level"]),
                result["bullish_mc_strength"] + level_factor,
                result["bullish_mc_strength"],
            )

            result["bearish_mc_strength"] = np.where(
                (result["bearish_mc"]) & (result["at_key_level"]),
                result["bearish_mc_strength"] + level_factor,
                result["bearish_mc_strength"],
            )

    # Cap strength at 1.0
    result["bullish_mc_strength"] = np.minimum(result["bullish_mc_strength"], 1.0)
    result["bearish_mc_strength"] = np.minimum(result["bearish_mc_strength"], 1.0)

    # Count occurrences
    bullish_count = result["bullish_mc"].sum()
    bearish_count = result["bearish_mc"].sum()

    logger.info("identified_manipulation_candles", bullish_count=bullish_count, bearish_count=bearish_count)

    return result


def identify_advanced_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify advanced candlestick patterns specific to crypto markets.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with additional pattern indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Calculate required metrics if not present
    if "candle_range" not in result.columns:
        result["candle_range"] = result["high"] - result["low"]

    if "candle_body" not in result.columns:
        result["candle_body"] = abs(result["close"] - result["open"])

    # Determine candle direction if not present
    if "bullish" not in result.columns:
        result["bullish"] = result["close"] > result["open"]
        result["bearish"] = result["close"] < result["open"]

    # 1. Crypto-specific "Stop Hunt" pattern (sharp move in one direction then reversal)

    # Look back 3 candles
    for i in range(3, len(result)):
        # Check for bullish stop hunt (sharp drop followed by reversal)
        if (
            result.iloc[i - 3 : i - 1]["bearish"].all()  # 2 bearish candles
            and result.iloc[i - 1 : i + 1]["bullish"].all()  # 2 bullish candles
            and result.iloc[i - 2]["low"] < result.iloc[i - 3 : i - 2]["low"].min()  # Lower low
            and result.iloc[i]["close"] > result.iloc[i - 2]["open"]
        ):  # Recovery above start
            result.iloc[i, result.columns.get_loc("bullish_stop_hunt")] = True
        else:
            result.iloc[i, result.columns.get_loc("bullish_stop_hunt")] = False

        # Check for bearish stop hunt (sharp rise followed by reversal)
        if (
            result.iloc[i - 3 : i - 1]["bullish"].all()  # 2 bullish candles
            and result.iloc[i - 1 : i + 1]["bearish"].all()  # 2 bearish candles
            and result.iloc[i - 2]["high"] > result.iloc[i - 3 : i - 2]["high"].max()  # Higher high
            and result.iloc[i]["close"] < result.iloc[i - 2]["open"]
        ):  # Drop below start
            result.iloc[i, result.columns.get_loc("bearish_stop_hunt")] = True
        else:
            result.iloc[i, result.columns.get_loc("bearish_stop_hunt")] = False

    # 2. "Wyckoff Spring" pattern (test of support with quick rejection)
    result["wyckoff_spring"] = False

    # Find instances where price drops below recent low then quickly recovers
    for i in range(10, len(result)):
        # Calculate recent low (last 10 candles)
        recent_low = result.iloc[i - 10 : i - 1]["low"].min()

        # Check if current candle broke below then recovered
        if (
            result.iloc[i]["low"] < recent_low  # Broke below recent low
            and result.iloc[i]["close"] > recent_low  # Closed back above
            and result.iloc[i]["bullish"]  # Bullish candle
            and result.iloc[i]["lower_wick_ratio"] > 0.5
        ):  # Strong lower wick
            result.iloc[i, result.columns.get_loc("wyckoff_spring")] = True

    # 3. "Wyckoff Upthrust" pattern (test of resistance with quick rejection)
    result["wyckoff_upthrust"] = False

    # Find instances where price rises above recent high then quickly drops
    for i in range(10, len(result)):
        # Calculate recent high (last 10 candles)
        recent_high = result.iloc[i - 10 : i - 1]["high"].max()

        # Check if current candle broke above then rejected
        if (
            result.iloc[i]["high"] > recent_high  # Broke above recent high
            and result.iloc[i]["close"] < recent_high  # Closed back below
            and result.iloc[i]["bearish"]  # Bearish candle
            and result.iloc[i]["upper_wick_ratio"] > 0.5
        ):  # Strong upper wick
            result.iloc[i, result.columns.get_loc("wyckoff_upthrust")] = True

    # 4. "Flash Crash Recovery" pattern (sharp drop with immediate recovery)
    result["flash_crash_recovery"] = False

    for i in range(2, len(result)):
        # Check for sharp drop with immediate recovery
        if (
            result.iloc[i - 1]["bearish"]  # Previous candle bearish
            and result.iloc[i - 1]["candle_body"] / result.iloc[i - 1]["candle_range"] > 0.7  # Strong bearish body
            and result.iloc[i]["bullish"]  # Current candle bullish
            and result.iloc[i]["candle_body"] / result.iloc[i]["candle_range"] > 0.6  # Strong bullish body
            and result.iloc[i]["close"] > result.iloc[i - 2]["close"]
        ):  # Recovery above pre-crash
            result.iloc[i, result.columns.get_loc("flash_crash_recovery")] = True

    # Count occurrences of each pattern
    pattern_counts = {
        "bullish_stop_hunt": result["bullish_stop_hunt"].sum(),
        "bearish_stop_hunt": result["bearish_stop_hunt"].sum(),
        "wyckoff_spring": result["wyckoff_spring"].sum(),
        "wyckoff_upthrust": result["wyckoff_upthrust"].sum(),
        "flash_crash_recovery": result["flash_crash_recovery"].sum(),
    }

    logger.info("identified_advanced_patterns", pattern_counts=pattern_counts)

    return result


def get_valid_signals(
    df: pd.DataFrame, timeframe: str = "scalp", min_strength: float = 0.5, enhanced_filtering: bool = True
) -> pd.DataFrame:
    """
    Filter valid trading signals from manipulation candles.

    Args:
        df: DataFrame with manipulation candle indicators
        timeframe: Trading timeframe ('scalp', 'day', 'swing')
        min_strength: Minimum signal strength threshold (0-1)
        enhanced_filtering: Apply additional crypto-specific filters

    Returns:
        DataFrame with valid trading signals
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Verify we have the necessary columns
    required_cols = ["bullish_mc", "bearish_mc", "bullish_mc_strength", "bearish_mc_strength"]
    missing_cols = [col for col in required_cols if col not in result.columns]

    if missing_cols:
        logger.error("missing_required_columns", missing_columns=missing_cols)
        logger.info("attempting_to_generate_manipulation_candle_indicators")
        result = identify_manipulation_candles(result, enhanced_detection=enhanced_filtering)

    # Initialize signal columns
    result["valid_bullish_signal"] = False
    result["valid_bearish_signal"] = False
    result["signal_type"] = "none"
    result["signal_strength"] = 0.0

    # Apply strength threshold
    result["valid_bullish_signal"] = result["bullish_mc"] & (result["bullish_mc_strength"] >= min_strength)

    result["valid_bearish_signal"] = result["bearish_mc"] & (result["bearish_mc_strength"] >= min_strength)

    # Set signal type
    result.loc[result["valid_bullish_signal"], "signal_type"] = "buy"
    result.loc[result["valid_bearish_signal"], "signal_type"] = "sell"

    # Set signal strength
    result.loc[result["valid_bullish_signal"], "signal_strength"] = result.loc[
        result["valid_bullish_signal"], "bullish_mc_strength"
    ]
    result.loc[result["valid_bearish_signal"], "signal_strength"] = result.loc[
        result["valid_bearish_signal"], "bearish_mc_strength"
    ]

    # Apply additional timeframe-specific filtering
    if timeframe == "scalp":
        # For scalping, we want quick in-and-out signals
        # No additional filtering needed, default settings are suitable
        pass

    elif timeframe == "day":
        # For day trading, we want stronger confirmation
        result.loc[result["signal_strength"] < 0.6, "signal_type"] = "none"
        result.loc[result["signal_strength"] < 0.6, "signal_strength"] = 0

        # Reset valid signal flags
        result["valid_bullish_signal"] = result["signal_type"] == "buy"
        result["valid_bearish_signal"] = result["signal_type"] == "sell"

    elif timeframe == "swing":
        # For swing trading, we want very strong confirmation and trend alignment
        result.loc[result["signal_strength"] < 0.7, "signal_type"] = "none"
        result.loc[result["signal_strength"] < 0.7, "signal_strength"] = 0

        # Reset valid signal flags
        result["valid_bullish_signal"] = result["signal_type"] == "buy"
        result["valid_bearish_signal"] = result["signal_type"] == "sell"

    # Enhanced filtering for crypto
    if enhanced_filtering:
        # Add trend alignment check if we have enough data
        if len(result) >= 50:
            # Simple trend using 50-period close prices
            result["trend_50"] = result["close"] > result["close"].shift(50)

            # Only keep signals aligned with trend for swing trading
            if timeframe == "swing":
                # In swing trading, only take signals aligned with the trend
                result.loc[(result["valid_bullish_signal"]) & (~result["trend_50"]), "signal_type"] = "none"
                result.loc[(result["valid_bearish_signal"]) & (result["trend_50"]), "signal_type"] = "none"

                # Reset valid signal flags
                result["valid_bullish_signal"] = result["signal_type"] == "buy"
                result["valid_bearish_signal"] = result["signal_type"] == "sell"

        # Avoid signals in choppy periods (if ATR data is available)
        if "atr" in result.columns:
            # Calculate ATR volatility ratio
            result["atr_ratio"] = result["atr"] / result["close"] * 100

            # Define choppy market as low ATR ratio
            choppy_threshold = 0.5  # 0.5% ATR ratio
            result["is_choppy"] = result["atr_ratio"] < choppy_threshold

            # Filter out signals in choppy periods
            result.loc[result["is_choppy"], "signal_type"] = "none"
            result.loc[result["is_choppy"], "signal_strength"] = 0

            # Reset valid signal flags
            result["valid_bullish_signal"] = result["signal_type"] == "buy"
            result["valid_bearish_signal"] = result["signal_type"] == "sell"

    # Count final signals
    bullish_count = result["valid_bullish_signal"].sum()
    bearish_count = result["valid_bearish_signal"].sum()

    logger.info(
        "generated_valid_signals",
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        timeframe=timeframe,
    )

    return result


def generate_entry_signals(df: pd.DataFrame, atr_multiplier: float = 1.0, dynamic_targets: bool = True) -> pd.DataFrame:
    """
    Generate entry signals with stop-loss and take-profit levels.

    Args:
        df: DataFrame with valid trading signals
        atr_multiplier: Multiplier for ATR-based stop-loss calculation
        dynamic_targets: Whether to use dynamic take-profit targets

    Returns:
        DataFrame with entry signals and risk parameters
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Verify we have the necessary columns
    required_cols = ["signal_type", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in result.columns]

    if missing_cols:
        logger.error("missing_required_columns", missing_columns=missing_cols)
        return df

    # Calculate ATR if not present (for stop-loss calculation)
    if "atr" not in result.columns:
        # Calculate True Range
        result["tr"] = np.maximum(
            result["high"] - result["low"],
            np.maximum(abs(result["high"] - result["close"].shift(1)), abs(result["low"] - result["close"].shift(1))),
        )

        # 14-period ATR
        result["atr"] = result["tr"].rolling(window=14).mean()

    # Initialize columns
    result["entry_signal"] = False
    result["entry_price"] = np.nan
    result["stop_loss"] = np.nan
    result["risk_pips"] = np.nan

    # Process buy signals
    buy_signals = result["signal_type"] == "buy"

    if buy_signals.any():
        # Entry price is the next candle's open (or current close as proxy)
        result.loc[buy_signals, "entry_price"] = result.loc[buy_signals, "close"]

        # Stop loss is below the signal candle's low
        result.loc[buy_signals, "stop_loss"] = (
            result.loc[buy_signals, "low"] - result.loc[buy_signals, "atr"] * atr_multiplier
        )

        # Risk in pips
        result.loc[buy_signals, "risk_pips"] = (
            result.loc[buy_signals, "entry_price"] - result.loc[buy_signals, "stop_loss"]
        )

        # Set entry signal flag
        result.loc[buy_signals, "entry_signal"] = True

    # Process sell signals
    sell_signals = result["signal_type"] == "sell"

    if sell_signals.any():
        # Entry price is the next candle's open (or current close as proxy)
        result.loc[sell_signals, "entry_price"] = result.loc[sell_signals, "close"]

        # Stop loss is above the signal candle's high
        result.loc[sell_signals, "stop_loss"] = (
            result.loc[sell_signals, "high"] + result.loc[sell_signals, "atr"] * atr_multiplier
        )

        # Risk in pips
        result.loc[sell_signals, "risk_pips"] = (
            result.loc[sell_signals, "stop_loss"] - result.loc[sell_signals, "entry_price"]
        )

        # Set entry signal flag
        result.loc[sell_signals, "entry_signal"] = True

    # Generate take-profit levels at various risk-reward ratios
    for rr in [1.0, 1.5, 2.0, 3.0]:
        tp_col = f"tp_rr_{rr}"
        result[tp_col] = np.nan

        # Buy take-profit
        result.loc[buy_signals, tp_col] = (
            result.loc[buy_signals, "entry_price"] + result.loc[buy_signals, "risk_pips"] * rr
        )

        # Sell take-profit
        result.loc[sell_signals, tp_col] = (
            result.loc[sell_signals, "entry_price"] - result.loc[sell_signals, "risk_pips"] * rr
        )

    # Add dynamic targets if requested
    if dynamic_targets:
        # For crypto, we often need to adjust targets based on volatility
        # Calculate volatility-adjusted targets (scale based on ATR ratio)

        # First, get a typical ATR ratio for this instrument
        if "atr_ratio" not in result.columns:
            result["atr_ratio"] = result["atr"] / result["close"] * 100

        avg_atr_ratio = result["atr_ratio"].median()

        # Apply volatility adjustment to targets for high volatility periods
        vol_adjusted = result["atr_ratio"] > avg_atr_ratio * 1.5

        if vol_adjusted.any():
            # Calculate volatility factor (how much more volatile than average)
            result.loc[vol_adjusted, "vol_factor"] = result.loc[vol_adjusted, "atr_ratio"] / avg_atr_ratio

            # Adjust take-profit targets for high volatility periods
            for rr in [1.5, 2.0, 3.0]:  # Don't adjust 1.0 RR
                tp_col = f"tp_rr_{rr}"

                # For buys, increase target
                buy_adjust = buy_signals & vol_adjusted
                if buy_adjust.any():
                    result.loc[buy_adjust, tp_col] = (
                        result.loc[buy_adjust, "entry_price"]
                        + result.loc[buy_adjust, "risk_pips"] * rr * result.loc[buy_adjust, "vol_factor"]
                    )

                # For sells, decrease target (move further down)
                sell_adjust = sell_signals & vol_adjusted
                if sell_adjust.any():
                    result.loc[sell_adjust, tp_col] = (
                        result.loc[sell_adjust, "entry_price"]
                        - result.loc[sell_adjust, "risk_pips"] * rr * result.loc[sell_adjust, "vol_factor"]
                    )

    # Count final entry signals
    entry_count = result["entry_signal"].sum()

    logger.info("generated_entry_signals", entry_count=entry_count)

    return result


def filter_signals_for_crypto(
    df: pd.DataFrame, min_strength: float = 0.5, respect_key_levels: bool = True
) -> pd.DataFrame:
    """
    Apply crypto-specific filters to trading signals.

    Args:
        df: DataFrame with entry signals
        min_strength: Minimum signal strength threshold
        respect_key_levels: Whether to respect key price levels

    Returns:
        DataFrame with filtered crypto signals
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Verify we have the necessary columns
    if "entry_signal" not in result.columns:
        logger.error("missing_entry_signal_column")
        return df

    # Apply strength threshold if signal_strength column exists
    if "signal_strength" in result.columns:
        weak_signals = result["entry_signal"] & (result["signal_strength"] < min_strength)
        result.loc[weak_signals, "entry_signal"] = False

        filtered_count = weak_signals.sum()
        if filtered_count > 0:
            logger.info("filtered_weak_signals", filtered_count=filtered_count, min_strength=min_strength)

    # Respect key levels if requested and available
    if respect_key_levels:
        key_level_columns = [
            col
            for col in result.columns
            if col.startswith(("at_", "in_", "near_"))
            and any(term in col for term in ["level", "zone", "support", "resistance"])
        ]

        if key_level_columns:
            # Create a combined mask for signals at key levels
            at_key_level = result[key_level_columns].any(axis=1)

            # Filter signals not at key levels
            not_at_level = result["entry_signal"] & ~at_key_level
            result.loc[not_at_level, "entry_signal"] = False

            filtered_count = not_at_level.sum()
            if filtered_count > 0:
                logger.info("filtered_non_key_level_signals", filtered_count=filtered_count)

    # Filter signals near high volatility events if we have volatility data
    if "volatility_pct" in result.columns:
        # Calculate extreme volatility threshold (2x 20-period average)
        result["avg_volatility"] = result["volatility_pct"].rolling(window=20).mean()
        extreme_vol = result["volatility_pct"] > result["avg_volatility"] * 3

        # Look ahead a few periods to check for extreme volatility
        for i in range(1, 4):
            extreme_vol_ahead = extreme_vol.shift(-i).fillna(False)

            # Filter signals that would be caught in extreme volatility
            avoid_signals = result["entry_signal"] & extreme_vol_ahead
            result.loc[avoid_signals, "entry_signal"] = False

            filtered_count = avoid_signals.sum()
            if filtered_count > 0:
                logger.info("filtered_extreme_volatility_signals", filtered_count=filtered_count)

    # Filter signals during weekend low volume periods if we have session data
    weekend_columns = [col for col in result.columns if "weekend" in col.lower()]
    if weekend_columns:
        for col in weekend_columns:
            if "low" in col.lower() or "avoid" in col.lower():
                # Filter signals during weekend low volume periods
                avoid_signals = result["entry_signal"] & result[col]
                result.loc[avoid_signals, "entry_signal"] = False

                filtered_count = avoid_signals.sum()
                if filtered_count > 0:
                    logger.info("filtered_signals_during_period", filtered_count=filtered_count, period=col)

    # Count final signals after filtering
    final_count = result["entry_signal"].sum()

    logger.info("crypto_signal_filtering_complete", final_count=final_count)

    return result


def identify_fair_value_gaps(df: pd.DataFrame, gap_threshold: float = 0.001) -> pd.DataFrame:
    """
    Identify fair value gaps in price data.

    Args:
        df: DataFrame with OHLC data
        gap_threshold: Minimum gap size as percentage of price

    Returns:
        DataFrame with fair value gap indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Initialize indicator columns
    result["bullish_fvg"] = False
    result["bearish_fvg"] = False
    result["fvg_size"] = 0.0
    result["fvg_top"] = np.nan
    result["fvg_bottom"] = np.nan
    result["in_bullish_fvg"] = False
    result["in_bearish_fvg"] = False

    # Need at least 3 candles to identify gaps
    if len(result) < 3:
        logger.warning("insufficient_data_for_fair_value_gaps")
        return result

    # Crypto tends to have mitigated gaps due to 24/7 trading
    # Look for candles that don't overlap instead of traditional gaps

    # Process each candle starting from the third one
    for i in range(2, len(result)):
        # Get current and two previous candles
        curr_candle = result.iloc[i]
        prev_candle = result.iloc[i - 1]
        prev_prev_candle = result.iloc[i - 2]

        # Check for bullish FVG (current high > previous high, previous low < prev-prev low)
        if curr_candle["high"] > prev_candle["high"] and prev_candle["low"] < prev_prev_candle["low"]:
            # Calculate gap size
            gap_top = prev_prev_candle["low"]
            gap_bottom = prev_candle["high"]

            # Only consider real gaps
            if gap_top > gap_bottom:
                # Calculate gap size as percentage
                gap_size = (gap_top - gap_bottom) / curr_candle["close"]

                # Apply threshold
                if gap_size >= gap_threshold:
                    result.iloc[i, result.columns.get_loc("bullish_fvg")] = True
                    result.iloc[i, result.columns.get_loc("fvg_size")] = gap_size
                    result.iloc[i, result.columns.get_loc("fvg_top")] = gap_top
                    result.iloc[i, result.columns.get_loc("fvg_bottom")] = gap_bottom

        # Check for bearish FVG (current low < previous low, previous high > prev-prev high)
        if curr_candle["low"] < prev_candle["low"] and prev_candle["high"] > prev_prev_candle["high"]:
            # Calculate gap size
            gap_top = prev_candle["low"]
            gap_bottom = prev_prev_candle["high"]

            # Only consider real gaps
            if gap_top > gap_bottom:
                # Calculate gap size as percentage
                gap_size = (gap_top - gap_bottom) / curr_candle["close"]

                # Apply threshold
                if gap_size >= gap_threshold:
                    result.iloc[i, result.columns.get_loc("bearish_fvg")] = True
                    result.iloc[i, result.columns.get_loc("fvg_size")] = gap_size
                    result.iloc[i, result.columns.get_loc("fvg_top")] = gap_top
                    result.iloc[i, result.columns.get_loc("fvg_bottom")] = gap_bottom

    # Track when price is inside an FVG
    active_bullish_fvgs = []
    active_bearish_fvgs = []

    # First pass: identify all gaps
    for i in range(len(result)):
        if result.iloc[i]["bullish_fvg"]:
            top = result.iloc[i]["fvg_top"]
            bottom = result.iloc[i]["fvg_bottom"]
            active_bullish_fvgs.append((top, bottom, i))

        if result.iloc[i]["bearish_fvg"]:
            top = result.iloc[i]["fvg_top"]
            bottom = result.iloc[i]["fvg_bottom"]
            active_bearish_fvgs.append((top, bottom, i))

    # Second pass: determine when price is in a gap
    for i in range(len(result)):
        curr_price = result.iloc[i]["close"]

        # Check if price is in any active bullish FVG
        for top, bottom, start_idx in active_bullish_fvgs[:]:
            if bottom <= curr_price <= top:
                result.iloc[i, result.columns.get_loc("in_bullish_fvg")] = True

            # Remove filled gaps
            if i > start_idx and result.iloc[i]["low"] <= bottom and result.iloc[i]["high"] >= top:
                active_bullish_fvgs.remove((top, bottom, start_idx))

        # Check if price is in any active bearish FVG
        for top, bottom, start_idx in active_bearish_fvgs[:]:
            if bottom <= curr_price <= top:
                result.iloc[i, result.columns.get_loc("in_bearish_fvg")] = True

            # Remove filled gaps
            if i > start_idx and result.iloc[i]["low"] <= bottom and result.iloc[i]["high"] >= top:
                active_bearish_fvgs.remove((top, bottom, start_idx))

    # Count FVGs
    bullish_count = result["bullish_fvg"].sum()
    bearish_count = result["bearish_fvg"].sum()

    logger.info("identified_fair_value_gaps", bullish_count=bullish_count, bearish_count=bearish_count)

    return result


def identify_supply_demand_zones(df: pd.DataFrame, zone_strength_threshold: int = 3) -> pd.DataFrame:
    """
    Identify supply and demand zones based on price rejections.

    Args:
        df: DataFrame with OHLC data
        zone_strength_threshold: Minimum number of tests for a strong zone

    Returns:
        DataFrame with supply/demand zone indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Initialize columns
    result["at_supply_zone"] = False
    result["at_demand_zone"] = False
    result["supply_zone_strength"] = 0
    result["demand_zone_strength"] = 0

    # Need sufficient data to identify zones
    if len(result) < 50:
        logger.warning("insufficient_data_for_supply_demand_zones")
        return result

    # Identify swing highs and lows
    result["swing_high"] = False
    result["swing_low"] = False

    # Use a 5-period window for swings
    window = 5

    for i in range(window, len(result) - window):
        # Center candle and surrounding windows
        center = result.iloc[i]
        left_window = result.iloc[i - window : i]
        right_window = result.iloc[i + 1 : i + window + 1]

        # Check for swing high
        if center["high"] > left_window["high"].max() and center["high"] > right_window["high"].max():
            result.iloc[i, result.columns.get_loc("swing_high")] = True

        # Check for swing low
        if center["low"] < left_window["low"].min() and center["low"] < right_window["low"].min():
            result.iloc[i, result.columns.get_loc("swing_low")] = True

    # Identify zones around swing points
    swing_highs = result[result["swing_high"]].copy()
    swing_lows = result[result["swing_low"]].copy()

    # Group nearby swing highs into supply zones
    supply_zones = []

    if not swing_highs.empty:
        # Sort by price to group nearby levels
        sorted_highs = swing_highs.sort_values(by="high")  # type: ignore[call-overload]

        current_zone = {
            "top": sorted_highs.iloc[0]["high"],
            "bottom": sorted_highs.iloc[0]["high"] * 0.995,  # 0.5% zone size
            "points": [sorted_highs.index[0]],
            "strength": 1,
        }

        for i in range(1, len(sorted_highs)):
            high_idx = sorted_highs.index[i]
            high_price = sorted_highs.loc[high_idx, "high"]

            # If this high is within 1% of the current zone, add to zone
            if high_price <= current_zone["top"] * 1.01:
                current_zone["points"].append(high_idx)
                current_zone["strength"] += 1
                # Expand zone if necessary
                current_zone["top"] = max(current_zone["top"], high_price)
                current_zone["bottom"] = min(current_zone["bottom"], high_price * 0.995)
            else:
                # Add current zone to list if strong enough
                if current_zone["strength"] >= zone_strength_threshold:
                    supply_zones.append(current_zone)

                # Start new zone
                current_zone = {"top": high_price, "bottom": high_price * 0.995, "points": [high_idx], "strength": 1}

        # Add final zone if strong enough
        if current_zone["strength"] >= zone_strength_threshold:
            supply_zones.append(current_zone)

    # Group nearby swing lows into demand zones
    demand_zones = []

    if not swing_lows.empty:
        # Sort by price to group nearby levels
        sorted_lows = swing_lows.sort_values(by="low", ascending=False)  # type: ignore[call-overload]

        current_zone = {
            "top": sorted_lows.iloc[0]["low"] * 1.005,  # 0.5% zone size
            "bottom": sorted_lows.iloc[0]["low"],
            "points": [sorted_lows.index[0]],
            "strength": 1,
        }

        for i in range(1, len(sorted_lows)):
            low_idx = sorted_lows.index[i]
            low_price = sorted_lows.loc[low_idx, "low"]

            # If this low is within 1% of the current zone, add to zone
            if low_price >= current_zone["bottom"] * 0.99:
                current_zone["points"].append(low_idx)
                current_zone["strength"] += 1
                # Expand zone if necessary
                current_zone["top"] = max(current_zone["top"], low_price * 1.005)
                current_zone["bottom"] = min(current_zone["bottom"], low_price)
            else:
                # Add current zone to list if strong enough
                if current_zone["strength"] >= zone_strength_threshold:
                    demand_zones.append(current_zone)

                # Start new zone
                current_zone = {"top": low_price * 1.005, "bottom": low_price, "points": [low_idx], "strength": 1}

        # Add final zone if strong enough
        if current_zone["strength"] >= zone_strength_threshold:
            demand_zones.append(current_zone)

    # Mark candles that are at supply or demand zones
    for i in range(len(result)):
        candle = result.iloc[i]

        # Check if the candle interacts with any supply zone
        for zone in supply_zones:
            if zone["bottom"] <= candle["high"] <= zone["top"] * 1.005:
                result.iloc[i, result.columns.get_loc("at_supply_zone")] = True
                result.iloc[i, result.columns.get_loc("supply_zone_strength")] = zone["strength"]
                break

        # Check if the candle interacts with any demand zone
        for zone in demand_zones:
            if zone["bottom"] * 0.995 <= candle["low"] <= zone["top"]:
                result.iloc[i, result.columns.get_loc("at_demand_zone")] = True
                result.iloc[i, result.columns.get_loc("demand_zone_strength")] = zone["strength"]
                break

    # Count zones
    supply_count = len(supply_zones)
    demand_count = len(demand_zones)

    logger.info("identified_supply_demand_zones", supply_count=supply_count, demand_count=demand_count)

    return result


def identify_key_levels(df: pd.DataFrame, round_digits: int = 0) -> pd.DataFrame:
    """
    Identify key price levels and psychological levels.

    Args:
        df: DataFrame with OHLC data
        round_digits: Number of digits to round to for identifying psychological levels

    Returns:
        DataFrame with key level indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Initialize columns
    result["at_key_level"] = False
    result["level_type"] = ""

    # Need price data
    if "close" not in result.columns:
        logger.error("missing_close_column")
        return result

    # 1. Identify psychological levels based on round numbers

    # Get min and max prices
    min_price = result["close"].min()
    max_price = result["close"].max()

    # For crypto like BTC, we need to handle different price magnitudes
    # Determine the appropriate psychological levels based on price range
    round_levels: list[dict[str, Any]] = []
    if max_price > 10000:
        # BTC-range prices
        round_levels = []

        # Major levels (multiples of $5000)
        for level in range(int(min_price // 5000) * 5000, int(max_price // 5000 + 1) * 5000, 5000):
            round_levels.append({"price": level, "type": "major"})

        # Intermediate levels (multiples of $1000)
        for level in range(int(min_price // 1000) * 1000, int(max_price // 1000 + 1) * 1000, 1000):
            if level % 5000 != 0:  # Skip major levels already added
                round_levels.append({"price": level, "type": "intermediate"})

        # Minor levels (multiples of $500)
        for level in range(int(min_price // 500) * 500, int(max_price // 500 + 1) * 500, 500):
            if level % 1000 != 0:  # Skip levels already added
                round_levels.append({"price": level, "type": "minor"})

    elif max_price > 1000:
        # Lower priced assets (e.g., ETH)
        round_levels = []

        # Major levels (multiples of $500)
        for level in range(int(min_price // 500) * 500, int(max_price // 500 + 1) * 500, 500):
            round_levels.append({"price": level, "type": "major"})

        # Intermediate levels (multiples of $100)
        for level in range(int(min_price // 100) * 100, int(max_price // 100 + 1) * 100, 100):
            if level % 500 != 0:  # Skip major levels already added
                round_levels.append({"price": level, "type": "intermediate"})

        # Minor levels (multiples of $50)
        for level in range(int(min_price // 50) * 50, int(max_price // 50 + 1) * 50, 50):
            if level % 100 != 0:  # Skip levels already added
                round_levels.append({"price": level, "type": "minor"})
    else:
        # Very low priced assets
        round_levels = []

        # Get the appropriate rounding factor based on price magnitude
        magnitude = 10 ** (round_digits)

        # Major levels (multiples of magnitude)
        for level in range(
            int(min_price // magnitude) * magnitude, int(max_price // magnitude + 1) * magnitude, magnitude
        ):
            round_levels.append({"price": level, "type": "major"})

        # Intermediate levels (multiples of magnitude/2)
        half_magnitude = magnitude / 2
        for level in range(
            int(min_price // half_magnitude) * half_magnitude,
            int(max_price // half_magnitude + 1) * half_magnitude,
            half_magnitude,
        ):
            if level % magnitude != 0:  # Skip major levels
                round_levels.append({"price": level, "type": "intermediate"})

        # Minor levels (multiples of magnitude/10)
        tenth_magnitude = magnitude / 10
        for level in range(
            int(min_price // tenth_magnitude) * tenth_magnitude,
            int(max_price // tenth_magnitude + 1) * tenth_magnitude,
            tenth_magnitude,
        ):
            if level % half_magnitude != 0:  # Skip major and intermediate levels
                round_levels.append({"price": level, "type": "minor"})

    # 2. Check each candle for interaction with key levels
    level_threshold = 0.001  # 0.1% threshold for being "at" a level

    for i in range(len(result)):
        candle = result.iloc[i]

        # Check each round level
        for lvl in round_levels:
            level_price = float(lvl["price"])

            # Calculate threshold based on price magnitude
            threshold = level_price * level_threshold

            # Check if the candle interacts with this level
            if abs(candle["high"] - level_price) <= threshold or abs(candle["low"] - level_price) <= threshold:
                result.iloc[i, result.columns.get_loc("at_key_level")] = True
                result.iloc[i, result.columns.get_loc("level_type")] = lvl["type"]
                break

    # Count key levels
    key_level_count = result["at_key_level"].sum()

    logger.info("identified_key_levels", key_level_count=key_level_count)

    return result


def identify_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify key session high/low levels.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with session level indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Initialize columns
    result["at_session_high"] = False
    result["at_session_low"] = False
    result["near_daily_open"] = False

    # Check if we have datetime index
    if not isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index = pd.to_datetime(result.index)
        except Exception:
            logger.error("dataframe_missing_datetime_index")
            return result

    # Extract date for grouping
    result["date"] = pd.DatetimeIndex(result.index).date  # type: ignore[attr-defined]

    # Group by date to get daily high/low
    daily_stats = result.groupby("date").agg({"high": "max", "low": "min", "open": "first"})

    # For each candle, check if it's near the session high/low
    threshold = 0.001  # 0.1% threshold for being "at" a level

    for i in range(len(result)):
        candle = result.iloc[i]
        date = candle["date"]

        if date in daily_stats.index:
            daily_high = daily_stats.loc[date, "high"]
            daily_low = daily_stats.loc[date, "low"]
            daily_open = daily_stats.loc[date, "open"]

            # Calculate thresholds
            high_threshold = daily_high * threshold
            low_threshold = daily_low * threshold
            open_threshold = daily_open * threshold

            # Check if the candle is at the session high
            if abs(candle["high"] - daily_high) <= high_threshold:
                result.iloc[i, result.columns.get_loc("at_session_high")] = True

            # Check if the candle is at the session low
            if abs(candle["low"] - daily_low) <= low_threshold:
                result.iloc[i, result.columns.get_loc("at_session_low")] = True

            # Check if the candle is near the daily open
            if abs(candle["high"] - daily_open) <= open_threshold or abs(candle["low"] - daily_open) <= open_threshold:
                result.iloc[i, result.columns.get_loc("near_daily_open")] = True

    # Add combined indicator for any session level
    result["at_session_level"] = result["at_session_high"] | result["at_session_low"] | result["near_daily_open"]

    # Remove temporary date column
    result = result.drop("date", axis=1)

    # Count session levels
    session_high_count = result["at_session_high"].sum()
    session_low_count = result["at_session_low"].sum()
    daily_open_count = result["near_daily_open"].sum()

    logger.info(
        "identified_session_levels",
        session_high_count=session_high_count,
        session_low_count=session_low_count,
        daily_open_count=daily_open_count,
    )

    return result


def identify_bitcoin_specific_levels(df: pd.DataFrame, fibonacci_base: float | None = None) -> pd.DataFrame:
    """
    Identify Bitcoin-specific key levels including historical supports/resistances,
    and market cycle levels.

    Args:
        df: DataFrame with OHLC data
        fibonacci_base: Base price for Fibonacci calculations (if None, auto-detect)

    Returns:
        DataFrame with Bitcoin-specific level indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Initialize indicator columns
    result["at_btc_key_level"] = False
    result["in_btc_accumulation_zone"] = False
    result["btc_level_type"] = ""

    # Historical BTC key levels (major supports/resistances from previous market cycles)
    btc_historical_levels = [
        # These values should be updated based on current market conditions
        {"price": 65000, "type": "ATH_2021"},
        {"price": 58300, "type": "resistance_2021"},
        {"price": 53000, "type": "resistance_2021"},
        {"price": 48000, "type": "resistance_2021"},
        {"price": 45000, "type": "resistance_2021"},
        {"price": 40000, "type": "support_resistance_2021"},
        {"price": 37000, "type": "support_2021"},
        {"price": 35000, "type": "support_2021"},
        {"price": 30000, "type": "support_2021"},
        {"price": 29000, "type": "support_2021"},
        {"price": 25000, "type": "support_2022"},
        {"price": 20000, "type": "ATH_2017_support_2022"},
        {"price": 18000, "type": "support_2022"},
        {"price": 16000, "type": "support_2022"},
        {"price": 15000, "type": "support_2019"},
        {"price": 13000, "type": "resistance_2019"},
        {"price": 10000, "type": "psychological_level"},
        {"price": 6000, "type": "support_2018"},
        {"price": 3000, "type": "support_2018"},
    ]

    # BTC accumulation zones (historical value areas)
    btc_accumulation_zones = [
        {"top": 22000, "bottom": 18000, "type": "accumulation_2022"},
        {"top": 16000, "bottom": 14000, "type": "accumulation_2022"},
        {"top": 11000, "bottom": 9000, "type": "accumulation_2019"},
        {"top": 6500, "bottom": 3500, "type": "accumulation_2018"},
    ]

    # Add Fibonacci levels if requested
    if fibonacci_base is None:
        # Auto-detect appropriate base price
        # Use the all-time high if in the dataset, or the max price
        max_price = result["high"].max()
        fibonacci_base = float(max_price)  # type: ignore[arg-type]

    # Calculate Fibonacci levels
    fib_levels = [
        {"ratio": 0.236, "type": "fib_0.236"},
        {"ratio": 0.382, "type": "fib_0.382"},
        {"ratio": 0.5, "type": "fib_0.5"},
        {"ratio": 0.618, "type": "fib_0.618"},
        {"ratio": 0.786, "type": "fib_0.786"},
        {"ratio": 1.0, "type": "fib_1.0"},
        {"ratio": 1.618, "type": "fib_1.618"},
        {"ratio": 2.618, "type": "fib_2.618"},
    ]

    # Calculate Fibonacci price levels
    for level in fib_levels:
        level["price"] = fibonacci_base * float(level["ratio"])  # type: ignore[arg-type]

    # Combine all levels
    all_levels = btc_historical_levels + fib_levels

    # Check each candle for intersection with levels
    threshold = 0.005  # 0.5% threshold for BTC levels (wider due to volatility)

    for i in range(len(result)):
        candle = result.iloc[i]

        # Check historical and Fibonacci levels
        for level in all_levels:
            level_price = float(level["price"])  # type: ignore[arg-type]
            level_threshold = level_price * threshold

            # Check if the candle interacts with this level
            if candle["low"] - level_threshold <= level_price <= candle["high"] + level_threshold:
                result.iloc[i, result.columns.get_loc("at_btc_key_level")] = True
                result.iloc[i, result.columns.get_loc("btc_level_type")] = level["type"]
                break

        # Check accumulation zones
        for zone in btc_accumulation_zones:
            # Check if the candle is within the accumulation zone
            if zone["bottom"] <= candle["low"] <= zone["top"] or zone["bottom"] <= candle["high"] <= zone["top"]:
                result.iloc[i, result.columns.get_loc("in_btc_accumulation_zone")] = True

                # If not already at a key level, set the level type to the zone type
                if not result.iloc[i]["at_btc_key_level"]:
                    result.iloc[i, result.columns.get_loc("btc_level_type")] = zone["type"]

                break

    # Count candles at BTC-specific levels
    btc_level_count = result["at_btc_key_level"].sum()
    accumulation_count = result["in_btc_accumulation_zone"].sum()

    logger.info(
        "identified_bitcoin_specific_levels",
        btc_level_count=btc_level_count,
        accumulation_count=accumulation_count,
    )

    return result


def is_price_in_area_of_interest(price: float, df: pd.DataFrame, threshold: float = 0.001) -> dict[str, Any]:
    """
    Check if a given price is in any area of interest.

    Args:
        price: Price to check
        df: DataFrame with identified areas of interest
        threshold: Percentage threshold for proximity

    Returns:
        Dictionary with area of interest information
    """
    # Initialize result
    result = {"is_in_area": False, "area_types": [], "distance": float("inf"), "closest_level": None}

    # Check for required columns
    required_cols = [col for col in df.columns if col.startswith(("at_", "in_")) and col != "at_key_level"]

    if not required_cols:
        logger.warning("no_area_of_interest_columns_found")
        return result

    # Find the closest candle to the given price
    df["price_distance"] = abs(df["close"] - price) / price
    closest_idx = df["price_distance"].idxmin()
    closest_candle = df.loc[closest_idx]

    # Check if the closest candle is in any area of interest
    in_area = False
    area_types = []

    for col in required_cols:
        if closest_candle[col]:
            in_area = True

            # Extract area type from column name
            area_type = col.replace("at_", "").replace("in_", "")
            area_types.append(area_type)

    # Check if the price is close enough to the closest candle
    is_close = closest_candle["price_distance"] <= threshold

    if in_area and is_close:
        result["is_in_area"] = True
        result["area_types"] = area_types
        result["distance"] = closest_candle["price_distance"]
        result["closest_level"] = closest_candle["close"]

    return result
