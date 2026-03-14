"""
Manipulation Model - Candle Patterns Module

This module provides functions to identify manipulation candles (MCs) and trap patterns
in financial market data, supporting both BTC and traditional markets.
"""

import numpy as np
import pandas as pd

from lib.core.logging_config import get_logger

# (utils.datetime_utils / utils.config_utils are from the original project; not needed here)

logger = get_logger(__name__)


def identify_manipulation_candles(
    df: pd.DataFrame, trend_period: int = 14, enhanced_detection: bool = False
) -> pd.DataFrame:
    """
    Identifies Manipulation Candles (MCs) in price data with enhanced vectorized operations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data with columns: 'open', 'high', 'low', 'close'
    trend_period : int, optional
        Period for calculating the trend direction (default: 14)
    enhanced_detection: bool, optional
        Use additional pattern recognition for crypto markets (default: False)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - 'bullish_mc': Boolean flag for bullish manipulation candles
        - 'bearish_mc': Boolean flag for bearish manipulation candles
        - 'trend': Current trend direction ('up', 'down', or 'sideways')
        - 'is_beartrap': Boolean flag for beartraps
        - 'is_bulltrap': Boolean flag for bulltraps
        - Additional pattern columns if enhanced_detection is True
    """
    try:
        # Check for required columns
        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error("missing_required_columns", missing_columns=missing_columns)
            return df

        # Check if dataframe is too small for trend calculation
        if len(df) < trend_period + 1:
            logger.warning(
                "dataframe_too_small_for_trend_calculation",
                min_required=trend_period + 1,
                row_count=len(df),
            )
            # Return the original dataframe with additional columns initialized
            result = df.copy()
            result["ema"] = np.nan
            result["trend"] = "sideways"
            result["bullish_mc"] = False
            result["bearish_mc"] = False
            result["is_beartrap"] = False
            result["is_bulltrap"] = False
            return result

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Calculate trend using EMA
        result["ema"] = result["close"].ewm(span=trend_period, adjust=False).mean()

        # Determine trend based on EMA direction - fully vectorized
        ema_diff = result["ema"].diff()
        result["trend"] = "sideways"  # Default value
        result.loc[ema_diff > 0, "trend"] = "up"
        result.loc[ema_diff < 0, "trend"] = "down"

        # Initialize MC columns
        result["bullish_mc"] = False
        result["bearish_mc"] = False

        # Identify Bullish Manipulation Candles - vectorized conditions
        # Price dips below the last candle's low and then closes above the high
        bullish_condition = (result["low"] < result["low"].shift(1)) & (result["close"] > result["high"].shift(1))
        result.loc[bullish_condition, "bullish_mc"] = True

        # Identify Bearish Manipulation Candles - vectorized conditions
        # Price spikes above the last candle's high and then closes below the low
        bearish_condition = (result["high"] > result["high"].shift(1)) & (result["close"] < result["low"].shift(1))
        result.loc[bearish_condition, "bearish_mc"] = True

        # Identify traps (counter-trend MCs) - vectorized operations
        result["is_beartrap"] = (result["trend"] == "up") & result["bearish_mc"]
        result["is_bulltrap"] = (result["trend"] == "down") & result["bullish_mc"]

        # Enhanced pattern detection for crypto markets if requested
        if enhanced_detection:
            # Add wick strength analysis
            result["upper_wick"] = result["high"] - np.maximum(result["open"], result["close"])
            result["lower_wick"] = np.minimum(result["open"], result["close"]) - result["low"]
            result["body_size"] = abs(result["close"] - result["open"])

            # Calculate wick-to-body ratios (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                result["upper_wick_ratio"] = np.where(
                    result["body_size"] > 0, result["upper_wick"] / result["body_size"], 0
                )
                result["lower_wick_ratio"] = np.where(
                    result["body_size"] > 0, result["lower_wick"] / result["body_size"], 0
                )

            # Identify pinbar patterns (long wick rejections)
            result["bullish_pinbar"] = (
                (result["lower_wick_ratio"] > 2.0)  # Lower wick at least 2x the body
                & (result["lower_wick_ratio"] > result["upper_wick_ratio"] * 2)  # Lower wick much larger than upper
                & (result["close"] > result["open"])  # Bullish close
            )

            result["bearish_pinbar"] = (
                (result["upper_wick_ratio"] > 2.0)  # Upper wick at least 2x the body
                & (result["upper_wick_ratio"] > result["lower_wick_ratio"] * 2)  # Upper wick much larger than lower
                & (result["close"] < result["open"])  # Bearish close
            )

            # Identify high volatility candles (useful for crypto markets)
            # Calculate candle height as percentage of price
            result["candle_height_pct"] = (result["high"] - result["low"]) / result["low"]

            # Mark high volatility candles (above 1.5x the median volatility)
            median_height = result["candle_height_pct"].median()
            result["high_volatility"] = result["candle_height_pct"] > (median_height * 1.5)

            # Mark engulfing patterns - these often lead to manipulation candles
            result["bullish_engulfing"] = (
                (result["open"] < result["close"])  # Current candle is bullish
                & (result["open"].shift(1) > result["close"].shift(1))  # Previous candle is bearish
                & (result["open"] <= result["close"].shift(1))  # Current open below previous close
                & (result["close"] >= result["open"].shift(1))  # Current close above previous open
            )

            result["bearish_engulfing"] = (
                (result["open"] > result["close"])  # Current candle is bearish
                & (result["open"].shift(1) < result["close"].shift(1))  # Previous candle is bullish
                & (result["open"] >= result["close"].shift(1))  # Current open above previous close
                & (result["close"] <= result["open"].shift(1))  # Current close below previous open
            )

        return result

    except Exception as e:
        logger.error("identify_manipulation_candles_failed", error=str(e))
        # If an error occurs, return the original DataFrame
        return df


def get_valid_signals(
    df: pd.DataFrame,
    use_countertrend: bool = False,
    timeframe: str = "H4",
    min_strength: float = 0.0,
    enhanced_filtering: bool = False,
) -> pd.DataFrame:
    """
    Filters manipulation candles based on the strategy rules with improved filtering.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with identified manipulation candles
    use_countertrend : bool, optional
        Whether to consider counter-trend signals (traps) (default: False)
    timeframe : str, optional
        Current timeframe being analyzed (default: 'H4')
    min_strength : float, optional
        Minimum strength ratio for valid signals (0.0-1.0) (default: 0.0)
    enhanced_filtering : bool, optional
        Apply additional filters for stronger signals (default: False)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with valid trading signals based on strategy rules
    """
    try:
        # Check if required columns are present
        required_columns = ["bullish_mc", "bearish_mc", "trend", "is_beartrap", "is_bulltrap"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(
                "missing_required_columns",
                missing_columns=missing_columns,
                hint="Run identify_manipulation_candles first.",
            )
            # Add the missing columns with appropriate default values
            result = df.copy()
            for col in missing_columns:
                if col in ["bullish_mc", "bearish_mc", "is_beartrap", "is_bulltrap"]:
                    result[col] = False
                elif col == "trend":
                    result[col] = "sideways"
            result["valid_signal"] = False
            return result

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Initialize signals column
        result["valid_signal"] = False
        result["signal_strength"] = 0.0

        # Pro-trend signals - vectorized conditions
        bullish_signals = result["bullish_mc"] & (result["trend"] == "up") & ~result["is_bulltrap"]
        bearish_signals = result["bearish_mc"] & (result["trend"] == "down") & ~result["is_beartrap"]

        # Set valid signals
        result.loc[bullish_signals | bearish_signals, "valid_signal"] = True

        # Add counter-trend signals if requested
        if use_countertrend:
            trap_signals = result["is_beartrap"] | result["is_bulltrap"]
            result.loc[trap_signals, "valid_signal"] = True

        # Apply timeframe-specific rules
        if timeframe != "H4" and (timeframe.startswith("D") or timeframe.startswith("W")):
            if timeframe == "D1" or timeframe == "Daily":
                # For daily timeframe, only keep very strong signals
                strong_signals = bullish_signals | bearish_signals
                if "candle_height_pct" in result.columns:
                    # Use candle height as a strength factor
                    avg_height = result["candle_height_pct"].mean()
                    strong_signals = strong_signals & (result["candle_height_pct"] > avg_height * 1.5)

                result["valid_signal"] = False
                result.loc[strong_signals, "valid_signal"] = True
            elif timeframe.startswith("W"):
                # Weekly timeframe - only the strongest signals
                logger.info("processing_strong_signals_only", timeframe=timeframe)
                result["valid_signal"] = False
                # If enhanced features are available, use them
                if "candle_height_pct" in result.columns and enhanced_filtering:
                    very_strong = result["candle_height_pct"] > result["candle_height_pct"].quantile(0.85)
                    if "bullish_engulfing" in result.columns:
                        very_strong = very_strong | result["bullish_engulfing"] | result["bearish_engulfing"]
                    result.loc[very_strong & (bullish_signals | bearish_signals), "valid_signal"] = True
            else:
                # For other long timeframes, be very selective
                result["valid_signal"] = False
                logger.info("no_valid_signals_for_timeframe", timeframe=timeframe, mode="standard")

        # Calculate signal strength if enhanced filtering is enabled
        if enhanced_filtering:
            # Base strength starts at 0.5 for valid signals
            result.loc[result["valid_signal"], "signal_strength"] = 0.5

            # Enhance strength based on available indicators
            if "candle_height_pct" in result.columns:
                # Normalize candle height to a 0-0.3 scale
                max_height = result["candle_height_pct"].quantile(0.95)  # 95th percentile to avoid outliers
                if max_height > 0:
                    norm_height = result["candle_height_pct"] / max_height * 0.3
                    # Add to strength (max contribution: 0.3)
                    result["signal_strength"] += norm_height.clip(0, 0.3)

            # Add strength for pinbars if available
            if "bullish_pinbar" in result.columns:
                # Add 0.2 for pinbar patterns
                result.loc[result["bullish_pinbar"] | result["bearish_pinbar"], "signal_strength"] += 0.2

            # Add strength for engulfing patterns if available
            if "bullish_engulfing" in result.columns:
                # Add 0.3 for engulfing patterns
                result.loc[result["bullish_engulfing"] | result["bearish_engulfing"], "signal_strength"] += 0.3

            # Filter signals by minimum strength if specified
            if min_strength > 0:
                weak_signals = (result["signal_strength"] < min_strength) & result["valid_signal"]
                result.loc[weak_signals, "valid_signal"] = False
                logger.info("removed_weak_signals", count=int(weak_signals.sum()), min_strength=min_strength)

        return result

    except Exception as e:
        logger.error("get_valid_signals_failed", error=str(e))
        # If an error occurs, return the original DataFrame with a valid_signal column
        result = df.copy()
        result["valid_signal"] = False
        return result


def generate_entry_signals(
    df: pd.DataFrame,
    risk_reward_ratio: float = 2.0,
    atr_period: int = 14,
    atr_multiplier: float = 0.5,
    dynamic_targets: bool = True,
) -> pd.DataFrame:
    """
    Generates entry signals with stop-loss and take-profit levels based on manipulation candles.
    Enhanced with dynamic target positioning and improved ATR handling.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with identified manipulation candles and valid signals
    risk_reward_ratio : float, optional
        Risk-to-reward ratio for calculating take-profit levels (default: 2.0)
    atr_period : int, optional
        Period for calculating Average True Range (default: 14)
    atr_multiplier : float, optional
        Multiplier for ATR to determine stop loss distance (default: 0.5)
    dynamic_targets : bool, optional
        Use dynamic take-profit targeting based on volatility (default: True)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with entry signals and calculated parameters
    """
    try:
        # Check if required columns are present
        required_columns = ["valid_signal", "bullish_mc", "bearish_mc", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error("missing_required_columns", missing_columns=missing_columns)
            return df

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Calculate ATR for dynamic stop-loss sizing - vectorized operation
        high_low = result["high"] - result["low"]
        high_close = abs(result["high"] - result["close"].shift(1))
        low_close = abs(result["low"] - result["close"].shift(1))

        # True Range is the greatest of these three values
        tr_values = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["tr"] = tr_values
        _atr_rolling: pd.Series = result["tr"].rolling(window=atr_period).mean()  # type: ignore[assignment]
        result["atr"] = _atr_rolling.where(_atr_rolling.notna(), result["tr"])

        # Initialize entry columns
        result["entry_signal"] = False
        result["signal_type"] = ""
        result["entry_price"] = np.nan
        result["stop_loss"] = np.nan

        # Generate take profit columns for different RR levels
        for rr in [1.0, 1.5, 2.0, 3.0]:
            result[f"tp_rr_{rr}"] = np.nan

        # Find valid bullish signals
        bullish_entries = result["valid_signal"] & result["bullish_mc"]

        # Calculate entry parameters for bullish signals
        if bullish_entries.any():
            # Set basic signal info
            result.loc[bullish_entries, "entry_signal"] = True
            result.loc[bullish_entries, "signal_type"] = "buy"

            # Entry price: the high of the manipulation candle
            result.loc[bullish_entries, "entry_price"] = result.loc[bullish_entries, "high"]

            # Stop loss calculation (with safety against invalid values)
            if dynamic_targets:
                # Dynamic stop loss based on volatility and candle structure
                sl_distance = atr_multiplier * result.loc[bullish_entries, "atr"]

                # For bullish MCs, find the lowest point of the candle
                result.loc[bullish_entries, "stop_loss"] = result.loc[bullish_entries, "low"] - sl_distance

                # Ensure stop loss is not too far (max 2% of price)
                max_distance = result.loc[bullish_entries, "entry_price"] * 0.02
                too_far_sl = (
                    result.loc[bullish_entries, "entry_price"] - result.loc[bullish_entries, "stop_loss"]
                ) > max_distance

                # Adjust stop loss if needed
                if too_far_sl.any():
                    result.loc[bullish_entries & too_far_sl, "stop_loss"] = (
                        result.loc[bullish_entries & too_far_sl, "entry_price"] - max_distance
                    )
            else:
                # Standard fixed ATR-based stop loss
                result.loc[bullish_entries, "stop_loss"] = (
                    result.loc[bullish_entries, "low"] - atr_multiplier * result.loc[bullish_entries, "atr"]
                )

            # Calculate take-profit levels for different RR ratios
            for rr in [1.0, 1.5, 2.0, 3.0]:
                risk = result.loc[bullish_entries, "entry_price"] - result.loc[bullish_entries, "stop_loss"]

                # Ensure risk is positive to avoid negative targets
                risk = np.maximum(risk, 0.0001)  # Minimum risk distance

                # Standard take profit calculation
                result.loc[bullish_entries, f"tp_rr_{rr}"] = result.loc[bullish_entries, "entry_price"] + (risk * rr)

                # If dynamic targets enabled, adjust targets based on volatility
                if dynamic_targets and "candle_height_pct" in result.columns:
                    # For highly volatile markets, slightly reduce targets for more realistic results
                    volatile_idx = bullish_entries & (
                        result["candle_height_pct"] > result["candle_height_pct"].median() * 2
                    )
                    if volatile_idx.any():
                        # Adjust targets by up to 15% for highly volatile candles
                        adjustment = 0.85 + (
                            0.15
                            * (
                                result.loc[volatile_idx, "candle_height_pct"].median()
                                / result.loc[volatile_idx, "candle_height_pct"]
                            )
                        )
                        result.loc[volatile_idx, f"tp_rr_{rr}"] = result.loc[volatile_idx, "entry_price"] + (
                            risk.loc[volatile_idx] * rr * adjustment
                        )

        # Find valid bearish signals
        bearish_entries = result["valid_signal"] & result["bearish_mc"]

        # Calculate entry parameters for bearish signals
        if bearish_entries.any():
            # Set basic signal info
            result.loc[bearish_entries, "entry_signal"] = True
            result.loc[bearish_entries, "signal_type"] = "sell"

            # Entry price: the low of the manipulation candle
            result.loc[bearish_entries, "entry_price"] = result.loc[bearish_entries, "low"]

            # Stop loss calculation
            if dynamic_targets:
                # Dynamic stop loss based on volatility and candle structure
                sl_distance = atr_multiplier * result.loc[bearish_entries, "atr"]

                # For bearish MCs, find the highest point of the candle
                result.loc[bearish_entries, "stop_loss"] = result.loc[bearish_entries, "high"] + sl_distance

                # Ensure stop loss is not too far (max 2% of price)
                max_distance = result.loc[bearish_entries, "entry_price"] * 0.02
                too_far_sl = (
                    result.loc[bearish_entries, "stop_loss"] - result.loc[bearish_entries, "entry_price"]
                ) > max_distance

                # Adjust stop loss if needed
                if too_far_sl.any():
                    result.loc[bearish_entries & too_far_sl, "stop_loss"] = (
                        result.loc[bearish_entries & too_far_sl, "entry_price"] + max_distance
                    )
            else:
                # Standard fixed ATR-based stop loss
                result.loc[bearish_entries, "stop_loss"] = (
                    result.loc[bearish_entries, "high"] + atr_multiplier * result.loc[bearish_entries, "atr"]
                )

            # Calculate take-profit levels for different RR ratios
            for rr in [1.0, 1.5, 2.0, 3.0]:
                risk = result.loc[bearish_entries, "stop_loss"] - result.loc[bearish_entries, "entry_price"]

                # Ensure risk is positive to avoid negative targets
                risk = np.maximum(risk, 0.0001)  # Minimum risk distance

                # Standard take profit calculation
                result.loc[bearish_entries, f"tp_rr_{rr}"] = result.loc[bearish_entries, "entry_price"] - (risk * rr)

                # If dynamic targets enabled, adjust targets based on volatility
                if dynamic_targets and "candle_height_pct" in result.columns:
                    # For highly volatile markets, slightly reduce targets for more realistic results
                    volatile_idx = bearish_entries & (
                        result["candle_height_pct"] > result["candle_height_pct"].median() * 2
                    )
                    if volatile_idx.any():
                        # Adjust targets by up to 15% for highly volatile candles
                        adjustment = 0.85 + (
                            0.15
                            * (
                                result.loc[volatile_idx, "candle_height_pct"].median()
                                / result.loc[volatile_idx, "candle_height_pct"]
                            )
                        )
                        result.loc[volatile_idx, f"tp_rr_{rr}"] = result.loc[volatile_idx, "entry_price"] - (
                            risk.loc[volatile_idx] * rr * adjustment
                        )

        # Add risk-reward info for backtest analysis
        result["risk_distance"] = abs(result["entry_price"] - result["stop_loss"])
        result["reward_distance"] = abs(result["entry_price"] - result["tp_rr_2.0"])
        with np.errstate(divide="ignore", invalid="ignore"):
            result["risk_reward_ratio"] = np.where(
                result["risk_distance"] > 0, result["reward_distance"] / result["risk_distance"], np.nan
            )

        return result

    except Exception as e:
        logger.error("generate_entry_signals_failed", error=str(e))
        return df


def identify_advanced_patterns(df: pd.DataFrame, lookback_period: int = 5) -> pd.DataFrame:
    """
    Identifies additional advanced chart patterns useful for crypto markets.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data
    lookback_period : int, optional
        Number of candles to look back for pattern identification (default: 5)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional pattern columns
    """
    try:
        # Check required columns
        required_columns = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_columns):
            logger.error("missing_required_ohlc_columns")
            return df

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Skip if dataframe is too small
        if len(result) < lookback_period + 2:
            logger.warning(
                "dataframe_too_small_for_pattern_identification",
                min_required=lookback_period + 2,
                row_count=len(result),
            )
            return result

        # Calculate body size and wicks
        result["body_size"] = abs(result["close"] - result["open"])
        result["upper_wick"] = result["high"] - np.maximum(result["open"], result["close"])
        result["lower_wick"] = np.minimum(result["open"], result["close"]) - result["low"]

        # Identify doji patterns (small body, long wicks)
        avg_body = result["body_size"].rolling(window=lookback_period).mean()
        result["is_doji"] = result["body_size"] < (avg_body * 0.3)

        # Identify inside bars (current bar's range within previous bar's range)
        result["inside_bar"] = (result["high"] < result["high"].shift(1)) & (result["low"] > result["low"].shift(1))

        # Identify outside bars (current bar's range engulfs previous bar's range)
        result["outside_bar"] = (result["high"] > result["high"].shift(1)) & (result["low"] < result["low"].shift(1))

        # Identify tweezer tops (bearish reversal)
        result["tweezer_top"] = (
            (abs(result["high"] - result["high"].shift(1)) < avg_body * 0.2)  # Similar highs
            & (result["close"].shift(1) > result["open"].shift(1))  # Previous bullish
            & (result["close"] < result["open"])  # Current bearish
            & (result["upper_wick"] > result["body_size"])  # Significant upper wick
        )

        # Identify tweezer bottoms (bullish reversal)
        result["tweezer_bottom"] = (
            (abs(result["low"] - result["low"].shift(1)) < avg_body * 0.2)  # Similar lows
            & (result["close"].shift(1) < result["open"].shift(1))  # Previous bearish
            & (result["close"] > result["open"])  # Current bullish
            & (result["lower_wick"] > result["body_size"])  # Significant lower wick
        )

        # Identify three pushes pattern - common in crypto (three consecutive higher highs followed by reversal)
        if len(result) >= 4:
            # Create a window of high values for the last 4 candles
            highs = (
                result["high"].rolling(window=4).apply(lambda x: x[1] > x[0] and x[2] > x[1] and x[3] < x[2], raw=True)
            )
            result["three_pushes_top"] = highs

            # Same for lows (three consecutive lower lows followed by reversal)
            lows = (
                result["low"].rolling(window=4).apply(lambda x: x[1] < x[0] and x[2] < x[1] and x[3] > x[2], raw=True)
            )
            result["three_pushes_bottom"] = lows

        # Check for exhaustion moves (high volume followed by reversal)
        if "volume" in result.columns:
            # Identify high volume candles
            avg_volume = result["volume"].rolling(window=lookback_period).mean()
            high_volume = result["volume"] > (avg_volume * 1.5)

            # Bullish exhaustion (high volume down candle followed by reversal)
            result["bullish_exhaustion"] = (
                high_volume.shift(1)  # High volume on previous candle
                & (result["close"].shift(1) < result["open"].shift(1))  # Previous bearish
                & (result["close"] > result["open"])  # Current bullish
                & (result["close"] > result["close"].shift(1))  # Price reversal
            )

            # Bearish exhaustion (high volume up candle followed by reversal)
            result["bearish_exhaustion"] = (
                high_volume.shift(1)  # High volume on previous candle
                & (result["close"].shift(1) > result["open"].shift(1))  # Previous bullish
                & (result["close"] < result["open"])  # Current bearish
                & (result["close"] < result["close"].shift(1))  # Price reversal
            )

        return result

    except Exception as e:
        logger.error("identify_advanced_patterns_failed", error=str(e))
        return df


def get_pattern_strength(
    df: pd.DataFrame, index: int, pattern_weights: dict[str, float] | None = None
) -> dict[str, object]:
    """
    Calculates the strength of a pattern at a specific index.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing pattern information
    index : int
        Index position to evaluate
    pattern_weights : dict, optional
        Dictionary with pattern weights for strength calculation

    Returns:
    --------
    dict
        Dictionary with pattern strength information
    """
    try:
        # Default pattern weights
        if pattern_weights is None:
            pattern_weights = {
                "bullish_mc": 0.6,
                "bearish_mc": 0.6,
                "bullish_pinbar": 0.5,
                "bearish_pinbar": 0.5,
                "bullish_engulfing": 0.4,
                "bearish_engulfing": 0.4,
                "inside_bar": 0.3,
                "outside_bar": 0.4,
                "tweezer_top": 0.4,
                "tweezer_bottom": 0.4,
                "three_pushes_top": 0.5,
                "three_pushes_bottom": 0.5,
                "bullish_exhaustion": 0.5,
                "bearish_exhaustion": 0.5,
                "is_doji": 0.2,
            }

        # Get row at index
        if index < 0 or index >= len(df):
            logger.error("index_out_of_bounds", index=index, dataframe_length=len(df))
            return {"strength": 0.0, "patterns": []}

        row = df.iloc[index]

        # Calculate pattern strength
        strength = 0.0
        patterns = []

        # Check each pattern
        for pattern, weight in pattern_weights.items():
            if pattern in row and row[pattern]:
                strength += weight
                patterns.append(pattern)

        # Additional context factors
        if "trend" in row and (
            row["trend"] == "up"
            and any(p.startswith("bullish") for p in patterns)
            or row["trend"] == "down"
            and any(p.startswith("bearish") for p in patterns)
        ):
            strength += 0.2  # Bonus for trend-aligned patterns

        if "high_volatility" in row and row["high_volatility"]:
            strength += 0.15  # Bonus for high volatility context

        # Normalize to 0-1 range (cap at 1.0)
        strength = min(strength, 1.0)

        return {
            "strength": strength,
            "patterns": patterns,
            "is_bullish": any(p.startswith("bullish") for p in patterns),
            "is_bearish": any(p.startswith("bearish") for p in patterns),
        }

    except Exception as e:
        logger.error("get_pattern_strength_failed", error=str(e))
        return {"strength": 0.0, "patterns": []}


def filter_signals_for_crypto(
    df: pd.DataFrame, min_strength: float = 0.6, respect_key_levels: bool = True
) -> pd.DataFrame:
    """
    Apply crypto-specific filtering to trading signals.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with trading signals
    min_strength : float, optional
        Minimum pattern strength for valid crypto signals (default: 0.6)
    respect_key_levels : bool, optional
        Only trade signals that respect key levels (default: True)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with filtered signals
    """
    try:
        if df is None or df.empty or "entry_signal" not in df.columns:
            logger.error("invalid_dataframe_for_crypto_signal_filtering")
            return df

        result = df.copy()

        # Skip if no signals to filter
        if not bool(result["entry_signal"].any()):
            return result

        # Apply strength filtering if signal_strength column exists
        initial_count: int = result["entry_signal"].sum()  # type: ignore[assignment]

        if "signal_strength" in result.columns:
            # Filter by minimum strength
            weak_signals = (result["signal_strength"] < min_strength) & result["entry_signal"]
            result.loc[weak_signals, "entry_signal"] = False
            logger.info("removed_weak_signals", count=int(weak_signals.sum()), min_strength=min_strength)

        # Respect key levels if enabled and columns exist
        if respect_key_levels and "key_level" in result.columns:
            # Check if signals are near key levels
            non_key_signals: pd.Series = ~result["key_level"] & result["entry_signal"]

            # Check if the signal is in any area of interest
            aoi_columns = [col for col in result.columns if "in_" in col or "at_" in col or "is_" in col]

            if aoi_columns:
                # Create a combined mask for being in any area of interest
                aoi_mask = pd.Series(result[aoi_columns].any(axis=1), dtype=bool)
                non_key_signals = pd.Series(~aoi_mask & non_key_signals, dtype=bool)

            # Remove signals not near key levels or areas of interest
            result.loc[non_key_signals, "entry_signal"] = False
            logger.info("removed_non_key_level_signals", count=int(non_key_signals.sum()))

        # Add volatility-based filtering for crypto
        if "candle_height_pct" in result.columns:
            # Skip very low volatility signals
            low_vol_signals = (result["candle_height_pct"] < result["candle_height_pct"].median() * 0.5) & result[
                "entry_signal"
            ]
            result.loc[low_vol_signals, "entry_signal"] = False
            logger.info("removed_low_volatility_signals", count=int(low_vol_signals.sum()))

        # Log signals removed
        final_count = result["entry_signal"].sum()
        removed_count = initial_count - final_count
        logger.info(
            "crypto_filtering_complete",
            removed_count=removed_count,
            initial_count=initial_count,
            removed_pct=round(removed_count / initial_count * 100, 1) if initial_count else 0,
        )

        return result

    except Exception as e:
        logger.error("filter_signals_for_crypto_failed", error=str(e))
        return df
