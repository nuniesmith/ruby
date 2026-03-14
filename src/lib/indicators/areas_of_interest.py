"""
Manipulation Model - Areas of Interest Module

This module identifies key price levels and zones where manipulation candles
are more likely to occur, including fair value gaps, supply/demand zones,
key levels, and session highs/lows.
"""

import numpy as np
import pandas as pd

from lib.core.logging_config import get_logger

# (utils.datetime_utils / utils.config_utils are from the original project; not needed here)

logger = get_logger(__name__)


def identify_fair_value_gaps(df, gap_threshold=0.0005):
    """
    Identifies Fair Value Gaps (FVGs) where price moved too quickly.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data
    gap_threshold : float, optional
        Minimum size of gap as a percentage (default: 0.0005 or 0.05%)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - 'bullish_fvg': Boolean flag for bullish FVGs
        - 'bearish_fvg': Boolean flag for bearish FVGs
        - 'fvg_level': Price level of the FVG
    """
    try:
        result = df.copy()

        # Initialize FVG columns
        result["bullish_fvg"] = False
        result["bearish_fvg"] = False
        result["fvg_level"] = np.nan

        # Skip if dataframe is empty
        if len(result) < 2:
            logger.warning("DataFrame too short to identify fair value gaps")
            return result

        # Bullish FVG: Current candle's low is higher than previous candle's high
        bullish_fvg = result["low"] > result["high"].shift(1)
        result.loc[bullish_fvg, "bullish_fvg"] = True
        result.loc[bullish_fvg, "fvg_level"] = (result["low"] + result["high"].shift(1)) / 2

        # Bearish FVG: Current candle's high is lower than previous candle's low
        bearish_fvg = result["high"] < result["low"].shift(1)
        result.loc[bearish_fvg, "bearish_fvg"] = True
        result.loc[bearish_fvg, "fvg_level"] = (result["high"] + result["low"].shift(1)) / 2

        # Apply threshold to filter out small gaps
        gap_size = abs(result["fvg_level"] / result["close"] - 1)
        small_gaps = gap_size < gap_threshold
        result.loc[small_gaps, ["bullish_fvg", "bearish_fvg", "fvg_level"]] = [False, False, np.nan]

        return result
    except Exception as e:
        logger.error("identify_fair_value_gaps_failed", error=str(e))
        return df


def identify_supply_demand_zones(df, zone_length=5, zone_strength_threshold=3, lookforward_window=20):
    """
    Identifies supply and demand zones based on price action.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data
    zone_length : int, optional
        Number of candles to look for zone validation (default: 5)
    zone_strength_threshold : int, optional
        Minimum number of touches to consider a zone strong (default: 3)
    lookforward_window : int, optional
        Maximum number of candles to look forward for zone touches (default: 20)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - 'supply_zone': Boolean flag for supply zones
        - 'demand_zone': Boolean flag for demand zones
        - 'zone_high': Upper price of the zone
        - 'zone_low': Lower price of the zone
        - 'zone_strength': Number of touches validating the zone
    """
    try:
        result = df.copy()

        # Initialize zone columns
        result["supply_zone"] = False
        result["demand_zone"] = False
        result["zone_high"] = np.nan
        result["zone_low"] = np.nan
        result["zone_strength"] = 0

        # Skip if dataframe is too short
        if len(result) <= zone_length:
            logger.warning(
                "dataframe_too_short_for_zone_identification",
                row_count=len(result),
                min_required=zone_length + 1,
            )
            return result

        # Pre-compute vectorized conditions for better performance
        # For supply zones
        high_is_max = result["high"].rolling(zone_length).max() == result["high"]
        close_lt_open = result["close"] < result["open"]
        close_lt_prev_close = result["close"] < result["close"].shift(1)

        # Combine conditions for potential supply zones
        potential_supply = high_is_max & close_lt_open & close_lt_prev_close

        # For demand zones
        low_is_min = result["low"].rolling(zone_length).min() == result["low"]
        close_gt_open = result["close"] > result["open"]
        close_gt_prev_close = result["close"] > result["close"].shift(1)

        # Combine conditions for potential demand zones
        potential_demand = low_is_min & close_gt_open & close_gt_prev_close

        # Process supply zones - still need a loop for zone validation
        for i in range(zone_length, len(result)):
            # Skip if not a potential supply zone
            if not potential_supply.iloc[i]:
                continue

            zone_high = result["high"].iloc[i]
            zone_low = result["close"].iloc[i]

            # Create a mask for candles that touch the zone
            touches_mask = np.zeros(min(lookforward_window, len(result) - i), dtype=bool)

            # Vectorized approach for checking touches
            lookforward_slice = result.iloc[i : i + lookforward_window]
            touches_mask = (
                (lookforward_slice["high"] >= zone_low)
                & (lookforward_slice["high"] <= zone_high)
                & (lookforward_slice["close"] < lookforward_slice["open"])
            )

            touches = touches_mask.sum()

            if touches >= zone_strength_threshold:
                result.loc[result.index[i], "supply_zone"] = True
                result.loc[result.index[i], "zone_high"] = zone_high
                result.loc[result.index[i], "zone_low"] = zone_low
                result.loc[result.index[i], "zone_strength"] = touches

        # Process demand zones - still need a loop for zone validation
        for i in range(zone_length, len(result)):
            # Skip if not a potential demand zone
            if not potential_demand.iloc[i]:
                continue

            zone_high = result["open"].iloc[i]
            zone_low = result["low"].iloc[i]

            # Create a mask for candles that touch the zone
            touches_mask = np.zeros(min(lookforward_window, len(result) - i), dtype=bool)

            # Vectorized approach for checking touches
            lookforward_slice = result.iloc[i : i + lookforward_window]
            touches_mask = (
                (lookforward_slice["low"] >= zone_low)
                & (lookforward_slice["low"] <= zone_high)
                & (lookforward_slice["close"] > lookforward_slice["open"])
            )

            touches = touches_mask.sum()

            if touches >= zone_strength_threshold:
                result.loc[result.index[i], "demand_zone"] = True
                result.loc[result.index[i], "zone_high"] = zone_high
                result.loc[result.index[i], "zone_low"] = zone_low
                result.loc[result.index[i], "zone_strength"] = touches

        return result
    except Exception as e:
        logger.error("identify_supply_demand_zones_failed", error=str(e))
        return df


def identify_key_levels(df, round_digits=2):
    """
    Identifies key psychological levels like round numbers.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data
    round_digits : int, optional
        Number of decimal places for rounding (default: 2)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional column:
        - 'key_level': Boolean flag for prices near key levels
    """
    try:
        result = df.copy()

        # Initialize key level column
        result["key_level"] = False

        # Find round numbers based on the asset's typical price range
        multiplier = 10**round_digits

        # Vectorized approach instead of loop
        prices = result["close"]
        rounded_prices = np.round(prices * multiplier) / multiplier

        # Mark candles where price is close to a round number
        is_key_level = np.abs(prices - rounded_prices) / prices < 0.001  # Within 0.1%
        result.loc[is_key_level, "key_level"] = True

        # Add significant figure levels for crypto (BTC)
        # Check if price is near thousands (1000, 2000, etc.)
        if "close" in result.columns and result["close"].median() > 1000:
            thousands_level = np.round(prices / 1000) * 1000
            near_thousands = np.abs(prices - thousands_level) / prices < 0.005  # Within 0.5%
            result.loc[near_thousands, "key_level"] = True

            # For BTC, also check significant levels like 10k, 20k, etc.
            if result["close"].median() > 5000:  # Likely BTC
                ten_thousands_level = np.round(prices / 10000) * 10000
                near_ten_thousands = np.abs(prices - ten_thousands_level) / prices < 0.01  # Within 1%
                result.loc[near_ten_thousands, "key_level"] = True

        return result
    except Exception as e:
        logger.error("identify_key_levels_failed", error=str(e))
        return df


def identify_session_levels(df, time_column="datetime"):
    """
    Identifies daily opens, session highs/lows, NY midnight open, etc.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data with a datetime index or column
    time_column : str, optional
        Name of the column containing datetime information (default: 'datetime')

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns for session levels
    """
    try:
        result = df.copy()

        # Check if we have enough data
        if len(result) == 0:
            logger.warning("empty_dataframe_for_session_levels")
            return result

        # Ensure we have datetime information as a Series (not Index)
        if time_column in result.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(result[time_column]):
                time_data = pd.to_datetime(result[time_column], errors="coerce")
            else:
                time_data = result[time_column]
        else:
            # Convert index to Series to ensure .dt accessor works
            try:
                time_data = pd.Series(pd.to_datetime(result.index, errors="coerce"), index=result.index)
            except Exception:
                logger.error("could_not_extract_datetime_from_index")
                return result

        # Skip if we couldn't get valid datetime data
        if time_data.isna().all():
            logger.warning("no_valid_datetime_data_found")
            return result

        # Initialize session columns
        result["is_daily_open"] = False
        result["is_ny_midnight"] = False
        result["is_session_high"] = False
        result["is_session_low"] = False
        result["is_weekly_high"] = False
        result["is_weekly_low"] = False

        # Extract datetime components safely
        try:
            # Extract hour and minute
            hour_data = time_data.dt.hour
            minute_data = time_data.dt.minute

            # Mark daily opens (0:00) - vectorized
            result.loc[(hour_data == 0) & (minute_data == 0), "is_daily_open"] = True

            # Mark NY midnight (adjust based on data timezone) - vectorized
            ny_midnight_hour = 5  # Assuming data is in America/Toronto, NY midnight is 5:00 America/Toronto
            result.loc[(hour_data == ny_midnight_hour) & (minute_data == 0), "is_ny_midnight"] = True

            # Extract date
            date_data = time_data.dt.date

            # Create temporary date column for grouping
            result["temp_date"] = date_data

            # Group by day to find session highs/lows - this part is hard to vectorize
            result_clean = result.dropna(subset=["temp_date"])
            if not result_clean.empty:
                # Get daily high/low info with indices
                daily_high_idx = result_clean.groupby("temp_date")["high"].idxmax()
                daily_low_idx = result_clean.groupby("temp_date")["low"].idxmin()

                # Set session high/low flags
                result.loc[daily_high_idx, "is_session_high"] = True
                result.loc[daily_low_idx, "is_session_low"] = True

            # Extract year and week for weekly calculations
            result["temp_year"] = time_data.dt.isocalendar().year
            result["temp_week"] = time_data.dt.isocalendar().week

            # Group by week to find weekly highs/lows
            result_clean = result.dropna(subset=["temp_year", "temp_week"])
            if not result_clean.empty:
                # Get weekly high/low info with indices
                weekly_high_idx = result_clean.groupby(["temp_year", "temp_week"])["high"].idxmax()
                weekly_low_idx = result_clean.groupby(["temp_year", "temp_week"])["low"].idxmin()

                # Set weekly high/low flags
                result.loc[weekly_high_idx, "is_weekly_high"] = True
                result.loc[weekly_low_idx, "is_weekly_low"] = True

        except Exception as e:
            logger.error("error_extracting_datetime_components", error=str(e))

        # Clean up temporary columns
        result = result.drop(columns=["temp_date", "temp_year", "temp_week"], errors="ignore")

        return result
    except Exception as e:
        logger.error("identify_session_levels_failed", error=str(e))
        return df


def is_price_in_area_of_interest(df, current_price=None, buffer_percentage=0.001):
    """
    Determines if the current price is in any area of interest.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with identified areas of interest
    current_price : float, optional
        The current price to check (default: latest close price)
    buffer_percentage : float, optional
        Percentage buffer around levels to consider price "at" the level (default: 0.1%)

    Returns:
    --------
    dict
        Dictionary with Boolean flags for each type of area of interest
    """
    try:
        # Make sure we have a dataframe with data
        if df is None or df.empty:
            logger.warning("empty_dataframe_for_area_of_interest")
            return {
                "in_fair_value_gap": False,
                "in_supply_zone": False,
                "in_demand_zone": False,
                "at_key_level": False,
                "at_session_level": False,
                "in_any_area": False,
            }

        # Get current price if not provided
        if current_price is None:
            current_price = df["close"].iloc[-1]

        # Calculate buffer
        buffer = current_price * buffer_percentage

        # Initialize result dict
        result = {
            "in_fair_value_gap": False,
            "in_supply_zone": False,
            "in_demand_zone": False,
            "at_key_level": False,
            "at_session_level": False,
            "in_any_area": False,
        }

        # Vectorized approach for FVG check
        if "fvg_level" in df.columns:
            fvg_df = df[df["fvg_level"].notna()]
            if not fvg_df.empty:
                # Vectorized check if any FVG level is near current price
                near_fvg = np.any(np.abs(fvg_df["fvg_level"] - current_price) < buffer)
                result["in_fair_value_gap"] = bool(near_fvg)

        # Vectorized approach for supply zone
        if all(col in df.columns for col in ["supply_zone", "zone_low", "zone_high"]):
            supply_df = df[df["supply_zone"]]
            if not supply_df.empty:
                # Vectorized check if any supply zone contains current price
                in_supply = np.any((supply_df["zone_low"] <= current_price) & (current_price <= supply_df["zone_high"]))
                result["in_supply_zone"] = bool(in_supply)

        # Vectorized approach for demand zone
        if all(col in df.columns for col in ["demand_zone", "zone_low", "zone_high"]):
            demand_df = df[df["demand_zone"]]
            if not demand_df.empty:
                # Vectorized check if any demand zone contains current price
                in_demand = np.any((demand_df["zone_low"] <= current_price) & (current_price <= demand_df["zone_high"]))
                result["in_demand_zone"] = bool(in_demand)

        # Vectorized approach for key level
        if "key_level" in df.columns:
            key_level_df = df[df["key_level"]]
            if not key_level_df.empty:
                # Vectorized check if any key level is near current price
                at_key_level = np.any(np.abs(key_level_df["close"] - current_price) < buffer)
                result["at_key_level"] = bool(at_key_level)

        # Vectorized approach for session levels
        session_columns = [
            "is_daily_open",
            "is_ny_midnight",
            "is_session_high",
            "is_session_low",
            "is_weekly_high",
            "is_weekly_low",
        ]

        for col in session_columns:
            if col in df.columns:
                session_df = df[df[col]]
                if not session_df.empty:
                    # Vectorized check if any session level is near current price
                    at_session_level = np.any(np.abs(session_df["close"] - current_price) < buffer)
                    if bool(at_session_level):
                        result["at_session_level"] = True
                        break

        # Update the in_any_area flag
        result["in_any_area"] = (
            result["in_fair_value_gap"]
            or result["in_supply_zone"]
            or result["in_demand_zone"]
            or result["at_key_level"]
            or result["at_session_level"]
        )

        return result
    except Exception as e:
        logger.error("is_price_in_area_of_interest_failed", error=str(e))
        return {
            "in_fair_value_gap": False,
            "in_supply_zone": False,
            "in_demand_zone": False,
            "at_key_level": False,
            "at_session_level": False,
            "in_any_area": False,
        }


# Bitcoin-specific price level function
def identify_bitcoin_specific_levels(df, fibonacci_base=None):
    """
    Identifies Bitcoin-specific price levels like halving prices, all-time highs, etc.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing BTC/USD price data
    fibonacci_base : tuple, optional
        Base levels (low, high) for calculating Fibonacci retracements

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional columns for Bitcoin-specific levels
    """
    try:
        result = df.copy()

        # Initialize columns
        result["is_ath"] = False
        result["is_fibonacci_level"] = False
        result["is_halving_price"] = False

        # Find all-time high
        if len(result) > 0:
            all_time_high = result["high"].expanding().max()
            result.loc[result["high"] == all_time_high, "is_ath"] = True

        # Add historical halving prices (approximate values)
        halving_prices = [
            8000,  # 2020 halving (approximate)
            3500,  # 2016 halving (approximate)
            650,  # 2012 halving (approximate)
            21000,  # Projected 2024 halving
            42000,  # Key psychological level after 2021 ATH
            69000,  # 2021 ATH
        ]

        # Mark candles near halving prices
        for price in halving_prices:
            price_buffer = price * 0.02  # 2% buffer
            is_near_halving_price = (result["high"] >= price - price_buffer) & (result["low"] <= price + price_buffer)
            result.loc[is_near_halving_price, "is_halving_price"] = True

        # Calculate Fibonacci levels if base provided
        if fibonacci_base is not None:
            base_low, base_high = fibonacci_base
            range_size = base_high - base_low

            # Standard Fibonacci retracement levels
            fib_levels = [
                base_high,  # 0% retracement
                base_high - 0.236 * range_size,  # 23.6%
                base_high - 0.382 * range_size,  # 38.2%
                base_high - 0.5 * range_size,  # 50%
                base_high - 0.618 * range_size,  # 61.8%
                base_high - 0.786 * range_size,  # 78.6%
                base_low,  # 100% retracement
            ]

            # Extension levels
            fib_levels.extend(
                [
                    base_high + 0.618 * range_size,  # 61.8% extension
                    base_high + 1.0 * range_size,  # 100% extension
                    base_high + 1.618 * range_size,  # 161.8% extension
                ]
            )

            # Mark candles near Fibonacci levels
            for level in fib_levels:
                level_buffer = level * 0.01  # 1% buffer
                is_near_fib_level = (result["high"] >= level - level_buffer) & (result["low"] <= level + level_buffer)
                result.loc[is_near_fib_level, "is_fibonacci_level"] = True

        return result
    except Exception as e:
        logger.error("identify_bitcoin_specific_levels_failed", error=str(e))
        return df
