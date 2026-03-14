"""
Market Timing Module

This module provides functions to determine optimal trading times
based on session activity (Asian, London, New York) and volatility.
Supports both traditional forex and crypto markets with configurable
session timings.
"""

from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz  # type: ignore[import-untyped]

from lib.core.logging_config import get_logger


# Import shared utilities
def apply_datetime_patch():
    """No-op shim replacing utils.datetime_utils.apply_datetime_patch."""
    pass


def safe_convert_timezone(dt, tz):
    """No-op shim - returns dt unchanged."""
    return dt


class _Config:
    """No-op config shim."""

    def get_sessions(self) -> dict:
        return {}

    def get_crypto_sessions(self) -> dict:
        return {}

    def get_default_timezone(self) -> str:
        return ""


config = _Config()

logger = get_logger(__name__)

# Get session times from configuration, falling back to defaults if not configured
_cfg_sessions = config.get_sessions()
DEFAULT_SESSIONS: dict = (
    _cfg_sessions
    if _cfg_sessions
    else {
        "asian": {"start": time(0, 0), "end": time(8, 0)},
        "london": {"start": time(8, 0), "end": time(16, 30)},
        "ny": {"start": time(14, 0), "end": time(22, 0)},
        "overlap": {"start": time(14, 0), "end": time(16, 30)},  # London-NY overlap
    }
)

# Get crypto-specific session times
_cfg_crypto = config.get_crypto_sessions()
CRYPTO_SESSIONS: dict = (
    _cfg_crypto
    if _cfg_crypto
    else {
        "high_volume": {"start": time(14, 0), "end": time(22, 0)},
        "low_volume": {"start": time(0, 0), "end": time(6, 0)},
    }
)

# Get default timezone from configuration
_cfg_tz = config.get_default_timezone()
DEFAULT_TIMEZONE: str = _cfg_tz if _cfg_tz else "America/Toronto"


def is_active_session(
    timestamp: pd.Timestamp | datetime | None = None,
    tz: str = DEFAULT_TIMEZONE,
    session_times: dict[str, dict[str, time]] | None = None,
    asset_type: str = "forex",
) -> dict[str, bool]:
    """
    Determines if the current time is within an active trading session.

    Parameters:
    -----------
    timestamp : pandas.Timestamp or datetime.datetime, optional
        The timestamp to check (default: current time)
    tz : str, optional
        Timezone for session times (default: from config or 'America/Toronto')
    session_times : dict, optional
        Custom session times (default: use from config)
    asset_type : str, optional
        Type of asset ('forex', 'btc', or 'gold') for specific optimizations (default: 'forex')

    Returns:
    --------
    dict
        Dictionary with Boolean flags for each trading session and activity level
    """
    try:
        # Set timestamp to current time if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Use default session times if none provided
        if session_times is None:
            if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
                # Merge regular sessions with crypto-specific sessions
                session_times = DEFAULT_SESSIONS.copy()
                session_times.update(
                    {
                        "crypto_high": CRYPTO_SESSIONS.get("high_volume", {"start": time(14, 0), "end": time(22, 0)}),
                        "crypto_low": CRYPTO_SESSIONS.get("low_volume", {"start": time(0, 0), "end": time(6, 0)}),
                    }
                )
            else:
                session_times = DEFAULT_SESSIONS

        # Convert timestamp to the specified timezone
        ts = safe_convert_timezone(timestamp, tz)

        # Get current time
        current_time = ts.time()

        # Initialize result dictionary
        result = {
            "in_asian_session": False,
            "in_london_session": False,
            "in_ny_session": False,
            "in_overlap_session": False,
            "is_optimal_trading_time": False,
            "is_high_volume_session": False,
        }

        # Add crypto-specific session indicators if appropriate
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            result.update({"in_crypto_high_volume": False, "in_crypto_low_volume": False})

        # Check if current time is in each session
        for session_name, session_info in session_times.items():
            if "start" in session_info and "end" in session_info:
                session_start = session_info["start"]
                session_end = session_info["end"]

                # Build the session key based on session name
                session_key = f"in_{session_name}_session"
                if session_name.startswith("crypto_"):
                    session_key = f"in_{session_name}"

                # Check if we're in this session and add to result if we have a key for it
                if session_key in result:
                    result[session_key] = _is_time_in_session(current_time, session_start, session_end)

        # Determine optimal trading time
        result["is_optimal_trading_time"] = result["in_london_session"] or result["in_ny_session"]

        # For crypto, also consider crypto-specific high volume periods
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"] and "in_crypto_high_volume" in result:
            result["is_optimal_trading_time"] = result["is_optimal_trading_time"] or result["in_crypto_high_volume"]

        # Determine high volume session (overlap or crypto high volume)
        result["is_high_volume_session"] = result["in_overlap_session"]

        # For crypto, also consider crypto-specific high volume periods
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"] and "in_crypto_high_volume" in result:
            result["is_high_volume_session"] = result["is_high_volume_session"] or result["in_crypto_high_volume"]

        return result

    except Exception as e:
        logger.error("is_active_session_failed", error=str(e))
        # Return all False as default
        default_result = {
            "in_asian_session": False,
            "in_london_session": False,
            "in_ny_session": False,
            "in_overlap_session": False,
            "is_optimal_trading_time": False,
            "is_high_volume_session": False,
        }

        # Add crypto-specific session indicators if appropriate
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            default_result.update({"in_crypto_high_volume": False, "in_crypto_low_volume": False})

        return default_result


def _is_time_in_session(current_time: time, session_start: time, session_end: time) -> bool:
    """
    Helper function to check if a time is within a session.
    Handles sessions that cross midnight.

    Parameters:
    -----------
    current_time : datetime.time
        The time to check
    session_start : datetime.time
        Session start time
    session_end : datetime.time
        Session end time

    Returns:
    --------
    bool
        True if time is in session, False otherwise
    """
    try:
        # Regular case: start < end
        if session_start < session_end:
            return session_start <= current_time <= session_end
        # Overnight case: start > end (crosses midnight)
        else:
            return current_time >= session_start or current_time <= session_end
    except Exception as e:
        logger.error("is_time_in_session_failed", error=str(e))
        return False


def analyze_session_performance(
    df: pd.DataFrame, time_column: str = "datetime", asset_type: str = "forex"
) -> pd.DataFrame:
    """
    Analyzes trading performance during different market sessions.
    Optimized with vectorized operations for better performance.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data with MC signals and a datetime column
    time_column : str, optional
        Name of the column containing datetime information (default: 'datetime')
    asset_type : str, optional
        Type of asset ('forex', 'btc', or 'gold') for specific analysis (default: 'forex')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with session statistics and performance metrics
    """
    try:
        # Check if dataframe is empty
        if df is None or df.empty:
            logger.warning("empty_dataframe_for_session_performance")
            return pd.DataFrame(
                {
                    "session": ["Asian", "London", "New York", "Overlap", "Other"],
                    "count": [0, 0, 0, 0, 0],
                    "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )

        # Ensure OHLC columns exist
        required_columns = ["high", "low"]
        for col in required_columns:
            if col not in df.columns:
                logger.error("required_column_not_found", column=col)
                return pd.DataFrame(
                    {
                        "session": ["Asian", "London", "New York", "Overlap", "Other"],
                        "count": [0, 0, 0, 0, 0],
                        "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "error": f"Missing column: {col}",
                    }
                )

        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Get datetime information with proper error handling
        try:
            # Try different methods to get datetime information
            if time_column in df_copy.columns:
                df_copy["datetime"] = pd.to_datetime(df_copy[time_column], errors="coerce")
            elif isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy["datetime"] = df_copy.index
            else:
                try:
                    df_copy["datetime"] = pd.to_datetime(df_copy.index, errors="coerce")
                except Exception as e1:
                    logger.error("could_not_extract_datetime_from_index", error=str(e1))
                    # Try using a temporary Series with .dt accessor
                    try:
                        datetime_series = pd.Series(df_copy.index).apply(pd.to_datetime)
                        df_copy["datetime"] = datetime_series.values
                    except Exception as e2:
                        logger.error("all_datetime_extraction_attempts_failed", error=str(e2))
                        return pd.DataFrame(
                            {
                                "session": ["Asian", "London", "New York", "Overlap", "Other"],
                                "count": [0, 0, 0, 0, 0],
                                "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                                "error": "No datetime information available",
                            }
                        )
        except Exception as e:
            logger.error("error_extracting_datetime_information", error=str(e))
            return pd.DataFrame(
                {
                    "session": ["Asian", "London", "New York", "Overlap", "Other"],
                    "count": [0, 0, 0, 0, 0],
                    "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "error": f"Error extracting datetime: {e}",
                }
            )

        # Check if we have valid datetime data
        if bool(df_copy["datetime"].isna().all()):
            logger.error("all_datetime_values_invalid")
            return pd.DataFrame(
                {
                    "session": ["Asian", "London", "New York", "Overlap", "Other"],
                    "count": [0, 0, 0, 0, 0],
                    "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "error": "Invalid datetime values",
                }
            )

        # Extract hour with safe handling
        try:
            # Create a Series from datetime for .dt access if needed
            datetime_series = pd.Series(df_copy["datetime"])
            df_copy["hour"] = datetime_series.dt.hour
        except Exception as e:
            logger.error("error_extracting_hour_component", error=str(e))
            return pd.DataFrame(
                {
                    "session": ["Asian", "London", "New York", "Overlap", "Other"],
                    "count": [0, 0, 0, 0, 0],
                    "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "error": f"Hour extraction failed: {e}",
                }
            )

        # Define session hour ranges based on asset type
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            # Add crypto-specific sessions
            sessions = DEFAULT_SESSIONS.copy()
            sessions.update(
                {
                    "crypto_high": CRYPTO_SESSIONS.get("high_volume", {"start": time(14, 0), "end": time(22, 0)}),
                    "crypto_low": CRYPTO_SESSIONS.get("low_volume", {"start": time(0, 0), "end": time(6, 0)}),
                }
            )
        else:
            sessions = DEFAULT_SESSIONS

        # Mark session flags - vectorized operations
        # Handle regular session flags
        for session_name, session_info in sessions.items():
            start_hour = session_info["start"].hour
            end_hour = session_info["end"].hour

            # Handle overnight sessions correctly
            if start_hour < end_hour:
                df_copy[f"is_{session_name}"] = df_copy["hour"].between(start_hour, end_hour, inclusive="left")
            else:
                df_copy[f"is_{session_name}"] = ~df_copy["hour"].between(end_hour, start_hour, inclusive="both")

        # Define 'Other' as not in any regular session (excluding crypto sessions)
        regular_sessions = ["asian", "london", "ny", "overlap"]
        df_copy["is_other"] = ~df_copy[[f"is_{s}" for s in regular_sessions]].any(axis=1)

        # Calculate volatility and other stats by session
        session_stats = []
        session_mapping = [
            ("Asian", "is_asian"),
            ("London", "is_london"),
            ("New York", "is_ny"),
            ("Overlap", "is_overlap"),
            ("Other", "is_other"),
        ]

        # Add crypto sessions if applicable
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            session_mapping.extend([("Crypto High Volume", "is_crypto_high"), ("Crypto Low Volume", "is_crypto_low")])

        # Calculate session statistics with vectorized operations where possible
        for session_name, mask_col in session_mapping:
            if mask_col in df_copy.columns:
                session_mask = df_copy[mask_col]
                session_df = df_copy[session_mask]
                count = len(session_df)

                if count > 0:
                    # Calculate volatility - vectorized
                    volatility = (session_df["high"] - session_df["low"]).mean()
                    volatility = 0.0 if pd.isna(volatility) else volatility

                    # Calculate volume (if available) - vectorized
                    if "volume" in session_df.columns:
                        avg_volume = session_df["volume"].mean()
                        avg_volume = 0.0 if bool(pd.isna(avg_volume)) else avg_volume
                    else:
                        avg_volume = None

                    # Calculate price movement - vectorized
                    if "close" in session_df.columns and "open" in session_df.columns:
                        price_move = (session_df["close"] - session_df["open"]).mean()
                        price_move = 0.0 if pd.isna(price_move) else price_move
                    else:
                        price_move = None

                    # MC signal success rate calculation (needs loop for sequence analysis)
                    bullish_success = None
                    bearish_success = None

                    if "bullish_mc" in df.columns:
                        bullish_df = session_df[session_df["bullish_mc"]]
                        if isinstance(bullish_df, pd.DataFrame) and len(bullish_df) > 0:
                            # Setup for success rate calculation
                            success_count = 0
                            total_count = len(bullish_df)

                            # Process each bullish signal
                            for idx in bullish_df.index:  # type: ignore[union-attr]
                                if idx in df.index:
                                    _loc = df.index.get_loc(idx)
                                    idx_pos = int(_loc) if isinstance(_loc, (int, np.integer)) else -1
                                    if idx_pos >= 0 and idx_pos + 3 < len(df):
                                        next_prices = df.iloc[idx_pos + 1 : idx_pos + 4]["close"].values
                                        current_price = df.loc[idx, "close"]
                                        if any(next_prices > current_price):
                                            success_count += 1

                            bullish_success = success_count / total_count if total_count > 0 else 0.0

                    if "bearish_mc" in df.columns:
                        bearish_df = session_df[session_df["bearish_mc"]]
                        if isinstance(bearish_df, pd.DataFrame) and len(bearish_df) > 0:
                            # Setup for success rate calculation
                            success_count = 0
                            total_count = len(bearish_df)

                            # Process each bearish signal
                            for idx in bearish_df.index:  # type: ignore[union-attr]
                                if idx in df.index:
                                    _loc = df.index.get_loc(idx)
                                    idx_pos = int(_loc) if isinstance(_loc, (int, np.integer)) else -1
                                    if idx_pos >= 0 and idx_pos + 3 < len(df):
                                        next_prices = df.iloc[idx_pos + 1 : idx_pos + 4]["close"].values
                                        current_price = df.loc[idx, "close"]
                                        if any(next_prices < current_price):
                                            success_count += 1

                            bearish_success = success_count / total_count if total_count > 0 else 0.0

                    # Build stats dictionary
                    stats = {
                        "session": session_name,
                        "count": count,
                        "avg_volatility": volatility,
                        "bullish_mc_success_rate": bullish_success,
                        "bearish_mc_success_rate": bearish_success,
                    }

                    # Add additional metrics if available
                    if avg_volume is not None:
                        stats["avg_volume"] = avg_volume

                    if price_move is not None:
                        stats["avg_price_move"] = price_move

                    session_stats.append(stats)
                else:
                    # Empty session
                    session_stats.append(
                        {
                            "session": session_name,
                            "count": 0,
                            "avg_volatility": 0.0,
                            "bullish_mc_success_rate": None,
                            "bearish_mc_success_rate": None,
                        }
                    )

        # Convert to DataFrame
        result_df = pd.DataFrame(session_stats)

        # Add additional analysis and metrics if needed
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            # Add a ranking column based on volatility
            if "avg_volatility" in result_df.columns:
                result_df["volatility_rank"] = result_df["avg_volatility"].rank(ascending=False)

            # Add session recommendation
            result_df["recommended_for_trading"] = False

            # Recommend high volatility sessions with good success rates for crypto
            if "avg_volatility" in result_df.columns and "bullish_mc_success_rate" in result_df.columns:
                # Calculate median volatility
                median_vol = result_df["avg_volatility"].median()

                # Mark sessions with above-median volatility and good success rates
                good_sessions = (result_df["avg_volatility"] > median_vol) & (
                    (result_df["bullish_mc_success_rate"] > 0.5) | (result_df["bearish_mc_success_rate"] > 0.5)
                )

                result_df.loc[good_sessions, "recommended_for_trading"] = True

        return result_df

    except Exception as e:
        logger.error("analyze_session_performance_failed", error=str(e))
        # Return empty DataFrame with error message
        return pd.DataFrame(
            {
                "session": ["Asian", "London", "New York", "Overlap", "Other"],
                "count": [0, 0, 0, 0, 0],
                "avg_volatility": [0.0, 0.0, 0.0, 0.0, 0.0],
                "error": str(e),
            }
        )


def should_trade_now(
    df: pd.DataFrame | None = None,
    timestamp: pd.Timestamp | datetime | None = None,
    trading_style: str = "day",
    asset_type: str = "forex",
    respect_weekends: bool = True,
) -> dict[str, Any]:
    """
    Determines if current market conditions are favorable for trading.

    Parameters:
    -----------
    df : pandas.DataFrame, optional
        DataFrame with processed market data
    timestamp : pandas.Timestamp or datetime.datetime, optional
        Current timestamp (default: current time)
    trading_style : str, optional
        Trading style: 'scalp', 'day', or 'swing' (default: 'day')
    asset_type : str, optional
        Type of asset ('forex', 'btc', 'gold') for specific rules (default: 'forex')
    respect_weekends : bool, optional
        Whether to consider weekends as non-trading days (default: True, but typically False for crypto)

    Returns:
    --------
    dict
        Dictionary with trading recommendation and reason
    """
    try:
        # For swing trading, timing doesn't matter as much
        if trading_style.lower() == "swing":
            return {
                "should_trade": True,
                "reason": "Swing trading is not time-sensitive",
                "trading_style": trading_style,
            }

        # If no timestamp provided, use current time or latest from dataframe
        if timestamp is None:
            timestamp = datetime.now()

            # Try to get latest timestamp from DataFrame if available
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    latest_ts = df.index.max()
                    try:
                        if latest_ts is not pd.NaT:
                            timestamp = pd.Timestamp(latest_ts)  # type: ignore[arg-type]
                    except Exception:
                        pass
                elif "datetime" in df.columns:
                    latest_ts_val = df["datetime"].max()
                    try:
                        if latest_ts_val is not pd.NaT:
                            _candidate = pd.Timestamp(str(latest_ts_val))
                            if _candidate is not pd.NaT:
                                timestamp = _candidate  # type: ignore[assignment]
                    except Exception:
                        pass

        # Check if current time is in an active session
        session_info = is_active_session(timestamp, asset_type=asset_type)

        # Check if it's a weekend (skip for crypto if not respecting weekends)
        is_weekend = False
        if isinstance(timestamp, (datetime, pd.Timestamp)):
            is_weekend = timestamp.weekday() >= 5  # type: ignore[union-attr]  # 5=Saturday, 6=Sunday

        # For crypto, optionally ignore weekend restrictions
        if (
            is_weekend
            and asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]
            and not respect_weekends
        ):
            is_weekend = False

        if is_weekend and respect_weekends:
            return {
                "should_trade": False,
                "reason": "Weekend - markets closed",
                "session_info": session_info,
                "trading_style": trading_style,
            }

        # Trading decisions based on style
        if trading_style.lower() == "scalp":
            # Scalping requires high volume and liquidity
            if session_info["is_high_volume_session"]:
                return {
                    "should_trade": True,
                    "reason": "High volume session ideal for scalping",
                    "session_info": session_info,
                    "trading_style": trading_style,
                }
            else:
                return {
                    "should_trade": False,
                    "reason": "Not a high volume session for scalping",
                    "session_info": session_info,
                    "trading_style": trading_style,
                }
        elif trading_style.lower() == "day":
            # Day trading works well during active sessions
            if session_info["is_optimal_trading_time"]:
                return {
                    "should_trade": True,
                    "reason": "Optimal session for day trading",
                    "session_info": session_info,
                    "trading_style": trading_style,
                }
            else:
                return {
                    "should_trade": False,
                    "reason": "Not an optimal session for day trading",
                    "session_info": session_info,
                    "trading_style": trading_style,
                }
        else:
            # Default case for other styles
            return {
                "should_trade": True,
                "reason": f"Default trading conditions for {trading_style}",
                "session_info": session_info,
                "trading_style": trading_style,
            }

    except Exception as e:
        logger.error("should_trade_now_failed", error=str(e))
        # Default to conservative response
        return {
            "should_trade": False,
            "reason": f"Error determining trading conditions: {e}",
            "error": True,
            "trading_style": trading_style,
        }


def get_next_session_start(
    timestamp: pd.Timestamp | datetime | None = None,
    target_session: str = "london",
    tz: str = DEFAULT_TIMEZONE,
) -> pd.Timestamp | None:
    """
    Gets the start time of the next session.

    Parameters:
    -----------
    timestamp : pandas.Timestamp or datetime.datetime, optional
        Reference timestamp (default: current time)
    target_session : str, optional
        Target session: 'asian', 'london', 'ny', 'overlap', 'crypto_high', 'crypto_low' (default: 'london')
    tz : str, optional
        Timezone for calculations (default: from config or 'America/Toronto')

    Returns:
    --------
    pandas.Timestamp
        Next session start time
    """
    try:
        # If no timestamp provided, use current time
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Convert to pandas Timestamp if it's a datetime
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)  # type: ignore[arg-type]

        # Make sure it's timezone-aware (pd.Timestamp has .tz; datetime does not)
        ts_pd: pd.Timestamp = timestamp  # type: ignore[assignment]
        if ts_pd.tz is None:
            ts_pd = ts_pd.tz_localize(tz)

        # Convert to target timezone for session calculation
        timestamp_tz = ts_pd.tz_convert(tz)

        # Get session times based on target
        if target_session.startswith("crypto_"):
            if target_session == "crypto_high":
                session_start = CRYPTO_SESSIONS.get("high_volume", {"start": time(14, 0)})["start"]
            elif target_session == "crypto_low":
                session_start = CRYPTO_SESSIONS.get("low_volume", {"start": time(0, 0)})["start"]
            else:
                logger.warning("unknown_crypto_session", target_session=target_session, fallback="default")
                session_start = time(0, 0)
        else:
            # Get from regular sessions
            if target_session not in DEFAULT_SESSIONS:
                logger.warning("unknown_session", target_session=target_session, fallback="london")
                target_session = "london"

            session_start = DEFAULT_SESSIONS[target_session]["start"]

        # Create a datetime for the session start today
        today = timestamp_tz.date()
        session_start_today = datetime.combine(today, session_start, tzinfo=pytz.timezone(tz))

        # If the timestamp is before today's session start, return today's start
        if timestamp_tz < session_start_today:
            return pd.Timestamp(session_start_today)  # type: ignore[return-value]

        # Otherwise, return tomorrow's session start
        tomorrow = today + timedelta(days=1)
        session_start_tomorrow = datetime.combine(tomorrow, session_start, tzinfo=pytz.timezone(tz))

        return pd.Timestamp(session_start_tomorrow)  # type: ignore[return-value]

    except Exception as e:
        logger.error("get_next_session_start_failed", error=str(e))
        # Return current timestamp + 1 day as fallback
        return pd.Timestamp.now() + pd.Timedelta(days=1)  # type: ignore[return-value]


def get_current_session(timestamp: pd.Timestamp | datetime | None = None, asset_type: str = "forex") -> str:
    """
    Gets the name of the current trading session.

    Parameters:
    -----------
    timestamp : pandas.Timestamp or datetime.datetime, optional
        Timestamp to check (default: current time)
    asset_type : str, optional
        Type of asset for specific session handling (default: 'forex')

    Returns:
    --------
    str
        Current session name: 'Asian', 'London', 'New York', 'Overlap', 'Crypto High Volume',
        'Crypto Low Volume', or 'None'
    """
    try:
        # Get session information
        session_info = is_active_session(timestamp, asset_type=asset_type)

        # Check for crypto-specific sessions first if applicable
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            if session_info.get("in_crypto_high_volume", False):
                return "Crypto High Volume"
            elif session_info.get("in_crypto_low_volume", False):
                return "Crypto Low Volume"

        # Check regular sessions with priority
        if session_info["in_overlap_session"]:
            return "Overlap"
        elif session_info["in_london_session"]:
            return "London"
        elif session_info["in_ny_session"]:
            return "New York"
        elif session_info["in_asian_session"]:
            return "Asian"
        else:
            return "None"

    except Exception as e:
        logger.error("get_current_session_failed", error=str(e))
        return "Error"


def add_session_indicators(df: pd.DataFrame, time_column: str = "datetime", asset_type: str = "forex") -> pd.DataFrame:
    """
    Adds session indicator columns to the dataframe with vectorized operations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    time_column : str, optional
        Name of datetime column or index (default: 'datetime')
    asset_type : str, optional
        Type of asset for specific session indicators (default: 'forex')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added session indicator columns
    """
    try:
        if df is None or df.empty:
            return df

        result = df.copy()

        # Get datetime series with proper error handling
        time_series = None

        if time_column in result.columns:
            time_series = pd.to_datetime(result[time_column], errors="coerce")
        elif isinstance(result.index, pd.DatetimeIndex):
            time_series = result.index
        else:
            try:
                time_series = pd.to_datetime(result.index, errors="coerce")
            except Exception as e:
                logger.error("could_not_get_datetime_from_index", error=str(e))
                return result

        if time_series is None or time_series.isna().all():
            logger.error("no_valid_datetime_information_found")
            return result

        # Extract hour - vectorized
        _ts_for_hour = (
            time_series if not isinstance(time_series, pd.DatetimeIndex) else pd.Series(time_series, index=result.index)
        )
        hour_series = pd.Series(_ts_for_hour.dt.hour, index=result.index)

        # Get session times based on asset type
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            # Include crypto-specific sessions
            sessions = DEFAULT_SESSIONS.copy()
            sessions.update(
                {
                    "crypto_high": CRYPTO_SESSIONS.get("high_volume", {"start": time(14, 0), "end": time(22, 0)}),
                    "crypto_low": CRYPTO_SESSIONS.get("low_volume", {"start": time(0, 0), "end": time(6, 0)}),
                }
            )
        else:
            sessions = DEFAULT_SESSIONS

        # Add session indicators - vectorized operations
        for session_name, session_info in sessions.items():
            start_hour = session_info["start"].hour
            end_hour = session_info["end"].hour

            col_name = f"in_{session_name}_session"
            if session_name.startswith("crypto_"):
                col_name = f"in_{session_name}"

            # Handle regular and overnight sessions
            if start_hour < end_hour:
                # Regular session
                result[col_name] = hour_series.between(start_hour, end_hour)
            else:
                # Overnight session
                result[col_name] = ~hour_series.between(end_hour, start_hour, inclusive="both")

        # Add convenience flags for optimal trading times and high volume sessions
        result["is_optimal_trading_time"] = result["in_london_session"] | result["in_ny_session"]
        result["is_high_volume_session"] = result["in_overlap_session"]

        # Add crypto-specific optimizations
        if asset_type.lower() in ["btc", "bitcoin", "crypto", "eth", "ethereum"] and "in_crypto_high" in result.columns:
            result["is_optimal_trading_time"] = result["is_optimal_trading_time"] | result["in_crypto_high"]
            result["is_high_volume_session"] = result["is_high_volume_session"] | result["in_crypto_high"]

        # Add weekday/weekend indicator - vectorized
        result["is_weekend"] = pd.Series(_ts_for_hour.dt.dayofweek, index=result.index) >= 5

        return result

    except Exception as e:
        logger.error("add_session_indicators_failed", error=str(e))
        return df


def analyze_hourly_activity(
    df: pd.DataFrame, time_column: str = "datetime", measure: str = "volatility"
) -> pd.DataFrame:
    """
    Analyzes trading activity by hour of day for optimization.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    time_column : str, optional
        Name of datetime column or index (default: 'datetime')
    measure : str, optional
        What to measure: 'volatility', 'volume', 'range', or 'movement' (default: 'volatility')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with hourly activity statistics
    """
    try:
        if df is None or df.empty:
            logger.warning("empty_dataframe_for_hourly_activity")
            return pd.DataFrame()

        # Get datetime information with error handling
        time_series = None

        if time_column in df.columns:
            time_series = pd.to_datetime(df[time_column], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            time_series = df.index
        else:
            try:
                time_series = pd.to_datetime(df.index, errors="coerce")
            except Exception as e:
                logger.error("could_not_get_datetime_from_index", error=str(e))
                return pd.DataFrame()

        if time_series is None or time_series.isna().all():
            logger.error("no_valid_datetime_information_found")
            return pd.DataFrame()

        # Create a copy with the hour information
        df_copy = df.copy()
        _ts_series = (
            time_series
            if not isinstance(time_series, pd.DatetimeIndex)
            else pd.Series(time_series, index=df_copy.index)
        )
        df_copy["hour"] = _ts_series.dt.hour
        df_copy["day_of_week"] = _ts_series.dt.dayofweek

        # Group by hour
        hourly_stats = {}

        if measure == "volatility" and all(col in df.columns for col in ["high", "low"]):
            # Calculate candle range as percentage of price
            df_copy["volatility"] = (df_copy["high"] - df_copy["low"]) / df_copy["low"] * 100

            # Group by hour and calculate stats
            hourly_stats = df_copy.groupby("hour")["volatility"].agg(["mean", "median", "std", "count"]).reset_index()

        elif measure == "volume" and "volume" in df.columns:
            # Group by hour and calculate volume stats
            hourly_stats = df_copy.groupby("hour")["volume"].agg(["mean", "median", "std", "count"]).reset_index()

        elif measure == "range" and all(col in df.columns for col in ["high", "low"]):
            # Calculate absolute price range
            df_copy["range"] = df_copy["high"] - df_copy["low"]

            # Group by hour and calculate stats
            hourly_stats = df_copy.groupby("hour")["range"].agg(["mean", "median", "std", "count"]).reset_index()

        elif measure == "movement" and all(col in df.columns for col in ["open", "close"]):
            # Calculate directional price movement
            df_copy["movement"] = abs(df_copy["close"] - df_copy["open"])

            # Group by hour and calculate stats
            hourly_stats = df_copy.groupby("hour")["movement"].agg(["mean", "median", "std", "count"]).reset_index()

        else:
            logger.error("cannot_calculate_measure", measure=measure, reason="required_columns_missing")
            return pd.DataFrame()

        # Add session information
        hourly_stats["session"] = "Other"

        # Map hours to sessions
        for hour in range(24):
            if hour >= DEFAULT_SESSIONS["asian"]["start"].hour and hour < DEFAULT_SESSIONS["asian"]["end"].hour:
                hourly_stats.loc[hourly_stats["hour"] == hour, "session"] = "Asian"
            elif hour >= DEFAULT_SESSIONS["london"]["start"].hour and hour < DEFAULT_SESSIONS["london"]["end"].hour:
                hourly_stats.loc[hourly_stats["hour"] == hour, "session"] = "London"
            elif hour >= DEFAULT_SESSIONS["ny"]["start"].hour and hour < DEFAULT_SESSIONS["ny"]["end"].hour:
                hourly_stats.loc[hourly_stats["hour"] == hour, "session"] = "New York"

            # Overlap session
            if hour >= DEFAULT_SESSIONS["overlap"]["start"].hour and hour < DEFAULT_SESSIONS["overlap"]["end"].hour:
                hourly_stats.loc[hourly_stats["hour"] == hour, "session"] = "Overlap"

            # Crypto sessions
            if (
                hour >= CRYPTO_SESSIONS["high_volume"]["start"].hour
                and hour < CRYPTO_SESSIONS["high_volume"]["end"].hour
            ):
                hourly_stats.loc[hourly_stats["hour"] == hour, "crypto_volume"] = "High"
            elif (
                hour >= CRYPTO_SESSIONS["low_volume"]["start"].hour and hour < CRYPTO_SESSIONS["low_volume"]["end"].hour
            ):
                hourly_stats.loc[hourly_stats["hour"] == hour, "crypto_volume"] = "Low"
            else:
                hourly_stats.loc[hourly_stats["hour"] == hour, "crypto_volume"] = "Medium"

        # Calculate rankings
        hourly_stats["rank"] = hourly_stats["mean"].rank(ascending=False)

        # Add recommendation based on ranking
        top_third = hourly_stats["rank"] <= len(hourly_stats) / 3
        hourly_stats["recommendation"] = "Neutral"
        hourly_stats.loc[top_third, "recommendation"] = "Recommended"
        hourly_stats.loc[hourly_stats["rank"] > 2 * len(hourly_stats) / 3, "recommendation"] = "Avoid"

        return hourly_stats

    except Exception as e:
        logger.error("analyze_hourly_activity_failed", error=str(e))
        return pd.DataFrame()


def get_session_calendar(
    start_date: str | datetime,
    end_date: str | datetime,
    sessions: list[str] | None = None,
    tz: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """
    Creates a calendar of trading session times for a date range.

    Parameters:
    -----------
    start_date : str or datetime
        Start date for the calendar
    end_date : str or datetime
        End date for the calendar
    sessions : list of str, optional
        List of sessions to include (default: all sessions)
    tz : str, optional
        Timezone for the calendar (default: from config or 'America/Toronto')

    Returns:
    --------
    pandas.DataFrame
        DataFrame with session calendar
    """
    try:
        # Convert dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Define which sessions to include
        if sessions is None:
            sessions = list(DEFAULT_SESSIONS.keys())
            if "crypto_high" not in sessions and "crypto_low" not in sessions:
                sessions.extend(["crypto_high", "crypto_low"])

        # Create output list
        calendar = []

        # Generate session times for each date
        for date in dates:
            # Skip weekends if not a crypto session
            is_weekend = date.weekday() >= 5

            for session_name in sessions:
                # Handle crypto sessions specially (they run on weekends)
                if session_name.startswith("crypto_"):
                    if session_name == "crypto_high":
                        session_start = CRYPTO_SESSIONS.get("high_volume", {"start": time(14, 0)})["start"]
                        session_end = CRYPTO_SESSIONS.get("high_volume", {"end": time(22, 0)})["end"]
                    elif session_name == "crypto_low":
                        session_start = CRYPTO_SESSIONS.get("low_volume", {"start": time(0, 0)})["start"]
                        session_end = CRYPTO_SESSIONS.get("low_volume", {"end": time(6, 0)})["end"]
                    else:
                        continue  # Skip unknown crypto sessions
                else:
                    # Skip regular sessions on weekends
                    if is_weekend:
                        continue

                    # Get regular session times
                    if session_name not in DEFAULT_SESSIONS:
                        continue

                    session_start = DEFAULT_SESSIONS[session_name]["start"]
                    session_end = DEFAULT_SESSIONS[session_name]["end"]

                # Create session start and end datetimes in the target timezone
                start_datetime = datetime.combine(date.date(), session_start)
                start_datetime = pytz.timezone(tz).localize(start_datetime)

                end_datetime = datetime.combine(date.date(), session_end)
                end_datetime = pytz.timezone(tz).localize(end_datetime)

                # Handle overnight sessions
                if session_start > session_end:
                    end_datetime += timedelta(days=1)

                # Add to calendar
                calendar.append(
                    {
                        "date": date.date(),
                        "session": session_name,
                        "start": start_datetime,
                        "end": end_datetime,
                        "duration_hours": (end_datetime - start_datetime).total_seconds() / 3600,
                    }
                )

        # Convert to DataFrame
        calendar_df = pd.DataFrame(calendar)

        # Add weekday information
        if "date" in calendar_df.columns:
            calendar_df["day_of_week"] = pd.to_datetime(calendar_df["date"]).dt.day_name()

        return calendar_df

    except Exception as e:
        logger.error("get_session_calendar_failed", error=str(e))
        return pd.DataFrame()
