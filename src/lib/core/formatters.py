"""
Data Formatting Utilities

This module provides utility functions for formatting various data types in a
consistent manner for display in user interfaces.
"""

import json
from datetime import datetime, timedelta
from typing import Any

# -------------------------------------------------------------------------
# Constants and Helper Functions
# -------------------------------------------------------------------------

# Default values for None/invalid inputs
NA_PLACEHOLDER = "N/A"
EMPTY_DATETIME_PLACEHOLDER = "--"

# Currency symbols lookup
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "BTC": "₿"}

# Asset-specific price precision
ASSET_PRECISION = {
    "BTC": 2,  # $48,250.25
    "ETH": 2,  # $2,830.45
    "ADA": 4,  # $1.2530
    "XRP": 4,  # $0.5678
    "DOGE": 6,  # $0.123456
}


def _ensure_datetime(dt_value: datetime | str | float | int | None) -> datetime | None:
    """
    Convert various datetime representations to a datetime object.

    Args:
        dt_value: Value to convert (datetime, timestamp, or ISO string)

    Returns:
        datetime object or None if conversion failed
    """
    if dt_value is None:
        return None

    if isinstance(dt_value, datetime):
        return dt_value

    try:
        if isinstance(dt_value, (int, float)):
            return datetime.fromtimestamp(dt_value)
        else:
            # Try to parse string as ISO format
            return datetime.fromisoformat(str(dt_value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _parse_duration_string(duration_str: str) -> int:
    """
    Parse a duration string like "1d 2h 3m 4s" into seconds.

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds
    """
    parts = duration_str.split()
    total_seconds = 0

    for part in parts:
        if part.endswith("d"):
            total_seconds += int(part[:-1]) * 86400
        elif part.endswith("h"):
            total_seconds += int(part[:-1]) * 3600
        elif part.endswith("m"):
            total_seconds += int(part[:-1]) * 60
        elif part.endswith("s"):
            total_seconds += int(part[:-1])

    return total_seconds


# -------------------------------------------------------------------------
# Currency and Number Formatting
# -------------------------------------------------------------------------


def format_currency(
    value: float | int | str | None, currency: str = "USD", precision: int = 2, include_symbol: bool = True
) -> str:
    """
    Format a value as currency with appropriate symbol and precision.

    Args:
        value: Value to format
        currency: Currency code (USD, EUR, GBP, etc.)
        precision: Decimal precision
        include_symbol: Whether to include currency symbol

    Returns:
        Formatted currency string

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.56, currency="EUR", include_symbol=False)
        '1,234.56'
    """
    if value is None:
        return NA_PLACEHOLDER

    try:
        # Convert to float
        value_float = float(value)

        # Format with appropriate precision
        formatted = f"{value_float:,.{precision}f}"

        # Add currency symbol if requested
        if include_symbol:
            symbol = CURRENCY_SYMBOLS.get(currency, currency)
            return f"{symbol}{formatted}"
        else:
            return formatted
    except (ValueError, TypeError):
        return str(value)


def format_price(
    price: float | int | str | None, asset: str | None = None, precision: int | None = None, include_symbol: bool = True
) -> str:
    """
    Format a price value with asset-appropriate precision.

    Args:
        price: Price value to format
        asset: Asset symbol (e.g., "BTC", "ETH") to determine precision
        precision: Override automatic precision
        include_symbol: Whether to include the currency symbol

    Returns:
        Formatted price string

    Examples:
        >>> format_price(48250.253, asset="BTC")
        '$48,250.25'
        >>> format_price(0.000456, asset="DOGE")
        '$0.000456'
    """
    if price is None:
        return NA_PLACEHOLDER

    try:
        # Convert to float
        price_float = float(price)

        # Determine precision if not explicitly provided
        if precision is None:
            if asset:
                # Use asset-specific precision
                precision = ASSET_PRECISION.get(asset.upper(), 2)
            else:
                # Default precision based on price magnitude
                if price_float >= 1000:
                    precision = 2
                elif price_float >= 1:
                    precision = 4
                else:
                    precision = 6

        # Format with appropriate precision
        formatted = f"{price_float:,.{precision}f}"

        # Add currency symbol if requested
        if include_symbol:
            return f"${formatted}"
        else:
            return formatted
    except (ValueError, TypeError):
        return str(price)


def format_number(value: float | int | str | None, precision: int = 2, thousands_separator: bool = True) -> str:
    """
    Format a number with given precision and optional thousands separator.

    Args:
        value: Number to format
        precision: Decimal precision
        thousands_separator: Whether to include thousands separator

    Returns:
        Formatted number string

    Examples:
        >>> format_number(1234.56789)
        '1,234.57'
        >>> format_number(1234.56789, precision=4, thousands_separator=False)
        '1234.5679'
    """
    if value is None:
        return NA_PLACEHOLDER

    try:
        # Convert to float
        value_float = float(value)

        # Format with appropriate precision
        if thousands_separator:
            return f"{value_float:,.{precision}f}"
        else:
            return f"{value_float:.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def format_large_number(number: int | float | None, precision: int = 2) -> str:
    """
    Format large numbers with K, M, B, T suffixes.

    Args:
        number: Number to format
        precision: Decimal precision

    Returns:
        Formatted number string with suffix

    Examples:
        >>> format_large_number(1234)
        '1.23K'
        >>> format_large_number(1234567)
        '1.23M'
    """
    if number is None:
        return NA_PLACEHOLDER

    try:
        number_float = float(number)

        if abs(number_float) < 1000:
            return format_number(number_float, precision, False)

        for suffix in ["", "K", "M", "B", "T"]:
            if abs(number_float) < 1000:
                return f"{number_float:.{precision}f}{suffix}"
            number_float /= 1000

        return f"{number_float:.{precision}f}T"  # for numbers >= 10^15
    except (ValueError, TypeError):
        return str(number)


def format_percentage(value: float | int | str | None, precision: int = 2, include_sign: bool = False) -> str:
    """
    Format a value as a percentage.

    Args:
        value: Value to format as percentage
        precision: Decimal precision
        include_sign: Whether to include +/- sign

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(0.1234)
        '12.34%'
        >>> format_percentage(0.1234, include_sign=True)
        '+12.34%'
    """
    if value is None:
        return NA_PLACEHOLDER

    try:
        # Convert to float
        value_float = float(value)

        # If input is a decimal (0.xx), multiply by 100
        if abs(value_float) < 1 and not str(value_float).startswith("0."):
            value_float *= 100

        # Format with given precision
        formatted = f"{abs(value_float):.{precision}f}%"

        # Add sign if requested
        if include_sign:
            if value_float > 0:
                return f"+{formatted}"
            elif value_float < 0:
                return f"-{formatted}"
            else:
                return formatted
        else:
            return formatted
    except (ValueError, TypeError):
        return str(value)


def format_pnl(
    pnl: float | int | str | None, percentage: bool = False, precision: int = 2, include_sign: bool = True
) -> str:
    """
    Format profit and loss values with appropriate coloring and sign.

    Args:
        pnl: Profit/loss value
        percentage: Whether value is a percentage
        precision: Decimal precision
        include_sign: Whether to include +/- sign

    Returns:
        Formatted P&L string

    Examples:
        >>> format_pnl(123.45)
        '+$123.45'
        >>> format_pnl(-123.45, percentage=True)
        '-123.45%'
    """
    if pnl is None:
        return NA_PLACEHOLDER

    try:
        # Convert to float
        pnl_float = float(pnl)

        # Format with given precision
        formatted = f"{abs(pnl_float):.{precision}f}"

        # Add sign if requested
        if include_sign:
            sign = "+" if pnl_float > 0 else "-" if pnl_float < 0 else ""
            formatted = f"{sign}{formatted}"

        # Add percentage symbol or currency symbol
        formatted = f"{formatted}%" if percentage else f"${formatted}"

        return formatted
    except (ValueError, TypeError):
        return str(pnl)


# -------------------------------------------------------------------------
# Date, Time and Duration Formatting
# -------------------------------------------------------------------------


def format_date(date: str | datetime | float | int | None, format_str: str = "%Y-%m-%d") -> str:
    """
    Format a date with the specified format.

    Args:
        date: Date to format (datetime, timestamp or string)
        format_str: Format string for output

    Returns:
        Formatted date string

    Examples:
        >>> format_date("2023-05-15T12:30:45Z")
        '2023-05-15'
        >>> format_date(datetime(2023, 5, 15, 12, 30, 45), format_str="%d/%m/%Y")
        '15/05/2023'
    """
    dt = _ensure_datetime(date)
    if dt is None:
        return EMPTY_DATETIME_PLACEHOLDER

    return dt.strftime(format_str)


def format_time(time: str | datetime | float | int | None, format_str: str = "%H:%M:%S") -> str:
    """
    Format a time with the specified format.

    Args:
        time: Time to format (datetime, timestamp or string)
        format_str: Format string for output

    Returns:
        Formatted time string

    Examples:
        >>> format_time("2023-05-15T12:30:45Z")
        '12:30:45'
        >>> format_time(datetime(2023, 5, 15, 12, 30, 45), format_str="%I:%M %p")
        '12:30 PM'
    """
    dt = _ensure_datetime(time)
    if dt is None:
        return EMPTY_DATETIME_PLACEHOLDER

    return dt.strftime(format_str)


def format_datetime(dt: datetime | str | float | int | None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime with the specified format.

    Args:
        dt: Datetime to format (datetime, timestamp or string)
        format_str: Format string for output

    Returns:
        Formatted datetime string

    Examples:
        >>> format_datetime("2023-05-15T12:30:45Z")
        '2023-05-15 12:30:45'
        >>> format_datetime(1621084245)  # Unix timestamp
        '2021-05-15 12:30:45'
    """
    dt_obj = _ensure_datetime(dt)
    if dt_obj is None:
        return EMPTY_DATETIME_PLACEHOLDER

    return dt_obj.strftime(format_str)


def format_duration(duration: str | timedelta | int | float | None, include_seconds: bool = True) -> str:
    """
    Format a duration into a human-readable string (e.g., "1d 2h 3m 4s").

    Args:
        duration: Duration to format (timedelta, seconds, or formatted string)
        include_seconds: Whether to include seconds in output

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(3665)  # seconds
        '1h 1m 5s'
        >>> format_duration(timedelta(days=1, hours=2, minutes=3, seconds=4))
        '1d 2h 3m 4s'
    """
    if duration is None:
        return NA_PLACEHOLDER

    total_seconds: float = 0

    try:
        # Convert various input types to seconds
        if isinstance(duration, timedelta):
            total_seconds = duration.total_seconds()
        elif isinstance(duration, (int, float)):
            total_seconds = float(duration)
        elif isinstance(duration, str):
            total_seconds = _parse_duration_string(duration)
        else:
            return str(duration)

        # Calculate components
        days, remainder = divmod(int(total_seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Build output string
        parts = []

        if days > 0:
            parts.append(f"{days}d")

        if hours > 0:
            parts.append(f"{hours}h")

        if minutes > 0:
            parts.append(f"{minutes}m")

        if include_seconds and seconds > 0:
            parts.append(f"{seconds}s")

        # If everything is zero and we want seconds
        if not parts and include_seconds:
            return "0s"

        # If everything is zero and we don't want seconds
        if not parts:
            return "0m"

        return " ".join(parts)
    except (ValueError, TypeError, AttributeError):
        return str(duration)


# -------------------------------------------------------------------------
# Specialized Formatting
# -------------------------------------------------------------------------


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as pretty-printed JSON.

    Args:
        data: Data to format as JSON
        indent: JSON indentation level

    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent)
    except (TypeError, ValueError):
        return str(data)
