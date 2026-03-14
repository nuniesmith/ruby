import re
import time
from datetime import UTC, datetime, timedelta

import pytz  # type: ignore[import-untyped]


def now() -> datetime:
    """
    Get the current datetime in America/Toronto.

    Returns:
        Current America/Toronto datetime with timezone info
    """
    return datetime.now(UTC)


def timestamp_ms() -> int:
    """
    Get the current timestamp in milliseconds.

    Returns:
        Current timestamp as milliseconds since epoch
    """
    return int(time.time() * 1000)


def timestamp_ns() -> int:
    """
    Get the current timestamp in nanoseconds.

    Returns:
        Current timestamp as nanoseconds since epoch
    """
    return time.time_ns()


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert a datetime to Unix timestamp (seconds).

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp in seconds
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def datetime_to_timestamp_ms(dt: datetime) -> int:
    """
    Convert a datetime to Unix timestamp (milliseconds).

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp in milliseconds
    """
    return datetime_to_timestamp(dt) * 1000


def timestamp_to_datetime(timestamp: int | float) -> datetime:
    """
    Convert a Unix timestamp to datetime.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        Datetime in America/Toronto
    """
    return datetime.fromtimestamp(timestamp, tz=UTC)


def timestamp_ms_to_datetime(timestamp_ms: int) -> datetime:
    """
    Convert a Unix timestamp in milliseconds to datetime.

    Args:
        timestamp_ms: Unix timestamp in milliseconds

    Returns:
        Datetime in America/Toronto
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)


def parse_datetime(dt_str: str, formats: list[str] | None = None) -> datetime:
    """
    Parse a datetime string using multiple possible formats.

    Args:
        dt_str: Datetime string to parse
        formats: List of datetime formats to try

    Returns:
        Parsed datetime with America/Toronto timezone

    Raises:
        ValueError: If the string cannot be parsed with any of the provided formats
    """
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format without microseconds
            "%Y-%m-%d %H:%M:%S.%f",  # PostgreSQL format with microseconds
            "%Y-%m-%d %H:%M:%S",  # PostgreSQL format without microseconds
            "%Y-%m-%d",  # Date only
        ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue

    # Try to parse ISO format with timezone offset
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt
    except ValueError:
        pass

    raise ValueError(f"Could not parse datetime string: {dt_str}")


def datetime_to_str(dt: datetime, fmt: str = "%Y-%m-%dT%H:%M:%S.%fZ") -> str:
    """
    Convert a datetime to a formatted string.

    Args:
        dt: Datetime to convert
        fmt: Format string

    Returns:
        Formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.strftime(fmt)


def datetime_to_iso(dt: datetime) -> str:
    """
    Convert a datetime to ISO 8601 format.

    Args:
        dt: Datetime to convert

    Returns:
        ISO 8601 formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat().replace("+00:00", "Z")


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse a duration string into a timedelta.

    Supported formats:
    - Ns: N seconds
    - Nm: N minutes
    - Nh: N hours
    - Nd: N days
    - Nw: N weeks

    Args:
        duration_str: Duration string

    Returns:
        Parsed timedelta

    Raises:
        ValueError: If the string cannot be parsed
    """
    pattern = re.compile(r"^(\d+)([smhdw])$")
    match = pattern.match(duration_str)

    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")

    value, unit = int(match.group(1)), match.group(2)

    if unit == "s":
        return timedelta(seconds=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Invalid duration unit: {unit}")


def format_duration(td: timedelta) -> str:
    """
    Format a timedelta as a human-readable duration string.

    Args:
        td: Timedelta to format

    Returns:
        Formatted duration string
    """
    seconds = td.total_seconds()

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h"
    elif seconds < 604800:
        return f"{int(seconds / 86400)}d"
    else:
        return f"{int(seconds / 604800)}w"


def get_time_range(start: str | datetime, end: str | datetime | None = None, interval: str | None = None) -> tuple:
    """
    Get a time range between start and end or start and start + interval.

    Args:
        start: Start datetime or datetime string
        end: End datetime or datetime string
        interval: Interval as a duration string (e.g. '1d', '4h')

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        ValueError: If the parameters are invalid
    """
    # Parse start
    start_dt = parse_datetime(start) if isinstance(start, str) else start

    # Parse end
    if end is not None:
        end_dt = parse_datetime(end) if isinstance(end, str) else end
    elif interval is not None:
        end_dt = start_dt + parse_duration(interval)
    else:
        end_dt = now()

    return start_dt, end_dt


def localize_datetime(dt: datetime, tz_name: str) -> datetime:
    """
    Localize a datetime to a specific timezone.

    Args:
        dt: Datetime to localize
        tz_name: Timezone name

    Returns:
        Localized datetime
    """
    tz = pytz.timezone(tz_name)

    if dt.tzinfo is None:
        # Naive datetime, assume America/Toronto
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(tz)


def truncate_datetime(dt: datetime, unit: str) -> datetime:
    """
    Truncate a datetime to a specific unit.

    Args:
        dt: Datetime to truncate
        unit: Unit to truncate to ('day', 'hour', 'minute', 'second')

    Returns:
        Truncated datetime
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    if unit == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif unit == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    elif unit == "minute":
        return dt.replace(second=0, microsecond=0)
    elif unit == "second":
        return dt.replace(microsecond=0)
    else:
        raise ValueError(f"Invalid truncation unit: {unit}")


def date_range(start: str | datetime, end: str | datetime, freq: str = "1d") -> list[datetime]:
    """
    Generate a list of datetimes between start and end with a given frequency.

    Args:
        start: Start datetime or datetime string
        end: End datetime or datetime string
        freq: Frequency as a duration string (e.g. '1d', '4h')

    Returns:
        List of datetimes
    """
    start_dt, end_dt = get_time_range(start, end)

    step = parse_duration(freq)
    result = []

    current = start_dt
    while current <= end_dt:
        result.append(current)
        current += step

    return result
