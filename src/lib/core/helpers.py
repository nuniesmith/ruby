#!/usr/bin/env python3
"""
Ruby Futures System Dashboard - Utility Helpers
General purpose utility functions used throughout the application
"""

import functools
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries

    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (overrides dict1)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def timing_decorator(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to measure and log execution time of functions

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug("Function executed", function=func.__name__, elapsed_seconds=round(end_time - start_time, 4))
        return result

    return wrapper


def safe_json_serialize(obj: Any) -> dict[str, Any]:
    """
    Safely serialize objects to JSON, handling non-serializable objects

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable dictionary
    """

    def default_handler(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()
        # For more complex objects, return a string representation
        return str(o)

    try:
        return json.loads(json.dumps(obj, default=default_handler))
    except (TypeError, ValueError) as e:
        logger.warning("Failed to fully serialize object to JSON", error=str(e))
        return {"error": "Object could not be fully serialized", "type": str(type(obj))}


def error_handler(default_value: T | None = None) -> Callable[[Callable[..., R]], Callable[..., R | T | None]]:
    """
    Decorator to handle exceptions and return a default value on error

    Args:
        default_value: Value to return if an exception occurs

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R | T | None]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> R | T | None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error("Error in function", function=func.__name__, error=str(e), exc_info=True)
                return default_value

        return wrapper

    return decorator


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def parse_timestamp(timestamp: str, formats: list[str] | None = None) -> datetime | None:
    """
    Try to parse a timestamp string in multiple formats

    Args:
        timestamp: Timestamp string
        formats: List of format strings to try (defaults to common formats)

    Returns:
        Parsed datetime or None if parsing failed
    """
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format
            "%Y-%m-%d %H:%M:%S",  # Standard format
            "%Y-%m-%d",  # Date only
            "%d/%m/%Y %H:%M:%S",  # European format
            "%m/%d/%Y %H:%M:%S",  # US format
        ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue

    logger.warning("Failed to parse timestamp", timestamp=timestamp)
    return None


def batch_process(items: list[T], batch_size: int, process_func: Callable[[list[T]], Any]) -> list[Any]:
    """
    Process a list of items in batches

    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to process each batch

    Returns:
        List of results from processing each batch
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        result = process_func(batch)
        results.append(result)
    return results


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (how much to increase delay each time)
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_attempts, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Retrying after error", error=str(e), retry_in_seconds=mdelay)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator
