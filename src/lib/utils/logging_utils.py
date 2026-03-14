"""
logging_utils.py

A unified logging utility that provides enhanced logging functionality with a clean API.
Features include:
- Structured logging with JSON support
- Console and file logging with rotation
- Execution time tracking
- Context binding
- CLI argument integration
- Custom log levels
"""

import contextlib
import inspect
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from loguru import logger

warnings.warn(
    "lib.utils.logging_utils is deprecated and will be removed in a future version. "
    "Use lib.core.logging_config.get_logger() instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =====================================================================
# Custom Log Levels
# =====================================================================

TRACE_LEVEL = 5
DATA_LEVEL = 15
METRICS_LEVEL = 25
SUCCESS_LEVEL = 35

# Register custom log levels with standard logging
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.addLevelName(DATA_LEVEL, "DATA")
logging.addLevelName(METRICS_LEVEL, "METRICS")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# =====================================================================
# Formatters
# =====================================================================


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs structured log records as JSON strings.
    """

    def __init__(
        self,
        time_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
        extra_fields: dict[str, Any] | None = None,
        include_timestamp: bool = True,
    ):
        """
        Initialize a new JSONFormatter.

        Args:
            time_format: Format for the time field
            extra_fields: Additional fields to include in the JSON output
            include_timestamp: Whether to include the timestamp field
        """
        self.time_format = time_format
        self.extra_fields = extra_fields or {}
        self.include_timestamp = include_timestamp
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_dict: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_dict["timestamp"] = datetime.utcfromtimestamp(record.created).strftime(self.time_format)

        # Add extra fields
        log_dict.update(self.extra_fields)

        # Add extra contextual info
        if hasattr(record, "extra_data"):
            log_dict.update(record.extra_data)  # type: ignore[attr-defined]

        # Add source location
        log_dict["location"] = f"{record.pathname}:{record.lineno}"

        # Add any exception info
        if record.exc_info:
            exc_type = record.exc_info[0]
            log_dict["exception"] = {
                "type": exc_type.__name__ if exc_type is not None else "Unknown",
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_dict)


class ColorFormatter(logging.Formatter):
    """
    Formatter with color support for different log levels.
    """

    # ANSI escape sequences for colors
    COLORS = {
        "TRACE": "\033[38;5;247m",  # Light gray
        "DEBUG": "\033[38;5;39m",  # Blue
        "DATA": "\033[38;5;85m",  # Light teal
        "INFO": "\033[38;5;15m",  # White
        "METRICS": "\033[38;5;220m",  # Gold
        "WARNING": "\033[38;5;214m",  # Orange
        "SUCCESS": "\033[38;5;118m",  # Green
        "ERROR": "\033[38;5;196m",  # Red
        "CRITICAL": "\033[1;38;5;196m",  # Bold red
        "RESET": "\033[0m",  # Reset
    }

    # Format strings for different log levels
    DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ERROR_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s (%(filename)s:%(lineno)d)"

    def __init__(self, use_colors: bool = True, time_format: str = "%Y-%m-%d %H:%M:%S"):
        """
        Initialize a new ColorFormatter.

        Args:
            use_colors: Whether to use colors in the output
            time_format: Format for the time field
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.time_format = time_format
        super().__init__(fmt=self.DEFAULT_FORMAT, datefmt=time_format)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with appropriate colors and format."""
        # Use error format for higher levels
        if record.levelno >= logging.ERROR:
            self._style._fmt = self.ERROR_FORMAT
        else:
            self._style._fmt = self.DEFAULT_FORMAT

        formatted = super().format(record)

        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            return f"{color}{formatted}{self.COLORS['RESET']}"
        else:
            return formatted


# =====================================================================
# Handlers
# =====================================================================


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Extended rotating file handler with automatic directory creation.
    """

    def __init__(self, filename: str | Path, *args, **kwargs):
        """
        Initialize a new RotatingFileHandler.

        Args:
            filename: Log file path
            *args: Additional arguments for the parent class
            **kwargs: Additional keyword arguments for the parent class
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(filename), *args, **kwargs)


# =====================================================================
# Core Logger functionality
# =====================================================================


class Logger:
    """
    Main logger class that provides a unified interface to logging functionality.
    """

    def __init__(self, name: str | None = None):
        """
        Initialize a new Logger instance.

        Args:
            name: Logger name (defaults to the calling module name)
        """
        if name is None:
            # Get the caller's module name
            current_frame = inspect.currentframe()
            frame = current_frame.f_back if current_frame is not None else None
            name = frame.f_globals.get("__name__", "root") if frame is not None else "root"

        self._logger = logger.bind(name=name)
        self._configured = False

    def configure(
        self,
        log_level: int | str = "INFO",
        console: bool = True,
        log_file: str | Path | None = None,
        json_format: bool = False,
        use_colors: bool = True,
        service_name: str | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        Configure the logger with handlers and formatters.

        Args:
            log_level: Logging level (string or int)
            console: Whether to enable console logging
            log_file: Path to log file (None for no file logging)
            json_format: Whether to use JSON formatting
            use_colors: Whether to use colors in console output
            service_name: Optional service name for contextual logging
            max_file_size: Maximum size in bytes before rotating log files
            backup_count: Number of backup log files to keep
            extra_fields: Additional fields to include in JSON logs
        """
        # Convert string log level to int if necessary
        if isinstance(log_level, str):
            log_level = logging.getLevelName(log_level.upper())

        # Reset any existing configuration
        self._reset_config()

        # Set the logger level
        self._logger.setLevel(log_level)  # type: ignore[attr-defined]

        # Create extra fields including service name if provided
        all_extra_fields = extra_fields or {}
        if service_name:
            all_extra_fields["service"] = service_name

        # Configure console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)

            if json_format:
                formatter: logging.Formatter = JSONFormatter(extra_fields=all_extra_fields)
            else:
                formatter = ColorFormatter(use_colors=use_colors)

            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)  # type: ignore[attr-defined]

        # Configure file handler
        if log_file:
            try:
                # Ensure we have a Path object
                if isinstance(log_file, str):
                    log_file = Path(log_file)

                # Create parent directory if it doesn't exist
                log_file.parent.mkdir(parents=True, exist_ok=True)

                # Create the file handler
                file_handler = RotatingFileHandler(filename=log_file, maxBytes=max_file_size, backupCount=backup_count)
                file_handler.setLevel(log_level)

                if json_format:
                    formatter = JSONFormatter(extra_fields=all_extra_fields)
                else:
                    formatter = logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
                    )

                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)  # type: ignore[attr-defined]

                self._logger.info(f"File logging configured at: {log_file}")
            except Exception as e:
                self._logger.error(f"Failed to configure file logging: {e}", exc_info=True)

        self._configured = True
        self._logger.info("Logger configured successfully" + (f" for service: {service_name}" if service_name else ""))

    def _reset_config(self) -> None:
        """Remove any existing handlers from the logger."""
        for handler in self._logger.handlers[:]:  # type: ignore[attr-defined]
            self._logger.removeHandler(handler)  # type: ignore[attr-defined]

    def bind(self, **context) -> "ContextLogger":
        """
        Create a logger with bound context information.

        Args:
            **context: Keyword arguments to add as context

        Returns:
            A ContextLogger with the specified context
        """
        return ContextLogger(self._logger, context)

    def trace(self, msg, *args, **kwargs):
        """Log at TRACE level."""
        return self._logger.log(TRACE_LEVEL, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log at DEBUG level."""
        return self._logger.debug(msg, *args, **kwargs)

    def data(self, msg, *args, **kwargs):
        """Log at DATA level."""
        return self._logger.log(DATA_LEVEL, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log at INFO level."""
        return self._logger.info(msg, *args, **kwargs)

    def metrics(self, msg, *args, **kwargs):
        """Log at METRICS level."""
        return self._logger.log(METRICS_LEVEL, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log at WARNING level."""
        return self._logger.warning(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        """Log at SUCCESS level."""
        return self._logger.log(SUCCESS_LEVEL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log at ERROR level."""
        return self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log at CRITICAL level."""
        return self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """Log exception with traceback."""
        return self._logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def log_dict(self, data: dict[str, Any], title: str = "Data") -> None:
        """
        Log a dictionary in a formatted table-like structure.

        Args:
            data: Dictionary to log
            title: Title for the table
        """
        if not isinstance(data, dict):
            self.warning("Cannot log as table: data is not a dictionary")
            return

        # Find the longest key for formatting
        max_key_length = max((len(str(key)) for key in data), default=10)

        # Create a horizontal rule
        hr = "=" * (max_key_length + 25)

        lines = [f"\n📌 {title}", hr]

        # Add each key-value pair
        for key, value in sorted(data.items()):
            # Format the value for display
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value, indent=2)
                except Exception:
                    value = str(value)

            lines.append(f"{str(key):<{max_key_length}} : {value}")

        # Log the entire table as a single message
        self.info("\n".join(lines))

    def log_error(self, exception: Exception, context: dict[str, Any] | None = None) -> None:
        """
        Log detailed error information.

        Args:
            exception: The exception to log
            context: Optional context information
        """
        error_type = type(exception).__name__
        error_msg = str(exception)

        # Start with basic error message
        self.error(f"Exception: {error_type}: {error_msg}")

        if context is None:
            context = {}

        # Log context information if provided
        if context:
            self.error("Error context:")
            for key, value in context.items():
                self.error(f"  {key}: {value}")

        # Log the full traceback
        self.error(f"Traceback:\n{traceback.format_exc()}")

    def timed(self, operation_name: str, level: str = "INFO"):
        """
        Create a context manager for timing operations.

        Args:
            operation_name: Name of the operation being timed
            level: Log level to use for timing messages

        Returns:
            A TimedContext context manager
        """
        return TimedContext(self, operation_name, level)

    def set_level(self, level: int | str) -> None:
        """
        Set the logger's level.

        Args:
            level: New log level (string or int)
        """
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())

        self._logger.setLevel(level)  # type: ignore[attr-defined]


# =====================================================================
# Context Logger
# =====================================================================


class ContextLogger:
    """
    Logger that includes contextual information with each log message.
    """

    def __init__(self, logger: Any, context: dict[str, Any]):
        """
        Initialize a new ContextLogger.

        Args:
            logger: Base logger to use
            context: Context to include with log messages
        """
        self._logger = logger
        self._context = context

    def _process_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Add context information to the keyword arguments.

        Args:
            kwargs: Existing keyword arguments

        Returns:
            Updated keyword arguments
        """
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if "extra_data" not in kwargs["extra"]:
            kwargs["extra"]["extra_data"] = {}

        kwargs["extra"]["extra_data"].update(self._context)
        return kwargs

    def trace(self, msg, *args, **kwargs):
        """Log at TRACE level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.log(TRACE_LEVEL, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log at DEBUG level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.debug(msg, *args, **kwargs)

    def data(self, msg, *args, **kwargs):
        """Log at DATA level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.log(DATA_LEVEL, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log at INFO level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.info(msg, *args, **kwargs)

    def metrics(self, msg, *args, **kwargs):
        """Log at METRICS level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.log(METRICS_LEVEL, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log at WARNING level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.warning(msg, *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        """Log at SUCCESS level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.log(SUCCESS_LEVEL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log at ERROR level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log at CRITICAL level with context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """Log exception with traceback and context."""
        kwargs = self._process_kwargs(kwargs)
        return self._logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def bind(self, **additional_context) -> "ContextLogger":
        """
        Create a new context logger with additional context.

        Args:
            **additional_context: Additional context to include

        Returns:
            A new ContextLogger with combined context
        """
        combined_context = self._context.copy()
        combined_context.update(additional_context)
        return ContextLogger(self._logger, combined_context)


# =====================================================================
# Timing Context
# =====================================================================


class TimedContext:
    """
    Context manager for timing operations and logging the duration.
    """

    def __init__(self, logger: Logger, operation_name: str, level: str = "INFO"):
        """
        Initialize a new TimedContext.

        Args:
            logger: Logger to use
            operation_name: Name of the operation being timed
            level: Logging level for timing messages
        """
        self.logger = logger
        self.operation_name = operation_name
        self.level = logging.getLevelName(level.upper()) if isinstance(level, str) else level

    def __enter__(self) -> "TimedContext":
        """
        Enter the context manager.

        Returns:
            Self
        """
        self.start_time = time.time()
        self.logger._logger.log(self.level, f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger._logger.log(self.level, f"Completed {self.operation_name} in {duration:.4f} seconds")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.4f} seconds: {exc_val}")


# =====================================================================
# Decorators
# =====================================================================


def log_execution_time(logger=None, level: str = "INFO"):
    """
    Decorator to log the execution time of a function.

    Args:
        logger: Logger to use (if None, a new one will be created)
        level: Logging level for timing messages

    Returns:
        Decorator function

    Example:
        @log_execution_time(level="DEBUG")
        def my_function():
            # Function code
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create or use the provided logger
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            log_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level

            func_name = func.__name__
            module_name = func.__module__

            # Log start
            logger._logger.log(log_level, f"Starting {module_name}.{func_name}")

            # Measure execution time
            start_time = time.time()

            # Execute function
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger._logger.log(log_level, f"Completed {module_name}.{func_name} in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Exception in {module_name}.{func_name} after {execution_time:.4f} seconds: {type(e).__name__}: {e}"
                )
                raise

        return wrapper

    return decorator


def log_execution(logger=None):
    """
    Decorator to log function execution details including parameters and return values.

    Args:
        logger: Logger to use (if None, a new one will be created)

    Returns:
        Decorator function

    Example:
        @log_execution()
        def my_function(a, b):
            return a + b
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create or use the provided logger
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            # Get function details
            func_name = func.__name__
            module_name = func.__module__

            # Format arguments, handling 'self' for methods
            formatted_args = args
            if args and hasattr(args[0], func_name):
                # This is likely a method, skip logging 'self'
                formatted_args = args[1:]

            # Get source file and line number
            try:
                source_file = inspect.getsourcefile(func)
                source_line = inspect.getsourcelines(func)[1]
                location = f"{source_file}:{source_line}"
            except (TypeError, OSError):
                location = "unknown location"

            # Log entry with parameters
            logger.info(f"Entering {module_name}.{func_name} at {location}")
            logger.debug(f"Arguments: args={formatted_args}, kwargs={kwargs}")

            # Measure execution time
            start_time = time.time()

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Log successful completion
                logger.info(f"Exiting {module_name}.{func_name}; execution time: {execution_time:.4f} sec")

                # Log return value (only for debug level)
                try:
                    # Try to get a reasonable string representation of the result
                    result_str = str(result)
                    # Truncate if too long
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "... [truncated]"
                    logger.debug(f"Return value: {result_str}")
                except Exception as e:
                    logger.debug(f"Return value: <not loggable: {e}>")

                return result

            except Exception as e:
                # Calculate execution time until exception
                execution_time = time.time() - start_time

                # Log exception details
                logger.error(
                    f"Exception in {module_name}.{func_name} after {execution_time:.4f} sec: {type(e).__name__}: {e}"
                )

                # Re-raise the exception
                raise

        return wrapper

    return decorator


# =====================================================================
# Utility Functions
# =====================================================================


def get_logger(name: str | None = None) -> Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (defaults to the calling module name)

    Returns:
        A Logger instance
    """
    return Logger(name)


def configure_from_args(args, log_path: str | Path | None = None) -> Logger:
    """
    Configure a logger based on command-line arguments.

    Args:
        args: Parsed command-line arguments object
        log_path: Optional path for log files

    Returns:
        Configured Logger instance
    """
    logger = get_logger()

    # Configure based on command-line arguments
    logger.configure(
        log_level="INFO",  # Default level
        service_name=getattr(args, "service_name", None),
    )

    # If verbose mode is enabled, log the full argument namespace
    if getattr(args, "verbose", False):
        args_dict = {arg: getattr(args, arg) for arg in dir(args) if not arg.startswith("_")}
        logger.log_dict(args_dict, "Command-line arguments")

    return logger


def suppress_library_logs(library_names: list[str], log_level: str = "WARNING") -> None:
    """
    Suppress logs from specified libraries.

    Args:
        library_names: List of library names to suppress
        log_level: Level to set the libraries to
    """
    for library in library_names:
        with contextlib.suppress(AttributeError, TypeError):
            logging.getLogger(library).setLevel(getattr(logging, log_level.upper()))


def log_environment(
    logger: Logger | None = None, sensitive_vars: list[str] | None = None, exclude_empty: bool = False
) -> None:
    """
    Log all environment variables, excluding sensitive ones.

    Args:
        logger: Logger to use (if None, a new one will be created)
        sensitive_vars: List of patterns to identify sensitive variables
        exclude_empty: Whether to exclude environment variables with empty values
    """
    if logger is None:
        logger = get_logger()

    default_sensitive = ["PASSWORD", "SECRET", "TOKEN", "KEY", "CREDENTIAL", "AUTH"]
    sensitive_vars = sensitive_vars or default_sensitive

    logger.info("Logging environment variables...")

    env_vars = {}
    for key, value in sorted(os.environ.items()):
        if exclude_empty and not value:
            continue

        # Skip logging sensitive information
        if any(sensitive_key.upper() in key.upper() for sensitive_key in sensitive_vars):
            env_vars[key] = "********"
        else:
            env_vars[key] = value

    logger.log_dict(env_vars, "Environment Variables")


# =====================================================================
# Initialize default logger for simple usage
# =====================================================================

default_logger = get_logger("root")
