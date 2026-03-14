"""
Ruby Futures System
Core logging module for configuring system-wide logging.

This module provides a simplified interface to our comprehensive logging system,
focused on the initialization and setup of loggers for different components.
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# Import from our comprehensive logging utilities
from .logging_utils import (
    get_logger,
    suppress_library_logs,
)

warnings.warn(
    "lib.utils.setup_logging is deprecated and will be removed in a future version. "
    "Use lib.core.logging_config.get_logger() instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Default libraries to suppress verbose logging from
DEFAULT_SUPPRESSED_LIBS = [
    "matplotlib",
    "urllib3",
    "requests",
    "boto3",
    "botocore",
    "paramiko",
    "sklearn",
    "pandas",
    "numpy",
]


def setup_logging(
    log_file: str | Path | None = None,
    log_level: str = "INFO",
    verbose: bool = False,
    json_format: bool = False,
    service_name: str | None = None,
    suppress_libraries: bool = True,
    log_dir: str | None = None,
    backup_count: int = 5,
    include_timestamp: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    log_environment: bool = False,
    console: bool = True,
    include_system_info: bool = False,
    env_blacklist: list[str] | None = None,
    metrics_enabled: bool = False,
) -> Any:
    """
    Configure logging for the Ruby Futures System.

    This function provides a unified interface to our comprehensive logging system,
    setting up appropriate handlers and formatters based on the provided options.

    Args:
        log_file (str, optional): Path to the log file. If None, a default
                                 log file will be created in the logs directory.
        log_level (str, optional): Logging level. One of "TRACE", "DEBUG", "DATA", "INFO",
                                  "METRICS", "WARNING", "SUCCESS", "ERROR", "CRITICAL".
                                  Defaults to "INFO".
        verbose (bool, optional): Enable verbose logging. When True, console
                                 output will include more detailed information and
                                 the log level will be set to DEBUG.
                                 Defaults to False.
        json_format (bool, optional): Whether to use JSON formatting for logs.
                                     Useful for structured logging systems.
                                     Defaults to False.
        service_name (str, optional): Name of the service for contextual logging.
        suppress_libraries (bool, optional): Whether to suppress logs from common
                                           libraries. Defaults to True.
        log_dir (str, optional): Directory for log files. If specified and log_file
                                is not an absolute path, log_file will be relative
                                to this directory.
        backup_count (int, optional): Number of backup log files to keep. Defaults to 5.
        include_timestamp (bool, optional): Whether to include timestamps in log
                                          filenames. Defaults to True.
        max_file_size (int, optional): Maximum size of each log file before rotation in bytes.
                                      Defaults to 10 MB.
        log_environment (bool, optional): Whether to log environment variables at startup.
                                         Defaults to False.
        console (bool, optional): Whether to output logs to the console.
                                 Defaults to True.
        include_system_info (bool, optional): Whether to log system information at startup.
                                             Defaults to False.
        env_blacklist (List[str], optional): List of environment variable patterns to exclude
                                           when logging environment variables.
        metrics_enabled (bool, optional): Whether to enable metrics logging.
                                        Defaults to False.

    Returns:
        Logger: The configured logger instance

    Raises:
        OSError: If there's an issue with log file creation/access
        ValueError: If an invalid log level is provided
    """
    # Import locally to avoid circular imports
    import platform
    from datetime import datetime
    from pathlib import Path

    from .logging_utils import log_environment as log_env

    try:
        # If verbose, set log level to DEBUG
        if verbose and log_level == "INFO":
            log_level = "DEBUG"

        # Determine log file path
        if log_file is None:
            # Create default log file in the logs directory
            logs_dir = log_dir or Path("logs")
            logs_dir = Path(logs_dir)

            try:
                logs_dir.mkdir(exist_ok=True, parents=True)
            except Exception as e:
                sys.stderr.write(f"Error creating log directory: {e}\n")
                # Fallback to a local logs directory
                logs_dir = Path("./logs")
                logs_dir.mkdir(exist_ok=True, parents=True)

            # Generate filename with timestamp if requested
            base_name = f"fks_trading_{service_name}_" if service_name else "fks_trading_"

            if include_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = logs_dir / f"{base_name}{timestamp}.log"
            else:
                log_file = logs_dir / f"{base_name}log"
        else:
            # If log_file is not absolute and log_dir is specified, make it relative to log_dir
            if log_dir and not os.path.isabs(log_file):
                log_dir_path = Path(log_dir)
                log_dir_path.mkdir(exist_ok=True, parents=True)
                log_file = log_dir_path / log_file

        # Get the logger
        logger = get_logger()

        # Extra fields for structured logging
        extra_fields: dict[str, Any] = {
            "app": "Ruby Futures System",
            "version": os.environ.get("FKS_VERSION", "unknown"),
        }

        # Add service name if provided
        if service_name:
            extra_fields["service"] = service_name

        # If metrics are enabled, add a metrics handler
        if metrics_enabled:
            # Configure metrics collection
            metrics_dir = Path(log_dir or "logs") / "metrics"
            metrics_dir.mkdir(exist_ok=True, parents=True)

            metrics_file = metrics_dir / f"{service_name or 'fks'}_metrics.jsonl"
            extra_fields["metrics_enabled"] = True

            # Add metrics file path to extra_fields for use by metrics logger
            extra_fields["metrics_file"] = str(metrics_file)

        # Configure the logger
        logger.configure(
            log_level=log_level,
            console=console,
            log_file=str(log_file),
            json_format=json_format,
            use_colors=True,
            service_name=service_name,
            max_file_size=max_file_size,
            backup_count=backup_count,
            extra_fields=extra_fields,
        )

        # Suppress library logs if requested
        if suppress_libraries:
            DEFAULT_SUPPRESSED_LIBS = [
                "matplotlib",
                "urllib3",
                "requests",
                "boto3",
                "botocore",
                "paramiko",
                "sklearn",
                "pandas",
                "numpy",
            ]
            suppress_library_logs(DEFAULT_SUPPRESSED_LIBS)

        # Log setup information
        logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")

        if verbose:
            logger.debug("Verbose logging enabled")

            # Log system info in verbose mode or if explicitly requested
            if include_system_info or verbose:
                system_info = {
                    "Python Version": sys.version,
                    "Platform": sys.platform,
                    "OS": platform.platform(),
                    "Executable": sys.executable,
                    "Current Directory": os.getcwd(),
                    "PID": os.getpid(),
                    "CPU Count": os.cpu_count(),
                    "User": os.getlogin() if hasattr(os, "getlogin") else "unknown",
                }
                logger.log_dict(system_info, "System Information")

        # Log environment variables if requested
        if log_environment:
            log_env(logger, sensitive_vars=env_blacklist)

        # Return the configured logger
        return logger

    except Exception as e:
        # Fallback to basic stderr logging if setup fails
        import traceback

        sys.stderr.write(f"Error setting up logging: {e}\n")
        traceback.print_exc(file=sys.stderr)

        # Return a basic logger that writes to stderr
        basic_logger = get_logger()
        return basic_logger


def get_module_logger(module_name: str, **context) -> Any:
    """
    Get a logger configured for a specific module with optional context binding.

    This is a convenience wrapper around get_logger() that adds the module name
    and optional additional context information.

    Args:
        module_name (str): Name of the module requesting the logger
        **context: Additional context to bind to the logger

    Returns:
        Logger: Configured logger instance bound to the module name and context
    """

    logger = get_logger(module_name)
    if context:
        return logger.bind(module=module_name, **context)
    else:
        return logger.bind(module=module_name)


def configure_trading_system_logging(
    system_name: str,
    log_dir: str | None = None,
    log_level: str = "INFO",
    include_json: bool = True,
    include_metrics: bool = True,
    trade_logging: bool = True,
    position_logging: bool = True,
) -> Any:
    """
    Configure specialized logging for a trading system.

    Creates a dedicated logger with appropriate configuration for a specific
    trading system or strategy, including specialized handlers for trade and
    position tracking.

    Args:
        system_name (str): Name of the trading system/strategy
        log_dir (str, optional): Directory for logs
        log_level (str, optional): Logging level
        include_json (bool, optional): Whether to include JSON output
        include_metrics (bool, optional): Whether to enable metrics collection
        trade_logging (bool, optional): Whether to enable detailed trade logging
        position_logging (bool, optional): Whether to enable position tracking logs

    Returns:
        Logger: Configured logger for the trading system
    """

    # Create log directory if needed
    if log_dir is None:
        log_dir_path: Path = Path("logs") / "trading_systems"
    else:
        log_dir_path = Path(log_dir)

    log_dir_path.mkdir(exist_ok=True, parents=True)

    # Create the log file path
    log_file = log_dir_path / f"{system_name}.log"

    # Get a logger with the trading system name
    logger = get_logger(f"trading.{system_name}")

    # Create specialized directories for trade and position logging if enabled
    if trade_logging:
        trade_log_dir = log_dir_path / "trades"
        trade_log_dir.mkdir(exist_ok=True, parents=True)
        trade_log_file = trade_log_dir / f"{system_name}_trades.jsonl"
    else:
        trade_log_file = None

    if position_logging:
        position_log_dir = log_dir_path / "positions"
        position_log_dir.mkdir(exist_ok=True, parents=True)
        position_log_file = position_log_dir / f"{system_name}_positions.jsonl"
    else:
        position_log_file = None

    # Create metrics directory if metrics are enabled
    if include_metrics:
        metrics_dir = log_dir_path / "metrics"
        metrics_dir.mkdir(exist_ok=True, parents=True)
        metrics_file = metrics_dir / f"{system_name}_metrics.jsonl"
    else:
        metrics_file = None

    # Extra fields for structured logging
    extra_fields: dict[str, Any] = {
        "trading_system": system_name,
        "app": "Ruby Futures System",
        "start_time": datetime.now().isoformat(),
    }

    # Add specialized log files to extra fields if enabled
    if trade_logging:
        extra_fields["trade_log_file"] = str(trade_log_file)

    if position_logging:
        extra_fields["position_log_file"] = str(position_log_file)

    if include_metrics:
        extra_fields["metrics_file"] = str(metrics_file)

    # Configure the logger
    logger.configure(
        log_level=log_level,
        console=True,
        log_file=str(log_file),
        json_format=include_json,
        service_name=system_name,
        max_file_size=20 * 1024 * 1024,  # 20 MB for trading systems
        backup_count=10,  # Keep more backups for trading systems
        extra_fields=extra_fields,
    )

    # Bind common context
    logger = logger.bind(system=system_name, start_time=datetime.now().isoformat())  # type: ignore[assignment]

    logger.info(f"Trading system logger configured: {system_name}")

    # Log specialized configuration
    config_info = {
        "System Name": system_name,
        "Log Level": log_level,
        "JSON Format": include_json,
        "Metrics Enabled": include_metrics,
        "Trade Logging": trade_logging,
        "Position Logging": position_logging,
        "Log Directory": str(log_dir_path),
    }

    logger.log_dict(config_info, "Trading System Configuration")  # type: ignore[attr-defined]

    return logger
