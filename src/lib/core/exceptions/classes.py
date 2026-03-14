"""
Core exception classes for structured error handling.

This module provides a unified exception hierarchy for the application,
with consistent error handling, logging, and serialization capabilities.
"""

import json
import traceback
from datetime import datetime
from enum import Enum, auto
from typing import Any, TypeVar

from prometheus_client import CollectorRegistry, Counter

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseException")

# Custom Prometheus registry to avoid global duplication
ERROR_REGISTRY = CollectorRegistry()


class ErrorSeverity(Enum):
    """Enumeration for error severity levels."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


def _increment_error_counter(exception_name: str):
    """
    Increment Prometheus error counter for a given exception.

    Args:
        exception_name (str): Name of the exception class
    """
    try:
        counter = Counter(
            "application_exceptions_total",
            "Total number of exceptions by type",
            ["exception_type"],
            registry=ERROR_REGISTRY,
        )
        counter.labels(exception_type=exception_name).inc()
    except Exception as e:
        logger.error("error_incrementing_counter", error=str(e), exception_name=exception_name)


class BaseException(Exception):
    """
    Comprehensive base exception class for all application exceptions.

    Provides:
    - Rich error context
    - Automatic logging
    - Prometheus metrics tracking
    - JSON serialization
    - Detailed error tracking
    """

    # Default class-level attributes
    DEFAULT_MESSAGE = "An unspecified error occurred"
    DEFAULT_CODE = 1000
    DEFAULT_HTTP_STATUS = 500
    DEFAULT_SEVERITY = ErrorSeverity.ERROR

    def __init__(
        self,
        message: str | None = None,
        code: int | None = None,
        code_str: str | None = None,
        http_status: int | None = None,
        severity: ErrorSeverity | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the base exception with comprehensive error tracking.

        Args:
            message: Human-readable error message
            code: Numeric error code
            code_str: String error code for API responses
            http_status: HTTP status code for REST responses
            severity: Error severity level
            retryable: Whether the error can be retried
            details: Additional error context
            **kwargs: Additional keyword arguments to include in details
        """
        # Set default values
        self.timestamp = datetime.utcnow()
        self.message = message or self.DEFAULT_MESSAGE
        self.code = code or self.DEFAULT_CODE
        self.code_str = code_str or f"error_{self.code}"
        self.http_status = http_status or self.DEFAULT_HTTP_STATUS
        self.severity = severity or self.DEFAULT_SEVERITY
        self.retryable = retryable

        # Prepare details
        self.details = details or {}
        self.details.update(kwargs)

        # Add traceback information
        self.details["traceback"] = traceback.format_exc()

        # Initialize base Exception
        super().__init__(self.message)

        # Log the exception
        self._log_exception()

        # Increment Prometheus error counter
        _increment_error_counter(self.__class__.__name__)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the exception to a comprehensive dictionary.

        Returns:
            Detailed error dictionary for logging or API responses
        """
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "code_str": self.code_str,
                "http_status": self.http_status,
                "severity": self.severity.name,
                "retryable": self.retryable,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
            }
        }

    def to_json(self) -> str:
        """
        Convert the exception to a JSON string.

        Returns:
            JSON-formatted error representation
        """
        return json.dumps(self.to_dict(), indent=2)

    def _log_exception(self) -> None:
        """
        Log the exception details using structlog with appropriate log level.
        """
        level_map = {
            ErrorSeverity.INFO: "info",
            ErrorSeverity.WARNING: "warning",
            ErrorSeverity.ERROR: "error",
            ErrorSeverity.CRITICAL: "critical",
        }
        log_level = level_map.get(self.severity, "error")
        log_method = getattr(logger, log_level, logger.error)
        log_method("exception_raised", **self.to_dict()["error"])

    @classmethod
    def create(cls: type[T], message: str | None = None, code: int | None = None, **kwargs) -> T:
        """
        Factory method to create exception instances with flexible configuration.

        Args:
            message: Optional custom message
            code: Optional custom error code
            **kwargs: Additional configuration parameters

        Returns:
            Instantiated exception with specified parameters
        """
        # Use class or method-provided values, with fallbacks
        actual_message = message or getattr(cls, "DEFAULT_MESSAGE", "An error occurred")
        actual_code = code or getattr(cls, "DEFAULT_CODE", 1000)

        return cls(message=actual_message, code=actual_code, **kwargs)  # type: ignore[call-arg]

    def __repr__(self) -> str:
        """
        Provide a detailed string representation of the exception.

        Returns:
            Comprehensive exception representation
        """
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"code={self.code}, "
            f"severity={self.severity.name}, "
            f"retryable={self.retryable})"
        )


class GeneralError(BaseException):
    """Base class for general application errors."""

    DEFAULT_MESSAGE = "An unspecified error occurred"
    DEFAULT_CODE = 1000
    DEFAULT_HTTP_STATUS = 500


class ConfigError(GeneralError):
    """Error related to configuration issues."""

    DEFAULT_MESSAGE = "Configuration error"
    DEFAULT_CODE = 1001
    DEFAULT_HTTP_STATUS = 500


class ServiceInitializationError(GeneralError):
    """Error during service initialization."""

    DEFAULT_MESSAGE = "Service initialization error"
    DEFAULT_CODE = 1002
    DEFAULT_HTTP_STATUS = 500


class ErrorRegistry:
    """
    Central registry for tracking and managing application exceptions.
    """

    _registry: dict[str, type[BaseException]] = {}

    @classmethod
    def register(cls, exception_class: type[BaseException]):
        """
        Register an exception class in the central registry.

        Args:
            exception_class: Exception class to register
        """
        cls._registry[exception_class.__name__] = exception_class
        logger.info("exception_registered", exception_name=exception_class.__name__)

    @classmethod
    def get(cls, exception_name: str) -> type[BaseException] | None:
        """
        Retrieve a registered exception class.

        Args:
            exception_name: Name of the exception class

        Returns:
            Registered exception class or None
        """
        return cls._registry.get(exception_name)

    @classmethod
    def list_exceptions(cls) -> list[str]:
        """
        List all registered exception names.

        Returns:
            List of registered exception class names
        """
        return list(cls._registry.keys())


def create_exception(exception_class: type[T], message: str | None = None, code: int | None = None, **kwargs) -> T:
    """
    Factory function to create exceptions with proper typing.

    Args:
        exception_class: The exception class to instantiate
        message: Optional error message override
        code: Optional error code override
        **kwargs: Additional parameters for the exception

    Returns:
        An instance of the specified exception class
    """
    return exception_class.create(message=message, code=code, **kwargs)  # type: ignore[attr-defined]


# Automatically register base and core exception classes
def _register_core_exceptions():
    """
    Automatically register core exception classes from this module.
    """
    core_exceptions = [GeneralError, ConfigError, ServiceInitializationError, BaseException]

    for exc in core_exceptions:
        ErrorRegistry.register(exc)


# Initialize core exception registration
_register_core_exceptions()


# Networking Errors
class NetworkError(GeneralError):
    def __init__(self, message: str | None = "Network error occurred.", code: int = 7000, **kwargs):
        super().__init__(message=message, code=code, **kwargs)


class TimeoutError(NetworkError):
    def __init__(self, message: str | None = "The operation timed out.", code: int = 7001, **kwargs):
        super().__init__(message=message, code=code, **kwargs)


class RateLimitExceededError(NetworkError):
    def __init__(self, message: str | None = "Rate limit has been exceeded.", code: int = 7004, **kwargs):
        super().__init__(message=message, code=code, **kwargs)


class ConfigurationValidationError(GeneralError):
    """Error related to configuration validation."""

    DEFAULT_MESSAGE = "Configuration validation error"
    DEFAULT_CODE = 1003
    DEFAULT_HTTP_STATUS = 400


class DatabaseConnectionError(GeneralError):
    """Error related to database connection issues."""

    DEFAULT_MESSAGE = "Database connection error"
    DEFAULT_CODE = 1004
    DEFAULT_HTTP_STATUS = 500
