"""
Base exception classes for the application framework.

This module defines a comprehensive set of base exceptions
to provide consistent error handling across the application.
"""

from typing import Any


class BaseException(Exception):
    """
    Custom base exception class with enhanced error reporting.

    Attributes:
        message (str): The error message.
        details (Dict[str, Any]): Additional details about the error.
        code (Optional[int]): An optional error code for categorizing the error.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None, code: int | None = None):
        """
        Initialize a BaseException instance.

        Args:
            message (str): A descriptive error message.
            details (Dict[str, Any], optional): Additional context or details about the error.
            code (Optional[int]): An optional error code for categorizing the error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.code = code

    def __str__(self) -> str:
        """
        Return a string representation of the error, including details and code if available.

        Returns:
            str: The string representation of the error.
        """
        base_message = f"{self.message}"
        if self.code is not None:
            base_message += f" | Code: {self.code}"
        if self.details:
            base_message += f" | Details: {self.details}"
        return base_message

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the error message, details, and code.
        """
        return {
            "message": self.message,
            "details": self.details,
            "code": self.code,
        }


class FrameworkException(Exception):
    """
    Base exception class for all framework-specific exceptions.

    All framework-specific exceptions should inherit from this class
    to allow for unified exception handling.
    """

    def __init__(self, message: str = "", code: str = "FRAMEWORK_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new framework exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        self.message = message or "An unexpected error occurred"
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Get a string representation of the exception.

        Returns:
            String representation with code and message
        """
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception to a dictionary.

        Returns:
            Dictionary representation of the exception
        """
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationException(FrameworkException):
    """
    Exception raised for configuration errors.

    This includes:
    - Missing required configuration
    - Invalid configuration values
    - Configuration schema violations
    """

    def __init__(self, message: str = "", code: str = "CONFIG_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new configuration exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "A configuration error occurred"
        super().__init__(message=message, code=code, details=details)


class ValidationException(FrameworkException):
    """
    Exception raised for validation errors.

    This includes:
    - Invalid input data
    - Schema violations
    - Business rule violations
    """

    def __init__(self, message: str = "", code: str = "VALIDATION_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new validation exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "A validation error occurred"
        super().__init__(message=message, code=code, details=details)


class ComponentException(FrameworkException):
    """
    Exception raised for component errors.

    This includes:
    - Component initialization errors
    - Component lifecycle errors
    - Component dependency errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "COMPONENT_ERROR",
        details: dict[str, Any] | None = None,
        component_id: str | None = None,
    ):
        """
        Initialize a new component exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            component_id: ID of the component that raised the exception
        """
        message = message or "A component error occurred"
        details = details or {}

        if component_id:
            details["component_id"] = component_id

        super().__init__(message=message, code=code, details=details)


class ServiceException(ComponentException):
    """
    Exception raised for service errors.

    This includes:
    - Service initialization errors
    - Service execution errors
    - Service dependency errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "SERVICE_ERROR",
        details: dict[str, Any] | None = None,
        service_id: str | None = None,
    ):
        """
        Initialize a new service exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            service_id: ID of the service that raised the exception
        """
        message = message or "A service error occurred"
        super().__init__(message=message, code=code, details=details, component_id=service_id)


class NotImplementedException(FrameworkException):
    """
    Exception raised for not implemented features.

    This should be used for features that are planned but not yet implemented.
    """

    def __init__(self, message: str = "", code: str = "NOT_IMPLEMENTED", details: dict[str, Any] | None = None):
        """
        Initialize a new not implemented exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "This feature is not implemented yet"
        super().__init__(message=message, code=code, details=details)


class AuthenticationError(FrameworkException):
    """
    Exception raised for authentication-related errors.

    This includes:
    - Invalid credentials
    - Unauthorized access
    - Authentication failures
    """

    def __init__(self, message: str = "", code: str = "AUTH_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new authentication exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "Authentication failed"
        super().__init__(message=message, code=code, details=details)


class NotFoundError(FrameworkException):
    """
    Exception raised when a requested resource is not found.

    This includes:
    - Missing database records
    - Non-existent resources
    - Unavailable endpoints
    """

    def __init__(
        self,
        message: str = "",
        code: str = "NOT_FOUND",
        details: dict[str, Any] | None = None,
        resource: str | None = None,
    ):
        """
        Initialize a new not found exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            resource: Name or identifier of the resource not found
        """
        message = message or "Resource not found"
        details = details or {}

        if resource:
            details["resource"] = resource

        super().__init__(message=message, code=code, details=details)
