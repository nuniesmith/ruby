"""
Comprehensive API exception classes for handling various API-related errors.

This module provides a structured set of exceptions for different
API error scenarios, inheriting from a base ApiException class.
"""

from typing import Any

from lib.core.exceptions.base import FrameworkException


class ApiException(FrameworkException):
    """
    Base exception class for API-related errors.

    This includes:
    - Client errors
    - Server errors
    - Authentication errors
    - Rate limit errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "API_ERROR",
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ):
        """
        Initialize a new API exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            status_code: HTTP status code
        """
        details = details or {}

        if status_code is not None:
            details["status_code"] = status_code

        super().__init__(message=message, code=code, details=details)


class ApiClientError(ApiException):
    """
    Exception raised for general client-side API errors.
    """

    def __init__(self, message: str = "", code: str = "CLIENT_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new client error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        super().__init__(
            message=message or "API client error occurred",
            code=code,
            details=details,
            status_code=400,  # Bad Request
        )


class ApiTimeoutError(ApiException):
    """
    Raised when an API request times out.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "TIMEOUT_ERROR",
        details: dict[str, Any] | None = None,
        timeout_duration: int | float | None = None,
    ):
        """
        Initialize the timeout error.

        Args:
            message: Timeout error message
            code: Error code
            details: Additional error context
            timeout_duration: Duration after which the timeout occurred
        """
        details = details or {}
        if timeout_duration is not None:
            details["timeout_duration"] = timeout_duration

        super().__init__(
            message=message or "API request timed out",
            code=code,
            details=details,
            status_code=504,  # Gateway Timeout
        )


class ApiRateLimitError(ApiException):
    """
    Raised when rate limits are exceeded.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "RATE_LIMIT",
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ):
        """
        Initialize a new rate limit exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            retry_after: Seconds to wait before retrying
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(
            message=message or "API rate limit exceeded",
            code=code,
            details=details,
            status_code=429,  # Too Many Requests
        )


class ApiCircuitOpenError(ApiException):
    """
    Raised when the API circuit breaker is open.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "CIRCUIT_OPEN",
        details: dict[str, Any] | None = None,
        retry_after: int | float | None = None,
    ):
        """
        Initialize the circuit open error.

        Args:
            message: Error message
            code: Error code
            details: Additional error context
            retry_after: Suggested time to wait before retrying
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(
            message=message or "API circuit is open",
            code=code,
            details=details,
            status_code=503,  # Service Unavailable
        )


class ApiValidationError(ApiException):
    """
    Raised when API request validation fails.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
        validation_errors: dict[str, Any] | None = None,
    ):
        """
        Initialize the validation error.

        Args:
            message: Validation error message
            code: Error code
            details: Additional error context
            validation_errors: Dictionary of specific field validation errors
        """
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors

        super().__init__(
            message=message or "API request validation failed",
            code=code,
            details=details,
            status_code=400,  # Bad Request
        )


class ApiAuthenticationError(ApiException):
    """
    Raised when API authentication fails.
    """

    def __init__(self, message: str = "", code: str = "AUTHENTICATION_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the authentication error.

        Args:
            message: Authentication error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "API authentication failed",
            code=code,
            details=details,
            status_code=401,  # Unauthorized
        )


class ApiConnectionError(ApiException):
    """
    Raised when there is a connection-related error with the API.
    """

    def __init__(self, message: str = "", code: str = "CONNECTION_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the connection error.

        Args:
            message: Connection error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "API connection failed",
            code=code,
            details=details,
            status_code=503,  # Service Unavailable
        )


class ApiResponseError(ApiException):
    """
    Raised when there is an error processing the API response.
    """

    def __init__(self, message: str = "", code: str = "RESPONSE_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the response error.

        Args:
            message: Response error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "Error processing API response",
            code=code,
            details=details,
            status_code=500,  # Internal Server Error
        )


class SecurityError(ApiException):
    """
    Raised for security-related errors in API interactions.
    """

    def __init__(self, message: str = "", code: str = "SECURITY_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the security error.

        Args:
            message: Security error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "Security violation in API interaction",
            code=code,
            details=details,
            status_code=403,  # Forbidden
        )


class ConfigurationError(ApiException):
    """
    Raised when there is an error in API configuration.
    """

    def __init__(self, message: str = "", code: str = "CONFIGURATION_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the configuration error.

        Args:
            message: Configuration error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "API configuration error",
            code=code,
            details=details,
            status_code=500,  # Internal Server Error
        )


# Ensure compatibility with built-in ValueError
class ValueError(ApiException):
    """
    Extended ValueError that inherits from ApiException.
    """

    def __init__(self, message: str = "", code: str = "VALUE_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize the value error.

        Args:
            message: Value error message
            code: Error code
            details: Additional error context
        """
        super().__init__(
            message=message or "Invalid value provided",
            code=code,
            details=details,
            status_code=400,  # Bad Request
        )


class SourceUnavailableError(ApiException):
    """
    Raised when an API data source is unavailable.

    This includes:
    - External service outages
    - Resource unavailability
    - Temporary source failures
    """

    def __init__(
        self,
        message: str = "",
        code: str = "SOURCE_UNAVAILABLE",
        details: dict[str, Any] | None = None,
        source_name: str | None = None,
    ):
        """
        Initialize the source unavailable error.

        Args:
            message: Error message
            code: Error code
            details: Additional error context
            source_name: Name of the unavailable source
        """
        details = details or {}
        if source_name is not None:
            details["source_name"] = source_name

        message = message or (
            f"API source '{source_name}' is unavailable" if source_name else "API source is unavailable"
        )

        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=503,  # Service Unavailable
        )


class CircuitOpenError(ApiException):
    """
    Raised when a circuit breaker is open in the API.

    This is a simplified version of ApiCircuitOpenError for scenarios
    where the full circuit breaker implementation details are not needed.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "CIRCUIT_OPEN",
        details: dict[str, Any] | None = None,
        service_name: str | None = None,
    ):
        """
        Initialize the circuit open error.

        Args:
            message: Error message
            code: Error code
            details: Additional error context
            service_name: Name of the service with open circuit
        """
        details = details or {}
        if service_name is not None:
            details["service_name"] = service_name

        message = message or (f"Circuit is open for service '{service_name}'" if service_name else "Circuit is open")

        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=503,  # Service Unavailable
        )


class ClientNotFoundError(ApiException):
    """
    Raised when a client is not found in the API.

    This includes:
    - Missing client ID
    - Invalid client credentials
    """

    def __init__(
        self,
        message: str = "",
        code: str = "CLIENT_NOT_FOUND",
        details: dict[str, Any] | None = None,
        client_id: str | None = None,
    ):
        """
        Initialize the client not found error.

        Args:
            message: Error message
            code: Error code
            details: Additional error context
            client_id: ID of the missing client
        """
        details = details or {}
        if client_id is not None:
            details["client_id"] = client_id

        message = message or (f"Client with ID '{client_id}' not found" if client_id else "Client not found")

        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=404,  # Not Found
        )


class TokenExpiredError(Exception):
    """
    Exception raised when a JWT token has expired.

    This exception is used during token validation to indicate that
    the token's expiration timestamp (exp claim) has passed.
    """

    def __init__(self, message="Token has expired"):
        self.message = message
        super().__init__(self.message)


class InvalidTokenError(Exception):
    """
    Exception raised when a JWT token is invalid.

    This could be due to invalid signature, malformed token structure,
    missing required claims, or other validation failures.
    """

    def __init__(self, message="Invalid token"):
        self.message = message
        super().__init__(self.message)


class AuthenticationError(Exception):
    """
    General authentication error.

    Used for authentication failures that don't fall into more specific
    categories like token expiration or invalid tokens. Examples include
    insufficient permissions, token creation failures, or revocation issues.
    """

    def __init__(self, message="Authentication failed"):
        self.message = message
        super().__init__(self.message)
