from typing import Any

from lib.core.exceptions.base import FrameworkException


class DataException(FrameworkException):
    """
    Base exception class for data-related errors.

    This includes:
    - Data source errors
    - Data processing errors
    - Data validation errors
    - Data storage errors
    """

    def __init__(self, message: str | None = None, code: str | None = None, details: dict[str, Any] | None = None):
        """
        Initialize a new data exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        code = code or "DATA_ERROR"
        message = message or "A data error occurred"
        super().__init__(message=message, code=code, details=details)


class DataSourceException(DataException):
    """
    Exception raised for data source errors.

    This includes:
    - Connection errors
    - Authentication errors
    - Rate limiting
    - Source-specific errors
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        source_name: str | None = None,
    ):
        """
        Initialize a new data source exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            source_name: Name of the data source
        """
        code = code or "DATA_SOURCE_ERROR"

        details = details or {}
        if source_name:
            details["source"] = source_name

        message = message or (f"Error from data source: {source_name}" if source_name else "Data source error")

        super().__init__(message=message, code=code, details=details)


class DataSourceNotFoundError(DataSourceException):
    """
    Exception raised when a requested data source is not found or cannot be accessed.

    This includes:
    - Missing data sources
    - Unavailable data sources
    - Undefined data source configurations
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        source_name: str | None = None,
        source_type: str | None = None,
    ):
        """
        Initialize a new data source not found exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            source_name: Name of the data source
            source_type: Type of the data source
        """
        code = code or "DATA_SOURCE_NOT_FOUND"

        details = details or {}
        if source_name:
            details["source_name"] = source_name
        if source_type:
            details["source_type"] = source_type

        if source_name and source_type:
            message = message or f"Data source '{source_name}' of type '{source_type}' not found"
        elif source_name:
            message = message or f"Data source '{source_name}' not found"
        elif source_type:
            message = message or f"Data source of type '{source_type}' not found"
        else:
            message = message or "Data source not found"

        super().__init__(message=message, code=code, details=details, source_name=source_name)


class DataValidationException(DataException):
    """
    Exception raised for data validation errors.

    This includes:
    - Schema violations
    - Constraint violations
    - Data integrity issues
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        validation_errors: dict[str, Any] | None = None,
    ):
        """
        Initialize a new data validation exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            validation_errors: Dictionary of field-specific validation errors
        """
        code = code or "DATA_VALIDATION_ERROR"
        message = message or "Data validation error"

        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors

        super().__init__(message=message, code=code, details=details)


class DataProcessingException(DataException):
    """
    Exception raised for data processing errors.

    This includes:
    - Transformation errors
    - Normalization errors
    - Feature extraction errors
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        processor_name: str | None = None,
    ):
        """
        Initialize a new data processing exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            processor_name: Name of the data processor
        """
        code = code or "DATA_PROCESSING_ERROR"

        details = details or {}
        if processor_name:
            details["processor"] = processor_name

        if processor_name:
            message = message or f"Error in data processor: {processor_name}"
        else:
            message = message or "Data processing error"

        super().__init__(message=message, code=code, details=details)


class DataStorageException(DataException):
    """
    Exception raised for data storage errors.

    This includes:
    - Database errors
    - File system errors
    - Cache errors
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        storage_type: str | None = None,
    ):
        """
        Initialize a new data storage exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            storage_type: Type of storage (e.g., "database", "file", "cache")
        """
        code = code or "DATA_STORAGE_ERROR"

        details = details or {}
        if storage_type:
            details["storage_type"] = storage_type

        message = message or (f"Error in {storage_type} storage" if storage_type else "Data storage error")

        super().__init__(message=message, code=code, details=details)


class DataNotFoundError(DataException):
    """
    Exception raised when requested data is not found.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        data_type: str | None = None,
        data_id: str | int | None = None,
    ):
        """
        Initialize a new data not found exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            data_type: Type of data that was not found
            data_id: ID of data that was not found
        """
        code = code or "DATA_NOT_FOUND"

        details = details or {}
        if data_type:
            details["data_type"] = data_type
        if data_id:
            details["data_id"] = data_id

        if data_type and data_id:
            message = message or f"{data_type} with ID '{data_id}' not found"
        elif data_type:
            message = message or f"{data_type} data not found"
        else:
            message = message or "Data not found"

        super().__init__(message=message, code=code, details=details)


class InsufficientDataException(DataException):
    """
    Exception raised when there is not enough data for an operation.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        required: int | None = None,
        available: int | None = None,
    ):
        """
        Initialize a new insufficient data exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            required: Required amount of data
            available: Available amount of data
        """
        code = code or "INSUFFICIENT_DATA"

        details = details or {}
        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available

        if required is not None and available is not None:
            message = message or f"Insufficient data: required {required}, available {available}"
        else:
            message = message or "Insufficient data for operation"

        super().__init__(message=message, code=code, details=details)


class DataInconsistencyError(DataException):
    """
    Exception raised when data is inconsistent or malformed.

    This includes:
    - Missing values
    - Duplicates
    - Incorrect formats
    """

    def __init__(self, message: str | None = None, code: str | None = None, details: dict[str, Any] | None = None):
        """
        Initialize a new data inconsistency exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        code = code or "DATA_INCONSISTENCY"
        message = message or "Data inconsistency error"

        details = details or {}

        super().__init__(message=message, code=code, details=details)


class CacheError(DataException):
    """
    Exception raised for cache-related errors.

    This includes:
    - Cache misses
    - Cache corruption
    - Cache invalidation errors
    """

    def __init__(self, message: str | None = None, code: str | None = None, details: dict[str, Any] | None = None):
        """
        Initialize a new cache error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        code = code or "CACHE_ERROR"
        message = message or "Cache error"

        details = details or {}

        super().__init__(message=message, code=code, details=details)


# Enhanced Database Exception Hierarchy


class DatabaseError(DataStorageException):
    """
    Exception raised for database-related errors.

    This includes:
    - Connection errors
    - Query errors
    - Transaction errors
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
        sql: str | None = None,
    ):
        """
        Initialize a new database error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
            sql: SQL statement that caused the error (sanitized)
        """
        code = code or "DATABASE_ERROR"
        message = message or "Database error"

        details = details or {}
        if db_type:
            details["db_type"] = db_type
        if sql:
            details["sql"] = sql

        super().__init__(message=message, code=code, details=details, storage_type="database")


class DatabaseConnectionError(DatabaseError):
    """
    Exception raised when a database connection cannot be established or is lost.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        """
        Initialize a new database connection error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
            host: Database host
            port: Database port
        """
        code = code or "DATABASE_CONNECTION_ERROR"

        details = details or {}
        if host:
            # Sanitize host info for security
            sanitized_host = host.split(".")[0] + ".*" if "." in host else host
            details["host"] = sanitized_host
        if port:
            details["port"] = port

        if host and port:
            message = message or f"Could not connect to database at {host}:{port}"
        elif host:
            message = message or f"Could not connect to database at {host}"
        else:
            message = message or "Database connection failed"

        super().__init__(message=message, code=code, details=details, db_type=db_type)


class DatabaseQueryError(DatabaseError):
    """
    Exception raised when a database query fails.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
        sql: str | None = None,
        params: list | None = None,
    ):
        """
        Initialize a new database query error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
            sql: SQL statement that caused the error (sanitized)
            params: Query parameters (sanitized)
        """
        code = code or "DATABASE_QUERY_ERROR"
        message = message or "Database query failed"

        details = details or {}
        if sql:
            # Remove potential sensitive data in the query
            sanitized_sql = self._sanitize_sql(sql)
            details["query"] = sanitized_sql
        if params:
            # Sanitize parameter values
            sanitized_params = self._sanitize_params(params)
            details["params"] = sanitized_params

        super().__init__(message=message, code=code, details=details, db_type=db_type, sql=sql)

    @staticmethod
    def _sanitize_sql(sql: str) -> str:
        """Sanitize SQL query to remove potential sensitive information."""
        # This is a simple sanitization - in production you might want more sophisticated filtering
        return sql

    @staticmethod
    def _sanitize_params(params: list) -> list:
        """Sanitize query parameters to remove potential sensitive information."""
        # This is a simple sanitization - in production you might want more sophisticated filtering
        return ["***" if isinstance(p, str) and len(p) > 20 else p for p in params]


class DatabaseTransactionError(DatabaseError):
    """
    Exception raised when a database transaction fails.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
    ):
        """
        Initialize a new database transaction error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
        """
        code = code or "DATABASE_TRANSACTION_ERROR"
        message = message or "Database transaction failed"

        super().__init__(message=message, code=code, details=details, db_type=db_type)


class DatabaseConstraintViolationError(DatabaseError):
    """
    Exception raised when a database constraint is violated.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
        constraint: str | None = None,
        table: str | None = None,
    ):
        """
        Initialize a new database constraint violation error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
            constraint: Name of the violated constraint
            table: Table where the constraint was violated
        """
        code = code or "DATABASE_CONSTRAINT_VIOLATION"

        details = details or {}
        if constraint:
            details["constraint"] = constraint
        if table:
            details["table"] = table

        if constraint and table:
            message = message or f"Constraint '{constraint}' violated on table '{table}'"
        elif constraint:
            message = message or f"Constraint '{constraint}' violated"
        elif table:
            message = message or f"Database constraint violated on table '{table}'"
        else:
            message = message or "Database constraint violated"

        super().__init__(message=message, code=code, details=details, db_type=db_type)


class DatabaseDeadlockError(DatabaseError):
    """
    Exception raised when a database deadlock is detected.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
    ):
        """
        Initialize a new database deadlock error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
        """
        code = code or "DATABASE_DEADLOCK"
        message = message or "Database deadlock detected"

        super().__init__(message=message, code=code, details=details, db_type=db_type)


class DatabaseTimeoutError(DatabaseError):
    """
    Exception raised when a database operation times out.
    """

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        db_type: str | None = None,
        timeout_seconds: float | None = None,
    ):
        """
        Initialize a new database timeout error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            db_type: Database type (postgres, mysql, etc.)
            timeout_seconds: Timeout duration in seconds
        """
        code = code or "DATABASE_TIMEOUT"

        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds

        if timeout_seconds is not None:
            message = message or f"Database operation timed out after {timeout_seconds} seconds"
        else:
            message = message or "Database operation timed out"

        super().__init__(message=message, code=code, details=details, db_type=db_type)


class DatabaseConfigurationError(DatabaseError):
    """
    Exception raised when there's an issue with database configuration.
    """

    def __init__(self, message: str | None = None, code: str | None = None, details: dict[str, Any] | None = None):
        """
        Initialize a new database configuration error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        code = code or "DATABASE_CONFIGURATION_ERROR"
        message = message or "Database configuration error"

        super().__init__(message=message, code=code, details=details)
