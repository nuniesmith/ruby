import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from lib.core.exceptions import FrameworkException
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for validation errors."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationException(FrameworkException):
    """
    Base exception for all validation-related errors.
    """

    DEFAULT_MESSAGE = "Validation error"
    DEFAULT_CODE = "VALIDATION_ERROR"
    DEFAULT_HTTP_STATUS = 422

    def __init__(self, message: str = "", code: str = "", details: dict[str, Any] | None = None, http_status: int = 0):
        """
        Initialize a new validation exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            http_status: HTTP status code to return
        """
        message = message or self.DEFAULT_MESSAGE
        code = code or self.DEFAULT_CODE
        http_status = http_status or self.DEFAULT_HTTP_STATUS
        super().__init__(message=message, code=code, details=details)


@dataclass
class ValidationErrorDetail:
    """Structured validation error detail."""

    loc: tuple = field(default_factory=tuple)
    msg: str = ""
    type: str = ""
    ctx: dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {"loc": self.loc, "msg": self.msg, "type": self.type, "ctx": self.ctx, "severity": self.severity.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationErrorDetail":
        """Create from dictionary."""
        severity = data.get("severity")
        if severity is None:
            severity = ErrorSeverity.ERROR
        elif isinstance(severity, str):
            try:
                severity = ErrorSeverity[severity]
            except KeyError:
                severity = ErrorSeverity.ERROR

        return cls(
            loc=tuple(data.get("loc", ())),
            msg=data.get("msg", ""),
            type=data.get("type", ""),
            ctx=data.get("ctx", {}),
            severity=severity,
        )


class ValidationError(ValidationException):
    """
    Exception raised for input validation errors.

    This can represent both simple validation errors and complex
    validation structures with multiple error details.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "",
        details: dict[str, Any] | list[ValidationErrorDetail] | None = None,
        field: str = "",
        value: Any = None,
    ):
        """
        Initialize a validation error exception.

        Args:
            message: Human-readable error message
            code: Error code
            details: Additional details or list of validation error details
            field: Field name that failed validation (for simple cases)
            value: Invalid value (for simple cases)
        """
        if field and not message:
            message = f"Validation failed for field '{field}'"

        if field and value is not None and not details:
            details = {"field": field, "value": value}

        # Convert list of ValidationErrorDetail to dict format if needed
        if isinstance(details, list):
            details = {"errors": [err.to_dict() for err in details]}

        super().__init__(message=message, code=code, details=details)
        self.field = field
        self.value = value


class ConfigurationValidationError(ValidationError):
    """Exception raised specifically for configuration validation errors."""

    DEFAULT_CODE = "CONFIG_VALIDATION_ERROR"

    def __init__(self, key: str, value: Any = None, message: str = "", details: dict[str, Any] | None = None):
        message = message or f"Invalid configuration value for '{key}'"
        super().__init__(message=message, field=key, value=value, details=details)


class SchemaValidationError(ValidationError):
    """Exception raised specifically for schema validation failures."""

    DEFAULT_CODE = "SCHEMA_VALIDATION_ERROR"


def validate_exception_schema(schema_path, config_data):
    """
    Validate exception schema to ensure compliance.

    Args:
        schema_path: Path to the schema file
        config_data: Data to validate against the schema

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        with open(schema_path) as schema_file:
            schema = json.load(schema_file)
        jsonschema_validate(instance=config_data, schema=schema)
        logger.info("exception_schema_validated")
    except JsonSchemaValidationError as e:
        logger.error("schema_validation_error", error=e.message, schema_path=str(schema_path))
        raise SchemaValidationError(
            message=f"Schema validation error: {e.message}", details={"schema_path": schema_path, "error": str(e)}
        ) from e
