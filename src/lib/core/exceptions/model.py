"""
Model exception classes for the application framework.

This module defines a comprehensive set of model-related exceptions
to provide consistent error handling across the model components of the application.
"""

from typing import Any

from lib.core.exceptions.base import FrameworkException


class ModelError(FrameworkException):
    """
    Base exception class for all model-related exceptions.

    All model-specific exceptions should inherit from this class
    to allow for unified exception handling.
    """

    def __init__(self, message: str = "", code: str = "MODEL_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new model exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "A model-related error occurred"
        super().__init__(message=message, code=code, details=details)


class ModelInitializationError(ModelError):
    """
    Exception raised during model initialization.

    This includes:
    - Model configuration errors
    - Model parameter validation failures
    - Initialization dependency errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_INIT_ERROR",
        details: dict[str, Any] | None = None,
        model_id: str | None = None,
    ):
        """
        Initialize a new model initialization exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            model_id: ID of the model that failed to initialize
        """
        message = message or "Failed to initialize model"
        details = details or {}

        if model_id:
            details["model_id"] = model_id

        super().__init__(message=message, code=code, details=details)


class ModelValidationError(ModelError):
    """
    Exception raised when a model fails validation.

    This includes:
    - Schema validation failures
    - Model integrity checks
    - Constraint violations
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
        validation_errors: list[str] | None = None,
    ):
        """
        Initialize a new model validation exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            validation_errors: List of specific validation errors
        """
        message = message or "Model validation failed"
        details = details or {}

        if validation_errors:
            details["validation_errors"] = validation_errors

        super().__init__(message=message, code=code, details=details)


class ModelPersistenceError(ModelError):
    """
    Exception raised for model persistence operations.

    This includes:
    - Model saving errors
    - Model loading errors
    - Storage system failures
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_PERSISTENCE_ERROR",
        details: dict[str, Any] | None = None,
        operation: str | None = None,
    ):
        """
        Initialize a new model persistence exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            operation: The persistence operation that failed (e.g., 'save', 'load')
        """
        message = message or "Model persistence operation failed"
        details = details or {}

        if operation:
            details["operation"] = operation

        super().__init__(message=message, code=code, details=details)


class ModelNotFoundError(ModelError):
    """
    Exception raised when a model is not found.

    This includes:
    - Model ID not found in storage
    - Model file missing
    - Model reference invalid
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_NOT_FOUND",
        details: dict[str, Any] | None = None,
        model_id: str | None = None,
    ):
        """
        Initialize a new model not found exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            model_id: ID of the model that was not found
        """
        message = message or "Model not found"
        details = details or {}

        if model_id:
            details["model_id"] = model_id

        super().__init__(message=message, code=code, details=details)


class ModelVersionError(ModelError):
    """
    Exception raised for model version issues.

    This includes:
    - Version incompatibility
    - Version conflict
    - Unsupported version
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_VERSION_ERROR",
        details: dict[str, Any] | None = None,
        current_version: str | None = None,
        required_version: str | None = None,
    ):
        """
        Initialize a new model version exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            current_version: Current model version
            required_version: Required model version
        """
        message = message or "Model version error"
        details = details or {}

        if current_version:
            details["current_version"] = current_version
        if required_version:
            details["required_version"] = required_version

        super().__init__(message=message, code=code, details=details)


class ModelSerializationError(ModelError):
    """
    Exception raised during model serialization/deserialization.

    This includes:
    - Format conversion errors
    - Serialization compatibility issues
    - Corrupt serialized data
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_SERIALIZATION_ERROR",
        details: dict[str, Any] | None = None,
        operation: str | None = None,
    ):
        """
        Initialize a new model serialization exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            operation: The serialization operation that failed (e.g., 'serialize', 'deserialize')
        """
        message = message or "Model serialization error"
        details = details or {}

        if operation:
            details["operation"] = operation

        super().__init__(message=message, code=code, details=details)


class ModelOperationError(ModelError):
    """
    Exception raised during model operations.

    This includes:
    - Method execution errors
    - Operation timeout
    - Operation constraint violations
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_OPERATION_ERROR",
        details: dict[str, Any] | None = None,
        operation: str | None = None,
    ):
        """
        Initialize a new model operation exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            operation: The model operation that failed
        """
        message = message or "Model operation failed"
        details = details or {}

        if operation:
            details["operation"] = operation

        super().__init__(message=message, code=code, details=details)


class ModelMigrationError(ModelError):
    """
    Exception raised during model migration.

    This includes:
    - Migration script failures
    - Version migration incompatibilities
    - Data structure migration errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_MIGRATION_ERROR",
        details: dict[str, Any] | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
    ):
        """
        Initialize a new model migration exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            from_version: Source model version
            to_version: Target model version
        """
        message = message or "Model migration failed"
        details = details or {}

        if from_version:
            details["from_version"] = from_version
        if to_version:
            details["to_version"] = to_version

        super().__init__(message=message, code=code, details=details)


class ModelRelationshipError(ModelError):
    """
    Exception raised for model relationship errors.

    This includes:
    - Invalid relationships
    - Relationship constraint violations
    - Circular dependencies
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_RELATIONSHIP_ERROR",
        details: dict[str, Any] | None = None,
        relationship_type: str | None = None,
        related_models: list[str] | None = None,
    ):
        """
        Initialize a new model relationship exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            relationship_type: Type of relationship that caused the error
            related_models: List of models involved in the relationship
        """
        message = message or "Model relationship error"
        details = details or {}

        if relationship_type:
            details["relationship_type"] = relationship_type
        if related_models:
            details["related_models"] = related_models

        super().__init__(message=message, code=code, details=details)
