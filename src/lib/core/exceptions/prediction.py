"""
Prediction exception classes for the application framework.

This module defines a comprehensive set of prediction-related exceptions
to provide consistent error handling across the prediction components of the application.
"""

from typing import Any

from lib.core.exceptions.base import FrameworkException, ServiceException


class PredictionException(FrameworkException):
    """
    Base exception class for all prediction-related exceptions.

    All prediction-specific exceptions should inherit from this class
    to allow for unified exception handling.
    """

    def __init__(self, message: str = "", code: str = "PREDICTION_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new prediction exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "A prediction-related error occurred"
        super().__init__(message=message, code=code, details=details)


class PredictionServiceException(ServiceException):
    """
    Exception raised for prediction service errors.

    This includes:
    - Prediction service initialization errors
    - Prediction service execution errors
    - Prediction service dependency errors
    """

    def __init__(
        self,
        message: str = "",
        code: str = "PREDICTION_SERVICE_ERROR",
        details: dict[str, Any] | None = None,
        service_id: str | None = None,
    ):
        """
        Initialize a new prediction service exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            service_id: ID of the prediction service that raised the exception
        """
        message = message or "A prediction service error occurred"
        super().__init__(message=message, code=code, details=details, service_id=service_id)


class ModelLoadingException(PredictionException):
    """
    Exception raised when there is an error loading a prediction model.

    This includes:
    - Model file not found
    - Model version incompatibility
    - Model corruption
    - Insufficient resources for model loading
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_LOADING_ERROR",
        details: dict[str, Any] | None = None,
        model_id: str | None = None,
    ):
        """
        Initialize a new model loading exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            model_id: ID of the model that failed to load
        """
        message = message or "Failed to load prediction model"
        details = details or {}

        if model_id:
            details["model_id"] = model_id

        super().__init__(message=message, code=code, details=details)


class PredictionInputException(PredictionException):
    """
    Exception raised for invalid prediction input data.

    This includes:
    - Missing required input features
    - Invalid input data types
    - Input data out of expected range
    - Input schema violations
    """

    def __init__(self, message: str = "", code: str = "PREDICTION_INPUT_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new prediction input exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "Invalid prediction input data"
        super().__init__(message=message, code=code, details=details)


class PredictionProcessingException(PredictionException):
    """
    Exception raised during prediction processing.

    This includes:
    - Feature preprocessing errors
    - Computation errors during prediction
    - Resource exhaustion during prediction
    - Internal model errors
    """

    def __init__(
        self, message: str = "", code: str = "PREDICTION_PROCESSING_ERROR", details: dict[str, Any] | None = None
    ):
        """
        Initialize a new prediction processing exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "Error during prediction processing"
        super().__init__(message=message, code=code, details=details)


class PredictionOutputException(PredictionException):
    """
    Exception raised for issues with prediction output.

    This includes:
    - Output validation failures
    - Output transformation errors
    - Output format errors
    """

    def __init__(self, message: str = "", code: str = "PREDICTION_OUTPUT_ERROR", details: dict[str, Any] | None = None):
        """
        Initialize a new prediction output exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        message = message or "Error in prediction output"
        super().__init__(message=message, code=code, details=details)


class ModelNotFoundException(PredictionException):
    """
    Exception raised when a requested prediction model is not found.

    This includes:
    - Non-existent model ID
    - Model version not found
    - Model file missing
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
        message = message or "Prediction model not found"
        details = details or {}

        if model_id:
            details["model_id"] = model_id

        super().__init__(message=message, code=code, details=details)


class PredictionTimeoutException(PredictionException):
    """
    Exception raised when a prediction operation times out.

    This includes:
    - Model inference timeout
    - Prediction service response timeout
    - Resource allocation timeout
    """

    def __init__(
        self,
        message: str = "",
        code: str = "PREDICTION_TIMEOUT",
        details: dict[str, Any] | None = None,
        timeout: float | None = None,
    ):
        """
        Initialize a new prediction timeout exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            timeout: The timeout limit in seconds
        """
        message = message or "Prediction operation timed out"
        details = details or {}

        if timeout is not None:
            details["timeout"] = timeout

        super().__init__(message=message, code=code, details=details)


class FeatureExtractionException(PredictionException):
    """
    Exception raised during feature extraction for prediction.

    This includes:
    - Data transformation errors
    - Feature normalization issues
    - Missing feature extractors
    - Feature extraction pipeline failures
    """

    def __init__(
        self,
        message: str = "",
        code: str = "FEATURE_EXTRACTION_ERROR",
        details: dict[str, Any] | None = None,
        feature_name: str | None = None,
    ):
        """
        Initialize a new feature extraction exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            feature_name: Name of the feature that caused the error
        """
        message = message or "Error extracting features for prediction"
        details = details or {}

        if feature_name:
            details["feature_name"] = feature_name

        super().__init__(message=message, code=code, details=details)


class ModelVersionException(PredictionException):
    """
    Exception raised for model version compatibility issues.

    This includes:
    - Version mismatch between model and framework
    - Deprecated model version
    - Incompatible model format
    """

    def __init__(
        self,
        message: str = "",
        code: str = "MODEL_VERSION_ERROR",
        details: dict[str, Any] | None = None,
        expected_version: str | None = None,
        actual_version: str | None = None,
    ):
        """
        Initialize a new model version exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            expected_version: The expected model version
            actual_version: The actual model version
        """
        message = message or "Model version incompatibility"
        details = details or {}

        if expected_version:
            details["expected_version"] = expected_version
        if actual_version:
            details["actual_version"] = actual_version

        super().__init__(message=message, code=code, details=details)
