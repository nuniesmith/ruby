"""
Framework exceptions.
This package contains the exception hierarchy used throughout the framework.
"""

# Import exceptions from modules
from .api import (
    ApiAuthenticationError,
    ApiCircuitOpenError,
    ApiClientError,
    ApiConnectionError,
    ApiException,
    ApiRateLimitError,
    ApiResponseError,
    ApiTimeoutError,
    ApiValidationError,
    ConfigurationError,
    SecurityError,
    ValueError,
)
from .base import (
    AuthenticationError,
    BaseException,
    ComponentException,
    ConfigurationException,
    FrameworkException,
    NotFoundError,
    NotImplementedException,
    ServiceException,
    ValidationException,
)
from .boundary import ErrorBoundary
from .classes import ConfigError, ErrorRegistry, ErrorSeverity, GeneralError, ServiceInitializationError
from .data import (
    DataException,
    DataNotFoundError,
    DataProcessingException,
    DataSourceException,
    DataSourceNotFoundError,
    DataStorageException,
    DataValidationException,
    InsufficientDataException,
)
from .model import (
    ModelError,
    ModelInitializationError,
    ModelMigrationError,
    ModelNotFoundError,
    ModelOperationError,
    ModelPersistenceError,
    ModelRelationshipError,
    ModelSerializationError,
    ModelValidationError,
    ModelVersionError,
)
from .prediction import (
    FeatureExtractionException,
    ModelLoadingException,
    ModelNotFoundException,
    ModelVersionException,
    PredictionException,
    PredictionInputException,
    PredictionOutputException,
    PredictionProcessingException,
    PredictionServiceException,
    PredictionTimeoutException,
)
from .trading import (
    ExecutionException,
    InsufficientFundsException,
    MarketConditionException,
    OrderException,
    PositionLimitException,
    StrategyException,
    TradingException,
)
from .validation import (
    ValidationError,
)

__all__ = [
    # Base exceptions
    "BaseException",
    "FrameworkException",
    "ConfigurationException",
    "ValidationException",
    "ComponentException",
    "ServiceException",
    "NotImplementedException",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    # API exceptions
    "ApiException",
    "ApiClientError",
    "ApiTimeoutError",
    "ApiRateLimitError",
    "ApiCircuitOpenError",
    "ApiValidationError",
    "ApiAuthenticationError",
    "ApiConnectionError",
    "ApiResponseError",
    "SecurityError",
    "ConfigurationError",
    "ValueError",
    # Data exceptions
    "DataException",
    "DataSourceException",
    "DataSourceNotFoundError",
    "DataValidationException",
    "DataProcessingException",
    "DataStorageException",
    "DataNotFoundError",
    "InsufficientDataException",
    # Trading exceptions
    "TradingException",
    "OrderException",
    "StrategyException",
    "ExecutionException",
    "MarketConditionException",
    "InsufficientFundsException",
    "PositionLimitException",
    # Prediction exceptions
    "PredictionException",
    "PredictionServiceException",
    "ModelLoadingException",
    "PredictionInputException",
    "PredictionProcessingException",
    "PredictionOutputException",
    "ModelNotFoundException",
    "PredictionTimeoutException",
    "FeatureExtractionException",
    "ModelVersionException",
    # Model exceptions
    "ModelError",
    "ModelInitializationError",
    "ModelValidationError",
    "ModelPersistenceError",
    "ModelNotFoundError",
    "ModelVersionError",
    "ModelSerializationError",
    "ModelOperationError",
    "ModelMigrationError",
    "ModelRelationshipError",
    # Classes exceptions and utilities
    "ErrorSeverity",
    "GeneralError",
    "ConfigError",
    "ServiceInitializationError",
    "ErrorRegistry",
    # Error handling utilities
    "ErrorBoundary",
]
