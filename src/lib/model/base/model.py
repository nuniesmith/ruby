"""
Base model class for all forecasting models.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import log_execution, logger

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None

# Try to import the original ModelMetadata, but don't assign it to avoid type conflicts
try:
    from lib.model.utils.metadata import ModelMetadata

    HAS_ORIGINAL_METADATA = True
    logger.info("Using ModelMetadata models.model_metadata")
except ImportError:
    HAS_ORIGINAL_METADATA = False
    logger.warning("ModelMetadata not found, using local implementation")


class BaseModel(ABC):
    """
    Abstract base class for forecasting models with evaluation metrics, saving/loading capabilities,
    and confidence interval support.

    Each model has a `metadata` attribute that stores model information such as
    model type, creation time, hyperparameters, and metrics.
    """

    def __init__(self):
        # Initialize metadata as a ModelMetadata object for proper type checking
        self._metadata = ModelMetadata(model_type=self.__class__.__name__)

    @property
    def metadata(self) -> ModelMetadata:
        """Get the model metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict | ModelMetadata | SimpleNamespace):
        """Set the model metadata."""
        if isinstance(value, dict):
            self._metadata = ModelMetadata.from_dict(value)
        elif isinstance(value, ModelMetadata):
            self._metadata = value
        elif isinstance(value, SimpleNamespace):
            # Convert SimpleNamespace to dict, then to ModelMetadata
            self._metadata = ModelMetadata.from_dict(vars(value))
        else:
            logger.warning(f"Expected dict, ModelMetadata, or SimpleNamespace for metadata, got {type(value)}.")
            # Create a new metadata object with model_type set
            self._metadata = ModelMetadata(model_type=self.__class__.__name__)

    def set_params(self, **params) -> BaseModel:
        """
        Update model parameters.

        Args:
            **params: Parameters to update

        Returns:
            self: The model instance for chaining
        """
        self._metadata.params.update(params)
        self._metadata.updated_at = datetime.now().isoformat()
        return self

    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns:
            dict: Dictionary of parameter name to value
        """
        return self._metadata.params

    def configure(self, **config) -> BaseModel:
        """
        Configure the model with the given parameters.

        Args:
            **config: Configuration parameters

        Returns:
            self: The model instance for chaining
        """
        return self.set_params(**config)

    @abstractmethod
    def fit(self, train_data: Any, target_column: Any = None, **kwargs) -> BaseModel:
        """
        Fit the model to training data.

        Args:
            train_data: DataFrame containing training data
            target_column: Name of the target column
            **kwargs: Additional fitting parameters

        Returns:
            self: The fitted model instance
        """

    @abstractmethod
    def train(
        self, data: Any = None, target: Any = None, X_train: Any = None, y_train: Any = None, **kwargs
    ) -> BaseModel:
        """
        Train the model with flexible parameter options.

        This method provides a flexible interface that can accept either:
        1. A DataFrame and target column name, or
        2. Pre-split X_train and y_train data

        Args:
            data: DataFrame containing training data
            target: Target column name or Series
            X_train: Alternative input features
            y_train: Alternative target values
            **kwargs: Additional training parameters

        Returns:
            self: The trained model instance
        """

    @abstractmethod
    def predict(self, data: Any = None, horizon: int = 0, return_ci: bool = False) -> Any:
        """
        Generate forecasts with optional confidence intervals.

        Args:
            data: Optional data to use for prediction
            horizon: Number of future time steps to forecast
            return_ci: Whether to return confidence intervals

        Returns:
            Predictions, optionally with confidence intervals.
        """

    @log_execution
    def evaluate(self, y_test: pd.Series, y_pred: pd.Series) -> dict:
        """
        Evaluate model performance using multiple metrics.

        Args:
            y_test: Actual values
            y_pred: Predicted values

        Returns:
            dict: Evaluation metrics including MSE, RMSE, MAE, and R²
        """
        if mean_squared_error is None:
            raise ImportError("scikit-learn is required for evaluate()")

        # Ensure data is properly aligned
        if len(y_test) != len(y_pred):
            logger.warning(f"Length mismatch in evaluation: y_test={len(y_test)}, y_pred={len(y_pred)}")

        # Convert inputs to numpy arrays for consistency
        y_test_np = np.array(y_test).flatten()
        y_pred_np = np.array(y_pred).flatten()

        metrics = {
            "mse": mean_squared_error(y_test_np, y_pred_np),
            "rmse": np.sqrt(mean_squared_error(y_test_np, y_pred_np)),
            "mae": mean_absolute_error(y_test_np, y_pred_np),
            "r2": r2_score(y_test_np, y_pred_np),
        }

        # Store metrics in metadata
        self._metadata.metrics.update(metrics)
        self._metadata.updated_at = datetime.now().isoformat()

        logger.info(f"Evaluation metrics computed: {metrics}")
        return metrics

    @log_execution
    def compute_mse(self, y_pred, y_true) -> float:
        """
        Compute Mean Squared Error between predictions and true values.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            float: Mean Squared Error
        """
        y_true_np = np.array(y_true).flatten()
        y_pred_np = np.array(y_pred).flatten()
        mse = float(np.mean((y_true_np - y_pred_np) ** 2))
        logger.info(f"Computed MSE: {mse}")
        return mse

    @log_execution
    def summary(self) -> str:
        """
        Returns a summary of the model including parameters and evaluation metrics.

        Returns:
            str: A formatted summary of the model
        """
        params = self.get_params()
        metrics = self._metadata.metrics

        summary_lines = [
            f"Model: {self.__class__.__name__}",
            f"Created: {self._metadata.created_at}",
            f"Last Updated: {self._metadata.updated_at or 'Never'}\n",
            "Parameters:",
        ]

        for param, value in params.items():
            summary_lines.append(f"  - {param}: {value}")

        if metrics:
            summary_lines.append("\nMetrics:")
            for metric, value in metrics.items():
                # Check if value is a number before formatting
                if isinstance(value, (int, float)):
                    summary_lines.append(f"  - {metric}: {value:.6f}")
                else:
                    summary_lines.append(f"  - {metric}: {value}")

        summary_str = "\n".join(summary_lines)
        logger.info("Model summary generated")
        return summary_str

    @log_execution
    def save_model(self, file_path: str) -> None:
        """
        Save the model using pickle.

        Args:
            file_path: Path where the model will be saved
        """
        try:
            # Update the saved_at metadata
            self._metadata.saved_at = datetime.now().isoformat()

            with open(file_path, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            logger.exception(f"Error saving model to {file_path}: {e}")
            raise

    @staticmethod
    @log_execution
    def load_model(file_path: str) -> BaseModel:
        """
        Load the model from disk.

        Args:
            file_path: Path from which to load the model

        Returns:
            BaseModel: The loaded model instance
        """
        try:
            with open(file_path, "rb") as f:
                model = pickle.load(f)  # noqa: S301

            # Update the loaded_at metadata
            if hasattr(model, "_metadata"):
                model._metadata.loaded_at = datetime.now().isoformat()

            logger.info(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            logger.exception(f"Error loading model from {file_path}: {e}")
            raise

    @abstractmethod
    def generate_samples(self, n_samples: int = 10, **kwargs) -> pd.DataFrame:
        """
        Generate sample predictions, useful for uncertainty quantification.

        Args:
            n_samples: Number of sample predictions to generate
            **kwargs: Additional parameters for sample generation

        Returns:
            DataFrame: Generated samples
        """

    def validate_data(self, data: pd.DataFrame, expected_columns: list[str] | None = None) -> bool:
        """
        Validate that input data meets expected requirements.

        Args:
            data: DataFrame to validate
            expected_columns: Optional list of required columns

        Returns:
            bool: True if data is valid, raises exception otherwise
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")

        if expected_columns:
            missing_cols = set(expected_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        if data.empty:
            raise ValueError("Empty DataFrame provided")

        return True

    def cleanup(self) -> None:
        """
        Clean up resources used by the model.
        """
        logger.info("Cleaning up model resources")

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (type: {self._metadata.model_type})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.get_params()})"
