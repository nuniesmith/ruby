"""
XGBoost models for regression and classification.
"""

from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import ModelError
from lib.model.base.classifier import Classifier
from lib.model.base.regressor import Regressor
from lib.model.registry import register_model

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    HAS_XGBOOST = False


@register_model
class XGBoostRegressor(Regressor):
    """
    XGBoost regressor for predicting continuous values.

    XGBoost is a powerful gradient boosting algorithm that works well
    for a wide range of regression tasks.
    """

    def __init__(self, name: str = "XGBoostRegressor", params: dict[str, Any] | None = None):
        """
        Initialize the XGBoost regressor.

        Args:
            name: The name of the model.
            params: Parameters for the model. These are passed directly to XGBoost.
                Common parameters include:
                - learning_rate: Step size shrinkage to prevent overfitting, defaults to 0.1.
                - max_depth: Maximum depth of a tree, defaults to 3.
                - n_estimators: Number of trees, defaults to 100.
                - subsample: Subsample ratio of the training instances, defaults to 1.0.
                - colsample_bytree: Subsample ratio of columns, defaults to 1.0.
                - objective: Objective function, defaults to 'reg:squarederror'.
        """
        super().__init__(name, params)

        # Default XGBoost parameters
        self.xgb_params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        # Update with user-provided parameters
        if params:
            self.xgb_params.update(params)

        # Initialize model
        self._model: Any = None
        if HAS_XGBOOST and xgb is not None:
            self._model = xgb.XGBRegressor(**self.xgb_params)

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "XGBoostRegressor":
        """
        Fit the XGBoost regressor to the data.

        Args:
            X: Features.
            y: Target values.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during model fitting.
        """
        try:
            # Extract fit parameters if present
            fit_params = self.params.get("fit_params", {})

            # Fit the model
            self._model.fit(X, y, **fit_params)
            self._is_fitted = True

            return self

        except Exception as e:
            raise ModelError(f"Error fitting XGBoost regressor: {str(e)}") from e

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # type: ignore[override]
        """
        Make predictions using the fitted XGBoost regressor.

        Args:
            X: Features.

        Returns:
            Predicted values.

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict().")

        try:
            return self._model.predict(X)

        except Exception as e:
            raise ModelError(f"Error predicting with XGBoost regressor: {str(e)}") from e

    def feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance scores.

        Returns:
            DataFrame with feature importance scores.

        Raises:
            ModelError: If the model is not fitted.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before feature_importance().")

        # Get feature importances
        importances = self._model.feature_importances_

        # Create DataFrame with feature names if available
        if hasattr(self._model, "feature_names_in_"):
            feature_names = self._model.feature_names_in_
            return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            )
        else:
            return pd.DataFrame(
                {"feature": [f"feature_{i}" for i in range(len(importances))], "importance": importances}
            ).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        """
        Save the fitted model to a file.

        Args:
            path: Path to save the model to.

        Raises:
            ModelError: If the model is not fitted or there's an error during saving.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before save().")

        try:
            self._model.save_model(path)
        except Exception as e:
            raise ModelError(f"Error saving XGBoost regressor: {str(e)}") from e

    def load(self, path: str) -> "XGBoostRegressor":
        """
        Load a fitted model from a file.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during loading.
        """
        try:
            self._model.load_model(path)
            self._is_fitted = True
            return self

        except Exception as e:
            raise ModelError(f"Error loading XGBoost regressor: {str(e)}") from e


@register_model
class XGBoostClassifier(Classifier):
    """
    XGBoost classifier for predicting discrete classes.

    XGBoost is a powerful gradient boosting algorithm that works well
    for a wide range of classification tasks.
    """

    def __init__(self, name: str = "XGBoostClassifier", params: dict[str, Any] | None = None):
        """
        Initialize the XGBoost classifier.

        Args:
            name: The name of the model.
            params: Parameters for the model. These are passed directly to XGBoost.
                Common parameters include:
                - learning_rate: Step size shrinkage to prevent overfitting, defaults to 0.1.
                - max_depth: Maximum depth of a tree, defaults to 3.
                - n_estimators: Number of trees, defaults to 100.
                - subsample: Subsample ratio of the training instances, defaults to 1.0.
                - colsample_bytree: Subsample ratio of columns, defaults to 1.0.
                - objective: Objective function, defaults to 'binary:logistic'.
        """
        super().__init__(name, params)

        # Default XGBoost parameters
        self.xgb_params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "binary:logistic",
            "random_state": 42,
        }

        # Update with user-provided parameters
        if params:
            self.xgb_params.update(params)

        # Initialize model
        self._model: Any = None
        if HAS_XGBOOST and xgb is not None:
            self._model = xgb.XGBClassifier(**self.xgb_params)

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "XGBoostClassifier":
        """
        Fit the XGBoost classifier to the data.

        Args:
            X: Features.
            y: Target classes.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during model fitting.
        """
        try:
            # Store the unique classes
            super().fit(X, y)

            # Extract fit parameters if present
            fit_params = self.params.get("fit_params", {})

            # Fit the model
            self._model.fit(X, y, **fit_params)
            self._is_fitted = True

            return self

        except Exception as e:
            raise ModelError(f"Error fitting XGBoost classifier: {str(e)}") from e

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # type: ignore[override]
        """
        Predict class labels for the input samples.

        Args:
            X: Features.

        Returns:
            Predicted class labels.

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict().")

        try:
            return self._model.predict(X)

        except Exception as e:
            raise ModelError(f"Error predicting with XGBoost classifier: {str(e)}") from e

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input samples.

        Args:
            X: Features.

        Returns:
            Predicted class probabilities.

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict_proba().")

        try:
            return self._model.predict_proba(X)

        except Exception as e:
            raise ModelError(f"Error predicting probabilities with XGBoost classifier: {str(e)}") from e

    def feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance scores.

        Returns:
            DataFrame with feature importance scores.

        Raises:
            ModelError: If the model is not fitted.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before feature_importance().")

        # Get feature importances
        importances = self._model.feature_importances_

        # Create DataFrame with feature names if available
        if hasattr(self._model, "feature_names_in_"):
            feature_names = self._model.feature_names_in_
            return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            )
        else:
            return pd.DataFrame(
                {"feature": [f"feature_{i}" for i in range(len(importances))], "importance": importances}
            ).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        """
        Save the fitted model to a file.

        Args:
            path: Path to save the model to.

        Raises:
            ModelError: If the model is not fitted or there's an error during saving.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before save().")

        try:
            self._model.save_model(path)
        except Exception as e:
            raise ModelError(f"Error saving XGBoost classifier: {str(e)}") from e

    def load(self, path: str) -> "XGBoostClassifier":
        """
        Load a fitted model from a file.

        Args:
            path: Path to load the model from.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during loading.
        """
        try:
            self._model.load_model(path)
            self._is_fitted = True
            return self

        except Exception as e:
            raise ModelError(f"Error loading XGBoost classifier: {str(e)}") from e


class ConcreteXGBoostModel(XGBoostClassifier):
    def __init__(self, name: str = "ConcreteXGBoostModel", params: dict[str, Any] | None = None):
        super().__init__()

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "ConcreteXGBoostModel":
        super().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # type: ignore[override]
        return super().predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return super().predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        return super().feature_importance()

    def save(self, path: str) -> None:
        return super().save(path)

    def load(self, path: str) -> "ConcreteXGBoostModel":
        super().load(path)
        return self
