"""
ARIMAModel (AutoRegressive Integrated Moving Average) models.
"""

from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import ModelError

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMAModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAModelX

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    sm = None  # type: ignore[assignment]
    SARIMAModelX = None  # type: ignore[assignment]
    StatsARIMAModel = None  # type: ignore[assignment]

from lib.model.base.estimator import Estimator
from lib.model.registry import register_model


@register_model
class ARIMAModel(Estimator):
    """
    AutoRegressive Integrated Moving Average (ARIMAModel) model.

    ARIMAModel is a popular statistical method for time series forecasting that
    combines autoregressive (AR), differencing (I), and moving average (MA) components.
    """

    def __init__(self, name: str = "ARIMAModel", params: dict[str, Any] | None = None):
        """
        Initialize the ARIMAModel model.

        Args:
            name: The name of the model.
            params: Parameters for the model.
                - order: ARIMAModel order (p, d, q), defaults to (1, 0, 0).
                - seasonal_order: Seasonal order (P, D, Q, s), defaults to None.
                - trend: Trend component, defaults to 'c' (constant).
                - enforce_stationarity: Whether to enforce stationarity, defaults to True.
                - enforce_invertibility: Whether to enforce invertibility, defaults to True.
        """
        super().__init__(name, params)
        self.order = self.params.get("order", (1, 0, 0))
        self.seasonal_order = self.params.get("seasonal_order", None)
        self.trend = self.params.get("trend", "c")
        self.enforce_stationarity = self.params.get("enforce_stationarity", True)
        self.enforce_invertibility = self.params.get("enforce_invertibility", True)

        # Initialize model
        self._model: Any | None = None
        self._results: Any | None = None
        self._last_data: Any | None = None

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | pd.Series, y: pd.Series | np.ndarray | None = None
    ) -> "ARIMAModel":
        """
        Fit the ARIMAModel model to the data.

        Args:
            X: Time series data. If y is None, this is treated as the target variable.
            y: Target values (optional). If provided, X is treated as exogenous variables.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during model fitting.
        """
        if not HAS_STATSMODELS:
            raise ModelError("statsmodels is required for ARIMAModel but is not installed")
        try:
            # Handle different input types
            if y is None:
                # X is the target
                if isinstance(X, pd.DataFrame):
                    if X.shape[1] != 1:
                        raise ValueError("When y is None, X must be a 1-dimensional array or DataFrame with one column")
                    endog = X.iloc[:, 0]
                    exog = None
                elif isinstance(X, pd.Series):
                    endog = X
                    exog = None
                else:
                    endog = X
                    exog = None
            else:
                # X is exogenous, y is the target
                endog = y
                exog = X

            # Store the last data points for forecasting
            if isinstance(endog, pd.Series):
                self._last_data = endog.iloc[-self.order[0] :].values if self.order[0] > 0 else None
            elif isinstance(endog, np.ndarray):
                self._last_data = endog[-self.order[0] :] if self.order[0] > 0 else None

            # Create and fit the model
            if self.seasonal_order:
                # Use SARIMAModelX for seasonal models
                self._model = SARIMAModelX(  # type: ignore[operator]
                    endog=endog,
                    exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )
            else:
                # Use ARIMAModel for non-seasonal models
                self._model = StatsARIMAModel(  # type: ignore[operator]
                    endog=endog,
                    exog=exog,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )

            self._results = self._model.fit()
            self._is_fitted = True

            return self

        except Exception as e:
            raise ModelError(f"Error fitting ARIMAModel model: {str(e)}") from e

    def predict(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | None = None, steps: int = 1
    ) -> np.ndarray:
        """
        Make predictions using the fitted ARIMAModel model.

        Args:
            X: Exogenous variables for prediction (optional).
            steps: Number of steps to forecast, defaults to 1.

        Returns:
            Predicted values.

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict().")

        try:
            assert self._results is not None
            # Make predictions
            forecast = self._results.forecast(steps=steps, exog=X)

            # Convert to numpy array
            if isinstance(forecast, pd.Series):
                return forecast.values
            return forecast

        except Exception as e:
            raise ModelError(f"Error predicting with ARIMAModel model: {str(e)}") from e

    def predict_interval(
        self, X: pd.DataFrame | np.ndarray | None = None, steps: int = 1, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict confidence intervals for the forecast.

        Args:
            X: Exogenous variables for prediction (optional).
            steps: Number of steps to forecast, defaults to 1.
            alpha: Significance level, defaults to 0.05 (95% confidence).

        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds).

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict_interval().")

        try:
            assert self._results is not None
            # Get forecast with confidence intervals
            forecast = self._results.get_forecast(steps=steps, exog=X)
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=alpha)

            # Extract lower and upper bounds
            if isinstance(conf_int, pd.DataFrame):
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
                mean = mean_forecast.values
            else:
                lower = conf_int[:, 0]
                upper = conf_int[:, 1]
                mean = mean_forecast

            return mean, lower, upper

        except Exception as e:
            raise ModelError(f"Error predicting intervals with ARIMAModel model: {str(e)}") from e

    def get_summary(self) -> str:
        """
        Get the summary of the fitted model.

        Returns:
            String representation of the model summary.

        Raises:
            ModelError: If the model is not fitted.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before get_summary().")

        assert self._results is not None
        return self._results.summary().as_text()

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
            import pickle

            with open(path, "wb") as f:
                pickle.dump(
                    {"results": self._results, "params": self.params, "name": self.name, "last_data": self._last_data},
                    f,
                )
        except Exception as e:
            raise ModelError(f"Error saving ARIMAModel model: {str(e)}") from e

    def load(self, path: str) -> "ARIMAModel":
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
            import pickle

            with open(path, "rb") as f:
                data = pickle.load(f)

            self._results = data.get("results")
            self.params = data.get("params", {})
            self.name = data.get("name", self.name)
            self._last_data = data.get("last_data")

            # Update parameters
            self.order = self.params.get("order", (1, 0, 0))
            self.seasonal_order = self.params.get("seasonal_order", None)
            self.trend = self.params.get("trend", "c")
            self.enforce_stationarity = self.params.get("enforce_stationarity", True)
            self.enforce_invertibility = self.params.get("enforce_invertibility", True)

            self._is_fitted = self._results is not None
            return self

        except Exception as e:
            raise ModelError(f"Error loading ARIMAModel model: {str(e)}") from e


class ConcreteARIMAModel(ARIMAModel):
    def __init__(self, name: str = "ARIMAModel", params: dict[str, Any] | None = None):
        super().__init__(name, params)  # pyright: ignore[reportCallIssue]

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | pd.Series, y: pd.Series | np.ndarray | None = None
    ) -> "ARIMAModel":
        return super().fit(X, y)  # type: ignore

    def predict(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | None = None, steps: int = 1
    ) -> np.ndarray:
        return super().predict(X, steps)

    def predict_interval(
        self, X: pd.DataFrame | np.ndarray | None = None, steps: int = 1, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().predict_interval(X, steps, alpha)

    def get_summary(self) -> str:
        return super().get_summary()

    def save(self, path: str) -> None:
        return super().save(path)

    def load(self, path: str) -> "ARIMAModel":
        return super().load(path)
