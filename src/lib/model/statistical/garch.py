"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models.
"""

from typing import Any

import numpy as np
import pandas as pd

try:
    import arch
    from arch import arch_model

    HAS_ARCH = True
except ImportError:
    arch = None  # type: ignore[assignment]
    arch_model = None  # type: ignore[assignment]
    HAS_ARCH = False

from lib.model._shims import ModelError
from lib.model.base.estimator import Estimator
from lib.model.registry import register_model


@register_model
class GARCH(Estimator):
    """
    Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.

    GARCH models are used to estimate the volatility of financial time series
    and are particularly useful for modeling periods of turbulence and stability.
    """

    def __init__(self, name: str = "GARCH", params: dict[str, Any] | None = None):
        """
        Initialize the GARCH model.

        Args:
            name: The name of the model.
            params: Parameters for the model.
                - p: Order of the GARCH terms, defaults to 1.
                - q: Order of the ARCH terms, defaults to 1.
                - o: Order of the asymmetric terms, defaults to 0.
                - vol: Volatility model type ('GARCH', 'EGARCH', 'TARCH', etc.), defaults to 'GARCH'.
                - dist: Error distribution ('normal', 'studentst', 'skewstudent'), defaults to 'normal'.
                - mean: Mean model ('Zero', 'Constant', 'AR', 'ARX', etc.), defaults to 'Constant'.
                - lags: Number of lags if using AR mean model, defaults to 1.
                - scale: Whether to scale the data before fitting, defaults to True.
        """
        super().__init__(name, params)
        self.p = self.params.get("p", 1)
        self.q = self.params.get("q", 1)
        self.o = self.params.get("o", 0)
        self.vol = self.params.get("vol", "GARCH")
        self.dist = self.params.get("dist", "normal")
        self.mean = self.params.get("mean", "Constant")
        self.lags = self.params.get("lags", 1)
        self.scale = self.params.get("scale", True)

        # Initialize model
        self._model: Any | None = None
        self._results: Any | None = None
        self._scaler: float = 1.0
        self._last_data: Any | None = None

    def fit(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | pd.Series, y: pd.Series | np.ndarray | None = None
    ) -> "GARCH":
        """
        Fit the GARCH model to the data.

        Args:
            X: Time series data of returns. If y is None, this is treated as the target variable.
            y: Not used, maintained for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ModelError: If there's an error during model fitting.
        """
        try:
            # Handle different input types
            if isinstance(X, pd.DataFrame):
                if X.shape[1] != 1:
                    raise ValueError("X must be a 1-dimensional array or DataFrame with one column")
                returns = X.iloc[:, 0]
            elif isinstance(X, pd.Series):
                returns = X
            else:
                returns = pd.Series(X)

            # Scale data if specified
            if self.scale:
                returns_std = returns.std()
                if returns_std > 0:
                    returns = returns / returns_std
                    self._scaler = returns_std
                else:
                    self._scaler = 1.0
            else:
                self._scaler = 1.0

            # Store the last data points for forecasting
            self._last_data = returns.copy()

            # Create and fit the model
            self._model = arch_model(  # type: ignore[operator]
                returns,
                vol=self.vol,
                p=self.p,
                q=self.q,
                o=self.o,
                dist=self.dist,
                mean=self.mean,
                lags=self.lags if self.mean == "AR" else None,
            )

            self._results = self._model.fit(disp="off")
            self._is_fitted = True

            return self

        except Exception as e:
            raise ModelError(f"Error fitting GARCH model: {str(e)}") from e

    def predict(  # type: ignore[override]
        self, X: pd.DataFrame | np.ndarray | None = None, horizon: int = 1, method: str = "analytic"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted GARCH model.

        Args:
            X: Exogenous variables for prediction (if using ARX model). Optional.
            horizon: Number of steps to forecast, defaults to 1.
            method: Forecast method ('analytic', 'simulation', 'bootstrap'), defaults to 'analytic'.

        Returns:
            Tuple of (mean predictions, volatility predictions).

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict().")

        try:
            assert self._results is not None
            # Make forecasts
            forecasts = self._results.forecast(horizon=horizon, method=method, reindex=False)

            # Extract mean and variance forecasts
            mean_forecast = forecasts.mean.values[-1]
            var_forecast = forecasts.variance.values[-1]
            vol_forecast = np.sqrt(var_forecast)

            # Rescale if necessary
            if self.scale and self._scaler != 1.0:
                mean_forecast = mean_forecast * self._scaler
                vol_forecast = vol_forecast * self._scaler

            return mean_forecast, vol_forecast

        except Exception as e:
            raise ModelError(f"Error predicting with GARCH model: {str(e)}") from e

    def predict_volatility(self, horizon: int = 1, method: str = "analytic") -> np.ndarray:
        """
        Predict volatility using the fitted GARCH model.

        Args:
            horizon: Number of steps to forecast, defaults to 1.
            method: Forecast method ('analytic', 'simulation', 'bootstrap'), defaults to 'analytic'.

        Returns:
            Predicted volatility values.

        Raises:
            ModelError: If the model is not fitted or there's an error during prediction.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before predict_volatility().")

        try:
            assert self._results is not None
            # Make forecasts
            forecasts = self._results.forecast(horizon=horizon, method=method, reindex=False)

            # Extract variance forecasts and convert to volatility
            var_forecast = forecasts.variance.values[-1]
            vol_forecast = np.sqrt(var_forecast)

            # Rescale if necessary
            if self.scale and self._scaler != 1.0:
                vol_forecast = vol_forecast * self._scaler

            return vol_forecast

        except Exception as e:
            raise ModelError(f"Error predicting volatility with GARCH model: {str(e)}") from e

    def simulate(self, horizon: int = 100, n_simulations: int = 1000, x: np.ndarray | None = None) -> pd.DataFrame:
        """
        Simulate future paths from the fitted GARCH model.

        Args:
            horizon: Number of steps to simulate, defaults to 100.
            n_simulations: Number of simulations, defaults to 1000.
            x: Exogenous variables for simulation (if using ARX model). Optional.

        Returns:
            DataFrame with simulated paths.

        Raises:
            ModelError: If the model is not fitted or there's an error during simulation.
        """
        if not self._is_fitted:
            raise ModelError("Model not fitted. Call fit() before simulate().")

        try:
            assert self._results is not None
            # Make simulations
            simulation = self._results.forecast(horizon=horizon, method="simulation", simulations=n_simulations, x=x)

            # Extract simulated values
            sim_values = simulation.simulations.values

            # Rescale if necessary
            if self.scale and self._scaler != 1.0:
                sim_values = sim_values * self._scaler

            # Create DataFrame with simulation results
            sim_df = pd.DataFrame(sim_values.reshape(n_simulations, horizon))

            return sim_df

        except Exception as e:
            raise ModelError(f"Error simulating with GARCH model: {str(e)}") from e

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
                    {
                        "results": self._results,
                        "params": self.params,
                        "name": self.name,
                        "last_data": self._last_data,
                        "scaler": self._scaler,
                    },
                    f,
                )
        except Exception as e:
            raise ModelError(f"Error saving GARCH model: {str(e)}") from e

    def load(self, path: str) -> "GARCH":
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
            self._scaler = data.get("scaler", 1.0)

            # Update parameters
            self.p = self.params.get("p", 1)
            self.q = self.params.get("q", 1)
            self.o = self.params.get("o", 0)
            self.vol = self.params.get("vol", "GARCH")
            self.dist = self.params.get("dist", "normal")
            self.mean = self.params.get("mean", "Constant")
            self.lags = self.params.get("lags", 1)
            self.scale = self.params.get("scale", True)

            self._is_fitted = self._results is not None
            return self

        except Exception as e:
            raise ModelError(f"Error loading GARCH model: {str(e)}") from e
