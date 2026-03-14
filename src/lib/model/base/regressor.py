"""
Base regressor class for all regression models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from lib.model.base.estimator import Estimator

if TYPE_CHECKING:
    import pandas as pd


class Regressor(Estimator):
    """
    Base class for all regressor models.

    Regressors are estimators that predict continuous values.
    This class provides a common interface for all regressor models.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        """
        Initialize the regressor with a name and parameters.

        Args:
            name: The name of the regressor.
            params: Optional parameters for the regressor.
        """
        super().__init__(name, params)

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Regressor:  # type: ignore[override]
        """
        Fit the regressor to the data.

        Args:
            X: Features.
            y: Target values.

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | pd.Series:  # type: ignore[override]
        """
        Predict target values for the input samples.

        Args:
            X: Features.

        Returns:
            Predicted target values.
        """

    def predict_interval(
        self, X: pd.DataFrame | np.ndarray, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict confidence intervals for the input samples.

        Args:
            X: Features.
            confidence: Confidence level, defaults to 0.95 (95%).

        Returns:
            Tuple of (predictions, lower bounds, upper bounds).
        """
        raise NotImplementedError("Interval prediction not implemented for this regressor")

    def score(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        """
        Calculate the coefficient of determination (R^2) for the predictions.

        Args:
            X: Features.
            y: True target values.

        Returns:
            R^2 score.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)

        # Calculate R^2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0  # If all true values are the same

        return 1 - (ss_res / ss_tot)
