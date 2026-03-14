"""
Base classifier class for all classification models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from lib.model.base.estimator import Estimator

if TYPE_CHECKING:
    import pandas as pd


class Classifier(Estimator):
    """
    Base class for all classifier models.

    Classifiers are estimators that predict discrete classes or labels.
    This class provides a common interface for all classifier models.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        """
        Initialize the classifier with a name and parameters.

        Args:
            name: The name of the classifier.
            params: Optional parameters for the classifier.
        """
        super().__init__(name, params)
        self._classes: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Classifier:  # type: ignore[override]
        """
        Fit the classifier to the data.

        Args:
            X: Features.
            y: Target classes.

        Returns:
            Self for method chaining.
        """
        # Store the unique classes
        self._classes = np.unique(y)
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | pd.Series:  # type: ignore[override]
        """
        Predict class labels for the input samples.

        Args:
            X: Features.

        Returns:
            Predicted class labels.
        """

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | pd.DataFrame:
        """
        Predict class probabilities for the input samples.

        Args:
            X: Features.

        Returns:
            Predicted class probabilities.
        """
        raise NotImplementedError("Probability prediction not implemented for this classifier")

    @property
    def classes(self) -> np.ndarray | None:
        """
        Get the class labels.

        Returns:
            Array of class labels or None if not fitted.
        """
        return self._classes

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the classifier to a dictionary.

        Returns:
            Dictionary representation of the classifier.
        """
        result = super().to_dict()
        if self._classes is not None:
            result["classes"] = self._classes.tolist()
        return result
