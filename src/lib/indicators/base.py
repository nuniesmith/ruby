"""
Base class for technical indicators.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from lib.indicators._shims import Component, validate_dataframe


class Indicator(Component, ABC):
    """
    Base class for all technical indicators.

    Indicators take market data (e.g., OHLCV data) and calculate technical values
    that can be used for analysis and trading decisions.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        """
        Initialize the indicator with a name and parameters.

        Args:
            name: The name of the indicator.
            params: Optional parameters for the indicator.
        """
        super().__init__({"name": name})
        self.name = name
        self.params = params or {}
        self._values = None

    @abstractmethod
    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the indicator values based on the input data.

        Args:
            data: Input dataframe containing market data.

        Returns:
            DataFrame with indicator values.
        """
        pass

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Call the indicator on input data to calculate values.

        Args:
            data: Input dataframe containing market data.

        Returns:
            DataFrame with indicator values.
        """
        validate_dataframe(data, required_columns=self.required_columns())
        self._values = self.calculate(data)
        return self._values

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["open", "high", "low", "close", "volume"]

    @property
    def values(self) -> pd.DataFrame | None:
        """
        Get the calculated indicator values.

        Returns:
            DataFrame with indicator values or None if not calculated yet.
        """
        return self._values

    def reset(self) -> None:
        """
        Reset the indicator's state.
        """
        self._values = None

    def get_value(self, index: int = -1) -> float | dict[str, float]:
        """
        Get the indicator value at the specified index.

        Args:
            index: Index position, defaults to -1 (latest value).

        Returns:
            Indicator value or dictionary of values.
        """
        if self._values is None:
            raise ValueError("Indicator values not calculated yet")

        if isinstance(self._values, pd.DataFrame):
            row = self._values.iloc[index]
            if len(row) == 1:
                return row.iloc[0]
            return row.to_dict()
        else:
            return self._values[index]

    def get_series(self, column: str | None = None) -> pd.Series:
        """
        Get the indicator values as a pandas Series.

        Args:
            column: Optional column name if the indicator has multiple outputs.

        Returns:
            Series of indicator values.
        """
        if self._values is None:
            raise ValueError("Indicator values not calculated yet")

        if column is not None and isinstance(self._values, pd.DataFrame):
            return self._values[column]  # type: ignore[return-value]
        elif isinstance(self._values, pd.DataFrame) and self._values.shape[1] == 1:
            return self._values.iloc[:, 0]
        elif isinstance(self._values, pd.DataFrame):
            raise ValueError(f"Multiple columns available, specify one: {list(self._values.columns)}")
        else:
            return pd.Series(self._values)
