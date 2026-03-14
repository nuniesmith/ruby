"""
Moving average indicators.
"""

from typing import Any

import numpy as np
import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator


@register_indicator
class SMA(Indicator):
    """
    Simple Moving Average (SMA) indicator.

    Calculates the arithmetic mean of a price over a specified period.
    """

    def __init__(self, name: str = "SMA", params: dict[str, Any] | None = None):
        """
        Initialize the SMA indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The lookback period for the moving average.
                - column: The column to calculate SMA on, defaults to 'close'.
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.column = self.params.get("column", "close")

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the SMA based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: Unused; the column is determined by self.column from params.

        Returns:
            DataFrame with SMA values.
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")

        sma = data[self.column].rolling(window=self.period).mean()
        return pd.DataFrame({f"{self.name}_{self.period}": sma}, index=data.index)

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["close"]


@register_indicator
class EMA(Indicator):
    """
    Exponential Moving Average (EMA) indicator.

    Applies more weight to recent prices compared to SMA.
    """

    def __init__(self, name: str = "EMA", params: dict[str, Any] | None = None):
        """
        Initialize the EMA indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The lookback period for the moving average.
                - column: The column to calculate EMA on, defaults to 'close'.
                - alpha: Optional smoothing factor, defaults to 2/(period+1).
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.column = self.params.get("column", "close")
        self.alpha = self.params.get("alpha", 2 / (self.period + 1))

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the EMA based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: Unused; the column is determined by self.column from params.

        Returns:
            DataFrame with EMA values.
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")

        ema = data[self.column].ewm(span=self.period, adjust=False, alpha=self.alpha).mean()

        return pd.DataFrame({f"{self.name}_{self.period}": ema}, index=data.index)

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["close"]


@register_indicator
class WMA(Indicator):
    """
    Weighted Moving Average (WMA) indicator.

    Applies more weight to recent prices using a linear weighting scheme.
    """

    def __init__(self, name: str = "WMA", params: dict[str, Any] | None = None):
        """
        Initialize the WMA indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The lookback period for the moving average.
                - column: The column to calculate WMA on, defaults to 'close'.
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.column = self.params.get("column", "close")

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the WMA based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: Unused; the column is determined by self.column from params.

        Returns:
            DataFrame with WMA values.
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")

        weights = np.arange(1, self.period + 1)
        wma = (
            data[self.column].rolling(window=self.period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        )

        return pd.DataFrame({f"{self.name}_{self.period}": wma}, index=data.index)

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["close"]


@register_indicator
class VWAP(Indicator):
    """
    Volume-Weighted Average Price (VWAP) indicator.

    Calculates the average price weighted by volume.
    """

    def __init__(self, name: str = "VWAP", params: dict[str, Any] | None = None):
        """
        Initialize the VWAP indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The lookback period for the VWAP, defaults to None (all data).
        """
        super().__init__(name, params)
        self.period = self.params.get("period", None)

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the VWAP based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: Unused; VWAP uses high, low, close, and volume columns.

        Returns:
            DataFrame with VWAP values.
        """
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        volume_price = typical_price * data["volume"]

        if self.period is None:
            # Cumulative VWAP
            cumulative_volume = data["volume"].cumsum()
            cumulative_volume_price = volume_price.cumsum()
            vwap = cumulative_volume_price / cumulative_volume
        else:
            # Rolling VWAP
            rolling_volume = data["volume"].rolling(window=self.period).sum()
            rolling_volume_price = volume_price.rolling(window=self.period).sum()
            vwap = rolling_volume_price / rolling_volume

        result_name = f"{self.name}"
        if self.period is not None:
            result_name += f"_{self.period}"

        return pd.DataFrame({result_name: vwap}, index=data.index)
