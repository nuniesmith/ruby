"""
Bollinger Bands indicator.
"""

from typing import Any

import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator


@register_indicator
class BollingerBands(Indicator):
    """
    Bollinger Bands indicator.

    Bollinger Bands consist of a middle band (SMA) with upper and lower bands
    that are standard deviations away from the middle band. They provide
    information about price volatility and potential overbought/oversold conditions.
    """

    def __init__(self, name: str = "BollingerBands", params: dict[str, Any] | None = None):
        """
        Initialize the Bollinger Bands indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The period for the moving average, defaults to 20.
                - std_dev: Number of standard deviations for the bands, defaults to 2.
                - column: The column to calculate Bollinger Bands on, defaults to 'close'.
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 20)
        self.std_dev = self.params.get("std_dev", 2)
        self.column = self.params.get("column", "close")

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the Bollinger Bands based on the input data.

        Args:
            data: Input dataframe containing market data.

        Returns:
            DataFrame with Bollinger Bands values (middle, upper, and lower bands).
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")

        # Calculate the middle band (SMA)
        middle_band = data[self.column].rolling(window=self.period).mean()

        # Calculate the standard deviation
        std = data[self.column].rolling(window=self.period).std()

        # Calculate the upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)

        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band

        # Calculate %B (position within the bands)
        percent_b = (data[self.column] - lower_band) / (upper_band - lower_band)

        # Create result DataFrame
        result = pd.DataFrame(
            {
                f"{self.name}_middle": middle_band,
                f"{self.name}_upper": upper_band,
                f"{self.name}_lower": lower_band,
                f"{self.name}_bandwidth": bandwidth,
                f"{self.name}_percent_b": percent_b,
            },
            index=data.index,
        )

        return result

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["close"]
