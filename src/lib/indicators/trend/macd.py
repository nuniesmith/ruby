"""
Moving Average Convergence Divergence (MACD) indicator.
"""

from typing import Any

import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator


@register_indicator
class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """

    def __init__(self, name: str = "MACD", params: dict[str, Any] | None = None):
        """
        Initialize the MACD indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - fast_period: The period for the fast EMA, defaults to 12.
                - slow_period: The period for the slow EMA, defaults to 26.
                - signal_period: The period for the signal line, defaults to 9.
                - column: The column to calculate MACD on, defaults to 'close'.
        """
        super().__init__(name, params)
        self.fast_period = self.params.get("fast_period", 12)
        self.slow_period = self.params.get("slow_period", 26)
        self.signal_period = self.params.get("signal_period", 9)
        self.column = self.params.get("column", "close")

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the MACD based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: The price column to use (ignored; uses self.column instead).

        Returns:
            DataFrame with MACD values (MACD line, signal line, and histogram).
        """
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")

        # Calculate the fast and slow EMAs
        fast_ema = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate the MACD line
        macd_line = fast_ema - slow_ema

        # Calculate the signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate the MACD histogram
        histogram = macd_line - signal_line

        # Create result DataFrame
        result = pd.DataFrame(
            {f"{self.name}_line": macd_line, f"{self.name}_signal": signal_line, f"{self.name}_histogram": histogram},
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
