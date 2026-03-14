"""
Average True Range (ATR) indicator.
"""

from typing import Any

import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator


@register_indicator
class ATR(Indicator):
    """
    Average True Range (ATR) indicator.

    ATR measures market volatility by decomposing the entire range of an asset
    price for a specific period. It is a useful tool for position sizing and stop-loss placement.
    """

    def __init__(self, name: str = "ATR", params: dict[str, Any] | None = None):
        """
        Initialize the ATR indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - period: The period for the ATR calculation, defaults to 14.
                - method: Smoothing method ('sma' or 'ema'), defaults to 'sma'.
        """
        super().__init__(name, params)
        self.period = self.params.get("period", 14)
        self.method = self.params.get("method", "sma").lower()

        if self.method not in ("sma", "ema"):
            raise ValueError("Method must be 'sma' or 'ema'")

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the ATR based on the input data.

        Args:
            data: Input dataframe containing market data.

        Returns:
            DataFrame with ATR values.
        """
        # Calculate True Range series
        high_low = data["high"] - data["low"]
        high_close_prev = abs(data["high"] - data["close"].shift(1))
        low_close_prev = abs(data["low"] - data["close"].shift(1))

        # True Range is the maximum of the three calculations
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate ATR based on the specified method
        if self.method == "sma":
            atr = tr.rolling(window=self.period).mean()
        else:  # 'ema'
            atr = tr.ewm(span=self.period, adjust=False).mean()

        # Calculate Normalized ATR (ATR divided by close price)
        normalized_atr = atr / data["close"] * 100  # As percentage

        # Create result DataFrame
        result = pd.DataFrame(
            {f"{self.name}_{self.period}": atr, f"{self.name}_{self.period}_normalized": normalized_atr},
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
        return ["high", "low", "close"]
