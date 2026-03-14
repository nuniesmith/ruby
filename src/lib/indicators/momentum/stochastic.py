"""
Stochastic oscillator indicators.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator


@register_indicator
class Stochastic(Indicator):
    """
    Stochastic Oscillator indicator.

    A momentum indicator that compares a closing price to its price range over a period.
    It consists of two lines: %K (fast stochastic) and %D (slow stochastic).
    """

    def __init__(self, name: str = "Stochastic", params: dict[str, Any] | None = None):
        """
        Initialize the Stochastic Oscillator indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - k_period: The lookback period for %K calculation, defaults to 14.
                - d_period: The smoothing period for %D calculation, defaults to 3.
                - smooth_k: The smoothing period for %K, defaults to 1 (no smoothing).
        """
        super().__init__(name, params)
        self.k_period = self.params.get("k_period", 14)
        self.d_period = self.params.get("d_period", 3)
        self.smooth_k = self.params.get("smooth_k", 1)

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the Stochastic Oscillator based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: The price column to use (unused; Stochastic uses high/low/close).

        Returns:
            DataFrame with Stochastic Oscillator values (%K and %D).
        """
        # Get the highest high and lowest low over the k_period
        high_k = data["high"].rolling(window=self.k_period).max()
        low_k = data["low"].rolling(window=self.k_period).min()

        # Calculate raw %K
        raw_k = 100 * ((data["close"] - low_k) / (high_k - low_k))

        # Apply smoothing to %K if required
        k = raw_k.rolling(window=self.smooth_k).mean() if self.smooth_k > 1 else raw_k

        # Calculate %D (SMA of %K)
        d = k.rolling(window=self.d_period).mean()

        # Create result DataFrame
        result = pd.DataFrame({f"{self.name}_K": k, f"{self.name}_D": d}, index=data.index)

        return result

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["high", "low", "close"]


@register_indicator
class StochasticRSI(Indicator):
    """
    Stochastic RSI indicator.

    Applies the Stochastic Oscillator formula to RSI values instead of price data.
    This creates a more sensitive indicator that oscillates between 0 and 100.
    """

    def __init__(self, name: str = "StochasticRSI", params: dict[str, Any] | None = None):
        """
        Initialize the Stochastic RSI indicator.

        Args:
            name: The name of the indicator.
            params: Parameters for the indicator.
                - rsi_period: The period for RSI calculation, defaults to 14.
                - stoch_period: The period for Stochastic calculation, defaults to 14.
                - k_period: The smoothing period for %K, defaults to 3.
                - d_period: The smoothing period for %D, defaults to 3.
        """
        super().__init__(name, params)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.stoch_period = self.params.get("stoch_period", 14)
        self.k_period = self.params.get("k_period", 3)
        self.d_period = self.params.get("d_period", 3)

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        Calculate the Stochastic RSI based on the input data.

        Args:
            data: Input dataframe containing market data.
            price_column: The price column to use (unused; StochasticRSI always uses close).

        Returns:
            DataFrame with Stochastic RSI values (%K and %D).
        """
        # Calculate RSI
        price_diff = data["close"].diff()
        gain = price_diff.copy()
        loss = price_diff.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        avg_gain = cast("pd.Series", gain.rolling(window=self.rsi_period).mean())
        avg_loss = cast("pd.Series", loss.rolling(window=self.rsi_period).mean())
        avg_loss_safe = cast("pd.Series", avg_loss.replace(0, np.nan))
        rs = cast("pd.Series", np.where(avg_loss == 0, 100.0, avg_gain / avg_loss_safe))
        rs = pd.Series(rs.values if hasattr(rs, "values") else rs, index=data.index, dtype=float)
        rsi = cast("pd.Series", 100 - (100 / (1 + rs)))

        # Apply Stochastic formula to RSI
        min_rsi = rsi.rolling(window=self.stoch_period).min()
        max_rsi = rsi.rolling(window=self.stoch_period).max()
        stoch_rsi = 100 * ((rsi - min_rsi) / (max_rsi - min_rsi))

        # Apply smoothing to %K
        k = stoch_rsi.rolling(window=self.k_period).mean()

        # Calculate %D
        d = k.rolling(window=self.d_period).mean()

        # Create result DataFrame
        result = pd.DataFrame({f"{self.name}_K": k, f"{self.name}_D": d}, index=data.index)
        return result

    @classmethod
    def required_columns(cls) -> list[str]:
        """
        List of required columns in the input dataframe.

        Returns:
            List of column names.
        """
        return ["close"]
