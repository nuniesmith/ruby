"""
Weighted Moving Average (WMA) indicator module.

This indicator gives higher weights to more recent prices and decreasing weights to older prices,
making it more responsive to recent price changes than a Simple Moving Average.
"""

import numpy as np


class WMA:
    """
    Weighted Moving Average indicator.

    The WMA assigns more weight to recent prices and less weight to older prices.
    This makes it more responsive to recent price movements compared to a simple
    moving average (SMA).

    Parameters
    ----------
    period : int
        The number of periods to use for the WMA calculation. Default is 14.
    """

    def __init__(self, period=14):
        """
        Initialize the WMA indicator.

        Parameters
        ----------
        period : int
            The number of periods to use for the WMA calculation. Default is 14.
        """
        self.period = period
        self.name = f"Weighted Moving Average ({period})"
        self.description = "Assigns more weight to recent prices, making it more responsive to new data."
        self.category = "trend"

    def calculate(self, data, price_column="close"):
        """
        Calculate the Weighted Moving Average.

        Parameters
        ----------
        data : pd.DataFrame
            The OHLCV data to use for the calculation.
        price_column : str, optional
            The column to use for price data. Default is 'close'.

        Returns
        -------
        pd.DataFrame
            The input dataframe with WMA column added.
        """
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in dataframe")

        # Create column name
        column_name = f"wma_{self.period}"
        if price_column != "close":
            column_name += f"_{price_column}"

        # Get the price series
        prices = data[price_column].values

        # Calculate weights - linear weights from 1 to period
        weights = np.arange(1, self.period + 1)

        # Calculate the WMA
        wma_values = np.zeros_like(prices)

        # For the first period-1 values, calculate partial WMAs
        for i in range(1, self.period):
            if i == 0:
                wma_values[i] = prices[i]
            else:
                curr_weights = weights[-i:]
                weight_sum = np.sum(curr_weights)
                wma_values[i] = np.sum(prices[max(0, i - self.period + 1) : i + 1] * curr_weights) / weight_sum

        # Calculate WMA for the rest of the series
        weight_sum = np.sum(weights)
        for i in range(self.period - 1, len(prices)):
            price_slice = prices[i - self.period + 1 : i + 1]
            wma_values[i] = np.sum(price_slice * weights) / weight_sum

        # Add the WMA to the dataframe
        data[column_name] = wma_values

        return data

    def get_signal(self, data):
        """
        Generate trading signals based on WMA.

        For this simple implementation, we'll just indicate when price crosses above or
        below the WMA line.

        Parameters
        ----------
        data : pd.DataFrame
            The OHLCV data with WMA values.

        Returns
        -------
        dict
            Dictionary with buy, sell, and exit signals.
        """
        column_name = f"wma_{self.period}"
        if column_name not in data.columns:
            self.calculate(data)

        # Get the last row
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest

        # Get signal based on price crossing the WMA
        signal = {"buy": False, "sell": False, "exit_long": False, "exit_short": False}

        # Generate signals for the last candle
        if latest["close"] > latest[column_name] and prev["close"] <= prev[column_name]:
            signal["buy"] = True
            signal["exit_short"] = True
        elif latest["close"] < latest[column_name] and prev["close"] >= prev[column_name]:
            signal["sell"] = True
            signal["exit_long"] = True

        return signal
