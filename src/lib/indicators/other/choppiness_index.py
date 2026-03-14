import numpy as np


class ChoppinessIndex:
    """
    Choppiness Index (CHOP) indicator module.

    The Choppiness Index is designed to determine if the market is choppy (trading sideways)
    or trending. It ranges from 0 to 100, with readings above 61.8 indicating a choppy market
    and readings below 38.2 indicating a trending market.
    """

    def __init__(self, period=14):
        self.period = period

    def calculate(self, df):
        """
        Calculate the Choppiness Index.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'High', 'Low' columns

        Returns:
        --------
        pandas.Series
            Choppiness Index values
        """
        high_low_range = df["High"] - df["Low"]
        atr = high_low_range.rolling(window=self.period).sum()
        max_high = df["High"].rolling(window=self.period).max()
        min_low = df["Low"].rolling(window=self.period).min()

        # Avoid division by zero and log of zero
        denominator = max_high - min_low
        denominator = denominator.replace(0, np.nan)

        # Calculate CHOP using the formula: 100 * LOG10(SUM(ATR(1)) / (MaxHigh - MinLow)) / LOG10(period)
        chop = 100 * np.log10(atr / denominator) / np.log10(self.period)

        return chop
