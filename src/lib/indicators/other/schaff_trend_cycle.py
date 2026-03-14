import numpy as np


class SchaffTrendCycle:
    """
    Schaff Trend Cycle (STC) indicator module.

    The Schaff Trend Cycle is a momentum oscillator based on the MACD and stochastic oscillator.
    It moves between 0 and 100, with readings above 75 indicating overbought conditions
    and readings below 25 indicating oversold conditions.
    """

    def __init__(self, short_ema=12, long_ema=26, stoch_period=10, signal_period=3):
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.stoch_period = stoch_period
        self.signal_period = signal_period

    def calculate(self, df):
        """
        Calculate the Schaff Trend Cycle indicator.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'Close' price column

        Returns:
        --------
        pandas.Series
            Schaff Trend Cycle values
        """
        # Calculate MACD
        short_ema = df["Close"].ewm(span=self.short_ema).mean()
        long_ema = df["Close"].ewm(span=self.long_ema).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=9).mean()
        macd_diff = macd - macd_signal

        # Calculate Stochastic of MACD
        lowest_macd = macd_diff.rolling(self.stoch_period).min()
        highest_macd = macd_diff.rolling(self.stoch_period).max()

        # Avoid division by zero
        range_macd = highest_macd - lowest_macd
        range_macd = range_macd.replace(0, np.nan)

        stc = 100 * ((macd_diff - lowest_macd) / range_macd)

        # Apply EMA smoothing if signal_period is specified
        if self.signal_period > 0:
            stc = stc.ewm(span=self.signal_period).mean()

        return stc
