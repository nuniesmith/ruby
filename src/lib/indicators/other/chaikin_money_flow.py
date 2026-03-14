import numpy as np


class ChaikinMoneyFlow:
    """
    Chaikin Money Flow (CMF) indicator module.

    The Chaikin Money Flow measures the amount of Money Flow Volume over a specific period.
    Values above +0.20 indicate strong buying pressure, while values below -0.20 indicate
    strong selling pressure.
    """

    def __init__(self, period=20):
        self.period = period

    def calculate(self, df):
        """
        Calculate the Chaikin Money Flow.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns

        Returns:
        --------
        pandas.Series
            Chaikin Money Flow values
        """
        # Calculate Money Flow Multiplier
        high_low_range = df["High"] - df["Low"]

        # Avoid division by zero
        high_low_range = high_low_range.replace(0, np.nan)

        money_flow_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_range

        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * df["Volume"]

        # Calculate sum of Money Flow Volume and Volume over period
        sum_money_flow_volume = money_flow_volume.rolling(window=self.period).sum()
        sum_volume = df["Volume"].rolling(window=self.period).sum()

        # Avoid division by zero
        sum_volume = sum_volume.replace(0, np.nan)

        # Calculate CMF
        cmf = sum_money_flow_volume / sum_volume

        return cmf
