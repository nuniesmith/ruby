import numpy as np


class VolumeZoneOscillator:
    """
    Volume Zone Oscillator (VZO) indicator module.

    The Volume Zone Oscillator measures volume flowing into and out of a security.
    It oscillates between +100% and -100%, with readings above +5% indicating bullish volume
    and readings below -5% indicating bearish volume.
    """

    def __init__(self, period=14):
        self.period = period

    def calculate(self, df):
        """
        Calculate the Volume Zone Oscillator.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing 'Close' and 'Volume' columns

        Returns:
        --------
        pandas.Series
            Volume Zone Oscillator values
        """
        # Calculate price change
        close_diff = df["Close"].diff()

        # Assign volume based on price direction
        positive_volume = df["Volume"].where(close_diff > 0, 0)
        negative_volume = df["Volume"].where(close_diff < 0, 0)

        # Calculate total volume over the period
        total_volume = df["Volume"].rolling(window=self.period).sum()

        # Avoid division by zero
        total_volume = total_volume.replace(0, np.nan)

        # Calculate VZO: 100 * (positive_volume_sum - negative_volume_sum) / total_volume_sum
        vzo = (
            100
            * (positive_volume.rolling(window=self.period).sum() - negative_volume.rolling(window=self.period).sum())
            / total_volume
        )

        return vzo
