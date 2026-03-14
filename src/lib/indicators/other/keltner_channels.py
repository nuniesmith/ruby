from typing import Any

import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class KeltnerChannelsIndicator:
    """
    Keltner Channels Indicator.

    Calculates Keltner Channels, which include an upper band, a lower band, and a middle line (EMA).
    The indicator uses an EMA calculated on the closing prices and an ATR-based offset to determine
    the upper and lower bands.

    Attributes:
        period_ema (int): The period for calculating the EMA (and ATR, in this implementation).
        multiplier_atr (float): The multiplier for the ATR to determine the channel offsets.
    """

    REQUIRED_COLUMNS = ["Close", "High", "Low"]

    def __init__(self, period_ema: int = 20, multiplier_atr: float = 2.0):
        """
        Initialize the Keltner Channels Indicator.

        Args:
            period_ema (int): The period used for both the EMA and ATR calculations.
            multiplier_atr (float): The multiplier applied to the ATR for offset calculations.
        """
        if period_ema <= 0:
            raise ValueError("EMA period must be a positive integer.")
        if multiplier_atr <= 0:
            raise ValueError("ATR multiplier must be positive.")

        self.logger = logger
        self.period_ema = period_ema
        self.multiplier_atr = multiplier_atr
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        # current_value will hold the latest channels as a dictionary:
        # {'upper': value, 'lower': value, 'middle': value}
        self.current_value: dict[str, Any] = {}

    def _calculate_keltner_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels based on the provided OHLC data.

        Args:
            data (pd.DataFrame): DataFrame containing 'Close', 'High', and 'Low' columns.

        Returns:
            pd.DataFrame: DataFrame containing the calculated 'UpperBand', 'LowerBand', and 'EMA' columns.

        Raises:
            ValueError: If any required columns are missing.
        """
        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

        self.logger.info(
            "calculating_keltner_channels",
            period_ema=self.period_ema,
            multiplier_atr=self.multiplier_atr,
        )

        # Calculate the Exponential Moving Average (EMA) for the 'Close' prices
        ema = data["Close"].ewm(span=self.period_ema, adjust=False).mean()

        # Calculate True Range components
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift(1)).abs()
        low_close = (data["Low"] - data["Close"].shift(1)).abs()

        # True Range is the maximum of the three measures
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Calculate the Average True Range (ATR) over the given period
        atr = true_range.rolling(window=self.period_ema, min_periods=1).mean()

        # Calculate Upper and Lower Bands
        upper_band = ema + (atr * self.multiplier_atr)
        lower_band = ema - (atr * self.multiplier_atr)

        self.logger.info("keltner_channels_calculated")
        return pd.DataFrame(
            {
                "UpperBand": upper_band,
                "LowerBand": lower_band,
                "EMA": ema,  # The EMA is used as the middle channel
            }
        )

    def update(self, data_point: dict) -> dict | None:
        """
        Update the Keltner Channels indicator with a new market data point.

        Args:
            data_point (dict): Dictionary containing keys 'high', 'low', and 'close'.

        Returns:
            dict or None: The latest Keltner Channels values as a dictionary with keys 'upper', 'lower', and 'middle',
                          or None if the data point is missing required values.
        """
        try:
            # Extract values from the data point
            high = data_point.get("high")
            low = data_point.get("low")
            close = data_point.get("close")

            if high is None or low is None or close is None:
                self.logger.warning(
                    "missing_required_fields",
                    indicator="keltner_channels",
                    required=["high", "low", "close"],
                )
                return None

            # Append the new data point to the history DataFrame
            new_row = pd.DataFrame([{"High": high, "Low": low, "Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            # Only calculate channels if enough data is available (at least equal to period_ema)
            if len(self.history_df) >= self.period_ema:
                kc_df = self._calculate_keltner_channels(self.history_df.copy())
                self.current_value = {
                    "upper": kc_df["UpperBand"].iloc[-1],
                    "lower": kc_df["LowerBand"].iloc[-1],
                    "middle": kc_df["EMA"].iloc[-1],
                }
            else:
                self.logger.debug("insufficient_data", indicator="keltner_channels")
                self.current_value = {}

            return self.current_value

        except Exception as e:
            self.logger.error("keltner_channels_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Keltner Channels indicator to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the required columns ('Close', 'High', 'Low').

        Returns:
            pd.DataFrame: A new DataFrame with 'UpperBand', 'LowerBand', and 'EMA' columns added.

        Raises:
            RuntimeError: If the calculation fails.
        """
        try:
            # Validate that required columns exist
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_cols)}")

            self.logger.info("applying_keltner_channels")
            data = data.copy()
            kc_df = self._calculate_keltner_channels(data)
            data["UpperBand"] = kc_df["UpperBand"]
            data["LowerBand"] = kc_df["LowerBand"]
            data["EMA"] = kc_df["EMA"]
            self.logger.info("keltner_channels_applied")
            return data

        except Exception as e:
            self.logger.error("keltner_channels_apply_failed", error=str(e), exc_info=True)
            raise RuntimeError("Failed to apply Keltner Channels indicator.") from e
