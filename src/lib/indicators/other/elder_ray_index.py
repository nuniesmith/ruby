from typing import Any

import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ElderRayIndexIndicator:
    """
    Elder Ray Index Indicator.

    This indicator calculates Bull Power and Bear Power using the exponential moving average (EMA)
    of the closing prices. It maintains an internal history of market data points and provides methods
    to update the calculation with new data points and to apply the calculation to a complete DataFrame.
    """

    # Define required columns for validation
    REQUIRED_COLUMNS = ["High", "Low", "Close"]

    def __init__(self, fast_period: int = 14):
        """
        Initialize the Elder Ray Index Indicator.

        Args:
            fast_period (int): The period used for calculating the EMA of the closing prices.
                               Defaults to 14.
        """
        self.logger = logger
        self.fast_period = fast_period if fast_period > 0 else 14
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        # current_value holds the most recent calculation results as a dictionary:
        # {'bull_power': <value>, 'bear_power': <value>}
        self.current_value: dict[str, Any] = {}

    def _calculate_eri(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Elder Ray Index using the given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: DataFrame containing 'BullPower' and 'BearPower' columns.

        Raises:
            ValueError: If required columns are missing.
        """
        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

        self.logger.info("calculating_elder_ray_index", ema_period=self.fast_period)
        # Calculate the EMA of the closing prices
        ema = data["Close"].ewm(span=self.fast_period, adjust=False).mean()
        # Calculate Bull Power and Bear Power
        bull_power = data["High"] - ema
        bear_power = data["Low"] - ema

        self.logger.info("elder_ray_index_calculated")
        return pd.DataFrame({"BullPower": bull_power, "BearPower": bear_power})

    def update(self, data_point: dict) -> dict | None:
        """
        Update the Elder Ray Index with a new data point.

        Args:
            data_point (dict): Market data point containing keys 'high', 'low', and 'close'.

        Returns:
            dict or None: A dictionary with keys 'bull_power' and 'bear_power' containing the latest
                          computed values, or None if the update fails.
        """
        try:
            # Extract required values
            high = data_point.get("high")
            low = data_point.get("low")
            close = data_point.get("close")

            if high is None or low is None or close is None:
                self.logger.warning(
                    "missing_required_fields", indicator="elder_ray_index", required_fields=["high", "low", "close"]
                )
                return None

            # Append the new data point to the history DataFrame
            new_row = pd.DataFrame([{"High": high, "Low": low, "Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            # Calculate the indicator if there is data available
            if len(self.history_df) > 0:
                eri_df = self._calculate_eri(self.history_df.copy())
                self.current_value = {
                    "bull_power": eri_df["BullPower"].iloc[-1],
                    "bear_power": eri_df["BearPower"].iloc[-1],
                }
            else:
                self.current_value = {}

            return self.current_value

        except Exception as e:
            self.logger.error("elder_ray_index_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Elder Ray Index calculation to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: A new DataFrame with additional columns 'BullPower' and 'BearPower'.

        Raises:
            RuntimeError: If the calculation fails.
        """
        try:
            data = data.copy()
            eri_df = self._calculate_eri(data)
            data["BullPower"] = eri_df["BullPower"]
            data["BearPower"] = eri_df["BearPower"]
            return data
        except Exception as e:
            self.logger.error("elder_ray_index_apply_failed", error=str(e), exc_info=True)
            raise RuntimeError("Failed to apply Elder Ray Index indicator.") from e
