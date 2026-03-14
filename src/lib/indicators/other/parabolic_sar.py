import numpy as np
import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ParabolicSARIndicator:
    """
    Parabolic SAR (Stop and Reverse) Indicator.

    This indicator calculates the Parabolic SAR values for a given series of high and low prices.
    It maintains an internal history of market data and provides methods to update the indicator
    with new data points as well as to apply the indicator to a complete DataFrame.
    """

    REQUIRED_COLUMNS = ["High", "Low"]

    def __init__(self, step: float = 0.02, max_step: float = 0.2):
        """
        Initialize the Parabolic SAR Indicator.

        Args:
            step (float): The step increment for the acceleration factor.
            max_step (float): The maximum limit for the acceleration factor.
        """
        self.logger = logger
        self.step = step
        self.max_step = max_step
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        self.current_value = None

    def _calculate_sar(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parabolic SAR for the provided DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

        Returns:
            pd.Series: Series of calculated Parabolic SAR values.
        """
        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

        self.logger.info("calculating_parabolic_sar", step=self.step, max_step=self.max_step)

        # Initialize SAR array and key variables
        sar = np.zeros(len(data))
        trend = 1  # 1 for uptrend, -1 for downtrend
        ep = data["Low"].iloc[0]  # Extreme Point; for an uptrend, EP is the highest high
        af = self.step  # Initial acceleration factor

        for i in range(1, len(data)):
            prev_sar = sar[i - 1]
            # Compute current SAR
            sar[i] = prev_sar + af * (ep - prev_sar)

            if trend == 1:  # Uptrend logic
                if data["High"].iloc[i] > ep:
                    ep = data["High"].iloc[i]
                    af = min(af + self.step, self.max_step)
                if data["Low"].iloc[i] < sar[i]:
                    # Switch to downtrend
                    trend = -1
                    sar[i] = ep  # Reset SAR at the reversal
                    ep = data["Low"].iloc[i]
                    af = self.step
            else:  # Downtrend logic
                if data["Low"].iloc[i] < ep:
                    ep = data["Low"].iloc[i]
                    af = min(af + self.step, self.max_step)
                if data["High"].iloc[i] > sar[i]:
                    # Switch to uptrend
                    trend = 1
                    sar[i] = ep  # Reset SAR at the reversal
                    ep = data["High"].iloc[i]
                    af = self.step

        self.logger.info("parabolic_sar_calculated_successfully")
        return pd.Series(sar, index=data.index)

    def update(self, data_point: dict) -> float | None:
        """
        Update the Parabolic SAR indicator with a new market data point.

        Args:
            data_point (dict): Market data containing 'high' and 'low'.

        Returns:
            float or None: The latest Parabolic SAR value, or None if update fails.
        """
        try:
            high = data_point.get("high")
            low = data_point.get("low")

            if high is None or low is None:
                self.logger.warning("missing_required_fields", indicator="parabolic_sar", required=["high", "low"])
                return None

            # Append new data point
            new_row = pd.DataFrame([{"High": high, "Low": low}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if not self.history_df.empty:
                sar_series = self._calculate_sar(self.history_df.copy())
                self.current_value = sar_series.iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error("parabolic_sar_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Parabolic SAR indicator to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

        Returns:
            pd.DataFrame: A new DataFrame with a 'ParabolicSAR' column added.
        """
        try:
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_cols)}")

            self.logger.info("applying_parabolic_sar_to_dataframe")
            data = data.copy()
            data["ParabolicSAR"] = self._calculate_sar(data)
            self.logger.info("parabolic_sar_added_successfully")
            return data

        except Exception as e:
            self.logger.error("error_applying_parabolic_sar", error=str(e), exc_info=True)
            raise
