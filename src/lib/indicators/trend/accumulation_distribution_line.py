import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ADLineIndicator:
    """
    Accumulation/Distribution Line (ADL) Indicator.

    This indicator calculates the ADL using market data. It accepts new data points
    via the update method, maintains an internal history, and computes the ADL using
    a standardized parameter interface.
    """

    # Define required columns centrally for validation and future parameter flexibility.
    REQUIRED_COLUMNS = ["High", "Low", "Close", "Volume"]

    def __init__(self):
        self.logger = logger
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        self.current_value = None

    def _calculate_adl(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the Accumulation/Distribution Line (ADL) based on input DataFrame.

        Args:
            data (pd.DataFrame): DataFrame with required columns.

        Returns:
            pd.Series: Calculated ADL as a cumulative sum of Money Flow Volume.

        Raises:
            ValueError: If the DataFrame is missing required columns.
            RuntimeError: For any unexpected error during calculation.
        """
        try:
            # Check for required columns
            missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
            if missing_columns:
                raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

            self.logger.info("calculating_adl")

            # Avoid division by zero when high equals low
            zero_range_mask = (data["High"] - data["Low"]) == 0
            mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (data["High"] - data["Low"])
            mfm[zero_range_mask] = 0  # Set multiplier to 0 where range is zero

            mfv = mfm * data["Volume"]
            adl = mfv.cumsum()

            self.logger.info("adl_calculation_successful")
            return adl

        except ValueError as ve:
            self.logger.error("validation_error", error=str(ve))
            raise
        except Exception as e:
            self.logger.exception("adl_calculation_failed", error=str(e))
            raise RuntimeError("Failed to calculate Accumulation/Distribution Line.") from e

    def update(self, data_point: dict) -> float | None:
        """
        Update the indicator with a new data point.

        Args:
            data_point (dict): Market data with keys 'high', 'low', 'close', and 'volume'.

        Returns:
            float: The latest ADL value or None if update fails.
        """
        try:
            # Extract and validate data
            high = data_point.get("high")
            low = data_point.get("low")
            close = data_point.get("close")
            volume = data_point.get("volume")

            if None in (high, low, close, volume):
                self.logger.warning("missing_required_keys", keys=["high", "low", "close", "volume"])
                return None

            # Append new data point
            new_row = pd.DataFrame([{"High": high, "Low": low, "Close": close, "Volume": volume}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            # Calculate ADL using a copy of the history to avoid side effects
            adl_series = self._calculate_adl(self.history_df.copy())
            self.current_value = adl_series.iloc[-1]

            return self.current_value

        except Exception as e:
            self.logger.error("adl_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the ADL indicator calculation to an entire DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame with required columns.

        Returns:
            pd.DataFrame: DataFrame with an added 'ADL' column containing the computed values.
        """
        try:
            self.logger.info("applying_adl_indicator")
            data = data.copy()  # Work on a copy to prevent side effects.
            data["ADL"] = self._calculate_adl(data)
            self.logger.info("adl_column_added_successfully")
            return data
        except Exception as e:
            self.logger.error("adl_apply_failed", error=str(e))
            raise
