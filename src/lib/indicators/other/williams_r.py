import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class WilliamsRIndicator:
    """
    Williams %R Indicator.

    Calculates Williams %R using the formula:

        Williams %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

    where Highest High and Lowest Low are computed over a specified look-back period.
    """

    REQUIRED_COLUMNS = ["High", "Low", "Close"]

    def __init__(self, period: int = 14):
        """
        Initialize the WilliamsRIndicator.

        Args:
            period (int): Look-back period for calculating Williams %R.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.logger = logger
        self.period = period
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        self.current_value = None

    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R for the provided DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
            pd.Series: Series representing the Williams %R values.
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

        self.logger.info("calculating_williams_r", period=self.period)
        highest_high = data["High"].rolling(window=self.period).max()
        lowest_low = data["Low"].rolling(window=self.period).min()
        # Compute Williams %R; note the multiplication by -100 to normalize
        will_r_raw = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)
        will_r = pd.Series(will_r_raw, index=data.index, name=f"Williams_%R_{self.period}")
        self.logger.info("williams_r_calculated_successfully")
        return will_r

    def update(self, data_point: dict) -> float | None:
        """
        Update the Williams %R indicator with a new data point.

        Args:
            data_point (dict): Market data containing 'high', 'low', and 'close'.

        Returns:
            float or None: The latest Williams %R value, or None if there is insufficient data.
        """
        try:
            high = data_point.get("high")
            low = data_point.get("low")
            close = data_point.get("close")
            if high is None or low is None or close is None:
                self.logger.warning(
                    "missing_required_fields", indicator="Williams %R", required=["high", "low", "close"]
                )
                return None

            new_row = pd.DataFrame([{"High": high, "Low": low, "Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if len(self.history_df) >= self.period:
                wr_series = self._calculate_williams_r(self.history_df.copy())
                self.current_value = wr_series.iloc[-1]
            else:
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error("williams_r_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Williams %R indicator to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: A new DataFrame with an added column for Williams %R.
        """
        try:
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing the following columns: {', '.join(missing_cols)}")

            self.logger.info("applying_williams_r_to_dataframe")
            data = data.copy()
            data[f"Williams_%R_{self.period}"] = self._calculate_williams_r(data)
            self.logger.info("williams_r_added_successfully")
            return data

        except Exception as e:
            self.logger.error("williams_r_apply_failed", error=str(e), exc_info=True)
            raise
