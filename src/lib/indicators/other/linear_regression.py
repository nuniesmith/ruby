import numpy as np
import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class LinearRegressionIndicator:
    """
    Linear Regression Slope Indicator.

    This indicator calculates the rolling linear regression slope for a given window (typically on the 'Close' prices).
    It maintains an internal history of market data and provides methods to update the calculation with new data points
    and to apply the calculation to an entire DataFrame.
    """

    REQUIRED_COLUMNS = ["Close"]

    def __init__(self, period: int = 14):
        """
        Initialize the Linear Regression Indicator.

        Args:
            period (int): The rolling window period for the linear regression calculation.
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer.")
        self.logger = logger
        self.period = period
        self.history_df = pd.DataFrame(columns=pd.Index(self.REQUIRED_COLUMNS))
        self.current_value = None

    def _calculate_lr_slope(self, data: pd.Series) -> pd.Series:  # type: ignore[type-arg]
        """
        Calculate the rolling linear regression slope for the provided Series.

        Args:
            data (pd.Series): Series representing data (e.g., closing prices).

        Returns:
            pd.Series: Series of calculated linear regression slopes.

        Raises:
            ValueError: If the input data is not a pandas Series or the window size is invalid.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")
        if len(data) < self.period:
            self.logger.warning(
                "insufficient_data_for_regression",
                data_length=len(data),
                regression_window=self.period,
            )
            return pd.Series([np.nan] * len(data), index=data.index)

        self.logger.info("calculating_linear_regression_slopes", window=self.period)
        X = np.arange(self.period)
        # Rolling apply to compute slope using numpy polyfit
        slopes = data.rolling(window=self.period).apply(
            lambda y: np.polyfit(X, y, 1)[0] if len(y) == self.period else np.nan, raw=True
        )
        self.logger.info("linear_regression_slopes_calculated")
        return slopes  # type: ignore[return-value]

    def update(self, data_point: dict) -> float | None:
        """
        Update the Linear Regression Slope with a new data point.

        Args:
            data_point (dict): Market data point containing the key 'close'.

        Returns:
            float or None: The latest linear regression slope value, or None if data is insufficient.
        """
        try:
            close = data_point.get("close")
            if close is None:
                self.logger.warning("missing_close_for_linear_regression_update")
                return None

            new_row = pd.DataFrame([{"Close": close}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

            if len(self.history_df) >= self.period:
                lr_slope_series = self._calculate_lr_slope(self.history_df["Close"].copy())  # type: ignore[arg-type]
                self.current_value = lr_slope_series.iloc[-1]
            else:
                self.logger.debug("insufficient_data_for_linear_regression_slope")
                self.current_value = None

            return self.current_value

        except Exception as e:
            self.logger.error("linear_regression_slope_update_failed", error=str(e), exc_info=True)
            return None

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Linear Regression Slope calculation to an entire DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing a 'Close' column.

        Returns:
            pd.DataFrame: A new DataFrame with an added 'LinearRegressionSlope' column.

        Raises:
            ValueError: If the 'Close' column is missing.
        """
        try:
            if "Close" not in data.columns:
                raise ValueError("DataFrame must contain a 'Close' column.")
            self.logger.info("adding_linear_regression_slope_to_dataframe")
            data = data.copy()
            data["LinearRegressionSlope"] = self._calculate_lr_slope(data["Close"])  # type: ignore[arg-type]
            self.logger.info("linear_regression_slope_added_successfully")
            return data

        except Exception as e:
            self.logger.error("error_applying_linear_regression_slope", error=str(e), exc_info=True)
            raise
