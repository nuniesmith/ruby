import pickle
from typing import Any

import pandas as pd

from lib.model._shims import log_execution, logger
from lib.model.base.model import BaseModel

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    Prophet = None
    HAS_PROPHET = False


class ProphetModel(BaseModel):
    """
    Prophet model for time series forecasting conforming to the BaseModel interface.

    The model expects training data as a DataFrame with a datetime index and converts it
    into Prophet's required format (columns: 'ds' for datetime and 'y' for target values).

    It fits the Prophet model using configurable parameters such as changepoint_prior_scale,
    seasonality_mode, and generates forecasts for a specified horizon, returning a DataFrame
    with 'PredictedPrice' as the forecasted values.

    Notes:
      - If the data frequency cannot be inferred, it defaults to daily ('D').
      - Custom seasonalities (e.g., weekly seasonality for daily data) can be configured.
      - The model currently uses only the point forecast (yhat); uncertainty intervals can be considered in future extensions.
    """

    def __init__(self, changepoint_prior_scale: float = 0.05, seasonality_mode: str = "multiplicative"):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model: Any | None = None
        self.train_data: pd.DataFrame | None = None
        self.target_column = "close"
        logger.info(
            f"Initialized ProphetModel with changepoint_prior_scale={changepoint_prior_scale}, "
            f"seasonality_mode={seasonality_mode}"
        )

    @log_execution
    def build_model(self, **kwargs) -> Any:
        """
        Build the Prophet model with the specified parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to Prophet().

        Returns:
            self: The model instance.
        """
        if not HAS_PROPHET:
            raise ImportError("prophet is required but not installed")
        self.model = Prophet(  # type: ignore[operator]
            changepoint_prior_scale=self.changepoint_prior_scale, seasonality_mode=self.seasonality_mode, **kwargs
        )
        logger.info("Built Prophet model instance.")
        return self

    @log_execution
    def fit(self, train_data: pd.DataFrame, target_column: str = "close") -> Any:
        """
        Fit the Prophet model using the provided training data.

        Args:
            train_data (pd.DataFrame): DataFrame with a datetime index.
            target_column (str): Column name containing the target values (default 'close').

        Returns:
            self: The fitted model instance.
        """
        self.train_data = train_data.copy()
        self.target_column = target_column
        logger.info(f"Copied training data with {len(self.train_data)} records and target column '{target_column}'")

        # Convert training data to Prophet's format
        prophet_data = pd.DataFrame({"ds": self.train_data.index, "y": self.train_data[target_column].values})
        logger.info("Converted training data to Prophet format with columns ['ds', 'y']")

        return self.train(prophet_data)

    @log_execution
    def train(self, train_df: pd.DataFrame, date_col: str = "ds", y_col: str = "y", **kwargs) -> Any:
        """
        Train the Prophet model on the provided DataFrame.

        Args:
            train_df (pd.DataFrame): DataFrame with columns [date_col, y_col].
            date_col (str): Name of the date column (default 'ds'). Must be 'ds' for Prophet.
            y_col (str): Name of the target column (default 'y'). Must be 'y' for Prophet.
            **kwargs: Additional keyword arguments to pass to Prophet(), e.g.,
                      custom seasonalities or holidays.

        Returns:
            self: The trained Prophet model.
        """
        if not HAS_PROPHET:
            raise ImportError("prophet is required but not installed")
        logger.info("Training Prophet model...")
        self.model = Prophet(  # type: ignore[operator]
            changepoint_prior_scale=self.changepoint_prior_scale, seasonality_mode=self.seasonality_mode, **kwargs
        )
        logger.info("Created Prophet model instance.")

        # Optionally add daily seasonality if data frequency is high (e.g., minute or hourly data)
        if self.train_data is not None and isinstance(self.train_data.index, pd.DatetimeIndex):
            freq = self.train_data.index.freq
            if freq in ["T", "min", "H", "1H", "4H"]:
                self.model.add_seasonality(name="daily", period=24, fourier_order=5)
                logger.info("Added daily seasonality to Prophet model.")

        if date_col != "ds" or y_col != "y":
            logger.warning(
                f"Prophet requires columns named 'ds' and 'y'. "
                f"Parameters date_col={date_col} and y_col={y_col} are for documentation only."
            )

        self.model.fit(train_df)
        logger.info(f"Prophet model training completed with {len(train_df)} data points.")
        return self

    @log_execution
    def predict(self, horizon: int = 1) -> pd.DataFrame:
        """
        Generate a forecast for the next 'horizon' time steps.

        Args:
            horizon (int): Number of future periods to forecast.

        Returns:
            pd.DataFrame: Forecast DataFrame with index as forecasted dates and a column 'PredictedPrice'.
        """
        if self.model is None or self.train_data is None:
            raise ValueError("Model not fitted. Call fit() first.")
        try:
            # Determine frequency from the training data's index; warn if not inferable.
            if isinstance(self.train_data.index, pd.DatetimeIndex):
                freq = self.train_data.index.freq
                if freq is None:
                    freq = "D"
                    logger.warning("Could not infer data frequency; defaulting to daily ('D').")
                else:
                    freq = str(freq)
                    logger.info(f"Inferred data frequency as: {freq}")
            else:
                freq = "D"
                logger.warning("Index is not a DatetimeIndex; defaulting to daily ('D').")

            # Create a future dataframe including both history and future periods.
            future = self.model.make_future_dataframe(periods=horizon, freq=freq)
            logger.info(f"Created future dataframe with {horizon} periods using frequency {freq}.")
            forecast = self.model.predict(future)
            logger.info("Generated forecast using Prophet model.")

            # Extract only the forecasts for future periods (beyond the training data)
            last_date = self.train_data.index[-1]
            forecast_values = forecast[forecast["ds"] > last_date]["yhat"].values
            logger.info(f"Extracted forecast values for periods beyond {last_date}.")

            # Generate forecast index using the same frequency
            forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
            forecast_df = pd.DataFrame({"PredictedPrice": forecast_values}, index=forecast_index)
            logger.info(f"Generated forecast DataFrame with index starting from {forecast_index[0]}")
            return forecast_df

        except Exception as e:
            logger.exception(f"Error predicting with Prophet model: {e}")
            raise

    @log_execution
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series | list) -> dict:
        """
        Evaluate the Prophet model using Mean Squared Error (MSE).

        Args:
            X_test (pd.DataFrame): Future DataFrame with a 'ds' column containing dates.
            y_test (pd.Series or list): Ground truth values corresponding to X_test.

        Returns:
            dict: Dictionary containing evaluation metric(s), e.g., {"mse": value}.
        """
        logger.info("Evaluating Prophet model...")
        forecast_df = self.predict(horizon=len(X_test))
        y_pred = forecast_df["PredictedPrice"].to_numpy(dtype=float)
        y_true = pd.Series(y_test).to_numpy(dtype=float)
        mse = ((y_true - y_pred) ** 2).mean()
        logger.info(f"Prophet model MSE: {mse:.4f}")
        return {"mse": mse}

    @log_execution
    def save_model(self, file_path: str) -> None:
        """
        Save the trained Prophet model to a file using pickle.

        Args:
            file_path (str): Path where the model should be saved.
        """
        if self.model is None:
            raise ValueError("ProphetModel is not trained; nothing to save.")
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info(f"Prophet model saved to {file_path}")
        except Exception as e:
            logger.exception(f"Error saving Prophet model: {e}")
            raise

    @log_execution
    def load_model(self, file_path: str) -> None:
        """
        Load the Prophet model from a pickle file.

        Args:
            file_path (str): Path from which to load the model.
        """
        try:
            with open(file_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Prophet model loaded from {file_path}")
        except Exception as e:
            logger.exception(f"Error loading Prophet model: {e}")
            raise

    @log_execution
    def generate_samples(self, n_samples: int = 10) -> Any:
        """
        Prophet does not natively support sample generation.
        """
        logger.info("generate_samples() not implemented for ProphetModel.")
        raise NotImplementedError("Sample generation is not supported for ProphetModel.")

    def __str__(self) -> str:
        trained = "Yes" if self.model is not None else "No"
        return f"ProphetModel(trained={trained})"


class ConcreteProphetModel(ProphetModel):
    """
    Concrete implementation of the Prophet model with default parameters.

    This class is used to create an instance of the Prophet model with default parameters.
    """

    def __init__(self):
        super().__init__(changepoint_prior_scale=0.05, seasonality_mode="multiplicative")
        logger.info("Initialized ConcreteProphetModel with default parameters.")
