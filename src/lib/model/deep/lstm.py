import json
import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import log_execution, logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment,misc]
    HAS_TORCH = False

try:
    import lightning.pytorch as pl

    HAS_LIGHTNING = True
except ImportError:
    pl = None  # type: ignore[assignment]
    HAS_LIGHTNING = False

from lib.model.base.model import BaseModel
from lib.model.utils.metadata import ModelMetadata


class LSTMNetwork(nn.Module):
    """
    A simple LSTM network for univariate time series forecasting.
    Consists of an LSTM layer followed by a linear layer.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Any) -> Any:
        lstm_out, _ = self.lstm(x)
        # Use the last output in the sequence for prediction.
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class LSTMModel(BaseModel):
    """
    PyTorch-based LSTM model for time series forecasting.

    Key improvements:
      - Uses a configurable lookback window (default 60) to create training sequences.
      - Implements predict(horizon) to generate an iterative multi-step forecast.
      - Scales training data based only on the training split and stores scaling parameters.
      - Returns predictions as a DataFrame with a 'PredictedPrice' column.
    """

    def __init__(
        self,
        lookback: int = 60,
        input_size: int = 1,
        hidden_size: int = 128,
        dropout: float = 0.2,
        num_layers: int = 1,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: str = device
        self.lookback: int = lookback
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.dropout: float = dropout
        self.num_layers: int = num_layers

        self.model: LSTMNetwork | None = None
        self.min_val: float | None = None
        self.max_val: float | None = None
        # To store the last training sequence for forecasting
        self.last_window: np.ndarray | None = None

    @log_execution
    def build_model(self) -> None:
        try:
            net = LSTMNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                num_layers=self.num_layers,
            )
            self.model = net.to(self.device)
            logger.info(f"Built LSTMNetwork with lookback={self.lookback} on device: {self.device}")
        except Exception as e:
            logger.exception(f"Error building model: {e}")
            raise

    @log_execution
    def fit(
        self,
        train_series: np.ndarray | pd.Series,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int | None = 42,
    ) -> Any:
        """
        Fit the LSTM model to the provided univariate time series.

        This method:
          - Builds the model if not already built.
          - Computes min-max scaling on the training data.
          - Creates sliding-window training samples of length `lookback`.
          - Trains the model using mini-batch gradient descent via DataLoader.
          - Stores the last window for forecasting.

        Args:
            train_series: The univariate time series for training.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Mini-batch size.
            seed: Random seed for reproducibility.

        Returns:
            self for method chaining.
        """
        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Random seed set to {seed}")

        if self.model is None:
            self.build_model()

        if self.model is None:
            raise RuntimeError("Failed to build model. Check logs for details.")

        # Convert training series to numpy array of type float32
        if isinstance(train_series, pd.Series):
            arr = train_series.values.astype(np.float32)
        else:
            arr = np.asarray(train_series, dtype=np.float32)
        logger.info(f"Training data converted to numpy array with shape: {arr.shape}")

        # Ensure there are enough data points for the lookback window
        if len(arr) <= self.lookback:
            raise ValueError("Training series length must be greater than lookback window.")

        # Set up min-max scaling parameters using only the training data
        self.min_val = float(arr.min())
        self.max_val = float(arr.max())
        logger.info(f"Computed min_val: {self.min_val}, max_val: {self.max_val}")
        if self.max_val == self.min_val:
            raise ValueError("All values are identical; cannot scale.")
        scaled = (arr - self.min_val) / (self.max_val - self.min_val)

        # Create training samples using sliding window
        X_train: Any = []
        y_train: Any = []
        for i in range(self.lookback, len(scaled)):
            X_train.append(scaled[i - self.lookback : i])
            y_train.append(scaled[i])
        X_train = np.array(X_train)  # Shape: (num_samples, lookback)
        y_train = np.array(y_train)  # Shape: (num_samples,)
        logger.info(f"Created training samples: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Reshape X_train to [num_samples, lookback, input_size] and y_train to [num_samples, 1]
        X_train = X_train.reshape(-1, self.lookback, self.input_size)
        y_train = y_train.reshape(-1, 1)

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        logger.info(f"Converted training samples to tensors on device: {self.device}")

        # Create a DataLoader for mini-batch training
        assert TensorDataset is not None, "torch is required but not installed"
        assert DataLoader is not None, "torch is required but not installed"
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"DataLoader created with batch_size: {batch_size}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        logger.info(f"Training started for {epochs} epochs with learning rate: {lr}")

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss = {epoch_loss:.4f}")

        # Store the last window for forecasting (last lookback values from training)
        self.last_window = scaled[-self.lookback :].reshape(1, self.lookback, self.input_size)
        logger.info("Stored last training window for forecasting.")
        return self

    def generate_samples(self, n_samples=10, **kwargs):
        """
        Generate samples from the trained LSTM model.

        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters

        Returns:
            Generated samples
        """
        if self.model is None or self.last_window is None:
            raise RuntimeError("Model not trained or missing last window")

        self.model.eval()
        samples = []
        current_window = torch.tensor(self.last_window.copy(), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            for _ in range(n_samples):
                # Get prediction for current window
                pred = self.model(current_window).cpu().numpy()[0]
                samples.append(float(pred * (self.max_val - self.min_val) + self.min_val))

                # Update window with prediction (rolling window)
                new_point = torch.tensor([[pred]], dtype=torch.float32).to(self.device)
                current_window = torch.cat((current_window[:, 1:, :], new_point.view(1, 1, 1)), dim=1)

        return np.array(samples)

    # Override train method to match BaseModel signature while using fit internally
    @log_execution
    def train(self, X_train, y_train=None, **kwargs):
        return self.fit(train_series=X_train, **kwargs)

    @log_execution
    def predict(self, horizon: int) -> pd.DataFrame:
        """
        Perform an iterative multi-step forecast for the next 'horizon' time steps.

        Starting from the stored last training window, the model predicts one step ahead,
        appends that prediction to the window, and repeats iteratively.

        Returns:
            pd.DataFrame: A DataFrame with a 'PredictedPrice' column (in original scale)
                          and a generated date index (default daily frequency).
        """
        if self.model is None or self.last_window is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaling parameters not set. Call fit() first.")

        self.model.eval()
        current_window = torch.tensor(self.last_window, dtype=torch.float32).to(self.device)
        predictions: Any = []
        with torch.no_grad():
            for _ in range(horizon):
                pred_t = self.model(current_window)
                pred = pred_t.cpu().numpy().flatten()[0]
                predictions.append(pred)
                # Update the window: remove the first value and append the new prediction.
                new_input = np.array([[pred]])
                current_window_np = current_window.cpu().numpy()
                updated_window = np.concatenate(
                    [current_window_np[:, 1:, :], new_input.reshape(1, 1, self.input_size)], axis=1
                )
                current_window = torch.tensor(updated_window, dtype=torch.float32).to(self.device)
        logger.info(f"Raw predictions (scaled): {predictions}")

        # Invert scaling for predictions
        predictions = np.array(predictions) * (self.max_val - self.min_val) + self.min_val
        logger.info(f"Inverted predictions: {predictions}")

        # Create a date range for the forecast. Here we default to daily frequency.
        # In practice, the frequency should match the original data's frequency.
        last_index = pd.Timestamp.today()  # Replace with last training date if available.
        forecast_index = pd.date_range(start=last_index, periods=horizon + 1, freq="D")[1:]
        forecast_df = pd.DataFrame({"PredictedPrice": predictions}, index=forecast_index)
        logger.info(f"Generated forecast with date index starting from {forecast_index[0]}")
        return forecast_df


class LightningLSTMModel(pl.LightningModule):
    """
    PyTorch Lightning implementation of the LSTM model.
    This class wraps the LSTMNetwork in a Lightning module for easier training.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        dropout: float = 0.2,
        num_layers: int = 1,
        lookback: int = 60,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        # Save all hyperparameters except modules which are saved separately
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])

        # Create the network
        self.network = LSTMNetwork(
            input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers
        )

        # Define loss function
        self.loss = nn.MSELoss()

        # Store additional parameters
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.min_val = None
        self.max_val = None
        self.last_window = None

        # Create metadata
        self.metadata = ModelMetadata(model_type="LightningLSTM")
        self.metadata.params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers": num_layers,
            "lookback": lookback,
            "learning_rate": learning_rate,
            **kwargs,
        }

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self, train_series):
        """Prepare data for training"""
        if isinstance(train_series, pd.Series):
            arr = train_series.values.astype(np.float32)
        else:
            arr = np.asarray(train_series, dtype=np.float32)

        # Set up min-max scaling
        self.min_val = float(arr.min())
        self.max_val = float(arr.max())
        if self.max_val == self.min_val:
            raise ValueError("All values are identical; cannot scale.")

        scaled = (arr - self.min_val) / (self.max_val - self.min_val)

        # Create sliding window samples
        X_train = []
        y_train = []
        for i in range(self.lookback, len(scaled)):
            X_train.append(scaled[i - self.lookback : i])
            y_train.append(scaled[i])

        X_train = np.array(X_train).reshape(-1, self.lookback, self.hparams.input_size)
        y_train = np.array(y_train).reshape(-1, 1)

        # Store last window for forecasting
        self.last_window = scaled[-self.lookback :].reshape(1, self.lookback, self.hparams.input_size)

        return X_train, y_train

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Generate a forecast for the next horizon steps"""
        if self.last_window is None:
            raise RuntimeError("No training data has been processed.")

        self.eval()
        current_window = torch.tensor(self.last_window, dtype=torch.float32)
        predictions = []

        with torch.no_grad():
            for _ in range(horizon):
                pred_t = self(current_window)
                pred = pred_t.cpu().numpy().flatten()[0]
                predictions.append(pred)

                # Update window for next prediction
                new_input = np.array([[pred]])
                current_window_np = current_window.cpu().numpy()
                updated_window = np.concatenate(
                    [current_window_np[:, 1:, :], new_input.reshape(1, 1, self.hparams.input_size)], axis=1
                )
                current_window = torch.tensor(updated_window, dtype=torch.float32)

        # Invert scaling
        predictions = np.array(predictions) * (self.max_val - self.min_val) + self.min_val

        # Create forecast DataFrame
        last_index = pd.Timestamp.today()
        forecast_index = pd.date_range(start=last_index, periods=horizon + 1, freq="D")[1:]
        forecast_df = pd.DataFrame({"PredictedPrice": predictions}, index=forecast_index)

        return forecast_df

    def save_model(self, path: str) -> None:
        """Save the model with metadata"""
        # Update metadata
        self.metadata.updated_at = datetime.now().isoformat()

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        checkpoint_path = path
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"

        try:
            # Save Lightning model
            self.training.save_checkpoint(checkpoint_path)  # type: ignore[attr-defined]

            # Save additional metadata
            with open(metadata_path, "w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

            # Save scaling parameters separately
            scaling_path = f"{os.path.splitext(path)[0]}_scaling.json"
            with open(scaling_path, "w") as f:
                json.dump({"min_val": self.min_val, "max_val": self.max_val}, f, indent=2)

            if self.last_window is not None:
                window_path = f"{os.path.splitext(path)[0]}_last_window.npy"
                np.save(window_path, self.last_window)

            logger.info(f"Successfully saved model to {path}")
        except Exception as e:
            logger.exception(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load the model with metadata"""
        if not os.path.exists(path):
            logger.warning(f"Model file {path} not found.")
            return

        try:
            # Load Lightning model
            self.load_from_checkpoint(path)

            # Load metadata if available
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.metadata = ModelMetadata.from_dict(json.load(f))

            # Load scaling parameters
            scaling_path = f"{os.path.splitext(path)[0]}_scaling.json"
            if os.path.exists(scaling_path):
                with open(scaling_path) as f:
                    scaling_data = json.load(f)
                    self.min_val = scaling_data.get("min_val")
                    self.max_val = scaling_data.get("max_val")

            # Load last window if available
            window_path = f"{os.path.splitext(path)[0]}_last_window.npy"
            if os.path.exists(window_path):
                self.last_window = np.load(window_path)

            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            logger.exception(f"Error loading model: {str(e)}")
            raise


class ConcreteLSTMModel(LightningLSTMModel):
    """
    A concrete implementation of the Lightning LSTM model with additional functionality.
    This class provides a simpler API for training and prediction.
    """

    def __init__(
        self,
        input_size: int = 1,
        lookback: int = 60,
        hidden_size: int = 50,
        dropout: float = 0.2,
        num_layers: int = 1,
        device: str | None = None,
        use_amp: bool = False,
        **kwargs: Any,
    ):
        # Set device
        self.device_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Enable automatic mixed precision for performance if on CUDA
        self.use_amp = use_amp and self.device_str == "cuda" and torch.cuda.is_available()

        # Initialize Lightning module
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            lookback=lookback,
            learning_rate=kwargs.get("learning_rate", 1e-3),
            **kwargs,
        )

        # Store training parameters
        self.training_params: dict[str, Any] = {}

        # Set up scaler for input normalization if requested
        self.scaler = None
        if kwargs.get("use_scaler", True):
            try:
                from sklearn.preprocessing import StandardScaler

                self.scaler = StandardScaler()
            except ImportError:
                logger.warning("StandardScaler not available, proceeding without scaling")

    def train(  # type: ignore[override]
        self,
        train_data: pd.DataFrame,
        target_column: str,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Train the model with given parameters

        Args:
            train_data: DataFrame containing training data
            target_column: Name of the target column
            epochs: Number of training epochs
            batch_size: Size of training batches
            learning_rate: Learning rate for optimization
            **kwargs: Additional training parameters

        Returns:
            self: The trained model instance
        """
        # Update training parameters
        self.training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate if learning_rate is not None else self.learning_rate,
            **kwargs,
        }

        # Log training parameters
        logger.info(f"Training LSTM model with params: {self.training_params}")

        # Extract target series
        train_series = train_data[target_column]

        # Apply StandardScaler if available
        if self.scaler is not None:
            train_series = pd.Series(
                self.scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten(), index=train_series.index
            )

        # Prepare data
        X_train, y_train = self.prepare_data(train_series)

        # Create TensorDataset and DataLoader
        assert TensorDataset is not None, "torch is required but not installed"
        assert DataLoader is not None, "torch is required but not installed"
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up validation data if provided
        val_loader = None
        if "val_data" in kwargs and "val_target" in kwargs:
            val_series = kwargs["val_data"][kwargs["val_target"]]
            if self.scaler is not None:
                val_series = pd.Series(
                    self.scaler.transform(val_series.values.reshape(-1, 1)).flatten(), index=val_series.index
                )
            X_val, y_val = self.prepare_data(val_series)
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize trainer
        training = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.device_str if self.device_str == "cpu" else "gpu",
            precision="16-mixed" if self.use_amp else "32",
            **{k: v for k, v in kwargs.items() if k in ["gradient_clip_val", "accumulate_grad_batches", "logger"]},
        )

        # Train the model
        training.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Store training for later use
        self.training = training  # type: ignore[attr-defined,assignment]

        # Update metadata
        self.metadata.updated_at = datetime.now().isoformat()

        return self

    def predict(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        horizon: int | None = None,
        batch_size: int = 256,
    ) -> np.ndarray | pd.DataFrame:
        """
        Make predictions with the model

        Args:
            data: Data to make predictions on
            target_column: Target column name (for preprocessing)
            horizon: Number of future steps to predict (for forecasting)
            batch_size: Batch size for prediction

        Returns:
            Either a numpy array of predictions or a DataFrame for forecasting
        """
        try:
            # Set model to evaluation mode
            self.eval()

            # If horizon is specified, do forecasting
            if horizon is not None:
                return self.forecast(horizon)

            # Otherwise, make predictions on the provided data
            if target_column is not None:
                series = data[target_column]
                if self.scaler is not None:
                    series = pd.Series(
                        self.scaler.transform(series.values.reshape(-1, 1)).flatten(), index=series.index
                    )

                # Create sliding windows
                windows = []
                for i in range(len(series) - self.lookback + 1):
                    windows.append(series.iloc[i : i + self.lookback].values)

                X_pred = np.array(windows).reshape(-1, self.lookback, self.hparams["input_size"])  # type: ignore[index]

                # Disable gradients for inference
                with torch.no_grad():
                    # Process in batches
                    all_preds = []
                    for i in range(0, len(X_pred), batch_size):
                        batch = torch.tensor(X_pred[i : i + batch_size], dtype=torch.float32)
                        preds = self(batch).cpu().numpy().flatten()
                        all_preds.extend(preds)

                # Invert scaling if needed
                if self.scaler is not None:
                    all_preds = self.scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()

                return np.array(all_preds)
            else:
                logger.warning("No target_column provided for prediction")
                return np.array([])

        except Exception as e:
            logger.exception(f"Error during prediction: {str(e)}")
            return np.array([])

    def generate_samples(self, n_samples: int = 10, **kwargs: Any) -> list[float]:
        """Generate samples from the model"""
        try:
            # Get the number of samples to generate
            num_samples = kwargs.get("num_samples", n_samples)
            logger.info(f"Generating {num_samples} samples with LSTM model")

            # If we have a last window, use it to generate samples
            if self.last_window is not None:
                self.eval()
                current_window = torch.tensor(self.last_window, dtype=torch.float32)
                predictions = []

                with torch.no_grad():
                    for _ in range(num_samples):
                        pred_t = self(current_window)
                        pred = pred_t.cpu().numpy().flatten()[0]
                        predictions.append(pred)

                        # Update window for next prediction
                        new_input = np.array([[pred]])
                        current_window_np = current_window.cpu().numpy()
                        updated_window = np.concatenate(
                            [current_window_np[:, 1:, :], new_input.reshape(1, 1, self.hparams.input_size)], axis=1
                        )
                        current_window = torch.tensor(updated_window, dtype=torch.float32)

                # Invert scaling
                if self.min_val is not None and self.max_val is not None:
                    predictions = np.array(predictions) * (self.max_val - self.min_val) + self.min_val

                return predictions.tolist()
            else:
                logger.warning("No training data has been processed, generating random samples")
                return [random.random() for _ in range(num_samples)]

        except Exception as e:
            logger.exception(f"Error generating samples: {str(e)}")
            return [0.0] * num_samples

    def cleanup(self) -> None:
        """Clean up resources used by the model"""
        try:
            # Move model to CPU to free GPU memory
            if self.device_str == "cuda":
                self.to("cpu")
                torch.cuda.empty_cache()
                logger.info("Moved LSTM model to CPU and cleared CUDA cache")
        except Exception as e:
            logger.warning(f"Error during model cleanup: {str(e)}")
