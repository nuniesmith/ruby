"""
Temporal Fusion Transformer (TFT) model implementation.
"""

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import DEFAULT_DEVICE, log_execution, logger

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE, RMSE, SMAPE

    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    GroupNormalizer = None
    MAE = None
    RMSE = None
    SMAPE = None
    logger.warning("PyTorch Forecasting not available - TFT model will have limited functionality")
    PYTORCH_FORECASTING_AVAILABLE = False

from lib.model.base.model import BaseModel


def create_sequences(data, context_length, prediction_length, feature_columns, target_column, is_train=True):
    """Local helper replacing utils.data_utils.create_sequences."""
    import numpy as np

    if isinstance(data, pd.DataFrame):
        features = data[feature_columns].values if feature_columns else data.drop(columns=[target_column]).values
        target = data[target_column].values
    else:
        features = np.asarray(data)
        target = features

    X, y = [], []
    total = len(features) - context_length - prediction_length + 1
    for i in range(max(total, 0)):
        X.append(features[i : i + context_length])
        if is_train:
            y.append(target[i + context_length : i + context_length + prediction_length])
        else:
            y.append(
                target[i + context_length : i + context_length + prediction_length]
                if i + context_length + prediction_length <= len(target)
                else []
            )
    return np.array(X) if X else np.empty((0, context_length, features.shape[1] if features.ndim > 1 else 1)), np.array(
        y
    ) if y else np.empty((0, prediction_length))


class TFTModel(BaseModel):
    """
    Temporal Fusion Transformer model for time series forecasting.

    This model wraps the PyTorch Forecasting implementation of TFT when available,
    or provides a simplified version otherwise.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 128,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_encoder_length: int = 120,
        max_prediction_length: int = 10,
        gradient_clip_val: float = 0.1,
        context_length: int = 32,
        device: str = DEFAULT_DEVICE,
        **kwargs,
    ):
        """
        Initialize the TFT model with hyperparameters.

        Args:
            hidden_size: Hidden size for attention layers
            lstm_layers: Number of LSTM layers
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for processing continuous variables
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            max_encoder_length: Max length of encoder/lookback window
            max_prediction_length: Prediction horizon length
            gradient_clip_val: Gradient clipping value
            context_length: Context length for historical values
            device: Device to use ('cuda' or 'cpu')
            **kwargs: Additional parameters
        """
        super().__init__()

        # Store hyperparameters
        self.hyperparams = {
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "attention_head_size": attention_head_size,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "gradient_clip_val": gradient_clip_val,
            "context_length": context_length,
        }
        self.hyperparams.update(kwargs)

        # Initialize model
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.training = None

        # Set device
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"

        # Initialize metadata
        self.metadata = {
            "model_type": "TFT",
            "params": self.hyperparams.copy(),
            "metrics": {},
            "feature_importance": {},
            "created_at": pd.Timestamp.now().isoformat(),
            "updated_at": pd.Timestamp.now().isoformat(),
        }

        # Check if PyTorch Forecasting is available
        self.has_pytorch_forecasting = PYTORCH_FORECASTING_AVAILABLE
        if not self.has_pytorch_forecasting:
            logger.warning("Using simplified TFT implementation as PyTorch Forecasting is not available")

        # Initialize training parameters
        self.training_params: dict[str, Any] = {
            "target_column": None,
            "time_idx_column": None,
            "group_ids": None,
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": [],
        }

    def _prepare_data(
        self, data: pd.DataFrame, target_column: str, time_idx_column: str = "", group_ids: list[str] | None = None
    ):
        """
        Prepare data for TFT model.

        Args:
            data: Input dataframe
            target_column: Target column name
            time_idx_column: Column containing time indices
            group_ids: List of columns identifying different time series

        Returns:
            Prepared data
        """
        if group_ids is None:
            group_ids = []
        if not self.has_pytorch_forecasting:
            # Simplified data preparation
            if time_idx_column is None:
                time_idx_column = "timestamp" if "timestamp" in data.columns else data.index.name or "index"
                if time_idx_column not in data.columns:
                    data = data.reset_index().rename(columns={"index": "time_idx"})
                    time_idx_column = "time_idx"

            # Ensure we have a time index
            if time_idx_column not in data.columns:
                data["time_idx"] = range(len(data))
                time_idx_column = "time_idx"

            # Store important columns
            self.time_idx_column = time_idx_column
            self.target_column = target_column

            # Normalize data
            self.feature_columns = [col for col in data.columns if col != target_column and col != time_idx_column]

            # Store preprocessed data for later use
            return data

        else:
            # Full PyTorch Forecasting data preparation
            if time_idx_column is None:
                if "time_idx" in data.columns:
                    time_idx_column = "time_idx"
                elif "timestamp" in data.columns:
                    # Convert timestamp to time_idx if needed
                    if pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                        data["time_idx"] = (data["timestamp"] - data["timestamp"].min()).dt.days
                    else:
                        data["time_idx"] = range(len(data))
                    time_idx_column = "time_idx"
                else:
                    # Create time index
                    data["time_idx"] = range(len(data))
                    time_idx_column = "time_idx"

            # If no group_ids provided, use a single group
            if group_ids is None:
                data["series_id"] = 0
                group_ids = ["series_id"]

            # Determine variable groups
            self.time_varying_unknown_reals = [
                col
                for col in data.columns
                if col != time_idx_column and col not in group_ids and pd.api.types.is_numeric_dtype(data[col])
            ]

            if target_column not in self.time_varying_unknown_reals:
                self.time_varying_unknown_reals.append(target_column)

            # Store data preparation parameters
            self.training_params.update(
                {
                    "target_column": target_column,  # type: ignore[dict-item]
                    "time_idx_column": time_idx_column,  # type: ignore[dict-item]
                    "group_ids": group_ids,
                    "time_varying_unknown_reals": self.time_varying_unknown_reals,
                }
            )

            return data

    @log_execution
    def build_model(self, **kwargs) -> "BaseModel":
        """
        Build the TFT model.

        Args:
            **kwargs: Additional parameters
        """
        if not self.has_pytorch_forecasting:
            logger.warning("Using simplified TFT implementation. Full features require PyTorch Forecasting.")
        return self

    @log_execution
    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs) -> "BaseModel":
        """
        Fit the TFT model to the provided dataset.

        Args:
            train_data: Data to train on
            target_column: Name of the target column
            **kwargs: Additional parameters for training

        Returns:
            self: For method chaining
        """
        # Call the existing train method
        self.train(data=train_data, target_column=target_column, **kwargs)
        return self

    @log_execution
    def generate_samples(self, n_samples=10, **kwargs):
        """
        Generate samples from the TFT model.

        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters

        Returns:
            Generated samples
        """
        if not self.has_pytorch_forecasting or self.model is None:
            raise ValueError("Model must be trained before generating samples")

        # For TFT, sample generation typically involves some input data
        if "input_data" not in kwargs:
            raise ValueError("input_data is required for TFT sample generation")

        input_data = kwargs["input_data"]
        horizon = kwargs.get("horizon", n_samples)

        # Generate predictions with the model - TFT normally doesn't do sample generation
        # but we can return predictions as samples
        predictions = self.predict(data=input_data, horizon=horizon)
        return predictions

    @log_execution
    def train(
        self,
        data: pd.DataFrame,
        target_column: str,
        validation_data: pd.DataFrame | None = None,
        time_idx_column: str = "",
        group_ids: list[str] | None = None,
        **kwargs,
    ):
        """
        Train the TFT model on data.

        Args:
            data: Training dataframe
            target_column: Target column name
            validation_data: Optional validation data
            time_idx_column: Column containing time indices
            group_ids: List of columns identifying different time series
            **kwargs: Additional parameters passed to fit method
        """
        # Update hyperparameters if provided
        if group_ids is None:
            group_ids = []
        for k, v in kwargs.items():
            if k in self.hyperparams:
                self.hyperparams[k] = v

        # Prepare data
        prepared_data = self._prepare_data(data, target_column, time_idx_column, group_ids)

        if not self.has_pytorch_forecasting:
            # Simplified TFT implementation
            logger.warning("Using simplified TFT implementation. Full features require PyTorch Forecasting.")

            # Create sequences from data for training
            X, y = create_sequences(
                prepared_data,
                self.hyperparams["context_length"],
                self.hyperparams["max_prediction_length"],
                self.feature_columns,
                target_column,
            )

            # Initialize a simple LSTM model as a fallback
            self.model = self._create_simple_model(X.shape[-2:])

            # Train the model
            X_train = torch.tensor(X, dtype=torch.float32)
            y_train = torch.tensor(y, dtype=torch.float32)

            # Create a simple dataset and dataloader
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=int(self.hyperparams["batch_size"]), shuffle=True
            )

            # Train the model
            self._train_simple_model(dataloader)

            # Update metadata
            self.metadata["updated_at"] = pd.Timestamp.now().isoformat()  # type: ignore[index]
            logger.info(f"Model trained with simplified implementation, target: {target_column}")

        else:
            # Full PyTorch Forecasting implementation
            assert TimeSeriesDataSet is not None, "pytorch_forecasting is required but not installed"
            assert GroupNormalizer is not None, "pytorch_forecasting is required but not installed"
            assert RMSE is not None, "pytorch_forecasting is required but not installed"
            max_encoder_length = self.hyperparams["max_encoder_length"]
            max_prediction_length = self.hyperparams["max_prediction_length"]

            # Create training dataset
            self.training_dataset = TimeSeriesDataSet(
                data=prepared_data,
                time_idx=time_idx_column,
                target=target_column,
                group_ids=group_ids,
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=[],
                static_reals=[],
                time_varying_known_categoricals=[],
                time_varying_known_reals=[],
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=self.time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
            )

            # Create validation dataset if provided
            if validation_data is not None:
                self.validation_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, validation_data)
                val_dataloader = self.validation_dataset.to_dataloader(batch_size=self.hyperparams["batch_size"])  # type: ignore[attr-defined]
            else:
                val_dataloader = None

            # Create dataloaders
            train_dataloader = self.training_dataset.to_dataloader(  # type: ignore[attr-defined]
                batch_size=self.hyperparams["batch_size"], train=True
            )

            # Create PyTorch Lightning trainer
            self.training = pl.Trainer(  # type: ignore[assignment]
                max_epochs=kwargs.get("epochs", 20),
                accelerator="gpu" if self.device == "cuda" else "cpu",
                gradient_clip_val=self.hyperparams["gradient_clip_val"],
                devices=1 if torch.cuda.is_available() and self.device == "cuda" else 1,
                enable_progress_bar=True,
                logger=kwargs.get("logger", True),
            )

            # Create TFT model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_dataset,
                hidden_size=self.hyperparams["hidden_size"],
                attention_head_size=self.hyperparams["attention_head_size"],
                dropout=self.hyperparams["dropout"],
                hidden_continuous_size=self.hyperparams["hidden_continuous_size"],
                lstm_layers=self.hyperparams["lstm_layers"],
                learning_rate=self.hyperparams["learning_rate"],
                loss=RMSE(),
            )

            # Fit the model
            self.training.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)  # type: ignore[attr-defined]

            # Calculate and store feature importance if available
            try:
                feature_importance = self.model.feature_importance()  # type: ignore[attr-defined]
                self.metadata["feature_importance"] = {  # type: ignore[index]
                    name: float(importance)
                    for name, importance in zip(
                        self.time_varying_unknown_reals, feature_importance.mean(0), strict=False
                    )
                }
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")

            # Update metadata
            self.metadata["updated_at"] = pd.Timestamp.now().isoformat()  # type: ignore[index]
            logger.info(f"TFT model trained with {len(data)} samples, target: {target_column}")

    @log_execution
    def _create_simple_model(self, input_shape):
        """
        Create a simple LSTM model as a fallback for TFT.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Simple PyTorch model
        """

        class SimpleLSTM(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.lstm = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
                self.fc = torch.nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])
                return output

        # Extract input dimensions
        seq_len, n_features = input_shape
        hidden_size = self.hyperparams["hidden_size"]
        prediction_length = self.hyperparams["max_prediction_length"]

        return SimpleLSTM(n_features, hidden_size, prediction_length, self.hyperparams["lstm_layers"]).to(self.device)

    @log_execution
    def _train_simple_model(self, dataloader):
        """
        Train simple LSTM model.

        Args:
            dataloader: PyTorch DataLoader containing training data
        """
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams["learning_rate"])
        criterion = torch.nn.MSELoss()

        # Set model to training mode
        self.model.train()

        # Train for multiple epochs
        epochs = 20
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                assert self.model is not None, "Model must be built before training"
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

        # Set model to evaluation mode
        self.model.eval()

    @log_execution
    def predict(self, data=None, horizon=None, return_ci=False):
        """
        Generate forecasts with the TFT model.

        Args:
            data: Data for prediction. If None, uses horizon parameter
            horizon: Number of steps to forecast (used if data is None)
            return_ci: Whether to return confidence intervals

        Returns:
            Predictions and optionally confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Set model to evaluation mode
        if hasattr(self.model, "eval"):
            self.model.eval()

        if not self.has_pytorch_forecasting:
            # Simplified prediction
            if data is None:
                raise ValueError("Data must be provided for simplified TFT implementation")

            # Prepare data for prediction
            X_test, _ = create_sequences(
                data,
                self.hyperparams["context_length"],
                self.hyperparams["max_prediction_length"],
                self.feature_columns,
                self.target_column,
                is_train=False,
            )

            # Convert to tensor
            X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

            # Make predictions
            with torch.no_grad():
                predictions = self.model(X_test).cpu().numpy()

            return predictions

        else:
            # Full PyTorch Forecasting prediction
            if self.training_dataset is None:
                raise ValueError("Model must be trained before making predictions")

            assert self.model is not None, "Model must be trained before making predictions"

            # If data is provided, create a test dataset
            if data is not None:
                # Create test dataset
                test_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, data, stop_randomization=True)
                test_dataloader = test_dataset.to_dataloader(batch_size=self.hyperparams["batch_size"], train=False)

                # Make predictions
                predictions = self.model.predict(test_dataloader, mode="prediction", return_x=False, return_y=False)  # type: ignore[union-attr]

                if return_ci:
                    # Generate predictions with quantiles for confidence intervals
                    raw_predictions = self.model.predict(test_dataloader, mode="raw", return_x=True, return_y=True)  # type: ignore[union-attr]

                    # Extract quantiles
                    lower_quantile = raw_predictions.output.quantiles(0.1).cpu().numpy()
                    upper_quantile = raw_predictions.output.quantiles(0.9).cpu().numpy()

                    # Extract mean predictions
                    mean_predictions = predictions.cpu().numpy()

                    return {"predictions": mean_predictions, "lower_ci": lower_quantile, "upper_ci": upper_quantile}
                else:
                    return predictions.cpu().numpy()

            elif horizon is not None:
                # Use encoder data from training dataset to make future predictions
                # This is a simplified approach for demonstration
                train_dataloader = self.training_dataset.to_dataloader(batch_size=1, train=False)

                # Get the first batch to use as encoder data
                x, _ = next(iter(train_dataloader))

                # Make predictions
                predictions = self.model.predict(x, mode="prediction", n_samples=1)  # type: ignore[union-attr]

                # Return the specified horizon
                return predictions[:, : min(horizon, self.hyperparams["max_prediction_length"])].cpu().numpy()  # type: ignore[index]

            else:
                raise ValueError("Either data or horizon must be provided")

    @log_execution
    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path where to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata and hyperparameters
        metadata_path = save_path.with_suffix(".metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "hyperparams": self.hyperparams,
                    "training_params": self.training_params,
                    "has_pytorch_forecasting": self.has_pytorch_forecasting,
                },
                f,
            )

        if not self.has_pytorch_forecasting:
            # Save simplified model
            if self.model is not None:
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Saved simplified TFT model to {save_path}")
        else:
            # Save full TFT model
            if self.model is not None:
                self.model.save(save_path)  # type: ignore[operator]
                logger.info(f"Saved TFT model to {save_path}")

    @log_execution
    def load_model(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        load_path = Path(path)

        # Load metadata and hyperparameters
        metadata_path = load_path.with_suffix(".metadata.pkl")
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                saved_data = pickle.load(f)
                self.metadata = saved_data.get("metadata", self.metadata)
                self.hyperparams = saved_data.get("hyperparams", self.hyperparams)
                self.training_params = saved_data.get("training_params", self.training_params)
                saved_has_pytorch_forecasting = saved_data.get("has_pytorch_forecasting", self.has_pytorch_forecasting)

                # Check if saved with different PyTorch Forecasting availability
                if saved_has_pytorch_forecasting != self.has_pytorch_forecasting:
                    logger.warning(
                        f"Model was saved with PyTorch Forecasting {'available' if saved_has_pytorch_forecasting else 'unavailable'} "
                        f"but now it is {'available' if self.has_pytorch_forecasting else 'unavailable'}. "
                        f"This may cause compatibility issues."
                    )

        if not self.has_pytorch_forecasting:
            # Load simplified model
            # First create model structure
            if hasattr(self, "feature_columns") and hasattr(self, "target_column"):
                # Create a dummy model with correct dimensions
                seq_len = self.hyperparams["context_length"]
                n_features = len(self.feature_columns) if hasattr(self, "feature_columns") else 1
                self.model = self._create_simple_model((seq_len, n_features))

                # Load state dict
                assert self.model is not None
                self.model.load_state_dict(torch.load(load_path, map_location=self.device))
                self.model.eval()
                logger.info(f"Loaded simplified TFT model from {load_path}")
            else:
                logger.error("Cannot load model: missing feature information")
                raise ValueError("Cannot load model: missing feature information")
        else:
            # Load full TFT model
            if os.path.exists(load_path):
                self.model = TemporalFusionTransformer.load_from_checkpoint(load_path)
                logger.info(f"Loaded TFT model from {load_path}")
            else:
                logger.error(f"Model file not found at {load_path}")
                raise FileNotFoundError(f"Model file not found at {load_path}")

    @log_execution
    def cleanup(self):
        """Clean up resources used by the model."""
        if self.model is not None:
            # Remove model from GPU if applicable
            if hasattr(self.model, "to"):
                self.model.to("cpu")

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Cleaned up TFT model resources")

    @log_execution
    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters."""
        return self.hyperparams.copy()

    @log_execution
    def set_params(self, **params: Any) -> None:
        """
        Set model hyperparameters.

        Args:
            **params: Model hyperparameters to set
        """
        for key, value in params.items():
            if key in self.hyperparams:
                self.hyperparams[key] = value

        # Update metadata
        if hasattr(self, "metadata") and isinstance(self.metadata, dict):
            if "params" in self.metadata:
                self.metadata["params"].update(params)
            else:
                self.metadata["params"] = params.copy()

            self.metadata["updated_at"] = pd.Timestamp.now().isoformat()


class ConcreteTFTModel(TFTModel):
    """
    Concrete implementation of the TFT model with default hyperparameters.

    This class ensures all abstract methods from BaseModel are properly implemented
    while maintaining the functionality of the TFTModel parent class.
    """

    def __init__(self, **kwargs):
        """Initialize the model with default hyperparameters."""
        super().__init__(**kwargs)

    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs) -> "BaseModel":  # type: ignore[override]
        """
        Fit the TFT model to the provided dataset.

        Args:
            train_data: Data to train on
            target_column: Name of the target column
            **kwargs: Additional parameters for training

        Returns:
            self: For method chaining
        """
        # Call the existing train method with appropriate arguments
        self.train(data=train_data, target_column=target_column, **kwargs)
        return self

    def generate_samples(self, n_samples=10, **kwargs):
        """
        Generate samples from the TFT model by adding noise to input features.

        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters, must include 'input_data'

        Returns:
            List of predictions representing the generated samples

        Raises:
            ValueError: If model not trained or input_data not provided
        """
        if not self.has_pytorch_forecasting or self.model is None:
            raise ValueError("Model not trained or PyTorch Forecasting not available")

        # For TFT, we need input data to generate samples
        input_data = kwargs.get("input_data")
        if input_data is None:
            raise ValueError("input_data required for TFT sample generation")

        # Generate n different predictions by adding noise to inputs
        samples = []
        for _i in range(n_samples):
            # Add small random noise to input features
            noisy_input = input_data.copy()
            if isinstance(noisy_input, pd.DataFrame):
                for col in noisy_input.select_dtypes(include=[np.number]).columns:
                    noisy_input[col] = noisy_input[col] * (1 + np.random.normal(0, 0.01))

            # Generate prediction with the noisy input
            pred = self.predict(noisy_input)
            samples.append(pred)

        return samples
