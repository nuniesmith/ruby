from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE, MAPE, RMSE

import numpy as np
import pandas as pd

from lib.model._shims import logger

try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

try:
    import torch as _torch  # type: ignore[import-untyped]

    HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    HAS_TORCH = False

if not TYPE_CHECKING:
    torch = _torch  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

try:
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting import TimeSeriesDataSet as _TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE as _MAE
    from pytorch_forecasting.metrics import MAPE as _MAPE
    from pytorch_forecasting.metrics import RMSE as _RMSE

    HAS_PYTORCH_FORECASTING = True
except ImportError:
    TemporalFusionTransformer = None  # type: ignore[assignment]
    _TimeSeriesDataSet = None  # type: ignore[assignment]
    _MAPE = None  # type: ignore[assignment]
    _MAE = None  # type: ignore[assignment]
    _RMSE = None  # type: ignore[assignment]
    GroupNormalizer = None  # type: ignore[assignment]
    HAS_PYTORCH_FORECASTING = False

if not TYPE_CHECKING:
    TimeSeriesDataSet = _TimeSeriesDataSet  # type: ignore[assignment,misc]
    MAE = _MAE  # type: ignore[assignment,misc]
    MAPE = _MAPE  # type: ignore[assignment,misc]
    RMSE = _RMSE  # type: ignore[assignment,misc]


class EvaluationService:
    def __init__(
        self,
        config_path: str,
        model_path: str,
        output_dir: str = "eval_outputs",
        accelerator: str = "cuda",
        interpret: bool = False,
        series_id: str | None = None,
    ):
        """
        Initialize the evaluation service.

        Args:
            config_path: Path to the YAML configuration file
            model_path: Path to the trained model checkpoint
            output_dir: Directory to save evaluation outputs
            accelerator: Hardware accelerator to use ('cpu', 'cuda', 'mps')
            interpret: Whether to generate model interpretation
            series_id: Optional identifier for the time series (used for filtering/grouping)
        """
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.interpret = interpret
        self.series_id = series_id

        # Initialize attributes
        self.config = None
        self.model = None
        self.training_dataset = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.evaluation_results: dict[str, Any] = {}
        self.df_train = None
        self.df_test = None
        self.df_val = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Evaluation service initialized")

    def load_config(self) -> bool:
        """Load and validate configuration file."""
        try:
            logger.info(f"Loading configuration from: {self.config_path}")
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

            if not isinstance(self.config, dict):
                logger.error("Configuration file did not produce a valid mapping")
                return False

            config: dict[str, Any] = self.config

            # Ensure dataset section has default empty lists for critical parameters
            if "dataset" in config:
                dataset_defaults: dict[str, Any] = {
                    "time_varying_known_categoricals": [],
                    "time_varying_unknown_categoricals": [],
                    "static_categoricals": [],
                    "static_reals": [],
                }

                for key, default_value in dataset_defaults.items():
                    if key not in config["dataset"]:
                        logger.warning(f"Missing '{key}' in dataset config. Using empty list as default.")
                        config["dataset"][key] = default_value

            if not self._validate_config(config):
                logger.error("Invalid configuration file")
                return False

            # Log dataset configuration for debugging
            if "dataset" in config:
                logger.debug("Dataset configuration:")
                for key in ["time_idx", "target", "group_ids"]:
                    if key in config["dataset"]:
                        logger.debug(f"  {key}: {config['dataset'][key]}")

                # Log categorical and real variables
                for key in [
                    "time_varying_known_categoricals",
                    "time_varying_unknown_categoricals",
                    "time_varying_known_reals",
                    "time_varying_unknown_reals",
                ]:
                    if key in config["dataset"]:
                        logger.debug(f"  {key}: {config['dataset'][key]}")

            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Validate that the config has the required sections and parameters."""
        required_sections = ["data", "dataset", "training"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing '{section}' section in config")
                return False

        # Validate data section
        required_data_params = ["train_path", "test_path"]
        for param in required_data_params:
            if param not in config["data"]:
                logger.error(f"Missing '{param}' in data section")
                return False

        # Check if paths exist
        for path_key in ["train_path", "test_path", "val_path"]:
            if path_key in config["data"]:
                path = config["data"][path_key]
                if not os.path.exists(path):
                    logger.warning(f"Path specified in config '{path_key}': {path} does not exist")
                    # We don't return False here since input_data might override this

        # Validate dataset section
        required_dataset_params = [
            "time_idx",
            "target",
            "group_ids",
            "min_encoder_length",
            "max_encoder_length",
            "min_prediction_length",
            "max_prediction_length",
            "time_varying_known_reals",
            "time_varying_known_categoricals",
            "time_varying_unknown_reals",
            "time_varying_unknown_categoricals",  # Critical for TFT
            "static_categoricals",
            "static_reals",
        ]

        missing_params = []
        for param in required_dataset_params:
            if param not in config["dataset"]:
                missing_params.append(param)

        if missing_params:
            logger.error(f"Missing required dataset parameters: {', '.join(missing_params)}")
            return False

        # Validate training section
        if "batch_size" not in config["training"]:
            logger.error("Missing 'batch_size' in training section")
            return False

        return True

    def _validate_data_columns(self, df: pd.DataFrame, required_columns: list[str]) -> bool:
        """Check if all required columns exist in the dataframe."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in data: {', '.join(missing_columns)}")
            return False
        return True

    def load_data(self, input_data=None) -> bool:
        """Load train, test, and optional validation datasets."""
        try:
            # Check if config is loaded
            if self.config is None:
                logger.error("Configuration not loaded. Call load_config() before load_data().")
                return False

            # Validate input_data if provided
            if input_data is not None:
                if not os.path.exists(input_data):
                    logger.error(f"Input data path does not exist: {input_data}")
                    return False
                logger.info(f"Using provided input data: {input_data}")

            logger.info("Loading datasets...")

            # Use input_data path if provided, otherwise use config path
            test_path = input_data if input_data else self.config["data"]["test_path"]
            logger.info(f"Using test data from: {test_path}")

            # Load datasets with error handling
            try:
                self.df_train = pd.read_parquet(self.config["data"]["train_path"])
                logger.info(f"Loaded training data with shape {self.df_train.shape}")
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                return False

            try:
                self.df_test = pd.read_parquet(test_path)
                logger.info(f"Loaded test data with shape {self.df_test.shape}")
            except Exception as e:
                logger.error(f"Failed to load test data: {e}")
                return False

            # Add loading for validation data
            self.df_val = None
            if "val_path" in self.config["data"] and os.path.exists(self.config["data"]["val_path"]):
                try:
                    self.df_val = pd.read_parquet(self.config["data"]["val_path"])
                    logger.info(f"Loaded validation data with shape {self.df_val.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load validation data: {e}. Skipping validation metrics.")
            else:
                logger.warning("Validation path not specified or file not found. Skipping validation metrics.")

            # If series_id is provided, ensure it exists in the dataframes
            if self.series_id is not None:
                # For test data
                if "series_id" not in self.df_test.columns:
                    logger.info(f"Adding series_id '{self.series_id}' to test data")
                    self.df_test["series_id"] = self.series_id

                # For validation data if it exists
                if self.df_val is not None and "series_id" not in self.df_val.columns:
                    logger.info(f"Adding series_id '{self.series_id}' to validation data")
                    self.df_val["series_id"] = self.series_id

                # For training data
                if "series_id" not in self.df_train.columns:
                    logger.info(f"Adding series_id '{self.series_id}' to training data")
                    self.df_train["series_id"] = self.series_id

            # Validate required columns exist in the data
            required_columns = [self.config["dataset"]["time_idx"], self.config["dataset"]["target"]]
            required_columns.extend(self.config["dataset"]["group_ids"])

            if not self._validate_data_columns(self.df_train, required_columns):
                return False

            if not self._validate_data_columns(self.df_test, required_columns):
                return False

            if self.df_val is not None and not self._validate_data_columns(self.df_val, required_columns):
                logger.warning("Validation data lacks required columns. Skipping validation.")
                self.df_val = None

            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def create_datasets(self) -> bool:
        """Create training dataset and dataloaders for test and validation sets."""
        try:
            # Check if config is loaded
            if self.config is None:
                logger.error("Configuration not loaded. Call load_config() before create_datasets().")
                return False

            # Check if data is loaded
            if self.df_train is None or self.df_test is None:
                logger.error("Data not loaded. Call load_data() before create_datasets().")
                return False

            # Validate feature columns exist in data
            dataset_config = self.config["dataset"]
            all_feature_columns = []

            for feature_list_name in [
                "time_varying_known_categoricals",
                "time_varying_unknown_categoricals",
                "time_varying_known_reals",
                "time_varying_unknown_reals",
                "static_categoricals",
                "static_reals",
            ]:
                if feature_list_name in dataset_config:
                    all_feature_columns.extend(dataset_config[feature_list_name])

            # Add required columns
            all_feature_columns.append(dataset_config["time_idx"])
            all_feature_columns.append(dataset_config["target"])
            all_feature_columns.extend(dataset_config["group_ids"])

            # Remove duplicates
            all_feature_columns = list(set(all_feature_columns))

            # Check if columns exist in training data
            missing_columns = [col for col in all_feature_columns if col not in self.df_train.columns]
            if missing_columns:
                logger.error(f"Missing columns in training data: {missing_columns}")
                logger.error("These columns are required by the dataset configuration.")
                return False

            # Create training dataset
            try:
                self.training_dataset = self._create_dataset(self.config, self.df_train, is_training=True)
                logger.info("Successfully created training dataset")
            except KeyError as ke:
                logger.error(f"KeyError when creating training dataset: {ke}")
                logger.error("This is likely due to a missing column in the data or configuration mismatch.")
                return False
            except Exception as e:
                logger.error(f"Failed to create training dataset: {e}")
                return False

            # Create test dataset and dataloader
            try:
                test_dataset = TimeSeriesDataSet.from_dataset(
                    self.training_dataset, self.df_test.reset_index(drop=True), predict=True, stop_randomization=True
                )
                batch_size = self.config["training"]["batch_size"]
                self.test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
                logger.info("Successfully created test dataloader")
            except Exception as e:
                logger.error(f"Failed to create test dataset: {e}")
                return False

            # Create validation dataloader if validation data available
            if self.df_val is not None:
                try:
                    self.val_dataloader = self._validation_dataloader_from_training_dataset(
                        self.training_dataset, self.df_val.reset_index(drop=True), self.config["training"]["batch_size"]
                    )
                    logger.info("Successfully created validation dataloader")
                except Exception as e:
                    logger.warning(f"Failed to create validation dataloader: {e}")
                    self.val_dataloader = None

            return True
        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            return False

    def _create_dataset(self, config, df, is_training=True) -> TimeSeriesDataSet:
        """
        Create a TimeSeriesDataSet from a configuration and dataframe.

        Args:
            config: Configuration dictionary with dataset parameters
            df: DataFrame with time series data
            is_training: Whether this is for training (True) or inference (False)

        Returns:
            TimeSeriesDataSet object
        """
        logger.info(f"Creating {'training' if is_training else 'evaluation'} dataset")

        # Extract dataset configuration
        dataset_config = config["dataset"]

        # Log configuration being used
        logger.debug("TimeSeriesDataSet configuration:")
        logger.debug(f"  time_idx: {dataset_config['time_idx']}")
        logger.debug(f"  target: {dataset_config['target']}")
        logger.debug(f"  group_ids: {dataset_config['group_ids']}")
        logger.debug(f"  time_varying_known_categoricals: {dataset_config['time_varying_known_categoricals']}")
        logger.debug(f"  time_varying_unknown_categoricals: {dataset_config['time_varying_unknown_categoricals']}")

        # Ensure all required lists exist (with defaults if missing)
        time_varying_known_categoricals = dataset_config.get("time_varying_known_categoricals", [])
        time_varying_unknown_categoricals = dataset_config.get("time_varying_unknown_categoricals", [])
        time_varying_known_reals = dataset_config.get("time_varying_known_reals", [])
        time_varying_unknown_reals = dataset_config.get("time_varying_unknown_reals", [])
        static_categoricals = dataset_config.get("static_categoricals", [])
        static_reals = dataset_config.get("static_reals", [])

        # Create dataset
        assert TimeSeriesDataSet is not None and GroupNormalizer is not None
        dataset = TimeSeriesDataSet(
            data=df,
            time_idx=dataset_config["time_idx"],
            target=dataset_config["target"],
            group_ids=dataset_config["group_ids"],
            min_encoder_length=dataset_config["min_encoder_length"],
            max_encoder_length=dataset_config["max_encoder_length"],
            min_prediction_length=dataset_config["min_prediction_length"] if is_training else 1,
            max_prediction_length=dataset_config["max_prediction_length"],
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=GroupNormalizer(
                groups=dataset_config["group_ids"],
                transformation=dataset_config.get("target_transformation", "softplus"),
            ),
        )

        return dataset

    def _validation_dataloader_from_training_dataset(self, training_dataset, df_val, batch_size: int):
        """
        Create a validation dataloader from a training dataset.

        Args:
            training_dataset: The training TimeSeriesDataSet
            df_val: DataFrame containing validation data
            batch_size: Batch size for the dataloader

        Returns:
            DataLoader for validation data
        """
        # Create validation dataset from the training dataset
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, df_val, predict=False, stop_randomization=True
        )

        # Create validation dataloader
        validation_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        return validation_dataloader

    def load_model(self) -> bool:
        """Load the trained TFT model from checkpoint."""
        try:
            logger.info(f"Loading model from checkpoint: {self.model_path}")
            if not os.path.exists(self.model_path):
                logger.error(f"Model checkpoint file not found: {self.model_path}")
                return False

            self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)

            # Set model to evaluation mode and move to correct device
            assert self.model is not None
            self.model.eval()
            if self.accelerator == "cuda" and torch.cuda.is_available():  # type: ignore[union-attr]
                self.model.to("cuda")
                logger.info("Using CUDA accelerator")
            elif self.accelerator == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[union-attr]
                self.model.to("mps")
                logger.info("Using MPS accelerator")
            else:
                logger.info("No GPU available, using CPU")
                # CPU is default so no need to move model

            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def evaluate_test_set(self) -> dict[str, float]:
        """
        Evaluate the model on the test set and return metrics.

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return {}

            if self.test_dataloader is None:
                logger.error("Test dataloader not created. Call create_datasets() first.")
                return {}

            logger.info("Evaluating model on test set...")

            # Get predictions using helper method
            predictions, targets = self.get_predictions(self.test_dataloader)
            if len(predictions) == 0:  # Fixed: check for empty tensor instead of None
                logger.error("Failed to get predictions for test set")
                return {}

            # Create evaluation metrics
            assert MAE is not None and RMSE is not None and MAPE is not None
            metrics = {"mae": MAE(), "rmse": RMSE(), "mape": MAPE()}

            # Compute metrics
            metric_results = {}
            for name, metric in metrics.items():
                value = metric(predictions, targets)
                metric_results[name] = value.item()

            logger.info(f"Test metrics: {metric_results}")

            # Generate and save test predictions plot
            self._save_predictions_plot(predictions, targets, "test_predictions.png")

            # Store metrics in evaluation results
            self.evaluation_results["test"] = {"metrics": metric_results, "n_samples": len(targets)}

            return metric_results

        except Exception as e:
            logger.error(f"Error evaluating test set: {e}")
            return {}

    def evaluate_validation_set(self) -> dict[str, float]:
        """
        Evaluate the model on the validation set and return metrics.
        Only runs if a validation dataloader was created.

        Returns:
            Dictionary of evaluation metrics (empty if validation data not available)
        """
        if self.val_dataloader is None:
            logger.warning("No validation dataloader available. Skipping validation evaluation.")
            return {}

        try:
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return {}

            logger.info("Evaluating model on validation set...")

            # Get predictions using helper method
            predictions, targets = self.get_predictions(self.val_dataloader)
            if len(predictions) == 0:  # Fixed: check for empty tensor instead of None
                logger.error("Failed to get predictions for validation set")
                return {}

            # Create evaluation metrics
            assert MAE is not None and RMSE is not None and MAPE is not None
            metrics = {"mae": MAE(), "rmse": RMSE(), "mape": MAPE()}

            # Compute metrics
            metric_results = {}
            for name, metric in metrics.items():
                value = metric(predictions, targets)
                metric_results[name] = value.item()

            logger.info(f"Validation metrics: {metric_results}")

            # Generate and save validation predictions plot
            self._save_predictions_plot(predictions, targets, "validation_predictions.png")

            # Store metrics in evaluation results
            self.evaluation_results["validation"] = {"metrics": metric_results, "n_samples": len(targets)}

            return metric_results

        except Exception as e:
            logger.error(f"Error evaluating validation set: {e}")
            return {}

    def get_predictions(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get raw predictions and targets from a dataloader.
        Args:
            dataloader: DataLoader containing batches of data
        Returns:
            Tuple of (predictions, targets) tensors; empty tensors on error
        """
        try:
            # Extract batch structure from first batch to understand the format
            for batch in dataloader:
                logger.debug(f"Batch type: {type(batch)}")
                if isinstance(batch, tuple):
                    if len(batch) >= 2:
                        targets = batch[1]  # Second element typically contains targets
                        if isinstance(targets, tuple) and len(targets) > 0:
                            targets = targets[0]  # First element of target tuple
                        logger.debug(f"Target shape: {targets.shape if hasattr(targets, 'shape') else 'unknown'}")
                    else:
                        logger.error("Tuple batch doesn't have enough elements")
                        return torch.tensor([]), torch.tensor([])
                elif isinstance(batch, dict):
                    targets = batch["target"]
                    logger.debug(f"Target shape: {targets.shape}")
                else:
                    logger.error(f"Unexpected batch type: {type(batch)}")
                    return torch.tensor([]), torch.tensor([])
                break  # Just check the first batch

            logger.info("Using CPU-based batch-by-batch prediction for TFT")
            with torch.no_grad():
                if self.model is None:
                    logger.error("Model not loaded")
                    return torch.tensor([]), torch.tensor([])

                # Temporarily move model to CPU to avoid CUDA memory issues
                original_device = self.model.device
                self.model = self.model.cpu()

                # Process in smaller batches to avoid CUDA errors
                all_predictions = []
                all_targets = []

                try:
                    # Create a new dataloader with smaller batch size if needed
                    batch_size = min(32, dataloader.batch_size)  # Use smaller batch size
                    if batch_size < dataloader.batch_size:
                        logger.info(f"Using smaller batch size of {batch_size} for prediction")
                        # Recreate dataset with smaller batches
                        prediction_dataloader = dataloader.dataset.to_dataloader(
                            train=False, batch_size=batch_size, num_workers=0
                        )
                    else:
                        prediction_dataloader = dataloader

                    # Process each batch individually
                    for i, batch in enumerate(prediction_dataloader):
                        if i % 10 == 0:  # Log progress every 10 batches
                            logger.debug(f"Processing batch {i}")

                        # Extract x (features) from batch
                        if isinstance(batch, tuple) and len(batch) > 0:
                            x = batch[0]
                            # Extract target
                            if len(batch) > 1:
                                if isinstance(batch[1], tuple) and len(batch[1]) > 0:
                                    target = batch[1][0].cpu()
                                else:
                                    target = batch[1].cpu()
                                all_targets.append(target)
                        else:
                            logger.error("Unexpected batch structure")
                            continue

                        # Direct forward pass on CPU
                        try:
                            # For TimeSeriesDataset batches, we can use direct model prediction
                            outputs = self.model(x)  # Direct forward pass
                            prediction = outputs.prediction.detach().cpu()
                            all_predictions.append(prediction)
                        except Exception as batch_e:
                            logger.error(f"Error in batch {i}: {batch_e}")
                            import traceback

                            logger.debug(f"Batch error traceback: {traceback.format_exc()}")
                            continue

                    # Move model back to original device
                    self.model = self.model.to(original_device)

                    # Combine results
                    if not all_predictions:
                        logger.error("No predictions were generated")
                        return torch.tensor([]), torch.tensor([])

                    predictions = torch.cat(all_predictions)
                    targets = torch.cat(all_targets)

                    # Ensure predictions and targets have matching dimensions
                    min_len = min(len(predictions), len(targets))
                    if min_len == 0:
                        logger.error("No valid predictions/targets to evaluate")
                        return torch.tensor([]), torch.tensor([])

                    logger.debug(f"Final shapes - predictions: {predictions.shape}, targets: {targets.shape}")
                    predictions = predictions[:min_len]
                    targets = targets[:min_len]

                    return predictions, targets

                except Exception as e:
                    # Move model back to original device before raising
                    self.model = self.model.to(original_device)
                    logger.error(f"Error in prediction: {e}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return torch.tensor([]), torch.tensor([])

        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return torch.tensor([]), torch.tensor([])

    def _save_predictions_plot(self, predictions, targets, filename):
        """Save a plot of predictions vs targets."""
        try:
            plt.figure(figsize=(12, 8))

            # Extract shapes for debugging
            pred_shape = predictions.shape
            target_shape = targets.shape

            logger.info(f"Prediction shape: {pred_shape}, Target shape: {target_shape}")

            # Plot the actual values
            plt.plot(targets[:100].cpu().numpy(), label="Actual")

            # For multi-horizon predictions, plot first and last prediction horizons
            if len(pred_shape) == 3:  # Shape: [samples, horizons, features]
                # Plot short-term prediction (first horizon)
                plt.plot(predictions[:100, 0, 0].cpu().numpy(), label="Predicted (t+1)")

                # Plot long-term prediction (last horizon)
                if pred_shape[1] > 1:
                    plt.plot(predictions[:100, -1, 0].cpu().numpy(), label=f"Predicted (t+{pred_shape[1]})")
            else:
                # Fall back to direct plotting if predictions are already 1D/2D
                plt.plot(predictions[:100].cpu().numpy(), label="Predicted")

            plt.legend()
            plt.title("Actual vs Predicted Values (First 100 samples)")
            plt.ylabel("Value")
            plt.xlabel("Sample")
            plt.tight_layout()

            # Save plot to output directory
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Saved predictions plot to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create predictions plot: {e}")
            logger.debug(f"Error details: {str(e)}")

    def interpret_model(self):
        """Generate model interpretation visualizations."""
        if not self.interpret:
            logger.info("Model interpretation disabled. Skipping.")
            return

        try:
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                return

            logger.info("Generating model interpretation...")

            # Get a batch of data for interpretation
            if self.test_dataloader is None:
                logger.error("Test dataloader is not available; cannot perform model interpretation.")
                return
            interpretation_batch = next(iter(self.test_dataloader))

            # Move batch to model device
            device = self.model.device
            interpretation_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in interpretation_batch.items()
            }

            # Get variable importance with newer vs older API support
            try:
                # Try newer API first
                interpretation = self.model.interpret_output(
                    interpretation_batch, attention_prediction_horizon=0, reduction="mean"
                )
            except (AttributeError, TypeError):
                # Fall back to older API if needed
                interpretation = self.model.interpret_output(interpretation_batch)

            # Extract variable importances
            variable_importances = interpretation["variable_importance"]

            # Convert to list of dictionaries for easier serialization
            importance_list = []
            for _idx, (var, imp) in enumerate(
                zip(interpretation["variables"], variable_importances.mean(0), strict=False)
            ):
                importance_list.append({"variable": var, "importance": float(imp)})

            # Sort by importance
            importance_list = sorted(importance_list, key=lambda x: x["importance"], reverse=True)

            # Plot and save variable importance
            self._plot_variable_importance(importance_list)

            # Store in results
            self.evaluation_results["interpretation"] = {"importance": importance_list}

            logger.info("Model interpretation completed")

        except Exception as e:
            logger.error(f"Error in model interpretation: {e}")

    def _plot_variable_importance(self, importance_list):
        """Plot and save variable importance."""
        try:
            # Extract variables and importance values
            variables = [item["variable"] for item in importance_list]
            importances = [item["importance"] for item in importance_list]

            # Create plot
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(variables))
            plt.barh(y_pos, importances)
            plt.yticks(y_pos, variables)
            plt.xlabel("Importance")
            plt.title("Variable Importance")
            plt.tight_layout()

            # Save plot
            importance_plot_path = os.path.join(self.output_dir, "variable_importance.png")
            plt.savefig(importance_plot_path)
            plt.close()

            logger.info(f"Saved variable importance plot to {importance_plot_path}")

        except Exception as e:
            logger.error(f"Failed to create variable importance plot: {e}")

    def run_evaluation(self, input_data=None) -> dict[str, Any]:
        """Run the complete evaluation pipeline and return results."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate input_data if provided
        if input_data is not None:
            if not os.path.exists(input_data):
                logger.error(f"Input data path does not exist: {input_data}")
                return {"status": "error", "message": f"Input data path not found: {input_data}"}
            logger.info(f"Using provided input data: {input_data}")

        # Sequential pipeline execution with early return on failure
        if not self.load_config():
            return {"status": "error", "message": "Failed to load configuration"}

        if not self.load_data(input_data):  # Pass input_data to load_data
            return {"status": "error", "message": "Failed to load datasets"}

        try:
            if not self.create_datasets():
                return {"status": "error", "message": "Failed to create datasets"}
        except KeyError as ke:
            error_msg = f"Missing required key in configuration: {ke}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        if not self.load_model():
            return {"status": "error", "message": "Failed to load model"}

        # Run evaluations
        test_metrics = self.evaluate_test_set()
        if not test_metrics:
            return {"status": "error", "message": "Failed to evaluate on test set"}

        # Optional validation evaluation
        self.evaluate_validation_set()

        # Optional model interpretation
        if self.interpret:
            self.interpret_model()

        # Compile final results
        results = {
            "status": "success",
            "output_dir": os.path.abspath(self.output_dir),
            "results": self.evaluation_results,
        }

        # Save results to JSON file
        results_file = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            # Convert numpy values to Python native types
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))

        logger.info(f"Evaluation completed successfully. Results saved to {results_file}")
        return results

    def free_memory(self):
        """Release memory used by datasets and model."""
        logger.info("Releasing memory...")
        self.df_train = None
        self.df_test = None
        self.df_val = None
        self.training_dataset = None
        self.test_dataloader = None
        self.val_dataloader = None
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model.cpu()
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Memory freed")
