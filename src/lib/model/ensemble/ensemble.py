import json
import os
import pickle
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import logger
from lib.model.base.model import BaseModel
from lib.model.utils.metadata import ModelMetadata


class EnsembleMethod(Enum):
    """Methods for combining predictions in an ensemble"""

    WEIGHTED_AVERAGE = auto()
    MAJORITY_VOTE = auto()
    STACKING = auto()
    BAGGING = auto()
    BOOSTING = auto()
    NONE = auto()  # Added NONE option for empty ensembles


class EnsembleModel(BaseModel):
    """Enhanced ensemble model that combines multiple base models"""

    def __init__(
        self,
        models: list[BaseModel],
        weights: list[float] | None = None,
        ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        name: str | None = None,
        meta_model: BaseModel | None = None,
    ):
        """
        Initialize the ensemble model.
        Args:
            models: List of base models in the ensemble
            weights: Optional weights for weighted average prediction (normalized automatically)
            ensemble_method: Method used to combine predictions
            name: Optional name for the ensemble
            meta_model: Model used for stacking (required if ensemble_method is STACKING)
        """
        super().__init__()

        # Input validation
        if not models and ensemble_method != EnsembleMethod.NONE:
            raise ValueError("At least one model must be provided for ensemble")

        if weights is not None and len(models) > 0 and len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")

        self.models = models
        self.ensemble_method = ensemble_method
        self.name = name or f"Ensemble_{uuid.uuid4().hex[:8]}"
        self.meta_model = meta_model

        # Set up weights based on ensemble method
        if weights is not None:
            self.weights: list[float] | None = weights
            total_weight = sum(self.weights)
            if total_weight > 0:  # Avoid division by zero
                self.weights = [w / total_weight for w in self.weights]
        else:
            if ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                if len(models) > 0:  # Prevent division by zero
                    self.weights = [1.0 / len(models)] * len(models)
                else:
                    self.weights = []
            else:
                self.weights = None

        # Validate stacking configuration
        if ensemble_method == EnsembleMethod.STACKING and meta_model is None:
            raise ValueError("Meta model must be provided for stacking ensemble")

        # Initialize metadata
        self.metadata = ModelMetadata(model_type="Ensemble")
        self.metadata.params = {
            "model_names": [model.metadata.model_type for model in self.models] if models else [],
            "ensemble_method": ensemble_method.name,
            "weights": self.weights,
            "name": self.name,
        }
        self.metadata.description = f"{ensemble_method.name} ensemble of {len(models)} models"

        # Initialize performance tracking
        self.model_performances: dict[int, dict[str, Any]] = {i: {} for i in range(len(models))}

    def predict(self, data: Any) -> Any:  # type: ignore[override]
        """
        Generate predictions using the ensemble.

        Args:
            data: Input data for prediction

        Returns:
            Combined prediction based on the ensemble method
        """
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._predict_weighted_average(data)
        elif self.ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            return self._predict_majority_vote(data)
        elif self.ensemble_method == EnsembleMethod.STACKING:
            return self._predict_stacking(data)
        elif self.ensemble_method == EnsembleMethod.BAGGING:
            return self._predict_bagging(data)
        elif self.ensemble_method == EnsembleMethod.BOOSTING:
            return self._predict_boosting(data)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

    def _predict_weighted_average(self, data: Any) -> Any:
        """Generate predictions using weighted average"""
        predictions = []
        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)

        # Handle different data types (numpy arrays, pandas DataFrames, lists)
        if isinstance(predictions[0], np.ndarray):
            predictions = [p.reshape(-1, 1) if p.ndim == 1 else p for p in predictions]
            stacked = np.column_stack([p for p in predictions])
            return np.average(stacked, axis=1, weights=self.weights)
        elif isinstance(predictions[0], (pd.DataFrame, pd.Series)):
            # Convert to numpy for weighted average
            np_preds = [p.values for p in predictions]
            np_preds = [p.reshape(-1, 1) if p.ndim == 1 else p for p in np_preds]
            stacked = np.column_stack(np_preds)
            result = np.average(stacked, axis=1, weights=self.weights)
            # Convert back to DataFrame/Series
            if isinstance(predictions[0], pd.DataFrame):
                return pd.DataFrame(result, index=predictions[0].index, columns=predictions[0].columns)
            else:
                return pd.Series(result, index=predictions[0].index)
        else:
            # Assume list or scalar values
            return sum(w * p for w, p in zip(self.weights, predictions, strict=False))  # type: ignore[arg-type]

    def _predict_majority_vote(self, data: Any) -> Any:
        """Generate predictions using majority vote (for classification)"""
        predictions = []
        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)

        # Handle different data types
        if isinstance(predictions[0], np.ndarray):
            if predictions[0].ndim == 1:
                # For 1D arrays (binary/multiclass)
                stacked = np.column_stack(predictions)
                # Return the most common value for each row
                return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=stacked)
            else:
                # For 2D arrays (multi-label or probabilities)
                logger.warning("Majority vote for 2D predictions may not be appropriate")
                return self._predict_weighted_average(data)
        elif isinstance(predictions[0], pd.Series):
            # Convert to numpy for voting
            np_preds = np.column_stack([p.values for p in predictions])
            result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=np_preds)
            return pd.Series(result, index=predictions[0].index)
        else:
            # Count occurrences of each prediction and return the most common
            from collections import Counter

            return Counter(predictions).most_common(1)[0][0]

    def _predict_stacking(self, data: Any) -> Any:
        """Generate predictions using stacking"""
        if self.meta_model is None:
            raise ValueError("Meta model is required for stacking")

        # Get base model predictions as features for meta model
        meta_features = self._get_meta_features(data)

        # Use meta model to make final prediction
        return self.meta_model.predict(meta_features)

    def _predict_bagging(self, data: Any) -> Any:
        """Generate predictions using bagging (simple average)"""
        # For bagging, we use equal weights regardless of what was provided
        old_weights = self.weights
        self.weights = [1.0 / len(self.models)] * len(self.models)
        result = self._predict_weighted_average(data)
        self.weights = old_weights
        return result

    def _predict_boosting(self, data: Any) -> Any:
        """
        Generate predictions using boosting.
        Note: This is a simplified implementation and assumes models were trained in a boosting fashion.
        """
        # In a proper boosting implementation, each model would have been trained to correct
        # the errors of the previous models. Here we're using the weights to approximate that.
        return self._predict_weighted_average(data)

    def _get_meta_features(self, data: Any) -> pd.DataFrame:
        """Generate meta features for stacking"""
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(data)

            # Convert to appropriate format
            if isinstance(pred, np.ndarray):
                if pred.ndim == 1:
                    pred_df = pd.DataFrame({f"model_{i}": pred})
                else:
                    # Handle multi-output predictions
                    pred_df = pd.DataFrame(pred, columns=[f"model_{i}_out_{j}" for j in range(pred.shape[1])])
            elif isinstance(pred, pd.Series):
                pred_df = pd.DataFrame({f"model_{i}": pred})
            elif isinstance(pred, pd.DataFrame):
                # Rename columns to avoid conflicts
                pred_df = pred.copy()
                pred_df.columns = [f"model_{i}_{col}" for col in pred_df.columns]
            else:
                # Scalar or list
                pred_df = pd.DataFrame({f"model_{i}": [pred] * len(data) if hasattr(data, "__len__") else [pred]})

            predictions.append(pred_df)

        # Combine all predictions
        if isinstance(data, pd.DataFrame):
            meta_features = pd.concat(
                [data.reset_index(drop=True)] + [p.reset_index(drop=True) for p in predictions], axis=1
            )
        else:
            meta_features = pd.concat([p.reset_index(drop=True) for p in predictions], axis=1)

        return meta_features

    def save_model(self, path: str) -> None:
        """
        Save the ensemble model to disk.

        Args:
            path: Path to save the model
        """
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save each base model
        model_paths = []
        for idx, model in enumerate(self.models):
            model_path = str(model_dir / f"{self.name}_{model.metadata.model_type}_{idx}.model")
            model.save_model(model_path)
            model_paths.append(model_path)

        # Save the meta model if using stacking
        meta_model_path = None
        if self.meta_model and self.ensemble_method == EnsembleMethod.STACKING:
            meta_model_path = str(model_dir / f"{self.name}_meta_model.model")
            self.meta_model.save_model(meta_model_path)

        # Update and save metadata
        self.metadata.updated_at = datetime.now().isoformat()
        self.metadata.saved_at = datetime.now().isoformat()

        metadata = {
            "model_paths": model_paths,
            "meta_model_path": meta_model_path,
            "weights": self.weights,
            "ensemble_method": self.ensemble_method.name,
            "name": self.name,
            "model_performances": self.model_performances,
            "metadata": self.metadata.to_dict(),
        }

        # Save metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save the ensemble configuration
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "ensemble_method": self.ensemble_method,
                    "weights": self.weights,
                    "name": self.name,
                    "model_performances": self.model_performances,
                },
                f,
            )

        logger.info(f"Saved ensemble model '{self.name}' to {path}")

    @classmethod
    def load_model(cls, path: str, model_loader_func: Callable[[str], BaseModel]) -> "EnsembleModel":
        """
        Load an ensemble model from disk.

        Args:
            path: Path to load the model from
            model_loader_func: Function to load individual models (takes path, returns model)

        Returns:
            Loaded ensemble model
        """
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata not found: {metadata_path}")
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load configuration
        with open(path, "rb") as f:
            pickle.load(f)

        # Load individual models
        models = []
        for model_path in metadata.get("model_paths", []):
            if not os.path.exists(model_path):
                logger.warning(f"Model path not found: {model_path}")
                continue

            try:
                model = model_loader_func(model_path)
                models.append(model)
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {str(e)}")
                raise

        # Load meta model if using stacking
        meta_model = None
        meta_model_path = metadata.get("meta_model_path")
        if meta_model_path and os.path.exists(meta_model_path):
            try:
                meta_model = model_loader_func(meta_model_path)
            except Exception as e:
                logger.error(f"Failed to load meta model {meta_model_path}: {str(e)}")

        # Create and configure the ensemble
        ensemble_method = EnsembleMethod[metadata.get("ensemble_method", "WEIGHTED_AVERAGE")]
        weights = metadata.get("weights")
        name = metadata.get("name", f"Ensemble_{uuid.uuid4().hex[:8]}")

        ensemble = cls(
            models=models, weights=weights, ensemble_method=ensemble_method, name=name, meta_model=meta_model
        )

        # Restore performance data
        ensemble.model_performances = metadata.get("model_performances", {})

        # Restore metadata
        if "metadata" in metadata:
            ensemble.metadata = ModelMetadata.from_dict(metadata["metadata"])
            ensemble.metadata.loaded_at = datetime.now().isoformat()

        logger.info(f"Loaded ensemble model '{ensemble.name}' from {path}")
        return ensemble

    def train(self, train_data: Any, target_column: str | None = None, **kwargs: Any) -> None:  # type: ignore[override]
        """
        Train individual models in the ensemble.

        Args:
            train_data: Training data
            target_column: Target column name (if applicable)
            **kwargs: Additional training arguments
        """
        logger.info(f"Training {len(self.models)} individual models in the ensemble")

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i + 1}/{len(self.models)}: {model.metadata.model_type}")
            try:
                if target_column is not None:
                    model.train(train_data, target_column=target_column, **kwargs)
                else:
                    model.train(train_data, **kwargs)

                # Store model performance if available
                if hasattr(model, "metadata") and hasattr(model.metadata, "metrics"):
                    self.model_performances[i] = model.metadata.metrics

            except Exception as e:
                logger.error(f"Error training model {i}: {str(e)}")
                raise

        # Train meta model for stacking
        if self.ensemble_method == EnsembleMethod.STACKING and self.meta_model is not None:
            logger.info("Training meta model for stacking")

            # Generate meta features
            meta_features = self._get_meta_features(train_data)

            # Train meta model
            self.meta_model.train(meta_features, target_column=target_column, **kwargs)

        self.metadata.updated_at = datetime.now().isoformat()

        # Update overall ensemble metrics
        self._update_ensemble_metrics()

    def _update_ensemble_metrics(self) -> None:
        """Update ensemble metrics based on individual model metrics"""
        if not self.model_performances:
            return

        # Aggregate metrics from all models
        all_metrics: dict[str, list[Any]] = {}
        for model_metrics in self.model_performances.values():
            for metric, value in model_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        # Calculate aggregate statistics
        for metric, values in all_metrics.items():
            self.metadata.metrics[f"mean_{metric}"] = np.mean(values)
            self.metadata.metrics[f"min_{metric}"] = np.min(values)
            self.metadata.metrics[f"max_{metric}"] = np.max(values)
            self.metadata.metrics[f"std_{metric}"] = np.std(values)

    def evaluate(self, test_data: Any, target_column: str | None = None, **kwargs: Any) -> dict[str, float]:
        """
        Evaluate the ensemble model.

        Args:
            test_data: Test data
            target_column: Target column name (if applicable)
            **kwargs: Additional evaluation arguments

        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate individual models
        for i, model in enumerate(self.models):
            if hasattr(model, "evaluate"):
                try:
                    eval_kwargs = {"target_column": target_column, **kwargs}
                    model_any: Any = model
                    model_metrics = model_any.evaluate(test_data, **eval_kwargs)
                    self.model_performances[i] = model_metrics
                except Exception as e:
                    logger.error(f"Error evaluating model {i}: {str(e)}")

        # Update ensemble metrics
        self._update_ensemble_metrics()

        return self.metadata.metrics

    def optimize_weights(
        self, validation_data: Any, target_column: str | None = None, metric: str = "accuracy", **kwargs: Any
    ) -> list[float]:
        """
        Optimize ensemble weights using validation data.

        Args:
            validation_data: Validation data
            target_column: Target column name (if applicable)
            metric: Metric to optimize (must be implemented in each model's evaluate method)
            **kwargs: Additional arguments

        Returns:
            Optimized weights
        """
        if self.ensemble_method != EnsembleMethod.WEIGHTED_AVERAGE:
            logger.warning(f"Weight optimization only applicable for WEIGHTED_AVERAGE, not {self.ensemble_method}")
            return self.weights  # type: ignore[return-value]

        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy is required for weight optimization")
            return self.weights  # type: ignore[return-value]

        # Get individual model predictions
        y_true = validation_data[target_column] if target_column else kwargs.get("y_true")
        if y_true is None:
            raise ValueError("Target data must be provided either as a column or y_true")

        predictions = []
        for model in self.models:
            pred = model.predict(validation_data)
            predictions.append(pred)

        # Define the objective function (negative metric to minimize)
        def objective(weights):
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Weighted prediction
            if isinstance(predictions[0], (np.ndarray, pd.Series)):
                weighted_pred = sum(w * p for w, p in zip(weights, predictions, strict=False))
            else:
                weighted_pred = sum(w * p for w, p in zip(weights, predictions, strict=False))

            # Calculate metric (negative for minimization)
            try:
                from sklearn import metrics as sk_metrics
            except ImportError:
                logger.error("scikit-learn is required for weight optimization metrics")
                return 0

            if hasattr(sk_metrics, metric):
                metric_func = getattr(sk_metrics, metric)
                score = -metric_func(y_true, weighted_pred)
                return score
            else:
                logger.error(f"Metric {metric} not found in sklearn.metrics")
                return 0

        # Initial weights
        initial_weights = np.ones(len(self.models)) / len(self.models)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(len(self.models))]

        # Optimize
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

        if result.success:
            optimized_weights = result.x
            logger.info(f"Optimized weights: {optimized_weights}")
            self.weights = list(optimized_weights)

            # Update metadata
            self.metadata.params["weights"] = self.weights
            self.metadata.updated_at = datetime.now().isoformat()

            return self.weights
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
            return self.weights  # type: ignore[return-value]

    def cleanup(self) -> None:
        """Clean up resources for the ensemble model"""
        for model in self.models:
            if hasattr(model, "cleanup"):
                try:
                    model.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up ensemble component: {str(e)}")

        if self.meta_model and hasattr(self.meta_model, "cleanup"):
            try:
                self.meta_model.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up meta model: {str(e)}")

    def __repr__(self) -> str:
        """String representation of the ensemble"""
        return (
            f"{self.name} ({self.ensemble_method.name}) with {len(self.models)} models: "
            f"{[model.metadata.model_type for model in self.models]}"
        )


class ConcreteEnsembleModel(EnsembleModel):
    """Concrete implementation of EnsembleModel with specific behaviors"""

    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs: Any) -> Any:  # type: ignore[override]
        """
        Fit the ensemble on training data by fitting each base model.

        Args:
            train_data: Training data
            target_column: Target column name
            **kwargs: Additional fitting arguments

        Returns:
            Self for chaining
        """
        # Train each base model
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i + 1}/{len(self.models)}: {model.metadata.model_type}")
            try:
                model.fit(train_data, target_column, **kwargs)
            except Exception as e:
                logger.error(f"Error fitting model {i}: {str(e)}")

        # Train meta model for stacking
        if self.ensemble_method == EnsembleMethod.STACKING and self.meta_model is not None:
            logger.info("Fitting meta model for stacking")

            # Generate meta features
            X = train_data.drop(columns=[target_column])
            meta_features = self._get_meta_features(X)

            # Add target column to meta features
            meta_features[target_column] = train_data[target_column]

            # Fit meta model
            self.meta_model.fit(meta_features, target_column, **kwargs)

        self.metadata.updated_at = datetime.now().isoformat()
        return self

    def generate_samples(self, n_samples: int = 10, **kwargs: Any) -> pd.DataFrame:
        """
        Generate synthetic samples by aggregating samples from base models.

        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional generation arguments

        Returns:
            DataFrame of generated samples
        """
        models_with_generation = [model for model in self.models if hasattr(model, "generate_samples")]

        if not models_with_generation:
            raise NotImplementedError("None of the base models support sample generation")

        # Generate samples from each capable model
        all_samples = []
        for model in models_with_generation:
            try:
                samples = model.generate_samples(n_samples=n_samples // len(models_with_generation), **kwargs)
                all_samples.append(samples)
            except Exception as e:
                logger.warning(f"Error generating samples with {model.metadata.model_type}: {str(e)}")

        if not all_samples:
            raise RuntimeError("Failed to generate samples from any model")

        # Combine samples
        if isinstance(all_samples[0], pd.DataFrame):
            combined = pd.concat(all_samples, ignore_index=True)
            # Take only n_samples rows
            if len(combined) > n_samples:
                combined = combined.sample(n_samples)
            return combined
        else:
            raise TypeError(f"Unexpected sample type: {type(all_samples[0])}")


class TimeSeriesEnsemble(EnsembleModel):
    """Specialized ensemble for time series forecasting"""

    def __init__(
        self,
        models: list[BaseModel],
        weights: list[float] | None = None,
        ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,  # Fixed: Changed from EnsembleModel to EnsembleMethod
        name: str | None = None,
        meta_model: BaseModel | None = None,
        forecast_horizon: int = 1,
    ):
        """
        Initialize the time series ensemble.

        Args:
            models: List of base models in the ensemble
            weights: Optional weights for weighted average prediction
            ensemble_method: Method used to combine predictions
            name: Optional name for the ensemble
            meta_model: Model used for stacking
            forecast_horizon: Number of time steps to forecast
        """
        super().__init__(
            models=models,
            weights=weights,
            ensemble_method=ensemble_method,
            name=name or f"TSEnsemble_{uuid.uuid4().hex[:8]}",
            meta_model=meta_model,
        )

        self.forecast_horizon = forecast_horizon
        self.metadata.params["forecast_horizon"] = forecast_horizon

    def forecast(self, data: Any, steps: int | None = None, **kwargs: Any) -> pd.DataFrame:
        """
        Generate a time series forecast.

        Args:
            data: Input data for forecasting
            steps: Number of steps to forecast (defaults to self.forecast_horizon)
            **kwargs: Additional forecasting arguments

        Returns:
            DataFrame with forecasted values
        """
        steps = steps or self.forecast_horizon

        forecasts = []
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, "forecast"):
                    forecast = model.forecast(data, steps=steps, **kwargs)
                else:
                    # Fall back to predict if forecast not available
                    forecast = model.predict(data)

                forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Error forecasting with model {i}: {str(e)}")

        if not forecasts:
            raise RuntimeError("No successful forecasts generated")

        # Combine forecasts based on ensemble method
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Convert all forecasts to DataFrames for consistent handling
            forecast_dfs = []
            for forecast in forecasts:
                if isinstance(forecast, pd.DataFrame):
                    forecast_dfs.append(forecast)
                elif isinstance(forecast, pd.Series):
                    forecast_dfs.append(forecast.to_frame())
                elif isinstance(forecast, np.ndarray):
                    if forecast.ndim == 1:
                        forecast_dfs.append(pd.DataFrame(forecast.reshape(-1, 1)))
                    else:
                        forecast_dfs.append(pd.DataFrame(forecast))
                else:
                    forecast_dfs.append(pd.DataFrame([forecast]))

            # Ensure all forecasts have the same shape
            if len(set(df.shape[0] for df in forecast_dfs)) > 1:
                # Different lengths - use the minimum length
                min_length = min(df.shape[0] for df in forecast_dfs)
                forecast_dfs = [df.iloc[:min_length] for df in forecast_dfs]

            # Weighted average across models
            weighted_forecast = sum(w * df for w, df in zip(self.weights, forecast_dfs, strict=False))  # type: ignore[arg-type]

            return weighted_forecast
        elif self.ensemble_method == EnsembleMethod.STACKING:
            # Use meta model to combine forecasts
            if self.meta_model is None:
                raise ValueError("Meta model required for stacking ensemble")

            # Prepare meta features
            meta_features = pd.concat([df.reset_index(drop=True) for df in forecasts], axis=1)
            meta_features.columns = [
                f"model_{i}_forecast_{j}" for i in range(len(forecasts)) for j in range(forecasts[i].shape[1])
            ]

            # Generate final forecast with meta model
            return self.meta_model.predict(meta_features)
        else:
            # Default to first forecast for unsupported methods
            logger.warning(f"{self.ensemble_method} not fully implemented for forecasting")
            return forecasts[0]
