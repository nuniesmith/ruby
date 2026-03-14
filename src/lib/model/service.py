"""
Model service for managing predictive models with advanced lifecycle management.

This module provides a comprehensive service for:
1. Model registration and discovery
2. Model lifecycle management (training, prediction, persistence)
3. Version tracking and metrics collection
4. Flexible model interfaces
5. Gold price prediction model support
"""

import datetime
import inspect
import json
import os
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import pandas as pd

from lib.model._shims import logger

# Type variable for model types
T = TypeVar("T")


class ModelStatus(Enum):
    """Status of a model in its lifecycle."""

    INITIALIZED = auto()
    TRAINING = auto()
    TRAINED = auto()
    FAILED = auto()
    LOADED = auto()


@dataclass
class ModelVersionInfo:
    """Information about a model version."""

    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    status: ModelStatus = ModelStatus.INITIALIZED
    samples_count: int | None = None
    features: list[str] = field(default_factory=list)
    target: str | None = None
    training_params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    loaded_from: str | None = None
    error: str | None = None
    model_type: str | None = None  # Added field for model type identification

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.name
        return result


class ModelInterface(Protocol):
    """Protocol defining the base interface for models."""

    def fit(self, train_data: pd.DataFrame, target_column: str, **kwargs) -> dict[str, Any]:
        """Train the model."""
        ...

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions using the model."""
        ...


class ModelRegistry:
    """Registry of available model factories and configurations."""

    def __init__(self):
        """Initialize an empty registry."""
        self._factories: dict[str, Callable] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    def register(self, name: str, factory: Callable, config: dict[str, Any] | None = None) -> None:
        """
        Register a model factory with configuration.

        Args:
            name: Unique model name
            factory: Factory function to create model instances
            config: Default configuration for this model type
        """
        if config is None:
            config = {}
        self._factories[name] = factory
        self._configs[name] = config or {}

    def get_factory(self, name: str) -> Callable | None:
        """Get factory function for a model type."""
        return self._factories.get(name)

    def get_config(self, name: str) -> dict[str, Any]:
        """Get default configuration for a model type."""
        return self._configs.get(name, {}).copy()

    def list_models(self) -> list[str]:
        """Get list of all registered model types."""
        return list(self._factories.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a model type is registered."""
        return name in self._factories


class ModelService:
    """
    Service for managing predictive models with advanced lifecycle management.

    This service provides:
    1. Model registration and instance management
    2. Training and prediction interfaces
    3. Model versioning and metrics tracking
    4. Persistence operations (save/load)
    5. Lifecycle hooks for extensibility
    6. Gold price prediction model support
    """

    def __init__(
        self,
        selected_models: list[str] | None = None,
        config: dict[str, Any] | None = None,
        plugin_dir: str | None = None,
        model_dir: str | None = None,
    ):
        """
        Initialize the model service.

        Args:
            selected_models: List of model names to initialize (or all available if None)
            config: Configuration for model initialization
            plugin_dir: Directory of model plugins to load
            model_dir: Directory for model storage (used for loading/saving)
        """
        # Core components
        self.registry = ModelRegistry()
        self.config = config or {}
        self.model_dir = Path(model_dir) if model_dir else None

        # Model storage
        self._models: dict[str, Any] = {}
        self._versions: dict[str, ModelVersionInfo] = {}

        # Runtime state
        self._active_model: str | None = None
        self._lifecycle_hooks: dict[str, list[Callable]] = {
            "pre_train": [],
            "post_train": [],
            "pre_predict": [],
            "post_predict": [],
            "pre_save": [],
            "post_save": [],
            "pre_load": [],
            "post_load": [],
        }

        # Initialize built-in models
        self._register_builtin_models()

        # Load plugins if specified
        if plugin_dir:
            self._load_plugins(plugin_dir)

        # Select and initialize models
        available_models = self.registry.list_models()
        self.selected_models = [m for m in (selected_models or available_models) if m in available_models]
        self._initialize_selected_models()

        # Try to load any pre-existing models
        if self.model_dir:
            self._load_existing_models()

        logger.info(f"ModelService initialized with {len(self._models)} models")

    def _register_builtin_models(self) -> None:
        """Register built-in model types with the registry."""
        logger.debug("Registering built-in model types")

        # Register all model types
        self._register_standard_models()
        self._register_ml_models()
        self._register_gold_models()
        self._register_deep_learning_models()

    def _register_standard_models(self) -> None:
        """Register standard statistical models."""
        try:
            # ARIMA
            self.registry.register("ARIMA", self._create_arima_model, self.config.get("ARIMA", {}))

            # Prophet
            self.registry.register("Prophet", self._create_prophet_model, self.config.get("Prophet", {}))

            logger.debug("Registered standard statistical models")
        except Exception as e:
            logger.error(f"Error registering standard models: {e}")

    def _register_ml_models(self) -> None:
        """Register machine learning models."""
        try:
            # Linear Regression
            self.registry.register(
                "LinearRegression", self._create_linear_regression_model, self.config.get("LinearRegression", {})
            )

            # XGBoost Regression
            self.registry.register(
                "XGBoostRegressor", self._create_xgboost_regressor, self.config.get("XGBoostRegressor", {})
            )

            # RandomForest
            self.registry.register(
                "RandomForest", self._create_random_forest_model, self.config.get("RandomForest", {})
            )

            logger.debug("Registered machine learning models")
        except Exception as e:
            logger.error(f"Error registering ML models: {e}")

    def _register_gold_models(self) -> None:
        """Register gold price prediction models."""
        try:
            # Gold price direction model
            self.registry.register(
                "GoldPriceDirection",
                self._create_gold_direction_model,
                self.config.get("GoldPriceDirection", {"is_gold_model": True, "model_type": "gold_direction"}),
            )

            # Gold price regression model
            self.registry.register(
                "GoldPriceRegression",
                self._create_gold_regression_model,
                self.config.get("GoldPriceRegression", {"is_gold_model": True, "model_type": "gold_regression"}),
            )

            # Gold price ensemble model
            self.registry.register(
                "GoldPriceEnsemble",
                self._create_gold_ensemble_model,
                self.config.get("GoldPriceEnsemble", {"is_gold_model": True, "model_type": "gold_ensemble"}),
            )

            logger.debug("Registered gold price prediction models")
        except Exception as e:
            logger.error(f"Error registering gold models: {e}")

    def _register_deep_learning_models(self) -> None:
        """Register deep learning models."""
        try:
            # LSTM
            self.registry.register("LSTM", self._create_lstm_model, self.config.get("LSTM", {}))

            # Neural Network
            self.registry.register("NeuralNetwork", self._create_neural_network, self.config.get("NeuralNetwork", {}))

            logger.debug("Registered deep learning models")
        except Exception as e:
            logger.error(f"Error registering deep learning models: {e}")

    # Model factory methods

    def _create_arima_model(self, **kwargs) -> Any:
        """Create an ARIMA model instance."""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            class ARIMAModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.model = None
                    self.features: list[str] = []
                    self.results: Any = None

                def fit(self, train_data, target_column, **_kwargs):
                    self.features = [target_column]
                    # Extract time series
                    y = train_data[target_column]
                    # Create and fit ARIMA model
                    self.model = ARIMA(y, **self.params)
                    self.results = self.model.fit()
                    return {"status": "success"}

                def predict(self, X=None, steps=1, **_kwargs):
                    if self.model is None:
                        raise ValueError("Model not trained")
                    # Create forecast
                    forecast = self.results.forecast(steps=steps)
                    return forecast

                def save(self, path):
                    if self.model is None:
                        raise ValueError("Model not trained")
                    # Save model results
                    self.results.save(f"{path}.pkl")
                    return path

                def load(self, path):
                    from statsmodels.tsa.arima.model import ARIMAResults

                    self.results = ARIMAResults.load(f"{path}.pkl")
                    self.model = True  # Just a flag to indicate model is loaded
                    return True

            return ARIMAModel(**kwargs)
        except ImportError:
            logger.warning("statsmodels not available, using stub ARIMA model")
            return self._create_stub_model("ARIMA", **kwargs)

    def _create_prophet_model(self, **kwargs) -> Any:
        """Create a Prophet model instance."""
        try:
            from prophet import Prophet

            class ProphetModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.model = Prophet(**_kwargs)
                    self.features: list[str] = []

                def fit(self, train_data, target_column, date_column="ds", **_kwargs):
                    # Prepare data - Prophet requires 'ds' (date) and 'y' (target) columns
                    df = train_data.copy()
                    if date_column != "ds":
                        df = df.rename(columns={date_column: "ds"})
                    if target_column != "y":
                        df = df.rename(columns={target_column: "y"})

                    self.features = [date_column, target_column]
                    # Fit model
                    self.model.fit(df)
                    return {"status": "success"}

                def predict(self, X=None, periods=1, **_kwargs):
                    if not hasattr(self.model, "params"):  # Check if model is fitted
                        raise ValueError("Model not trained")

                    # Create future dataframe
                    if X is not None and isinstance(X, pd.DataFrame) and "ds" in X.columns:
                        future = X.copy()
                    else:
                        future = self.model.make_future_dataframe(periods=periods)

                    # Generate forecast
                    forecast = self.model.predict(future)
                    return forecast["yhat"].values

                def save(self, path):
                    if not hasattr(self.model, "params"):
                        raise ValueError("Model not trained")
                    # Save model as JSON
                    from prophet.serialize import model_to_json

                    with open(f"{path}.json", "w") as f:
                        json.dump(model_to_json(self.model), f)
                    return path

                def load(self, path):
                    from prophet.serialize import model_from_json

                    with open(f"{path}.json") as f:
                        self.model = model_from_json(json.load(f))
                    return True

            return ProphetModel(**kwargs)
        except ImportError:
            logger.warning("Prophet not available, using stub Prophet model")
            return self._create_stub_model("Prophet", **kwargs)

    def _create_linear_regression_model(self, **kwargs) -> Any:
        """Create a Linear Regression model instance."""
        try:
            from sklearn.linear_model import LinearRegression

            class LinearRegressionModel:
                def __init__(self, **_kwargs):
                    self.model = LinearRegression(**_kwargs)
                    self.features: list[str] = []

                def fit(self, train_data, target_column, **_kwargs):
                    # Extract features and target
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features]
                    y = train_data[target_column]

                    # Fit model
                    self.model.fit(X, y)

                    # Return metrics if validation data provided
                    metrics: dict[str, float] = {}
                    if "validation_data" in _kwargs:
                        val_data = _kwargs["validation_data"]
                        val_pred = self.predict(val_data)
                        val_true = val_data[target_column]
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                        metrics["val_mse"] = mean_squared_error(val_true, val_pred)
                        metrics["val_mae"] = mean_absolute_error(val_true, val_pred)
                        metrics["val_r2"] = r2_score(val_true, val_pred)

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.model, "coef_"):
                        raise ValueError("Model not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions
                    return self.model.predict(X_features)

                def save(self, path):
                    if not hasattr(self.model, "coef_"):
                        raise ValueError("Model not trained")
                    import pickle

                    with open(f"{path}.pkl", "wb") as f:
                        pickle.dump(self.model, f)
                    # Save features
                    with open(f"{path}_features.json", "w") as f:
                        json.dump(self.features, f)
                    return path

                def load(self, path):
                    import pickle

                    with open(f"{path}.pkl", "rb") as f:
                        self.model = pickle.load(f)
                    # Load features
                    try:
                        with open(f"{path}_features.json") as f:
                            self.features = json.load(f)
                    except FileNotFoundError:
                        logger.warning(f"Features file not found for {path}, using empty features list")
                        self.features = []
                    return True

            return LinearRegressionModel(**kwargs)
        except ImportError:
            logger.warning("scikit-learn not available, using stub Linear Regression model")
            return self._create_stub_model("LinearRegression", **kwargs)

    def _create_xgboost_regressor(self, **kwargs) -> Any:
        """Create an XGBoost regressor model instance."""
        try:
            import xgboost as xgb

            class XGBoostRegressorModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.model = xgb.XGBRegressor(**_kwargs)
                    self.features: list[str] = []
                    self.is_gold_model = _kwargs.get("is_gold_model", False)
                    self.model_type = _kwargs.get("model_type", "xgboost_regressor")

                def fit(self, train_data, target_column, validation_data=None, **_kwargs):
                    # Extract features and target
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features]
                    y = train_data[target_column]

                    # Prepare validation data if provided
                    eval_set = None
                    if validation_data is not None:
                        X_val = validation_data[self.features]
                        y_val = validation_data[target_column]
                        eval_set = [(X, y), (X_val, y_val)]

                    # Fit model
                    self.model.fit(X, y, eval_set=eval_set, **_kwargs)

                    # Return metrics
                    metrics: dict[str, Any] = {}
                    if hasattr(self.model, "evals_result_"):
                        results = self.model.evals_result_
                        # Add validation metrics if available
                        if results and len(results) > 1:
                            metrics["val_rmse"] = results["validation_1"]["rmse"][-1]

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions
                    return self.model.predict(X_features)

                def save(self, path):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Save model
                    self.model.save_model(f"{path}.json")

                    # Save features and metadata
                    metadata = {
                        "features": self.features,
                        "is_gold_model": self.is_gold_model,
                        "model_type": self.model_type,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    # Load model
                    self.model.load_model(f"{path}.json")

                    # Load features and metadata
                    try:
                        with open(f"{path}_metadata.json") as f:
                            metadata = json.load(f)
                            self.features = metadata.get("features", [])
                            self.is_gold_model = metadata.get("is_gold_model", False)
                            self.model_type = metadata.get("model_type", "xgboost_regressor")
                    except FileNotFoundError:
                        logger.warning(f"Metadata file not found for {path}")

                    return True

                def get_feature_importance(self):
                    """Get feature importance scores."""
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Create dictionary of feature to importance
                    return dict(zip(self.features, self.model.feature_importances_, strict=False))

                def is_gold_price_model(self):
                    """Check if this is a gold price model."""
                    return self.is_gold_model

            return XGBoostRegressorModel(**kwargs)
        except ImportError:
            logger.warning("XGBoost not available, using stub XGBoost model")
            return self._create_stub_model("XGBoostRegressor", **kwargs)

    def _create_random_forest_model(self, **kwargs) -> Any:
        """Create a Random Forest model instance."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            class RandomForestModel:
                def __init__(self, **_kwargs):
                    self.model = RandomForestRegressor(**_kwargs)
                    self.features: list[str] = []

                def fit(self, train_data, target_column, **_kwargs):
                    # Extract features and target
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features]
                    y = train_data[target_column]

                    # Fit model
                    self.model.fit(X, y)

                    # Return metrics if validation data provided
                    metrics: dict[str, float] = {}
                    if "validation_data" in _kwargs:
                        val_data = _kwargs["validation_data"]
                        val_pred = self.predict(val_data)
                        val_true = val_data[target_column]
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                        metrics["val_mse"] = mean_squared_error(val_true, val_pred)
                        metrics["val_mae"] = mean_absolute_error(val_true, val_pred)
                        metrics["val_r2"] = r2_score(val_true, val_pred)

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions
                    return self.model.predict(X_features)

                def save(self, path):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")
                    import pickle

                    with open(f"{path}.pkl", "wb") as f:
                        pickle.dump(self.model, f)
                    # Save features
                    with open(f"{path}_features.json", "w") as f:
                        json.dump(self.features, f)
                    return path

                def load(self, path):
                    import pickle

                    with open(f"{path}.pkl", "rb") as f:
                        self.model = pickle.load(f)
                    # Load features
                    try:
                        with open(f"{path}_features.json") as f:
                            self.features = json.load(f)
                    except FileNotFoundError:
                        logger.warning(f"Features file not found for {path}, using empty features list")
                        self.features = []
                    return True

                def get_feature_importance(self):
                    """Get feature importance scores."""
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Create dictionary of feature to importance
                    return dict(zip(self.features, self.model.feature_importances_, strict=False))

            return RandomForestModel(**kwargs)
        except ImportError:
            logger.warning("scikit-learn not available, using stub Random Forest model")
            return self._create_stub_model("RandomForest", **kwargs)

    def _create_gold_direction_model(self, **kwargs) -> Any:
        """Create a gold price direction prediction model."""
        try:
            import xgboost as xgb

            class GoldDirectionModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.model = xgb.XGBClassifier(
                        objective="binary:logistic",
                        **{k: v for k, v in _kwargs.items() if k != "is_gold_model" and k != "model_type"},
                    )
                    self.features: list[str] = []
                    self.is_gold_model = True
                    self.model_type = "gold_direction"

                def fit(self, train_data, target_column, validation_data=None, **_kwargs):
                    # Extract features and target
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features]
                    y = train_data[target_column]

                    # Prepare validation data if provided
                    eval_set = None
                    X_val = None
                    y_val = None
                    if validation_data is not None:
                        X_val = validation_data[self.features]
                        y_val = validation_data[target_column]
                        eval_set = [(X, y), (X_val, y_val)]

                    # Fit model
                    self.model.fit(X, y, eval_set=eval_set, **_kwargs)

                    # Return metrics
                    metrics: dict[str, float] = {}
                    if validation_data is not None and X_val is not None and y_val is not None:
                        from sklearn.metrics import accuracy_score, roc_auc_score

                        y_pred = self.model.predict(X_val)
                        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
                        metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
                        metrics["roc_auc"] = float(roc_auc_score(y_val, y_pred_proba))

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions
                    if _kwargs.get("return_proba", False):
                        return self.model.predict_proba(X_features)[:, 1]
                    else:
                        return self.model.predict(X_features)

                def save(self, path):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Save model
                    self.model.save_model(f"{path}.json")

                    # Save features and metadata
                    metadata = {
                        "features": self.features,
                        "is_gold_model": self.is_gold_model,
                        "model_type": self.model_type,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    # Load model
                    self.model.load_model(f"{path}.json")

                    # Load features and metadata
                    try:
                        with open(f"{path}_metadata.json") as f:
                            metadata = json.load(f)
                            self.features = metadata.get("features", [])
                            self.is_gold_model = metadata.get("is_gold_model", True)
                            self.model_type = metadata.get("model_type", "gold_direction")
                    except FileNotFoundError:
                        logger.warning(f"Metadata file not found for {path}")

                    return True

                def get_feature_importance(self):
                    """Get feature importance scores."""
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Create dictionary of feature to importance
                    return dict(zip(self.features, self.model.feature_importances_, strict=False))

                def is_gold_price_model(self):
                    """Check if this is a gold price model."""
                    return True

                def validate_hyperparameters(self):
                    """Validate the model hyperparameters."""
                    # XGBoost-specific validation
                    if "max_depth" in self.params and self.params["max_depth"] > 10:
                        logger.warning("GoldDirectionModel: max_depth > 10 may cause overfitting")
                    if "learning_rate" in self.params and self.params["learning_rate"] > 0.3:
                        logger.warning("GoldDirectionModel: learning_rate > 0.3 may cause unstable training")
                    return True

            return GoldDirectionModel(**kwargs)
        except ImportError:
            logger.warning("XGBoost not available, using stub Gold Direction model")
            return self._create_stub_model("GoldDirection", is_gold_model=True, **kwargs)

    def _create_gold_regression_model(self, **kwargs) -> Any:
        """Create a gold price regression model."""
        try:
            import xgboost as xgb

            class GoldRegressionModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.model = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        **{k: v for k, v in _kwargs.items() if k != "is_gold_model" and k != "model_type"},
                    )
                    self.features: list[str] = []
                    self.is_gold_model = True
                    self.model_type = "gold_regression"

                def fit(self, train_data, target_column, validation_data=None, **_kwargs):
                    # Extract features and target
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features]
                    y = train_data[target_column]

                    # Prepare validation data if provided
                    eval_set = None
                    X_val = None
                    y_val = None
                    if validation_data is not None:
                        X_val = validation_data[self.features]
                        y_val = validation_data[target_column]
                        eval_set = [(X, y), (X_val, y_val)]

                    # Fit model
                    self.model.fit(X, y, eval_set=eval_set, **_kwargs)

                    # Return metrics
                    metrics: dict[str, float] = {}
                    if validation_data is not None and X_val is not None and y_val is not None:
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                        y_pred = self.model.predict(X_val)
                        metrics["mse"] = mean_squared_error(y_val, y_pred)
                        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                        metrics["mae"] = mean_absolute_error(y_val, y_pred)
                        metrics["r2"] = r2_score(y_val, y_pred)

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions
                    return self.model.predict(X_features)

                def save(self, path):
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Save model
                    self.model.save_model(f"{path}.json")

                    # Save features and metadata
                    metadata = {
                        "features": self.features,
                        "is_gold_model": self.is_gold_model,
                        "model_type": self.model_type,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    # Load model
                    self.model.load_model(f"{path}.json")

                    # Load features and metadata
                    try:
                        with open(f"{path}_metadata.json") as f:
                            metadata = json.load(f)
                            self.features = metadata.get("features", [])
                            self.is_gold_model = metadata.get("is_gold_model", True)
                            self.model_type = metadata.get("model_type", "gold_regression")
                    except FileNotFoundError:
                        logger.warning(f"Metadata file not found for {path}")

                    return True

                def get_feature_importance(self):
                    """Get feature importance scores."""
                    if not hasattr(self.model, "feature_importances_"):
                        raise ValueError("Model not trained")

                    # Create dictionary of feature to importance
                    return dict(zip(self.features, self.model.feature_importances_, strict=False))

                def is_gold_price_model(self):
                    """Check if this is a gold price model."""
                    return True

                def validate_hyperparameters(self):
                    """Validate the model hyperparameters."""
                    # XGBoost-specific validation
                    if "max_depth" in self.params and self.params["max_depth"] > 10:
                        logger.warning("GoldRegressionModel: max_depth > 10 may cause overfitting")
                    if "learning_rate" in self.params and self.params["learning_rate"] > 0.3:
                        logger.warning("GoldRegressionModel: learning_rate > 0.3 may cause unstable training")
                    return True

            return GoldRegressionModel(**kwargs)
        except ImportError:
            logger.warning("XGBoost not available, using stub Gold Regression model")
            return self._create_stub_model("GoldRegression", is_gold_model=True, **kwargs)

    def _create_gold_ensemble_model(self, **kwargs) -> Any:
        """Create a gold price ensemble model."""
        try:
            import xgboost as xgb

            class GoldEnsembleModel:
                def __init__(self, **_kwargs):
                    self.params = _kwargs
                    self.direction_model = self._create_direction_model(_kwargs)
                    self.regression_model = self._create_regression_model(_kwargs)
                    self.features: list[str] = []
                    self.is_gold_model = True
                    self.model_type = "gold_ensemble"

                def _create_direction_model(self, _kwargs):
                    """Create the direction sub-model."""
                    direction_params = _kwargs.get("direction_params", {})
                    return xgb.XGBClassifier(objective="binary:logistic", **direction_params)

                def _create_regression_model(self, _kwargs):
                    """Create the regression sub-model."""
                    regression_params = _kwargs.get("regression_params", {})
                    return xgb.XGBRegressor(objective="reg:squarederror", **regression_params)

                def fit(self, train_data, target_column, direction_column=None, **_kwargs):
                    # Extract common features
                    self.features = [
                        col
                        for col in train_data.columns
                        if col != target_column and (direction_column is None or col != direction_column)
                    ]

                    # Prepare direction target if provided
                    if direction_column is None:
                        # Create direction column (price goes up or down)
                        direction_column = f"{target_column}_direction"
                        train_data[direction_column] = (
                            train_data[target_column].shift(-1) > train_data[target_column]
                        ).astype(int)

                    # Extract features and targets
                    X = train_data[self.features]
                    y_reg = train_data[target_column]
                    y_dir = train_data[direction_column]

                    # Fit direction model
                    self.direction_model.fit(X, y_dir)

                    # Fit regression model
                    self.regression_model.fit(X, y_reg)

                    # Return metrics
                    metrics: dict[str, float] = {}
                    if "validation_data" in _kwargs:
                        val_data = _kwargs["validation_data"]
                        X_val = val_data[self.features]

                        # Direction metrics
                        from sklearn.metrics import accuracy_score, roc_auc_score

                        y_dir_val = val_data[direction_column]
                        y_dir_pred = self.direction_model.predict(X_val)
                        y_dir_prob = self.direction_model.predict_proba(X_val)[:, 1]
                        metrics["dir_accuracy"] = float(accuracy_score(y_dir_val, y_dir_pred))
                        metrics["dir_roc_auc"] = float(roc_auc_score(y_dir_val, y_dir_prob))

                        # Regression metrics
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                        y_reg_val = val_data[target_column]
                        y_reg_pred = self.regression_model.predict(X_val)
                        metrics["reg_mse"] = mean_squared_error(y_reg_val, y_reg_pred)
                        metrics["reg_rmse"] = float(np.sqrt(metrics["reg_mse"]))
                        metrics["reg_mae"] = mean_absolute_error(y_reg_val, y_reg_pred)
                        metrics["reg_r2"] = r2_score(y_reg_val, y_reg_pred)

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if not hasattr(self.direction_model, "feature_importances_"):
                        raise ValueError("Models not trained")

                    # Prepare features
                    X_features = X[self.features] if isinstance(X, pd.DataFrame) else X

                    # Generate predictions for both models
                    direction_pred = self.direction_model.predict(X_features)
                    direction_prob = self.direction_model.predict_proba(X_features)[:, 1]
                    price_pred = self.regression_model.predict(X_features)

                    # Return different results based on mode
                    mode = _kwargs.get("mode", "price")
                    if mode == "direction":
                        return direction_pred
                    elif mode == "direction_prob":
                        return direction_prob
                    elif mode == "both":
                        return np.column_stack((price_pred, direction_prob))
                    else:  # default: price
                        return price_pred

                def save(self, path):
                    if not hasattr(self.direction_model, "feature_importances_"):
                        raise ValueError("Models not trained")

                    # Save direction model
                    self.direction_model.save_model(f"{path}_direction.json")

                    # Save regression model
                    self.regression_model.save_model(f"{path}_regression.json")

                    # Save features and metadata
                    metadata = {
                        "features": self.features,
                        "is_gold_model": self.is_gold_model,
                        "model_type": self.model_type,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    # Load direction model
                    self.direction_model.load_model(f"{path}_direction.json")

                    # Load regression model
                    self.regression_model.load_model(f"{path}_regression.json")

                    # Load features and metadata
                    try:
                        with open(f"{path}_metadata.json") as f:
                            metadata = json.load(f)
                            self.features = metadata.get("features", [])
                            self.is_gold_model = metadata.get("is_gold_model", True)
                            self.model_type = metadata.get("model_type", "gold_ensemble")
                    except FileNotFoundError:
                        logger.warning(f"Metadata file not found for {path}")

                    return True

                def get_feature_importance(self):
                    """Get feature importance scores for both models."""
                    if not hasattr(self.direction_model, "feature_importances_"):
                        raise ValueError("Models not trained")

                    # Get importance from both models
                    dir_imp = dict(zip(self.features, self.direction_model.feature_importances_, strict=False))
                    reg_imp = dict(zip(self.features, self.regression_model.feature_importances_, strict=False))

                    # Combine importances
                    return {"direction": dir_imp, "regression": reg_imp}

                def is_gold_price_model(self):
                    """Check if this is a gold price model."""
                    return True

                def validate_hyperparameters(self):
                    """Validate the model hyperparameters."""
                    # Validate direction model parameters
                    dir_params = self.params.get("direction_params", {})
                    if "max_depth" in dir_params and dir_params["max_depth"] > 10:
                        logger.warning("GoldEnsembleModel (direction): max_depth > 10 may cause overfitting")

                    # Validate regression model parameters
                    reg_params = self.params.get("regression_params", {})
                    if "max_depth" in reg_params and reg_params["max_depth"] > 10:
                        logger.warning("GoldEnsembleModel (regression): max_depth > 10 may cause overfitting")

                    return True

            return GoldEnsembleModel(**kwargs)
        except ImportError:
            logger.warning("XGBoost not available, using stub Gold Ensemble model")
            return self._create_stub_model("GoldEnsemble", is_gold_model=True, **kwargs)

    def _create_lstm_model(self, **kwargs) -> Any:
        """Create an LSTM model instance."""
        try:
            import tensorflow as tf  # type: ignore[import-unresolved]

            class LSTMModel:
                def __init__(self, **_kwargs):
                    self.units = _kwargs.get("units", 50)
                    self.dropout = _kwargs.get("dropout", 0.2)
                    self.sequence_length = _kwargs.get("sequence_length", 10)
                    self.model: Any = None
                    self.features: list[str] = []
                    self.scaler: Any = None
                    self.scaler_X: Any = None
                    self.scaler_y: Any = None

                def _build_model(self, input_shape):
                    """Build the LSTM model architecture."""
                    model = tf.keras.Sequential(
                        [
                            tf.keras.layers.LSTM(
                                units=self.units, return_sequences=True, input_shape=input_shape, dropout=self.dropout
                            ),
                            tf.keras.layers.LSTM(units=self.units, dropout=self.dropout),
                            tf.keras.layers.Dense(units=1),
                        ]
                    )
                    model.compile(optimizer="adam", loss="mse")
                    return model

                def _prepare_sequences(self, data, seq_length):
                    """Convert time series data to sequences for LSTM."""
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i : i + seq_length])
                        y.append(data[i + seq_length])
                    return np.array(X), np.array(y)

                def fit(self, train_data, target_column, **_kwargs):
                    from sklearn.preprocessing import MinMaxScaler

                    # Extract features
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features].values
                    y = train_data[target_column].values.reshape(-1, 1)

                    # Scale data
                    self.scaler_X = MinMaxScaler()
                    self.scaler_y = MinMaxScaler()
                    X_scaled = self.scaler_X.fit_transform(X)
                    self.scaler_y.fit_transform(y)

                    # Create sequences
                    X_seq, y_seq = self._prepare_sequences(X_scaled, self.sequence_length)

                    # Build model
                    self.model = self._build_model((self.sequence_length, X.shape[1]))

                    # Train model
                    history = self.model.fit(
                        X_seq,
                        y_seq,
                        epochs=_kwargs.get("epochs", 50),
                        batch_size=_kwargs.get("batch_size", 32),
                        validation_split=_kwargs.get("validation_split", 0.2),
                        verbose=_kwargs.get("verbose", 0),
                    )

                    # Return metrics
                    metrics: dict[str, Any] = {
                        "loss": history.history["loss"][-1],
                        "val_loss": history.history["val_loss"][-1] if "val_loss" in history.history else None,
                    }

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if self.model is None:
                        raise ValueError("Model not trained")

                    # Prepare input data
                    X_values = X[self.features].values if isinstance(X, pd.DataFrame) else X

                    # Scale input
                    X_scaled = self.scaler_X.transform(X_values)

                    # Create sequences
                    seq_length = self.sequence_length
                    if len(X_scaled) < seq_length:
                        raise ValueError(f"Input data too short for sequence length {seq_length}")

                    X_seq = []
                    for i in range(len(X_scaled) - seq_length + 1):
                        X_seq.append(X_scaled[i : i + seq_length])
                    X_seq = np.array(X_seq)

                    # Generate predictions
                    y_pred_scaled = self.model.predict(X_seq)

                    # Inverse transform to get actual values
                    y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

                    return y_pred.flatten()

                def save(self, path):
                    if self.model is None:
                        raise ValueError("Model not trained")

                    # Save keras model
                    self.model.save(f"{path}.h5")

                    # Save scaler and metadata
                    import pickle

                    with open(f"{path}_scaler_X.pkl", "wb") as f:
                        pickle.dump(self.scaler_X, f)
                    with open(f"{path}_scaler_y.pkl", "wb") as f:
                        pickle.dump(self.scaler_y, f)

                    # Save features and parameters
                    metadata = {
                        "features": self.features,
                        "sequence_length": self.sequence_length,
                        "units": self.units,
                        "dropout": self.dropout,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    import pickle

                    import tensorflow as tf  # type: ignore[import-unresolved]

                    # Load keras model
                    self.model = tf.keras.models.load_model(f"{path}.h5")

                    # Load scaler
                    with open(f"{path}_scaler_X.pkl", "rb") as f:
                        self.scaler_X = pickle.load(f)
                    with open(f"{path}_scaler_y.pkl", "rb") as f:
                        self.scaler_y = pickle.load(f)

                    # Load features and parameters
                    with open(f"{path}_metadata.json") as f:
                        metadata = json.load(f)
                        self.features = metadata.get("features", [])
                        self.sequence_length = metadata.get("sequence_length", 10)
                        self.units = metadata.get("units", 50)
                        self.dropout = metadata.get("dropout", 0.2)

                    return True

            return LSTMModel(**kwargs)
        except ImportError:
            logger.warning("TensorFlow not available, using stub LSTM model")
            return self._create_stub_model("LSTM", **kwargs)

    def _create_neural_network(self, **kwargs) -> Any:
        """Create a neural network model instance."""
        try:
            import tensorflow as tf  # type: ignore[import-unresolved]

            class NeuralNetworkModel:
                def __init__(self, **_kwargs):
                    self.hidden_layers = _kwargs.get("hidden_layers", [64, 32])
                    self.activation = _kwargs.get("activation", "relu")
                    self.model: Any = None
                    self.features: list[str] = []
                    self.scaler: Any = None
                    self.scaler_X: Any = None
                    self.scaler_y: Any = None

                def _build_model(self, input_shape):
                    """Build the neural network architecture."""
                    model = tf.keras.Sequential()

                    # Input layer
                    model.add(tf.keras.layers.Input(shape=(input_shape,)))
                    model.add(tf.keras.layers.Dense(self.hidden_layers[0], activation=self.activation))

                    # Hidden layers
                    for units in self.hidden_layers[1:]:
                        model.add(tf.keras.layers.Dense(units, activation=self.activation))

                    # Output layer
                    model.add(tf.keras.layers.Dense(1))

                    model.compile(optimizer="adam", loss="mse")
                    return model

                def fit(self, train_data, target_column, **_kwargs):
                    from sklearn.preprocessing import StandardScaler

                    # Extract features
                    self.features = [col for col in train_data.columns if col != target_column]
                    X = train_data[self.features].values
                    y = train_data[target_column].values.reshape(-1, 1)

                    # Scale data
                    self.scaler_X = StandardScaler()
                    self.scaler_y = StandardScaler()
                    X_scaled = self.scaler_X.fit_transform(X)
                    y_scaled = self.scaler_y.fit_transform(y)

                    # Build model
                    self.model = self._build_model(X.shape[1])

                    # Train model
                    history = self.model.fit(
                        X_scaled,
                        y_scaled,
                        epochs=_kwargs.get("epochs", 50),
                        batch_size=_kwargs.get("batch_size", 32),
                        validation_split=_kwargs.get("validation_split", 0.2),
                        verbose=_kwargs.get("verbose", 0),
                    )

                    # Return metrics
                    metrics: dict[str, Any] = {
                        "loss": history.history["loss"][-1],
                        "val_loss": history.history["val_loss"][-1] if "val_loss" in history.history else None,
                    }

                    return {"status": "success", "metrics": metrics}

                def predict(self, X, **_kwargs):
                    if self.model is None:
                        raise ValueError("Model not trained")

                    # Prepare input data
                    X_values = X[self.features].values if isinstance(X, pd.DataFrame) else X

                    # Scale input
                    X_scaled = self.scaler_X.transform(X_values)

                    # Generate predictions
                    y_pred_scaled = self.model.predict(X_scaled)

                    # Inverse transform to get actual values
                    y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

                    return y_pred.flatten()

                def save(self, path):
                    if self.model is None:
                        raise ValueError("Model not trained")

                    # Save keras model
                    self.model.save(f"{path}.h5")

                    # Save scaler and metadata
                    import pickle

                    with open(f"{path}_scaler_X.pkl", "wb") as f:
                        pickle.dump(self.scaler_X, f)
                    with open(f"{path}_scaler_y.pkl", "wb") as f:
                        pickle.dump(self.scaler_y, f)

                    # Save features and parameters
                    metadata = {
                        "features": self.features,
                        "hidden_layers": self.hidden_layers,
                        "activation": self.activation,
                    }
                    with open(f"{path}_metadata.json", "w") as f:
                        json.dump(metadata, f)

                    return path

                def load(self, path):
                    import pickle

                    import tensorflow as tf  # type: ignore[import-unresolved]

                    # Load keras model
                    self.model = tf.keras.models.load_model(f"{path}.h5")

                    # Load scaler
                    with open(f"{path}_scaler_X.pkl", "rb") as f:
                        self.scaler_X = pickle.load(f)
                    with open(f"{path}_scaler_y.pkl", "rb") as f:
                        self.scaler_y = pickle.load(f)

                    # Load features and parameters
                    with open(f"{path}_metadata.json") as f:
                        metadata = json.load(f)
                        self.features = metadata.get("features", [])
                        self.hidden_layers = metadata.get("hidden_layers", [64, 32])
                        self.activation = metadata.get("activation", "relu")

                    return True

            return NeuralNetworkModel(**kwargs)
        except ImportError:
            logger.warning("TensorFlow not available, using stub Neural Network model")
            return self._create_stub_model("NeuralNetwork", **kwargs)

    def _create_stub_model(self, model_type: str, **kwargs) -> Any:
        """Create a stub model for when a dependency is missing."""

        class StubModel:
            """Stub model implementation for when dependencies are missing."""

            def __init__(self, _model_type, **_kwargs):
                self.model_type = _model_type
                self.params = _kwargs
                self.features: list[str] = []
                self.is_trained = False
                self.is_gold_model = _kwargs.get("is_gold_model", False)

            def fit(self, train_data, target_column, **_kwargs):
                """Stub fit method."""
                logger.warning(f"Using stub {self.model_type} model - no actual training performed")
                self.features = [col for col in train_data.columns if col != target_column]
                self.is_trained = True
                return {"status": "warning", "message": f"Stub {self.model_type} model used"}

            def predict(self, X, **_kwargs):
                """Stub predict method."""
                if not self.is_trained:
                    raise ValueError("Model not trained")

                # Return random predictions
                size = len(X) if isinstance(X, pd.DataFrame) else len(X) if hasattr(X, "__len__") else 1

                # For classification, return 0/1
                if self.model_type in ["GoldDirection"]:
                    return np.random.randint(0, 2, size=size)

                # For regression, return random values centered at 0
                return np.random.randn(size)

            def save(self, path):
                """Stub save method."""
                if not self.is_trained:
                    raise ValueError("Model not trained")

                # Save minimal metadata
                metadata = {
                    "model_type": self.model_type,
                    "features": self.features,
                    "is_stub": True,
                    "is_gold_model": self.is_gold_model,
                }
                with open(f"{path}_stub_metadata.json", "w") as f:
                    json.dump(metadata, f)

                return path

            def load(self, path):
                """Stub load method."""
                # Load minimal metadata
                try:
                    with open(f"{path}_stub_metadata.json") as f:
                        metadata = json.load(f)
                        self.features = metadata.get("features", [])
                except FileNotFoundError:
                    logger.warning(f"Stub metadata file not found for {path}")

                self.is_trained = True
                return True

            def is_gold_price_model(self):
                """Check if this is a gold price model."""
                return self.is_gold_model

            def get_feature_importance(self):
                """Stub feature importance."""
                if not self.is_trained:
                    raise ValueError("Model not trained")

                # Return random importance scores
                return {feature: float(np.random.random()) for feature in self.features}

        return StubModel(model_type, **kwargs)

    def _initialize_selected_models(self) -> None:
        """Initialize instances of all selected models."""
        logger.info(f"Initializing {len(self.selected_models)} selected models")

        for model_name in self.selected_models:
            try:
                self._initialize_model(model_name)
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
                logger.debug(traceback.format_exc())

    def _initialize_model(self, model_name: str) -> None:
        """
        Initialize a specific model instance.

        Args:
            model_name: Name of the model to initialize

        Raises:
            ValueError: If the model type is not registered
        """
        if not self.registry.is_registered(model_name):
            raise ValueError(f"Model type not registered: {model_name}")

        try:
            # Get factory and config
            factory = self.registry.get_factory(model_name)
            config = self.registry.get_config(model_name)

            if factory is None:
                raise ValueError(f"No factory function registered for model type: {model_name}")

            # Create model instance
            logger.debug(f"Creating model {model_name} with config: {config}")
            model = factory(**config)

            # Store model
            self._models[model_name] = model

            # Create version info
            version_info = ModelVersionInfo()
            # Add model type info if available
            if hasattr(model, "model_type"):
                version_info.model_type = model.model_type
            elif model_name.lower().startswith("gold"):
                version_info.model_type = f"gold_{model_name.lower()}"

            self._versions[model_name] = version_info

            logger.info(f"Successfully initialized model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            self._versions[model_name] = ModelVersionInfo(status=ModelStatus.FAILED, error=str(e))
            raise

    def _load_existing_models(self) -> None:
        """Attempt to load any existing models from the model directory."""
        if not self.model_dir or not self.model_dir.exists():
            return

        logger.info(f"Checking for existing models in {self.model_dir}")

        # Look for model directories
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name

                # Check if this model is in our registry
                if model_name in self._models:
                    logger.debug(f"Found existing model directory: {model_dir}")

                    # Look for latest model file
                    model_files = list(model_dir.glob("model*"))
                    if model_files:
                        try:
                            # Try to load the model
                            latest_model = sorted(model_files)[-1]
                            self.load_model(model_name, str(latest_model))
                            logger.info(f"Loaded existing model: {model_name}")
                        except Exception as e:
                            logger.warning(f"Could not load existing model {model_name}: {e}")

    def _load_plugins(self, plugin_dir: str) -> None:
        """
        Load model plugins from a directory.

        Args:
            plugin_dir: Directory containing plugins
        """
        if not os.path.exists(plugin_dir):
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return

        logger.info(f"Loading plugins from {plugin_dir}")

        # Look for Python modules
        plugin_path = Path(plugin_dir)
        for py_file in plugin_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                module_name = py_file.stem
                import importlib.util

                # Load module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for models
                    models_found = 0
                    for name, obj in inspect.getmembers(module):
                        # Find model classes
                        if inspect.isclass(obj) and hasattr(obj, "fit") and hasattr(obj, "predict"):
                            # Register model
                            self.registry.register(name, lambda cls=obj, **kwargs: cls(**kwargs))
                            models_found += 1

                    logger.info(f"Loaded {models_found} model(s) from plugin {module_name}")
            except Exception as e:
                logger.error(f"Error loading plugin {py_file.name}: {e}")

    def _run_lifecycle_hooks(self, hook_type: str, model_name: str, **context) -> None:
        """
        Run lifecycle hooks for a specific event.

        Args:
            hook_type: Type of hook to run
            model_name: Name of the model
            **context: Additional context for the hooks
        """
        if hook_type not in self._lifecycle_hooks:
            logger.warning(f"Unknown hook type: {hook_type}")
            return

        for hook in self._lifecycle_hooks[hook_type]:
            try:
                hook(model_name=model_name, model=self._models.get(model_name), **context)
            except Exception as e:
                logger.warning(f"Error in {hook_type} hook for {model_name}: {e}")

    def _verify_model_interface(self, model: Any, model_name: str) -> bool:
        """
        Verify that a model implements the required interface.

        Args:
            model: Model instance to verify
            model_name: Name of the model (for logging)

        Returns:
            True if the model implements the required interface, False otherwise
        """
        required_methods = ["fit", "predict"]

        missing_methods = []
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                missing_methods.append(method)

        if missing_methods:
            logger.warning(f"Model {model_name} is missing required methods: {', '.join(missing_methods)}")
            return False
        return True

    @contextmanager
    def _model_operation(self, model_name: str, operation: str):
        """
        Context manager for model operations with error handling.

        Args:
            model_name: Name of the model
            operation: Operation being performed

        Yields:
            The model instance

        Raises:
            ValueError: If the model is not initialized
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not initialized")

        model = self._models[model_name]
        version_info = self._versions.get(model_name, ModelVersionInfo())

        previous_active = self._active_model
        try:
            # Set as active model during operation
            self._active_model = model_name

            yield model

        except Exception as e:
            logger.error(f"Error during {operation} for model {model_name}: {e}")
            # Update version info on error
            version_info.error = str(e)
            version_info.status = ModelStatus.FAILED
            self._versions[model_name] = version_info
            raise

        finally:
            # Restore previous active model
            self._active_model = previous_active

    # Public API methods

    def get_models(self) -> dict[str, Any]:
        """
        Get all initialized model instances.

        Returns:
            Dictionary of model name to model instance
        """
        return self._models

    def set_models(self, models: dict[str, Any]) -> None:
        """
        Set multiple model instances at once.

        Args:
            models: Dictionary of model name to model instance
        """
        # Verify and add each model
        for name, model in models.items():
            if self._verify_model_interface(model, name):
                self._models[name] = model
                if name not in self._versions:
                    version_info = ModelVersionInfo()
                    # Add model type if available
                    if hasattr(model, "model_type"):
                        version_info.model_type = model.model_type
                    self._versions[name] = version_info
            else:
                logger.warning(f"Model {name} does not implement required interface")

        logger.info(f"Set {len(models)} models")

    def add_model(self, name: str, model: Any, metrics: dict[str, float] | None = None) -> None:
        """
        Add a model instance to the service.

        Args:
            name: Model name
            model: Model instance
            metrics: Optional performance metrics

        Raises:
            ValueError: If the model does not implement required interface
        """
        if not self._verify_model_interface(model, name):
            raise ValueError(f"Model {name} does not implement required interface")

        self._models[name] = model

        version_info = self._versions.get(name, ModelVersionInfo())
        if metrics:
            version_info.metrics = metrics

        # Set a trained status if metrics are provided
        if metrics and version_info.status == ModelStatus.INITIALIZED:
            version_info.status = ModelStatus.TRAINED

        # Store model type if available
        if hasattr(model, "model_type"):
            version_info.model_type = model.model_type
        elif name.lower().startswith("gold"):
            version_info.model_type = f"gold_{name.lower()}"

        self._versions[name] = version_info

        # Update selected models list if needed
        if name not in self.selected_models:
            self.selected_models.append(name)

        logger.info(f"Added model: {name}")

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the service.

        Args:
            name: Model name

        Returns:
            True if the model was removed, False otherwise
        """
        if name in self._models:
            del self._models[name]
            if name in self._versions:
                del self._versions[name]
            if name in self.selected_models:
                self.selected_models.remove(name)
            logger.info(f"Removed model: {name}")
            return True
        return False

    def get_model(self, name: str) -> Any | None:
        """
        Get a specific model instance.

        Args:
            name: Model name

        Returns:
            Model instance or None if not found
        """
        return self._models.get(name)

    def get_model_metrics(self) -> dict[str, dict[str, float]]:
        """
        Get performance metrics for all models.

        Returns:
            Dictionary of model name to metrics dictionary
        """
        return {name: info.metrics for name, info in self._versions.items() if info.metrics}

    def set_model_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        """
        Set performance metrics for multiple models.

        Args:
            metrics: Dictionary of model name to metrics dictionary
        """
        for name, model_metrics in metrics.items():
            if name in self._versions:
                self._versions[name].metrics = model_metrics
            else:
                version_info = ModelVersionInfo(metrics=model_metrics, status=ModelStatus.TRAINED)
                self._versions[name] = version_info

        logger.info(f"Set metrics for {len(metrics)} models")

    def clear_models(self) -> None:
        """Clear all models from the service."""
        self._models = {}
        self._versions = {}
        self._active_model = None
        logger.info("Cleared all models")

    def get_available_model_types(self) -> list[str]:
        """
        Get a list of all registered model types.

        Returns:
            List of model type names
        """
        return self.registry.list_models()

    def get_version_info(self, model_name: str) -> dict[str, Any] | None:
        """
        Get version information for a model.

        Args:
            model_name: Model name

        Returns:
            Dictionary with version information or None if not found
        """
        if model_name in self._versions:
            return self._versions[model_name].to_dict()
        return None

    def get_all_version_info(self) -> dict[str, dict[str, Any]]:
        """
        Get version information for all models.

        Returns:
            Dictionary of model name to version information
        """
        return {name: info.to_dict() for name, info in self._versions.items()}

    def register_hook(self, hook_type: str, hook_func: Callable) -> None:
        """
        Register a lifecycle hook function.

        Args:
            hook_type: Type of hook (pre_train, post_train, etc.)
            hook_func: Hook function

        Raises:
            ValueError: If hook_type is invalid
        """
        if hook_type not in self._lifecycle_hooks:
            raise ValueError(f"Invalid hook type: {hook_type}")

        self._lifecycle_hooks[hook_type].append(hook_func)
        logger.debug(f"Registered {hook_type} hook: {hook_func.__name__}")

    def train_model(
        self, model_name: str, data: pd.DataFrame, target_column: str, features: list[str] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Train a specific model with the provided data.

        Args:
            model_name: Name of the model to train
            data: Training data
            target_column: Target column name
            features: Feature column names (uses all except target if None)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results

        Raises:
            ValueError: If the model is not initialized
        """
        logger.info(f"Training model {model_name} with {len(data)} samples")

        with self._model_operation(model_name, "train") as model:
            # Update version info
            version_info = self._versions[model_name]
            version_info.status = ModelStatus.TRAINING
            version_info.updated_at = datetime.datetime.now().isoformat()

            # Determine features if not specified
            if features is None:
                features = [col for col in data.columns if col != target_column]

            # Run pre-train hooks
            self._run_lifecycle_hooks(
                "pre_train", model_name, data=data, target_column=target_column, features=features
            )

            try:
                # Try calling fit method first (standard interface)
                train_result = model.fit(train_data=data, target_column=target_column, **kwargs)
            except (TypeError, AttributeError) as e:
                logger.debug(f"Standard fit method failed, trying alternative: {e}")

                # Try alternative interfaces
                if hasattr(model, "train"):
                    # Try train(X, y) interface
                    try:
                        train_result = model.train(X=data[features], y=data[target_column], **kwargs)
                    except TypeError:
                        # Try train(X_train, y_train) interface
                        train_result = model.train(X_train=data[features], y_train=data[target_column], **kwargs)
                else:
                    raise ValueError(f"Model {model_name} has no valid training method") from None

            # Update version info with training details
            version_info.status = ModelStatus.TRAINED
            version_info.updated_at = datetime.datetime.now().isoformat()
            version_info.samples_count = len(data)
            version_info.features = features
            version_info.target = target_column
            version_info.training_params = kwargs

            # Add metrics if returned
            if isinstance(train_result, dict) and "metrics" in train_result:
                version_info.metrics.update(train_result["metrics"])

            # Run post-train hooks
            self._run_lifecycle_hooks("post_train", model_name, result=train_result)

            return train_result or {}

    def predict(self, model_name: str, data: pd.DataFrame, features: list[str] | None = None, **kwargs) -> np.ndarray:
        """
        Generate predictions using a model.

        Args:
            model_name: Name of the model to use
            data: Input data
            features: Feature columns (uses training features if None)
            **kwargs: Additional prediction parameters

        Returns:
            Numpy array of predictions

        Raises:
            ValueError: If the model is not initialized or trained
        """
        logger.info(f"Generating predictions with model {model_name} for {len(data)} samples")

        with self._model_operation(model_name, "predict") as model:
            # Check if model is trained
            version_info = self._versions[model_name]
            if version_info.status not in (ModelStatus.TRAINED, ModelStatus.LOADED):
                raise ValueError(f"Model {model_name} is not trained")

            # Use training features if not specified
            if features is None:
                features = version_info.features
                if not features:
                    raise ValueError("No features specified and no training features found")

            # Run pre-predict hooks
            self._run_lifecycle_hooks("pre_predict", model_name, data=data, features=features)

            try:
                # Try standard predict interface
                predictions = model.predict(X=data[features], **kwargs)
            except (TypeError, KeyError, AttributeError) as e:
                logger.debug(f"Standard predict interface failed, trying alternatives: {e}")

                # Try alternative interfaces
                try:
                    # Try predict(data) interface
                    predictions = model.predict(data=data, **kwargs)
                except (TypeError, AttributeError):
                    # Try predict(X_test) interface
                    predictions = model.predict(X_test=data[features], **kwargs)

            # Run post-predict hooks
            self._run_lifecycle_hooks("post_predict", model_name, predictions=predictions)

            return predictions

    def save_model(self, model_name: str, path: str | None = None) -> str:
        """
        Save a model to disk.

        Args:
            model_name: Name of the model to save
            path: Directory path (uses model_dir if None)

        Returns:
            Path to the saved model

        Raises:
            ValueError: If the model is not initialized
        """
        # Use model_dir as default path
        if path is None:
            if self.model_dir is None:
                raise ValueError("No path specified and no model_dir configured")
            path = str(self.model_dir / model_name)

        logger.info(f"Saving model {model_name} to {path}")

        with self._model_operation(model_name, "save") as model:
            # Create version-specific path
            version_info = self._versions[model_name]
            version_id = version_info.version_id
            model_path = f"{path}/{model_name}_{version_id}"

            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Run pre-save hooks
            self._run_lifecycle_hooks("pre_save", model_name, path=model_path)

            # Try different save methods
            save_path: str | None = None
            if hasattr(model, "save") and callable(model.save):
                result = model.save(model_path)
                save_path = str(result) if result is not None else None
            elif hasattr(model, "save_model") and callable(model.save_model):
                result = model.save_model(model_path)
                save_path = str(result) if result is not None else None
            else:
                # For models without a save method, try pickle
                import pickle

                try:
                    with open(f"{model_path}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    save_path = f"{model_path}.pkl"
                except Exception as e:
                    raise ValueError(f"Failed to save model {model_name} with pickle: {e}") from e

            # Save version info
            version_path = f"{model_path}_info.json"
            with open(version_path, "w") as f:
                json.dump(version_info.to_dict(), f, indent=2)

            # Run post-save hooks
            self._run_lifecycle_hooks("post_save", model_name, save_path=save_path)

            return save_path if save_path is not None else model_path

    def load_model(self, model_name: str, path: str) -> None:
        """
        Load a model from disk.

        Args:
            model_name: Name of the model to load
            path: Path to the saved model

        Raises:
            ValueError: If the model is not initialized or the path doesn't exist
        """
        logger.info(f"Loading model {model_name} from {path}")

        if not os.path.exists(path) and not (
            os.path.exists(f"{path}.pkl") or os.path.exists(f"{path}.json") or os.path.exists(f"{path}.h5")
        ):
            raise ValueError(f"Model path does not exist: {path}")

        with self._model_operation(model_name, "load") as model:
            # Run pre-load hooks
            self._run_lifecycle_hooks("pre_load", model_name, path=path)

            # Try different load methods
            if hasattr(model, "load") and callable(model.load):
                model.load(path)
            elif hasattr(model, "load_model") and callable(model.load_model):
                model.load_model(path)
            else:
                # For models without a load method, try pickle
                pickle_path = f"{path}.pkl" if not path.endswith(".pkl") else path
                if os.path.exists(pickle_path):
                    import pickle

                    with open(pickle_path, "rb") as f:
                        loaded_model = pickle.load(f)
                        # Copy attributes from loaded model
                        for attr_name, attr_value in vars(loaded_model).items():
                            setattr(model, attr_name, attr_value)
                else:
                    raise ValueError(f"Model {model_name} has no load method and no pickle file found")

            # Update version info
            version_info = self._versions[model_name]
            version_info.status = ModelStatus.LOADED
            version_info.updated_at = datetime.datetime.now().isoformat()
            version_info.loaded_from = path

            # Try to load version info
            version_path = f"{path}_info.json"
            if os.path.exists(version_path):
                try:
                    with open(version_path) as f:
                        loaded_info = json.load(f)

                    # Update features, target, etc. from loaded info
                    if "features" in loaded_info:
                        version_info.features = loaded_info["features"]
                    if "target" in loaded_info:
                        version_info.target = loaded_info["target"]
                    if "metrics" in loaded_info:
                        version_info.metrics = loaded_info["metrics"]
                    if "model_type" in loaded_info:
                        version_info.model_type = loaded_info["model_type"]

                    logger.debug(f"Loaded version info for {model_name}")
                except Exception as e:
                    logger.warning(f"Error loading version info: {e}")

            # Run post-load hooks
            self._run_lifecycle_hooks("post_load", model_name, path=path)

    def evaluate_model(
        self,
        model_name: str,
        data: pd.DataFrame,
        target_column: str,
        features: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a model against test data.

        Args:
            model_name: Name of the model to evaluate
            data: Test data
            target_column: Target column name
            features: Feature columns (uses training features if None)
            metrics: List of metrics to calculate

        Returns:
            Dictionary of metric name to value

        Raises:
            ValueError: If the model is not trained
        """
        logger.info(f"Evaluating model {model_name} with {len(data)} samples")

        metrics_list: list[str] = metrics if metrics else ["mae", "mse", "rmse", "r2"]

        # Check if model exists and is trained
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not initialized")

        version_info = self._versions.get(model_name)
        if not version_info or version_info.status not in (ModelStatus.TRAINED, ModelStatus.LOADED):
            raise ValueError(f"Model {model_name} is not trained")

        # Check if this is a classification model
        is_classification = False
        if version_info.model_type:
            is_classification = (
                "classifier" in version_info.model_type.lower() or "direction" in version_info.model_type.lower()
            )

        # Adjust metrics for classification models
        if is_classification:
            metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        # Get features list
        if features is None:
            features = version_info.features
            if not features:
                raise ValueError("No features specified and no training features found")

        # Generate predictions
        y_pred = self.predict(model_name, data, features)
        y_true = data[target_column].values

        # Calculate metrics
        results: dict[str, float] = {}

        # Classification metrics
        if is_classification:
            try:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

                # For binary classification
                if "accuracy" in metrics_list:
                    results["accuracy"] = float(accuracy_score(y_true, y_pred))

                # Try to get probability predictions for ROC AUC
                if "roc_auc" in metrics_list:
                    try:
                        y_pred_proba = self.predict(model_name, data, features, return_proba=True)
                        results["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {e}")

                # Other classification metrics
                if "precision" in metrics_list:
                    results["precision"] = float(precision_score(y_true, y_pred, average="weighted"))
                if "recall" in metrics_list:
                    results["recall"] = float(recall_score(y_true, y_pred, average="weighted"))
                if "f1" in metrics_list:
                    results["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
            except Exception as e:
                logger.warning(f"Error calculating classification metrics: {e}")
        else:
            # Regression metrics
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                if "mae" in metrics_list:
                    results["mae"] = float(mean_absolute_error(y_true, y_pred))

                if "mse" in metrics_list:
                    results["mse"] = float(mean_squared_error(y_true, y_pred))

                if "rmse" in metrics_list:
                    results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

                if "r2" in metrics_list:
                    results["r2"] = float(r2_score(y_true, y_pred))
            except Exception as e:
                logger.warning(f"Error calculating regression metrics: {e}")

        # Update model metrics
        version_info.metrics.update(results)

        return results

    def get_gold_price_models(self) -> dict[str, Any]:
        """
        Get all gold price models.

        Returns:
            Dictionary of model name to model instance, containing only gold price models
        """
        gold_models = {}

        for name, model in self._models.items():
            # Check if model is a gold price model
            is_gold = False

            # Check model attributes
            if hasattr(model, "is_gold_price_model") and callable(model.is_gold_price_model):
                is_gold = model.is_gold_price_model()
            elif hasattr(model, "is_gold_model"):
                is_gold = model.is_gold_model
            elif name.lower().startswith("gold"):
                is_gold = True

            # Check version info
            version_info = self._versions.get(name)
            if version_info and version_info.model_type and "gold" in version_info.model_type.lower():
                is_gold = True

            if is_gold:
                gold_models[name] = model

        return gold_models

    def __repr__(self) -> str:
        """String representation of ModelService."""
        return f"ModelService(models={list(self._models.keys())})"
