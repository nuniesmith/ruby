import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from lib.model._shims import logger

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import GridSearchCV, train_test_split

    HAS_SKLEARN = True
except ImportError:
    train_test_split = None
    GridSearchCV = None
    RandomForestClassifier = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    HAS_SKLEARN = False


class AssetPricePredictor:
    """Base class for asset price prediction models."""

    # Supported asset types
    SUPPORTED_ASSETS = ["gold", "silver", "bitcoin", "ethereum", "forex"]

    def __init__(self, asset_type: str, model_path: str | None = None, data_manager=None):
        """
        Initialize the predictor.

        Args:
            asset_type (str): Type of asset to predict
            model_path (str, optional): Path to load a saved model
            data_manager: Data manager instance for fetching asset data
        """
        if asset_type not in self.SUPPORTED_ASSETS:
            raise ValueError(f"Unsupported asset type: {asset_type}. Supported types: {self.SUPPORTED_ASSETS}")

        self.asset_type = asset_type
        self.model = None
        self.model_path = model_path
        self.data_manager = data_manager
        self.last_training_date: datetime | None = None
        self.model_metrics: dict[str, Any] = {}

        # Core technical indicator features
        self.feature_columns = [
            "DayOfWeek",
            "Hour",
            "Month",
            "MarketHour",
            "SMA5_Distance",
            "SMA20_Distance",
            "RSI",
            "MACD",
            "MACD_Hist",
            "Volatility",
            "Return",
            "BB_Width",
        ]

        # Additional optional features
        self.volume_features = ["Volume_Change", "Volume_Ratio"]
        self.advanced_features = ["ATR", "ADX", "OBV_Change", "StochK", "StochD"]

        # Feature importance scores (will be populated after training)
        self.feature_importance: dict[str, float] = {}

        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model file

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.feature_importance = model_data.get("feature_importance", {})
            self.last_training_date = model_data.get("training_date")
            self.model_metrics = model_data.get("metrics", {})

            logger.info(f"Loaded {self.asset_type} model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False

    def train_model(
        self,
        train_data: pd.DataFrame | None = None,
        test_size: float = 0.2,
        save_path: str | None = None,
        hyperparameter_tuning: bool = False,
    ) -> dict[str, Any]:
        """
        Train the prediction model with optional hyperparameter tuning.

        Args:
            train_data (DataFrame, optional): Training data with features
            test_size (float): Proportion of data to use for testing
            save_path (str, optional): Path to save the trained model
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning

        Returns:
            dict: Trained model and evaluation metrics
        """
        # Record training start time
        training_start = datetime.now()

        # Download data if not provided
        if train_data is None:
            if self.data_manager is None:
                raise ValueError("No data manager available to download data")

            raw_data = self.data_manager.download_data(
                self.asset_type, start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            )
            train_data = self.data_manager.prepare_features(raw_data, include_advanced=True)

        if train_data is None or train_data.empty:
            logger.error("No training data available")
            return {"error": "No training data available"}

        # Ensure all features exist
        features_to_use = self._prepare_feature_list(train_data)

        # Prepare features and target
        X = train_data[features_to_use]
        y = train_data["Target"]

        # Check if we have enough data
        if len(X) < 100:
            logger.warning(f"Limited data available ({len(X)} samples). Model accuracy may be affected.")

        # Split data
        assert train_test_split is not None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Train the model
        if hyperparameter_tuning:
            self.model, best_params = self._train_with_hyperparameter_tuning(X_train, y_train)
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            assert RandomForestClassifier is not None
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
            )
            assert self.model is not None
            self.model.fit(X_train, y_train)

        # Evaluate model
        assert self.model is not None
        y_pred = self.model.predict(X_test)
        assert accuracy_score is not None and classification_report is not None and confusion_matrix is not None
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate feature importance
        self.feature_importance = dict(zip(features_to_use, self.model.feature_importances_, strict=False))

        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Log results
        logger.info(f"{self.asset_type.capitalize()} model accuracy: {accuracy:.4f}")
        logger.info(
            f"{self.asset_type.capitalize()} classification report:\n{classification_report(y_test, y_pred)}"
        )  # classification_report asserted non-None above
        logger.info("Top 5 most important features:")
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")

        # Record metrics
        self.model_metrics = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "training_samples": len(X_train),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True), strict=False)),
        }

        # Record training date
        training_end = datetime.now()
        self.last_training_date = training_end

        # Calculate training duration
        training_duration = (training_end - training_start).total_seconds()

        # Save the model if requested
        model_data = {
            "model": self.model,
            "feature_importance": self.feature_importance,
            "training_date": self.last_training_date,
            "metrics": self.model_metrics,
            "features_used": features_to_use,
        }

        if save_path:
            save_path = save_path if save_path.endswith(".pkl") else f"{save_path}.pkl"
            joblib.dump(model_data, save_path)
            logger.info(f"Model saved to {save_path}")
            self.model_path = save_path

        return {
            "model": self.model,
            "accuracy": accuracy,
            "report": report,
            "feature_importance": self.feature_importance,
            "training_duration_seconds": training_duration,
            "features_used": features_to_use,
        }

    def _train_with_hyperparameter_tuning(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "tuple[Any, dict[str, Any]]":
        """
        Train model with hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Tuple containing the best model and the best parameters
        """
        logger.info("Performing hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # Create GridSearchCV object
        assert GridSearchCV is not None and RandomForestClassifier is not None
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_grid=param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        # Train with grid search
        grid_search.fit(X_train, y_train)

        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        return best_model, best_params

    def _prepare_feature_list(self, data: pd.DataFrame) -> list[str]:
        """
        Prepare the list of features to use based on what's available in the data.

        Args:
            data: DataFrame containing potential features

        Returns:
            List of feature column names to use
        """
        # Start with core features
        features_to_use = []

        # Add core features that exist in the data
        for feature in self.feature_columns:
            if feature in data.columns:
                features_to_use.append(feature)
            else:
                logger.warning(f"Core feature {feature} not found in training data. Adding zeros.")
                data[feature] = 0
                features_to_use.append(feature)

        # Add volume features if they exist
        for feature in self.volume_features:
            if feature in data.columns:
                features_to_use.append(feature)

        # Add advanced features if they exist
        for feature in self.advanced_features:
            if feature in data.columns:
                features_to_use.append(feature)

        return features_to_use

    def predict_next_movement(
        self,
        current_data: pd.DataFrame | None = None,
        prediction_horizon: str = "market_open",
        confidence_threshold: float = 0.6,
    ) -> dict[str, Any]:
        """
        Predict the next significant price movement.

        Args:
            current_data (DataFrame, optional): Current market data
            prediction_horizon (str): 'market_open', '24h', or '1h'
            confidence_threshold (float): Minimum confidence to make a prediction

        Returns:
            dict: Prediction results
        """
        if self.model is None:
            logger.error("Model not trained. Please train the model first or load a pre-trained model.")
            return {"error": "Model not trained"}

        # Download recent data if not provided
        if current_data is None:
            current_data = self._get_current_data()

        # Check if we have valid data
        if current_data is None or current_data.empty:
            logger.warning("No valid data available for prediction. Using placeholder data.")
            current_data = self._create_placeholder_data()

        # Get the latest data point
        latest_data = current_data.iloc[-1:].copy()

        # Extract the current date and time
        current_date = latest_data.index[0] if isinstance(latest_data.index, pd.DatetimeIndex) else datetime.now()

        # Make predictions based on the requested horizon
        if prediction_horizon == "market_open":
            # Generate prediction data for market openings
            prediction_data = self._prepare_market_opening_data(current_date)
            return self._make_predictions(prediction_data, latest_data, confidence_threshold)
        elif prediction_horizon == "24h":
            # Prepare data for 24-hour prediction
            prediction_data = self._prepare_time_horizon_data(current_date, hours=24)
            return self._make_time_horizon_prediction(prediction_data, latest_data, confidence_threshold)
        elif prediction_horizon == "1h":
            # Prepare data for 1-hour prediction
            prediction_data = self._prepare_time_horizon_data(current_date, hours=1)
            return self._make_time_horizon_prediction(prediction_data, latest_data, confidence_threshold)
        else:
            return {"error": f"Invalid prediction horizon: {prediction_horizon}"}

    def _get_current_data(self) -> pd.DataFrame | None:
        """Download current market data for prediction."""
        if self.data_manager is None:
            logger.error("No data manager available")
            return None

        # Get appropriate date range
        today = datetime.now().date()

        # If it's a weekend, use data from Friday
        if today.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            days_back = today.weekday() - 4  # Go back to Friday
            start_date = (today - timedelta(days=days_back + 10)).strftime("%Y-%m-%d")
            end_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        else:
            start_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

        # Download the latest data
        try:
            raw_data = self.data_manager.download_data(
                self.asset_type, start_date=start_date, end_date=end_date, interval="1h"
            )
            return self.data_manager.prepare_features(raw_data, include_advanced=True)
        except Exception as e:
            logger.error(f"Error downloading current data: {e}")
            return None

    def _create_placeholder_data(self) -> pd.DataFrame:
        """Create placeholder data when no real data is available."""
        now = datetime.now()
        placeholder_data = pd.DataFrame(
            {
                "DayOfWeek": [now.weekday()],
                "Hour": [now.hour],
                "Month": [now.month],
                "MarketHour": [0],
                "SMA5_Distance": [0],
                "SMA20_Distance": [0],
                "RSI": [50],
                "MACD": [0],
                "MACD_Hist": [0],
                "Volatility": [5],
                "Return": [0],
                "BB_Width": [5],
            },
            index=[now],
        )

        # Add volume features
        for feature in self.volume_features:
            placeholder_data[feature] = 0

        # Add advanced features
        for feature in self.advanced_features:
            placeholder_data[feature] = 0

        return placeholder_data

    def _prepare_market_opening_data(self, current_date: datetime) -> dict[str, datetime]:
        """Prepare prediction data for market openings."""
        if self.data_manager is None:
            logger.error("No data manager available")
            # Use default market opening times
            london_open = "08:00:00"
            newyork_open = "14:30:00"
            tokyo_open = "00:00:00"
        else:
            london_open = self.data_manager.london_open
            newyork_open = self.data_manager.newyork_open
            tokyo_open = "00:00:00"  # Default for Tokyo opening

        current_time = current_date.time()

        # Create prediction data for London opening
        london_open_time = datetime.strptime(london_open, "%H:%M:%S").time()
        london_date = datetime.combine(current_date.date(), london_open_time)

        # If London opening has already passed today, predict for tomorrow
        if current_time > london_open_time:
            london_date = london_date + timedelta(days=1)

            # Skip to Monday if the next day is a weekend
            while london_date.weekday() >= 5:
                london_date = london_date + timedelta(days=1)

        # Create prediction data for New York opening
        newyork_open_time = datetime.strptime(newyork_open, "%H:%M:%S").time()
        newyork_date = datetime.combine(current_date.date(), newyork_open_time)

        # If New York opening has already passed today, predict for tomorrow
        if current_time > newyork_open_time:
            newyork_date = newyork_date + timedelta(days=1)

            # Skip to Monday if the next day is a weekend
            while newyork_date.weekday() >= 5:
                newyork_date = newyork_date + timedelta(days=1)

        # Add Tokyo opening
        tokyo_open_time = datetime.strptime(tokyo_open, "%H:%M:%S").time()
        tokyo_date = datetime.combine(current_date.date(), tokyo_open_time)

        # If Tokyo opening has already passed today, predict for tomorrow
        if current_time > tokyo_open_time:
            tokyo_date = tokyo_date + timedelta(days=1)

            # Skip to Monday if the next day is a weekend
            while tokyo_date.weekday() >= 5:
                tokyo_date = tokyo_date + timedelta(days=1)

        return {"London": london_date, "NewYork": newyork_date, "Tokyo": tokyo_date}

    def _prepare_time_horizon_data(self, current_date: datetime, hours: int) -> dict[str, datetime]:
        """Prepare prediction data for specific time horizons."""
        # Calculate target prediction time
        target_time = current_date + timedelta(hours=hours)

        # Skip weekends for traditional assets
        if self.asset_type in ["gold", "silver", "forex"]:
            while target_time.weekday() >= 5:
                target_time = target_time + timedelta(days=1)

        return {f"{hours}h_ahead": target_time}

    def _make_predictions(
        self, prediction_dates: dict[str, datetime], latest_data: pd.DataFrame, confidence_threshold: float = 0.6
    ) -> dict[str, Any]:
        """Make predictions for the specified market opening dates."""
        # Prepare prediction records for each market
        prediction_records = []

        for market, market_date in prediction_dates.items():
            record = latest_data.copy()

            # Update date-related features
            if isinstance(record.index, pd.DatetimeIndex):
                record.index = [market_date]

            record["DayOfWeek"] = market_date.weekday()
            record["Hour"] = market_date.hour
            record["Month"] = market_date.month

            # Set MarketHour feature
            if market == "London":
                record["MarketHour"] = 1
            elif market == "NewYork":
                record["MarketHour"] = 2
            elif market == "Tokyo":
                record["MarketHour"] = 3
            else:
                record["MarketHour"] = 0

            prediction_records.append((market, record))

        # Define features to use
        features_to_use = self._prepare_feature_list(latest_data)

        # Make predictions
        results: dict[str, Any] = {}
        movement_map = {
            -1: "Significant Downward Movement (> 0.5% drop)",
            0: "Sideways Movement (within ±0.5%)",
            1: "Significant Upward Movement (> 0.5% rise)",
        }

        for market, record in prediction_records:
            try:
                X_pred = record[features_to_use]
                assert self.model is not None
                prediction = self.model.predict(X_pred)[0]
                probabilities = self.model.predict_proba(X_pred)[0]
                confidence = max(probabilities) * 100

                market_date = (
                    record.index[0] if isinstance(record.index, pd.DatetimeIndex) else prediction_dates[market]
                )

                # Only make high-confidence predictions
                prediction_text = movement_map[prediction]
                if confidence < confidence_threshold * 100:
                    prediction_text = "Uncertain - confidence below threshold"

                results[f"{market} Opening"] = {
                    "datetime": market_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "day": market_date.strftime("%A"),
                    "prediction": prediction_text,
                    "confidence": round(float(confidence), 2),
                    "raw_prediction": int(prediction),
                    "probabilities": {
                        movement_map[i]: round(float(prob) * 100, 2) for i, prob in enumerate(probabilities)
                    },
                }
            except Exception as e:
                logger.error(f"Error making prediction for {market}: {e}")
                results[f"{market} Opening"] = {"error": f"Failed to make prediction: {str(e)}"}

        return results

    def _make_time_horizon_prediction(
        self, prediction_dates: dict[str, datetime], latest_data: pd.DataFrame, confidence_threshold: float = 0.6
    ) -> dict[str, Any]:
        """Make predictions for specific time horizons."""
        # This is a wrapper around _make_predictions that formats the output differently
        raw_predictions = self._make_predictions(prediction_dates, latest_data, confidence_threshold)

        # Reformat the results
        results: dict[str, Any] = {"predictions": []}

        for time_frame, prediction_data in raw_predictions.items():
            if "error" in prediction_data:
                results["error"] = prediction_data["error"]
                continue

            results["predictions"].append(
                {
                    "time_frame": time_frame,
                    "target_datetime": prediction_data["datetime"],
                    "prediction": prediction_data["prediction"],
                    "confidence": prediction_data["confidence"],
                    "probability_breakdown": prediction_data["probabilities"],
                }
            )

        return results

    def feature_analysis(self) -> dict[str, Any]:
        """
        Analyze feature importance.

        Returns:
            dict: Feature importance analysis
        """
        if not self.feature_importance:
            return {"error": "No feature importance data available. Train model first."}

        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Group features by type
        feature_groups: dict[str, list[tuple[str, float]]] = {
            "time_based": [],
            "technical_indicators": [],
            "volume_related": [],
            "advanced_indicators": [],
        }

        time_features = ["DayOfWeek", "Hour", "Month", "MarketHour"]
        volume_features = self.volume_features
        advanced_features = self.advanced_features

        for feature, importance in sorted_features:
            if feature in time_features:
                feature_groups["time_based"].append((feature, importance))
            elif feature in volume_features:
                feature_groups["volume_related"].append((feature, importance))
            elif feature in advanced_features:
                feature_groups["advanced_indicators"].append((feature, importance))
            else:
                feature_groups["technical_indicators"].append((feature, importance))

        # Calculate relative importance of each group
        group_importance = {}
        for group, features in feature_groups.items():
            if features:
                total_importance = sum(imp for _, imp in features)
                group_importance[group] = total_importance

        # Normalize group importance
        total_importance = sum(group_importance.values())
        for group in group_importance:
            group_importance[group] = round((group_importance[group] / total_importance) * 100, 2)

        return {
            "top_features": [(f, round(i, 4)) for f, i in sorted_features[:10]],
            "group_importance": group_importance,
            "feature_groups": {k: [(f, round(i, 4)) for f, i in v] for k, v in feature_groups.items()},
        }

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"error": "No model loaded"}

        model_info = {
            "asset_type": self.asset_type,
            "model_type": type(self.model).__name__,
            "last_training_date": self.last_training_date.strftime("%Y-%m-%d %H:%M:%S")
            if self.last_training_date
            else "Unknown",
            "model_path": self.model_path or "Not saved",
            "model_params": self.model.get_params(),
            "features_used": len(self.feature_importance) if self.feature_importance else 0,
        }

        # Add metrics if available
        if self.model_metrics:
            model_info["metrics"] = {
                "accuracy": round(self.model_metrics.get("accuracy", 0), 4),
                "f1_scores": {
                    k: round(v["f1-score"], 4)
                    for k, v in self.model_metrics.get("report", {}).items()
                    if k in ["-1", "0", "1"] and isinstance(v, dict) and "f1-score" in v
                },
                "training_samples": self.model_metrics.get("training_samples", 0),
                "class_distribution": self.model_metrics.get("class_distribution", {}),
            }

        return model_info
