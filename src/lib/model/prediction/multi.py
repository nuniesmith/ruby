import os
from datetime import datetime, timedelta
from typing import Any

from lib.model._shims import logger

# Stub imports for modules not yet available in this project
AssetDataManager = None  # stub: from lib.model.data.manager import AssetDataManager
AssetPricePredictor = None  # stub: from lib.model.prediction.single import AssetPricePredictor


class MultiAssetPredictor:
    """Combined predictor for multiple asset types."""

    def __init__(self, assets: list[str] | None = None):
        """
        Initialize the multi-asset predictor.

        Args:
            assets (list, optional): List of asset types to predict
        """
        self.assets = assets or ["gold", "bitcoin", "ethereum"]
        self.predictors = {}

        # Create a shared data manager instance
        self.data_manager = AssetDataManager()  # type: ignore[misc]

        for asset in self.assets:
            predictor = AssetPricePredictor(asset_type=asset)  # type: ignore[misc]
            # Assign the shared data manager to each predictor
            predictor.data_manager = self.data_manager
            self.predictors[asset] = predictor

    def train_all_models(
        self, save_dir: str = "models", hyperparameter_tuning: bool = False
    ) -> dict[str, dict[str, Any]]:
        """
        Train models for all assets.

        Args:
            save_dir (str): Directory to save trained models
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning

        Returns:
            dict: Training results for each asset
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results = {}

        for asset, predictor in self.predictors.items():
            logger.info(f"Training model for {asset}...")

            # Train the model
            model_result = predictor.train_model(
                save_path=os.path.join(save_dir, f"{asset}_model.pkl"), hyperparameter_tuning=hyperparameter_tuning
            )

            results[asset] = model_result

        return results

    def predict_all(self, prediction_horizon: str = "market_open") -> dict[str, dict[str, Any]]:
        """
        Make predictions for all assets.

        Args:
            prediction_horizon (str): 'market_open', '24h', or '1h'

        Returns:
            dict: Predictions for each asset
        """
        predictions = {}

        for asset, predictor in self.predictors.items():
            if predictor.model is None:
                logger.warning(f"No trained model available for {asset}. Attempting to load...")

                # Try to load from default path
                model_path = f"models/{asset}_model.pkl"
                if os.path.exists(model_path):
                    if predictor._load_model(model_path):
                        logger.info(f"Loaded model for {asset} from {model_path}")
                    else:
                        predictions[asset] = {"error": f"Failed to load model for {asset}"}
                        continue
                else:
                    logger.error(f"No model file found for {asset}")
                    predictions[asset] = {"error": f"No trained model available for {asset}"}
                    continue

            # Make prediction
            logger.info(f"Making predictions for {asset} with horizon {prediction_horizon}...")
            try:
                predictions[asset] = predictor.predict_next_movement(prediction_horizon=prediction_horizon)
            except Exception as e:
                logger.error(f"Error predicting {asset} movements: {e}")
                predictions[asset] = {"error": str(e)}

        return predictions

    def get_latest_prices(self) -> dict[str, dict[str, Any]]:
        """
        Get the latest prices for all assets.

        Returns:
            dict: Latest prices and basic statistics
        """
        results = {}

        for asset in self.assets:
            try:
                # Get today's data
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)

                data = self.data_manager.download_data(
                    asset, start_date=yesterday.strftime("%Y-%m-%d"), end_date=today.strftime("%Y-%m-%d"), interval="1h"
                )

                if data is None or data.empty:
                    results[asset] = {"error": "No recent data available"}
                    continue

                latest = data.iloc[-1]

                # Calculate daily change
                if len(data) > 1:
                    first_price = data.iloc[0]["Close"]
                    latest_price = latest["Close"]
                    daily_change_pct = ((latest_price - first_price) / first_price) * 100
                else:
                    daily_change_pct = 0

                results[asset] = {"latest_price": latest["Close"], "daily_change_pct": daily_change_pct}
            except Exception as e:
                logger.error(f"Error fetching latest prices for {asset}: {e}")
                results[asset] = {"error": str(e)}

        return results
