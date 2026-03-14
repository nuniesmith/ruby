"""
Prediction Generator Module

This module handles the core logic for generating predictions
using trained models for different assets.
"""

import asyncio
import os
from datetime import datetime
from typing import Any

from lib.model._shims import logger
from lib.model.prediction.manager import ModelManager

AssetDataManager: Any = None  # stub: data.manager.AssetDataManager
get_config: Any = None  # stub: core.constants.manager.get_config


class PredictionGenerator:
    """
    Handles generating predictions from trained models.
    """

    def __init__(self, model_manager: ModelManager | None = None, data_fetcher: Any | None = None):
        """
        Initialize the prediction generator.

        Args:
            model_manager: Model manager instance for accessing models
            data_fetcher: Data fetcher for accessing market data
        """
        # Get constants
        self.constants = get_config() if get_config is not None else None  # type: ignore[misc]

        self.model_manager = model_manager or ModelManager()
        self.data_fetcher = data_fetcher or self.model_manager.data_fetcher

    async def predict(
        self, asset: str, days_ahead: int = 7, include_confidence: bool = True, include_probabilities: bool = False
    ) -> dict[str, Any]:
        """
        Make predictions for a specific asset.

        Args:
            asset: Asset to predict (e.g., 'gold', 'bitcoin')
            days_ahead: Number of days ahead to predict
            include_confidence: Whether to include confidence scores
            include_probabilities: Whether to include probability distributions

        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making predictions for {asset}")

        model = self.model_manager.get_model(asset)
        if not model:
            logger.error(f"No model available for {asset}")
            return {"asset": asset, "error": f"No model available for {asset}"}

        try:
            import pathlib

            model_path = str(pathlib.Path(self.model_manager.model_dir) / f"{asset}_model.pkl")

            # Check if model file exists and load it
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found for {asset}: {model_path}")
                logger.info("Attempting to predict with untrained model")
            elif not hasattr(model, "is_trained") or not model.is_trained:
                # Load the model if it exists but isn't loaded
                try:
                    await asyncio.to_thread(model.load, model_path)  # type: ignore[union-attr]
                    logger.info(f"Loaded existing model for {asset}")
                except Exception as e:
                    logger.error(f"Error loading model for {asset}: {str(e)}")

            # Make predictions
            predict_fn: Any = model.predict  # type: ignore[union-attr]
            predictions = await asyncio.to_thread(
                predict_fn,
                days_ahead=days_ahead,
                include_confidence=include_confidence,
                include_probabilities=include_probabilities,
            )

            # Get latest price data for reference
            latest_data: Any = None
            if self.data_fetcher is not None:
                latest_data = await asyncio.to_thread(self.data_fetcher.get_latest_prices, asset)  # type: ignore[union-attr]

            return {
                "asset": asset,
                "model_type": self.model_manager.model_types.get(asset, "unknown"),
                "predictions": predictions,
                "latest_data": latest_data,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error making predictions for {asset}: {str(e)}")
            return {"asset": asset, "error": str(e)}

    async def predict_all(
        self,
        assets: list[str] | None = None,
        days_ahead: int = 7,
        include_confidence: bool = True,
        include_probabilities: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Make predictions for multiple assets.

        Args:
            assets: List of assets to predict (defaults to all supported assets)
            days_ahead: Number of days ahead to predict
            include_confidence: Whether to include confidence scores
            include_probabilities: Whether to include probability distributions

        Returns:
            Dictionary mapping asset names to their prediction results
        """
        if assets is None:
            assets = self.constants.SUPPORTED_ASSETS if self.constants is not None else []

        assert assets is not None

        # Use gather to run predictions concurrently
        tasks = []
        for asset in assets:
            tasks.append(
                self.predict(
                    asset,
                    days_ahead=days_ahead,
                    include_confidence=include_confidence,
                    include_probabilities=include_probabilities,
                )
            )

        # Wait for all predictions to complete
        results_list = await asyncio.gather(*tasks)

        # Convert list of results to dictionary keyed by asset name
        results = {}
        for result in results_list:
            asset_name = result.get("asset")
            if asset_name:
                predictions = result.get("predictions")
                if predictions:
                    results[asset_name] = predictions
                else:
                    results[asset_name] = {"error": "No predictions available"}

        return results

    async def get_price_info(self, assets: list[str]) -> dict[str, Any]:
        """
        Get the latest price information for specified assets.

        Args:
            assets: List of assets to get prices for

        Returns:
            Dictionary with latest price information by asset
        """
        price_info = {}
        for asset in assets:
            try:
                if self.data_fetcher is None:
                    continue
                latest = await asyncio.to_thread(self.data_fetcher.get_latest_prices, asset)  # type: ignore[union-attr]
                if latest:
                    price_info[asset] = latest
            except Exception as e:
                logger.warning(f"Error getting latest prices for {asset}: {e}")

        return price_info
