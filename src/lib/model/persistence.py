"""
Model persistence utilities for saving and loading trained models.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from datetime import datetime
from typing import Any

from lib.model._shims import logger


class ModelLoader:
    """
    Utility for loading trained models from disk.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize the model loader.

        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        logger.debug(f"ModelLoader initialized with directory: {model_dir}")

    def load_latest_models(self) -> dict[str, Any]:
        """
        Load the latest version of each model.

        Returns:
            Dictionary of model name to model object
        """
        models: dict[str, Any] = {}

        try:
            # Get all model files
            model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
            if not model_files:
                logger.warning(f"No model files found in {self.model_dir}")
                return models

            # Group by model name and find latest version
            model_versions: dict[str, tuple[str, float]] = {}
            for model_file in model_files:
                basename = os.path.basename(model_file)
                # Expected format: model_name-timestamp.pkl
                if "-" in basename:
                    name_part = basename.split("-")[0]
                    timestamp = os.path.getmtime(model_file)

                    if name_part not in model_versions or timestamp > model_versions[name_part][1]:
                        model_versions[name_part] = (model_file, timestamp)

            # Load each latest model
            for name, (file_path, _) in model_versions.items():
                with open(file_path, "rb") as f:
                    models[name] = pickle.load(f)
                logger.debug(f"Loaded model {name} from {file_path}")

            # Load model metadata if exists
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    json.load(f)
                logger.debug(f"Loaded model metadata from {metadata_path}")

                # We could use metadata to enhance models or verify integrity

            logger.info(f"Loaded {len(models)} models")
            return models

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}

    def load_model(self, model_name: str) -> Any | None:
        """
        Load a specific model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Model object or None if not found
        """
        try:
            # Find the latest version of this model
            model_files = glob.glob(os.path.join(self.model_dir, f"{model_name}-*.pkl"))
            if not model_files:
                logger.warning(f"No files found for model {model_name}")
                return None

            # Sort by modification time (newest first)
            latest_file = sorted(model_files, key=os.path.getmtime, reverse=True)[0]

            # Load the model
            with open(latest_file, "rb") as f:
                model = pickle.load(f)

            logger.info(f"Loaded model {model_name} from {latest_file}")
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None


class ModelSaver:
    """
    Utility for saving trained models to disk.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize the model saver.

        Args:
            model_dir: Directory to save model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        logger.debug(f"ModelSaver initialized with directory: {model_dir}")

    def save_models(self, models: dict[str, Any]) -> bool:
        """
        Save multiple models to disk.

        Args:
            models: Dictionary of model name to model object

        Returns:
            True if successful, False otherwise
        """
        try:
            if not models:
                logger.warning("No models to save")
                return False

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Save each model
            for name, model in models.items():
                file_path = os.path.join(self.model_dir, f"{name}-{timestamp}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump(model, f)
                logger.debug(f"Saved model {name} to {file_path}")

            # Save metadata
            metadata = {"timestamp": timestamp, "model_count": len(models), "model_names": list(models.keys())}

            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved {len(models)} models with timestamp {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False

    def save_model(self, name: str, model: Any) -> bool:
        """
        Save a single model to disk.

        Args:
            name: Model name
            model: Model object

        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(self.model_dir, f"{name}-{timestamp}.pkl")

            with open(file_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Saved model {name} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model {name}: {e}")
            return False

    def cleanup_old_models(self, max_versions: int = 3) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.

        Args:
            max_versions: Maximum number of versions to keep per model

        Returns:
            Number of files deleted
        """
        try:
            # Group files by model name
            model_files: dict[str, list[tuple[str, float]]] = {}
            all_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))

            for file_path in all_files:
                basename = os.path.basename(file_path)
                if "-" in basename:
                    name_part = basename.split("-")[0]
                    if name_part not in model_files:
                        model_files[name_part] = []
                    model_files[name_part].append((file_path, os.path.getmtime(file_path)))

            # Delete older versions
            deleted_count = 0
            for _name, files in model_files.items():
                # Sort by modification time (newest first)
                sorted_files = sorted(files, key=lambda x: x[1], reverse=True)

                # Keep only max_versions
                for file_path, _ in sorted_files[max_versions:]:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"Deleted old model file: {file_path}")

            logger.info(f"Cleaned up {deleted_count} old model files")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            return 0
