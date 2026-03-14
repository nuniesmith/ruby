"""
Model metadata tracking and registry module.

This module provides classes for tracking model metadata and maintaining a registry of models.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from lib.model._shims import logger


@dataclass
class ModelMetadata:
    """Metadata for storing model information and tracking.

    This class provides functionality to store, update, and serialize model metadata
    including model type, parameters, metrics, and other tracking information.
    """

    model_type: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str | None = None
    saved_at: str | None = None
    loaded_at: str | None = None
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    training_data_hash: str | None = None
    features: list[str] = field(default_factory=list)
    target: str = ""
    is_gold_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def to_json(self, pretty: bool = False) -> str:
        """Convert metadata to JSON string.

        Args:
            pretty: If True, format the JSON with indentation for readability

        Returns:
            JSON string representation of the metadata
        """
        indent = 4 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """Save metadata to file.

        Args:
            filepath: Path where the metadata JSON file will be saved
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        self.saved_at = datetime.now().isoformat()
        logger.info(f"Metadata saved to {filepath}")

    def update(self, **kwargs) -> None:
        """Update metadata fields.

        Args:
            **kwargs: Key-value pairs of fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Field '{key}' not found in ModelMetadata")
        self.updated_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create metadata from dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            ModelMetadata instance
        """
        # Filter out unknown fields
        known_fields = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**known_fields)

    @classmethod
    def load(cls, filepath: str) -> ModelMetadata:
        """Load metadata from file.

        Args:
            filepath: Path to the metadata JSON file

        Returns:
            ModelMetadata instance loaded from file

        Raises:
            Exception: If loading fails
        """
        try:
            with open(filepath) as f:
                data = json.load(f)
            metadata = cls.from_dict(data)
            metadata.loaded_at = datetime.now().isoformat()
            logger.info(f"Metadata loaded from {filepath}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata from {filepath}: {e}")
            raise


@dataclass
class ModelRegistry:
    """Registry for managing multiple model metadata objects.

    This class provides functionality to register, retrieve, filter, and save
    multiple model metadata objects.
    """

    models: dict[str, ModelMetadata] = field(default_factory=dict)

    def register(self, model_id: str, metadata: ModelMetadata) -> None:
        """Register a model with the registry.

        Args:
            model_id: Unique identifier for the model
            metadata: ModelMetadata instance for the model
        """
        self.models[model_id] = metadata
        logger.info(f"Model '{model_id}' registered")

    def get(self, model_id: str) -> ModelMetadata | None:
        """Get a model by ID.

        Args:
            model_id: Unique identifier for the model

        Returns:
            ModelMetadata instance if found, None otherwise
        """
        return self.models.get(model_id)

    def list_models(self) -> list[str]:
        """List all registered model IDs.

        Returns:
            List of model IDs in the registry
        """
        return list(self.models.keys())

    def filter(self, **kwargs) -> list[str]:
        """Filter models by metadata attributes.

        Args:
            **kwargs: Key-value pairs of attributes to filter by

        Returns:
            List of model IDs matching the filter criteria
        """
        results = []
        for model_id, metadata in self.models.items():
            match = True
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    attr_value = getattr(metadata, key)
                    if isinstance(value, list) and isinstance(attr_value, list):
                        # Check for any common elements
                        if not set(value).intersection(set(attr_value)):
                            match = False
                            break
                    elif attr_value != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                results.append(model_id)
        return results

    def save(self, directory: str) -> None:
        """Save registry to a directory.

        Args:
            directory: Directory where the registry will be saved
        """
        os.makedirs(directory, exist_ok=True)
        registry_file = os.path.join(directory, "registry.json")
        model_ids = list(self.models.keys())
        with open(registry_file, "w") as f:
            json.dump(model_ids, f, indent=4)

        for model_id, metadata in self.models.items():
            model_file = os.path.join(directory, f"{model_id}.json")
            metadata.save(model_file)

        logger.info(f"Registry saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> ModelRegistry:
        """Load registry from a directory.

        Args:
            directory: Directory where the registry is saved

        Returns:
            ModelRegistry instance loaded from directory

        Raises:
            Exception: If loading fails
        """
        registry = cls()
        registry_file = os.path.join(directory, "registry.json")

        try:
            with open(registry_file) as f:
                model_ids = json.load(f)

            for model_id in model_ids:
                model_file = os.path.join(directory, f"{model_id}.json")
                metadata = ModelMetadata.load(model_file)
                registry.register(model_id, metadata)

            logger.info(f"Registry loaded from {directory}")
            return registry
        except Exception as e:
            logger.error(f"Failed to load registry from {directory}: {e}")
            raise
