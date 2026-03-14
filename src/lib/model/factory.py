"""
Factory for creating models with metadata tracking capabilities.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TypeVar

from lib.model.base.classifier import Classifier
from lib.model.base.estimator import Estimator
from lib.model.base.model import BaseModel
from lib.model.base.regressor import Regressor
from lib.model.registry import model_registry
from lib.model.utils.metadata import ModelMetadata, ModelRegistry

logger = logging.getLogger(__name__)

# Create a type variable for our BaseModel to help with type annotations
ModelType = TypeVar("ModelType", bound=BaseModel)


class ModelFactory:
    """
    Factory for creating models with optional metadata tracking.

    This factory uses the model registry to create instances of models
    based on their name and configuration. It can also track model metadata
    in a ModelRegistry if provided.
    """

    def __init__(self, metadata_registry: ModelRegistry | None = None):
        """
        Initialize the model factory.

        Args:
            metadata_registry: Optional registry for tracking model metadata
        """
        self.metadata_registry = metadata_registry

    @staticmethod
    def create(name: str, **params: Any) -> BaseModel:
        """
        Create a model by name with parameters.

        Args:
            name: The name of the model.
            **params: Parameters to pass to the model constructor.

        Returns:
            A model instance.

        Raises:
            ValueError: If the model is not found.
        """
        model_cls = model_registry.get(name)
        if model_cls is None:
            raise ValueError(f"BaseModel not found: {name}")
        model = model_cls(**params)
        return model

    def create_with_metadata(
        self,
        name: str,
        model_id: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        **params: Any,
    ) -> BaseModel:
        """
        Create a model with metadata tracking.

        Args:
            name: The name of the model.
            model_id: Unique identifier for the model. If None, a timestamp-based ID is generated.
            description: Description of the model.
            tags: List of tags for the model.
            **params: Parameters to pass to the model constructor.

        Returns:
            A model instance.

        Raises:
            ValueError: If the model is not found or metadata registry is not set.
        """
        if self.metadata_registry is None:
            raise ValueError(
                "Metadata registry not set. Initialize ModelFactory with a ModelRegistry or use create() instead."
            )

        # Create the model
        model = self.create(name, **params)

        # Generate model_id if not provided
        if model_id is None:
            model_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine model type for metadata
        model_type = "unknown"
        if isinstance(model, Classifier):
            model_type = "classifier"
        elif isinstance(model, Regressor):
            model_type = "regressor"
        elif isinstance(model, Estimator):
            model_type = "estimator"

        # Create metadata
        metadata = ModelMetadata(
            model_type=model_type,
            params=params,
            description=description,
            tags=tags or [],  # Empty list as default if tags is None
            version=getattr(model, "version", "1.0.0"),
        )

        # Register metadata
        self.metadata_registry.register(model_id, metadata)

        # Optionally attach metadata to model if supported
        try:
            # The BaseModel from src.lib.model.base.model has a metadata property
            model.metadata = metadata
        except (AttributeError, TypeError) as e:
            logger.debug(f"Model {model_id} does not support metadata attribute attachment: {e}")

        return model

    def create_from_config(self, config: dict[str, Any], with_metadata: bool = False) -> BaseModel:
        """
        Create a model from a configuration dictionary.

        Args:
            config: A dictionary with 'name' and optional 'params'.
            with_metadata: Whether to track metadata for the created model.

        Returns:
            A model instance.

        Raises:
            ValueError: If the configuration is invalid or the model is not found.
        """
        if not isinstance(config, dict):
            raise ValueError("BaseModel configuration must be a dictionary")

        name = config.get("name")
        if name is None:
            raise ValueError("BaseModel configuration must include 'name'")

        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("BaseModel parameters must be a dictionary")

        # Additional configuration options
        fit_params = config.get("fit_params", {})

        # Metadata options
        if with_metadata and self.metadata_registry is not None:
            model_id = config.get("id")
            description = config.get("description", "")
            tags = config.get("tags", [])

            # Create the model with metadata
            model = self.create_with_metadata(name, model_id=model_id, description=description, tags=tags, **params)
        else:
            # Create the model without metadata
            model = self.create(name, **params)

        # Store fit params with the model if it's an estimator
        if isinstance(model, Estimator) and fit_params:
            model.params["fit_params"] = fit_params

        return model

    def create_multiple(self, configs: list[dict[str, Any]], with_metadata: bool = False) -> list[BaseModel]:
        """
        Create multiple models from a list of configurations.

        Args:
            configs: A list of model configuration dictionaries.
            with_metadata: Whether to track metadata for the created models.

        Returns:
            A list of model instances.
        """
        return [self.create_from_config(config, with_metadata) for config in configs]

    @staticmethod
    def list_available() -> list[str]:
        """
        List all available models.

        Returns:
            List of model names.
        """
        # The model_registry has a different interface than ModelRegistry
        return model_registry.list()

    @staticmethod
    def list_available_by_type(model_type: type) -> list[str]:
        """
        List all available models of a specific type.

        Args:
            model_type: The model type (e.g., Classifier, Regressor).

        Returns:
            List of model names.
        """
        all_models = model_registry.list()
        filtered_models = []

        for model_name in all_models:
            # model_registry.get retrieves the model class by name
            model_cls = model_registry.get(model_name)
            if model_cls and issubclass(model_cls, model_type):
                filtered_models.append(model_name)

        return filtered_models

    @staticmethod
    def list_classifiers() -> list[str]:
        """
        List all available classifier models.

        Returns:
            List of classifier model names.
        """
        return ModelFactory.list_available_by_type(Classifier)

    @staticmethod
    def list_regressors() -> list[str]:
        """
        List all available regressor models.

        Returns:
            List of regressor model names.
        """
        return ModelFactory.list_available_by_type(Regressor)

    def get_metadata(self, model_id: str) -> ModelMetadata | None:
        """
        Get metadata for a model by ID.

        Args:
            model_id: The unique identifier for the model.

        Returns:
            ModelMetadata if found, None otherwise.
        """
        if self.metadata_registry is None:
            logger.warning("Metadata registry not set. Cannot retrieve metadata.")
            return None

        return self.metadata_registry.get(model_id)

    def list_models_with_metadata(self) -> list[str]:
        """
        List all model IDs with metadata in the registry.

        Returns:
            List of model IDs.
        """
        if self.metadata_registry is None:
            logger.warning("Metadata registry not set. No models with metadata available.")
            return []

        # Use list_models instead of list which is for the model_registry
        return self.metadata_registry.list_models()

    def filter_models(self, **kwargs) -> list[str]:
        """
        Filter models by metadata attributes.

        Args:
            **kwargs: Key-value pairs of metadata attributes to filter by.

        Returns:
            List of model IDs matching the filter criteria.
        """
        if self.metadata_registry is None:
            logger.warning("Metadata registry not set. Cannot filter models.")
            return []

        return self.metadata_registry.filter(**kwargs)
