"""
ModelRegistry for model registration, persistence, and lifecycle management.

This module provides a comprehensive registry for models with:
1. Class-based model registration
2. File-based model persistence
3. Version tracking
4. Active model management
5. Metadata storage
"""

from __future__ import annotations

import inspect
import json
import os
import pickle
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.model._shims import logger
from lib.model.utils.metadata import ModelMetadata

if TYPE_CHECKING:
    import builtins

    from lib.model.base.model import BaseModel


class ModelRegistryEntry:
    """Entry in the model registry."""

    def __init__(
        self,
        model_id: str,
        path: str,
        metadata: ModelMetadata,
        is_active: bool = False,
        class_name: str = "",
    ):
        """
        Initialize a registry entry.

        Args:
            model_id: Unique model identifier
            path: Path to the saved model
            metadata: Model metadata
            is_active: Whether this is the active model of its type
            class_name: Original model class name
        """
        self.model_id = model_id
        self.path = path
        self.metadata = metadata
        self.is_active = is_active
        self.class_name = class_name

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": self.model_id,
            "path": self.path,
            "metadata": self.metadata.to_dict(),
            "is_active": self.is_active,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRegistryEntry:
        """Create entry from dictionary."""
        metadata = ModelMetadata.from_dict(data.get("metadata", {}))
        return cls(
            model_id=data.get("id") or "",
            path=data.get("path") or "",
            metadata=metadata,
            is_active=data.get("is_active", False),
            class_name=data.get("class_name") or "",
        )


class ModelRegistry:
    """
    Unified registry for models with class registration and persistence.

    This registry provides:
    1. Class-based model registration (in-memory)
    2. File-based model persistence
    3. Version tracking
    4. Active model management
    5. Metadata storage
    """

    def __init__(self, registry_path: str = ""):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to store the registry and models
        """
        # Set up logging
        self.logger = logger

        # In-memory class registry
        self._model_classes: dict[str, type[BaseModel]] = {}

        # File-based registry
        self.registry_path = Path(registry_path or os.path.join(os.environ.get("DATA_DIR", "."), "model_registry"))
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"

        # Load registry from disk
        self._entries: dict[str, ModelRegistryEntry] = self._load_registry()

        self.logger.info(
            f"ModelRegistry initialized with {len(self._model_classes)} classes and {len(self._entries)} saved models"
        )

    def _load_registry(self) -> dict[str, ModelRegistryEntry]:
        """Load registry from disk."""
        entries = {}
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)

                # Convert entries from dictionary
                for entry_data in data.get("models", []):
                    entry = ModelRegistryEntry.from_dict(entry_data)
                    entries[entry.model_id] = entry

                self.logger.debug(f"Loaded {len(entries)} entries from registry")
            except Exception as e:
                self.logger.error(f"Error loading registry: {e}")

        return entries

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "models": [entry.to_dict() for entry in self._entries.values()],
            }

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Saved {len(self._entries)} entries to registry")
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")

    def register_class(self, cls: type[BaseModel]) -> type[BaseModel]:
        """
        Register a model class.

        Args:
            cls: The model class to register

        Returns:
            The registered model class
        """
        class_name = cls.__name__.lower()
        self._model_classes[class_name] = cls
        self.logger.debug(f"Registered model class: {class_name}")
        return cls

    def get_class(self, name: str) -> type[BaseModel] | None:
        """
        Get a model class by name.

        Args:
            name: The name of the model class

        Returns:
            The model class or None if not found
        """
        class_name = name.lower()
        cls = self._model_classes.get(class_name)
        if cls is None:
            self.logger.warning(f"Model class not found: {class_name}")
        return cls

    def list_classes(self) -> builtins.list[str]:
        """
        List all registered model classes.

        Returns:
            List of model class names
        """
        return list(self._model_classes.keys())

    # Add compatibility methods for factory
    def list(self) -> builtins.list[str]:
        """
        List all registered model classes.
        (Compatibility method for ModelFactory)

        Returns:
            List of model class names
        """
        return self.list_classes()

    def get(self, name: str) -> type[BaseModel] | None:
        """
        Get a model class by name.
        (Compatibility method for ModelFactory)

        Args:
            name: The name of the model class

        Returns:
            The model class or None if not found
        """
        return self.get_class(name)

    def create(self, name: str, **params: Any) -> BaseModel | None:
        """
        Create a model instance by class name with parameters.
        (Compatibility method for ModelFactory)

        Args:
            name: The name of the model class
            **params: Parameters to pass to the model constructor

        Returns:
            A model instance or None if class not found
        """
        return self.create_instance(name, **params)

    def create_instance(self, name: str, **params: Any) -> BaseModel | None:
        """
        Create a model instance by class name with parameters.

        Args:
            name: The name of the model class
            **params: Parameters to pass to the model constructor

        Returns:
            A model instance or None if class not found
        """
        cls = self.get_class(name)
        if cls is None:
            return None

        try:
            # Get the constructor signature
            sig = inspect.signature(cls.__init__)
            # Prepare arguments that match the constructor
            ctor_params = sig.parameters
            init_args = {}
            # Exclude 'self'
            for param in ctor_params:
                if param == "self":
                    continue
                if param == "name":
                    init_args["name"] = name
                elif param in params:
                    init_args[param] = params[param]
            return cls(**init_args)
        except Exception as e:
            self.logger.error(f"Error creating instance of {name}: {e}")
            return None

    def register_model(
        self,
        model: BaseModel,
        model_type: str,
        version: str = "",
        metadata: dict[str, Any] | None = None,
        activate: bool = False,
    ) -> str:
        """
        Register a model instance with the registry.

        Args:
            model: The model instance to register
            model_type: Type identifier for the model
            version: Version string (defaults to timestamp)
            metadata: Additional metadata
            activate: Whether to set this model as active

        Returns:
            The generated model ID
        """
        # Generate unique model ID
        if metadata is None:
            metadata = {}
        version = version or datetime.now().strftime("%Y%m%d%H%M%S")
        model_id = f"{model_type}_{version}_{uuid.uuid4().hex[:8]}"

        # Create model directory
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{model_id}.pkl"

        # Create metadata
        meta_dict = metadata or {}
        if not isinstance(meta_dict, dict):
            meta_dict = {}

        # Add model class information
        class_name = model.__class__.__name__

        # Try to extract features and parameters from model
        features = getattr(model, "features", meta_dict.get("features", []))

        params = getattr(model, "params", meta_dict.get("params", {}))
        if not params and callable(getattr(model, "get_params", None)):
            try:
                params = model.get_params()
            except Exception:
                params = {}

        # Create metadata object
        model_metadata = ModelMetadata(
            model_type=model_type,
            version=version,
            params=params if isinstance(params, dict) else {},
            features=features if isinstance(features, list) else [],
            **meta_dict,
        )

        # Save the model
        try:
            # Try using the model's save method first
            save_method = getattr(model, "save", None)
            if callable(save_method):
                saved_path = save_method(str(model_path))
                if saved_path:
                    model_path = Path(str(saved_path))
            else:
                # Fall back to pickle
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # Save metadata separately
            meta_path = model_dir / f"{model_id}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(model_metadata.to_dict(), f, indent=2)

            self.logger.info(f"Saved model {model_id} to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
            raise

        # Create registry entry
        entry = ModelRegistryEntry(
            model_id=model_id, path=str(model_path), metadata=model_metadata, is_active=False, class_name=class_name
        )

        # Add to registry
        self._entries[model_id] = entry

        # Activate if requested
        if activate:
            self.activate_model(model_id)

        # Save registry
        self._save_registry()

        return model_id

    def load_model(self, model_id: str) -> tuple[BaseModel | None, ModelMetadata | None]:
        """
        Load a model from the registry.

        Args:
            model_id: The ID of the model to load

        Returns:
            Tuple of (model, metadata) or (None, None) if not found
        """
        entry = self._entries.get(model_id)
        if not entry:
            self.logger.warning(f"Model ID {model_id} not found in registry")
            return None, None

        try:
            # Check if model file exists
            if not os.path.exists(entry.path):
                self.logger.error(f"Model file not found: {entry.path}")
                return None, entry.metadata

            # Try to get class
            if entry.class_name and entry.class_name.lower() in self._model_classes:
                # Create a new instance
                cls = self._model_classes[entry.class_name.lower()]
                model = cls()

                # Try to use its load method if it exists
                load_method = getattr(model, "load", None)
                if callable(load_method):
                    load_method(entry.path)
                    return model, entry.metadata

            # Fall back to pickle
            with open(entry.path, "rb") as f:
                model = pickle.load(f)

            return model, entry.metadata
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            return None, entry.metadata

    def get_active_model(self, model_type: str) -> tuple[BaseModel | None, ModelMetadata | None]:
        """
        Get the active model of the specified type.

        Args:
            model_type: The type of model to get

        Returns:
            Tuple of (model, metadata) or (None, None) if no active model
        """
        for entry in self._entries.values():
            if entry.metadata.model_type == model_type and entry.is_active:
                return self.load_model(entry.model_id)

        self.logger.warning(f"No active model found for type: {model_type}")
        return None, None

    def activate_model(self, model_id: str) -> bool:
        """
        Set a model as the active one for its type.

        Args:
            model_id: The ID of the model to activate

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If model ID not found
        """
        entry = self._entries.get(model_id)
        if not entry:
            raise ValueError(f"Model ID {model_id} not found in registry")

        # Get model type
        model_type = entry.metadata.model_type

        # Deactivate all models of the same type
        for e in self._entries.values():
            if e.metadata.model_type == model_type:
                e.is_active = False

        # Activate this model
        entry.is_active = True

        # Save registry
        self._save_registry()

        self.logger.info(f"Activated model {model_id} for type {model_type}")
        return True

    def get_model_metadata(self, model_id: str) -> ModelMetadata | None:
        """
        Get metadata for a model.

        Args:
            model_id: The ID of the model

        Returns:
            The model metadata or None if not found
        """
        entry = self._entries.get(model_id)
        return entry.metadata if entry else None

    def update_model_metadata(self, model_id: str, **metadata: Any) -> bool:
        """
        Update metadata for a model.

        Args:
            model_id: The ID of the model
            **metadata: Metadata fields to update

        Returns:
            True if successful, False otherwise
        """
        entry = self._entries.get(model_id)
        if not entry:
            self.logger.warning(f"Model ID {model_id} not found in registry")
            return False

        # Update metadata fields
        for key, value in metadata.items():
            if key == "metrics" and hasattr(entry.metadata, "metrics"):
                # Merge metrics dictionaries
                entry.metadata.metrics.update(value)
            elif hasattr(entry.metadata, key):
                setattr(entry.metadata, key, value)
            else:
                # Skip unknown metadata fields or log a warning
                self.logger.warning(f"Unknown metadata field '{key}' for model {model_id}; skipping update.")

        # Update timestamp
        entry.metadata.updated_at = datetime.now().isoformat()

        # Save metadata file
        model_dir = Path(entry.path).parent
        meta_path = model_dir / f"{model_id}_metadata.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(entry.metadata.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata for {model_id}: {e}")
            return False

        # Save registry
        self._save_registry()

        return True

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: The ID of the model to delete

        Returns:
            True if successful, False otherwise
        """
        entry = self._entries.get(model_id)
        if not entry:
            self.logger.warning(f"Model ID {model_id} not found in registry")
            return False

        # Check if this is an active model
        if entry.is_active:
            self.logger.warning(f"Deleting active model {model_id}")

        # Remove from registry
        del self._entries[model_id]

        # Try to delete files
        try:
            model_dir = Path(entry.path).parent
            if model_dir.exists() and model_dir.name == model_id:
                shutil.rmtree(model_dir)
                self.logger.info(f"Deleted model directory: {model_dir}")
            else:
                # Just delete the model file
                if os.path.exists(entry.path):
                    os.remove(entry.path)
                    self.logger.info(f"Deleted model file: {entry.path}")

                # Try to delete metadata file
                meta_path = f"{os.path.splitext(entry.path)[0]}_metadata.json"
                if os.path.exists(meta_path):
                    os.remove(meta_path)
        except Exception as e:
            self.logger.error(f"Error deleting model files for {model_id}: {e}")

        # Save registry
        self._save_registry()

        return True

    def list_models(self, model_type: str = "") -> builtins.list[dict[str, Any]]:
        """
        List all registered model instances, optionally filtered by type.

        Args:
            model_type: Optional type to filter by

        Returns:
            List of model entries as dictionaries
        """
        results = []

        for entry in self._entries.values():
            if model_type is None or entry.metadata.model_type == model_type:
                model_info = {
                    "id": entry.model_id,
                    "type": entry.metadata.model_type,
                    "version": entry.metadata.version,
                    "created_at": entry.metadata.created_at,
                    "is_active": entry.is_active,
                    "metrics": entry.metadata.metrics,
                    "features": entry.metadata.features,
                    "is_gold_model": entry.metadata.is_gold_model,
                }
                results.append(model_info)

        return results

    def get_gold_models(self) -> builtins.list[dict[str, Any]]:
        """
        Get all registered gold price models.

        Returns:
            List of gold model entries
        """
        results = []

        for entry in self._entries.values():
            if entry.metadata.is_gold_model:
                model_info = {
                    "id": entry.model_id,
                    "type": entry.metadata.model_type,
                    "version": entry.metadata.version,
                    "created_at": entry.metadata.created_at,
                    "is_active": entry.is_active,
                    "metrics": entry.metadata.metrics,
                }
                results.append(model_info)

        return results

    def get_best_model(
        self, model_type: str, metric: str = "mse", higher_is_better: bool = False
    ) -> tuple[str | None, float | None]:
        """
        Get the best model of a type based on a metric.

        Args:
            model_type: The type of model to search for
            metric: The metric to compare
            higher_is_better: Whether higher values are better

        Returns:
            Tuple of (model_id, metric_value) or (None, None) if no models found
        """
        candidates = []

        for model_id, entry in self._entries.items():
            if entry.metadata.model_type == model_type and metric in entry.metadata.metrics:
                candidates.append((model_id, entry.metadata.metrics[metric]))

        if not candidates:
            return None, None

        if higher_is_better:
            best_id, best_value = max(candidates, key=lambda x: x[1])
        else:
            best_id, best_value = min(candidates, key=lambda x: x[1])

        return best_id, best_value

    def export_model(self, model_id: str, export_dir: str) -> str | None:
        """
        Export a model to a different directory.

        Args:
            model_id: The ID of the model to export
            export_dir: Directory to export to

        Returns:
            The path to the exported model or None if failed
        """
        entry = self._entries.get(model_id)
        if not entry:
            self.logger.warning(f"Model ID {model_id} not found in registry")
            return None

        # Create export directory
        export_path = Path(export_dir)
        export_path.mkdir(exist_ok=True, parents=True)

        # Define export files
        model_file = export_path / f"{model_id}.pkl"
        meta_file = export_path / f"{model_id}_metadata.json"

        try:
            # Copy model file
            shutil.copy2(entry.path, model_file)

            # Save metadata
            with open(meta_file, "w") as f:
                json.dump(entry.metadata.to_dict(), f, indent=2)

            self.logger.info(f"Exported model {model_id} to {export_dir}")
            return str(model_file)
        except Exception as e:
            self.logger.error(f"Error exporting model {model_id}: {e}")
            return None

    def import_model(self, import_path: str, activate: bool = False) -> str | None:
        """
        Import a model from a file.

        Args:
            import_path: Path to the model file
            activate: Whether to set this model as active

        Returns:
            The imported model ID or None if failed
        """
        import_file = Path(import_path)
        if not import_file.exists():
            self.logger.error(f"Import file not found: {import_path}")
            return None

        # Look for metadata file
        meta_path = import_file.with_name(f"{import_file.stem}_metadata.json")
        metadata = {}

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading metadata from {meta_path}: {e}")

        try:
            # Load the model
            with open(import_file, "rb") as f:
                model = pickle.load(f)

            # Extract model type
            model_type = metadata.get("model_type", "unknown")
            if hasattr(model, "model_type"):
                model_type = model.model_type

            # Extract version
            version = metadata.get("version", datetime.now().strftime("%Y%m%d%H%M%S"))

            # Register the model
            model_id = self.register_model(
                model=model, model_type=model_type, version=version, metadata=metadata, activate=activate
            )

            self.logger.info(f"Imported model as {model_id}")
            return model_id
        except Exception as e:
            self.logger.error(f"Error importing model: {e}")
            return None

    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._entries)


# Create decorator for registering model classes
def register_model(cls: type[BaseModel]) -> type[BaseModel]:
    """
    Decorator for registering a model class.

    Args:
        cls: The model class to register

    Returns:
        The registered model class
    """
    # Always use the global model_registry to register the class
    model_registry.register_class(cls)
    return cls


# Global model registry instance
model_registry = ModelRegistry()
