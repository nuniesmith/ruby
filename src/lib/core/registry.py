import builtins
import inspect
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any, Generic, TypeVar

from lib.core.base import BaseComponent as Component
from lib.core.exceptions.validation import ValidationException
from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
K = TypeVar("K")


# Define missing classes that are referenced in the code
class RegistryCategory(Enum):
    """Categories for registry items"""

    COMPONENT = "component"
    SERVICE = "service"
    MODEL = "model"
    STRATEGY = "strategy"
    OTHER = "other"


class RegistrationError(Exception):
    """Error raised when registration fails"""

    pass


class NotRegisteredError(KeyError):
    """Error raised when an item is not registered"""

    pass


class RegistryItem(Generic[T]):
    """Container for items in the registry with metadata"""

    def __init__(
        self,
        item: T,
        tags: set[str] | None = None,
        category: str | RegistryCategory | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.item = item
        self.tags = tags or set()

        # Convert string category to enum if needed
        if isinstance(category, str):
            try:
                self.category = RegistryCategory(category)
            except ValueError:
                self.category = RegistryCategory.OTHER
        else:
            self.category = category or RegistryCategory.OTHER

        self.metadata = metadata or {}

    def add_tag(self, tag: str) -> None:
        """Add a tag to the item"""
        self.tags.add(tag)


class BaseRegistry(Generic[T]):
    """
    Basic registry for components or other objects.

    Provides a centralized registry for components, allowing:
    - Registration of components by name
    - Retrieval of components by name
    - Discovery of available components
    - Factory pattern for component creation
    """

    def __init__(self, base_class: type[T] | None = None, name: str = "BaseRegistry"):
        """
        Initialize a new registry.

        Args:
            base_class: Base class for registry entries (optional)
            name: Registry name for logging
        """
        self._registry: dict[str, type[T]] = {}
        self._instances: dict[str, T] = {}
        self._base_class = base_class
        self._name = name

    def register(self, name: str, cls: type[T]) -> None:
        """
        Register a class in the registry.

        Args:
            name: Name to register under
            cls: Class to register

        Raises:
            ValidationException: If the name is already registered or
                                if the class doesn't inherit from the base class
        """
        # Check if name is already registered
        if name in self._registry:
            raise ValidationException(
                message=f"'{name}' is already registered in the registry",
                details={"name": name, "existing_class": self._registry[name].__name__},
            )

        # Check if the class inherits from the base class
        if self._base_class and not issubclass(cls, self._base_class):
            raise ValidationException(
                message=f"Class {cls.__name__} must inherit from {self._base_class.__name__}",
                details={"class": cls.__name__, "base_class": self._base_class.__name__},
            )

        # Register the class
        self._registry[name] = cls
        logger.debug("Registered item in registry", registry=self._name, name=name, cls=cls.__name__)

    def unregister(self, name: str) -> None:
        """
        Unregister a class from the registry.

        Args:
            name: Name to unregister

        Raises:
            KeyError: If the name is not registered
        """
        if name in self._instances:
            del self._instances[name]

        if name in self._registry:
            cls = self._registry[name]
            del self._registry[name]
            logger.debug("Unregistered item from registry", registry=self._name, name=name, cls=cls.__name__)
        else:
            raise KeyError(f"'{name}' is not registered in the registry")

    def get(self, name: str) -> type[T]:
        """
        Get a class from the registry.

        Args:
            name: Name to retrieve

        Returns:
            Registered class

        Raises:
            KeyError: If the name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"'{name}' is not registered in the registry")

        return self._registry[name]

    def get_instance(self, name: str, *args, **kwargs) -> T:
        """
        Get or create an instance of a registered class.

        If an instance already exists, it will be returned.
        Otherwise, a new instance will be created.

        Args:
            name: Name of the class to instantiate
            *args: Arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            Instance of the registered class

        Raises:
            KeyError: If the name is not registered
        """
        if name not in self._instances:
            cls = self.get(name)
            self._instances[name] = cls(*args, **kwargs)
            logger.debug("Created instance", registry=self._name, name=name, cls=cls.__name__)

        return self._instances[name]

    def create(self, name: str, *args, **kwargs) -> T:
        """
        Create a new instance of a registered class.

        Unlike get_instance, this always creates a new instance.

        Args:
            name: Name of the class to instantiate
            *args: Arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            New instance of the registered class

        Raises:
            KeyError: If the name is not registered
        """
        cls = self.get(name)
        instance = cls(*args, **kwargs)
        logger.debug("Created instance", registry=self._name, name=name, cls=cls.__name__)
        return instance

    def has(self, name: str) -> bool:
        """
        Check if a name is registered.

        Args:
            name: Name to check

        Returns:
            True if the name is registered, False otherwise
        """
        return name in self._registry

    def list(self) -> list[str]:
        """
        Get a list of all registered names.

        Returns:
            List of registered names
        """
        return list(self._registry.keys())

    def clear(self) -> None:
        """
        Clear the registry.
        """
        self._instances.clear()
        self._registry.clear()
        logger.debug("Registry cleared", registry=self._name)

    def register_module(self, module: Any) -> builtins.list[str]:
        """
        Register all eligible classes from a module.

        A class is eligible if:
        - It's directly defined in the module
        - It inherits from the base class
        - It's not the base class itself

        Args:
            module: Module to scan for classes

        Returns:
            List of registered names
        """
        if not self._base_class:
            raise ValidationException(
                message="Cannot register module without a base class", details={"module": module.__name__}
            )

        registered = []

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip classes not defined in this module
            if obj.__module__ != module.__name__:
                continue

            # Skip the base class itself
            if obj is self._base_class:
                continue

            # Check if the class inherits from the base class
            if issubclass(obj, self._base_class):
                # Use the class name as the registration name
                reg_name = obj.__name__
                self.register(reg_name, obj)
                registered.append(reg_name)

        logger.debug(
            "Registered classes from module", registry=self._name, count=len(registered), module=module.__name__
        )
        return registered

    def __len__(self) -> int:
        """
        Get the number of registered entries.

        Returns:
            Number of registered entries
        """
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """
        Check if a name is registered.

        Args:
            name: Name to check

        Returns:
            True if the name is registered, False otherwise
        """
        return self.has(name)

    def __getitem__(self, name: str) -> type[T]:
        """
        Get a class from the registry.

        Args:
            name: Name to retrieve

        Returns:
            Registered class

        Raises:
            KeyError: If the name is not registered
        """
        return self.get(name)


class ComponentRegistry(BaseRegistry[Component]):
    """
    Registry specifically for Component classes.

    This is a specialized registry that adds component-specific
    functionality.
    """

    def __init__(self, name: str = "ComponentRegistry"):
        """
        Initialize a new component registry.

        Args:
            name: Registry name for logging
        """
        super().__init__(base_class=Component, name=name)

    def start_all(self) -> None:
        """
        Start all registered component instances.
        """
        for name, instance in self._instances.items():
            if not instance.is_running:  # type: ignore[attr-defined]
                logger.debug("Starting component", registry=self._name, name=name)
                instance.start()  # type: ignore[attr-defined]

    def stop_all(self) -> None:
        """
        Stop all registered component instances.
        """
        for name, instance in self._instances.items():
            if instance.is_running:  # type: ignore[attr-defined]
                logger.debug("Stopping component", registry=self._name, name=name)
                instance.stop()  # type: ignore[attr-defined]


class SingletonRegistry(BaseRegistry[T]):
    """
    Registry that enforces singleton instances.

    This registry ensures that only one instance of each registered
    class exists at any time.
    """

    def get_instance(self, name: str, *args, **kwargs) -> T:
        """
        Get the singleton instance of a registered class.

        If an instance already exists, it will be returned regardless
        of the provided arguments.

        Args:
            name: Name of the class to instantiate
            *args: Arguments to pass to the constructor (ignored if instance exists)
            **kwargs: Keyword arguments to pass to the constructor (ignored if instance exists)

        Returns:
            Singleton instance of the registered class

        Raises:
            KeyError: If the name is not registered
        """
        return super().get_instance(name, *args, **kwargs)

    def create(self, name: str, *args, **kwargs) -> T:
        """
        Create or get the singleton instance of a registered class.

        This behaves the same as get_instance to enforce the singleton pattern.

        Args:
            name: Name of the class to instantiate
            *args: Arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            Singleton instance of the registered class

        Raises:
            KeyError: If the name is not registered
        """
        return self.get_instance(name, *args, **kwargs)


class FactoryRegistry(Generic[T]):
    """
    Registry for factory functions.

    This registry stores factory functions that create instances
    of a specific type.
    """

    def __init__(self, return_type: type[T] | None = None, name: str = "FactoryRegistry"):
        """
        Initialize a new factory registry.

        Args:
            return_type: Expected return type of factory functions
            name: Registry name for logging
        """
        self._factories: dict[str, Callable[..., T]] = {}
        self._return_type = return_type
        self._name = name

    def register(self, name: str, factory: Callable[..., T]) -> None:
        """
        Register a factory function.

        Args:
            name: Name to register under
            factory: Factory function to register

        Raises:
            ValidationException: If the name is already registered
        """
        if name in self._factories:
            raise ValidationException(
                message=f"'{name}' is already registered in the factory registry", details={"name": name}
            )

        self._factories[name] = factory
        logger.debug("Registered factory in registry", registry=self._name, name=name)

    def unregister(self, name: str) -> None:
        """
        Unregister a factory function.

        Args:
            name: Name to unregister

        Raises:
            KeyError: If the name is not registered
        """
        if name in self._factories:
            del self._factories[name]
            logger.debug("Unregistered factory from registry", registry=self._name, name=name)
        else:
            raise KeyError(f"'{name}' is not registered in the factory registry")

    def get(self, name: str) -> Callable[..., T]:
        """
        Get a factory function.

        Args:
            name: Name to retrieve

        Returns:
            Registered factory function

        Raises:
            KeyError: If the name is not registered
        """
        if name not in self._factories:
            raise KeyError(f"'{name}' is not registered in the factory registry")

        return self._factories[name]

    def create(self, name: str, *args, **kwargs) -> T:
        """
        Create an instance using a registered factory function.

        Args:
            name: Name of the factory to use
            *args: Arguments to pass to the factory
            **kwargs: Keyword arguments to pass to the factory

        Returns:
            Instance created by the factory

        Raises:
            KeyError: If the name is not registered
            ValidationException: If the return type doesn't match the expected type
        """
        factory = self.get(name)
        instance = factory(*args, **kwargs)

        if self._return_type and not isinstance(instance, self._return_type):
            raise ValidationException(
                message=f"Factory '{name}' returned wrong type: expected {self._return_type.__name__}, got {type(instance).__name__}",
                details={
                    "name": name,
                    "expected_type": self._return_type.__name__,
                    "actual_type": type(instance).__name__,
                },
            )

        return instance

    def has(self, name: str) -> bool:
        """
        Check if a factory is registered.

        Args:
            name: Name to check

        Returns:
            True if the name is registered, False otherwise
        """
        return name in self._factories

    def list(self) -> list[str]:
        """
        Get a list of all registered factory names.

        Returns:
            List of registered factory names
        """
        return list(self._factories.keys())

    def clear(self) -> None:
        """
        Clear the registry.
        """
        self._factories.clear()
        logger.debug("Factory registry cleared", registry=self._name)


# Global basic component registry
base_component_registry = ComponentRegistry()


class Registry(Generic[K, T]):
    """
    A thread-safe registry for managing components throughout the application.

    The registry provides a centralized location to register and retrieve
    named components, with support for categorization, tagging, and filtering.

    Type Parameters:
        K: Type of keys in the registry (typically str)
        T: Type of items stored in the registry

    Example:
        # Create a registry for service components
        services = Registry[str, ServiceComponent]()

        # Register a service
        services.register("data_service", DataService())

        # Get a service
        data_service = services.get("data_service")

        # Filter services by tag
        db_services = services.get_by_tag("database")
    """

    def __init__(self, case_sensitive: bool = True):
        """
        Initialize a new registry.

        Args:
            case_sensitive: Whether keys should be case-sensitive
        """
        self._registry: dict[K, RegistryItem[T]] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._case_sensitive = case_sensitive

    def _normalize_key(self, key: K) -> K:
        """Normalize key based on case sensitivity setting."""
        if not self._case_sensitive and isinstance(key, str):
            return key.lower()  # type: ignore
        return key

    def register(
        self,
        name: K,
        item: T,
        tags: set[str] | None = None,
        category: str | RegistryCategory | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register an item with a given name.

        Args:
            name: Unique identifier for the item
            item: The item to register
            tags: Optional set of tags for categorization
            category: Optional category for the item
            metadata: Optional additional metadata
            overwrite: Whether to overwrite if item exists

        Raises:
            RegistrationError: If an item with the same name is already registered
                              and overwrite is False
        """
        name = self._normalize_key(name)

        with self._lock:
            if name in self._registry and not overwrite:
                raise RegistrationError(f"Item '{name}' is already registered.")

            registry_item = RegistryItem(item=item, tags=tags, category=category, metadata=metadata)

            self._registry[name] = registry_item
            logger.debug("Registered item in registry", name=name)

    def unregister(self, name: K) -> T:
        """
        Unregister an item by its name.

        Args:
            name: The name of the item to unregister

        Returns:
            The unregistered item

        Raises:
            NotRegisteredError: If no item with the given name is registered
        """
        name = self._normalize_key(name)

        with self._lock:
            if name not in self._registry:
                raise NotRegisteredError(f"Item '{name}' is not registered.")

            item = self._registry[name].item
            del self._registry[name]
            logger.debug("Unregistered item from registry", name=name)
            return item

    def get(self, name: K) -> T:
        """
        Retrieve an item by its name.

        Args:
            name: The name of the item to retrieve

        Returns:
            The registered item

        Raises:
            NotRegisteredError: If no item with the given name is registered
        """
        name = self._normalize_key(name)

        with self._lock:
            if name not in self._registry:
                raise NotRegisteredError(f"Item '{name}' is not registered.")

            return self._registry[name].item

    def get_with_metadata(self, name: K) -> RegistryItem[T]:
        """
        Retrieve an item with its metadata by name.

        Args:
            name: The name of the item to retrieve

        Returns:
            The registry item container with the item and its metadata

        Raises:
            NotRegisteredError: If no item with the given name is registered
        """
        name = self._normalize_key(name)

        with self._lock:
            if name not in self._registry:
                raise NotRegisteredError(f"Item '{name}' is not registered.")

            return self._registry[name]

    def get_safe(self, name: K, default: T | None = None) -> T | None:
        """
        Safely retrieve an item by its name, returning a default if not found.

        Args:
            name: The name of the item to retrieve
            default: Default value to return if item isn't found

        Returns:
            The registered item or the default value
        """
        name = self._normalize_key(name)

        with self._lock:
            if name not in self._registry:
                return default

            return self._registry[name].item

    def exists(self, name: K) -> bool:
        """
        Check if an item exists in the registry.

        Args:
            name: The name to check

        Returns:
            True if an item with the given name is registered, False otherwise
        """
        name = self._normalize_key(name)

        with self._lock:
            return name in self._registry

    def get_all(self) -> dict[K, T]:
        """
        Get all registered items.

        Returns:
            Dictionary mapping names to registered items
        """
        with self._lock:
            return {name: item.item for name, item in self._registry.items()}

    def get_all_with_metadata(self) -> dict[K, RegistryItem[T]]:
        """
        Get all registered items with their metadata.

        Returns:
            Dictionary mapping names to registry item containers
        """
        with self._lock:
            return self._registry.copy()

    def get_names(self) -> list[K]:
        """
        Get all registered item names.

        Returns:
            List of registered item names
        """
        with self._lock:
            return list(self._registry.keys())

    def get_by_tag(self, tag: str) -> dict[K, T]:
        """
        Get all items with a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            Dictionary mapping names to items with the specified tag
        """
        with self._lock:
            return {name: item.item for name, item in self._registry.items() if tag in item.tags}

    def get_by_category(self, category: str | RegistryCategory) -> dict[K, T]:
        """
        Get all items in a specific category.

        Args:
            category: The category to filter by

        Returns:
            Dictionary mapping names to items in the specified category
        """
        # Convert string to enum if needed
        if isinstance(category, str):
            try:
                category_enum = RegistryCategory(category)
            except ValueError:
                return {}
        else:
            category_enum = category

        with self._lock:
            return {name: item.item for name, item in self._registry.items() if item.category == category_enum}

    def filter(self, predicate: Callable[[RegistryItem[T]], bool]) -> dict[K, T]:
        """
        Filter items using a custom predicate function.

        Args:
            predicate: Function that takes a registry item and returns a boolean

        Returns:
            Dictionary mapping names to items for which the predicate returns True
        """
        with self._lock:
            return {name: item.item for name, item in self._registry.items() if predicate(item)}

    def add_tag(self, name: K, tag: str) -> None:
        """
        Add a tag to an existing registry item.

        Args:
            name: The name of the item
            tag: The tag to add

        Raises:
            NotRegisteredError: If no item with the given name is registered
        """
        name = self._normalize_key(name)

        with self._lock:
            if name not in self._registry:
                raise NotRegisteredError(f"Item '{name}' is not registered.")

            self._registry[name].add_tag(tag)

    def count(self) -> int:
        """
        Get the number of registered items.

        Returns:
            Number of items in the registry
        """
        with self._lock:
            return len(self._registry)

    def clear(self) -> None:
        """Clear all items from the registry."""
        with self._lock:
            self._registry.clear()
            logger.debug("Registry cleared")

    def __contains__(self, name: K) -> bool:
        """
        Check if an item exists in the registry.

        Enables the 'in' operator: `if name in registry:`

        Args:
            name: The name to check

        Returns:
            True if an item with the given name is registered, False otherwise
        """
        return self.exists(name)

    def __len__(self) -> int:
        """
        Get the number of registered items.

        Enables the len() function: `len(registry)`

        Returns:
            Number of items in the registry
        """
        return self.count()

    def __iter__(self):
        """
        Iterate over registered item names.

        Enables iteration: `for name in registry:`

        Yields:
            Names of registered items
        """
        with self._lock:
            yield from self._registry


# Create singleton registries for common use cases
ServiceRegistry = Registry[str, Any]()
ThreadSafeComponentRegistry = Registry[str, Any]()
StrategyRegistry = Registry[str, Any]()
ModelRegistry = Registry[str, Any]()


def get_registry(registry_type: str) -> Registry:
    """
    Get a singleton registry of the specified type.

    Args:
        registry_type: Type of registry to get

    Returns:
        The requested registry

    Raises:
        ValueError: If the registry type doesn't exist
    """
    registries = {
        "service": ServiceRegistry,
        "component": ThreadSafeComponentRegistry,
        "strategy": StrategyRegistry,
        "model": ModelRegistry,
    }

    if registry_type.lower() not in registries:
        raise ValueError(f"Unknown registry type '{registry_type}'. Available types: {', '.join(registries.keys())}")

    return registries[registry_type.lower()]
