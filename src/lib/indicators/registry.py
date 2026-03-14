"""
Registry for technical indicators.
"""

from typing import Any

from lib.core.logging_config import get_logger
from lib.indicators._shims import Registry
from lib.indicators.base import Indicator

logger = get_logger(__name__)


class IndicatorRegistry(Registry):
    """
    Registry for technical indicators.
    This registry maintains a mapping of indicator names to their implementations
    and provides methods for registering and retrieving indicators.
    """

    def __init__(self):
        """Initialize the indicator registry."""
        super().__init__("indicator")
        self._indicators: dict[str, type[Indicator]] = {}

        # Ensure logger is initialized
        if not hasattr(self, "logger"):
            self.logger = get_logger(__name__)

    def register(self, indicator_cls: type[Indicator]) -> None:
        """
        Register an indicator class.
        Args:
            indicator_cls: The indicator class to register.
        """
        indicator_name = indicator_cls.__name__.lower()
        self._indicators[indicator_name] = indicator_cls

        try:
            self.logger.debug("registered_indicator", indicator_name=indicator_name)
        except AttributeError:
            # Fallback if logger is still not available
            print(f"Registered indicator: {indicator_name}")

    def get(self, name: str) -> type[Indicator] | None:
        """
        Get an indicator class by name.
        Args:
            name: The name of the indicator.
        Returns:
            The indicator class or None if not found.
        """
        indicator_name = name.lower()
        indicator_cls = self._indicators.get(indicator_name)

        if indicator_cls is None:
            try:
                self.logger.warning("indicator_not_found", indicator_name=indicator_name)
            except AttributeError:
                print(f"Warning: Indicator not found: {indicator_name}")

        return indicator_cls

    def list(self) -> list[str]:
        """
        List all registered indicators.
        Returns:
            List of indicator names.
        """
        return list(self._indicators.keys())

    def create(self, name: str, **params: Any) -> Indicator | None:
        """
        Create an indicator instance by name with parameters.
        Args:
            name: The name of the indicator.
            **params: Parameters to pass to the indicator constructor.
        Returns:
            An indicator instance or None if not found.
        """
        indicator_cls = self.get(name)
        if indicator_cls is None:
            return None
        return indicator_cls(name=name, params=params)


# Global indicator registry instance
indicator_registry = IndicatorRegistry()


def register_indicator(cls: type[Indicator]) -> type[Indicator]:
    """
    Decorator for registering an indicator class.
    Args:
        cls: The indicator class to register.
    Returns:
        The registered indicator class.
    """
    indicator_registry.register(cls)
    return cls
