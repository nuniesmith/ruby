"""
Factory for creating technical indicators.
"""

from typing import Any

from lib.core.logging_config import get_logger
from lib.indicators.base import Indicator
from lib.indicators.registry import indicator_registry

logger = get_logger(__name__)


class IndicatorFactory:
    """
    Factory for creating technical indicators.
    This factory uses the indicator registry to create instances of indicators
    based on their name and configuration.
    """

    # Add a logger for the factory
    _logger = get_logger(__name__)

    def __init__(self):
        """Private constructor to prevent instantiation."""
        raise NotImplementedError("IndicatorFactory should not be instantiated")

    @staticmethod
    def create(name: str, **params: Any) -> Indicator:
        """
        Create an indicator by name with parameters.
        Args:
            name: The name of the indicator.
            **params: Parameters to pass to the indicator constructor.
        Returns:
            An indicator instance.
        Raises:
            ValueError: If the indicator is not found.
        """
        IndicatorFactory._logger.debug("creating_indicator", name=name, params=params)
        indicator = indicator_registry.create(name, **params)
        if indicator is None:
            IndicatorFactory._logger.error("indicator_not_found", name=name)
            raise ValueError(f"Indicator not found: {name}")
        return indicator

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> Indicator:
        """
        Create an indicator from a configuration dictionary.
        Args:
            config: A dictionary with 'name' and optional 'params'.
        Returns:
            An indicator instance.
        Raises:
            ValueError: If the configuration is invalid or the indicator is not found.
        """
        if not isinstance(config, dict):
            raise ValueError("Indicator configuration must be a dictionary")

        name = config.get("name")
        if name is None:
            raise ValueError("Indicator configuration must include 'name'")

        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Indicator parameters must be a dictionary")

        return IndicatorFactory.create(name, **params)

    @staticmethod
    def validate_config(config: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate an indicator configuration without creating the indicator.
        Args:
            config: A dictionary with 'name' and optional 'params'.
        Returns:
            A tuple of (is_valid, error_message).
        """
        if not isinstance(config, dict):
            return False, "Indicator configuration must be a dictionary"

        name = config.get("name")
        if name is None:
            return False, "Indicator configuration must include 'name'"

        params = config.get("params", {})
        if not isinstance(params, dict):
            return False, "Indicator parameters must be a dictionary"

        # Check if the indicator exists
        if indicator_registry.get(name) is None:
            return False, f"Indicator not found: {name}"

        return True, None

    @staticmethod
    def create_multiple(configs: list[dict[str, Any]], ignore_errors: bool = False) -> list[Indicator]:
        """
        Create multiple indicators from a list of configurations.
        Args:
            configs: A list of indicator configuration dictionaries.
            ignore_errors: If True, skip invalid configurations instead of raising errors.
        Returns:
            A list of indicator instances.
        """
        indicators = []

        for i, config in enumerate(configs):
            try:
                indicator = IndicatorFactory.create_from_config(config)
                indicators.append(indicator)
            except ValueError as e:
                if ignore_errors:
                    IndicatorFactory._logger.warning("skipping_invalid_config", index=i, error=str(e))
                    continue
                else:
                    raise ValueError(f"Error in configuration at index {i}: {str(e)}") from e

        return indicators

    @staticmethod
    def list_available() -> list[str]:
        """
        List all available indicators.
        Returns:
            List of indicator names.
        """
        return indicator_registry.list()

    @staticmethod
    def create_custom(indicator_class, **params: Any) -> Indicator:
        """
        Create a custom indicator directly from its class, bypassing the registry.
        Args:
            indicator_class: The class of the indicator to create.
            **params: Parameters to pass to the indicator constructor.
        Returns:
            An indicator instance.
        """
        try:
            return indicator_class(name=indicator_class.__name__.lower(), params=params)
        except Exception as e:
            IndicatorFactory._logger.error("custom_indicator_creation_failed", error=str(e))
            raise ValueError(f"Failed to create custom indicator: {str(e)}") from e
