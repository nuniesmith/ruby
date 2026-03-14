"""
Manager for technical indicators.

This module provides a centralized way to manage, calculate, and organize technical indicators.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from lib.core.logging_config import get_logger
from lib.indicators.base import Indicator
from lib.indicators.factory import IndicatorFactory

logger = get_logger(__name__)


class IndicatorManager:
    """
    Manager for technical indicators.

    This class provides a centralized way to manage multiple indicators,
    handle their calculation, and organize them into groups.

    Example:
        # Create a manager
        manager = IndicatorManager()

        # Add indicators
        manager.add_indicator({'name': 'rsi', 'params': {'period': 14}}, groups=['momentum'])
        manager.add_indicator({'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}, groups=['trend'])

        # Calculate indicators on data
        results = manager.calculate_all(df)

        # Get results
        rsi_value = manager.get_indicator('rsi').get_value()

        # Calculate only momentum indicators
        momentum_results = manager.calculate_group('momentum', df)
    """

    def __init__(self, indicators: list[Indicator | dict[str, Any]] | None = None):
        """
        Initialize the indicator manager with optional indicators.

        Args:
            indicators: Optional list of indicator instances or configurations to add.
        """
        self.logger = get_logger(__name__)
        self._indicators: dict[str, Indicator] = {}
        self._groups: dict[str, set[str]] = {}

        # Add initial indicators if provided
        if indicators:
            for ind in indicators:
                self.add_indicator(ind)

        self.logger.info("indicator_manager_initialized", indicator_count=len(self._indicators))

    def add_indicator(
        self,
        indicator_or_symbol: Indicator | dict[str, Any] | str,
        type_or_groups: str | list[str] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Add an indicator to the manager.

        This method supports different calling styles:
        1. add_indicator(indicator, groups=None) - Original style with indicator object or config dict
        2. add_indicator(symbol, indicator_type, params) - Legacy style with separate arguments
        3. add_indicator(indicator, groups=['group1']) - Keyword style

        Args:
            indicator_or_symbol: An indicator instance, configuration dictionary, or a symbol string.
            type_or_groups: Optional list of group names, or indicator type string in legacy mode.
            params: Optional indicator parameters dictionary (used only in legacy mode).
            **kwargs: Additional keyword arguments:
                - groups: Alternative way to specify groups (takes precedence over type_or_groups)

        Returns:
            The ID of the added indicator.

        Raises:
            ValueError: If the indicator is invalid or a duplicate.
        """
        # Handle groups keyword argument if provided
        groups = kwargs.get("groups")

        # If groups is explicitly provided as a keyword arg, it takes precedence
        if groups is not None:
            # Ensure groups is a list
            if isinstance(groups, str):
                groups = [groups]
            # Now call the internal implementation with these arguments
            return self._add_indicator_internal(indicator_or_symbol, groups)  # type: ignore[arg-type]

        # Legacy call format: (symbol, indicator_type, params)
        if isinstance(indicator_or_symbol, str) and isinstance(type_or_groups, str):
            symbol = indicator_or_symbol
            indicator_type = type_or_groups

            # Create indicator config dictionary
            indicator_config = {
                "name": indicator_type,
                "params": params or {},
                "symbol": symbol,  # Store symbol info in the config
            }

            # Use symbol as group
            legacy_groups = [symbol]

            # Now call the original implementation with the transformed arguments
            return self._add_indicator_internal(indicator_config, legacy_groups)

        # Original call format: (indicator, groups)
        else:
            if not isinstance(indicator_or_symbol, (Indicator, dict)):
                raise ValueError(
                    "When not using legacy format, indicator_or_symbol must be an Indicator instance or a configuration dictionary"
                )

            # Convert type_or_groups to a list if it's a string
            passed_groups = None
            if type_or_groups is not None:
                passed_groups = [type_or_groups] if isinstance(type_or_groups, str) else type_or_groups

            return self._add_indicator_internal(indicator_or_symbol, passed_groups)

    def _add_indicator_internal(self, indicator: Indicator | dict[str, Any], groups: list[str] | None = None) -> str:
        """
        Internal implementation of add_indicator.
        """
        # Handle indicator instance
        if isinstance(indicator, Indicator):
            indicator_id = indicator.name
            self._indicators[indicator_id] = indicator
        # Handle indicator configuration
        elif isinstance(indicator, dict):
            try:
                ind = IndicatorFactory.create_from_config(indicator)
                indicator_id = ind.name
                self._indicators[indicator_id] = ind
            except ValueError as e:
                self.logger.error("failed_to_create_indicator_from_config", error=str(e))
                raise
        else:
            raise ValueError("Indicator must be an Indicator instance or a configuration dictionary")

        # Add to groups if specified
        if groups:
            for group in groups:
                if group not in self._groups:
                    self._groups[group] = set()
                self._groups[group].add(indicator_id)

        self.logger.debug("added_indicator", indicator_id=indicator_id)
        return indicator_id

    def remove_indicator(self, indicator_id: str) -> bool:
        """
        Remove an indicator from the manager.

        Args:
            indicator_id: The ID of the indicator to remove.

        Returns:
            True if the indicator was removed, False if not found.
        """
        if indicator_id not in self._indicators:
            self.logger.warning("indicator_not_found", indicator_id=indicator_id)
            return False

        # Remove from indicators dict
        del self._indicators[indicator_id]

        # Remove from all groups
        for group in self._groups.values():
            if indicator_id in group:
                group.remove(indicator_id)

        self.logger.debug("removed_indicator", indicator_id=indicator_id)
        return True

    def get_indicator(self, indicator_id: str) -> Indicator | None:
        """
        Get an indicator by ID.

        Args:
            indicator_id: The ID of the indicator.

        Returns:
            The indicator instance or None if not found.
        """
        return self._indicators.get(indicator_id)

    def list_indicators(self) -> list[str]:
        """
        List all indicators in the manager.

        Returns:
            List of indicator IDs.
        """
        return list(self._indicators.keys())

    def add_group(self, group_name: str, indicator_ids: list[str] | None = None) -> None:
        """
        Add a new group or update an existing group.

        Args:
            group_name: The name of the group.
            indicator_ids: Optional list of indicator IDs to include in the group.
        """
        if group_name not in self._groups:
            self._groups[group_name] = set()

        if indicator_ids:
            for ind_id in indicator_ids:
                if ind_id in self._indicators:
                    self._groups[group_name].add(ind_id)
                else:
                    self.logger.warning("indicator_not_found", indicator_id=ind_id)

        self.logger.debug("added_updated_group", group_name=group_name)

    def remove_group(self, group_name: str) -> bool:
        """
        Remove a group (but not the indicators in it).

        Args:
            group_name: The name of the group to remove.

        Returns:
            True if the group was removed, False if not found.
        """
        if group_name not in self._groups:
            self.logger.warning("group_not_found", group_name=group_name)
            return False

        del self._groups[group_name]
        self.logger.debug("removed_group", group_name=group_name)
        return True

    def list_groups(self) -> list[str]:
        """
        List all groups in the manager.

        Returns:
            List of group names.
        """
        return list(self._groups.keys())

    def get_group_indicators(self, group_name: str) -> list[str]:
        """
        Get all indicator IDs in a group.

        Args:
            group_name: The name of the group.

        Returns:
            List of indicator IDs in the group.
        """
        if group_name not in self._groups:
            self.logger.warning("group_not_found", group_name=group_name)
            return []

        return list(self._groups[group_name])

    def calculate_all(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate all indicators on the provided data.

        Args:
            data: Input dataframe containing market data.

        Returns:
            Dictionary mapping indicator IDs to their calculated values.
        """
        results = {}
        for ind_id, indicator in self._indicators.items():
            try:
                results[ind_id] = indicator(data)
                self.logger.debug("calculated_indicator", indicator_id=ind_id)
            except Exception as e:
                self.logger.error("indicator_calculation_failed", indicator_id=ind_id, error=str(e))
                results[ind_id] = None

        return results

    def calculate_group(self, group_name: str, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate indicators in a group on the provided data.

        Args:
            group_name: The name of the group.
            data: Input dataframe containing market data.

        Returns:
            Dictionary mapping indicator IDs to their calculated values.
        """
        if group_name not in self._groups:
            self.logger.warning("group_not_found", group_name=group_name)
            return {}

        results = {}
        for ind_id in self._groups[group_name]:
            if ind_id in self._indicators:
                try:
                    results[ind_id] = self._indicators[ind_id](data)
                    self.logger.debug("calculated_indicator", indicator_id=ind_id)
                except Exception as e:
                    self.logger.error("indicator_calculation_failed", indicator_id=ind_id, error=str(e))
                    results[ind_id] = None
            else:
                self.logger.warning("indicator_not_found", indicator_id=ind_id)

        return results

    def calculate_indicators(self, indicator_ids: list[str], data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calculate specific indicators on the provided data.

        Args:
            indicator_ids: List of indicator IDs to calculate.
            data: Input dataframe containing market data.

        Returns:
            Dictionary mapping indicator IDs to their calculated values.
        """
        results = {}
        for ind_id in indicator_ids:
            if ind_id in self._indicators:
                try:
                    results[ind_id] = self._indicators[ind_id](data)
                    self.logger.debug("calculated_indicator", indicator_id=ind_id)
                except Exception as e:
                    self.logger.error("indicator_calculation_failed", indicator_id=ind_id, error=str(e))
                    results[ind_id] = None
            else:
                self.logger.warning("indicator_not_found", indicator_id=ind_id)

        return results

    def filter_indicators(self, filter_func: Callable[[Indicator], bool]) -> list[str]:
        """
        Filter indicators based on a custom function.

        Args:
            filter_func: Function that takes an Indicator and returns a boolean.

        Returns:
            List of indicator IDs that match the filter.
        """
        return [ind_id for ind_id, ind in self._indicators.items() if filter_func(ind)]

    def reset_all(self) -> None:
        """
        Reset the state of all indicators.
        """
        for indicator in self._indicators.values():
            indicator.reset()

        self.logger.debug("reset_all_indicators")

    def reset_group(self, group_name: str) -> bool:
        """
        Reset the state of indicators in a group.

        Args:
            group_name: The name of the group.

        Returns:
            True if the group was found and reset, False otherwise.
        """
        if group_name not in self._groups:
            self.logger.warning("group_not_found", group_name=group_name)
            return False

        for ind_id in self._groups[group_name]:
            if ind_id in self._indicators:
                self._indicators[ind_id].reset()

        self.logger.debug("reset_group_indicators", group_name=group_name)
        return True

    def get_all_values(self) -> dict[str, float | dict[str, float] | None]:
        """
        Get the latest values of all calculated indicators.

        Returns:
            Dictionary mapping indicator IDs to their latest values.
        """
        values: dict[str, float | dict[str, float] | None] = {}
        for ind_id, indicator in self._indicators.items():
            if indicator.values is not None:
                try:
                    values[ind_id] = indicator.get_value()
                except Exception as e:
                    self.logger.error("error_getting_indicator_value", indicator_id=ind_id, error=str(e))
                    values[ind_id] = None

        return values

    def get_group_values(self, group_name: str) -> dict[str, float | dict[str, float] | None]:
        """
        Get the latest values of indicators in a group.

        Args:
            group_name: The name of the group.

        Returns:
            Dictionary mapping indicator IDs to their latest values.
        """
        if group_name not in self._groups:
            self.logger.warning("group_not_found", group_name=group_name)
            return {}

        values: dict[str, float | dict[str, float] | None] = {}
        for ind_id in self._groups[group_name]:
            if ind_id in self._indicators and self._indicators[ind_id].values is not None:
                try:
                    values[ind_id] = self._indicators[ind_id].get_value()
                except Exception as e:
                    self.logger.error("error_getting_indicator_value", indicator_id=ind_id, error=str(e))
                    values[ind_id] = None

        return values

    def is_indicator_calculated(self, indicator_id: str) -> bool:
        """
        Check if an indicator has been calculated.

        Args:
            indicator_id: The ID of the indicator.

        Returns:
            True if the indicator has values, False otherwise.
        """
        if indicator_id not in self._indicators:
            return False

        return self._indicators[indicator_id].values is not None

    def save_to_config(self) -> dict[str, Any]:
        """
        Save the manager's state to a configuration dictionary.

        Returns:
            Dictionary with manager configuration.
        """
        indicators_config = []
        for ind_id, indicator in self._indicators.items():
            indicators_config.append({"name": ind_id, "params": indicator.params})

        groups_config = {}
        for group_name, ind_ids in self._groups.items():
            groups_config[group_name] = list(ind_ids)

        return {"indicators": indicators_config, "groups": groups_config}

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> "IndicatorManager":
        """
        Create an IndicatorManager from a configuration dictionary.

        Args:
            config: Dictionary with manager configuration.

        Returns:
            IndicatorManager instance.
        """
        manager = cls()

        # Add indicators
        if "indicators" in config:
            for ind_config in config["indicators"]:
                try:
                    manager.add_indicator(ind_config)
                except ValueError as e:
                    manager.logger.error("failed_to_load_indicator", error=str(e))

        # Add groups
        if "groups" in config:
            for group_name, ind_ids in config["groups"].items():
                manager.add_group(group_name, ind_ids)

        return manager

    def add_indicators_by_category(self, category: str) -> list[str]:
        """
        Add all indicators of a specific category.

        Args:
            category: The category of indicators to add (e.g., 'trend', 'momentum', 'volatility').

        Returns:
            List of added indicator IDs.
        """
        from lib.indicators import technical_indicators

        added_indicators = []
        for ind_cls in technical_indicators:
            # Extract category from module path
            ind_category = ind_cls.__module__.split(".")[-2]

            if ind_category == category:
                try:
                    indicator = IndicatorFactory.create(ind_cls.__name__.lower())
                    self.add_indicator(indicator, [category])
                    added_indicators.append(indicator.name)
                except Exception as e:
                    self.logger.error("failed_to_add_indicator", indicator_class=ind_cls.__name__, error=str(e))

        self.logger.info("added_indicators_by_category", count=len(added_indicators), category=category)
        return added_indicators

    def create_combined_dataframe(self, data: pd.DataFrame, indicator_ids: list[str] | None = None) -> pd.DataFrame:
        """
        Calculate indicators and combine them into a single DataFrame with the original data.

        Args:
            data: Input dataframe containing market data.
            indicator_ids: Optional list of indicator IDs to include (default: all).

        Returns:
            DataFrame with original data and indicator values.
        """
        # Start with a copy of the original data
        result = data.copy()

        # Determine which indicators to calculate
        if indicator_ids is None:
            indicator_ids = self.list_indicators()

        # Calculate each indicator and add to the result
        for ind_id in indicator_ids:
            if ind_id in self._indicators:
                try:
                    ind_result = self._indicators[ind_id](data)

                    if isinstance(ind_result, pd.DataFrame):
                        # If multi-column result, prefix columns with indicator name
                        for col in ind_result.columns:
                            result[f"{ind_id}_{col}"] = ind_result[col]
                    else:
                        # Single-column result
                        result[ind_id] = ind_result

                    self.logger.debug("added_indicator_to_combined_dataframe", indicator_id=ind_id)
                except Exception as e:
                    self.logger.error("indicator_calculation_failed", indicator_id=ind_id, error=str(e))

        return result

    def create_batch_indicator_calculation(
        self, df: pd.DataFrame, indicator_configs: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create and calculate multiple indicators in a single operation.

        Args:
            df: DataFrame with market data
            indicator_configs: List of indicator configurations

        Returns:
            DataFrame with original data and added indicator columns
        """
        result = df.copy()

        # Create temporary indicators for batch calculation
        temp_indicators = []
        for config in indicator_configs:
            try:
                ind = IndicatorFactory.create_from_config(config)
                temp_indicators.append(ind)
            except ValueError as e:
                self.logger.error("failed_to_create_indicator_from_config", error=str(e))

        # Calculate each indicator and add to result
        for ind in temp_indicators:
            try:
                ind_result = ind(df)

                if isinstance(ind_result, pd.DataFrame):
                    # Multi-column result
                    for col in ind_result.columns:
                        result[f"{ind.name}_{col}"] = ind_result[col]
                else:
                    # Single-column result
                    result[ind.name] = ind_result

                self.logger.debug("added_indicator_to_result_dataframe", indicator_name=ind.name)
            except Exception as e:
                self.logger.error("indicator_calculation_failed", indicator_name=ind.name, error=str(e))

        return result
