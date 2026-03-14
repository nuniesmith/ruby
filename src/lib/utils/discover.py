"""
Module and Configuration Discovery Utilities.

This module provides functions for discovering modules and configuration files.
"""

import importlib
import os
from pathlib import Path
from typing import Any

# Define defaults
DEFAULT_CONFIG_PATHS = ["/app/config/fks/{service_name}.yaml", "{base_dir}/config/fks/{service_name}.yaml"]


def try_import(module_name: str) -> Any | None:
    """
    Try to import a module and return it, or None if not found.

    Args:
        module_name: Name of the module to import

    Returns:
        Module if successfully imported, None otherwise
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def find_config_file(service_name: str, base_dir: Path) -> str:
    """
    Find a configuration file for the service.

    Args:
        service_name: Service name
        base_dir: Base directory for the service

    Returns:
        str: Path to configuration file if found, or default path
    """
    # Try environment variable first
    env_config = os.environ.get(f"{service_name.upper()}_CONFIG")
    if env_config and Path(env_config).exists() and os.access(Path(env_config), os.R_OK):
        return env_config

    # Try default paths
    for path_template in DEFAULT_CONFIG_PATHS:
        path_str = path_template.format(service_name=service_name, base_dir=base_dir)
        path = Path(path_str)
        if path.exists() and os.access(path, os.R_OK):
            return str(path)

    # Try to find any yaml file in the config directories
    search_dirs = [
        Path("/app/config/fks"),
        Path("/etc/fks"),
        Path("config/fks"),
        base_dir / "config",
        base_dir / "config/fks",
    ]

    for directory in search_dirs:
        if directory.exists() and directory.is_dir():
            for extension in [".yaml", ".yml", ".json"]:
                config_file = directory / f"{service_name}{extension}"
                if config_file.exists() and os.access(config_file, os.R_OK):
                    return str(config_file)

    # Return the first path as default (it will be created if needed)
    return DEFAULT_CONFIG_PATHS[0].format(service_name=service_name, base_dir=base_dir)
