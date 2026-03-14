"""
Configuration Loading Module.

This module provides functions for loading and processing configuration files.
It supports YAML and JSON formats with environment variable expansion.
"""

import json
import os
import traceback
from pathlib import Path
from typing import Any

# Try to import yaml libraries
try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_configuration(config_path: str) -> tuple[bool, dict[str, Any]]:
    """
    Load configuration from a file with environment variable expansion.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple[bool, Dict]: Tuple with success flag and configuration dictionary
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        print(f"Warning: Configuration file not found: {config_path}")
        return False, {}

    try:
        file_extension = config_path_obj.suffix.lower()

        with open(config_path) as f:
            if file_extension in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    print(f"Warning: YAML configuration file specified but PyYAML not installed: {config_path}")
                    return False, {}

                config_data = yaml.safe_load(f)  # type: ignore[possibly-undefined]
            elif file_extension == ".json":
                config_data = json.load(f)
            else:
                # Try to guess format based on content
                content = f.read()
                f.seek(0)  # Reset file pointer

                if content.strip().startswith("{"):
                    # Looks like JSON
                    config_data = json.load(f)
                else:
                    # Try as YAML if available
                    if YAML_AVAILABLE:
                        config_data = yaml.safe_load(f)  # type: ignore[possibly-undefined]
                    else:
                        print(f"Warning: Could not determine file format for {config_path} and PyYAML not installed")
                        return False, {}

        # Expand environment variables in string values
        expanded_config = _expand_env_vars(config_data)

        return True, expanded_config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        traceback.print_exc()
        return False, {}


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.

    Args:
        obj: Configuration object (dict, list, str, etc.)

    Returns:
        Configuration with environment variables expanded
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Only try to expand if the string contains a potential env var pattern $VAR or ${VAR}
        if "$" in obj:
            try:
                import re

                # Two patterns: ${VAR} and $VAR
                pattern1 = r"\${([^}]+)}"  # ${VAR} format
                pattern2 = r"\$([A-Za-z0-9_]+)"  # $VAR format

                # Process ${VAR} format first
                def replace_with_env1(match):
                    env_var = match.group(1)
                    return os.environ.get(env_var) or match.group(0)

                result = re.sub(pattern1, replace_with_env1, obj)

                # Then process $VAR format
                def replace_with_env2(match):
                    env_var = match.group(1)
                    return os.environ.get(env_var) or match.group(0)

                result = re.sub(pattern2, replace_with_env2, result)

                return result
            except ImportError:
                # If re module not available, use a simpler approach for ${VAR} format only
                if "${" in obj and "}" in obj:
                    # Simple replacement for ${VAR} format without regex
                    in_var = False
                    var_name = ""
                    result = ""
                    i = 0
                    while i < len(obj):
                        if not in_var and obj[i : i + 2] == "${":
                            in_var = True
                            var_name = ""
                            i += 2  # Skip ${
                        elif in_var and obj[i] == "}":
                            in_var = False
                            env_value = os.environ.get(var_name, "${" + var_name + "}")
                            result += env_value
                            i += 1  # Skip }
                        elif in_var:
                            var_name += obj[i]
                            i += 1
                        else:
                            result += obj[i]
                            i += 1
                    return result

        # If no $ or error in processing, return original string
        return obj
    else:
        # Return other types (int, float, bool, etc.) unchanged
        return obj
