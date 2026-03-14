"""Pine Script Generator — assembles modular ``.pine`` files into complete indicators."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import tempfile
from typing import Any

import yaml  # type: ignore[import-untyped]


class PineScriptGenerator:
    """
    Advanced Pine Script Generator for creating trading indicators.

    This class handles the assembly of modular Pine Script files into complete
    trading indicators based on configuration settings in a params.yaml file.

    The primary focus is generating the Ruby indicator.
    """

    def __init__(self, base_dir: str, logger_level: int = logging.INFO):
        """
        Initialize the Pine Script generator.

        Args:
            base_dir: Base directory containing Pine Script source files
            logger_level: Logging level (default: INFO)
        """
        self.base_dir = base_dir
        self.modules_dir = os.path.join(base_dir, "modules")

        # Setup logging
        logging.basicConfig(
            level=logger_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("PineScriptGenerator")

        # Load parameters and validate
        self.params = self._load_params()
        self.validation_result = self._validate_params()

        # Use file orders from params.yaml if available, otherwise use defaults
        self.file_orders = self._get_file_orders()

        # Get output filename configurations
        self.output_filenames = self._get_output_filenames()

        # Track processed files and their order
        self.processed_files: dict[str, list[str]] = {}

        # Set up cache directory with environment variable or fallbacks
        self._setup_cache_directory()

        # Set default indicator type to Ruby
        self.default_indicator = "ruby"

        self.logger.info("PineScriptGenerator initialized with base directory: %s", base_dir)
        self.logger.info("Using cache directory: %s", self.cache_dir)
        self.logger.info("Default indicator set to: %s", self.default_indicator)

    def _setup_cache_directory(self) -> None:
        """Set up a cache directory with proper fallback mechanisms for containerised environments."""
        # Try different locations in order of preference
        cache_locations = [
            os.environ.get("PINE_CACHE_DIR"),  # Environment variable (first priority)
            os.path.join("/app/data/cache", "pine"),  # App data directory
            os.path.join(self.base_dir, ".cache"),  # Original location in source directory
            os.path.join(tempfile.gettempdir(), "pine_cache"),  # System temp directory (last resort)
        ]

        for location in cache_locations:
            if location is None:
                continue

            try:
                os.makedirs(location, exist_ok=True)
                # Test if the directory is writable by creating a test file
                test_file = os.path.join(location, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)

                # If we reach here, the directory is writable
                self.cache_dir: str | None = location
                self.logger.info("Using cache directory: %s", self.cache_dir)
                return
            except OSError as e:
                self.logger.warning("Cache location %s is not writable: %s", location, e)

        # If all locations fail, use a memory-only "cache"
        self.logger.warning("No writable cache directory found, using memory-only cache")
        self.cache_dir = None
        self.memory_cache: dict[str, Any] = {}  # In-memory cache for fallback

    def _load_params(self) -> dict[str, Any]:
        """
        Load parameters from params.yaml.

        Returns:
            Dictionary containing parsed YAML parameters
        """
        params_path = os.path.join(self.base_dir, "params.yaml")
        try:
            with open(params_path) as f:
                params = yaml.safe_load(f)
                self.logger.info("Successfully loaded parameters from %s", params_path)
                return params
        except FileNotFoundError:
            self.logger.error("Parameters file not found at %s", params_path)
            return {}
        except yaml.YAMLError as e:
            self.logger.error("Error parsing YAML file: %s", e)
            return {}

    def _validate_params(self) -> dict[str, list[str]]:
        """
        Validate parameters for completeness and correctness.

        Returns:
            Dictionary containing validation errors and warnings
        """
        validation_result: dict[str, list[str]] = {"errors": [], "warnings": []}

        # Check if version info exists
        if "version" not in self.params:
            validation_result["warnings"].append("Version information missing from params")
            self.params["version"] = {"major": 1, "minor": 0, "patch": 0, "name": "Default"}

        # Check if file_orders exists
        if "file_orders" not in self.params:
            validation_result["warnings"].append("File orders missing from params, will use defaults")

        # Check if output_filenames exists
        if "output_filenames" not in self.params:
            validation_result["warnings"].append("Output filenames missing from params, will use defaults")

        # Check for empty keys in output_filenames and correct them
        if "output_filenames" in self.params:
            for key in list(self.params["output_filenames"].keys()):
                if key == "":
                    validation_result["warnings"].append("Empty key found in output_filenames. Changed to 'slim_1m'.")
                    # Create a modified copy with the empty key replaced
                    output_filenames: dict[str, str] = {}
                    for k, v in self.params["output_filenames"].items():
                        if k == "":
                            output_filenames["slim_1m"] = v
                        else:
                            output_filenames[k] = v
                    self.params["output_filenames"] = output_filenames

        # Check that modules directory exists
        if not os.path.exists(self.modules_dir):
            validation_result["errors"].append(f"Modules directory not found at {self.modules_dir}")

        # Check if key Ruby files exist
        if "file_orders" in self.params and "ruby" in self.params["file_orders"]:
            missing_files: list[str] = []
            critical_files = ["header.pine", "inputs.pine", "core_calculations.pine"]
            for filename in critical_files:
                file_path = os.path.join(self.modules_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)

            if missing_files:
                validation_result["errors"].append(f"Critical Ruby files missing: {', '.join(missing_files)}")

        # Log validation results
        if validation_result["errors"]:
            self.logger.error("Validation errors: %s", validation_result["errors"])
        if validation_result["warnings"]:
            self.logger.warning("Validation warnings: %s", validation_result["warnings"])

        return validation_result

    def _get_file_orders(self) -> dict[str, list[str]]:
        """
        Get file orders from params.yaml if available, otherwise use defaults.

        Returns:
            Dictionary of file orders by indicator type
        """
        # Default file orders — follows the params.yaml structure with focus on Ruby
        default_file_orders: dict[str, list[str]] = {
            "ruby": [
                "header.pine",
                "inputs.pine",
                "core_calculations.pine",
                "top_g_channel.pine",
                "wave_analysis.pine",
                "market_regime.pine",
                "volatility.pine",
                "session_orb.pine",
                "quality_score.pine",
                "signal_logic.pine",
                "trade_management.pine",
                "visualization.pine",
                "signal_labels.pine",
                "info_table.pine",
                "alerts.pine",
                "footer.pine",
            ],
        }

        # Check if file_orders exists in params
        if "file_orders" in self.params:
            file_orders: dict[str, list[str]] = {}
            params_file_orders = self.params["file_orders"]

            # Process each indicator type
            for indicator_type, files in params_file_orders.items():
                # Skip custom if not enabled
                if indicator_type == "custom" and isinstance(files, dict) and not files.get("enabled", False):
                    continue

                # Handle custom indicator type
                if (
                    indicator_type == "custom"
                    and isinstance(files, dict)
                    and files.get("enabled", False)
                    and "files" in files
                ):
                    file_orders["custom"] = files["files"]
                # Handle regular indicator types
                elif indicator_type != "custom":
                    if isinstance(files, list):
                        file_orders[indicator_type] = files
                    else:
                        self.logger.warning(
                            "Unexpected type for file order '%s': %s, skipping",
                            indicator_type,
                            type(files),
                        )

            # Ensure Ruby is in the file orders
            if "ruby" not in file_orders and "ruby" in default_file_orders:
                file_orders["ruby"] = default_file_orders["ruby"]
                self.logger.warning("Ruby indicator not found in file_orders, using default")

            self.logger.info("Using file orders from params.yaml: %s", ", ".join(file_orders.keys()))
            return file_orders

        self.logger.warning("No file_orders found in params.yaml, using defaults")
        return default_file_orders

    def _get_output_filenames(self) -> dict[str, str]:
        """
        Get output filenames from params if available, otherwise use defaults.

        Returns:
            Dictionary of output filenames by indicator type
        """
        # Default output filenames with focus on Ruby
        default_filenames: dict[str, str] = {
            "ruby": "ruby.pine",
        }

        # Add other indicators from params.yaml if available
        if "output_filenames" in self.params:
            output_filenames: dict[str, str] = {}

            # Process the output filenames, converting any empty key to 'slim_1m'
            for indicator_type, filename in self.params["output_filenames"].items():
                if indicator_type == "":
                    output_filenames["slim_1m"] = filename
                    self.logger.info("Converted empty key in output_filenames to 'slim_1m'")
                else:
                    output_filenames[indicator_type] = filename

            # Ensure Ruby is in the output filenames
            if "ruby" not in output_filenames:
                output_filenames["ruby"] = default_filenames["ruby"]

            self.logger.info("Using output filenames from params.yaml: %s", output_filenames)
            return output_filenames

        self.logger.info("No output_filenames found in params.yaml, using defaults: %s", default_filenames)
        return default_filenames

    def _read_pine_file(self, filename: str) -> str:
        """
        Read contents of a Pine Script file from the modules directory.

        Args:
            filename: Name of the Pine Script file

        Returns:
            File contents as string
        """
        file_path = os.path.join(self.modules_dir, filename)

        if not os.path.exists(file_path):
            self.logger.warning("File %s not found at %s", filename, file_path)
            return ""  # Return empty string to continue the process

        try:
            with open(file_path) as f:
                content = f.read()
                self.logger.debug("Successfully read file: %s (%d bytes)", filename, len(content))
                return content
        except Exception as e:
            self.logger.error("Error reading file %s: %s", filename, e)
            return ""  # Return empty string to continue the process

    def _get_parameter_value(self, param_key: str) -> Any:
        """
        Get a parameter value from the nested parameters dictionary.

        Args:
            param_key: Dot-separated parameter key (e.g., 'top_g.range_length.default')

        Returns:
            Parameter value or None if not found
        """
        parts = param_key.split(".")
        current = self.params

        for part in parts:
            if part in current:
                current = current[part]
            else:
                return None

        return current

    def _replace_params(self, content: str) -> str:
        """
        Replace parameter placeholders in Pine Script with actual values.

        Args:
            content: Pine Script content

        Returns:
            Modified Pine Script with parameter values inserted
        """
        # Identify all parameter placeholders in the content
        placeholder_pattern = r"\{params\.([a-zA-Z0-9_\.]+)\}"
        placeholders = re.findall(placeholder_pattern, content)

        # Process each placeholder
        for param_key in placeholders:
            try:
                # Get parameter value
                value = self._get_parameter_value(param_key)

                if value is None:
                    self.logger.warning("Parameter not found: %s", param_key)
                    continue

                # Convert parameter value to Pine Script format
                if isinstance(value, bool):
                    str_value = str(value).lower()
                elif isinstance(value, (int, float)):
                    str_value = str(value)
                elif isinstance(value, str):
                    # Escape string if needed
                    str_value = f'"{value}"'
                elif isinstance(value, list):
                    # Handle arrays in Pine Script format
                    if all(isinstance(item, str) for item in value):
                        # List of strings needs quotes
                        quoted = [f'"{item}"' for item in value]
                        str_value = f"[{', '.join(quoted)}]"
                    else:
                        # List of numbers or mixed content
                        str_value = f"[{', '.join(str(item) for item in value)}]"
                elif isinstance(value, dict):
                    # For dictionary values, serialize to JSON-like format
                    self.logger.warning("Dictionary value for %s might not convert correctly", param_key)
                    pairs = []
                    for k, v in value.items():
                        if isinstance(v, str):
                            pairs.append(f'"{k}": "{v}"')
                        else:
                            pairs.append(f'"{k}": {v}')
                    str_value = f"{{{', '.join(pairs)}}}"
                else:
                    self.logger.warning("Unsupported parameter type for %s: %s", param_key, type(value))
                    continue

                # Replace placeholder with value
                placeholder = f"{{params.{param_key}}}"
                if placeholder in content:
                    self.logger.debug("Replacing %s with %s", placeholder, str_value)
                    content = content.replace(placeholder, str_value)

            except Exception as e:
                self.logger.error("Error processing parameter %s: %s", param_key, e)

        return content

    def _process_imports(self, content: str) -> str:
        """
        Process and remove import statements to prevent duplicates.

        Args:
            content: Pine Script content

        Returns:
            Content with import statements removed
        """
        # Remove Pine Script import statements
        return re.sub(r"^import.*$", "", content, flags=re.MULTILINE)

    def get_available_indicator_types(self) -> list[str]:
        """
        Get list of available indicator types.

        Returns:
            List of indicator type names
        """
        return list(self.file_orders.keys())

    def _generate_file_hash(self, file_path: str) -> str:
        """
        Generate a hash of a file's contents for caching.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash of the file content
        """
        if not os.path.exists(file_path):
            return "file_not_found"

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                return hashlib.md5(file_content).hexdigest()  # noqa: S324
        except Exception as e:
            self.logger.error("Error generating hash for %s: %s", file_path, e)
            return "hash_error"

    def _check_cache(self, indicator_type: str) -> str | None:
        """
        Check if a cached version of the generated script exists.

        Args:
            indicator_type: Type of indicator to check

        Returns:
            Cached script content if available and valid, None otherwise
        """
        # Skip caching if cache_dir is None (memory-only mode)
        if self.cache_dir is None:
            # Check memory cache instead
            if hasattr(self, "memory_cache") and indicator_type in self.memory_cache:
                cache_entry = self.memory_cache[indicator_type]
                # Check expiration (1 hour)
                now = datetime.datetime.now()
                if (now - cache_entry["timestamp"]).total_seconds() <= 3600:
                    self.logger.info("Using in-memory cached version for %s", indicator_type)
                    return cache_entry["content"]
            return None

        # Generate cache file path
        cache_file = os.path.join(self.cache_dir, f"{indicator_type}_cache.pine")
        meta_file = os.path.join(self.cache_dir, f"{indicator_type}_meta.json")

        # Check if cache files exist
        if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
            return None

        try:
            # Load metadata
            with open(meta_file) as f:
                metadata = json.load(f)

            # Check timestamp (cache expiration after 1 hour)
            cache_time = datetime.datetime.fromisoformat(metadata["timestamp"])
            now = datetime.datetime.now()
            if (now - cache_time).total_seconds() > 3600:  # 1 hour expiration
                self.logger.info("Cache expired for %s", indicator_type)
                return None

            # Check if file hashes match
            current_hashes = {}
            for filename in metadata["files"]:
                fp = os.path.join(self.modules_dir, filename)
                current_hashes[filename] = self._generate_file_hash(fp)

            # Compare hashes
            if current_hashes != metadata["hashes"]:
                self.logger.info("Source files changed for %s, cache invalid", indicator_type)
                return None

            # Load cache content
            with open(cache_file) as f:
                cache_content = f.read()
                self.logger.info("Using cached version for %s", indicator_type)
                return cache_content

        except Exception as e:
            self.logger.error("Error checking cache: %s", e)
            return None

    def _update_cache(self, indicator_type: str, content: str, processed_files: list[str]) -> None:
        """
        Update the cache with a newly generated script.

        Args:
            indicator_type: Type of indicator
            content: Generated script content
            processed_files: List of processed files
        """
        # If cache_dir is None, use memory cache instead
        if self.cache_dir is None:
            try:
                if not hasattr(self, "memory_cache"):
                    self.memory_cache = {}

                # Generate file hashes for metadata
                file_hashes = {}
                for filename in processed_files:
                    file_path = os.path.join(self.modules_dir, filename)
                    file_hashes[filename] = self._generate_file_hash(file_path)

                # Store in memory cache
                self.memory_cache[indicator_type] = {
                    "content": content,
                    "timestamp": datetime.datetime.now(),
                    "files": processed_files,
                    "hashes": file_hashes,
                }
                self.logger.info("Updated in-memory cache for %s", indicator_type)
                return
            except Exception as e:
                self.logger.error("Error updating in-memory cache: %s", e)
                return

        # Handle file-based caching
        try:
            # Generate cache file paths
            cache_file = os.path.join(self.cache_dir, f"{indicator_type}_cache.pine")
            meta_file = os.path.join(self.cache_dir, f"{indicator_type}_meta.json")

            # Generate file hashes
            file_hashes = {}
            for filename in processed_files:
                file_path = os.path.join(self.modules_dir, filename)
                file_hashes[filename] = self._generate_file_hash(file_path)

            # Create metadata
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "files": processed_files,
                "hashes": file_hashes,
                "indicator_type": indicator_type,
            }

            # Save cache
            with open(cache_file, "w") as f:
                f.write(content)

            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info("Updated cache for %s", indicator_type)

        except Exception as e:
            self.logger.error("Error updating cache (continuing without caching): %s", e)

    def generate_full_script(self, indicator_type: str) -> tuple[str, list[str]]:
        """
        Generate the full Pine Script by combining all source files.

        Args:
            indicator_type: Type of indicator ('ruby', 'slim_1m', or custom)

        Returns:
            Tuple containing complete Pine Script as a string and list of processed files
        """
        # Normalize indicator type (convert empty string to 'slim_1m')
        if indicator_type == "":
            indicator_type = "slim_1m"
            self.logger.info("Empty indicator type converted to 'slim_1m'")

        # Verify indicator type is supported
        if indicator_type not in self.file_orders:
            msg = f"Unsupported indicator type: {indicator_type}. Supported types: {', '.join(self.file_orders.keys())}"
            raise ValueError(msg)

        # Check if a valid cached version exists
        cached_script = self._check_cache(indicator_type)
        if cached_script:
            return cached_script, self.processed_files.get(indicator_type, [])

        # Get file order for the indicator
        file_order = self.file_orders[indicator_type]

        # Collect and process each file
        script_parts: list[str] = []
        proc_files: list[str] = []

        for filename in file_order:
            self.logger.info("Processing file: %s", filename)

            content = self._read_pine_file(filename)

            # Skip empty content (files not found)
            if not content:
                self.logger.warning("Skipping empty or missing file: %s", filename)
                continue

            # Process parameter replacements
            content = self._replace_params(content)

            # Process imports
            content = self._process_imports(content)

            script_parts.append(content)
            proc_files.append(filename)

        self.logger.info("Processed %d files for %s: %s", len(proc_files), indicator_type, ", ".join(proc_files))

        # Save processed files list
        self.processed_files[indicator_type] = proc_files

        # Combine all parts
        full_script = "\n\n".join(script_parts)

        # Add generation metadata as comment
        version_info = self.params.get("version", {})
        version_str = (
            f"{version_info.get('major', '1')}.{version_info.get('minor', '0')}.{version_info.get('patch', '0')}"
        )
        version_name = version_info.get("name", "Standard")

        metadata = (
            f"\n// {indicator_type.upper()} Indicator v{version_str} - {version_name}\n"
            f"// Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"// Description: {version_info.get('description', 'Pine Script trading indicator')}\n"
            f"// This file was automatically generated using the PineScriptGenerator\n"
        )

        full_script = metadata + full_script

        # Update cache
        self._update_cache(indicator_type, full_script, proc_files)

        return full_script, proc_files

    def save_script(self, indicator_type: str, output_dir: str) -> str:
        """
        Save the generated script to a file.

        Args:
            indicator_type: Type of indicator ('ruby', 'slim_1m', or custom)
            output_dir: Directory to save the generated script

        Returns:
            Path to the saved file
        """
        # Generate the script
        full_script, _processed_files = self.generate_full_script(indicator_type)

        # Define output filename based on indicator type
        if indicator_type in self.output_filenames:
            output_filename = self.output_filenames[indicator_type]
        else:
            output_filename = f"{indicator_type}.pine"

        # Combine output directory and filename
        output_path = os.path.join(output_dir, output_filename)

        # Ensure output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.logger.error("Cannot create output directory %s: %s", output_dir, e)
            # Try a fallback location
            fallback_dir = os.path.join(tempfile.gettempdir(), "pine_output")
            os.makedirs(fallback_dir, exist_ok=True)
            output_path = os.path.join(fallback_dir, output_filename)
            self.logger.warning("Using fallback output directory: %s", fallback_dir)

        # Save the script
        try:
            with open(output_path, "w") as f:
                f.write(full_script)
            self.logger.info("%s Pine Script generated and saved to %s", indicator_type.upper(), output_path)
        except OSError as e:
            self.logger.error("Error saving script to %s: %s", output_path, e)
            raise

        return output_path

    def get_indicator_stats(self, indicator_type: str) -> dict[str, Any]:
        """
        Get statistics about an indicator.

        Args:
            indicator_type: Type of indicator

        Returns:
            Dictionary containing statistics about the indicator
        """
        full_script, proc_files = self.generate_full_script(indicator_type)

        # Calculate statistics
        return {
            "indicator_type": indicator_type,
            "module_count": len(proc_files),
            "modules": proc_files,
            "total_lines": full_script.count("\n") + 1,
            "total_chars": len(full_script),
            "generation_time": datetime.datetime.now().isoformat(),
        }

    def generate_default_indicator(self, output_dir: str) -> str:
        """
        Generate the default Ruby indicator.

        Args:
            output_dir: Directory to save the generated script

        Returns:
            Path to the saved file
        """
        return self.save_script(self.default_indicator, output_dir)


# ---------------------------------------------------------------------------
# Default output directory — overridable via PINE_OUTPUT_DIR env-var
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get(
    "PINE_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pine_output"),
)


def main() -> None:
    """Main function for command line execution."""
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Generate Pine Script indicators")
    parser.add_argument(
        "--indicators",
        nargs="+",
        default=["ruby"],
        help='Indicators to generate (default: ruby, use "all" for all available)',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated scripts",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available indicator types and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics for generated indicators",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate parameters and exit",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Clean cache before generating",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Set the cache directory (overrides environment variable)",
    )

    args = parser.parse_args()

    # Set cache directory if specified
    if args.cache_dir:
        os.environ["PINE_CACHE_DIR"] = args.cache_dir

    # Configure logging level
    logger_level = logging.DEBUG if args.verbose else logging.INFO

    # Get the pine directory path (parent of modules directory)
    pine_dir = os.path.dirname(os.path.abspath(__file__))

    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(pine_dir, "output")

    # Create generator instance
    generator = PineScriptGenerator(pine_dir, logger_level=logger_level)

    # Clean cache if requested
    if args.clean_cache and generator.cache_dir:
        try:
            for file in os.listdir(generator.cache_dir):
                os.remove(os.path.join(generator.cache_dir, file))
            print(f"Cache cleaned in {generator.cache_dir}")
        except OSError as e:
            print(f"Error cleaning cache: {e}")

    # Validate parameters if requested
    if args.validate:
        validation_result = generator.validation_result
        if validation_result["errors"]:
            print("Validation errors:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
        if validation_result["warnings"]:
            print("Validation warnings:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")
        if not validation_result["errors"] and not validation_result["warnings"]:
            print("Validation passed with no issues")
        return

    # List available indicators if requested
    if args.list:
        available = generator.get_available_indicator_types()
        print(f"Available indicator types: {', '.join(available)}")
        return

    # Determine which indicators to generate
    indicators_to_generate = generator.get_available_indicator_types() if "all" in args.indicators else args.indicators

    # Generate and save indicator scripts
    for indicator in indicators_to_generate:
        try:
            output_path = generator.save_script(indicator, output_dir)
            print(f"Generated {indicator} script: {output_path}")

            # Show statistics if requested
            if args.stats:
                stats = generator.get_indicator_stats(indicator)
                print(f"\nStatistics for {indicator}:")
                print(f"  Module count: {stats['module_count']}")
                print(f"  Total lines: {stats['total_lines']}")
                print(f"  Total characters: {stats['total_chars']}")

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error generating {indicator}: {e}")


# This script generates the Ruby indicator by default
if __name__ == "__main__":
    main()
