import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from jsonschema import ValidationError, validate

from lib.core.logging_config import get_logger

from .classes import ConfigError
from .general_error import GeneralError

logger = get_logger(__name__)


class ExceptionLoader:
    """
    Dynamically loads exceptions from a YAML configuration file and validates them against a schema.
    """

    def __init__(self, config_path: Path, schema_path: Path):
        """
        Initialize the ExceptionLoader with paths to the configuration and schema files.

        Args:
            config_path (Path): Path to the exceptions configuration file.
            schema_path (Path): Path to the schema file for validation.
        """
        self.config_path = self._validate_path(config_path, "Config")
        self.schema_path = self._validate_path(schema_path, "Schema")

    def _validate_path(self, path: Path, name: str) -> Path:
        """
        Validate the existence and readability of a given path.

        Args:
            path (Path): The path to validate.
            name (str): Name of the file being validated.

        Returns:
            Path: The validated path.

        Raises:
            ConfigError: If the file does not exist or is not readable.
        """
        if not path.is_file() or not path.exists():
            logger.error("file_not_found", file_type=name, path=str(path))
            raise ConfigError(f"{name} file not found or inaccessible: {path}")
        logger.info("file_validated", file_type=name, path=str(path))
        return path

    def load_exceptions(self) -> dict[str, type[GeneralError]]:
        """
        Load exceptions from a YAML file and validate against the schema.

        Returns:
            Dict[str, Type[GeneralError]]: Mapping of exception names to their dynamically created classes.

        Raises:
            ConfigError: If the configuration file is not found, malformed, or invalid.
        """
        try:
            with self.config_path.open("r") as file:
                data = yaml.safe_load(file)

            self._validate_schema(data)

            if not isinstance(data, dict) or not data.get("exceptions"):
                raise ConfigError(f"Malformed or empty configuration file: {self.config_path}")

            exceptions = self._create_exceptions(data["exceptions"])
            logger.info("exceptions_loaded", count=len(exceptions))
            return exceptions

        except (yaml.YAMLError, ValidationError) as e:
            logger.error("exceptions_load_error", error=str(e))
            raise ConfigError(f"Error loading exceptions: {e}") from e
        except Exception as e:
            logger.exception("exceptions_load_unexpected_error", error=str(e))
            raise ConfigError(f"Unexpected error: {e}") from e

    def _validate_schema(self, data: dict):
        """
        Validate the loaded YAML data against the exceptions schema.

        Args:
            data (Dict): Loaded YAML data.

        Raises:
            ConfigError: If the schema file is missing or the data does not comply with the schema.
        """
        if not self.schema_path.is_file():
            logger.error("schema_file_not_found", path=str(self.schema_path))
            raise ConfigError(f"Schema file not found at: {self.schema_path}")

        try:
            with self.schema_path.open("r") as schema_file:
                schema = json.load(schema_file)
            validate(instance=data, schema=schema)
            logger.info("schema_validation_passed")
        except ValidationError as e:
            logger.error("schema_validation_error", error=e.message)
            raise ConfigError(f"Validation error: {e.message}") from e
        except json.JSONDecodeError as e:
            logger.error("schema_json_decode_error", error=str(e))
            raise ConfigError(f"Error decoding schema JSON file: {e}") from e

    @staticmethod
    def _create_exceptions(data: dict[str, dict]) -> dict[str, type[GeneralError]]:
        """
        Dynamically create exception classes from the provided data.

        Args:
            data (Dict[str, Dict]): Exception configuration data.

        Returns:
            Dict[str, Type[GeneralError]]: A dictionary mapping exception names to classes.

        Raises:
            ConfigError: If any configuration is invalid.
        """
        exceptions = {}
        for category, category_exceptions in data.items():
            if not isinstance(category_exceptions, dict):
                logger.error("invalid_category_structure", category=category, value=str(category_exceptions))
                raise ConfigError(f"Invalid structure for category '{category}'")

            for name, config in category_exceptions.items():
                if not all(key in config for key in ["message", "code"]):
                    logger.error("invalid_exception_config", exception_name=name, config=str(config))
                    raise ConfigError(f"Invalid exception configuration for '{name}'")

                # Dynamically create exception classes
                exceptions[name] = type(
                    name,
                    (GeneralError,),
                    {
                        "DEFAULT_ERROR_CODE": config["code"],
                        "__init__": ExceptionLoader._generate_init(config["message"], config["code"]),
                    },
                )
        logger.info("exceptions_created_dynamically", count=len(exceptions))
        return exceptions

    @staticmethod
    def _generate_init(message: str, code: int):
        """
        Generate the __init__ method for dynamically created exceptions.

        Args:
            message (str): Default error message.
            code (int): Default error code.

        Returns:
            Callable: A dynamically generated __init__ method.
        """

        def __init__(self, details=None, **kwargs):
            logger.debug("initializing_exception", message=message, code=code)
            super(self.__class__, self).__init__(message=message, code=code, details=details, **kwargs)

        return __init__
