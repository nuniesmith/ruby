import os
from pathlib import Path


class ExceptionsErrorCodes:
    """
    Configuration class for managing error codes.

    This class handles loading, validating, and providing access to error codes from
    environment-specific configuration files.
    """

    def __init__(self, base_path: Path, environment: str):
        """
        Initialize the ErrorCodesConfig.

        Args:
            base_path (Path): Base directory for configuration files.
            environment (str): Application environment (e.g., development, production).
        """
        self.base_path = base_path
        self.environment = environment
        self.error_codes: dict[int, str] = {}

    def resolve_placeholders(self):
        """
        Resolve environment variable placeholders in error descriptions.
        """

        def resolve(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                parts = value[2:-1].split(":")
                env_var = parts[0]
                default = parts[1] if len(parts) > 1 else None
                return os.getenv(env_var, default) or ""
            return value

        def recursive_resolve(codes: dict[int, str]):
            for code, description in codes.items():
                codes[code] = resolve(description)

        recursive_resolve(self.error_codes)

    def get_error_description(self, code: int, default: str = "Unknown Error Code") -> str:
        """
        Retrieve the description for a specific error code.

        Args:
            code (int): The error code to retrieve.
            default (str): Default description if the code is not found.

        Returns:
            str: Error description.
        """
        return self.error_codes.get(code, default)

    def is_valid_code(self, code: int) -> bool:
        """
        Check if an error code exists.

        Args:
            code (int): The error code to check.

        Returns:
            bool: True if the code exists, False otherwise.
        """
        return code in self.error_codes
