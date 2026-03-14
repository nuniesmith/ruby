import yaml  # type: ignore[import-untyped]

from lib.core.exceptions.base import FrameworkException
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ExceptionsUtils:
    """
    Utility class to handle dynamic exception loading, validation, and raising.
    """

    def __init__(self, app_manager, env: str = "development"):
        """
        Initialize the ExceptionsUtils with an application manager and environment.

        Args:
            app_manager: The AppManager instance providing access to configurations.
            env (str, optional): Current environment (e.g., "development", "production").
        """
        self.app_manager = app_manager
        self.env = env

        # Get the exceptions configuration path from the AppManager
        self.config_path = app_manager.config_manager.get_config_path("exceptions.yaml")

        # Load and validate configuration
        self._config = self._load_config()
        self.validate_config()

    def _load_config(self):
        """
        Load the exceptions configuration from a YAML file.

        Returns:
            dict: The loaded configuration dictionary.
        """
        try:
            if not self.config_path.is_file():
                raise FileNotFoundError(f"Exception config file not found: {self.config_path}")
            logger.info("loading_exceptions_config", config_path=str(self.config_path))
            with open(self.config_path) as file:
                config = yaml.safe_load(file)
            exceptions_config = config.get("exceptions", {})
            if not exceptions_config:
                logger.warning("no_exceptions_configured", action="using_default_fallback")
                exceptions_config = self._default_fallback()
            return exceptions_config
        except FileNotFoundError:
            logger.error(
                "exception_config_not_found", config_path=str(self.config_path), action="using_default_fallback"
            )
            return self._default_fallback()
        except yaml.YAMLError as e:
            logger.error("exception_config_parse_error", error=str(e), action="using_default_fallback")
            return self._default_fallback()

    @staticmethod
    def _default_fallback():
        """
        Provide a default fallback configuration.

        Returns:
            dict: Default exception configuration.
        """
        return {
            "default": {
                "DefaultError": {
                    "message": "An unspecified error occurred.",
                    "code": 9999,
                }
            }
        }

    def validate_config(self):
        """
        Validate that every exception entry has required keys like `message` and `code`.

        Logs a warning instead of raising an exception for better fault tolerance.
        """
        required_keys = {"message", "code"}
        for category, errors in self._config.items():
            for error_name, details in errors.items():
                missing_keys = required_keys - details.keys()
                if missing_keys:
                    logger.warning(
                        "exception_config_missing_keys",
                        error_name=error_name,
                        category=category,
                        missing_keys=", ".join(missing_keys),
                    )
        logger.info("exceptions_config_validation_completed")

    def reload(self):
        """
        Reload the exceptions configuration (e.g., if exceptions.yaml changes at runtime).
        """
        logger.info("reloading_exceptions_config")
        self._config = self._load_config()
        self.validate_config()
        logger.info("exceptions_config_reloaded")

    def get_exception(self, category: str, error_name: str) -> dict:
        """
        Retrieve exception details from the loaded configuration.

        Args:
            category (str): Exception category.
            error_name (str): Specific error name within the category.

        Returns:
            dict: The exception details (message, code, etc.).
        """
        category_config = self._config.get(category)
        if not category_config:
            logger.warning("exception_category_not_found", category=category, action="using_fallback")
            return self._fallback_error()

        exception_details = category_config.get(error_name)
        if not exception_details:
            logger.warning("exception_not_found", error_name=error_name, category=category, action="using_fallback")
            return self._fallback_error()

        message_key = f"message_{self.env}" if f"message_{self.env}" in exception_details else "message"
        return {
            "message": exception_details.get(message_key, exception_details.get("message")),
            "code": exception_details.get("code"),
            "http_status": exception_details.get("http_status"),
            "retryable": exception_details.get("retryable", False),
        }

    def _fallback_error(self) -> dict:
        """
        Return the default fallback error if a specified exception is not found.

        Returns:
            dict: The default error details.
        """
        return self._config.get("default", {}).get("DefaultError", self._default_fallback()["default"]["DefaultError"])

    def raise_exception(self, category: str, error_name: str, **kwargs):
        """
        Raise a CustomException based on the specified category and error_name.

        Args:
            category (str): The exception category.
            error_name (str): Specific error name within the category.
            **kwargs: Additional arguments to format the exception message.

        Raises:
            CustomException: Raised exception with the retrieved details.
        """
        exc = self.get_exception(category, error_name)
        message = exc["message"].format(**kwargs) if kwargs else exc["message"]
        raise FrameworkException(
            message=message,
            code=exc["code"],
            details={
                "http_status": exc.get("http_status"),
                "retryable": exc.get("retryable", False),
            },
        )
