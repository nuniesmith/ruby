"""
Base Service Module.

This module defines the abstract base class that all services must implement.
It provides common functionality and defines the interface that specific
service implementations must follow.
"""

import abc
import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from lib.core.config import load_configuration
from lib.core.initialization import initialize
from lib.core.lifespan import ApplicationLifecycle, get_app_lifecycle
from lib.core.logging_config import get_logger, setup_logging
from lib.core.teardown import emergency_shutdown, teardown
from lib.utils.discover import find_config_file
from lib.utils.system import detect_container_environment

# Define defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_DEBUG = False
DEFAULT_WORKERS = 1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "/app/logs"


class BaseService(abc.ABC):
    """
    Abstract base class for services.

    This class defines the interface that all services must implement
    and provides common functionality.
    """

    def __init__(self, service_name: str):
        """
        Initialize a new service.

        Args:
            service_name: The name of the service, used for configuration and logging
        """
        self.service_name = service_name
        # Configure structured logging for this service, then get a bound logger
        setup_logging(service=service_name)
        self.logger: Any = get_logger(service_name)

        # Set up default port based on service name
        self.default_port = int(os.environ.get(f"{service_name.upper()}_PORT", 8000))

        # Environment and container detection
        self.container_info = detect_container_environment()
        if self.container_info["is_docker"]:
            self.logger.info("Running in a container environment")

        # Get lifecycle
        self.app_lifecycle: ApplicationLifecycle | Any
        try:
            self.app_lifecycle = get_app_lifecycle()
        except Exception as e:
            self.logger.warning(f"Error loading app lifecycle: {e}")
            self.app_lifecycle = None

    def setup_environment(self) -> Path:
        """
        Set up environment variables and paths.

        Returns:
            Path: The base directory of the application
        """
        # Set up base paths using the enhanced detection methods
        base_dir = self._detect_base_dir()

        os.environ.setdefault(f"{self.service_name.upper()}_BASE_DIR", str(base_dir))

        # Make sure the src directory is in the Python path
        src_dir = str(base_dir / "src") if (base_dir / "src").exists() else str(base_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Log diagnostics info
        self._log_environment_diagnostics(base_dir)

        return base_dir

    def _detect_base_dir(self) -> Path:
        """
        Detect the base directory with enhanced container support.

        Returns:
            Path: Detected base directory
        """
        # Try environment variable first (highest priority)
        base_dir_env = os.environ.get(f"{self.service_name.upper()}_BASE_DIR", os.environ.get("FKS_BASE_DIR"))
        if base_dir_env:
            base_dir = Path(base_dir_env)
            self.logger.debug(f"Using base directory from environment: {base_dir}")
            return base_dir

        # Use base_dir from the file if this is being run as a script
        try:
            import inspect

            calling_frame = inspect.stack()[1]
            if calling_frame.filename != "<stdin>" and Path(calling_frame.filename).exists():
                # Go up one level if the calling file is likely 'main.py' or similar inside a service dir
                potential_base = Path(calling_frame.filename).parent
                # Heuristic: if parent contains 'src', go up one more level
                if "src" in potential_base.parts:
                    potential_base = potential_base.parent
                self.logger.debug(f"Using base directory from calling file heuristic: {potential_base}")
                return potential_base
        except (IndexError, AttributeError):
            self.logger.debug("Could not determine calling frame filename.")

        # Check common Docker paths
        if self.container_info["is_docker"]:
            docker_paths = [Path("/app"), Path("/usr/src/app"), Path("/home/app"), Path("/opt/app")]
            for docker_path in docker_paths:
                if docker_path.exists():
                    self.logger.debug(f"Using Docker path as base directory: {docker_path}")
                    return docker_path

        # Otherwise use the current working directory
        base_dir = Path.cwd()
        self.logger.debug(f"Using current directory as base directory: {base_dir}")
        return base_dir

    def _log_environment_diagnostics(self, base_dir: Path) -> None:
        """
        Log diagnostics information about the environment.

        Args:
            base_dir: Base directory
        """
        # Check if DEBUG level is enabled, compatible with both loguru and standard logging
        is_debug_enabled = False
        try:
            import logging

            # Detect if logger is loguru by checking for 'level' attribute or module name
            if self.logger.__class__.__module__.startswith("loguru"):
                log_level_str = os.environ.get(f"{self.service_name.upper()}_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
                is_debug_enabled = log_level_str == "DEBUG"
            else:
                is_debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        except Exception as e:
            # If there's any issue checking the log level, assume debug is not enabled
            print(f"Warning: Could not check debug log level: {e}", file=sys.stderr)
            is_debug_enabled = False

        if not is_debug_enabled:
            return

        self.logger.debug(f"Environment setup for {self.service_name} service:")
        self.logger.debug(f"  Base directory: {base_dir}")

        # Log Python paths
        self.logger.debug("Python path:")
        for path in sys.path:
            self.logger.debug(f"  - {path}")

        # Log relevant environment variables
        env_prefix = self.service_name.upper()
        self.logger.debug("Environment variables:")
        for key, value in sorted(os.environ.items()):
            if key.startswith(env_prefix) or key.startswith("FKS_"):
                self.logger.debug(f"  {key}={value}")

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command line arguments.

        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(description=f"{self.service_name.capitalize()} Service")

        # Create argument groups for better organization
        common_group = parser.add_argument_group("Common Options")
        server_group = parser.add_argument_group("Server Options")
        log_group = parser.add_argument_group("Logging Options")
        config_group = parser.add_argument_group("Configuration Options")

        # Server options
        server_group.add_argument(
            "--port",
            type=int,
            default=self.default_port,
            help=f"Port to run the service on (default: {self.default_port})",
        )
        server_group.add_argument(
            "--host",
            type=str,
            default=os.environ.get(f"{self.service_name.upper()}_HOST", DEFAULT_HOST),
            help=f"Host to bind the service to (default: {DEFAULT_HOST})",
        )
        server_group.add_argument(
            "--workers",
            type=int,
            default=int(os.environ.get(f"{self.service_name.upper()}_WORKERS", DEFAULT_WORKERS)),
            help=f"Number of worker processes (default: {DEFAULT_WORKERS})",
        )

        # Logging options
        log_group.add_argument(
            "--log-level",
            type=str,
            default=os.environ.get(f"{self.service_name.upper()}_LOG_LEVEL", DEFAULT_LOG_LEVEL),
            help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
        )
        log_group.add_argument(
            "--log-dir",
            type=str,
            default=os.environ.get(f"{self.service_name.upper()}_LOG_DIR", DEFAULT_LOG_DIR),  # Updated default
            help="Directory for log files",
        )

        # Common options
        common_group.add_argument(
            "--debug",
            action="store_true",
            default=os.environ.get("DEBUG", "").lower() in ("true", "1", "t"),
            help="Run in debug mode with more verbose logging",
        )
        common_group.add_argument("--diagnose", action="store_true", help="Run diagnostics and exit")

        # Configuration options
        config_group.add_argument(
            "--config",
            type=str,
            default=os.environ.get(f"{self.service_name.upper()}_CONFIG"),
            help="Path to configuration file",
        )

        # Allow subclasses to add their own arguments
        self.add_arguments(parser)

        return parser.parse_args()

    @abc.abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add service-specific command line arguments.

        Args:
            parser: ArgumentParser instance to add arguments to
        """
        # No-op in base class, override in subclasses to add custom arguments
        pass

    def load_settings(self) -> Any | None:
        """
        Load settings from a settings module if available.

        Returns:
            Settings object if available, None otherwise
        """
        try:
            # Try to load from core configuration
            settings_module = importlib.import_module("core.config.settings")
            if hasattr(settings_module, "get_settings"):
                return settings_module.get_settings()

            # Try service-specific settings
            service_settings = importlib.import_module(f"{self.service_name}.config.settings")
            if hasattr(service_settings, "get_settings"):
                return service_settings.get_settings()

        except Exception as e:
            self.logger.debug(f"Error loading settings: {e}")

        return None

    def initialize_components(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Initialize the service components.

        This method uses the initialization module to set up required components.

        Args:
            config: Configuration dictionary

        Returns:
            Dict[str, Any]: Context with initialized components
        """
        # Determine which components to initialize
        components = self.get_required_components()

        try:
            # Log components being initialized
            self.logger.debug(f"Initializing components: {components}")

            # Call the initialization function
            success, context = initialize(components, additional_config=config)

            if not success:
                self.logger.warning("Some components failed to initialize")

                # Log which components failed
                if isinstance(context, dict) and "failures" in context:
                    for component, error in context["failures"].items():
                        self.logger.error(f"Failed to initialize {component}: {error}")

            return context
        except Exception as e:
            self.logger.error(f"Error during component initialization: {e}")
            self.logger.debug(traceback.format_exc())
            return {"error": str(e)}

    def get_required_components(self) -> list[str]:
        """
        Get the list of components required by this service.

        Returns:
            List[str]: List of component names
        """
        # Default implementation returns an empty list
        # Override in subclasses to specify required components
        return []

    def preprocess_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess the configuration before using it.

        This is a hook for subclasses to modify the configuration.

        Args:
            config: Configuration dictionary

        Returns:
            dict: Modified configuration dictionary
        """
        # Add service info section if not present
        if "service" not in config:
            config["service"] = {}

        # Add base service information
        config["service"].update(
            {
                "name": self.service_name,
                "container": self.container_info["is_docker"],
                "debug": os.environ.get("DEBUG", "").lower() in ("true", "1", "t"),
                "environment": os.environ.get("FKS_ENV", os.environ.get("APP_ENVIRONMENT", "development")),
                "version": os.environ.get(
                    f"{self.service_name.upper()}_VERSION", os.environ.get("APP_VERSION", "0.1.0")
                ),
            }
        )

        return config

    def run_diagnostics(self) -> dict[str, Any]:
        """
        Run diagnostics on the service.

        Returns:
            Dict[str, Any]: Diagnostics results
        """
        from lib.utils.discover import try_import
        from lib.utils.permissions import verify_permissions
        from lib.utils.system import get_system_info

        self.logger.info("Running service diagnostics")
        results: dict[str, Any] = {
            "service": self.service_name,
            "system": get_system_info(),
            "container": self.container_info,
            "paths": {},
            "config": {},
            "components": {},
        }

        # Check base directory
        base_dir = self._detect_base_dir()
        results["paths"]["base_dir"] = verify_permissions(base_dir)

        # Check config
        config_path = find_config_file(self.service_name, base_dir)
        results["config"]["path"] = config_path
        results["config"]["exists"] = Path(config_path).exists()
        if Path(config_path).exists():
            results["config"]["permissions"] = verify_permissions(config_path)

            # Try to load config
            success, config = load_configuration(config_path)
            results["config"]["loaded"] = success
            if success:
                results["config"]["sections"] = list(config.keys())

        # Check components
        for component in self.get_required_components():
            try:
                # Basic check if the component's module is importable
                module_name = f"core.components.{component}"
                module = try_import(module_name)
                results["components"][component] = {"importable": module is not None}
            except Exception as e:
                results["components"][component] = {"importable": False, "error": str(e)}

        self.logger.info(f"Diagnostics complete for {self.service_name}")
        return results

    @abc.abstractmethod
    def create_app(self, config: dict[str, Any]) -> object:
        """
        Create and configure the application.

        Args:
            config: Configuration dictionary

        Returns:
            Application instance
        """
        pass

    @abc.abstractmethod
    def run_app(self, app: object, host: str, port: int, **kwargs) -> None:
        """
        Run the application.

        Args:
            app: Application instance
            host: Host to bind to
            port: Port to listen on
            **kwargs: Additional arguments for the runner
        """
        pass

    def cleanup(self) -> None:
        """
        Perform cleanup operations before shutting down.

        This method uses the teardown module to properly clean up resources.
        """
        self.logger.info(f"Cleaning up {self.service_name} service")
        try:
            teardown()
        except Exception as e:
            self.logger.error(f"Error during teardown: {e}")
            self.logger.debug(traceback.format_exc())

            # Try to perform emergency shutdown
            try:
                emergency_shutdown()
            except Exception as e2:
                self.logger.error(f"Error during emergency shutdown: {e2}")
