#!/usr/bin/env python3
"""
Service Base Module for Microservices.

This module provides a base template for creating service entry points
that share common functionality while allowing for service-specific customization.
It integrates the lifecycle components (initialization, teardown, lifespan)
into a cohesive framework for building robust services.

Usage:
    1. Create a concrete service by inheriting from BaseService
    2. Implement the required abstract methods
    3. Use ServiceRunner to handle lifecycle management

Example:
    # For a simple API service:
    class ApiService(BaseService):
        def create_app(self, config):
            # Create and return your app instance
            return FastAPI()

        def run_app(self, app, host, port, **kwargs):
            # Start your app
            import uvicorn
            uvicorn.run(app, host=host, port=port, **kwargs)

    # In your main.py:
    if __name__ == "__main__":
        service = ApiService("api")
        ServiceRunner(service).run()
"""

from typing import TypeVar

# Re-export helper functions
try:
    from lib.core.factory import ServiceFactory, create_fastapi_service, create_flask_service, run_service
    from lib.core.fastapi import FastApiService
    from lib.core.runner import ServiceRunner

    # Re-export classes and functions from refactored modules
    from lib.core.service import BaseService
except ImportError:
    pass

# Re-export lifecycle components
from lib.core.initialization import initialize, load_configuration
from lib.core.lifespan import get_app_lifecycle

# Direct use of the comprehensive logging system
from lib.core.logging_config import get_logger, setup_logging
from lib.core.teardown import emergency_shutdown, teardown
from lib.utils.discover import find_config_file, try_import
from lib.utils.permissions import verify_permissions

# Re-export system utilities
from lib.utils.system import detect_container_environment, get_system_info

# Type variables for better typing
T = TypeVar("T")  # Generic type for service
AppType = TypeVar("AppType")  # Type for the application object

# Define defaults - re-exported for compatibility
DEFAULT_HOST = "0.0.0.0"
DEFAULT_DEBUG = True
DEFAULT_WORKERS = 1
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "/app/logs"
DEFAULT_CONFIG_PATHS = ["/app/config/fks/{service_name}.yaml", "{base_dir}/config/fks/{service_name}.yaml"]


class BaseComponent:
    """
    BaseComponent serves as a common ancestor for all components in the service framework.
    It can be used to define shared attributes or methods that are common across different components.
    """

    pass


# For backward compatibility, re-export the entire contents of the modules
__all__ = [
    # Classes
    "BaseService",
    "ServiceRunner",
    "ServiceFactory",
    "FastApiService",
    # Functions
    "setup_logging",
    "get_logger",
    "get_system_info",
    "verify_permissions",
    "try_import",
    "find_config_file",
    "detect_container_environment",
    "initialize",
    "load_configuration",
    "teardown",
    "emergency_shutdown",
    "get_app_lifecycle",
    "create_fastapi_service",
    "create_flask_service",
    "run_service",
    # Constants
    "DEFAULT_HOST",
    "DEFAULT_DEBUG",
    "DEFAULT_WORKERS",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_DIR",
    "DEFAULT_CONFIG_PATHS",
]
