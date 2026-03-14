"""
Service Factory Module.

This module provides factory functions for creating and running different
types of services without directly importing their implementations.
"""

import sys
from typing import Any, TypeVar

# Use string imports to avoid circular imports
ServiceType = TypeVar("ServiceType")
RunnerType = TypeVar("RunnerType")


class ServiceFactory:
    """Factory for creating service instances."""

    @staticmethod
    def create(service_type: str, service_name: str, **kwargs) -> Any:
        """
        Create a service instance.

        Args:
            service_type: Type of service ('fastapi', 'flask', etc.)
            service_name: Name of the service
            **kwargs: Additional arguments for the service constructor

        Returns:
            BaseService: Service instance
        """
        # Import implementations here to avoid circular dependencies
        # This is done dynamically to maintain separation of concerns
        if service_type.lower() == "fastapi":
            try:
                from lib.core.fastapi import FastApiService
            except ImportError as e:
                raise ValueError("FastAPI service implementation not available") from e
            return FastApiService(service_name, **kwargs)
        else:
            raise ValueError(f"Unknown service type: {service_type}")


def create_fastapi_service(
    service_name: str, factory_module: str = "", factory_func: str = "create_app"
) -> tuple[Any, Any]:
    """
    Create a FastAPI service with runner.

    Args:
        service_name: Name of the service
        factory_module: Module containing app factory
        factory_func: Factory function name

    Returns:
        Tuple of (FastApiService, ServiceRunner)
    """
    try:
        from lib.core.fastapi import FastApiService
        from lib.core.runner import ServiceRunner
    except ImportError as e:
        raise ImportError(f"Required service modules not available: {e}") from e

    service = FastApiService(service_name, factory_module, factory_func)
    runner = ServiceRunner(service)
    return service, runner


def create_flask_service(
    service_name: str, factory_module: str = "", factory_func: str = "create_app"
) -> tuple[Any, Any]:
    """
    Create a Flask service with runner.

    Args:
        service_name: Name of the service
        factory_module: Module containing app factory
        factory_func: Factory function name

    Returns:
        Tuple of (FlaskService, ServiceRunner)
    """
    raise NotImplementedError("Flask service support is not yet implemented")


def run_service(service_type: str, service_name: str, **kwargs) -> None:
    """
    Create and run a service of the specified type.

    Args:
        service_type: Type of service ('fastapi' or 'flask')
        service_name: Name of the service
        **kwargs: Additional service creation parameters (passed to Service constructor)
    """
    try:
        service = ServiceFactory.create(service_type, service_name, **kwargs)
        try:
            from lib.core.runner import ServiceRunner
        except ImportError as e:
            print(f"!!! ERROR: ServiceRunner not available: {e}", file=sys.stderr)
            sys.exit(1)

        runner = ServiceRunner(service)
        runner.run()
    except ValueError as e:
        # Catch factory errors specifically
        print(
            f"!!! ERROR: Could not create service '{service_name}' of type '{service_type}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)
