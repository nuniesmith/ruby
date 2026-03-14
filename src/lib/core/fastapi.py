"""
FastAPI Service Implementation.

This module provides a concrete implementation of the BaseService
abstract class for FastAPI applications.
"""

import importlib
import os
import traceback
from typing import TYPE_CHECKING, Any

from lib.core.service import BaseService

if TYPE_CHECKING:
    import argparse


class FastApiService(BaseService):
    """
    Implementation of BaseService for FastAPI applications.
    """

    def __init__(self, service_name: str, factory_module: str = "", factory_func: str = "create_app"):
        """
        Initialize a FastAPI service.

        Args:
            service_name: The name of the service
            factory_module: Module that contains the app factory function
            factory_func: Name of the app factory function
        """
        super().__init__(service_name)
        self.factory_module = factory_module or f"{service_name}.factory"
        self.factory_func = factory_func

    def create_app(self, config: dict) -> Any:
        """
        Create and configure a FastAPI application.

        Args:
            config: Configuration dictionary

        Returns:
            FastAPI app instance or None on failure
        """
        try:
            # Import the app factory module
            module = importlib.import_module(self.factory_module)

            # Get the factory function
            factory = getattr(module, self.factory_func)

            # Create the app
            app = factory(config)

            # Set up lifespan context for FastAPI
            try:
                from lib.core.lifecycle.lifespan import setup_fastapi_lifespan  # type: ignore[import-not-found]

                setup_fastapi_lifespan(app)
            except ImportError:
                self.logger.debug("FastAPI lifespan setup not available or failed")

            return app
        except ImportError as e:
            self.logger.error(f"Could not import app factory module '{self.factory_module}': {e}")
            return self._create_fallback_fastapi(f"ImportError: {e}")
        except AttributeError as e:
            self.logger.error(f"Could not find factory function '{self.factory_func}' in '{self.factory_module}': {e}")
            return self._create_fallback_fastapi(f"AttributeError: {e}")
        except Exception as e:
            self.logger.error(f"Error creating FastAPI app from factory: {e}")
            self.logger.debug(traceback.format_exc())
            return self._create_fallback_fastapi(f"Exception: {e}")

    def _create_fallback_fastapi(self, error_message: str) -> Any | None:
        """
        Create a fallback FastAPI app if the main app creation fails.

        Args:
            error_message: Error message to display

        Returns:
            FastAPI app instance or None if FastAPI is not installed
        """
        # Optionally disable fallback in debug mode to see errors clearly
        if os.environ.get("DEBUG", "").lower() in ("true", "1", "t"):
            self.logger.warning("Debug mode enabled, not creating fallback app.")
            return None

        try:
            from fastapi import FastAPI  # type: ignore[attr-defined]

            app = FastAPI(title=f"{self.service_name.capitalize()} Service (Fallback)")

            @app.get("/")
            def read_root():
                return {
                    "service": f"{self.service_name.capitalize()} Service",
                    "status": "error",
                    "error": error_message,
                    "message": "Using fallback FastAPI application due to creation failure",
                }

            @app.get("/health")
            def health():
                return {"status": "warning", "message": "Fallback mode active"}

            self.logger.warning("Created fallback FastAPI application.")
            return app
        except ImportError:
            self.logger.error("Could not create fallback FastAPI app - FastAPI not installed")
            return None  # Return None if fallback cannot be created

    def run_app(self, app: Any, host: str, port: int, **kwargs) -> None:
        """
        Run the FastAPI application with uvicorn.

        Args:
            app: FastAPI application instance
            host: Host to bind to
            port: Port to listen on
            **kwargs: Additional arguments for uvicorn
        """
        try:
            import uvicorn

            # Try to use uvloop if available
            use_uvloop = False
            try:
                import asyncio

                import uvloop  # type: ignore[import-not-found]

                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                self.logger.info("Using uvloop for performance")
                use_uvloop = True
            except ImportError:
                self.logger.debug("uvloop not available, using default event loop")

            debug = kwargs.get("debug", False)
            workers = kwargs.get("workers", 1)

            # In debug mode, force 1 worker and enable reload
            if debug:
                workers = 1
                reload = True
                log_level = "debug"
            else:
                reload = False
                log_level = kwargs.get("log_level", "info").lower()  # Use level from args/defaults

            # Get app import string if needed (e.g., for reload or multiple workers)
            # If app is already an instance and not reloading/multi-worker, pass the instance directly.
            app_arg = app
            if reload or workers > 1:
                if not isinstance(app, str):
                    try:
                        module = app.__module__
                        name = getattr(app, "__name__", app.__class__.__name__)  # Handle app objects vs functions
                        app_arg = f"{module}:{name}"
                        self.logger.debug(f"Using app import string for uvicorn: {app_arg}")
                    except AttributeError:
                        self.logger.error(
                            "Could not determine app import string, uvicorn might fail with multiple workers or reload."
                        )
                        # Fallback to passing the app object, might not work as expected with workers/reload
                        app_arg = app
                else:
                    # app is already a string
                    app_arg = app

            # Additional uvicorn config
            uvicorn_config = {
                "host": host,
                "port": port,
                "reload": reload,
                "log_level": log_level,
                "workers": workers,
                "timeout_keep_alive": kwargs.get("keep_alive", 65),  # Keep-alive timeout
                "limit_concurrency": kwargs.get("max_concurrency", 1000),  # Max concurrent connections
                "backlog": kwargs.get("backlog", 2048),  # Connection queue size
                "loop": "uvloop" if use_uvloop else "auto",
                "lifespan": "on",  # Explicitly enable lifespan events
            }

            # If access log is explicitly disabled via args
            if kwargs.get("disable_access_log", False) or not debug:
                uvicorn_config["access_log"] = False

            # Run the uvicorn server
            self.logger.info(f"Starting uvicorn server with config: {uvicorn_config}")
            uvicorn.run(app_arg, **uvicorn_config)

        except ImportError as e:
            self.logger.error(f"Failed to start uvicorn server: {e}. Is uvicorn installed?")
            raise
        except Exception as e:
            self.logger.error(f"Error running FastAPI app with uvicorn: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def add_arguments(self, parser: "argparse.ArgumentParser") -> None:
        """
        Add FastAPI-specific command line arguments.

        Args:
            parser: ArgumentParser instance to add arguments to
        """
        pass

    def get_required_components(self) -> list[str]:
        """
        Get the components required by a FastAPI service.

        Returns:
            List[str]: Required component names
        """
        # Example components, adjust as needed
        return ["database", "cache", "event_bus"]
