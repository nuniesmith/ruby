"""
Health Check Utilities

This module provides functions for adding health check endpoints to services.
The health check endpoints can be used by Docker and other monitoring tools.
"""

import socket
import time
from typing import Any

import psutil


def system_health_check() -> dict[str, Any]:
    """
    Perform basic system health check.

    Returns:
        Dict with system health metrics
    """
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "uptime": time.time() - psutil.boot_time(),
        "hostname": socket.gethostname(),
    }


def add_fastapi_health_endpoints(app, prefix: str = ""):
    """
    Add health check endpoints to a FastAPI application.

    Args:
        app: The FastAPI application
        prefix: Optional URL prefix for the health endpoints
    """

    @app.get(f"{prefix}/health")
    async def health():
        """Basic health check endpoint"""
        return {"status": "ok"}

    @app.get(f"{prefix}/health/system")
    async def system_health():
        """System health check endpoint"""
        return system_health_check()

    @app.get(f"{prefix}/ready")
    async def ready():
        """Readiness check endpoint"""
        return {"status": "ready"}

    @app.get(f"{prefix}/live")
    async def live():
        """Liveness check endpoint"""
        return {"status": "alive"}


def add_flask_health_endpoints(app, prefix: str = ""):
    """
    Add health check endpoints to a Flask application.

    Args:
        app: The Flask application
        prefix: Optional URL prefix for the health endpoints
    """

    @app.route(f"{prefix}/health")
    def health():
        """Basic health check endpoint"""
        return {"status": "ok"}

    @app.route(f"{prefix}/health/system")
    def system_health():
        """System health check endpoint"""
        return system_health_check()

    @app.route(f"{prefix}/ready")
    def ready():
        """Readiness check endpoint"""
        return {"status": "ready"}

    @app.route(f"{prefix}/live")
    def live():
        """Liveness check endpoint"""
        return {"status": "alive"}


def add_health_endpoints(app, framework: str = "auto", prefix: str = ""):
    """
    Add health check endpoints to an application.

    Args:
        app: The application
        framework: The framework type ('fastapi', 'flask', or 'auto')
        prefix: Optional URL prefix for the health endpoints

    Returns:
        The modified application
    """
    if framework == "auto":
        # Try to detect the framework type
        if hasattr(app, "add_api_route"):
            framework = "fastapi"
        elif hasattr(app, "route"):
            framework = "flask"
        else:
            raise ValueError("Could not auto-detect framework type. Please specify 'fastapi' or 'flask'.")

    if framework.lower() == "fastapi":
        add_fastapi_health_endpoints(app, prefix)
    elif framework.lower() == "flask":
        add_flask_health_endpoints(app, prefix)
    else:
        raise ValueError(f"Unsupported framework type: {framework}")

    return app
