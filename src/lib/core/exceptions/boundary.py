"""Error boundary implementation for graceful error handling."""

from typing import Any

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class ErrorBoundary:
    """Error boundary to prevent cascading failures."""

    def __init__(self, name: str, fallback_value: Any = None):
        """
        Initialize error boundary.

        Args:
            name: Name of the context for error reporting
            fallback_value: Optional value to return in case of error
        """
        self.name = name
        self.fallback_value = fallback_value

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with error handling."""
        if exc_type is not None:
            logger.error("error_boundary_caught", boundary=self.name, error=str(exc_val))
            return True  # Suppress the exception
        return False

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit with error handling."""
        if exc_type is not None:
            logger.error("error_boundary_caught", boundary=self.name, error=str(exc_val))
            return True  # Suppress the exception
        return False
