"""
Database connection management for the Ruby Futures System.

This module provides centralized management for database connections
across the application, with support for multiple database types.
"""

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from lib.core.logging_config import get_logger

# Import public components from submodules
from .base import Database
from .orm import init_engine, shutdown_engine
from .postgres import PostgresConnection, close_postgres_connections
from .repository import BaseRepository

# Export public components
__all__ = [
    # Core components
    "Database",
    "PostgresConnection",
    "BaseRepository",
    # Connection management
    "register_connection",
    "unregister_connection",
    "get_connection_count",
    "close_all_connections",
    # ORM functions
    "init_engine",
    "shutdown_engine",
    # Async utilities
    "run_async",
    "run_in_thread",
]

# Active connection tracking
_connection_lock = threading.RLock()
_active_connections: dict[str, list[Any]] = {"postgres": [], "other": []}

# Module logger
_logger = get_logger("lib.core.db")


def register_connection(connection: Any, db_type: str = "other") -> None:
    """
    Register a database connection for tracking and cleanup.

    Args:
        connection: The database connection object
        db_type: Type of database ('postgres', 'other')
    """
    with _connection_lock:
        if db_type not in _active_connections:
            _active_connections[db_type] = []
        _active_connections[db_type].append(connection)

    _logger.debug("connection_registered", db_type=db_type)


def unregister_connection(connection: Any, db_type: str = "other") -> None:
    """
    Unregister a database connection that's no longer active.

    Args:
        connection: The database connection object
        db_type: Type of database ('postgres', 'other')
    """
    with _connection_lock:
        if db_type in _active_connections and connection in _active_connections[db_type]:
            _active_connections[db_type].remove(connection)

    _logger.debug("connection_unregistered", db_type=db_type)


def get_connection_count() -> dict[str, int]:
    """
    Get count of active connections by database type.

    Returns:
        Dictionary of connection counts by type
    """
    with _connection_lock:
        return {db_type: len(connections) for db_type, connections in _active_connections.items()}


def close_all_connections() -> bool:
    """
    Close all active database connections.

    Returns:
        True if all connections closed successfully, False otherwise
    """
    _logger.info("closing_all_connections")
    success = True

    # Close PostgreSQL connections
    if "close_postgres_connections" in globals() and close_postgres_connections is not None:
        try:
            with _connection_lock:
                postgres_connections = _active_connections.get("postgres", []).copy()

            if postgres_connections:
                postgres_success = close_postgres_connections(postgres_connections)
                if not postgres_success:
                    _logger.warning("postgres_connections_close_partial_failure")
                    success = False
            else:
                _logger.debug("no_postgres_connections_to_close")
        except Exception:
            _logger.error("postgres_connections_close_failed", exc_info=True)
            success = False

    # Close any remaining connections
    with _connection_lock:
        for db_type, connections in _active_connections.items():
            if (
                db_type == "postgres"
                and "close_postgres_connections" in globals()
                and close_postgres_connections is not None
            ):
                # Already handled above
                continue

            if not connections:
                continue

            _logger.info("closing_connections", db_type=db_type, count=len(connections))
            for connection in connections.copy():
                try:
                    if hasattr(connection, "close"):
                        connection.close()
                    elif hasattr(connection, "disconnect"):
                        connection.disconnect()
                    elif hasattr(connection, "shutdown"):
                        connection.shutdown()
                    connections.remove(connection)
                    _logger.debug("connection_closed", db_type=db_type)
                except Exception:
                    _logger.error("connection_close_failed", db_type=db_type, exc_info=True)
                    success = False

    # Reset active connections tracking
    with _connection_lock:
        for db_type in _active_connections:
            _active_connections[db_type] = []

    return success


# Async utilities for database operations
def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in a thread pool.

    This is useful for running blocking database operations
    without blocking the event loop.

    Args:
        func: The synchronous function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.submit(func, *args, **kwargs).result()


async def run_async(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in a thread pool asynchronously.

    This is useful for running blocking database operations
    without blocking the event loop.

    Args:
        func: The synchronous function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# Database instance cache for efficient reuse
_db_instances: dict[str, Database] = {}


def get_db(name: str = "default") -> Database:
    """
    Get a shared database instance by name.

    This allows different parts of the application to
    access the same database connection.

    Args:
        name: Name of the database instance

    Returns:
        Database: The database instance
    """
    if name not in _db_instances:
        _db_instances[name] = Database()
    return _db_instances[name]


# Add default database instance for convenience
default_db = get_db("default")
