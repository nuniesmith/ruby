"""
PostgreSQL database connection management.

This module provides functionality for creating, managing and closing
PostgreSQL database connections for the Ruby Futures System.
"""

import os
import threading
from contextlib import contextmanager
from typing import Any

# Import psycopg2 with proper error handling
try:
    import psycopg2
    from psycopg2 import pool

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Fallback to psycopg3 if psycopg2 is not available
if not HAS_PSYCOPG2:
    try:
        import psycopg
        from psycopg_pool import ConnectionPool

        HAS_PSYCOPG3 = True
    except ImportError:
        HAS_PSYCOPG3 = False
else:
    HAS_PSYCOPG3 = False

from lib.core.logging_config import get_logger

# Module logger
_logger = get_logger("lib.core.db.postgres")

# Connection pools by DSN
_pool_lock = threading.RLock()
_connection_pools: dict[str, Any] = {}

# Default connection parameters
DEFAULT_POOL_MIN_CONN = 1
DEFAULT_POOL_MAX_CONN = 10
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_QUERY_TIMEOUT = 30


class PostgresConnection:
    """
    Wrapper for a PostgreSQL database connection.

    This class provides a consistent interface for both psycopg2 and psycopg3
    connections, and handles connection pooling, timeouts, and error handling.
    """

    def __init__(
        self,
        dsn: str | None = None,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        use_pool: bool = True,
        min_connections: int = DEFAULT_POOL_MIN_CONN,
        max_connections: int = DEFAULT_POOL_MAX_CONN,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        application_name: str | None = None,
        cursor_factory=None,
    ):
        """
        Initialize a PostgreSQL connection.

        Args:
            dsn: Connection string (if provided, other connection params are ignored)
            host: Database host
            port: Database port
            dbname: Database name
            user: Database user
            password: Database password
            use_pool: Whether to use connection pooling
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
            connect_timeout: Connection timeout in seconds
            application_name: Client application name
            cursor_factory: Custom cursor factory
        """
        self.dsn: str | dict[str, int | str | None] | None = dsn
        self.use_pool = use_pool
        self.cursor_factory = cursor_factory
        self.conn = None
        self.pool_key = None

        # If no DSN provided, build connection parameters dict
        if not dsn:
            # Start with any environment variables
            conn_params = {
                "host": host or os.environ.get("POSTGRES_HOST", "localhost"),
                "port": port or os.environ.get("POSTGRES_PORT", 5432),
                "dbname": dbname or os.environ.get("POSTGRES_DB"),
                "user": user or os.environ.get("POSTGRES_USER"),
                "password": password or os.environ.get("POSTGRES_PASSWORD"),
                "connect_timeout": connect_timeout,
            }

            # Add application name if provided
            if application_name:
                conn_params["application_name"] = application_name

            # Filter out None values
            conn_params = {k: v for k, v in conn_params.items() if v is not None}

            # Convert to DSN string
            if HAS_PSYCOPG2:
                self.dsn = " ".join(f"{k}={v}" for k, v in conn_params.items())
            elif HAS_PSYCOPG3:
                # psycopg3 can work with dict or string
                self.dsn = conn_params

        # Generate a pool key
        if self.dsn is None:
            self.pool_key = "default"
        else:
            self.pool_key = self.dsn if isinstance(self.dsn, str) else str(sorted(self.dsn.items()))

        # Ensure pool_key is never None
        if self.pool_key is None:
            self.pool_key = "default"

        # Type assertion to satisfy type checker
        assert isinstance(self.pool_key, str)

        # Validate that we have either psycopg2 or psycopg3
        if not HAS_PSYCOPG2 and not HAS_PSYCOPG3:
            raise ImportError(
                "Neither psycopg2 nor psycopg3 is available. "
                "Please install either package: pip install psycopg2-binary or pip install psycopg"
            )

    def connect(self) -> bool:
        """
        Establish a connection to the PostgreSQL database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.use_pool:
                self.conn = self._get_from_pool()
            else:
                self.conn = self._create_direct_connection()

            # Register this connection for tracking
            from . import register_connection

            register_connection(self, "postgres")

            return True
        except Exception:
            _logger.error("postgres_connect_failed", exc_info=True)
            self.conn = None
            return False

    def _create_direct_connection(self) -> Any:
        """
        Create a direct connection (not from pool).

        Returns:
            Database connection
        """
        if HAS_PSYCOPG2:
            # Convert dict to string DSN if needed for psycopg2
            dsn_str = self.dsn
            if isinstance(self.dsn, dict):
                dsn_str = " ".join(f"{k}={v}" for k, v in self.dsn.items())

            # Ensure dsn_str is a string for psycopg2.connect
            if dsn_str is not None and not isinstance(dsn_str, str):
                dsn_str = str(dsn_str)

            if self.cursor_factory:
                return psycopg2.connect(dsn_str, cursor_factory=self.cursor_factory)  # type: ignore[possibly-undefined]
            else:
                return psycopg2.connect(dsn_str)  # type: ignore[possibly-undefined]
        elif HAS_PSYCOPG3:
            # psycopg3 doesn't use cursor_factory the same way
            if self.dsn is not None:
                if isinstance(self.dsn, dict):
                    return psycopg.connect(**self.dsn)  # type: ignore[arg-type, possibly-undefined]
                else:
                    return psycopg.connect(self.dsn)  # type: ignore[possibly-undefined]
            return None

    def _get_from_pool(self) -> Any:
        """
        Get a connection from the pool.

        Returns:
            Database connection from pool
        """
        with _pool_lock:
            # Check if pool exists for this DSN
            if self.pool_key not in _connection_pools:
                self._create_pool()

            # Get connection from pool
            assert self.pool_key is not None, "Pool key cannot be None"
            pool = _connection_pools[self.pool_key]["pool"]

            if HAS_PSYCOPG2:
                return pool.getconn()  # type: ignore[possibly-undefined]
            elif HAS_PSYCOPG3:
                return pool.connection()  # type: ignore[possibly-undefined]

    def _create_pool(self) -> None:
        """Create a new connection pool for this DSN."""
        with _pool_lock:
            if self.pool_key in _connection_pools:
                return  # Pool already exists

            min_conn = DEFAULT_POOL_MIN_CONN
            max_conn = DEFAULT_POOL_MAX_CONN

            if HAS_PSYCOPG2:
                # Create psycopg2 pool
                connection_pool = pool.ThreadedConnectionPool(min_conn, max_conn, self.dsn)  # type: ignore[possibly-undefined]
                assert self.pool_key is not None, "Pool key cannot be None"
                _connection_pools[self.pool_key] = {
                    "pool": connection_pool,
                    "min_conn": min_conn,
                    "max_conn": max_conn,
                    "type": "psycopg2",
                }
                _logger.debug("connection_pool_created", driver="psycopg2", pool_key=self.pool_key)
                # Create psycopg3 pool
                if isinstance(self.dsn, dict):
                    dsn_str = " ".join(f"{k}={v}" for k, v in self.dsn.items())
                elif self.dsn is None:
                    dsn_str = ""
                else:
                    dsn_str = str(self.dsn)
                connection_pool = ConnectionPool(dsn_str, min_size=min_conn, max_size=max_conn)  # type: ignore[possibly-undefined]
                assert self.pool_key is not None, "Pool key cannot be None"
                _connection_pools[self.pool_key] = {
                    "pool": connection_pool,
                    "min_conn": min_conn,
                    "max_conn": max_conn,
                    "type": "psycopg3",
                }
                _logger.debug("connection_pool_created", driver="psycopg3", pool_key=self.pool_key)

    def close(self) -> bool:
        """
        Close the database connection.

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            return True  # Already closed

        try:
            # Unregister this connection
            from . import unregister_connection

            unregister_connection(self, "postgres")

            if self.use_pool:
                self._return_to_pool()
            else:
                self._close_direct_connection()

            self.conn = None
            return True
        except Exception:
            _logger.error("postgres_close_failed", exc_info=True)
            return False

    def _close_direct_connection(self) -> None:
        """Close a direct connection (not from pool)."""
        if not self.conn:
            return

        try:
            self.conn.close()
        except Exception:
            _logger.error("direct_connection_close_failed", exc_info=True)

    def _return_to_pool(self) -> None:
        """Return connection to the pool."""
        if not self.conn or not self.pool_key:
            return

        with _pool_lock:
            if self.pool_key not in _connection_pools:
                # Pool doesn't exist, just close the connection
                self._close_direct_connection()
                return

            pool_info = _connection_pools[self.pool_key]

            try:
                if pool_info["type"] == "psycopg2":
                    pool_info["pool"].putconn(self.conn)
                # psycopg3 connections are automatically returned to pool when closed
                elif pool_info["type"] == "psycopg3":
                    self.conn.close()
            except Exception:
                _logger.error("pool_return_failed", exc_info=True)
                # Try to close it directly as a fallback
                self._close_direct_connection()


@contextmanager
def get_connection(
    dsn: str | None = None,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    use_pool: bool = True,
    cursor_factory=None,
) -> Any:
    """
    Context manager for getting a database connection.

    Usage:
        with get_connection(dsn="...") as conn:
            # use conn

    Args:
        dsn: Connection string
        host: Database host
        port: Database port
        dbname: Database name
        user: Database user
        password: Database password
        use_pool: Whether to use connection pooling
        cursor_factory: Custom cursor factory

    Yields:
        Database connection
    """
    conn = PostgresConnection(
        dsn=dsn,
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        use_pool=use_pool,
        cursor_factory=cursor_factory,
    )

    try:
        conn.connect()
        yield conn.conn
    finally:
        conn.close()


def close_postgres_connections(connections: list[Any] | None = None) -> bool:
    """
    Close PostgreSQL database connections.

    Args:
        connections: List of connections to close (if None, close all)

    Returns:
        True if all connections closed successfully, False otherwise
    """
    success = True

    # Close specific connections if provided
    if connections:
        for conn in connections:
            try:
                if isinstance(conn, PostgresConnection):
                    if not conn.close():
                        success = False
                else:
                    # Direct connection object
                    conn.close()
            except Exception:
                _logger.error("postgres_connection_close_failed", exc_info=True)
                success = False

    # Close all connection pools
    with _pool_lock:
        for dsn, pool_info in list(_connection_pools.items()):
            try:
                _logger.debug("closing_connection_pool", dsn=dsn)

                if pool_info["type"] == "psycopg2":
                    pool_info["pool"].closeall()
                elif pool_info["type"] == "psycopg3":
                    pool_info["pool"].close()

                del _connection_pools[dsn]
            except Exception:
                _logger.error("connection_pool_close_failed", dsn=dsn, exc_info=True)
                success = False

    return success
