"""
Database connection management.

Provides centralized database connection handling for the application.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import (
    Any,
    Self,  # For Python <3.11 compatibility
)

from pydantic import BaseModel, Field, field_validator

from lib.core.db import register_connection, unregister_connection
from lib.core.db.orm import init_engine, shutdown_engine
from lib.core.db.postgres import PostgresConnection
from lib.core.exceptions.data import DatabaseError
from lib.core.logging_config import get_logger

# Configure logger
logger = get_logger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration model."""

    dsn: str | None = None
    host: str | None = None
    port: int | None = Field(default=5432)
    dbname: str | None = None
    user: str | None = None
    password: str | None = None
    use_pool: bool = Field(default=True)
    use_orm: bool = Field(default=False)
    pool_min_size: int = Field(default=1)
    pool_max_size: int = Field(default=10)
    connection_timeout: float = Field(default=60.0)

    @field_validator("host", "dbname", "user", "password")
    @classmethod
    def check_connection_params(cls, v: str | None, info) -> str | None:
        """Validate that either DSN or individual parameters are provided."""
        # Skip validation if value is provided
        if v is not None:
            return v

        # Get the current values being validated
        field_name = info.field_name
        values = info.data

        # If we have a DSN, other fields can be None
        if values.get("dsn") is not None:
            return v

        # Otherwise, these fields are required
        if field_name in ["host", "dbname", "user", "password"] and v is None:
            raise ValueError(f"Either 'dsn' or '{field_name}' must be provided")

        return v

    def to_connection_params(self) -> dict[str, Any]:
        """Convert config to connection parameters dict."""
        return {
            "dsn": self.dsn,
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "use_pool": self.use_pool,
            "pool_min_size": self.pool_min_size,
            "pool_max_size": self.pool_max_size,
            "connection_timeout": self.connection_timeout,
        }


class Database:
    """
    Database connection manager.

    Provides unified interface for database connections, supporting both
    direct PostgreSQL connections and SQLAlchemy ORM.
    """

    def __init__(self):
        """Initialize the database manager."""
        self._pg_connection: PostgresConnection | None = None
        self._engine_initialized: bool = False
        self._config: DatabaseConfig | None = None
        self._is_connected: bool = False
        self._connecting: bool = False
        self._connection_lock = asyncio.Lock()

    def configure(self, config: DatabaseConfig | dict[str, Any]) -> Self:
        """
        Configure database connection parameters.

        Args:
            config: Either a DatabaseConfig object or a dictionary of parameters

        Returns:
            Self: For method chaining
        """
        if isinstance(config, dict):
            self._config = DatabaseConfig.model_validate(config)
        else:
            self._config = config

        return self

    async def connect(self) -> bool:
        """
        Connect to the database. Thread-safe and prevents multiple concurrent connections.

        Returns:
            True if successful, False otherwise

        Raises:
            DatabaseError: If configuration is missing
        """
        if self._is_connected:
            return True

        if not self._config:
            raise DatabaseError("Database not configured. Call configure() before connect().")

        # Prevent multiple concurrent connection attempts
        async with self._connection_lock:
            if self._is_connected:  # Double-check after acquiring lock
                return True

            if self._connecting:
                logger.warning("connection_attempt_already_in_progress")
                return False

            self._connecting = True

            try:
                # Run the synchronous PostgreSQL connection code in a thread pool
                loop = asyncio.get_running_loop()

                # Create PostgreSQL connection
                self._pg_connection = PostgresConnection(**self._config.to_connection_params())
                connected = await loop.run_in_executor(None, self._pg_connection.connect)

                if connected:
                    # Register the connection with the tracking system
                    register_connection(self._pg_connection, db_type="postgres")

                # Initialize ORM engine if requested
                if self._config.use_orm and connected:
                    # Build SQLAlchemy connection string if not provided
                    if self._config.dsn:
                        db_url = self._config.dsn
                    else:
                        db_url = (
                            f"postgresql://{self._config.user}:{self._config.password}"
                            f"@{self._config.host}:{self._config.port}/{self._config.dbname}"
                        )

                    # Initialize ORM engine
                    engine_result = await loop.run_in_executor(None, lambda: init_engine(db_url))
                    self._engine_initialized = engine_result is not None

                self._is_connected = connected
                return connected

            except Exception:
                logger.error("database_connection_error", exc_info=True)
                return False
            finally:
                self._connecting = False

    async def disconnect(self) -> bool:
        """
        Disconnect from the database.

        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected:
            return True  # Already disconnected

        async with self._connection_lock:
            if not self._is_connected:  # Double-check after acquiring lock
                return True

            loop = asyncio.get_running_loop()
            success = True

            try:
                # Shutdown ORM engine if it was initialized
                if self._engine_initialized:
                    orm_result = await loop.run_in_executor(None, lambda: shutdown_engine())
                    if not orm_result:
                        success = False

                # Close PostgreSQL connection
                if self._pg_connection:
                    # Unregister connection before closing
                    unregister_connection(self._pg_connection, db_type="postgres")

                    pg_result = await loop.run_in_executor(None, self._pg_connection.close)
                    if not pg_result:
                        success = False
                    self._pg_connection = None

                self._is_connected = False
                return success

            except Exception:
                logger.error("database_disconnection_error", exc_info=True)
                return False

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected

    @property
    def connection(self) -> Any:
        """
        Get the raw database connection object.

        Returns:
            Connection object or None if not connected

        Raises:
            DatabaseError: If connection is accessed but not connected
        """
        if not self._is_connected:
            raise DatabaseError("Database is not connected. Call connect() first.")
        return self._pg_connection.conn if self._pg_connection else None

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            async with database.transaction() as conn:
                # Perform database operations with conn

        Raises:
            DatabaseError: If database is not connected
            Exception: Any exceptions from the transaction are propagated
        """
        if not self._is_connected:
            await self.connect()
            if not self._is_connected:
                raise DatabaseError("Could not connect to database for transaction")

        conn = self.connection
        try:
            # Start transaction if applicable
            if hasattr(conn, "begin") and callable(conn.begin):
                conn.begin()

            yield conn

            # Commit transaction on exit
            if hasattr(conn, "commit"):
                conn.commit()
        except Exception:
            # Rollback on exception
            if hasattr(conn, "rollback"):
                conn.rollback()
            logger.error("transaction_error", exc_info=True)
            raise


# Global database instance for application-wide use
database = Database()
