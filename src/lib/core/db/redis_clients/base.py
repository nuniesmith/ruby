import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Concatenate,
    ParamSpec,
    TypeVar,
)

from redis import ConnectionError as RedisConnectionError
from redis import Redis

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class RedisError(Exception):
    """Custom exception for Redis-related errors."""

    def __init__(self, message="Redis error occurred"):
        logger.error("redis_error_instantiated", message=message)
        self.message = message
        super().__init__(self.message)


P = ParamSpec("P")
R = TypeVar("R")
RedisClientType = TypeVar("RedisClientType", bound="BaseRedisClient")  # Define a TypeVar bound to BaseRedisClient


def _ensure_connection(
    method: Callable[Concatenate[RedisClientType, P], R],
) -> Callable[Concatenate[RedisClientType, P], R]:
    """
    Decorator to ensure Redis connection is alive before calling methods.
    Now generic to work with BaseRedisClient and its subclasses.

    Args:
        method: The method to wrap with connection checking

    Returns:
        A wrapped method that ensures connection is valid before execution
    """

    def wrapper(self_instance: RedisClientType, *args: P.args, **kwargs: P.kwargs) -> R:  # Use RedisClientType here
        logger.debug("ensure_connection_start", method=method.__name__)
        logger.debug(
            "connection_status_check",
            method=method.__name__,
            connection=str(self_instance.connection),
            connection_type=str(type(self_instance.connection)),
        )
        if not self_instance.connection or not self_instance._check_connection_health(timeout=1):
            logger.warning("connection_lost_reconnecting", method=method.__name__)
            self_instance._initialize_connection()
            logger.debug("reconnection_attempt_completed", method=method.__name__)
        else:
            logger.debug("connection_active", method=method.__name__)
        result = method(self_instance, *args, **kwargs)
        logger.debug("ensure_connection_end", method=method.__name__)
        return result

    return wrapper


class BaseRedisClient(ABC):
    """
    Abstract base class for Redis clients (Sync and Async).
    Handles connection, reconnection, and common utilities.

    Attributes:
        timeout (int): Connection timeout in seconds
        max_retries (int): Maximum number of connection retries
        max_connections (int): Maximum number of connections in the pool
        connection_kwargs (dict): Additional connection parameters
        connection (Optional[redis.Redis]): The Redis connection
    """

    DEFAULT_TIMEOUT = 5
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_MAX_CONNECTIONS = 10

    # Class-level connection pool registry to enable reuse across instances
    _connection_pools: ClassVar[dict[str, Any]] = {}
    _pool_lock = threading.RLock()  # Thread-safe lock for connection pool access

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        connection_kwargs: dict[str, Any] | None = None,
    ):
        logger.debug(
            "base_redis_client_init_start",
            timeout=timeout,
            max_retries=max_retries,
            max_connections=max_connections,
        )
        self.max_connections = max_connections
        self.timeout = timeout
        self.max_retries = max_retries
        self.connection_kwargs = connection_kwargs or {}
        self.last_connection_attempt: float = 0  # For throttling reconnection attempts
        self.connection: Redis | None = None
        self.connection_params: dict[str, Any] = {}
        self._setup_connection_params()
        self._initialize_connection()
        logger.debug("base_redis_client_init_completed")

    def _setup_connection_params(self):
        """
        Set up Redis connection parameters from environment variables.
        Now supports Redis Sentinel for high availability.
        """
        logger.debug("setup_connection_params_start")

        # Import utils here to avoid circular imports
        from lib.core.db.redis_clients.utils import clean_redis_url, construct_redis_url

        # Check if we're using Sentinel
        use_sentinel = os.getenv("REDIS_USE_SENTINEL", "false").lower() == "true"

        # Check if we're using Cluster
        use_cluster = os.getenv("REDIS_USE_CLUSTER", "false").lower() == "true"

        if use_sentinel:
            # Get Sentinel parameters
            sentinel_hosts = os.getenv("REDIS_SENTINEL_HOSTS", "").strip().split(",")
            sentinel_port = int(os.getenv("REDIS_SENTINEL_PORT", "26379"))
            sentinel_master = os.getenv("REDIS_SENTINEL_MASTER", "mymaster").strip()

            self.connection_params = {
                "user": os.getenv("REDIS_USER", "default").strip(),
                "password": "****",  # Masked for logging
                "raw_password": os.getenv("REDIS_PASSWORD", "123456").strip(),  # Actual password
                "sentinel_hosts": sentinel_hosts,
                "sentinel_port": sentinel_port,
                "sentinel_master": sentinel_master,
                "db": os.getenv("REDIS_DB", "0"),
                "use_sentinel": use_sentinel,
                "use_cluster": False,
            }

            # For logging, construct a clean representation
            self.clean_url = f"sentinel://{','.join(sentinel_hosts)}:{sentinel_port}/{self.connection_params['db']} (master: {sentinel_master})"
            logger.debug("redis_sentinel_config", clean_url=self.clean_url)
        elif use_cluster:
            # Redis Cluster mode
            cluster_hosts = os.getenv("REDIS_CLUSTER_HOSTS", "").strip().split(",")
            cluster_port = int(os.getenv("REDIS_CLUSTER_PORT", "6379"))

            self.connection_params = {
                "user": os.getenv("REDIS_USER", "default").strip(),
                "password": "****",  # Masked for logging
                "raw_password": os.getenv("REDIS_PASSWORD", "123456").strip(),  # Actual password
                "cluster_hosts": cluster_hosts,
                "cluster_port": cluster_port,
                "use_sentinel": False,
                "use_cluster": True,
            }

            # For logging, construct a clean representation
            self.clean_url = f"cluster://{','.join(cluster_hosts)}:{cluster_port}"
            logger.debug("redis_cluster_config", clean_url=self.clean_url)
        else:
            # Standard Redis connection
            user = os.getenv("REDIS_USER", "default").strip()
            password_raw = os.getenv("REDIS_PASSWORD", "123456").strip()
            host = os.getenv("REDIS_HOST", "redis").strip()
            port = os.getenv("REDIS_PORT", "6379").strip()
            db = os.getenv("REDIS_DB", "0").strip()
            use_tls = os.getenv("REDIS_USE_TLS", "false").lower() == "true"

            # Store connection params with masked password for logging
            self.connection_params = {
                "user": user,
                "password": "****",  # Masked for logging
                "raw_password": password_raw,  # Actual password
                "host": host,
                "port": port,
                "db": db,
                "use_tls": use_tls,
                "use_sentinel": False,
                "use_cluster": False,
            }

            # Use raw_password for actual URL construction but don't log it
            conn_params_for_url = {**self.connection_params}
            conn_params_for_url["password"] = password_raw
            del conn_params_for_url["raw_password"]

            # Construct and clean Redis URL
            self.redis_url = construct_redis_url(**conn_params_for_url)
            self.clean_url = clean_redis_url(self.redis_url)
            logger.debug("redis_url_configured", clean_url=self.clean_url)

        logger.debug("setup_connection_params_completed")

    @abstractmethod
    def _create_connection(self) -> Redis:
        """
        Abstract method to create the Redis connection. Implement in subclasses.

        Returns:
            redis.Redis: A Redis connection

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def _get_connection_pool_kwargs(self) -> dict[str, Any]:
        """
        Get connection pool configuration with improved backoff and retry settings.

        Returns:
            Dict[str, Any]: Connection pool configuration
        """
        return {
            "max_connections": self.max_connections,
            "socket_timeout": self.timeout,
            "socket_connect_timeout": self.timeout,
            "retry_on_timeout": True,
            "health_check_interval": 30,  # Run health check every 30 seconds
            "retry": {
                "retries": self.max_retries,
                "backoff_factor": 0.5,  # Exponential backoff factor
                "status_forcelist": [500, 502, 503, 504],  # Retry on these HTTP statuses
                "allowed_methods": ["GET", "POST"],  # Methods to retry
            },
            **self.connection_kwargs,
        }

    def _get_connection_pool_key(self) -> str:
        """
        Generate a unique key for connection pool based on connection parameters.

        Returns:
            str: Unique key for connection pool
        """
        # Create a sorted string representation of key parameters
        key_parts = []

        if self.connection_params.get("use_sentinel", False):
            hosts = sorted(self.connection_params.get("sentinel_hosts", []))
            key_parts.extend(
                [
                    f"sentinel:{','.join(hosts)}",
                    f"port:{self.connection_params.get('sentinel_port', '26379')}",
                    f"master:{self.connection_params.get('sentinel_master', 'mymaster')}",
                ]
            )
        elif self.connection_params.get("use_cluster", False):
            hosts = sorted(self.connection_params.get("cluster_hosts", []))
            key_parts.extend(
                [f"cluster:{','.join(hosts)}", f"port:{self.connection_params.get('cluster_port', '6379')}"]
            )
        else:
            key_parts.extend(
                [
                    f"host:{self.connection_params.get('host', 'localhost')}",
                    f"port:{self.connection_params.get('port', '6379')}",
                ]
            )

        # Add common parameters
        key_parts.extend(
            [
                f"db:{self.connection_params.get('db', '0')}",
                f"user:{self.connection_params.get('user', 'default')}",
                f"tls:{self.connection_params.get('use_tls', False)}",
            ]
        )

        # Sort for consistency and join
        key_parts.sort()
        return ":".join(key_parts)

    def _get_or_create_connection_pool(self, pool_key: str, pool_factory: Callable, **kwargs) -> Any:
        """
        Get an existing connection pool or create a new one if it doesn't exist.
        Thread-safe implementation.

        Args:
            pool_key (str): Unique key for pool identification
            pool_factory (Callable): Factory function to create a new pool
            **kwargs: Additional arguments for pool creation

        Returns:
            Any: The connection pool
        """
        # Thread-safe access to connection pools
        with self._pool_lock:
            if pool_key in self._connection_pools:
                logger.debug("reusing_connection_pool", pool_key=pool_key)
                return self._connection_pools[pool_key]

            logger.debug("creating_connection_pool", pool_key=pool_key)
            new_pool = pool_factory(**kwargs)
            self._connection_pools[pool_key] = new_pool
            return new_pool

    def _initialize_connection(self):
        """
        Attempt to set up the Redis connection with retries and auto-reconnect.
        Uses exponential backoff with jitter.

        Raises:
            RedisError: If unable to connect after max_retries
        """
        logger.debug("initialize_connection_start", max_retries=self.max_retries)
        now = time.time()
        if now - self.last_connection_attempt < 5:
            logger.warning("connection_init_throttled")
            return

        self.last_connection_attempt = now
        failures = []
        for attempt in range(self.max_retries):
            attempt_num = attempt + 1
            logger.debug("connection_attempt", attempt=attempt_num, max_retries=self.max_retries)
            try:
                logger.debug("connecting_to_redis", clean_url=self.clean_url)
                self.connection = self._create_connection()  # Call subclass method to create connection
                if self.connection and self._check_connection_health():
                    logger.info("redis_connected", attempt=attempt_num)
                    return
            except RedisConnectionError as e:
                failures.append(str(e))
                logger.warning(
                    "connection_attempt_failed",
                    attempt=attempt_num,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                # Calculate adaptive backoff based on failure pattern
                base_sleep = min(30, 2**attempt)  # Cap at 30 seconds
                jitter = random.uniform(0, 2)  # More randomness to avoid thundering herd
                time_to_sleep = base_sleep + jitter
                logger.debug("connection_backoff", sleep_seconds=round(time_to_sleep, 2))
                time.sleep(time_to_sleep)

        error_message = f"Failed to connect to Redis after {self.max_retries} attempts. Errors: {failures}"
        logger.error("redis_connection_failed", max_retries=self.max_retries, errors=failures)
        raise RedisError(error_message)

    def _check_connection_health(self, timeout: float | None = None) -> bool:
        """
        Check if the Redis connection is healthy with an optional timeout.

        Args:
            timeout (Optional[float]): Timeout in seconds for the ping operation

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        logger.debug("health_check_start")

        if not self.connection:
            logger.debug("health_check_no_connection")
            return False

        try:
            # Use ping with timeout if specified
            if timeout is not None:
                # We use a simple command execution with timeout
                result = bool(self.connection.execute_command("PING", _timeout=timeout))
            else:
                result = bool(self.connection.ping())

            logger.debug("health_check_result", healthy=result)
            return result
        except Exception as e:  # Catch all exceptions, not just Redis-specific ones
            logger.error("health_check_error", error=str(e), exc_info=True)
            return False

    def clear_connection_pool(self):
        """Clear all connections in the connection pool."""
        logger.debug("clear_connection_pool_start")
        if self.connection and hasattr(self.connection, "connection_pool"):
            try:
                pool_key = self._get_connection_pool_key()
                with self._pool_lock:
                    if pool_key in self._connection_pools:
                        self.connection.connection_pool.disconnect()
                        del self._connection_pools[pool_key]
                        logger.info("connection_pool_cleared_and_removed", pool_key=pool_key)
                    else:
                        self.connection.connection_pool.disconnect()
                        logger.info("connection_pool_cleared_not_in_registry")
            except Exception as e:
                logger.error("clear_connection_pool_error", error=str(e), exc_info=True)
        else:
            logger.warning("no_connection_pool_to_clear")

    def reconnect_if_needed(self):
        """Reconnect to Redis if the connection is lost."""
        logger.debug("reconnect_check_start")
        if not self.connection or not self._check_connection_health(timeout=2):
            logger.warning("connection_unavailable_reconnecting")
            self._initialize_connection()
            logger.debug("reconnection_triggered")
        else:
            logger.debug("connection_healthy_no_reconnect_needed")
        logger.debug("reconnect_check_completed")

    def ping(self, timeout: float | None = None) -> Any:
        """
        Check if the Redis connection is alive with optional timeout.

        Args:
            timeout (Optional[float]): Timeout in seconds for the ping operation

        Returns:
            bool: True if ping succeeds, False otherwise
        """
        return self._check_connection_health(timeout=timeout)

    def close(self):
        """Closes the Redis connection gracefully."""
        logger.debug("close_connection_start")
        if self.connection:
            try:
                self.connection.close()
                logger.info("redis_connection_closed")
                self.connection = None  # Set to None after closing
            except Exception as e:
                logger.error("close_connection_error", error=str(e), exc_info=True)
        else:
            logger.warning("no_active_connection_to_close")

    def with_timeout(self, timeout: int) -> "BaseRedisClient":
        """
        Return a copy of the client with a different timeout.

        Args:
            timeout (int): New timeout in seconds

        Returns:
            BaseRedisClient: A new client instance with the specified timeout
        """
        logger.debug("creating_client_with_timeout", timeout=timeout)

        new_client = self.__class__(
            timeout=timeout,
            max_retries=self.max_retries,
            max_connections=self.max_connections,
            connection_kwargs=self.connection_kwargs.copy(),
        )
        return new_client

    # === Abstract Method Definitions for Common Redis Operations ===
    @abstractmethod
    def get(self, key: str) -> Any | None:
        """
        Get value from Redis by key.

        Args:
            key (str): The key to retrieve

        Returns:
            Optional[Any]: The value or None if key doesn't exist
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        """
        Set key-value pair in Redis with optional expiration.

        Args:
            key (str): The key
            value (Any): The value
            ex (Optional[int]): Expiration time in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> Any:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete

        Returns:
            bool: True if key was deleted, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> Any:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The key to check

        Returns:
            bool: True if key exists, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def setex(self, name: str, time: int, value: Any) -> Any:
        """
        Set key with expiration time.

        Args:
            name (str): The key
            time (int): Expiration time in seconds
            value (Any): The value

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def hset(self, name: str, key: str, value: Any) -> Any:
        """
        Set hash field.

        Args:
            name (str): Hash name
            key (str): Field name
            value (Any): Field value

        Returns:
            bool: True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def hget(self, name: str, key: str) -> Any:
        """
        Get hash field.

        Args:
            name (str): Hash name
            key (str): Field name

        Returns:
            Optional[Any]: Field value or None if it doesn't exist
        """
        raise NotImplementedError

    @abstractmethod
    def pipeline(self, transaction: bool = True):
        """
        Create a Redis pipeline for transaction support.

        Args:
            transaction (bool): Whether operations should be atomic

        Returns:
            A Redis pipeline object
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> Any:
        """
        Perform a comprehensive health check on the Redis connection.

        Returns:
            Dict[str, Any]: Health check results with metrics
        """
        raise NotImplementedError

    @abstractmethod
    def keys(self, pattern: str = "*") -> Any:
        """
        Find all keys matching the given pattern.

        Args:
            pattern (str): Pattern to match

        Returns:
            List[str]: List of matching keys
        """
        raise NotImplementedError

    @abstractmethod
    def publish(self, channel: str, message: str | bytes) -> Any:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish to
            message (Union[str, bytes]): The message to publish

        Returns:
            int: Number of clients that received the message
        """
        raise NotImplementedError

    @abstractmethod
    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> Any:
        """
        Return a range of members in a sorted set.

        Args:
            key (str): The sorted set key
            start (int): Start index
            end (int): End index
            desc (bool): Whether to sort in descending order
            withscores (bool): Whether to include scores in the result
            score_cast_func (Optional[Callable[[str], float]]): Function to cast scores

        Returns:
            list: List of members in the range
        """
        raise NotImplementedError

    # === Context Managers ===
    async def __aenter__(self):
        """Async context manager enter method."""
        logger.debug("async_context_enter")
        self.reconnect_if_needed()  # Ensure we have an active connection
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Async context manager exit method.
        Note: We no longer close the connection here to maintain connection pooling.
        """
        logger.debug("async_context_exit")
        # Only report errors if any occurred
        if exc_type:
            logger.error("async_context_exception", exc_type=exc_type.__name__, error=str(exc))

    def __enter__(self):
        """Synchronous context manager enter method."""
        logger.debug("sync_context_enter")
        self.reconnect_if_needed()  # Ensure we have an active connection
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        """
        Synchronous context manager exit method.
        Note: We no longer close the connection here to maintain connection pooling.
        """
        logger.debug("sync_context_exit")
        # Only report errors if any occurred
        if exc_type:
            logger.error("sync_context_exception", exc_type=exc_type.__name__, error=str(exc_val))
