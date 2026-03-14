from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio.client import Pipeline as AsyncPipeline
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import Pipeline, PubSub

from lib.core.db.redis_clients.async_client import AsyncRedisClient
from lib.core.db.redis_clients.queue import RedisQueue
from lib.core.db.redis_clients.sync_client import SyncRedisClient
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class RedisService:
    """
    Redis client factory that returns either a SyncRedisClient or an AsyncRedisClient.

    This implementation supports both synchronous and asynchronous usage.
    In asynchronous code with a sync client, wrap methods with asyncio.to_thread.

    Example usage:
        # Synchronous usage
        redis = RedisService()
        value = redis.get("my_key")

        # Asynchronous usage
        async_redis = RedisService(use_async=True)
        value = await async_redis.get("my_key")

        # Context manager usage
        with RedisService() as redis:
            redis.set("key", "value")

        # Async context manager
        async with RedisService(use_async=True) as redis:
            await redis.set("key", "value")
    """

    def __init__(
        self,
        use_async: bool = False,
        timeout: int = 5,
        max_retries: int = 5,
        max_connections: int = 10,
        connection_kwargs: dict[str, Any] | None = None,
    ):
        logger.debug(
            "redis_service_init_start",
            use_async=use_async,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.use_async = use_async
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_connections = max_connections
        self.connection_kwargs = connection_kwargs or {}
        self._client = self._create_client()  # self._client will be either SyncRedisService or AsyncRedisService
        logger.debug(
            "redis_service_init_completed",
            client_type="AsyncRedisClient" if use_async else "SyncRedisClient",
        )

    def _create_client(self):
        """
        Create and return either a SyncRedisClient or an AsyncRedisClient instance.

        Returns:
            Union[SyncRedisClient, AsyncRedisClient]: Redis client instance
        """
        if self.use_async:
            logger.debug("creating_async_redis_service")
            # Fix: Don't pass connection_kwargs if it's empty
            if not self.connection_kwargs:
                return AsyncRedisClient(
                    timeout=self.timeout, max_retries=self.max_retries, max_connections=self.max_connections
                )
            else:
                # Only pass non-empty connection_kwargs
                return AsyncRedisClient(
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    max_connections=self.max_connections,
                    connection_kwargs=self.connection_kwargs,
                )
        else:
            logger.debug("creating_sync_redis_service")
            # Apply the same fix for SyncRedisClient
            if not self.connection_kwargs:
                return SyncRedisClient(
                    timeout=self.timeout, max_retries=self.max_retries, max_connections=self.max_connections
                )
            else:
                return SyncRedisClient(
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    max_connections=self.max_connections,
                    connection_kwargs=self.connection_kwargs,
                )

    def queue(self, queue_name: str = "default_queue") -> RedisQueue:
        """
        Create a Redis queue for managing jobs between services.

        Args:
            queue_name (str): Base name for the queue keys (default: "default_queue")

        Returns:
            RedisQueue: A queue instance using this Redis client
        """
        logger.debug("creating_redis_queue", queue_name=queue_name)
        return RedisQueue(redis_client=self, queue_name=queue_name)

    def ping(self) -> bool | Awaitable[bool]:
        """
        Check if the Redis connection is alive.

        Returns:
            Union[bool, Awaitable[bool]]: True if the connection is successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("ping")
        result = self._client.ping()
        logger.debug("ping_result", result=result)
        return result

    def pubsub(self) -> PubSub | Awaitable[AsyncPubSub]:
        """
        Return a Redis PubSub object for pub/sub operations.

        Returns:
            Union[PubSub, Awaitable[AsyncPubSub]]: A Redis PubSub object.
            If the client is asynchronous, returns an awaitable that resolves to AsyncPubSub.
        """
        logger.debug("pubsub_request")
        result = self._client.pubsub()
        logger.debug("pubsub_obtained", result=str(result))
        return result

    def get(self, key: str) -> Any | None | Awaitable[Any | None]:
        """
        Retrieve data from Redis.

        Args:
            key (str): The key to retrieve.

        Returns:
            Union[Optional[Any], Awaitable[Optional[Any]]]: The value associated with the key,
            or None if the key does not exist. If the client is asynchronous, returns an awaitable.
        """
        logger.debug("get", key=key)
        result = self._client.get(key)
        logger.debug("get_result", key=key, result=result)
        return result

    def set(self, key: str, value: Any, ex: int | None = None) -> bool | Awaitable[bool]:
        """
        Store a value in Redis.

        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            ex (Optional[int]): Expiration time in seconds. If None, the key persists indefinitely.

        Returns:
            Union[bool, Awaitable[bool]]: True if the command was successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("set", key=key, expiration=ex)
        result = self._client.set(key, value, ex)
        logger.debug("set_result", key=key, result=result)
        return result

    def delete(self, key: str) -> bool | Awaitable[bool]:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            Union[bool, Awaitable[bool]]: True if the command was successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("delete", key=key)
        result = self._client.delete(key)
        logger.debug("delete_result", key=key, result=result)
        return result

    def exists(self, key: str) -> bool | Awaitable[bool]:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The key to check.

        Returns:
            Union[bool, Awaitable[bool]]: True if the key exists, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("exists", key=key)
        result = self._client.exists(key)
        logger.debug("exists_result", key=key, result=result)
        return result

    def keys(self, pattern: str = "*") -> list[str] | Awaitable[list[str]]:
        """
        Retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The pattern to match keys against (default: "*", matches all keys).

        Returns:
            Union[List[str], Awaitable[List[str]]]: A list of keys matching the pattern.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("keys", pattern=pattern)
        result = self._client.keys(pattern)
        logger.debug("keys_result", pattern=pattern, result=result)
        return result

    def hset(self, name: str, key: str, value: Any) -> bool | Awaitable[bool]:
        """
        Set the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field name within the hash.
            value (Any): Value to set.

        Returns:
            Union[bool, Awaitable[bool]]: True if successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("hset", hash_name=name, key=key)
        result = self._client.hset(name, key, value)
        logger.debug("hset_result", hash_name=name, key=key, result=result)
        return result

    def hget(self, name: str, key: str) -> Any | None | Awaitable[Any | None]:
        """
        Get the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field name within the hash.

        Returns:
            Union[Optional[Any], Awaitable[Optional[Any]]]: Value of the field or None if it doesn't exist.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("hget", hash_name=name, key=key)
        result = self._client.hget(name, key)
        logger.debug("hget_result", hash_name=name, key=key, result=result)
        return result

    def setex(self, key: str, time: int, value: Any) -> bool | Awaitable[bool]:
        """
        Set a value with expiration.

        Args:
            key (str): The key to set.
            time (int): Expiration time in seconds.
            value (Any): The value to set.

        Returns:
            Union[bool, Awaitable[bool]]: True if successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("setex", key=key, expiration=time)
        result = self._client.setex(key, time, value)
        logger.debug("setex_result", key=key, result=result)
        return result

    def publish(self, channel: str, message: str | bytes) -> int | Awaitable[int]:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish to.
            message (Union[str, bytes]): The message to publish.

        Returns:
            Union[int, Awaitable[int]]: The number of subscribers that received the message.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("publish", channel=channel)
        result = self._client.publish(channel, message)
        logger.debug("publish_result", channel=channel, result=result)
        return result

    def health_check(self) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        """
        Perform a comprehensive health check on the Redis connection.

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: Health check results with metrics.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("health_check")
        result = self._client.health_check()
        logger.debug("health_check_result", result=result)
        return result

    def close(self) -> None | Awaitable[None]:
        """
        Closes the Redis connection gracefully.

        Returns:
            Union[None, Awaitable[None]]: None or an awaitable that resolves to None.
        """
        logger.debug("close")
        result = self._client.close()
        logger.debug("connection_closed")
        return result

    def pipeline(self, transaction: bool = True) -> Pipeline | AsyncPipeline:
        """
        Return a Redis pipeline for batching multiple commands.

        Args:
            transaction (bool): Whether the pipeline should be transactional (default: True).

        Returns:
            Union[Pipeline, AsyncPipeline]: A Redis pipeline object (either synchronous or asynchronous
            depending on the factory configuration).
        """
        logger.debug("pipeline", transaction=transaction)
        result = self._client.pipeline(transaction)
        logger.debug("pipeline_created", pipeline_type=type(result).__name__)
        return result

    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> list | Awaitable[list]:
        """
        Retrieve members from a sorted set in Redis in the specified range.

        Args:
            key (str): The sorted set key.
            start (int): The start index (inclusive) of the range.
            end (int): The end index (inclusive) of the range.
            desc (bool): Whether to retrieve in descending order (default: False).
            withscores (bool): Whether to return scores along with members (default: False).
            score_cast_func (Optional[Callable[[str], float]]): An optional function to cast scores.

        Returns:
            Union[list, Awaitable[list]]: A list of members (and optionally scores) from the sorted set.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug(
            "zrange",
            key=key,
            start=start,
            end=end,
            desc=desc,
            withscores=withscores,
        )
        result = self._client.zrange(key, start, end, desc, withscores, score_cast_func)
        logger.debug("zrange_result_obtained", key=key)
        return result

    def get_last_fetched_timestamp(self, asset: str, timeframe: str) -> int | None | Awaitable[int | None]:
        """
        Retrieve the last fetched timestamp for a given asset and timeframe.

        Args:
            asset (str): The asset symbol (e.g., "BTCUSD").
            timeframe (str): The timeframe (e.g., "1m", "1h").

        Returns:
            Union[Optional[int], Awaitable[Optional[int]]]: The last fetched timestamp as an integer, or None if not found.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("get_last_fetched_timestamp", asset=asset, timeframe=timeframe)
        key = f"last_fetched:{asset.upper()}:{timeframe}"
        try:
            result = self._client.get(key)
            if isinstance(result, (bytes, str)):
                # For synchronous result that needs conversion
                try:
                    timestamp = int(result)
                    logger.debug(
                        "last_fetched_timestamp_retrieved", asset=asset, timeframe=timeframe, timestamp=timestamp
                    )
                    return timestamp
                except (ValueError, TypeError):
                    result_str = result.decode("utf-8", errors="replace") if isinstance(result, bytes) else result
                    logger.warning("invalid_timestamp_format", asset=asset, timeframe=timeframe, value=result_str)
                    return None
            # Return as is for async result (awaitable)
            return result
        except Exception as e:
            logger.error("get_last_fetched_timestamp_error", asset=asset, timeframe=timeframe, error=str(e))
            return None

    def set_last_fetched_timestamp(self, asset: str, timeframe: str, timestamp: int) -> bool | Awaitable[bool]:
        """
        Store the last fetched timestamp for a given asset and timeframe.

        Args:
            asset (str): The asset symbol (e.g., "BTCUSD").
            timeframe (str): The timeframe (e.g., "1m", "1h").
            timestamp (int): The timestamp to store.

        Returns:
            Union[bool, Awaitable[bool]]: True if the command was successful, False otherwise.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug(
            "set_last_fetched_timestamp",
            asset=asset,
            timeframe=timeframe,
            timestamp=timestamp,
        )
        key = f"last_fetched:{asset.upper()}:{timeframe}"
        return self._client.set(key, str(timestamp))

    def calculate_fetch_range(
        self, asset: str, timeframe: str, current_time: int, buffer_seconds: int = 120
    ) -> tuple[int, int] | Awaitable[tuple[int, int]]:
        """
        Calculate the fetch range for the next API call based on the last fetched timestamp and current time.

        Args:
            asset (str): The asset symbol.
            timeframe (str): The timeframe.
            current_time (int): The current timestamp.
            buffer_seconds (int): Buffer time in seconds to avoid fetching duplicate data.

        Returns:
            Union[Tuple[int, int], Awaitable[Tuple[int, int]]]: A tuple containing the start and end timestamps for the fetch range.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug(
            "calculate_fetch_range",
            asset=asset,
            timeframe=timeframe,
            current_time=current_time,
            buffer_seconds=buffer_seconds,
        )

        # For async client, implement the fetch range calculation logic
        if self.use_async:
            # We need to write a custom implementation that handles awaitable

            async def _async_calculate_range():
                last_timestamp = await self.get_last_fetched_timestamp(asset, timeframe)  # type: ignore[misc]
                start_time = last_timestamp + 1 if last_timestamp else 0
                end_time = current_time - buffer_seconds
                logger.debug("fetch_range_calculated", start_time=start_time, end_time=end_time)
                return start_time, end_time

            return _async_calculate_range()
        else:
            # For synchronous client
            last_timestamp = self.get_last_fetched_timestamp(asset, timeframe)
            start_time = last_timestamp + 1 if isinstance(last_timestamp, int) and last_timestamp else 0
            end_time = current_time - buffer_seconds
            logger.debug("fetch_range_calculated", start_time=start_time, end_time=end_time)
            return start_time, end_time

    def lpush(self, key: str, *values) -> int | Awaitable[int]:
        """
        Push one or more values to the left (head) of a list.

        Args:
            key (str): The list key
            *values: One or more values to push

        Returns:
            Union[int, Awaitable[int]]: The length of the list after the push operation.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("lpush", key=key)
        result = self._client.lpush(key, *values)
        logger.debug("lpush_result", key=key, result=result)
        return result

    def rpush(self, key: str, *values) -> int | Awaitable[int]:
        """
        Push one or more values to the right (tail) of a list.

        Args:
            key (str): The list key
            *values: One or more values to push

        Returns:
            Union[int, Awaitable[int]]: The length of the list after the push operation.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("rpush", key=key)
        result = self._client.rpush(key, *values)
        logger.debug("rpush_result", key=key, result=result)
        return result

    def lpop(self, key: str) -> Any | None | Awaitable[Any | None]:
        """
        Remove and return the first element of a list.

        Args:
            key (str): The list key

        Returns:
            Union[Optional[Any], Awaitable[Optional[Any]]]: The popped value or None if list is empty.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("lpop", key=key)
        result = self._client.lpop(key)
        logger.debug("lpop_result", key=key, result=result)
        return result

    def rpop(self, key: str) -> Any | None | Awaitable[Any | None]:
        """
        Remove and return the last element of a list.

        Args:
            key (str): The list key

        Returns:
            Union[Optional[Any], Awaitable[Optional[Any]]]: The popped value or None if list is empty.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("rpop", key=key)
        result = self._client.rpop(key)
        logger.debug("rpop_result", key=key, result=result)
        return result

    def llen(self, key: str) -> int | Awaitable[int]:
        """
        Get the length of a list.

        Args:
            key (str): The list key

        Returns:
            Union[int, Awaitable[int]]: The length of the list.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("llen", key=key)
        result = self._client.llen(key)
        logger.debug("llen_result", key=key, result=result)
        return result

    def lrem(self, key: str, count: int, value) -> int | Awaitable[int]:
        """
        Remove elements equal to value from the list.

        Args:
            key (str): The list key
            count (int): Number of occurrences to remove (0 = all, negative = from tail)
            value: The value to remove

        Returns:
            Union[int, Awaitable[int]]: Number of removed elements.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("lrem", key=key, count=count)
        result = self._client.lrem(key, count, value)
        logger.debug("lrem_result", key=key, result=result)
        return result

    def rpoplpush(self, source: str, destination: str) -> Any | None | Awaitable[Any | None]:
        """
        Remove the last element in a list and push it to another list.

        Args:
            source (str): Source list key
            destination (str): Destination list key

        Returns:
            Union[Optional[Any], Awaitable[Optional[Any]]]: The element being transferred or None if source is empty.
            If the client is asynchronous, returns an awaitable.
        """
        logger.debug("rpoplpush", source=source, destination=destination)
        result = self._client.rpoplpush(source, destination)
        logger.debug("rpoplpush_result", result=result)
        return result

    def execute_command(self, command: str, *args) -> Any | Awaitable[Any]:
        """
        Execute an arbitrary Redis command.

        Args:
            command (str): Redis command name
            *args: Command arguments

        Returns:
            Union[Any, Awaitable[Any]]: The command result or an awaitable.
        """
        logger.debug("execute_command", command=command)
        result = self._client.execute_command(command, *args)
        logger.debug("execute_command_result", command=command, result_type=type(result).__name__)
        return result

    @property
    def raw_client(self):
        """
        Access the raw Redis client directly.

        This bypasses the RedisService wrapper functionality and should be used with caution.

        Returns:
            The underlying Redis client instance
        """
        return self._client

    # === Context Managers ===
    async def __aenter__(self):
        """Async context manager enter method."""
        logger.debug("async_context_enter")
        if self.use_async:
            return await self._client.__aenter__()  # Delegate to async client
        else:
            # Convert sync client to work with async context
            logger.debug("sync_client_in_async_context")
            self._client.__enter__()  # Initialize the sync client
            return self  # Return self so methods can be called directly

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit method."""
        logger.debug("async_context_exit")
        if self.use_async:
            return await self._client.__aexit__(exc_type, exc, tb)  # Delegate to async client
        else:
            # Handle sync client in async context
            return self._client.__exit__(exc_type, exc, tb)

    def __enter__(self):
        """Synchronous context manager enter method."""
        logger.debug("sync_context_enter")
        if not self.use_async:
            return self._client.__enter__()  # Delegate to sync client
        else:
            raise RuntimeError(
                "Cannot use synchronous context manager with AsyncRedisClient. Use 'async with' instead."
            )

    def __exit__(self, exc_type, exc_val, traceback):
        """Synchronous context manager exit method."""
        logger.debug("sync_context_exit")
        if not self.use_async:
            return self._client.__exit__(exc_type, exc_val, traceback)  # Delegate to sync client
        else:
            raise RuntimeError(
                "Cannot use synchronous context manager with AsyncRedisClient. Use 'async with' instead."
            )
