import asyncio
import json
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, cast

import redis.asyncio as redis_async
from redis.asyncio import ConnectionError as RedisConnectionError
from redis.asyncio.client import Pipeline, PubSub

from lib.core.db.redis_clients import BaseRedisClient, RedisError, _ensure_connection
from lib.core.db.redis_clients.utils import CustomJSONEncoder
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class AsyncRedisClient(BaseRedisClient):
    """
    Asynchronous Redis client implementing Redis operations using redis.asyncio.
    Inherits connection and reconnection logic from BaseRedisClient.

    All data is stored in Redis as JSON strings. Values retrieved are deserialized from JSON.
    """

    # Override connection type to async Redis
    connection: redis_async.Redis | None  # type: ignore[assignment]

    def _create_connection(self) -> redis_async.Redis:  # type: ignore[override]
        """Creates an asynchronous Redis connection using connection pooling."""
        logger.debug("async_create_connection_start", clean_url=self.clean_url)

        # Get connection pool configuration
        pool_kwargs = self._get_connection_pool_kwargs()

        # Generate unique key for this connection config
        pool_key = self._get_connection_pool_key()

        if self.connection_params.get("use_cluster", False):
            # For cluster mode
            from redis.asyncio.cluster import RedisCluster  # type: ignore[attr-defined]

            # Define factory function to create a cluster connection pool
            def create_cluster_pool(**kwargs):  # type: ignore[return]
                from redis.asyncio.cluster import ClusterNode  # type: ignore[attr-defined]

                startup_nodes = [
                    ClusterNode(host=host, port=self.connection_params["cluster_port"])
                    for host in self.connection_params["cluster_hosts"]
                ]
                return RedisCluster(  # type: ignore[call-arg]
                    startup_nodes=startup_nodes,
                    username=self.connection_params.get("user"),
                    password=self.connection_params.get("password"),
                    ssl=self.connection_params.get("use_tls", False),
                    decode_responses=True,
                    **kwargs,
                )

            # Get or create a cluster connection pool
            connection: Any = self._get_or_create_connection_pool(pool_key, create_cluster_pool, **pool_kwargs)
            logger.debug("async_cluster_connection_created", connection=str(connection))
            return connection  # type: ignore[return-value]
        else:
            # For standard Redis or Sentinel
            # Define factory function for the connection pool
            def create_standard_pool(**kwargs):
                return redis_async.ConnectionPool.from_url(url=self.redis_url, **kwargs)

            # Get or create a standard connection pool
            pool = self._get_or_create_connection_pool(pool_key, create_standard_pool, **pool_kwargs)

            # Create Redis client with the pool
            connection = redis_async.Redis(connection_pool=pool, decode_responses=True)
            logger.debug("async_connection_created", connection=str(connection))
            return connection

    async def ping(self) -> bool:  # type: ignore[override]
        """
        Asynchronously ping the Redis server to verify the connection.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        logger.debug("async_ping_start")

        if not self.connection:
            logger.debug("async_ping_no_connection")
            return False

        try:
            result = await self.connection.ping()
            logger.debug("async_ping_result", result=result)
            return result
        except Exception as e:
            logger.error("async_ping_error", error=str(e), exc_info=True)
            return False

    async def _initialize_connection_async(self) -> None:
        """
        Asynchronously initialize the Redis connection with retry logic.
        """
        logger.debug("async_initialize_connection_start")

        now = time.time()
        last: float = self.last_connection_attempt  # type: ignore[attr-defined]
        if now - last < 5:
            logger.warning("async_connection_init_throttled")
            return

        self.last_connection_attempt = now  # type: ignore[attr-defined]

        for attempt in range(self.max_retries):
            attempt_num = attempt + 1
            logger.debug("async_connection_attempt", attempt=attempt_num, max_retries=self.max_retries)
            try:
                logger.debug("async_connecting_to_redis", clean_url=self.clean_url)
                self.connection = self._create_connection()  # type: ignore[assignment]

                # Verify connection with ping
                if self.connection and await self.ping():
                    logger.info("async_redis_connected", attempt=attempt_num)
                    return
            except RedisError as e:
                logger.warning(
                    "async_connection_attempt_failed",
                    attempt=attempt_num,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                base_sleep = 2**attempt
                jitter = random.uniform(0, 1)
                time_to_sleep = base_sleep + jitter
                logger.debug("async_connection_backoff", sleep_seconds=round(time_to_sleep, 2))
                await asyncio.sleep(time_to_sleep)

        error_message = "Failed to connect to Redis after multiple attempts."
        logger.error("async_redis_connection_failed", max_retries=self.max_retries)
        raise RedisError(error_message)

    async def pubsub(self) -> PubSub:
        """
        Return an asynchronous Redis PubSub object.

        Returns:
            PubSub: Asynchronous Redis PubSub object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_pubsub_start")

        self.reconnect_if_needed()
        if not self.connection:
            error_message = "Redis connection is None, cannot create PubSub."
            logger.error("async_pubsub_no_connection")
            raise RedisError(error_message)

        pubsub_obj = self.connection.pubsub()
        logger.debug("async_pubsub_created", pubsub=str(pubsub_obj))
        return pubsub_obj

    @_ensure_connection
    async def get(self, key: str) -> Any | None:
        """
        Asynchronously retrieve data from Redis by key, and deserialize from JSON.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The deserialized value if the key exists, otherwise None.
                           Value is deserialized from JSON.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_get_start", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_get_no_connection", key=key)
                raise RedisError(error_message)

            value_raw = await self.connection.get(key)
            logger.debug("async_get_raw_value", key=key, value_raw=str(value_raw))

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug("async_get_json_deserialized", key=key)
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug("async_get_raw_value_not_json", key=key)
                    return value_raw
            else:
                logger.debug("async_get_key_not_found", key=key)
                return None

        except RedisConnectionError as e:
            logger.error("async_get_connection_error", key=key, error=str(e), exc_info=True)
            await self._initialize_connection_async()
            return None

    @_ensure_connection
    async def set(self, key: str, value: Any, expiry: int | None = None) -> bool:  # type: ignore[override]
        """
        Asynchronously store a value in Redis, after serializing it to JSON.

        Args:
            key (str): The key to set.
            value (Any): The value to store (will be JSON serialized).
            expiry (Optional[int]): Expiry time in seconds (optional).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_set_start", key=key, expiry=expiry, value_type=str(type(value)))

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("async_set_json_serialized", key=key)

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_set_no_connection", key=key)
                raise RedisError(error_message)

            if expiry:
                result = await self.connection.setex(key, expiry, value_json)
                logger.debug("async_set_with_expiry", key=key, expiry=expiry, result=str(result))
            else:
                result = bool(await self.connection.set(key, value_json))
                logger.debug("async_set_without_expiry", key=key, result=result)

            logger.debug("async_set_success", key=key)
            return True

        except RedisError as e:
            logger.error("async_set_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    async def delete(self, key: str) -> bool:
        """
        Asynchronously delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was successfully deleted, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_delete_start", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_delete_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.delete(key)
            logger.debug("async_delete_result", key=key, result=bool(result))
            return bool(result)

        except RedisError as e:
            logger.error("async_delete_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    async def exists(self, key: str) -> bool:
        """
        Asynchronously check if a key exists in Redis.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_exists_start", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_exists_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.exists(key)
            logger.debug("async_exists_result", key=key, result=bool(result))
            return bool(result)

        except RedisError as e:
            logger.error("async_exists_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    async def keys(self, pattern: str = "*") -> list:
        """
        Asynchronously retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The key pattern to match (default: "*", i.e., all keys).

        Returns:
            list: A list of keys that match the pattern.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_keys_start", pattern=pattern)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_keys_no_connection", pattern=pattern)
                raise RedisError(error_message)

            keys_list = await self.connection.keys(pattern)
            keys_converted = list(keys_list) if isinstance(keys_list, list) else list(cast("Iterable", keys_list))
            logger.debug("async_keys_retrieved", pattern=pattern, count=len(keys_converted))
            return keys_converted

        except RedisError as e:
            logger.error("async_keys_error", pattern=pattern, error=str(e), exc_info=True)
            return []

    @_ensure_connection
    async def publish(self, channel: str, message: str | bytes) -> int:
        """
        Asynchronously publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish the message to.
            message (Union[str, bytes]): The message to publish.

        Returns:
            int: The number of subscribers who received the message.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_publish_start", channel=channel)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_publish_no_connection", channel=channel)
                raise RedisError(error_message)

            result_raw = await self.connection.publish(channel, message)
            result = int(result_raw) if result_raw is not None else 0  # type: ignore[redundant-cast]
            logger.debug("async_publish_success", channel=channel, subscribers=result)
            return result

        except RedisError as e:
            logger.error("async_publish_error", channel=channel, error=str(e), exc_info=True)
            return 0

    def pipeline(self, transaction: bool = True) -> Pipeline:
        """
        Return a Redis pipeline for batching multiple commands (async version).

        Args:
            transaction (bool): Whether to use a transaction (default: True).

        Returns:
            Pipeline: Redis pipeline object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_pipeline_start", transaction=transaction)

        self.reconnect_if_needed()
        if self.connection is None:
            error_message = "Redis connection is not available."
            logger.error("async_pipeline_no_connection")
            raise RedisError(error_message)

        pipeline_obj = self.connection.pipeline(transaction=transaction)
        logger.debug("async_pipeline_created", pipeline=str(pipeline_obj))
        return pipeline_obj  # type: ignore[return-value]

    @_ensure_connection
    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> list:
        """
        Retrieve members from a sorted set within a range (asynchronous).

        Args:
            key (str): Sorted set key.
            start (int): Start of range (inclusive).
            end (int): End of range (inclusive).
            desc (bool): Retrieve in descending order (default: False).
            withscores (bool): Include scores in the result (default: False).
            score_cast_func (Optional[Callable[[str], float]]): Function to cast scores (optional).

        Returns:
            list: List of members (and scores if withscores=True) in the specified range.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug(
            "async_zrange_start",
            key=key,
            start=start,
            end=end,
            desc=desc,
            withscores=withscores,
            score_cast_func_provided=score_cast_func is not None,
        )

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_zrange_no_connection", key=key)
                raise RedisError(error_message)

            if score_cast_func is None:
                result_raw = await self.connection.zrange(key, start, end, desc=desc, withscores=withscores)
            else:
                result_raw = await self.connection.zrange(
                    key, start, end, desc=desc, withscores=withscores, score_cast_func=score_cast_func
                )

            result: list = list(result_raw) if result_raw is not None else []  # type: ignore[redundant-cast]
            logger.debug("async_zrange_success", key=key, count=len(result))
            return result

        except RedisError as e:
            logger.error("async_zrange_error", key=key, error=str(e), exc_info=True)
            return []

    async def get_last_fetched_timestamp(self, asset: str, timeframe: str) -> int | None:
        """
        Retrieve the last fetched timestamp for an asset and timeframe (asynchronous).

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").

        Returns:
            Optional[int]: Last fetched timestamp as integer, or None if not found/invalid.
        """
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug("async_get_last_fetched_timestamp_start", asset=asset, timeframe=timeframe, key=key)

        value = await self.get(key)
        if value is not None:
            try:
                ts = int(value)
                logger.debug("async_get_last_fetched_timestamp_success", asset=asset, timeframe=timeframe, timestamp=ts)
                return ts
            except ValueError:
                logger.error("async_invalid_timestamp_format", asset=asset, timeframe=timeframe, value=str(value))
        else:
            logger.debug("async_get_last_fetched_timestamp_not_found", key=key)

        return None

    async def set_last_fetched_timestamp(self, asset: str, timeframe: str, timestamp: int) -> bool:
        """
        Store the last fetched timestamp for an asset and timeframe (asynchronous).

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").
            timestamp (int): Timestamp to store.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(
            "async_set_last_fetched_timestamp_start",
            asset=asset,
            timeframe=timeframe,
            timestamp=timestamp,
            key=key,
        )

        result = await self.set(key, timestamp)
        logger.debug("async_set_last_fetched_timestamp_result", result=result)
        return result

    async def calculate_fetch_range(
        self, asset: str, timeframe: str, current_time: int, buffer_seconds: int = 120
    ) -> tuple[int, int]:
        """
        Calculate the fetch range for the next API call (asynchronous).

        Args:
            asset (str): Asset symbol.
            timeframe (str): Timeframe.
            current_time (int): Current timestamp.
            buffer_seconds (int): Buffer in seconds to avoid duplicates (default: 120).

        Returns:
            Tuple[int, int]: Tuple of (start_timestamp, end_timestamp) for fetch range.
        """
        logger.debug(
            "async_calculate_fetch_range_start",
            asset=asset,
            timeframe=timeframe,
            current_time=current_time,
            buffer_seconds=buffer_seconds,
        )

        resolution = 60  # seconds per data point for 1-minute data
        last_timestamp = await self.get_last_fetched_timestamp(asset, timeframe)

        if last_timestamp is None:
            two_years = 7 * 24 * 3600  # using 7 days as placeholder; adjust as needed
            start_timestamp = current_time - two_years
            logger.debug("async_fetch_range_no_last_timestamp", start_timestamp=start_timestamp)
        else:
            start_timestamp = last_timestamp + resolution
            logger.debug(
                "async_fetch_range_from_last_timestamp",
                last_timestamp=last_timestamp,
                start_timestamp=start_timestamp,
            )

        end_timestamp = current_time - buffer_seconds
        logger.debug("async_fetch_range_end_timestamp", end_timestamp=end_timestamp)

        if start_timestamp >= end_timestamp:
            logger.debug(
                "async_fetch_range_no_new_data",
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        else:
            logger.debug(
                "async_fetch_range_calculated",
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )

        return start_timestamp, end_timestamp

    # Added missing abstract methods

    @_ensure_connection
    async def health_check(self) -> dict[str, Any]:
        """
        Check if the Redis connection is healthy.

        Returns:
            Dict[str, Any]: Health check information with status and details.
        """
        logger.debug("async_health_check_start")

        health_info: dict[str, Any] = {"status": False, "details": {}}

        try:
            if self.connection is None:
                logger.error("async_health_check_no_connection")
                return health_info

            # Basic ping check
            ping_result = await self.ping()
            health_info["status"] = ping_result
            health_info["details"]["ping"] = ping_result

            if ping_result:
                try:
                    info = await self.connection.info()  # type: ignore[union-attr]
                    health_info["details"]["version"] = info.get("redis_version", "unknown")
                    health_info["details"]["uptime_days"] = info.get("uptime_in_days", 0)
                    health_info["details"]["memory_used"] = info.get("used_memory_human", "unknown")
                    health_info["details"]["clients_connected"] = info.get("connected_clients", 0)
                except Exception as e:
                    logger.error("async_health_check_stats_error", error=str(e), exc_info=True)
                    health_info["details"]["stats_error"] = str(e)

            logger.debug("async_health_check_complete", status=health_info["status"])
            return health_info
        except Exception as e:
            logger.error("async_health_check_error", error=str(e), exc_info=True)
            health_info["details"]["error"] = str(e)
            return health_info

    @_ensure_connection
    async def hget(self, name: str, key: str) -> Any | None:
        """
        Get the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field in the hash.

        Returns:
            Optional[Any]: Value of the field if it exists, None otherwise.
                           Value is deserialized from JSON if possible.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_hget_start", hash_name=name, key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_hget_no_connection", hash_name=name, key=key)
                raise RedisError(error_message)

            value_raw = await self.connection.hget(name, key)
            logger.debug("async_hget_raw_value", hash_name=name, key=key, value_raw=str(value_raw))

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug("async_hget_json_deserialized", hash_name=name, key=key)
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug("async_hget_raw_value_not_json", hash_name=name, key=key)
                    return value_raw
            else:
                logger.debug("async_hget_key_not_found", hash_name=name, key=key)
                return None

        except RedisError as e:
            logger.error("async_hget_error", hash_name=name, key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    async def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Set the value of a hash field.

        Args:
            name (str): Name of the hash.
            key (str): Field in the hash.
            value (Any): Value to set (will be JSON serialized).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_hset_start", hash_name=name, key=key, value_type=str(type(value)))

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("async_hset_json_serialized", hash_name=name, key=key)

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_hset_no_connection", hash_name=name, key=key)
                raise RedisError(error_message)

            result = await self.connection.hset(name, key, value_json)
            logger.debug("async_hset_success", hash_name=name, key=key, result=str(result))
            return True

        except RedisError as e:
            logger.error("async_hset_error", hash_name=name, key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    async def setex(self, name: str, time: int, value: Any) -> bool:
        """
        Set the value and expiration of a key.

        Args:
            name (str): Key name.
            time (int): Expiration time in seconds.
            value (Any): Value to set (will be JSON serialized).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_setex_start", key=name, expiry=time, value_type=str(type(value)))

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("async_setex_json_serialized", key=name)

            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_setex_no_connection", key=name)
                raise RedisError(error_message)

            result = await self.connection.setex(name, time, value_json)
            logger.debug("async_setex_success", key=name, expiry=time, result=str(result))
            return True

        except RedisError as e:
            logger.error("async_setex_error", key=name, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    async def lpush(self, key: str, *values) -> int:
        """
        Push one or more values to the left (head) of a list.

        Args:
            key (str): The list key.
            *values: One or more values to push.

        Returns:
            int: The length of the list after the push operation, or 0 on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_lpush", key=key, count=len(values))

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_lpush_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.lpush(key, *values)  # type: ignore[union-attr]
            logger.debug("async_lpush_success", key=key, result=result)
            return int(result)

        except RedisError as e:
            logger.error("async_lpush_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    async def rpush(self, key: str, *values) -> int:
        """
        Push one or more values to the right (tail) of a list.

        Args:
            key (str): The list key.
            *values: One or more values to push.

        Returns:
            int: The length of the list after the push operation, or 0 on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_rpush", key=key, count=len(values))

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_rpush_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.rpush(key, *values)  # type: ignore[union-attr]
            logger.debug("async_rpush_success", key=key, result=result)
            return int(result)

        except RedisError as e:
            logger.error("async_rpush_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    async def lpop(self, key: str) -> Any | None:
        """
        Remove and return the first element of a list.

        Args:
            key (str): The list key.

        Returns:
            Optional[Any]: The popped value, or None if the list is empty or on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_lpop", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_lpop_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.lpop(key)  # type: ignore[union-attr]
            logger.debug("async_lpop_success", key=key, result=result)
            return result

        except RedisError as e:
            logger.error("async_lpop_error", key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    async def rpop(self, key: str) -> Any | None:
        """
        Remove and return the last element of a list.

        Args:
            key (str): The list key.

        Returns:
            Optional[Any]: The popped value, or None if the list is empty or on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_rpop", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_rpop_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.rpop(key)  # type: ignore[union-attr]
            logger.debug("async_rpop_success", key=key, result=result)
            return result

        except RedisError as e:
            logger.error("async_rpop_error", key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    async def llen(self, key: str) -> int:
        """
        Get the length of a list.

        Args:
            key (str): The list key.

        Returns:
            int: The length of the list, or 0 on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_llen", key=key)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_llen_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.llen(key)  # type: ignore[union-attr]
            logger.debug("async_llen_success", key=key, result=result)
            return int(result)

        except RedisError as e:
            logger.error("async_llen_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    async def lrem(self, key: str, count: int, value) -> int:
        """
        Remove elements equal to value from the list.

        Args:
            key (str): The list key.
            count (int): Number of occurrences to remove (0 = all, negative = from tail).
            value: The value to remove.

        Returns:
            int: The number of removed elements, or 0 on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_lrem", key=key, count=count)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_lrem_no_connection", key=key)
                raise RedisError(error_message)

            result = await self.connection.lrem(key, count, value)  # type: ignore[union-attr]
            logger.debug("async_lrem_success", key=key, result=result)
            return int(result)

        except RedisError as e:
            logger.error("async_lrem_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    async def rpoplpush(self, source: str, destination: str) -> Any | None:
        """
        Remove the last element in a list and push it to another list.

        Args:
            source (str): Source list key.
            destination (str): Destination list key.

        Returns:
            Optional[Any]: The element being transferred, or None if source is empty or on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_rpoplpush", source=source, destination=destination)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_rpoplpush_no_connection", source=source, destination=destination)
                raise RedisError(error_message)

            result = await self.connection.rpoplpush(source, destination)  # type: ignore[union-attr]
            logger.debug("async_rpoplpush_success", source=source, destination=destination, result=result)
            return result

        except RedisError as e:
            logger.error("async_rpoplpush_error", source=source, destination=destination, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    async def execute_command(self, command: str, *args) -> Any:
        """
        Execute an arbitrary Redis command.

        Args:
            command (str): Redis command name.
            *args: Command arguments.

        Returns:
            Any: The command result, or None on error.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("async_execute_command", command=command)

        try:
            if self.connection is None:
                error_message = "Redis connection is not established."
                logger.error("async_execute_command_no_connection", command=command)
                raise RedisError(error_message)

            result = await self.connection.execute_command(command, *args)  # type: ignore[union-attr]
            logger.debug("async_execute_command_success", command=command, result_type=type(result).__name__)
            return result

        except RedisError as e:
            logger.error("async_execute_command_error", command=command, error=str(e), exc_info=True)
            return None
