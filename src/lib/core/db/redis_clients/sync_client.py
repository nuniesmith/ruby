import json
from collections.abc import Callable
from typing import Any, cast

from redis import ConnectionError as RedisConnectionError
from redis import ConnectionPool, Redis
from redis.client import Pipeline, PubSub  # Import necessary types

from lib.core.db.redis_clients import BaseRedisClient, RedisError, _ensure_connection
from lib.core.db.redis_clients.utils import CustomJSONEncoder
from lib.core.logging_config import get_logger

logger = get_logger(__name__)


class SyncRedisClient(BaseRedisClient):
    """
    Synchronous Redis client implementing all Redis operations.
    Inherits connection and reconnection logic from BaseRedisClient.
    """

    def _create_connection(self) -> Redis:
        """Creates a synchronous Redis connection."""
        logger.debug("sync_create_connection_start", clean_url=self.clean_url)

        # Get connection pool configuration
        pool_kwargs = self._get_connection_pool_kwargs()

        # Generate unique key for this connection config
        pool_key = self._get_connection_pool_key()

        # Define factory function for the connection pool
        def create_pool(**kwargs):
            return ConnectionPool.from_url(url=self.redis_url, **kwargs)

        # Get or create a standard connection pool
        pool = self._get_or_create_connection_pool(pool_key, create_pool, **pool_kwargs)

        connection = Redis(connection_pool=pool, decode_responses=True)

        # Verify connection works - this is critical for early detection of connection issues
        ping_result = connection.ping()
        logger.debug("sync_create_connection_ping", result=ping_result)
        logger.debug("sync_create_connection_created", connection=str(connection))
        return connection

    # Override the base class ping method to ensure it works properly
    def ping(self) -> bool:  # type: ignore[override]
        """Check if the Redis connection is alive."""
        logger.debug("sync_ping_start")
        if not self.connection:
            logger.debug("sync_ping_no_connection")
            return False
        try:
            result = bool(self.connection.ping())
            logger.debug("sync_ping_result", result=result)
            return result
        except RedisError as e:
            logger.error("sync_ping_error", error=str(e), exc_info=True)
            return False

    # Override the reconnect_if_needed method to ensure it works correctly
    def reconnect_if_needed(self):
        """Reconnect to Redis if the connection is lost."""
        logger.debug("sync_reconnect_check_start")
        if not self.connection:
            logger.warning("sync_connection_none_reconnecting")
            self._initialize_connection()
        else:
            try:
                logger.debug("sync_reconnect_ping_check")
                self.connection.ping()
                logger.debug("sync_reconnect_ping_healthy")
            except (RedisConnectionError, BrokenPipeError) as e:
                logger.warning("sync_connection_error_reconnecting", error=str(e))
                self._initialize_connection()
                logger.debug("sync_reconnection_triggered")
        logger.debug("sync_reconnect_check_completed")

    def pubsub(self) -> PubSub:
        """
        Return a synchronous Redis PubSub object.

        Returns:
            PubSub: Synchronous Redis PubSub object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_pubsub_start")
        self.reconnect_if_needed()
        if not self.connection:
            error_message = "Redis connection is None, cannot create PubSub."
            logger.error("sync_pubsub_no_connection")
            raise RedisError(error_message)
        pubsub_obj = self.connection.pubsub()
        logger.debug("sync_pubsub_created", pubsub=str(pubsub_obj))
        return pubsub_obj

    def pipeline(self, transaction: bool = True) -> Pipeline:
        """
        Return a Redis pipeline object for batching commands.

        Args:
            transaction (bool): Whether to use a transaction (default: True).

        Returns:
            Pipeline: Redis pipeline object.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_pipeline_start", transaction=transaction)
        self.reconnect_if_needed()
        if self.connection is None:
            error_message = "Redis connection is not available."
            logger.error("sync_pipeline_no_connection")
            raise RedisError(error_message)
        pipeline_obj = self.connection.pipeline(transaction=transaction)
        logger.debug("sync_pipeline_created", pipeline=str(pipeline_obj))
        return pipeline_obj

    @_ensure_connection
    def get(self, key: str) -> Any | None:
        """
        Retrieve data from Redis by key, and deserialize from JSON.

        Args:
            key (str): The key to retrieve.

        Returns:
            Optional[Any]: The deserialized value if the key exists, otherwise None.
                           Value is deserialized from JSON if possible.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_get_start", key=key)
        try:
            value_raw = self.connection.get(key)  # type: ignore[union-attr]
            logger.debug("sync_get_raw_value", key=key, value_raw=str(value_raw))

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug("sync_get_json_deserialized", key=key)
                    return value
                except json.JSONDecodeError:
                    logger.warning("sync_get_not_valid_json", key=key)
                    return value_raw
            else:
                logger.debug("sync_get_key_not_found", key=key)
                return None

        except RedisConnectionError as e:
            logger.error("sync_get_connection_error", key=key, error=str(e), exc_info=True)
            self._initialize_connection()
            return None

    @_ensure_connection
    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """
        Store a value in Redis, after serializing it to JSON.

        Args:
            key (str): The key to set.
            value (Any): The value to store (will be JSON serialized).
            ex (Optional[int]): Expiry time in seconds (optional).

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_set_start", key=key, expiry=ex, value_type=str(type(value)))
        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("sync_set_json_serialized", key=key)

            if ex:
                result = self.connection.setex(key, ex, value_json)  # type: ignore[union-attr]
                logger.debug("sync_set_with_expiry", key=key, expiry=ex, result=str(result))
            else:
                result = self.connection.set(key, value_json)  # type: ignore[union-attr]
                logger.debug("sync_set_without_expiry", key=key, result=str(result))

            logger.debug("sync_set_success", key=key)
            return True

        except RedisError as e:
            logger.error("sync_set_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    def setex(self, name: str, time: int, value: Any) -> bool:
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
        logger.debug("sync_setex_start", key=name, expiry=time, value_type=str(type(value)))

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("sync_setex_json_serialized", key=name)

            result = self.connection.setex(name, time, value_json)  # type: ignore[union-attr]
            logger.debug("sync_setex_success", key=name, expiry=time, result=str(result))
            return True

        except RedisError as e:
            logger.error("sync_setex_error", key=name, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the key was successfully deleted, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_delete_start", key=key)
        try:
            result = self.connection.delete(key)  # type: ignore[union-attr]
            logger.debug("sync_delete_result", key=key, result=bool(result))
            return bool(result)

        except RedisError as e:
            logger.error("sync_delete_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_exists_start", key=key)
        try:
            result = bool(self.connection.exists(key))  # type: ignore[union-attr]
            logger.debug("sync_exists_result", key=key, result=result)
            return result

        except RedisError as e:
            logger.error("sync_exists_error", key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    def keys(self, pattern: str = "*") -> list:
        """
        Retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The key pattern to match (default: "*", i.e., all keys).

        Returns:
            list: A list of keys that match the pattern.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_keys_start", pattern=pattern)
        try:
            keys_list = self.connection.keys(pattern)  # type: ignore[union-attr]
            keys_converted = list(keys_list)  # Simplified conversion
            logger.debug("sync_keys_retrieved", pattern=pattern, count=len(keys_converted))
            return keys_converted

        except RedisError as e:
            logger.error("sync_keys_error", pattern=pattern, error=str(e), exc_info=True)
            return []

    @_ensure_connection
    def publish(self, channel: str, message: str | bytes) -> int:
        """
        Publish a message to a Redis channel.

        Args:
            channel (str): The channel to publish the message to.
            message (Union[str, bytes]): The message to publish.

        Returns:
            int: The number of subscribers who received the message.

        Raises:
            RedisError: If Redis connection is not established.
        """
        logger.debug("sync_publish_start", channel=channel)
        try:
            result_raw = self.connection.publish(channel, message)  # type: ignore[union-attr]
            result = cast("int", result_raw) if result_raw is not None else 0
            logger.debug("sync_publish_success", channel=channel, subscribers=result)
            return result

        except RedisError as e:
            logger.error("sync_publish_error", channel=channel, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    def hget(self, name: str, key: str) -> Any | None:
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
        logger.debug("sync_hget_start", hash_name=name, key=key)

        try:
            value_raw = self.connection.hget(name, key)  # type: ignore[union-attr]
            logger.debug("sync_hget_raw_value", hash_name=name, key=key, value_raw=str(value_raw))

            if value_raw:
                try:
                    value = json.loads(cast("str", value_raw))
                    logger.debug("sync_hget_json_deserialized", hash_name=name, key=key)
                    return value
                except json.JSONDecodeError:
                    # Return raw value if not a valid JSON string
                    logger.debug("sync_hget_raw_value_not_json", hash_name=name, key=key)
                    return value_raw
            else:
                logger.debug("sync_hget_key_not_found", hash_name=name, key=key)
                return None

        except RedisError as e:
            logger.error("sync_hget_error", hash_name=name, key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    def hset(self, name: str, key: str, value: Any) -> bool:
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
        logger.debug("sync_hset_start", hash_name=name, key=key, value_type=str(type(value)))

        try:
            value_json = json.dumps(value, cls=CustomJSONEncoder)
            logger.debug("sync_hset_json_serialized", hash_name=name, key=key)

            result = self.connection.hset(name, key, value_json)  # type: ignore[union-attr]
            logger.debug("sync_hset_success", hash_name=name, key=key, result=str(result))
            return True

        except RedisError as e:
            logger.error("sync_hset_error", hash_name=name, key=key, error=str(e), exc_info=True)
            return False

    @_ensure_connection
    def health_check(self) -> dict[str, Any]:
        """
        Check if the Redis connection is healthy.

        Returns:
            Dict[str, Any]: Health check information with status and details.
        """
        logger.debug("sync_health_check_start")

        health_info: dict[str, Any] = {"status": False, "details": {}}

        try:
            # Basic ping check
            ping_result = self.ping()
            health_info["status"] = ping_result
            health_info["details"]["ping"] = ping_result

            if ping_result:
                # Add additional stats if connection is working
                try:
                    info = self.connection.info()  # type: ignore[union-attr]
                    health_info["details"]["version"] = info.get("redis_version", "unknown")
                    health_info["details"]["uptime_days"] = info.get("uptime_in_days", 0)
                    health_info["details"]["memory_used"] = info.get("used_memory_human", "unknown")
                    health_info["details"]["clients_connected"] = info.get("connected_clients", 0)
                except Exception as e:
                    logger.error("sync_health_check_stats_error", error=str(e), exc_info=True)
                    health_info["details"]["stats_error"] = str(e)

            logger.debug("sync_health_check_complete", status=health_info["status"])
            return health_info

        except Exception as e:
            logger.error("sync_health_check_error", error=str(e), exc_info=True)
            health_info["details"]["error"] = str(e)
            return health_info

    @_ensure_connection
    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[str], float] | None = None,
    ) -> list:
        """
        Retrieve members from a sorted set within a range.

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
            "sync_zrange_start",
            key=key,
            start=start,
            end=end,
            desc=desc,
            withscores=withscores,
            score_cast_func_provided=score_cast_func is not None,
        )
        try:
            if score_cast_func is None:
                result_raw = self.connection.zrange(key, start, end, desc=desc, withscores=withscores)  # type: ignore[union-attr]
            else:
                result_raw = self.connection.zrange(  # type: ignore[union-attr]
                    key, start, end, desc=desc, withscores=withscores, score_cast_func=score_cast_func
                )

            result = cast("list", result_raw)
            logger.debug("sync_zrange_success", key=key, count=len(result))
            return result

        except RedisError as e:
            logger.error("sync_zrange_error", key=key, error=str(e), exc_info=True)
            return []

    @_ensure_connection
    def lpush(self, key: str, *values) -> int:
        """Push one or more values to the left of a list.

        Args:
            key (str): The list key.
            *values: One or more values to push.

        Returns:
            int: The length of the list after the push, or 0 on error.
        """
        logger.debug("sync_lpush", key=key, count=len(values))
        try:
            result = self.connection.lpush(key, *values)  # type: ignore[union-attr]
            logger.debug("sync_lpush_success", key=key, result=result)
            return int(result)
        except Exception as e:
            logger.error("sync_lpush_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    def rpush(self, key: str, *values) -> int:
        """Push one or more values to the right of a list.

        Args:
            key (str): The list key.
            *values: One or more values to push.

        Returns:
            int: The length of the list after the push, or 0 on error.
        """
        logger.debug("sync_rpush", key=key, count=len(values))
        try:
            result = self.connection.rpush(key, *values)  # type: ignore[union-attr]
            logger.debug("sync_rpush_success", key=key, result=result)
            return int(result)
        except Exception as e:
            logger.error("sync_rpush_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    def lpop(self, key: str) -> Any | None:
        """Pop a value from the left of a list.

        Args:
            key (str): The list key.

        Returns:
            Optional[Any]: The popped value, or None if the list is empty or on error.
        """
        logger.debug("sync_lpop", key=key)
        try:
            result = self.connection.lpop(key)  # type: ignore[union-attr]
            logger.debug("sync_lpop_success", key=key, result=result)
            return result
        except Exception as e:
            logger.error("sync_lpop_error", key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    def rpop(self, key: str) -> Any | None:
        """Pop a value from the right of a list.

        Args:
            key (str): The list key.

        Returns:
            Optional[Any]: The popped value, or None if the list is empty or on error.
        """
        logger.debug("sync_rpop", key=key)
        try:
            result = self.connection.rpop(key)  # type: ignore[union-attr]
            logger.debug("sync_rpop_success", key=key, result=result)
            return result
        except Exception as e:
            logger.error("sync_rpop_error", key=key, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    def llen(self, key: str) -> int:
        """Get the length of a list.

        Args:
            key (str): The list key.

        Returns:
            int: The length of the list, or 0 on error.
        """
        logger.debug("sync_llen", key=key)
        try:
            result = self.connection.llen(key)  # type: ignore[union-attr]
            logger.debug("sync_llen_success", key=key, result=result)
            return int(result)
        except Exception as e:
            logger.error("sync_llen_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    def lrem(self, key: str, count: int, value) -> int:
        """Remove elements from a list.

        Args:
            key (str): The list key.
            count (int): Number of occurrences to remove.
                         count > 0: remove from head to tail.
                         count < 0: remove from tail to head.
                         count = 0: remove all occurrences.
            value: The value to remove.

        Returns:
            int: The number of removed elements, or 0 on error.
        """
        logger.debug("sync_lrem", key=key, count=count, value=value)
        try:
            result = self.connection.lrem(key, count, value)  # type: ignore[union-attr]
            logger.debug("sync_lrem_success", key=key, result=result)
            return int(result)
        except Exception as e:
            logger.error("sync_lrem_error", key=key, error=str(e), exc_info=True)
            return 0

    @_ensure_connection
    def rpoplpush(self, source: str, destination: str) -> Any | None:
        """Pop from the right of source list and push to the left of destination list.

        Args:
            source (str): The source list key.
            destination (str): The destination list key.

        Returns:
            Optional[Any]: The popped/pushed value, or None if the source list is empty or on error.
        """
        logger.debug("sync_rpoplpush", source=source, destination=destination)
        try:
            result = self.connection.rpoplpush(source, destination)  # type: ignore[union-attr]
            logger.debug("sync_rpoplpush_success", source=source, destination=destination, result=result)
            return result
        except Exception as e:
            logger.error("sync_rpoplpush_error", source=source, destination=destination, error=str(e), exc_info=True)
            return None

    @_ensure_connection
    def execute_command(self, command: str, *args) -> Any:
        """Execute an arbitrary Redis command.

        Args:
            command (str): The Redis command to execute.
            *args: Arguments to pass to the command.

        Returns:
            Any: The result of the command, or None on error.
        """
        logger.debug("sync_execute_command", command=command, args=args)
        try:
            result = self.connection.execute_command(command, *args)  # type: ignore[union-attr]
            logger.debug("sync_execute_command_success", command=command, result=result)
            return result
        except Exception as e:
            logger.error("sync_execute_command_error", command=command, error=str(e), exc_info=True)
            return None

    def get_last_fetched_timestamp(self, asset: str, timeframe: str) -> int | None:
        """
        Retrieve the last fetched timestamp for an asset and timeframe.

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").

        Returns:
            Optional[int]: Last fetched timestamp as integer, or None if not found/invalid.
        """
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug("sync_get_last_fetched_timestamp_start", asset=asset, timeframe=timeframe, key=key)
        value = self.get(key)
        if value is not None:
            try:
                ts = int(value)
                logger.debug("sync_get_last_fetched_timestamp_success", asset=asset, timeframe=timeframe, timestamp=ts)
                return ts
            except ValueError:
                logger.error("sync_invalid_timestamp_format", asset=asset, timeframe=timeframe, value=str(value))
        else:
            logger.debug("sync_get_last_fetched_timestamp_not_found", key=key)
        return None

    def set_last_fetched_timestamp(self, asset: str, timeframe: str, timestamp: int) -> bool:
        """
        Store the last fetched timestamp for an asset and timeframe.

        Args:
            asset (str): Asset symbol (e.g., "BTCUSD").
            timeframe (str): Timeframe (e.g., "1m", "1h").
            timestamp (int): Timestamp to store.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        key = f"last_fetched:{asset}:{timeframe}"
        logger.debug(
            "sync_set_last_fetched_timestamp_start",
            asset=asset,
            timeframe=timeframe,
            timestamp=timestamp,
            key=key,
        )
        result = self.set(key, timestamp)
        logger.debug("sync_set_last_fetched_timestamp_result", result=result)
        return result

    def calculate_fetch_range(
        self, asset: str, timeframe: str, current_time: int, buffer_seconds: int = 120
    ) -> tuple[int, int]:
        """
        Calculate the fetch range for the next API call.

        Args:
            asset (str): Asset symbol.
            timeframe (str): Timeframe.
            current_time (int): Current timestamp.
            buffer_seconds (int): Buffer in seconds to avoid duplicates (default: 120).

        Returns:
            Tuple[int, int]: Tuple of (start_timestamp, end_timestamp) for fetch range.
        """
        logger.debug(
            "sync_calculate_fetch_range_start",
            asset=asset,
            timeframe=timeframe,
            current_time=current_time,
            buffer_seconds=buffer_seconds,
        )
        resolution = 60  # seconds per data point for 1-minute data
        last_timestamp = self.get_last_fetched_timestamp(asset, timeframe)
        if last_timestamp is None:
            one_week = 7 * 24 * 3600  # 7 days in seconds
            start_timestamp = current_time - one_week
            logger.debug("sync_fetch_range_no_last_timestamp", start_timestamp=start_timestamp)
        else:
            start_timestamp = last_timestamp + resolution
            logger.debug(
                "sync_fetch_range_from_last_timestamp",
                last_timestamp=last_timestamp,
                start_timestamp=start_timestamp,
            )
        end_timestamp = current_time - buffer_seconds
        logger.debug("sync_fetch_range_end_timestamp", end_timestamp=end_timestamp)
        if start_timestamp >= end_timestamp:
            logger.debug(
                "sync_fetch_range_no_new_data",
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        else:
            logger.debug(
                "sync_fetch_range_calculated",
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        return start_timestamp, end_timestamp
