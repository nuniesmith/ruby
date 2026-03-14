"""
Redis-based queue for managing jobs between services.
"""

import json
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

from lib.core.logging_config import get_logger

if TYPE_CHECKING:
    from lib.core.db.redis_clients.service import RedisService as RedisClient

logger = get_logger(__name__)


class RedisQueue:
    """Redis-based queue for managing jobs between services."""

    def __init__(self, redis_client: "RedisClient", queue_name: str = "training_jobs"):
        """
        Initialize the Redis queue.

        Args:
            redis_client: RedisClient instance
            queue_name: Base name for the queue keys
        """
        if isinstance(redis_client, dict):
            raise TypeError(
                "Expected RedisClient instance, got dict. Make sure to initialize the Redis client before passing it."
            )

        # Validate redis_client has required methods
        required_methods = ["hset", "lpush", "execute_command"]
        for method in required_methods:
            if not hasattr(redis_client, method):
                raise AttributeError(f"Redis client missing required method: {method}")

        self.redis = redis_client
        self.queue_name = queue_name
        self.processing_queue = f"{queue_name}:processing"
        self.pending_queue = f"{queue_name}:pending"
        self.completed_queue = f"{queue_name}:completed"
        self.failed_queue = f"{queue_name}:failed"
        self.data_hash = f"{queue_name}:data"

        # Determine if we're using async mode based on the client
        self.use_async = getattr(redis_client, "use_async", False)  # Default to False if attribute doesn't exist

        logger.info("redis_queue_initialized", queue_name=queue_name, async_mode=self.use_async)

    def enqueue_job(self, job_id: str, job_data: dict[str, Any]) -> bool | Awaitable[bool]:
        """
        Add a job to the pending queue.

        Returns:
            Union[bool, Awaitable[bool]]: Success status (or awaitable if using async client)
        """
        try:
            # Store job data
            job_data_str = json.dumps(job_data)

            if self.use_async:

                async def _async_enqueue():
                    # Store job data
                    await self.redis.hset(self.data_hash, job_id, job_data_str)  # type: ignore[misc]
                    # Add to pending queue
                    await self.redis.lpush(self.pending_queue, job_id)  # type: ignore[misc]
                    # Set expiration on job data (24 hours)
                    await self.redis.execute_command("EXPIRE", f"{self.data_hash}:{job_id}", 86400)  # type: ignore[misc]
                    return True

                return _async_enqueue()
            else:
                # Store job data
                self.redis.hset(self.data_hash, job_id, job_data_str)
                # Add to pending queue
                self.redis.lpush(self.pending_queue, job_id)
                # Set expiration on job data (24 hours)
                self.redis.execute_command("EXPIRE", f"{self.data_hash}:{job_id}", 86400)
                return True
        except Exception as e:
            logger.error("enqueue_job_failed", job_id=job_id, error=str(e))
            return False

    def dequeue_job(self) -> dict[str, Any] | None | Awaitable[dict[str, Any] | None]:
        """
        Get the next job from the pending queue and move to processing.

        Returns:
            Union[Optional[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]: Job data or None
            (or awaitable if using async client)
        """
        if self.use_async:

            async def _async_dequeue():
                try:
                    # Use rpoplpush if available or implement manually
                    try:
                        job_id = await self.redis.rpoplpush(self.pending_queue, self.processing_queue)  # type: ignore[misc]
                        if not job_id:
                            return None
                    except AttributeError:
                        # Implement rpoplpush manually with pipeline
                        pipeline = await self.redis.pipeline()  # type: ignore[misc]
                        await pipeline.rpop(self.pending_queue)
                        # We need to store the result to use in the next command
                        job_id_future = await self.redis.rpop(self.pending_queue)  # type: ignore[misc]
                        if not job_id_future:
                            return None
                        job_id = job_id_future
                        await self.redis.lpush(self.processing_queue, job_id)  # type: ignore[misc]

                    # Get job data
                    job_data_str = await self.redis.hget(self.data_hash, job_id)  # type: ignore[misc]
                    if not job_data_str:
                        logger.warning("job_missing_data", job_id=job_id)
                        return {"job_id": job_id}

                    # Parse job data
                    job_data = json.loads(job_data_str)
                    job_data["job_id"] = job_id
                    return job_data
                except Exception as e:
                    logger.error("dequeue_job_failed", error=str(e))
                    return None

            return _async_dequeue()
        else:
            try:
                # Atomic operation: move from pending to processing
                job_id = self.redis.rpoplpush(self.pending_queue, self.processing_queue)
                if not job_id:
                    return None

                # Get job data
                job_data_str = self.redis.hget(self.data_hash, job_id)  # type: ignore[arg-type]
                if not job_data_str:
                    logger.warning("job_missing_data", job_id=job_id)
                    return {"job_id": job_id}

                # Parse job data
                job_data = json.loads(job_data_str)  # type: ignore[arg-type]
                job_data["job_id"] = job_id
                return job_data
            except Exception as e:
                logger.error("dequeue_job_failed", error=str(e))
                return None

    def complete_job(self, job_id: str, result: dict[str, Any]) -> bool | Awaitable[bool]:
        """
        Mark a job as completed with results.

        Returns:
            Union[bool, Awaitable[bool]]: Success status (or awaitable if using async client)
        """
        if self.use_async:

            async def _async_complete():
                try:
                    # Update job data with results
                    job_data_str = await self.redis.hget(self.data_hash, job_id)  # type: ignore[misc]
                    if job_data_str:
                        job_data = json.loads(job_data_str)
                        job_data["result"] = result
                        job_data["status"] = "completed"
                        job_data["completed_at"] = time.time()
                        await self.redis.hset(self.data_hash, job_id, json.dumps(job_data))  # type: ignore[misc]

                    # Move from processing to completed
                    await self.redis.lrem(self.processing_queue, 0, job_id)  # type: ignore[misc]
                    await self.redis.lpush(self.completed_queue, job_id)  # type: ignore[misc]
                    return True
                except Exception as e:
                    logger.error("complete_job_failed", job_id=job_id, error=str(e))
                    return False

            return _async_complete()
        else:
            try:
                # Update job data with results
                job_data_str = self.redis.hget(self.data_hash, job_id)
                if job_data_str:
                    job_data = json.loads(job_data_str)  # type: ignore[arg-type]
                    job_data["result"] = result
                    job_data["status"] = "completed"
                    job_data["completed_at"] = time.time()
                    self.redis.hset(self.data_hash, job_id, json.dumps(job_data))

                # Move from processing to completed
                self.redis.lrem(self.processing_queue, 0, job_id)
                self.redis.lpush(self.completed_queue, job_id)
                return True
            except Exception as e:
                logger.error("complete_job_failed", job_id=job_id, error=str(e))
                return False

    def fail_job(self, job_id: str, error: str) -> bool | Awaitable[bool]:
        """
        Mark a job as failed with error information.

        Returns:
            Union[bool, Awaitable[bool]]: Success status (or awaitable if using async client)
        """
        if self.use_async:

            async def _async_fail():
                try:
                    # Update job data with error
                    job_data_str = await self.redis.hget(self.data_hash, job_id)  # type: ignore[misc]
                    if job_data_str:
                        job_data = json.loads(job_data_str)
                        job_data["error"] = error
                        job_data["status"] = "failed"
                        job_data["failed_at"] = time.time()
                        await self.redis.hset(self.data_hash, job_id, json.dumps(job_data))  # type: ignore[misc]

                    # Move from processing to failed
                    await self.redis.lrem(self.processing_queue, 0, job_id)  # type: ignore[misc]
                    await self.redis.lpush(self.failed_queue, job_id)  # type: ignore[misc]
                    return True
                except Exception as e:
                    logger.error("fail_job_failed", job_id=job_id, error=str(e))
                    return False

            return _async_fail()
        else:
            try:
                # Update job data with error
                job_data_str = self.redis.hget(self.data_hash, job_id)
                if job_data_str:
                    job_data = json.loads(job_data_str)  # type: ignore[arg-type]
                    job_data["error"] = error
                    job_data["status"] = "failed"
                    job_data["failed_at"] = time.time()
                    self.redis.hset(self.data_hash, job_id, json.dumps(job_data))

                # Move from processing to failed
                self.redis.lrem(self.processing_queue, 0, job_id)
                self.redis.lpush(self.failed_queue, job_id)
                return True
            except Exception as e:
                logger.error("fail_job_failed", job_id=job_id, error=str(e))
                return False

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics about the queues.

        Returns:
            Dict[str, Any]: Queue statistics
        """
        try:
            if self.use_async:
                return {
                    "pending": await self.redis.llen(self.pending_queue),  # type: ignore[misc]
                    "processing": await self.redis.llen(self.processing_queue),  # type: ignore[misc]
                    "completed": await self.redis.llen(self.completed_queue),  # type: ignore[misc]
                    "failed": await self.redis.llen(self.failed_queue),  # type: ignore[misc]
                }
            else:
                return {
                    "pending": self.redis.llen(self.pending_queue),
                    "processing": self.redis.llen(self.processing_queue),
                    "completed": self.redis.llen(self.completed_queue),
                    "failed": self.redis.llen(self.failed_queue),
                }
        except Exception as e:
            logger.error("get_queue_stats_failed", error=str(e))
            return {"error": str(e)}

    def get_queue_stats_sync(self) -> dict[str, Any]:
        """
        Synchronous version of get_queue_stats for use in non-async contexts.

        Returns:
            Dict[str, Any]: Queue statistics
        """
        if self.use_async:
            logger.warning("sync_stats_with_async_client")

        try:
            # Use the raw_client property if async client needs to be accessed synchronously
            # or use execute_command as a fallback
            return {
                "pending": self.redis.llen(self.pending_queue)
                if not self.use_async
                else self.redis.execute_command("LLEN", self.pending_queue),
                "processing": self.redis.llen(self.processing_queue)
                if not self.use_async
                else self.redis.execute_command("LLEN", self.processing_queue),
                "completed": self.redis.llen(self.completed_queue)
                if not self.use_async
                else self.redis.execute_command("LLEN", self.completed_queue),
                "failed": self.redis.llen(self.failed_queue)
                if not self.use_async
                else self.redis.execute_command("LLEN", self.failed_queue),
            }
        except Exception as e:
            logger.error("get_queue_stats_sync_failed", error=str(e))
            return {"error": str(e)}
