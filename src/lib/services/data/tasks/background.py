"""
Background task manager for the data service.

Wraps the DashboardEngine to run within the FastAPI lifespan context.
Manages startup/shutdown of the engine, Massive WebSocket feed,
and periodic background tasks (data refresh, optimization, backtesting).
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger("data_service.tasks")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[TASKS] %(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)


class BackgroundManager:
    """Manages all background tasks for the data service.

    This is the single coordination point that:
    - Starts/stops the DashboardEngine (data refresh, optimization, backtesting)
    - Manages the Massive WebSocket live feed
    - Provides status information to the API layer
    """

    def __init__(
        self,
        account_size: int = 150_000,
        interval: str = "5m",
        period: str = "5d",
    ):
        self.account_size = account_size
        self.interval = interval
        self.period = period
        self._engine = None
        self._started = False

    @property
    def engine(self):
        """Lazy-load the engine to avoid import-time side effects."""
        if self._engine is None:
            from lib.trading.engine import DashboardEngine

            self._engine = DashboardEngine(
                account_size=self.account_size,
                interval=self.interval,
                period=self.period,
            )
        return self._engine

    async def start(self) -> None:
        """Start all background tasks.

        Called during FastAPI lifespan startup. The engine runs its own
        daemon thread internally, so we just need to call start().
        """
        if self._started:
            logger.warning("Background manager already started")
            return

        logger.info(
            "Starting background manager: account=%s interval=%s period=%s",
            self.account_size,
            self.interval,
            self.period,
        )

        # Initialize the database
        from lib.core.models import init_db

        init_db()
        logger.info("Database initialized")

        # Start the engine (spawns daemon thread for refresh/optimize/backtest)
        self.engine.start()
        self._started = True
        logger.info("Background engine started successfully")

    async def stop(self) -> None:
        """Stop all background tasks.

        Called during FastAPI lifespan shutdown.
        """
        if not self._started:
            return

        logger.info("Stopping background manager...")

        if self._engine is not None:
            await self._engine.stop()

        self._started = False
        logger.info("Background manager stopped")

    def get_status(self) -> dict:
        """Return the current engine status dict."""
        if self._engine is None:
            return {"engine": "not_initialized"}
        return self.engine.get_status()

    def force_refresh(self) -> dict:
        """Trigger an immediate data refresh cycle."""
        if not self._started:
            return {"status": "error", "message": "Engine not started"}

        self.engine.force_refresh()
        return {"status": "ok", "message": "Refresh triggered"}

    def update_settings(
        self,
        account_size: int | None = None,
        interval: str | None = None,
        period: str | None = None,
    ) -> dict:
        """Update engine settings at runtime."""
        changed: dict[str, Any] = {}
        if account_size is not None and account_size != self.account_size:
            self.account_size = account_size
            changed["account_size"] = account_size
        if interval is not None and interval != self.interval:
            self.interval = interval
            changed["interval"] = interval
        if period is not None and period != self.period:
            self.period = period
            changed["period"] = period

        if changed and self._engine is not None:
            self.engine.update_settings(**changed)
            logger.info("Engine settings updated: %s", changed)

        return {"status": "ok", "changed": changed}

    def get_backtest_results(self) -> list:
        """Return the latest backtest results from the engine."""
        if self._engine is None:
            return []
        return self.engine.get_backtest_results()

    def get_strategy_history(self) -> dict:
        """Return per-asset strategy confidence history."""
        if self._engine is None:
            return {}
        return self.engine.get_strategy_history()

    def get_live_feed_status(self) -> dict:
        """Return Massive WebSocket feed status."""
        if self._engine is None:
            return {"status": "not_initialized"}
        return self.engine.get_live_feed_status()

    def start_live_feed(self) -> bool:
        """Start the Massive WebSocket live feed."""
        if self._engine is None:
            return False
        return self.engine.start_live_feed()

    async def stop_live_feed(self) -> None:
        """Stop the Massive WebSocket live feed."""
        if self._engine is not None:
            await self.engine.stop_live_feed()

    @property
    def is_running(self) -> bool:
        return self._started


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_manager: BackgroundManager | None = None


def get_background_manager(
    account_size: int = 150_000,
    interval: str = "5m",
    period: str = "5d",
) -> BackgroundManager:
    """Get or create the singleton BackgroundManager instance."""
    global _manager
    if _manager is None:
        _manager = BackgroundManager(
            account_size=account_size,
            interval=interval,
            period=period,
        )
    return _manager


@asynccontextmanager
async def lifespan_manager(app):
    """FastAPI lifespan context manager.

    Usage in main.py:
        from lib.services.data.tasks.background import lifespan_manager
        app = FastAPI(lifespan=lifespan_manager)

    Starts all background tasks on startup and cleanly shuts them
    down when the server is stopping.
    """
    import os

    account_size = int(os.getenv("DEFAULT_ACCOUNT_SIZE", "150000"))
    interval = os.getenv("DEFAULT_INTERVAL", "5m")
    period = os.getenv("DEFAULT_PERIOD", "5d")

    manager = get_background_manager(
        account_size=account_size,
        interval=interval,
        period=period,
    )

    try:
        await manager.start()
        logger.info("Data service is ready — background tasks running")
        yield
    finally:
        await manager.stop()
        logger.info("Data service shutdown complete")
