"""
Rithmic Position Engine — Live Position Data Wrapper
=====================================================
Wrapper around :class:`RithmicStreamManager` providing a clean,
position-oriented interface for the Position Intelligence Engine and
the dashboard.

Responsibilities:
  - Expose current positions per account (from Rithmic PNL plant)
  - Provide L1 (best bid/ask/last) snapshots per symbol
  - Provide L2 (depth-of-market) snapshots per symbol
  - Stream recent trades (time & sales) for a symbol
  - Aggregate real-time P&L per account

All methods are async-ready and return plain dicts / lists so they are
trivially JSON-serialisable for SSE / REST transport.

Usage:
    from lib.services.engine.rithmic_position_engine import (
        RithmicPositionEngine,
    )

    engine = RithmicPositionEngine()
    connected = await engine.connect()
    positions = await engine.get_positions()
    l1 = await engine.get_l1("MES")
    l2 = await engine.get_l2("MES", levels=10)
    trades = await engine.get_recent_trades("MES", n=20)
    pnl = await engine.get_pnl("my_account")
"""

from __future__ import annotations

import time
from typing import Any

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "RithmicPositionEngine",
]


# ---------------------------------------------------------------------------
# Lazy imports — avoid hard crash if rithmic_client deps are missing
# ---------------------------------------------------------------------------


def _import_stream_manager_factory() -> Any:
    """Import ``get_stream_manager`` lazily."""
    try:
        from lib.integrations.rithmic_client import get_stream_manager

        return get_stream_manager
    except Exception:
        logger.warning("rithmic_position_engine: could not import get_stream_manager")
        return None


def _import_stream_manager_class() -> Any:
    """Import ``RithmicStreamManager`` class lazily (for type checks)."""
    try:
        from lib.integrations.rithmic_client import RithmicStreamManager

        return RithmicStreamManager
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RithmicPositionEngine:
    """High-level wrapper around :class:`RithmicStreamManager`.

    Provides a position-centric API surface for the dashboard and the
    Position Intelligence Engine.  All public methods return plain dicts /
    lists suitable for JSON serialisation.

    When no live Rithmic connection is available (missing credentials,
    paper-trading mode, etc.) every method gracefully returns mock /
    empty data so callers never crash.

    Args:
        stream_manager: Optional pre-built ``RithmicStreamManager`` instance.
            When ``None`` (the default), the module-level singleton from
            ``lib.integrations.rithmic_client.get_stream_manager()`` is used
            on :meth:`connect`.
    """

    def __init__(self, stream_manager: Any | None = None) -> None:
        self._stream_manager: Any | None = stream_manager
        self._connected: bool = False
        self._connect_ts: float | None = None
        logger.info(
            "RithmicPositionEngine.__init__",
            has_stream_manager=stream_manager is not None,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Establish (or verify) the connection to Rithmic.

        If no ``stream_manager`` was provided at construction time, this
        method lazily resolves the module-level singleton via
        ``get_stream_manager()`` and calls ``start()``.

        Returns:
            ``True`` if the connection is live, ``False`` otherwise.
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.info("RithmicPositionEngine.connect called")

        if self._stream_manager is None:
            factory = _import_stream_manager_factory()
            if factory is not None:
                try:
                    self._stream_manager = factory()
                except Exception:
                    logger.exception("Failed to obtain RithmicStreamManager")
                    self._connected = False
                    return False

        if self._stream_manager is not None:
            try:
                # RithmicStreamManager.start() is the async entry point
                # await self._stream_manager.start()
                # self._connected = self._stream_manager.is_live()
                pass
            except Exception:
                logger.exception("Failed to start RithmicStreamManager")
                self._connected = False
                return False

        # For now, mark as connected with mock data available
        self._connected = True
        self._connect_ts = time.time()
        logger.info(
            "RithmicPositionEngine.connect result",
            connected=self._connected,
        )
        return self._connected

    def is_connected(self) -> bool:
        """Return ``True`` if the engine believes it has a live connection.

        This is a synchronous convenience check.  It does NOT perform a
        network round-trip — it only reflects the last known state.
        """
        # TODO: Wire to live Rithmic stream when creds available
        if self._stream_manager is not None:
            try:
                if hasattr(self._stream_manager, "is_live"):
                    return bool(self._stream_manager.is_live())
            except Exception:
                pass
        return self._connected

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(
        self,
        account_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return current open positions.

        Args:
            account_key: Optional Rithmic account identifier.  When
                ``None``, positions for all accounts are returned.

        Returns:
            List of position dicts, each containing at minimum:
            ``symbol``, ``qty``, ``avg_price``, ``side``,
            ``unrealized_pnl``, ``account``.
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.debug(
            "RithmicPositionEngine.get_positions",
            account_key=account_key,
        )

        # Mock data for development / UI scaffolding
        mock_positions: list[dict[str, Any]] = [
            {
                "symbol": "MES",
                "qty": 1,
                "avg_price": 5420.00,
                "side": "long",
                "unrealized_pnl": 12.50,
                "account": account_key or "SIM-001",
            },
        ]
        return mock_positions

    # ------------------------------------------------------------------
    # Market data — L1
    # ------------------------------------------------------------------

    async def get_l1(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Return the latest Level-1 quote for *symbol*.

        Args:
            symbol: Instrument symbol (e.g. ``"MES"``).

        Returns:
            Dict with ``bid``, ``ask``, ``last``, ``volume``.
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.debug("RithmicPositionEngine.get_l1", symbol=symbol)

        # Try the real stream manager if available
        if self._stream_manager is not None:
            try:
                if hasattr(self._stream_manager, "get_l1_snapshot"):
                    # snapshot = await self._stream_manager.get_l1_snapshot(symbol)
                    # if snapshot:
                    #     return snapshot
                    pass
            except Exception:
                logger.debug("get_l1: stream_manager call failed", symbol=symbol)

        # Mock L1 data
        return {
            "bid": 5419.75,
            "ask": 5420.00,
            "last": 5419.75,
            "volume": 142_387,
        }

    # ------------------------------------------------------------------
    # Market data — L2 (depth of market)
    # ------------------------------------------------------------------

    async def get_l2(
        self,
        symbol: str,
        levels: int = 10,
    ) -> dict[str, Any]:
        """Return Level-2 depth-of-market for *symbol*.

        Args:
            symbol: Instrument symbol.
            levels: Number of price levels per side to return.

        Returns:
            Dict with ``bids`` and ``asks`` lists.  Each entry is a dict
            with ``price`` (float) and ``size`` (int).
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.debug(
            "RithmicPositionEngine.get_l2",
            symbol=symbol,
            levels=levels,
        )

        # Generate mock L2 with realistic-looking price ladder
        base_bid = 5419.75
        base_ask = 5420.00
        tick_size = 0.25

        bids: list[dict[str, Any]] = []
        asks: list[dict[str, Any]] = []

        for i in range(levels):
            bids.append(
                {
                    "price": round(base_bid - i * tick_size, 2),
                    "size": max(10, 50 - i * 3),  # tapering depth
                }
            )
            asks.append(
                {
                    "price": round(base_ask + i * tick_size, 2),
                    "size": max(10, 45 - i * 3),
                }
            )

        return {"bids": bids, "asks": asks}

    # ------------------------------------------------------------------
    # Time & Sales
    # ------------------------------------------------------------------

    async def get_recent_trades(
        self,
        symbol: str,
        n: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the *n* most recent trades for *symbol*.

        Args:
            symbol: Instrument symbol.
            n: Maximum number of trades to return.

        Returns:
            List of trade dicts ordered newest-first.  Each dict contains
            ``price``, ``size``, ``side`` (``"buy"`` | ``"sell"``),
            ``timestamp`` (epoch float).
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.debug(
            "RithmicPositionEngine.get_recent_trades",
            symbol=symbol,
            n=n,
        )

        # Try the real stream manager if available
        if self._stream_manager is not None:
            try:
                if hasattr(self._stream_manager, "get_recent_ticks"):
                    # ticks = await self._stream_manager.get_recent_ticks(symbol, n)
                    # if ticks:
                    #     return ticks
                    pass
            except Exception:
                logger.debug(
                    "get_recent_trades: stream_manager call failed",
                    symbol=symbol,
                )

        # Mock trades
        now = time.time()
        trades: list[dict[str, Any]] = []
        for i in range(min(n, 20)):
            trades.append(
                {
                    "price": 5419.75 + (0.25 if i % 3 == 0 else 0.0),
                    "size": 1 + (i % 5),
                    "side": "buy" if i % 2 == 0 else "sell",
                    "timestamp": now - i * 0.3,
                }
            )
        return trades

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    async def get_pnl(
        self,
        account_key: str,
    ) -> dict[str, Any]:
        """Return real-time P&L summary for an account.

        Args:
            account_key: Rithmic account identifier.

        Returns:
            Dict with ``daily_pnl``, ``unrealized_pnl``,
            ``realized_pnl`` (all floats, USD).
        """
        # TODO: Wire to live Rithmic stream when creds available
        logger.debug(
            "RithmicPositionEngine.get_pnl",
            account_key=account_key,
        )

        return {
            "daily_pnl": 0.0,
            "unrealized_pnl": 12.50,
            "realized_pnl": 0.0,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<RithmicPositionEngine connected={self._connected} "
            f"stream_manager={'yes' if self._stream_manager else 'no'}>"
        )
