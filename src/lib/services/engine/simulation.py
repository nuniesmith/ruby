"""
KRAKEN-SIM — Paper Trading Simulation Engine
==============================================
Receives signals, executes mock fills against live tick prices from the
Kraken WebSocket feed, tracks positions and P&L, and publishes state to
Redis for dashboard consumption.

Supports both Kraken (crypto) and Rithmic (futures) data sources.
Signal flow is identical to live trading — only the execution layer is
simulated.

Public API::

    from lib.services.engine.simulation import SimulationEngine

    engine = SimulationEngine(initial_balance=10_000)
    engine.start()          # register tick callback, init DB table
    engine.submit_market_order("KRAKEN:XBTUSD", "long", qty=0.01)
    engine.get_status()     # full sim state dict
    engine.stop()           # unregister, close all, final publish

Environment:
    SIM_INITIAL_BALANCE  — Starting paper balance (default 10000)
    SIM_ENABLED          — "1" to enable simulation (default "0")
    SIM_DATA_SOURCE      — "kraken" (default) or "rithmic"
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("engine.simulation")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_INITIAL_BALANCE = float(os.getenv("SIM_INITIAL_BALANCE", "10000"))
SIM_ENABLED = os.getenv("SIM_ENABLED", "0").strip().lower() in ("1", "true", "yes")
SIM_DATA_SOURCE = os.getenv("SIM_DATA_SOURCE", "kraken").strip().lower()

# ---------------------------------------------------------------------------
# Database schemas
# ---------------------------------------------------------------------------

_SCHEMA_SIM_TRADES_SQLITE = """
CREATE TABLE IF NOT EXISTS sim_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    qty             REAL NOT NULL DEFAULT 1.0,
    pnl             REAL NOT NULL DEFAULT 0.0,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    exit_reason     TEXT NOT NULL DEFAULT 'manual',
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'simulation',
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);
"""

_SCHEMA_SIM_TRADES_POSTGRES = """
CREATE TABLE IF NOT EXISTS sim_trades (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    qty             DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    pnl             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    exit_reason     TEXT NOT NULL DEFAULT 'manual',
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'simulation',
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SimPosition:
    """An open simulated position."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: str
    qty: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    status: str = "open"  # "open" or "closed"


@dataclass
class SimOrder:
    """A pending limit order waiting for fill."""

    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = ""
    side: str = ""  # "long" or "short"
    qty: float = 1.0
    order_type: str = "limit"  # "limit"
    limit_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    created_at: str = ""
    status: str = "pending"  # "pending", "filled", "cancelled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_postgres() -> bool:
    """Check if the active DB backend is Postgres."""
    try:
        from lib.core.models import _is_using_postgres

        return _is_using_postgres()
    except ImportError:
        return False


def _get_conn():
    """Get a database connection (Postgres or SQLite)."""
    try:
        from lib.core.models import _get_conn

        return _get_conn()
    except ImportError:
        import sqlite3

        db_path = os.getenv("DB_PATH", "futures_journal.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def _convert_placeholders(sql: str) -> str:
    """Convert SQLite ``?`` placeholders to Postgres ``%s``."""
    return sql.replace("?", "%s")


def _get_point_value(symbol: str) -> float:
    """Return the point value for *symbol*, defaulting to 1.0."""
    try:
        from lib.core.models import POINT_VALUE

        return POINT_VALUE.get(symbol, 1.0)
    except ImportError:
        return 1.0


def _get_tick_size(symbol: str) -> float:
    """Return the tick size for *symbol*, defaulting to 0.01."""
    try:
        from lib.core.models import TICK_SIZE

        return TICK_SIZE.get(symbol, 0.01)
    except ImportError:
        return 0.01


def _redis_client():
    """Return (redis_client, available) — ``(None, False)`` if unavailable."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        return _r, REDIS_AVAILABLE
    except ImportError:
        return None, False


def _utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def _today_utc_str() -> str:
    """Return today's date string in UTC (YYYY-MM-DD)."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def _calc_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    point_value: float,
) -> float:
    """Calculate realised P&L for a trade.

    For crypto (point_value == 1.0), P&L = qty * price_diff.
    For futures, P&L = qty * price_diff * point_value.
    """
    if side == "long":
        return (exit_price - entry_price) * qty * point_value
    else:
        return (entry_price - exit_price) * qty * point_value


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """Paper trading simulation engine using live tick data.

    Receives signals, executes mock fills against live tick prices,
    tracks positions and P&L, and publishes state to Redis for
    dashboard consumption.

    Supports both Kraken (crypto) and Rithmic (futures) data sources.
    Signal flow is identical to live trading — only the execution
    layer is simulated.
    """

    def __init__(
        self,
        initial_balance: float | None = None,
        data_source: str | None = None,
    ) -> None:
        """Initialise the simulation engine.

        Args:
            initial_balance: Starting paper balance.  Falls back to
                ``SIM_INITIAL_BALANCE`` env var, then 10 000.
            data_source: ``"kraken"`` or ``"rithmic"``.  Falls back to
                ``SIM_DATA_SOURCE`` env var, then ``"kraken"``.
        """
        self._initial_balance = initial_balance or SIM_INITIAL_BALANCE
        self._data_source = (data_source or SIM_DATA_SOURCE).lower()

        # ── Shared state (protected by _lock) ────────────────────────────
        self._lock = threading.Lock()
        self._positions: dict[str, SimPosition] = {}
        self._pending_orders: dict[str, SimOrder] = {}  # order_id → SimOrder
        self._closed_trades: list[dict[str, Any]] = []
        self._latest_prices: dict[str, float] = {}  # symbol → last tick price

        # ── P&L tracking ─────────────────────────────────────────────────
        self._account_balance: float = self._initial_balance
        self._daily_pnl: float = 0.0
        self._total_realized_pnl: float = 0.0
        self._daily_date: str = _today_utc_str()

        # ── Lifecycle ────────────────────────────────────────────────────
        self._running: bool = False
        self._started_at: float | None = None

        logger.info(
            "SimulationEngine created",
            extra={
                "initial_balance": self._initial_balance,
                "data_source": self._data_source,
                "sim_enabled": SIM_ENABLED,
            },
        )

    # =====================================================================
    # Database
    # =====================================================================

    def _init_table(self) -> None:
        """Create the ``sim_trades`` table if it does not exist."""
        conn = _get_conn()
        try:
            schema = _SCHEMA_SIM_TRADES_POSTGRES if _is_postgres() else _SCHEMA_SIM_TRADES_SQLITE
            if _is_postgres():
                for stmt in schema.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                conn.commit()
            else:
                conn.executescript(schema)
            logger.info("sim_trades table ready")
        except Exception as exc:
            logger.warning("Failed to create sim_trades table: %s", exc)
            with contextlib.suppress(Exception):
                conn.rollback()
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _record_trade(self, trade: dict[str, Any]) -> None:
        """Persist a completed trade to the ``sim_trades`` table."""
        conn = _get_conn()
        try:
            if _is_postgres():
                sql = """
                    INSERT INTO sim_trades
                        (symbol, side, entry_price, exit_price, qty, pnl,
                         entry_time, exit_time, exit_reason, duration_seconds,
                         source, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            else:
                sql = """
                    INSERT INTO sim_trades
                        (symbol, side, entry_price, exit_price, qty, pnl,
                         entry_time, exit_time, exit_reason, duration_seconds,
                         source, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            params = (
                trade["symbol"],
                trade["side"],
                trade["entry_price"],
                trade["exit_price"],
                trade["qty"],
                trade["pnl"],
                trade["entry_time"],
                trade["exit_time"],
                trade.get("exit_reason", "manual"),
                trade.get("duration_seconds", 0),
                trade.get("source", "simulation"),
                json.dumps(trade.get("metadata", {})),
            )
            conn.execute(sql, params)
            conn.commit()
            logger.info(
                "sim trade recorded",
                extra={"symbol": trade["symbol"], "pnl": trade["pnl"]},
            )
        except Exception as exc:
            logger.warning("Failed to record sim trade: %s", exc)
            with contextlib.suppress(Exception):
                conn.rollback()
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def get_recorded_trades(self, today_only: bool = False) -> list[dict[str, Any]]:
        """Fetch completed sim trades from the database.

        Args:
            today_only: If ``True``, return only today's trades.

        Returns:
            A list of trade dicts.
        """
        conn = _get_conn()
        try:
            if today_only:
                today_str = _today_utc_str()
                if _is_postgres():
                    sql = "SELECT * FROM sim_trades WHERE exit_time >= %s ORDER BY id DESC"
                else:
                    sql = "SELECT * FROM sim_trades WHERE exit_time >= ? ORDER BY id DESC"
                cur = conn.execute(sql, (today_str,))
            else:
                cur = conn.execute("SELECT * FROM sim_trades ORDER BY id DESC")

            rows = cur.fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                if hasattr(row, "keys"):
                    results.append({k: row[k] for k in row})
                elif isinstance(row, dict):
                    results.append(row)
                else:
                    results.append(dict(row))
            return results
        except Exception as exc:
            logger.warning("Failed to fetch sim trades: %s", exc)
            return []
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    # =====================================================================
    # Redis publishing
    # =====================================================================

    def _publish_state(self) -> None:
        """Publish full simulation state to Redis.

        Called after every state change (fill, close, PnL update).
        Must be called with ``self._lock`` held.
        """
        r, available = _redis_client()
        if not available or r is None:
            return

        try:
            # Positions
            positions_data = [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time,
                    "qty": pos.qty,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "unrealized_pnl": round(pos.unrealized_pnl, 4),
                    "status": pos.status,
                }
                for pos in self._positions.values()
            ]
            r.setex("sim:positions", 300, json.dumps(positions_data))

            # Pending orders
            orders_data = [asdict(order) for order in self._pending_orders.values()]
            r.setex("sim:orders", 300, json.dumps(orders_data))

            # P&L summary
            pnl_data = {
                "balance": round(self._account_balance, 2),
                "daily_pnl": round(self._daily_pnl, 2),
                "total_pnl": round(self._total_realized_pnl, 2),
                "open_positions": len(self._positions),
                "closed_today": len(
                    [t for t in self._closed_trades if t.get("exit_time", "").startswith(self._daily_date)]
                ),
                "initial_balance": self._initial_balance,
                "unrealized_pnl": round(sum(p.unrealized_pnl for p in self._positions.values()), 2),
                "timestamp": _utc_iso(),
            }
            r.setex("sim:pnl", 300, json.dumps(pnl_data))

            # Today's completed trades
            today_trades = [t for t in self._closed_trades if t.get("exit_time", "").startswith(self._daily_date)]
            r.setex("sim:trades", 300, json.dumps(today_trades))

        except Exception as exc:
            logger.warning("Failed to publish sim state to Redis: %s", exc)

    def _publish_event(self, event: dict[str, Any]) -> None:
        """Publish an SSE event to the ``futures:events`` Redis channel.

        Args:
            event: Dict with at minimum an ``"event"`` key.
        """
        r, available = _redis_client()
        if not available or r is None:
            return
        with contextlib.suppress(Exception):
            r.publish("futures:events", json.dumps(event, default=str))

    # =====================================================================
    # Daily P&L reset
    # =====================================================================

    def _check_daily_reset(self) -> None:
        """Reset daily P&L if the UTC date has rolled over.

        Must be called with ``self._lock`` held.
        """
        today = _today_utc_str()
        if today != self._daily_date:
            logger.info(
                "daily P&L reset",
                extra={"prev_date": self._daily_date, "prev_daily_pnl": self._daily_pnl},
            )
            self._daily_pnl = 0.0
            self._daily_date = today

    # =====================================================================
    # Price / unrealised P&L updates
    # =====================================================================

    def _update_unrealized(self, symbol: str, price: float) -> None:
        """Recompute unrealised P&L for a position given the latest price.

        Must be called with ``self._lock`` held.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return
        pv = _get_point_value(symbol)
        pos.unrealized_pnl = round(_calc_pnl(pos.side, pos.entry_price, price, pos.qty, pv), 4)

    # =====================================================================
    # Stop-loss / take-profit checking
    # =====================================================================

    def _check_sl_tp(self, symbol: str, price: float) -> None:
        """Check and trigger stop-loss or take-profit for a position.

        Must be called with ``self._lock`` held.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return

        triggered = False
        exit_reason = "manual"
        exit_price = price

        if pos.stop_loss is not None and (
            pos.side == "long" and price <= pos.stop_loss or pos.side == "short" and price >= pos.stop_loss
        ):
            triggered = True
            exit_reason = "stop_loss"
            exit_price = pos.stop_loss

        if (
            not triggered
            and pos.take_profit is not None
            and (pos.side == "long" and price >= pos.take_profit or pos.side == "short" and price <= pos.take_profit)
        ):
            triggered = True
            exit_reason = "take_profit"
            exit_price = pos.take_profit

        if triggered:
            self._close_position_internal(symbol, exit_price, exit_reason)

    # =====================================================================
    # Pending limit order checking
    # =====================================================================

    def _check_pending_orders(self, symbol: str, price: float) -> None:
        """Check pending limit orders for possible fills at *price*.

        Must be called with ``self._lock`` held.
        """
        filled_ids: list[str] = []

        for order_id, order in self._pending_orders.items():
            if order.symbol != symbol or order.status != "pending":
                continue

            should_fill = False
            if (
                order.side == "long"
                and price <= order.limit_price
                or order.side == "short"
                and price >= order.limit_price
            ):
                should_fill = True

            if should_fill:
                # Fill at the limit price
                order.status = "filled"
                filled_ids.append(order_id)

                pos = SimPosition(
                    symbol=order.symbol,
                    side=order.side,
                    entry_price=order.limit_price,
                    entry_time=_utc_iso(),
                    qty=order.qty,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    status="open",
                )
                self._positions[order.symbol] = pos

                logger.info(
                    "limit order filled",
                    extra={
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.limit_price,
                        "qty": order.qty,
                    },
                )

                self._publish_event(
                    {
                        "event": "sim_fill",
                        "type": "limit_fill",
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": order.limit_price,
                        "qty": order.qty,
                        "order_id": order_id,
                        "timestamp": _utc_iso(),
                    }
                )

        # Remove filled orders
        for oid in filled_ids:
            del self._pending_orders[oid]

    # =====================================================================
    # Internal close (used by SL/TP and public close methods)
    # =====================================================================

    def _close_position_internal(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "manual",
    ) -> dict[str, Any] | None:
        """Close an open position and record the trade.

        Must be called with ``self._lock`` held.

        Returns:
            The trade record dict, or ``None`` if no position was open.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        pv = _get_point_value(symbol)
        pnl = round(_calc_pnl(pos.side, pos.entry_price, exit_price, pos.qty, pv), 4)

        exit_time = _utc_iso()

        # Compute duration
        try:
            entry_dt = datetime.fromisoformat(pos.entry_time)
            exit_dt = datetime.fromisoformat(exit_time)
            duration_seconds = int((exit_dt - entry_dt).total_seconds())
        except Exception:
            duration_seconds = 0

        trade = {
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "qty": pos.qty,
            "pnl": pnl,
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "duration_seconds": duration_seconds,
            "source": "simulation",
            "metadata": {
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "point_value": pv,
            },
        }

        # Update P&L
        self._account_balance += pnl
        self._daily_pnl += pnl
        self._total_realized_pnl += pnl

        # Move to closed
        self._closed_trades.append(trade)
        del self._positions[symbol]

        logger.info(
            "position closed",
            extra={
                "symbol": symbol,
                "side": pos.side,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "balance": round(self._account_balance, 2),
            },
        )

        # Publish events
        self._publish_event(
            {
                "event": "sim_fill",
                "type": "close",
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "timestamp": exit_time,
            }
        )
        self._publish_event(
            {
                "event": "sim_pnl",
                "balance": round(self._account_balance, 2),
                "daily_pnl": round(self._daily_pnl, 2),
                "total_pnl": round(self._total_realized_pnl, 2),
                "timestamp": exit_time,
            }
        )

        # Persist to database (outside of lock would be ideal but we keep it
        # simple here; DB writes are fast for single rows)
        self._publish_state()

        # Record to Postgres/SQLite in a background-safe manner
        # (we release the lock implicitly after this method returns,
        # and the caller re-publishes state anyway)
        threading.Thread(
            target=self._record_trade,
            args=(trade,),
            daemon=True,
            name=f"sim-record-{symbol}",
        ).start()

        return trade

    # =====================================================================
    # Tick processing (callback from Kraken / Rithmic feed)
    # =====================================================================

    def on_tick(self, internal_ticker: str, trade_data: dict[str, Any]) -> None:
        """Process a new tick from the live data feed.

        Called by the KrakenFeedManager ``on_trade`` callback (or a
        Rithmic equivalent in the future).  On each tick:

        1. Update the latest price cache.
        2. Check pending limit orders for fills.
        3. Update unrealised P&L on open positions.
        4. Check stop-loss / take-profit levels.
        5. Publish updated state to Redis.

        Args:
            internal_ticker: Internal symbol, e.g. ``"KRAKEN:XBTUSD"``.
            trade_data: Dict with at minimum ``"price"`` (float).
        """
        price = float(trade_data.get("price", 0))
        if price <= 0:
            return

        state_changed = False

        with self._lock:
            self._check_daily_reset()

            prev_price = self._latest_prices.get(internal_ticker)
            self._latest_prices[internal_ticker] = price

            # Check pending limit orders
            pending_before = len(self._pending_orders)
            self._check_pending_orders(internal_ticker, price)
            if len(self._pending_orders) != pending_before:
                state_changed = True

            # Update unrealised P&L
            if internal_ticker in self._positions:
                self._update_unrealized(internal_ticker, price)
                state_changed = True

            # Check SL/TP
            positions_before = len(self._positions)
            self._check_sl_tp(internal_ticker, price)
            if len(self._positions) != positions_before:
                state_changed = True

            # Publish state if anything changed, or periodically on price change
            if state_changed or (prev_price is not None and prev_price != price):
                self._publish_state()

    # =====================================================================
    # Order submission (public API)
    # =====================================================================

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float = 1.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Submit a market order — fills immediately at current tick price.

        If there is already an open position for *symbol* on the opposite
        side, the existing position is closed first (reversal).

        Args:
            symbol: Internal ticker (e.g. ``"KRAKEN:XBTUSD"``).
            side: ``"long"`` or ``"short"``.
            qty: Number of contracts / units.
            stop_loss: Optional stop-loss price.
            take_profit: Optional take-profit price.

        Returns:
            A dict describing the fill (or an error dict).
        """
        side = side.lower()
        if side not in ("long", "short"):
            return {"error": f"Invalid side: {side!r}. Use 'long' or 'short'."}

        with self._lock:
            self._check_daily_reset()

            # Get the current price
            price = self._latest_prices.get(symbol)
            if price is None or price <= 0:
                return {
                    "error": f"No live price available for {symbol}. Wait for the first tick from the data feed.",
                }

            # If there is an existing position on the opposite side, close it
            existing = self._positions.get(symbol)
            if existing is not None and existing.side != side:
                self._close_position_internal(symbol, price, "reversal")
            elif existing is not None and existing.side == side:
                return {
                    "error": f"Already have an open {existing.side} position in {symbol}. "
                    "Close it first or submit on the opposite side to reverse.",
                }

            # Create the position
            pos = SimPosition(
                symbol=symbol,
                side=side,
                entry_price=price,
                entry_time=_utc_iso(),
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="open",
            )
            self._positions[symbol] = pos

            logger.info(
                "market order filled",
                extra={
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "sl": stop_loss,
                    "tp": take_profit,
                },
            )

            fill_event = {
                "event": "sim_fill",
                "type": "market",
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": pos.entry_time,
            }
            self._publish_event(fill_event)
            self._publish_state()

            return {
                "status": "filled",
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": pos.entry_time,
            }

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float = 1.0,
        limit_price: float = 0.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Submit a limit order — pending until price crosses *limit_price*.

        Args:
            symbol: Internal ticker.
            side: ``"long"`` or ``"short"``.
            qty: Number of contracts / units.
            limit_price: The limit price for entry.
            stop_loss: Optional stop-loss price for the resulting position.
            take_profit: Optional take-profit price for the resulting position.

        Returns:
            A dict describing the pending order (or an error dict).
        """
        side = side.lower()
        if side not in ("long", "short"):
            return {"error": f"Invalid side: {side!r}. Use 'long' or 'short'."}
        if limit_price <= 0:
            return {"error": "limit_price must be > 0."}

        with self._lock:
            # Don't allow a limit order if there is already an open position
            if symbol in self._positions:
                return {
                    "error": f"Already have an open position in {symbol}. Close it before placing a new limit order.",
                }

            order = SimOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="limit",
                limit_price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=_utc_iso(),
                status="pending",
            )
            self._pending_orders[order.order_id] = order

            logger.info(
                "limit order submitted",
                extra={
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "limit_price": limit_price,
                    "qty": qty,
                },
            )

            self._publish_state()

            return {
                "status": "pending",
                "order_id": order.order_id,
                "symbol": symbol,
                "side": side,
                "limit_price": limit_price,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "created_at": order.created_at,
            }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a pending limit order.

        Args:
            order_id: The unique order identifier.

        Returns:
            A status dict.
        """
        with self._lock:
            order = self._pending_orders.get(order_id)
            if order is None:
                return {"error": f"Order {order_id!r} not found."}
            if order.status != "pending":
                return {"error": f"Order {order_id!r} is already {order.status}."}

            order.status = "cancelled"
            del self._pending_orders[order_id]

            logger.info("order cancelled", extra={"order_id": order_id, "symbol": order.symbol})
            self._publish_state()

            return {"status": "cancelled", "order_id": order_id, "symbol": order.symbol}

    # =====================================================================
    # Position closing (public API)
    # =====================================================================

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close the open position for *symbol* at the current market price.

        Args:
            symbol: Internal ticker.

        Returns:
            The trade record dict, or an error dict.
        """
        with self._lock:
            if symbol not in self._positions:
                return {"error": f"No open position for {symbol!r}."}

            price = self._latest_prices.get(symbol)
            if price is None or price <= 0:
                return {"error": f"No live price available for {symbol!r}."}

            trade = self._close_position_internal(symbol, price, "manual")
            return trade if trade is not None else {"error": "Close failed unexpectedly."}

    def close_all(self) -> dict[str, Any]:
        """Close all open positions at current market prices.

        Returns:
            A summary dict with per-symbol results.
        """
        with self._lock:
            symbols = list(self._positions.keys())
            results: dict[str, Any] = {}

            for symbol in symbols:
                price = self._latest_prices.get(symbol)
                if price is None or price <= 0:
                    results[symbol] = {"error": "No live price available."}
                    continue
                trade = self._close_position_internal(symbol, price, "close_all")
                results[symbol] = trade if trade is not None else {"error": "Close failed."}

            # Also cancel all pending orders
            cancelled = list(self._pending_orders.keys())
            for oid in cancelled:
                self._pending_orders[oid].status = "cancelled"
                del self._pending_orders[oid]

            self._publish_state()

            return {
                "closed_positions": results,
                "cancelled_orders": cancelled,
                "balance": round(self._account_balance, 2),
            }

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def start(self) -> None:
        """Start the simulation engine.

        - Initialises the ``sim_trades`` database table.
        - Registers the tick callback with the Kraken feed (if available).
        - Publishes the initial state to Redis.
        """
        if self._running:
            logger.warning("SimulationEngine is already running")
            return

        self._running = True
        self._started_at = time.time()

        # Init DB table
        self._init_table()

        # Register tick callback with the data feed
        if self._data_source == "kraken":
            try:
                from lib.integrations.kraken_client import get_kraken_feed

                feed = get_kraken_feed()
                if feed is not None:
                    feed.on_trade(self.on_tick)
                    logger.info("registered tick callback with KrakenFeedManager")
                else:
                    logger.warning("KrakenFeedManager not started yet — tick callback will need to be registered later")
            except ImportError:
                logger.warning("kraken_client not available — tick callback not registered")
        else:
            logger.info("data_source=%s — manual tick callback registration required", self._data_source)

        # Publish initial state
        with self._lock:
            self._publish_state()

        logger.info("SimulationEngine started", extra={"data_source": self._data_source})

    def stop(self) -> None:
        """Stop the simulation engine.

        - Closes all open positions.
        - Publishes final state.
        - Marks the engine as stopped.
        """
        if not self._running:
            logger.warning("SimulationEngine is not running")
            return

        logger.info("SimulationEngine stopping — closing all positions")

        # Close all positions
        with self._lock:
            symbols = list(self._positions.keys())
            for symbol in symbols:
                price = self._latest_prices.get(symbol)
                if price is not None and price > 0:
                    self._close_position_internal(symbol, price, "engine_stop")

            # Cancel pending orders
            for oid in list(self._pending_orders.keys()):
                self._pending_orders[oid].status = "cancelled"
                del self._pending_orders[oid]

            self._publish_state()

        self._running = False
        logger.info("SimulationEngine stopped")

    def reset(self) -> dict[str, Any]:
        """Reset the simulation — close all, clear history, reset balance.

        Returns:
            A status summary dict.
        """
        logger.info("SimulationEngine resetting")

        with self._lock:
            # Close all open positions at last known prices
            for symbol in list(self._positions.keys()):
                price = self._latest_prices.get(symbol)
                if price is not None and price > 0:
                    self._close_position_internal(symbol, price, "reset")

            # Cancel pending orders
            for oid in list(self._pending_orders.keys()):
                del self._pending_orders[oid]

            # Reset state
            self._positions.clear()
            self._pending_orders.clear()
            self._closed_trades.clear()
            self._latest_prices.clear()

            self._account_balance = self._initial_balance
            self._daily_pnl = 0.0
            self._total_realized_pnl = 0.0
            self._daily_date = _today_utc_str()

            self._publish_state()

        logger.info("SimulationEngine reset complete")
        return {
            "status": "reset",
            "balance": self._initial_balance,
            "timestamp": _utc_iso(),
        }

    # =====================================================================
    # Status / inspection
    # =====================================================================

    def get_status(self) -> dict[str, Any]:
        """Return the full simulation state as a dict.

        Returns:
            A dict with positions, orders, P&L, trades, and metadata.
        """
        with self._lock:
            self._check_daily_reset()

            positions = [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time,
                    "qty": pos.qty,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "unrealized_pnl": round(pos.unrealized_pnl, 4),
                    "status": pos.status,
                }
                for pos in self._positions.values()
            ]

            orders = [asdict(order) for order in self._pending_orders.values()]

            today_trades = [t for t in self._closed_trades if t.get("exit_time", "").startswith(self._daily_date)]

            return {
                "running": self._running,
                "data_source": self._data_source,
                "started_at": self._started_at,
                "positions": positions,
                "pending_orders": orders,
                "pnl": {
                    "balance": round(self._account_balance, 2),
                    "initial_balance": self._initial_balance,
                    "daily_pnl": round(self._daily_pnl, 2),
                    "total_pnl": round(self._total_realized_pnl, 2),
                    "unrealized_pnl": round(sum(p.unrealized_pnl for p in self._positions.values()), 2),
                    "open_positions": len(self._positions),
                    "closed_today": len(today_trades),
                },
                "today_trades": today_trades,
                "latest_prices": dict(self._latest_prices),
                "total_closed_trades": len(self._closed_trades),
                "timestamp": _utc_iso(),
            }

    def get_pnl_summary(self) -> dict[str, Any]:
        """Return a concise P&L summary dict."""
        with self._lock:
            self._check_daily_reset()

            today_count = len([t for t in self._closed_trades if t.get("exit_time", "").startswith(self._daily_date)])

            return {
                "balance": round(self._account_balance, 2),
                "initial_balance": self._initial_balance,
                "daily_pnl": round(self._daily_pnl, 2),
                "total_pnl": round(self._total_realized_pnl, 2),
                "unrealized_pnl": round(sum(p.unrealized_pnl for p in self._positions.values()), 2),
                "open_positions": len(self._positions),
                "pending_orders": len(self._pending_orders),
                "closed_today": today_count,
                "total_closed": len(self._closed_trades),
                "timestamp": _utc_iso(),
            }

    # =====================================================================
    # Convenience properties
    # =====================================================================

    @property
    def is_running(self) -> bool:
        """Whether the engine is currently active."""
        return self._running

    @property
    def positions(self) -> dict[str, SimPosition]:
        """Return a snapshot copy of open positions."""
        with self._lock:
            return dict(self._positions)

    @property
    def pending_orders(self) -> dict[str, SimOrder]:
        """Return a snapshot copy of pending orders."""
        with self._lock:
            return dict(self._pending_orders)

    @property
    def account_balance(self) -> float:
        """Current account balance."""
        return self._account_balance

    @property
    def daily_pnl(self) -> float:
        """Today's realised P&L."""
        return self._daily_pnl

    @property
    def total_realized_pnl(self) -> float:
        """Lifetime realised P&L."""
        return self._total_realized_pnl


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_sim_engine: SimulationEngine | None = None


def get_sim_engine() -> SimulationEngine | None:
    """Return the module-level SimulationEngine singleton, or ``None``."""
    return _sim_engine


def create_sim_engine(
    initial_balance: float | None = None,
    data_source: str | None = None,
) -> SimulationEngine:
    """Create (or recreate) the module-level SimulationEngine singleton.

    Args:
        initial_balance: Starting paper balance.
        data_source: ``"kraken"`` or ``"rithmic"``.

    Returns:
        The newly created engine instance.
    """
    global _sim_engine
    _sim_engine = SimulationEngine(
        initial_balance=initial_balance,
        data_source=data_source,
    )
    return _sim_engine
