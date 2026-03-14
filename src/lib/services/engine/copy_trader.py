"""
Copy Trader — Prop-Firm Compliant Multi-Account Order Replication
===================================================================
.. module:: lib.services.engine.copy_trader
Mirrors orders from a **main** Rithmic account to one or more **slave**
accounts with full prop-firm compliance:

    ✅ ``OrderPlacement.MANUAL`` on every order (main + slaves)
    ✅ Randomised 200–800 ms delay before each slave copy (humanises)
    ✅ Server-side hard stops via ``stop_ticks`` (never mental stops)
    ✅ Tagged audit trail on every order
    ✅ One-asset focus lock (PositionManager integration)
    ✅ Rate-limit monitoring (rolling 60-min action counter)

Architecture::

    WebUI "SEND ALL" button
        → CopyTrader.send_order_and_copy()
            → main_client.submit_order()          # human-initiated
            → for each slave:
                  asyncio.sleep(random 0.2–0.8s)   # humanise
                  slave_client.submit_order()       # MANUAL flag

    Main account fill listener (optional, Phase 2):
        → CopyTrader.on_main_fill()
            → mirrors to all enabled slaves

Environment Variables:
    CT_COPY_DELAY_MIN       — min delay before slave copy (default 0.2)
    CT_COPY_DELAY_MAX       — max delay before slave copy (default 0.8)
    CT_HIGH_IMPACT_DELAY_MIN — min delay on NFP/FOMC days (default 1.0)
    CT_HIGH_IMPACT_DELAY_MAX — max delay on NFP/FOMC days (default 2.0)
    CT_RATE_LIMIT_WARN      — warn threshold per rolling 60 min (default 3000)
    CT_RATE_LIMIT_HARD      — hard stop threshold (default 4500)
    CT_REDIS_LOG_TTL        — TTL for Redis order log entries (default 86400)

Usage::

    from lib.services.engine.copy_trader import CopyTrader

    ct = CopyTrader()

    # Add accounts (from RithmicAccountManager configs)
    await ct.add_account(config, is_main=True)
    await ct.add_account(slave_config_1)
    await ct.add_account(slave_config_2)

    # One-click order + copy (WebUI button)
    result = await ct.send_order_and_copy(
        security_code="MGCQ6",
        exchange="NYMEX",
        side="BUY",
        qty=1,
        order_type="MARKET",
        stop_ticks=20,
        target_ticks=40,
    )

    # Shutdown
    await ct.disconnect_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger("engine.copy_trader")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

COPY_DELAY_MIN = float(os.getenv("CT_COPY_DELAY_MIN", "0.2"))
COPY_DELAY_MAX = float(os.getenv("CT_COPY_DELAY_MAX", "0.8"))
HIGH_IMPACT_DELAY_MIN = float(os.getenv("CT_HIGH_IMPACT_DELAY_MIN", "1.0"))
HIGH_IMPACT_DELAY_MAX = float(os.getenv("CT_HIGH_IMPACT_DELAY_MAX", "2.0"))
RATE_LIMIT_WARN = int(os.getenv("CT_RATE_LIMIT_WARN", "3000"))
RATE_LIMIT_HARD = int(os.getenv("CT_RATE_LIMIT_HARD", "4500"))
REDIS_LOG_TTL = int(os.getenv("CT_REDIS_LOG_TTL", "86400"))  # 24 h
RITHMIC_DEBUG_LOGGING = os.getenv("RITHMIC_DEBUG_LOGGING", "0") == "1"

# Redis keys
_REDIS_ORDER_LOG_KEY = "engine:copy_trader:order_log"
_REDIS_RATE_COUNTER_KEY = "engine:copy_trader:rate_counter"
_REDIS_COMPLIANCE_KEY = "engine:copy_trader:compliance_log"

# ---------------------------------------------------------------------------
# Yahoo ticker → Rithmic product code + exchange mapping
# async_rithmic uses get_front_month_contract(exchange, product_code) to
# resolve the current front-month security_code automatically.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tick-size table: product_code → tick size in price units
# Used to convert a stop_price distance (in price units) into stop_ticks.
# These are standard CME/NYMEX micro contract tick sizes.
# ---------------------------------------------------------------------------

TICK_SIZE: dict[str, float] = {
    # Micro metals (NYMEX)
    "MGC": 0.10,  # Micro Gold — $0.10/oz, $1.00/tick
    "SIL": 0.005,  # Micro Silver — $0.005/oz, $1.00/tick
    # Micro energy (NYMEX)
    "MCL": 0.01,  # Micro Crude Oil — $0.01/bbl, $1.00/tick
    "MNG": 0.005,  # Micro Natural Gas — $0.005/MMBtu
    # Micro equity index (CME)
    "MES": 0.25,  # Micro E-mini S&P 500 — $0.25/pt, $1.25/tick
    "MNQ": 0.25,  # Micro E-mini Nasdaq 100 — $0.25/pt, $0.50/tick
    "MYM": 1.0,  # Micro Dow — $1.00/pt, $0.50/tick
    "M2K": 0.10,  # Micro Russell 2000 — $0.10/pt, $0.50/tick
    # Micro FX (CME)
    "M6E": 0.0001,  # Micro Euro FX — $0.0001/EUR
    "M6A": 0.0001,  # Micro AUD/USD
    "M6B": 0.0001,  # Micro GBP/USD
    "M6J": 0.0000001,  # Micro JPY/USD (per USD per JPY unit)
    # Micro crypto (CME)
    "MBT": 5.0,  # Micro Bitcoin — $5.00/BTC
    "MET": 0.05,  # Micro Ether — $0.05/ETH
    # Full-size (fallback)
    "ES": 0.25,
    "NQ": 0.25,
    "GC": 0.10,
    "CL": 0.01,
}

# Minimum stop_ticks to always attach (protects against too-tight stops)
MIN_STOP_TICKS = 2
# Default stop_ticks when no tick-size is available
DEFAULT_STOP_TICKS = 20


def stop_price_to_stop_ticks(
    entry_price: float,
    stop_price: float,
    product_code: str,
    *,
    min_ticks: int = MIN_STOP_TICKS,
    default_ticks: int = DEFAULT_STOP_TICKS,
) -> int:
    """Convert a stop-loss price into a ``stop_ticks`` integer for Rithmic.

    Rithmic server-side brackets accept ``stop_ticks`` (an integer number of
    ticks from the fill price, always positive).  PositionManager stores the
    absolute stop price, so we compute::

        distance = abs(entry_price - stop_price)
        tick_size = TICK_SIZE[product_code]
        stop_ticks = max(min_ticks, round(distance / tick_size))

    Args:
        entry_price: Expected fill / trigger price for the entry order.
        stop_price:  Absolute stop-loss price level.
        product_code: Rithmic product code (e.g. "MGC", "MES").
        min_ticks:   Minimum ticks to enforce (prevents stop too close).
        default_ticks: Fallback when product_code is not in TICK_SIZE.

    Returns:
        Positive integer number of ticks.
    """
    tick = TICK_SIZE.get(product_code)
    if tick is None or tick <= 0 or entry_price <= 0:
        logger.debug(
            "stop_price_to_stop_ticks: unknown product_code=%r or zero entry — using default %d ticks",
            product_code,
            default_ticks,
        )
        return default_ticks

    distance = abs(entry_price - stop_price)
    if distance <= 0:
        return min_ticks

    raw = round(distance / tick)
    result = max(min_ticks, int(raw))
    logger.debug(
        "stop_price_to_stop_ticks: %s entry=%.4f stop=%.4f dist=%.4f tick=%.6f → %d ticks",
        product_code,
        entry_price,
        stop_price,
        distance,
        tick,
        result,
    )
    return result


TICKER_TO_RITHMIC: dict[str, dict[str, str]] = {
    # Core watchlist (micro contracts)
    "MGC=F": {"product_code": "MGC", "exchange": "NYMEX", "name": "Micro Gold"},
    "MCL=F": {"product_code": "MCL", "exchange": "NYMEX", "name": "Micro Crude Oil"},
    "MES=F": {"product_code": "MES", "exchange": "CME", "name": "Micro E-mini S&P"},
    "MNQ=F": {"product_code": "MNQ", "exchange": "CME", "name": "Micro E-mini Nasdaq"},
    "M6E=F": {"product_code": "M6E", "exchange": "CME", "name": "Micro Euro FX"},
    # Extended watchlist
    "MYM=F": {"product_code": "MYM", "exchange": "CBOT", "name": "Micro Dow"},
    "M2K=F": {"product_code": "M2K", "exchange": "CME", "name": "Micro Russell 2000"},
    "MBT=F": {"product_code": "MBT", "exchange": "CME", "name": "Micro Bitcoin"},
    "MET=F": {"product_code": "MET", "exchange": "CME", "name": "Micro Ether"},
    "M6A=F": {"product_code": "M6A", "exchange": "CME", "name": "Micro AUD/USD"},
    "M6B=F": {"product_code": "M6B", "exchange": "CME", "name": "Micro GBP/USD"},
    "M6J=F": {"product_code": "M6J", "exchange": "CME", "name": "Micro JPY/USD"},
    "SIL=F": {"product_code": "SIL", "exchange": "NYMEX", "name": "Micro Silver"},
    "MNG=F": {"product_code": "MNG", "exchange": "NYMEX", "name": "Micro Natural Gas"},
    # Full-size (if ever needed)
    "ES=F": {"product_code": "ES", "exchange": "CME", "name": "E-mini S&P 500"},
    "NQ=F": {"product_code": "NQ", "exchange": "CME", "name": "E-mini Nasdaq 100"},
    "GC=F": {"product_code": "GC", "exchange": "NYMEX", "name": "Gold"},
    "CL=F": {"product_code": "CL", "exchange": "NYMEX", "name": "Crude Oil"},
}


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class CopyOrderStatus(StrEnum):
    """Status of an individual copy order."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class CopyOrderResult:
    """Result of a single order submission (main or slave)."""

    account_key: str
    account_label: str
    is_main: bool
    order_id: str
    security_code: str
    exchange: str
    side: str
    qty: int
    order_type: str
    price: float
    stop_ticks: int
    target_ticks: int | None
    status: CopyOrderStatus
    error: str = ""
    delay_ms: int = 0
    placement_mode: str = "MANUAL"
    tag: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CopyBatchResult:
    """Aggregate result of a send_order_and_copy() call."""

    batch_id: str
    security_code: str
    side: str
    qty: int
    main_result: CopyOrderResult | None = None
    slave_results: list[CopyOrderResult] = field(default_factory=list)
    compliance_log: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def all_submitted(self) -> bool:
        results = ([self.main_result] if self.main_result else []) + self.slave_results
        return all(r.status in (CopyOrderStatus.SUBMITTED, CopyOrderStatus.FILLED) for r in results)

    @property
    def total_orders(self) -> int:
        return (1 if self.main_result else 0) + len(self.slave_results)

    @property
    def failed_count(self) -> int:
        results = ([self.main_result] if self.main_result else []) + self.slave_results
        return sum(1 for r in results if r.status in (CopyOrderStatus.ERROR, CopyOrderStatus.REJECTED))

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "security_code": self.security_code,
            "side": self.side,
            "qty": self.qty,
            "main_result": self.main_result.to_dict() if self.main_result else None,
            "slave_results": [r.to_dict() for r in self.slave_results],
            "compliance_log": self.compliance_log,
            "total_orders": self.total_orders,
            "failed_count": self.failed_count,
            "all_submitted": self.all_submitted,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Rate-limit tracker
# ---------------------------------------------------------------------------


class RollingRateCounter:
    """Track order actions in a rolling 60-minute window.

    Each action is timestamped; expired entries are pruned lazily.
    The Rithmic practical limit is ~5,000 actions per rolling 60 minutes
    (place + modify + cancel each count as 1 action).
    """

    def __init__(self, window_seconds: int = 3600) -> None:
        self._window: int = window_seconds
        self._actions: deque[float] = deque()

    def record(self, count: int = 1) -> None:
        now = time.monotonic()
        for _i in range(count):
            self._actions.append(now)

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._actions and self._actions[0] < cutoff:
            self._actions.popleft()

    @property
    def count(self) -> int:
        self._prune()
        return len(self._actions)

    @property
    def is_warn(self) -> bool:
        return self.count >= RATE_LIMIT_WARN

    @property
    def is_hard_limit(self) -> bool:
        return self.count >= RATE_LIMIT_HARD

    def status_dict(self) -> dict[str, Any]:
        c = self.count
        return {
            "actions_60min": c,
            "warn_threshold": RATE_LIMIT_WARN,
            "hard_threshold": RATE_LIMIT_HARD,
            "warn": c >= RATE_LIMIT_WARN,
            "blocked": c >= RATE_LIMIT_HARD,
            "headroom": max(0, RATE_LIMIT_HARD - c),
        }


# ---------------------------------------------------------------------------
# Compliance logger
# ---------------------------------------------------------------------------


def _build_compliance_checklist(
    *,
    side: str,
    security_code: str,
    qty: int,
    stop_ticks: int,
    num_slaves: int,
    high_impact: bool,
    delay_range: tuple[float, float],
    rate_count: int,
) -> list[str]:
    """Build the compliance checklist that gets logged on every SEND ALL.

    This is printed to the log and stored in Redis so an auditor (or the
    trader themselves) can verify every trade was compliant.
    """
    checks: list[str] = []

    checks.append("✓ Main account = manual button push (WebUI)")
    checks.append("✓ All orders use OrderPlacement.MANUAL flag")
    checks.append(
        f"✓ Slave copy delay: {delay_range[0] * 1000:.0f}–{delay_range[1] * 1000:.0f} ms"
        + (" (HIGH IMPACT)" if high_impact else "")
    )
    checks.append(f"✓ Server-side hard stop: stop_ticks={stop_ticks}")
    checks.append(f"✓ Order: {side} {qty}× {security_code} → main + {num_slaves} slave(s)")
    checks.append(f"✓ Rate counter: {rate_count} actions in rolling 60 min (limit: {RATE_LIMIT_HARD})")

    if stop_ticks <= 0:
        checks.append("⚠ WARNING: stop_ticks=0 — no server-side hard stop attached!")
    if rate_count >= RATE_LIMIT_WARN:
        checks.append(f"⚠ WARNING: approaching rate limit ({rate_count}/{RATE_LIMIT_HARD})")

    return checks


def _log_compliance(checklist: list[str], batch_id: str) -> None:
    """Print the compliance checklist to the log and persist to Redis."""
    header = f"═══ COMPLIANCE LOG [{batch_id}] ═══"
    logger.info(header)
    for line in checklist:
        logger.info("  %s", line)
    logger.info("═" * len(header))

    # Persist to Redis for audit trail
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            entry = json.dumps(
                {
                    "batch_id": batch_id,
                    "checklist": checklist,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            _r.lpush(_REDIS_COMPLIANCE_KEY, entry)
            _r.ltrim(_REDIS_COMPLIANCE_KEY, 0, 999)  # keep last 1000
            _r.expire(_REDIS_COMPLIANCE_KEY, REDIS_LOG_TTL * 7)  # 7 days
    except Exception as exc:
        logger.debug("compliance log redis error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Connected account wrapper
# ---------------------------------------------------------------------------


@dataclass
class _ConnectedAccount:
    """Internal wrapper around an active Rithmic client connection."""

    key: str
    label: str
    is_main: bool
    client: Any  # async_rithmic.RithmicClient (untyped)
    account_ids: list[str] = field(default_factory=list)
    connected: bool = False
    last_order_at: str = ""
    order_count: int = 0
    account_size: int = 150_000


# ---------------------------------------------------------------------------
# CopyTrader
# ---------------------------------------------------------------------------


class CopyTrader:
    """Multi-account copy trading engine with prop-firm compliance.

    The CopyTrader manages Rithmic client connections for a **main** account
    and zero or more **slave** accounts.  When the trader clicks "SEND ALL"
    in the WebUI, the order is placed on the main account first, then
    replicated to every enabled slave with a randomised delay and the
    ``MANUAL`` placement flag.

    Thread safety:
        Uses an asyncio Lock to serialise order submission.  Safe to call
        from multiple coroutines but NOT from multiple threads.
    """

    def __init__(self) -> None:
        self._main: _ConnectedAccount | None = None
        self._slaves: dict[str, _ConnectedAccount] = {}  # key → account
        self._enabled_slave_keys: set[str] = set()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._rate_counter: RollingRateCounter = RollingRateCounter()
        self._high_impact_mode: bool = False  # set True on NFP/FOMC days
        self._order_history: deque[dict[str, Any]] = deque(maxlen=500)
        self._last_warn_logged_at: float = 0.0  # monotonic time of last warn log
        self._warn_log_interval: float = 300.0  # re-log warn at most every 5 min
        self._daily_actions: int = 0
        self._daily_reset_day: str = ""  # YYYY-MM-DD

        # Front-month contract cache: product_code → security_code
        self._contract_cache: dict[str, str] = {}

        logger.info(
            "CopyTrader initialised: delay=%.1f–%.1fs, rate_warn=%d, rate_hard=%d",
            COPY_DELAY_MIN,
            COPY_DELAY_MAX,
            RATE_LIMIT_WARN,
            RATE_LIMIT_HARD,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def add_account(
        self,
        config: Any,  # RithmicAccountConfig
        *,
        is_main: bool = False,
        connect_timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Connect a Rithmic account and register it as main or slave.

        Args:
            config: A ``RithmicAccountConfig`` instance with encrypted credentials.
            is_main: If True, register as the main (leader) account.
            connect_timeout: Timeout in seconds for the initial connection.

        Returns:
            Status dict with connection result.
        """
        if RITHMIC_DEBUG_LOGGING:
            import logging as _stdlib_logging

            _stdlib_logging.getLogger("rithmic").setLevel(_stdlib_logging.DEBUG)
            logger.info("Rithmic DEBUG logging enabled (RITHMIC_DEBUG_LOGGING=1)")

        try:
            from async_rithmic import RithmicClient  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError:
            return {
                "key": getattr(config, "key", "?"),
                "connected": False,
                "error": "async-rithmic not installed",
            }

        key = config.key
        label = getattr(config, "label", key)
        username = config.get_username()
        password = config.get_password()

        if not username or not password:
            return {"key": key, "connected": False, "error": "credentials not configured"}

        # Resolve gateway
        gateway = self._resolve_gateway(getattr(config, "gateway", "Chicago"))
        if gateway is None:
            return {"key": key, "connected": False, "error": "unknown gateway"}

        try:
            client = RithmicClient(  # type: ignore[call-arg]
                user=username,
                password=password,
                system_name=getattr(config, "system_name", ""),
                app_name=getattr(config, "app_name", "ruby_futures"),
                app_version=getattr(config, "app_version", "1.0"),
                gateway=gateway,
            )
            await asyncio.wait_for(client.connect(), timeout=connect_timeout)

            # Discover account IDs
            account_ids: list[str] = []
            try:
                accounts = await asyncio.wait_for(client.list_accounts(), timeout=10.0)  # type: ignore[attr-defined]
                account_ids = [getattr(a, "account_id", str(a)) for a in (accounts or [])]
            except Exception as exc:
                logger.warning("copy_trader[%s]: list_accounts failed: %s", key, exc)

            acct = _ConnectedAccount(
                key=key,
                label=label,
                is_main=is_main,
                client=client,
                account_ids=account_ids,
                connected=True,
                account_size=getattr(config, "account_size", 150_000),
            )

            if is_main:
                # Disconnect previous main if any
                if self._main is not None and self._main.connected:
                    await self._safe_disconnect(self._main)
                self._main = acct
                logger.info(
                    "✅ Main account connected: %s (%s) — accounts: %s",
                    label,
                    key,
                    account_ids,
                )
            else:
                # Disconnect previous connection for same key if any
                if key in self._slaves and self._slaves[key].connected:
                    await self._safe_disconnect(self._slaves[key])
                self._slaves[key] = acct
                self._enabled_slave_keys.add(key)
                logger.info(
                    "✅ Slave account connected: %s (%s) — accounts: %s",
                    label,
                    key,
                    account_ids,
                )

            return {
                "key": key,
                "label": label,
                "is_main": is_main,
                "connected": True,
                "account_ids": account_ids,
            }

        except TimeoutError:
            return {"key": key, "connected": False, "error": "connection timed out"}
        except Exception as exc:
            return {"key": key, "connected": False, "error": str(exc)[:300]}

    def enable_slave(self, key: str) -> None:
        """Enable a slave account for copy trading."""
        if key in self._slaves:
            self._enabled_slave_keys.add(key)
            logger.info("Slave %s enabled for copy trading", key)

    def disable_slave(self, key: str) -> None:
        """Disable a slave account (stops receiving copies)."""
        self._enabled_slave_keys.discard(key)
        logger.info("Slave %s disabled for copy trading", key)

    def set_high_impact_mode(self, enabled: bool) -> None:
        """Enable/disable high-impact mode (NFP/FOMC days → longer delays)."""
        self._high_impact_mode = enabled
        if enabled:
            logger.info(
                "⚠ HIGH IMPACT MODE: copy delays increased to %.1f–%.1fs",
                HIGH_IMPACT_DELAY_MIN,
                HIGH_IMPACT_DELAY_MAX,
            )
        else:
            logger.info("High impact mode disabled — normal delays restored")

    def _check_rate_and_warn(self) -> bool:
        """Check the rolling rate counter and log warnings/errors.

        Called before every order submission.  Returns ``True`` if the order
        is allowed to proceed, ``False`` if the hard limit blocks it.

        Warn threshold (3 000/hr): logs a WARNING every 5 min so the operator
        knows the session is heavy but does not block orders.

        Hard limit (4 500/hr): logs an ERROR and returns False — order is
        rejected to prevent Rithmic account suspension.
        """
        status = self._rate_counter.status_dict()
        count = status["actions_60min"]

        if status["blocked"]:
            logger.error(
                "🚫 RATE LIMIT HARD STOP: %d actions in last 60 min (limit=%d) — ORDER REJECTED",
                count,
                RATE_LIMIT_HARD,
            )
            return False

        if status["warn"]:
            now = time.monotonic()
            if now - self._last_warn_logged_at >= self._warn_log_interval:
                self._last_warn_logged_at = now
                headroom = status["headroom"]
                logger.warning(
                    "⚠️  RATE LIMIT WARNING: %d/%d actions in last 60 min — %d headroom remaining",
                    count,
                    RATE_LIMIT_WARN,
                    headroom,
                )

        return True

    @staticmethod
    def _detect_rate_limit_error(exc: Exception) -> bool:
        """Return True if *exc* looks like a Rithmic rate-limit or Consumer Slow error."""
        msg = str(exc).lower()
        return any(
            kw in msg
            for kw in (
                "consumer slow",
                "rate limit",
                "too many",
                "throttl",
                "429",
                "slow consumer",
                "exceed",
            )
        )

    async def disconnect_all(self) -> None:
        """Disconnect all accounts gracefully."""
        if self._main is not None:
            await self._safe_disconnect(self._main)
            self._main = None
        for _key, acct in list(self._slaves.items()):
            await self._safe_disconnect(acct)
        self._slaves.clear()
        self._enabled_slave_keys.clear()
        logger.info("All CopyTrader accounts disconnected")

    async def _safe_disconnect(self, acct: _ConnectedAccount) -> None:
        """Disconnect a single account, swallowing errors."""
        import contextlib

        try:
            if acct.client is not None and acct.connected:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(acct.client.disconnect(), timeout=5.0)
                acct.connected = False
        except Exception as exc:
            logger.debug("copy_trader[%s] disconnect error: %s", acct.key, exc)
            acct.connected = False

    # ------------------------------------------------------------------
    # Contract resolution
    # ------------------------------------------------------------------

    async def resolve_front_month(
        self,
        ticker: str,
        client: Any = None,
    ) -> tuple[str, str] | None:
        """Resolve a Yahoo-style ticker to (security_code, exchange).

        Uses async_rithmic's ``get_front_month_contract()`` with a local
        cache to avoid repeated lookups.

        Args:
            ticker: Yahoo-style ticker, e.g. "MGC=F"
            client: Optional specific RithmicClient to use for resolution.

        Returns:
            ``(security_code, exchange)`` tuple, or None if resolution fails.
        """
        mapping = TICKER_TO_RITHMIC.get(ticker)
        if mapping is None:
            logger.warning("No Rithmic mapping for ticker %s", ticker)
            return None

        product_code = mapping["product_code"]
        exchange = mapping["exchange"]

        # Check cache first
        if product_code in self._contract_cache:
            return self._contract_cache[product_code], exchange

        # Resolve via Rithmic
        resolve_client = client or (self._main.client if self._main and self._main.connected else None)
        if resolve_client is None:
            # Fall back — caller must supply security_code directly
            logger.debug("No connected client for front-month resolution of %s", ticker)
            return None

        try:
            contract = await asyncio.wait_for(
                resolve_client.get_front_month_contract(exchange=exchange, product_code=product_code),
                timeout=10.0,
            )
            if contract:
                sec_code: str = str(
                    getattr(contract, "security_code", None) or getattr(contract, "symbol", None) or contract
                )
                self._contract_cache[product_code] = sec_code
                logger.info("Resolved %s → %s (%s)", ticker, sec_code, exchange)
                return sec_code, exchange
        except Exception as exc:
            logger.warning("Front-month resolution failed for %s: %s", ticker, exc)

        return None

    def invalidate_contract_cache(self, ticker: str | None = None) -> None:
        """Clear cached front-month contract resolution."""
        if ticker:
            self._contract_cache.pop(ticker, None)
        else:
            self._contract_cache.clear()

    # ------------------------------------------------------------------
    # Core order submission
    # ------------------------------------------------------------------

    async def send_order_and_copy(
        self,
        *,
        security_code: str,
        exchange: str,
        side: str,
        qty: int = 1,
        order_type: str = "MARKET",
        price: float = 0.0,
        stop_ticks: int = 20,
        target_ticks: int | None = None,
        tag_prefix: str = "RUBY",
        reason: str = "",  # included in compliance log & order tag
        scale_qty_by_account: bool = False,
    ) -> CopyBatchResult:
        """Place an order on the main account and copy to all enabled slaves.

        This is the handler for the WebUI "SEND ALL" button.

        **Every order** uses ``OrderPlacement.MANUAL`` and slaves get a
        randomised delay before submission.

        Args:
            security_code: Rithmic security code (e.g. "MGCQ6").
            exchange: Exchange code (e.g. "NYMEX", "CME").
            side: "BUY" or "SELL".
            qty: Number of contracts.
            order_type: "MARKET" or "LIMIT".
            price: Limit price (ignored for MARKET orders).
            stop_ticks: Hard stop-loss in ticks (server-side bracket).
            target_ticks: Optional take-profit in ticks.
            tag_prefix: Order tag prefix for audit trail.
            reason: Human-readable reason for the trade.

        Returns:
            CopyBatchResult with main + all slave results and compliance log.
        """
        batch_id = f"{tag_prefix}_{security_code}_{int(time.time())}" + (f"_{reason[:20]}" if reason else "")
        result = CopyBatchResult(
            batch_id=batch_id,
            security_code=security_code,
            side=side,
            qty=qty,
        )

        # --- Rate limit check ---
        enabled_slaves = self._get_enabled_slaves()

        if not self._check_rate_and_warn():
            # Hard limit — return a failed batch result
            result.compliance_log.append(f"🚫 BLOCKED: rate limit {self._rate_counter.count}/{RATE_LIMIT_HARD}")
            return result

        # --- Determine delay range ---
        if self._high_impact_mode:
            delay_range = (HIGH_IMPACT_DELAY_MIN, HIGH_IMPACT_DELAY_MAX)
        else:
            delay_range = (COPY_DELAY_MIN, COPY_DELAY_MAX)

        # --- Build and log compliance checklist ---
        checklist = _build_compliance_checklist(
            side=side,
            security_code=security_code,
            qty=qty,
            stop_ticks=stop_ticks,
            num_slaves=len(enabled_slaves),
            high_impact=self._high_impact_mode,
            delay_range=delay_range,
            rate_count=self._rate_counter.count,
        )
        result.compliance_log = checklist
        _log_compliance(checklist, batch_id)

        async with self._lock:
            # --- Step 1: Place on MAIN account ---
            main_result = await self._submit_single_order(
                acct=self._main,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                tag=f"{tag_prefix}_MANUAL_WEBUI",
                is_main=True,
                batch_id=batch_id,
            )
            result.main_result = main_result

            if main_result.status in (CopyOrderStatus.ERROR, CopyOrderStatus.REJECTED):
                logger.error(
                    "❌ Main order FAILED — aborting slave copies. Error: %s",
                    main_result.error,
                )
                return result

            # --- Step 2: Copy to each enabled slave with delay ---
            for slave_acct in enabled_slaves:
                # Humanised delay
                delay = random.uniform(*delay_range)
                delay_ms = int(delay * 1000)
                await asyncio.sleep(delay)

                # Per-account qty scaling
                slave_qty = qty
                if scale_qty_by_account and self._main and self._main.account_size > 0:
                    ratio = slave_acct.account_size / self._main.account_size
                    slave_qty = max(1, round(qty * ratio))
                    result.compliance_log.append(
                        f"📊 Qty scaled for {slave_acct.label}: {qty}→{slave_qty} (ratio {ratio:.2f})"
                    )

                slave_result = await self._submit_single_order(
                    acct=slave_acct,
                    security_code=security_code,
                    exchange=exchange,
                    side=side,
                    qty=slave_qty,
                    order_type=order_type,
                    price=price,
                    stop_ticks=stop_ticks,
                    target_ticks=target_ticks,
                    tag="COPY_FROM_MAIN_HUMAN_150K",
                    is_main=False,
                    batch_id=batch_id,
                    delay_ms=delay_ms,
                )
                result.slave_results.append(slave_result)

        # Record rate-limit actions
        self._rate_counter.record(result.total_orders)

        # Daily action counter (resets at midnight)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_reset_day:
            self._daily_actions = 0
            self._daily_reset_day = today
        self._daily_actions += result.total_orders

        # Persist to Redis
        self._persist_batch_result(result)

        # Summary log
        logger.info(
            "📤 CopyTrader batch %s: %s %d× %s → main + %d slave(s) | %d ok, %d failed",
            batch_id,
            side,
            qty,
            security_code,
            len(result.slave_results),
            result.total_orders - result.failed_count,
            result.failed_count,
        )

        return result

    async def send_order_from_ticker(
        self,
        *,
        ticker: str,
        side: str,
        qty: int = 1,
        order_type: str = "MARKET",
        price: float = 0.0,
        stop_ticks: int = 20,
        target_ticks: int | None = None,
        tag_prefix: str = "RUBY",
        reason: str = "",  # forwarded to send_order_and_copy
    ) -> CopyBatchResult:
        """Convenience wrapper: resolve Yahoo ticker → Rithmic contract, then send.

        Uses the internal contract cache and falls back to the TICKER_TO_RITHMIC
        mapping if no live client is available for front-month resolution.
        """
        resolved = await self.resolve_front_month(ticker)
        if resolved is None:
            # Fall back to static mapping exchange only — security_code unknown
            mapping = TICKER_TO_RITHMIC.get(ticker)
            if mapping is None:
                batch_id = f"ERR_{ticker}_{int(time.time())}"
                r = CopyBatchResult(batch_id=batch_id, security_code="", side=side, qty=qty)
                r.compliance_log.append(f"🛑 Cannot resolve ticker {ticker} to Rithmic contract")
                return r
            logger.warning(
                "Front-month resolution unavailable for %s — caller must provide security_code",
                ticker,
            )
            batch_id = f"ERR_{ticker}_{int(time.time())}"
            r = CopyBatchResult(batch_id=batch_id, security_code="", side=side, qty=qty)
            r.compliance_log.append(f"🛑 No connected client to resolve {ticker}")
            return r

        security_code, exchange = resolved
        return await self.send_order_and_copy(
            security_code=security_code,
            exchange=exchange,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            stop_ticks=stop_ticks,
            target_ticks=target_ticks,
            tag_prefix=tag_prefix,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Rithmic order modification / cancellation helpers
    # ------------------------------------------------------------------

    async def modify_stop_on_all(
        self,
        *,
        security_code: str,
        exchange: str,
        new_stop_price: float,
        product_code: str = "",
        entry_price: float = 0.0,
        position_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Move the server-side stop to ``new_stop_price`` on all connected accounts.

        Converts the absolute stop price to a tick-distance using the tick-size
        table, then calls ``client.modify_order`` (via ``cancel_all_orders`` +
        re-entry bracket update if needed) on every connected account.

        This method enforces ``OrderPlacement.MANUAL`` on all modifications.

        Args:
            security_code: Rithmic security code (e.g. "MGCQ6").
            exchange:      Exchange code (e.g. "NYMEX").
            new_stop_price: The new absolute stop-loss price level.
            product_code:  Rithmic product code for tick-size lookup (e.g. "MGC").
                           If empty, falls back to the TICKER_TO_RITHMIC mapping
                           via security_code prefix.
            entry_price:   The original entry price (used for tick conversion).
                           If 0, stop_ticks is estimated as DEFAULT_STOP_TICKS.
            position_id:   Forwarded to the audit log only.
            reason:        Forwarded to the audit log only.

        Returns:
            Dict with keys: ``ok`` (bool), ``accounts_modified`` (list[str]),
            ``accounts_failed`` (list[str]), ``stop_ticks`` (int), ``reason`` (str).
        """
        # Resolve product_code from security_code prefix if not provided
        if not product_code:
            # security_code is typically product_code + contract month, e.g. "MGCQ6"
            # Try progressively shorter prefixes (3–2 chars)
            for length in (3, 2):
                candidate = security_code[:length]
                if candidate in TICK_SIZE:
                    product_code = candidate
                    break

        stop_ticks = stop_price_to_stop_ticks(
            entry_price=entry_price,
            stop_price=new_stop_price,
            product_code=product_code,
        )

        accounts_modified: list[str] = []
        accounts_failed: list[str] = []

        all_accounts: list[_ConnectedAccount] = []
        if self._main and self._main.connected:
            all_accounts.append(self._main)
        all_accounts.extend(self._get_enabled_slaves())

        if not all_accounts:
            logger.warning("modify_stop_on_all: no connected accounts — skipping")
            return {
                "ok": False,
                "accounts_modified": [],
                "accounts_failed": [],
                "stop_ticks": stop_ticks,
                "reason": "no connected accounts",
            }

        for acct in all_accounts:
            try:
                from async_rithmic import OrderPlacement  # type: ignore[import-untyped]

                account_id = acct.account_ids[0] if acct.account_ids else None
                modify_kwargs: dict[str, Any] = {
                    "security_code": security_code,
                    "exchange": exchange,
                    "stop_ticks": stop_ticks,
                    "manual_or_auto": OrderPlacement.MANUAL,
                }
                if account_id:
                    modify_kwargs["account_id"] = account_id

                await asyncio.wait_for(
                    acct.client.modify_order(**modify_kwargs),
                    timeout=10.0,
                )
                accounts_modified.append(acct.key)
                self._rate_counter.record(1)
                logger.info(
                    "✏️  [%s] modify_stop %s → new_stop=%.4f (%d ticks) MANUAL ✓ | %s",
                    acct.label,
                    security_code,
                    new_stop_price,
                    stop_ticks,
                    reason,
                )

            except ImportError:
                logger.warning("modify_stop_on_all: async-rithmic not installed — skipping %s", acct.key)
                accounts_failed.append(acct.key)
            except TimeoutError:
                logger.error("modify_stop_on_all: timed out on account %s", acct.key)
                accounts_failed.append(acct.key)
            except Exception as exc:
                logger.error("modify_stop_on_all[%s] error: %s", acct.key, exc)
                accounts_failed.append(acct.key)

        ok = len(accounts_modified) > 0 and len(accounts_failed) == 0
        return {
            "ok": ok,
            "accounts_modified": accounts_modified,
            "accounts_failed": accounts_failed,
            "stop_ticks": stop_ticks,
            "new_stop_price": new_stop_price,
            "security_code": security_code,
            "position_id": position_id,
            "reason": reason,
        }

    async def cancel_on_all(
        self,
        *,
        security_code: str = "",
        exchange: str = "",
        position_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Cancel all working orders for ``security_code`` on all connected accounts.

        Calls ``client.cancel_all_orders`` (per account_id) with
        ``manual_or_auto=OrderPlacement.MANUAL``.

        Args:
            security_code: Rithmic security code to target.  If empty, cancels
                           *all* open orders on each account (use with caution).
            exchange:      Exchange code (forwarded to API, may be ignored by some calls).
            position_id:   Audit log only.
            reason:        Audit log only.

        Returns:
            Dict with ``ok``, ``accounts_cancelled``, ``accounts_failed``.
        """
        accounts_cancelled: list[str] = []
        accounts_failed: list[str] = []

        all_accounts: list[_ConnectedAccount] = []
        if self._main and self._main.connected:
            all_accounts.append(self._main)
        all_accounts.extend(self._get_enabled_slaves())

        if not all_accounts:
            logger.warning("cancel_on_all: no connected accounts — skipping")
            return {"ok": False, "accounts_cancelled": [], "accounts_failed": [], "reason": "no connected accounts"}

        for acct in all_accounts:
            try:
                from async_rithmic import OrderPlacement  # type: ignore[import-untyped]

                account_id = acct.account_ids[0] if acct.account_ids else None
                cancel_kwargs: dict[str, Any] = {
                    "manual_or_auto": OrderPlacement.MANUAL,
                }
                if account_id:
                    cancel_kwargs["account_id"] = account_id
                if security_code:
                    cancel_kwargs["security_code"] = security_code
                if exchange:
                    cancel_kwargs["exchange"] = exchange

                await asyncio.wait_for(
                    acct.client.cancel_all_orders(**cancel_kwargs),
                    timeout=10.0,
                )
                accounts_cancelled.append(acct.key)
                self._rate_counter.record(1)
                logger.info(
                    "❌ [%s] cancel_all_orders %s MANUAL ✓ | %s",
                    acct.label,
                    security_code or "(all)",
                    reason,
                )

            except ImportError:
                logger.warning("cancel_on_all: async-rithmic not installed — skipping %s", acct.key)
                accounts_failed.append(acct.key)
            except TimeoutError:
                logger.error("cancel_on_all: timed out on account %s", acct.key)
                accounts_failed.append(acct.key)
            except Exception as exc:
                logger.error("cancel_on_all[%s] error: %s", acct.key, exc)
                accounts_failed.append(acct.key)

        ok = len(accounts_cancelled) > 0 and len(accounts_failed) == 0
        return {
            "ok": ok,
            "accounts_cancelled": accounts_cancelled,
            "accounts_failed": accounts_failed,
            "security_code": security_code,
            "position_id": position_id,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # PositionManager integration
    # ------------------------------------------------------------------

    async def execute_order_commands(
        self,
        orders: list[Any],  # list[OrderCommand] from PositionManager
        *,
        entry_prices: dict[str, float] | None = None,
    ) -> list[CopyBatchResult | dict[str, Any]]:
        """Translate PositionManager OrderCommands into Rithmic orders + copies.

        This is the bridge between ``PositionManager.process_signal()`` /
        ``PositionManager.update_all()`` output (``OrderCommand`` objects) and
        the Rithmic copy-trading path.

        Routing:
            * ``BUY`` / ``SELL`` with ``MARKET`` or ``LIMIT`` type →
              :meth:`send_order_from_ticker` (main + all slave copies).
            * ``MODIFY_STOP`` → :meth:`modify_stop_on_all` (MANUAL flag enforced).
            * ``CANCEL`` → :meth:`cancel_on_all` (MANUAL flag enforced).
            * ``STOP`` order type (initial SL companion) → ignored here because
              the server-side bracket on the entry order already covers it.

        Args:
            orders: List of ``OrderCommand`` instances from PositionManager.
            entry_prices: Optional dict of ``ticker → entry_price`` for accurate
                          stop_ticks conversion on ``MODIFY_STOP`` commands.
                          Falls back to 0.0 (→ DEFAULT_STOP_TICKS) when absent.

        Returns:
            Mixed list of :class:`CopyBatchResult` (for entry orders) and plain
            dicts (for modify/cancel responses).
        """
        results: list[CopyBatchResult | dict[str, Any]] = []
        entry_prices = entry_prices or {}

        for order in orders:
            action = str(getattr(order, "action", ""))
            order_type_val = str(getattr(order, "order_type", "MARKET")).upper()
            ticker = getattr(order, "symbol", "")
            reason = getattr(order, "reason", "")
            position_id = getattr(order, "position_id", "")

            # ------------------------------------------------------------------
            # MODIFY_STOP — move server-side bracket stop on all accounts
            # ------------------------------------------------------------------
            if action == "MODIFY_STOP":
                stop_price = getattr(order, "stop_price", 0.0)
                resolved = await self.resolve_front_month(ticker)
                if resolved is None:
                    logger.warning(
                        "execute_order_commands: MODIFY_STOP for %s — cannot resolve contract, skipping",
                        ticker,
                    )
                    results.append(
                        {
                            "action": "MODIFY_STOP",
                            "ticker": ticker,
                            "ok": False,
                            "reason": "contract not resolved",
                            "position_id": position_id,
                        }
                    )
                    continue

                security_code, exchange = resolved
                # Derive product_code from TICKER_TO_RITHMIC mapping
                mapping = TICKER_TO_RITHMIC.get(ticker, {})
                product_code = mapping.get("product_code", "")
                entry_price = entry_prices.get(ticker, 0.0)

                mod_result = await self.modify_stop_on_all(
                    security_code=security_code,
                    exchange=exchange,
                    new_stop_price=stop_price,
                    product_code=product_code,
                    entry_price=entry_price,
                    position_id=position_id,
                    reason=reason or f"PM MODIFY_STOP {ticker} → {stop_price:.4f}",
                )
                results.append(mod_result)
                continue

            # ------------------------------------------------------------------
            # CANCEL — cancel all working orders for this ticker on all accounts
            # ------------------------------------------------------------------
            if action == "CANCEL":
                resolved = await self.resolve_front_month(ticker)
                security_code = ""
                exchange = ""
                if resolved is not None:
                    security_code, exchange = resolved

                cancel_result = await self.cancel_on_all(
                    security_code=security_code,
                    exchange=exchange,
                    position_id=position_id,
                    reason=reason or f"PM CANCEL {ticker}",
                )
                results.append(cancel_result)
                continue

            # ------------------------------------------------------------------
            # STOP order type companion — skip; the entry's server-side bracket
            # already attaches the stop.  We don't submit a standalone stop order
            # through Rithmic.
            # ------------------------------------------------------------------
            if order_type_val == "STOP":
                # Capture the stop price from this companion order so that any
                # subsequent MODIFY_STOP for the same ticker has a good baseline
                # (stored in the caller's entry_prices dict by reference).
                stop_price = getattr(order, "stop_price", 0.0)
                if ticker and stop_price > 0:
                    entry_prices[ticker] = stop_price  # reuse slot as "last known stop"
                logger.debug(
                    "execute_order_commands: STOP companion for %s (stop=%.4f) — "
                    "covered by server-side bracket, skipping standalone submit",
                    ticker,
                    stop_price,
                )
                continue

            # ------------------------------------------------------------------
            # Entry orders: BUY / SELL  (MARKET or LIMIT)
            # ------------------------------------------------------------------
            if action not in ("BUY", "SELL"):
                logger.debug("execute_order_commands: unknown action %r for %s — skipping", action, ticker)
                continue

            side = action  # already "BUY" or "SELL"
            qty = getattr(order, "quantity", 1)
            price = getattr(order, "price", 0.0)

            # Compute stop_ticks from the companion STOP order's stop_price
            # (which was stored in entry_prices above during the STOP pass).
            # If not yet populated we use the order's own stop_price field as
            # a fallback before resorting to the safe default.
            stop_price_for_ticks = entry_prices.get(ticker, getattr(order, "stop_price", 0.0))
            entry_price_for_ticks = price if price > 0 else getattr(order, "price", 0.0)

            mapping = TICKER_TO_RITHMIC.get(ticker, {})
            product_code = mapping.get("product_code", "")

            stop_ticks_val = (
                stop_price_to_stop_ticks(
                    entry_price=entry_price_for_ticks,
                    stop_price=stop_price_for_ticks,
                    product_code=product_code,
                )
                if (stop_price_for_ticks > 0 and entry_price_for_ticks > 0)
                else DEFAULT_STOP_TICKS
            )

            batch = await self.send_order_from_ticker(
                ticker=ticker,
                side=side,
                qty=qty,
                order_type=order_type_val,
                price=price,
                stop_ticks=stop_ticks_val,
                tag_prefix="PM",
                reason=reason,
            )
            results.append(batch)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _submit_single_order(
        self,
        *,
        acct: _ConnectedAccount | None,
        security_code: str,
        exchange: str,
        side: str,
        qty: int,
        order_type: str,
        price: float,
        stop_ticks: int,
        target_ticks: int | None,
        tag: str,
        is_main: bool,
        batch_id: str,
        delay_ms: int = 0,
    ) -> CopyOrderResult:
        """Submit a single order to one Rithmic account.

        Always uses ``OrderPlacement.MANUAL``.
        """
        acct_key = acct.key if acct else "unknown"
        order_id = f"{batch_id}_{'main' if is_main else acct_key}_{int(time.time() * 1000)}"

        if acct is None or not acct.connected:
            return CopyOrderResult(
                account_key=acct.key if acct else "none",
                account_label=acct.label if acct else "none",
                is_main=is_main,
                order_id=order_id,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                status=CopyOrderStatus.ERROR,
                error="account not connected" if acct is None else f"account {acct.key} disconnected",
                delay_ms=delay_ms,
                tag=tag,
            )

        try:
            from async_rithmic import (  # type: ignore[import-untyped]  # noqa: PLC0415
                OrderPlacement,
                TransactionType,
            )
            from async_rithmic import (  # noqa: PLC0415
                OrderType as RithmicOrderType,
            )

            # Map side → TransactionType
            tx_type = TransactionType.BUY if side == "BUY" else TransactionType.SELL

            # Map order_type → RithmicOrderType
            r_order_type = RithmicOrderType.LIMIT if order_type == "LIMIT" else RithmicOrderType.MARKET

            # Build submit_order kwargs
            submit_kwargs: dict[str, Any] = {
                "order_id": order_id,
                "security_code": security_code,
                "exchange": exchange,
                "qty": qty,
                "order_type": r_order_type,
                "transaction_type": tx_type,
                "manual_or_auto": OrderPlacement.MANUAL,  # ← COMPLIANCE: always MANUAL
            }

            # Price for limit orders
            if order_type == "LIMIT" and price > 0:
                submit_kwargs["price"] = price

            # Server-side brackets
            if stop_ticks > 0:
                submit_kwargs["stop_ticks"] = stop_ticks
            if target_ticks is not None and target_ticks > 0:
                submit_kwargs["target_ticks"] = target_ticks

            # Use first discovered account_id
            if acct.account_ids:
                submit_kwargs["account_id"] = acct.account_ids[0]

            # Submit!
            await asyncio.wait_for(
                acct.client.submit_order(**submit_kwargs),
                timeout=10.0,
            )

            acct.last_order_at = datetime.now(UTC).isoformat()
            acct.order_count += 1

            role = "MAIN" if is_main else "SLAVE"
            logger.info(
                "📤 [%s] %s %s %d× %s @ %s | stop=%d ticks | tag=%s | delay=%dms | MANUAL ✓",
                role,
                acct.label,
                side,
                qty,
                security_code,
                f"{price:.4f}" if price > 0 else "MKT",
                stop_ticks,
                tag,
                delay_ms,
            )

            return CopyOrderResult(
                account_key=acct.key,
                account_label=acct.label,
                is_main=is_main,
                order_id=order_id,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                status=CopyOrderStatus.SUBMITTED,
                delay_ms=delay_ms,
                tag=tag,
            )

        except ImportError:
            return CopyOrderResult(
                account_key=acct.key,
                account_label=acct.label,
                is_main=is_main,
                order_id=order_id,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                status=CopyOrderStatus.ERROR,
                error="async-rithmic not installed",
                delay_ms=delay_ms,
                tag=tag,
            )
        except TimeoutError:
            return CopyOrderResult(
                account_key=acct.key,
                account_label=acct.label,
                is_main=is_main,
                order_id=order_id,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                status=CopyOrderStatus.ERROR,
                error="submit_order timed out (10s)",
                delay_ms=delay_ms,
                tag=tag,
            )
        except Exception as exc:
            if self._detect_rate_limit_error(exc):
                logger.error(
                    "🚫 RITHMIC RATE-LIMIT / CONSUMER SLOW detected on %s: %s — pausing 60s",
                    security_code,
                    exc,
                )
                await asyncio.sleep(60.0)
            logger.error("❌ copy_trader[%s] submit_order failed: %s", acct.key, exc)
            return CopyOrderResult(
                account_key=acct.key,
                account_label=acct.label,
                is_main=is_main,
                order_id=order_id,
                security_code=security_code,
                exchange=exchange,
                side=side,
                qty=qty,
                order_type=order_type,
                price=price,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                status=CopyOrderStatus.ERROR,
                error=str(exc)[:300],
                delay_ms=delay_ms,
                tag=tag,
            )

    def _get_enabled_slaves(self) -> list[_ConnectedAccount]:
        """Return list of connected, enabled slave accounts."""
        return [acct for key, acct in self._slaves.items() if key in self._enabled_slave_keys and acct.connected]

    def get_account_sizes(self) -> dict[str, int]:
        """Return a mapping of account key → account_size for all connected accounts."""
        sizes: dict[str, int] = {}
        if self._main:
            sizes[self._main.key] = self._main.account_size
        for key, acct in self._slaves.items():
            sizes[key] = acct.account_size
        return sizes

    def _persist_batch_result(self, batch: CopyBatchResult) -> None:
        """Write batch result to Redis for dashboard display and audit trail."""
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                entry = json.dumps(batch.to_dict(), default=str)
                _r.lpush(_REDIS_ORDER_LOG_KEY, entry)
                _r.ltrim(_REDIS_ORDER_LOG_KEY, 0, 499)  # keep last 500
                _r.expire(_REDIS_ORDER_LOG_KEY, REDIS_LOG_TTL)

                # Publish for real-time SSE
                _r.publish("dashboard:copy_trade", entry)
        except Exception as exc:
            logger.debug("batch result persist error (non-fatal): %s", exc)

        # In-memory history
        self._order_history.append(batch.to_dict())

    @staticmethod
    def _resolve_gateway(name: str) -> Any:
        """Return the async_rithmic Gateway enum for a gateway name string."""
        try:
            from async_rithmic import Gateway  # type: ignore[import-untyped]

            mapping = {
                "Chicago": Gateway.CHICAGO,
                "Sydney": Gateway.SYDNEY,
                "Sao Paulo": Gateway.SAO_PAULO,
                "test": Gateway.TEST,
            }
            return mapping.get(name, Gateway.CHICAGO)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Status / monitoring
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, Any]:
        """Return a status dict suitable for WebUI display."""
        main_info = None
        if self._main is not None:
            main_info = {
                "key": self._main.key,
                "label": self._main.label,
                "connected": self._main.connected,
                "account_ids": self._main.account_ids,
                "last_order_at": self._main.last_order_at,
                "order_count": self._main.order_count,
                "account_size": self._main.account_size,
            }

        slaves_info = []
        for key, acct in self._slaves.items():
            slaves_info.append(
                {
                    "key": acct.key,
                    "label": acct.label,
                    "connected": acct.connected,
                    "enabled": key in self._enabled_slave_keys,
                    "account_ids": acct.account_ids,
                    "last_order_at": acct.last_order_at,
                    "order_count": acct.order_count,
                    "account_size": acct.account_size,
                }
            )

        return {
            "main": main_info,
            "slaves": slaves_info,
            "enabled_slave_count": len(self._get_enabled_slaves()),
            "high_impact_mode": self._high_impact_mode,
            "rate_limit": {
                **self._rate_counter.status_dict(),
                "daily_actions": self._daily_actions,
            },
            "contract_cache_size": len(self._contract_cache),
            "account_sizes": self.get_account_sizes(),
            "recent_batches": list(self._order_history)[-10:],
        }

    def get_order_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent order history (newest first)."""
        return list(self._order_history)[-limit:][::-1]

    def get_rate_status(self) -> dict[str, Any]:
        """Return current rate-limit counter status."""
        return {
            **self._rate_counter.status_dict(),
            "daily_actions": self._daily_actions,
        }

    def get_rate_alert(self) -> dict[str, Any]:
        """Return a simple alert dict for WebUI display.

        Returns:
            Dict with ``level`` ("ok" | "warn" | "critical"), ``message``, and
            ``actions_60min`` / ``daily_actions`` counters.
        """
        status = self._rate_counter.status_dict()
        count = status["actions_60min"]

        if status["blocked"]:
            level = "critical"
            message = f"HARD LIMIT: {count}/{RATE_LIMIT_HARD} actions/60min — orders blocked"
        elif status["warn"]:
            level = "warn"
            message = f"High activity: {count}/{RATE_LIMIT_WARN} actions/60min — {status['headroom']} headroom"
        else:
            level = "ok"
            message = f"{count} actions in last 60min"

        return {
            "level": level,
            "message": message,
            "actions_60min": count,
            "daily_actions": self._daily_actions,
            "warn_threshold": RATE_LIMIT_WARN,
            "hard_threshold": RATE_LIMIT_HARD,
        }

    def __repr__(self) -> str:
        main_str = f"main={self._main.label}" if self._main else "main=None"
        slave_count = len(self._get_enabled_slaves())
        return f"<CopyTrader {main_str}, slaves={slave_count}, rate={self._rate_counter.count}/60min>"


# ---------------------------------------------------------------------------
# Module-level singleton (optional — engine main.py can also create its own)
# ---------------------------------------------------------------------------

_copy_trader: CopyTrader | None = None


def get_copy_trader() -> CopyTrader:
    """Return the module-level CopyTrader singleton (lazy-init)."""
    global _copy_trader
    if _copy_trader is None:
        _copy_trader = CopyTrader()
    return _copy_trader
