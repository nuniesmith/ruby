"""
Risk Rules Engine
==============================
Automated risk checks that feed into the dashboard and alerts.

Rules:
  - Max 2 open trades at once
  - Max risk per trade: $375 (0.75% of $50k)
  - Max daily loss: $500
  - No new entries after 10:00 AM ET
  - No overnight positions (force warning at 11:30 AM)
  - Micro contract stacking: add only if +0.5R and wave > 1.8x

Public API:
    rm = RiskManager(account_size=50_000)
    ok, reason = rm.can_enter_trade("MGC", "LONG", 2)
    status = rm.get_status()

The status dict is published to Redis for dashboard consumption.

Usage:
    from lib.services.engine.risk import RiskManager

    rm = RiskManager(account_size=50_000)
    allowed, reason = rm.can_enter_trade("MGC", "LONG", 2)
    if not allowed:
        print(f"Trade blocked: {reason}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
from zoneinfo import ZoneInfo

logger = logging.getLogger("engine.risk")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Default risk parameters
# ---------------------------------------------------------------------------
DEFAULT_ACCOUNT_SIZE = 150_000
DEFAULT_RISK_PCT_PER_TRADE = 0.0075  # 0.75%
DEFAULT_MAX_DAILY_LOSS = -3300.0  # TPT $150K daily loss limit
DEFAULT_MAX_OPEN_TRADES = 2
DEFAULT_NO_ENTRY_AFTER = dt_time(10, 0)  # 10:00 AM ET
DEFAULT_OVERNIGHT_WARNING = dt_time(11, 30)  # 11:30 AM ET — warn about overnight
DEFAULT_SESSION_END = dt_time(12, 0)  # 12:00 PM ET — session closes
DEFAULT_STACK_MIN_R = 0.5  # must be +0.5R before stacking
DEFAULT_STACK_MIN_WAVE = 1.8  # wave ratio > 1.8x to allow stacking

# ---------------------------------------------------------------------------
# TPT (Take Profit Trader) $150K account constants
# ---------------------------------------------------------------------------
TPT_PROFIT_TARGET = 9000
TPT_MAX_POSITION_SIZE = 15
TPT_EOD_TRAILING_DRAWDOWN = 4500
DAILY_PROFIT_GOAL_MIN = 500
DAILY_PROFIT_GOAL_MAX = 1800


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPTAccountRules:
    """Take Profit Trader $150K account rules."""

    account_size: int = 150_000
    profit_target: int = 9_000
    max_position_size: int = 15
    daily_loss_limit: float = 3_300.0
    eod_trailing_drawdown: float = 4_500.0
    daily_profit_goal_min: float = 500.0
    daily_profit_goal_max: float = 1_800.0


TPT_RULES = TPTAccountRules()


@dataclass
class TradeRecord:
    """Record of a single trade for daily P&L tracking."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    exit_price: float | None = None
    pnl: float = 0.0
    is_win: bool = False
    closed: bool = False
    opened_at: str | None = None
    closed_at: str | None = None


@dataclass
class RiskState:
    """Current risk state snapshot — published to Redis for dashboard."""

    account_size: int = DEFAULT_ACCOUNT_SIZE
    max_risk_per_trade: float = 0.0
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS
    max_open_trades: int = DEFAULT_MAX_OPEN_TRADES

    # Current state
    open_trade_count: int = 0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    total_risk_exposure: float = 0.0
    risk_pct_of_account: float = 0.0

    # Flags
    can_trade: bool = True
    block_reason: str = ""
    is_past_entry_cutoff: bool = False
    is_overnight_warning: bool = False
    is_daily_loss_exceeded: bool = False
    is_max_trades_reached: bool = False

    # Timestamps
    last_check: str = ""
    last_trade: str = ""
    trading_date: str = ""


class RiskManager:
    """Automated risk rules engine for Ruby Futures.

    Tracks open positions, daily P&L, and enforces risk rules to prevent
    overtrading and excessive drawdown.

    All time-based checks use Eastern Time (ET).
    """

    def __init__(
        self,
        account_size: int = DEFAULT_ACCOUNT_SIZE,
        risk_pct_per_trade: float = DEFAULT_RISK_PCT_PER_TRADE,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_open_trades: int = DEFAULT_MAX_OPEN_TRADES,
        no_entry_after: dt_time = DEFAULT_NO_ENTRY_AFTER,
        overnight_warning: dt_time = DEFAULT_OVERNIGHT_WARNING,
        session_end: dt_time = DEFAULT_SESSION_END,
        stack_min_r: float = DEFAULT_STACK_MIN_R,
        stack_min_wave: float = DEFAULT_STACK_MIN_WAVE,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.account_size = account_size
        self.risk_pct_per_trade = risk_pct_per_trade
        self.max_risk_per_trade = account_size * risk_pct_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_open_trades = max_open_trades
        self.no_entry_after = no_entry_after
        self.overnight_warning = overnight_warning
        self.session_end = session_end
        self.stack_min_r = stack_min_r
        self.stack_min_wave = stack_min_wave

        # Injected clock for testability; defaults to real time
        self._now_fn = now_fn or (lambda: datetime.now(tz=_EST))

        # State
        self._open_positions: dict[str, dict[str, Any]] = {}  # symbol -> position info
        self._closed_trades: list[TradeRecord] = []
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._consecutive_losses: int = 0
        self._trading_date: str | None = None
        self._last_trade_time: str | None = None

        # Initialize for today
        self._ensure_day_reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_enter_trade(
        self,
        symbol: str,
        side: str,
        size: int = 1,
        risk_per_contract: float = 0.0,
        is_stack: bool = False,
        wave_ratio: float = 1.0,
        unrealized_r: float = 0.0,
    ) -> tuple[bool, str]:
        """Check if a new trade entry is allowed.

        Args:
            symbol: Instrument symbol (e.g. "MGC", "MNQ")
            side: "LONG" or "SHORT"
            size: Number of contracts
            risk_per_contract: Dollar risk per contract (stop distance × point value)
            is_stack: True if this is adding to an existing position
            wave_ratio: Current wave ratio for the asset (for stacking check)
            unrealized_r: Current unrealized P&L in R-multiples (for stacking check)

        Returns:
            (allowed, reason) — True if trade is permitted, else False with reason.
        """
        self._ensure_day_reset()
        now = self._now_fn()
        now_time = now.time()
        reasons: list[str] = []

        # Rule 1: Max daily loss
        if self._daily_pnl <= self.max_daily_loss:
            reasons.append(f"Daily loss limit hit (${self._daily_pnl:,.2f} <= ${self.max_daily_loss:,.2f})")

        # Rule 2: Max open trades
        current_open = len(self._open_positions)
        if not is_stack and current_open >= self.max_open_trades:
            reasons.append(f"Max open trades reached ({current_open}/{self.max_open_trades})")

        # Rule 3: Per-trade risk limit
        total_risk = risk_per_contract * size if risk_per_contract > 0 else 0
        if total_risk > self.max_risk_per_trade:
            reasons.append(f"Risk ${total_risk:,.2f} exceeds per-trade max ${self.max_risk_per_trade:,.2f}")

        # Rule 4: No new entries after cutoff time
        if now_time >= self.no_entry_after and now_time < self.session_end:
            reasons.append(f"Past entry cutoff ({self.no_entry_after.strftime('%I:%M %p')} ET)")

        # Rule 5: Session ended
        if now_time >= self.session_end:
            reasons.append("Trading session has ended")

        # Rule 6: Micro contract stacking rules
        if is_stack:
            if unrealized_r < self.stack_min_r:
                reasons.append(f"Cannot stack: unrealized {unrealized_r:.2f}R < required +{self.stack_min_r}R")
            if wave_ratio < self.stack_min_wave:
                reasons.append(f"Cannot stack: wave ratio {wave_ratio:.2f}x < required {self.stack_min_wave}x")

        # Rule 7: Consecutive losses circuit breaker
        if self._consecutive_losses >= 3:
            reasons.append(f"Circuit breaker: {self._consecutive_losses} consecutive losses — take a break")

        # Rule 8: TPT max position size (total contracts across all positions)
        total_contracts = sum(p.get("quantity", 0) for p in self._open_positions.values()) + size
        if total_contracts > TPT_RULES.max_position_size:
            reasons.append(
                f"TPT max position size: {total_contracts} contracts would exceed {TPT_RULES.max_position_size} limit"
            )

        if reasons:
            combined = "; ".join(reasons)
            logger.warning("Trade BLOCKED: %s %s %dx — %s", side, symbol, size, combined)
            return False, combined

        logger.info("Trade ALLOWED: %s %s %dx", side, symbol, size)
        return True, ""

    def check_overnight_risk(self) -> tuple[bool, str]:
        """Check if any open positions face overnight risk.

        Should be called during active session approaching session end.

        Returns:
            (has_risk, warning_message)
        """
        self._ensure_day_reset()
        now = self._now_fn()
        now_time = now.time()

        if not self._open_positions:
            return False, ""

        if now_time >= self.overnight_warning:
            symbols = ", ".join(self._open_positions.keys())
            count = len(self._open_positions)
            msg = (
                f"⚠️ OVERNIGHT WARNING: {count} open position(s) [{symbols}] — "
                f"session ends at {self.session_end.strftime('%I:%M %p')} ET. "
                f"Close or set protective stops."
            )
            logger.warning(msg)
            return True, msg

        return False, ""

    def get_status(self, now: datetime | None = None) -> dict[str, Any]:
        """Return a comprehensive risk status dict for dashboard display.

        This is published to Redis and consumed by the data-service
        positions panel and risk indicators.
        """
        self._ensure_day_reset()
        if now is None:
            now = self._now_fn()
        now_time = now.time()

        total_exposure = sum(pos.get("risk_dollars", 0) for pos in self._open_positions.values())
        risk_pct = (total_exposure / self.account_size * 100) if self.account_size else 0

        is_past_cutoff = now_time >= self.no_entry_after and now_time < self.session_end
        is_overnight = now_time >= self.overnight_warning
        is_daily_exceeded = self._daily_pnl <= self.max_daily_loss
        is_max_trades = len(self._open_positions) >= self.max_open_trades

        can_trade = not (is_past_cutoff or is_daily_exceeded or is_max_trades)

        # Build block reason for dashboard display
        block_reasons = []
        if is_daily_exceeded:
            block_reasons.append("daily loss limit")
        if is_past_cutoff:
            block_reasons.append("past entry cutoff")
        if is_max_trades:
            block_reasons.append("max trades reached")
        if self._consecutive_losses >= 3:
            block_reasons.append("consecutive losses")
            can_trade = False

        return {
            "account_size": self.account_size,
            "max_risk_per_trade": round(self.max_risk_per_trade, 2),
            "max_daily_loss": self.max_daily_loss,
            "max_open_trades": self.max_open_trades,
            "open_trade_count": len(self._open_positions),
            "open_positions": {
                sym: {
                    "side": pos.get("side", "?"),
                    "quantity": pos.get("quantity", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "risk_dollars": pos.get("risk_dollars", 0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                }
                for sym, pos in self._open_positions.items()
            },
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trade_count": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "total_risk_exposure": round(total_exposure, 2),
            "risk_pct_of_account": round(risk_pct, 2),
            "can_trade": can_trade,
            "block_reason": "; ".join(block_reasons) if block_reasons else "",
            "is_past_entry_cutoff": is_past_cutoff,
            "is_overnight_warning": is_overnight and len(self._open_positions) > 0,
            "is_daily_loss_exceeded": is_daily_exceeded,
            "is_max_trades_reached": is_max_trades,
            "last_check": now.isoformat(),
            "last_trade": self._last_trade_time or "",
            "trading_date": self._trading_date or "",
            "rules": {
                "risk_pct_per_trade": self.risk_pct_per_trade,
                "no_entry_after": self.no_entry_after.strftime("%H:%M"),
                "overnight_warning": self.overnight_warning.strftime("%H:%M"),
                "session_end": self.session_end.strftime("%H:%M"),
                "stack_min_r": self.stack_min_r,
                "stack_min_wave": self.stack_min_wave,
            },
        }

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def register_open(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        risk_dollars: float = 0.0,
    ) -> None:
        """Register a newly opened position.

        Called when Rithmic reports a new fill, or when the engine
        opens a simulated position.
        """
        self._ensure_day_reset()
        now = self._now_fn()

        self._open_positions[symbol] = {
            "side": side.upper(),
            "quantity": quantity,
            "entry_price": entry_price,
            "risk_dollars": risk_dollars,
            "unrealized_pnl": 0.0,
            "opened_at": now.isoformat(),
        }
        self._daily_trade_count += 1
        self._last_trade_time = now.isoformat()

        logger.info(
            "Position opened: %s %s %dx @ %.2f (risk $%.2f)",
            side,
            symbol,
            quantity,
            entry_price,
            risk_dollars,
        )

    def register_close(
        self,
        symbol: str,
        exit_price: float,
        realized_pnl: float,
    ) -> None:
        """Register a position close and update daily P&L.

        Args:
            symbol: Instrument symbol
            exit_price: Exit fill price
            realized_pnl: Realized P&L in dollars (positive = profit)
        """
        self._ensure_day_reset()
        now = self._now_fn()

        pos = self._open_positions.pop(symbol, None)

        # Update daily P&L
        self._daily_pnl += realized_pnl

        # Track win/loss streak
        is_win = realized_pnl >= 0
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._last_trade_time = now.isoformat()

        # Record the trade
        if pos:
            self._closed_trades.append(
                TradeRecord(
                    symbol=symbol,
                    side=pos.get("side", "?"),
                    quantity=pos.get("quantity", 0),
                    entry_price=pos.get("entry_price", 0),
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    is_win=is_win,
                    closed=True,
                    opened_at=pos.get("opened_at"),
                    closed_at=now.isoformat(),
                )
            )

        logger.info(
            "Position closed: %s @ %.2f — P&L $%.2f (%s) | daily $%.2f | streak %d",
            symbol,
            exit_price,
            realized_pnl,
            "WIN" if is_win else "LOSS",
            self._daily_pnl,
            self._consecutive_losses,
        )

    def update_account_size(self, new_size: int) -> None:
        """Update account size and recalculate derived risk parameters."""
        self.account_size = new_size
        self.max_risk_per_trade = new_size * self.risk_pct_per_trade
        # Scale daily loss limit proportionally (1% of account)
        self.max_daily_loss = -(new_size * 0.01)
        logger.info(
            "risk_manager_account_size_updated, account_size=%d, max_risk=%.2f, max_daily_loss=%.2f",
            new_size,
            self.max_risk_per_trade,
            self.max_daily_loss,
        )

    def update_unrealized(self, symbol: str, unrealized_pnl: float) -> None:
        """Update the unrealized P&L for an open position.

        Called when Rithmic pushes updated position data.
        """
        if symbol in self._open_positions:
            self._open_positions[symbol]["unrealized_pnl"] = unrealized_pnl

    def sync_positions(self, positions: list[dict[str, Any]]) -> None:
        """Sync open positions from Rithmic snapshot.

        This replaces the internal position state with the latest from
        Rithmic, handling any positions that were opened/closed outside
        the engine's knowledge.
        """
        self._ensure_day_reset()
        now = self._now_fn()

        new_symbols = {p.get("symbol", ""): p for p in positions if p.get("symbol")}
        old_symbols = set(self._open_positions.keys())
        current_symbols = set(new_symbols.keys())

        # Positions that appeared (opened outside engine)
        for sym in current_symbols - old_symbols:
            p = new_symbols[sym]
            self._open_positions[sym] = {
                "side": p.get("side", "UNKNOWN"),
                "quantity": p.get("quantity", 0),
                "entry_price": p.get("avgPrice", p.get("entry_price", 0)),
                "risk_dollars": 0.0,  # unknown risk for externally opened
                "unrealized_pnl": p.get("unrealizedPnL", 0),
                "opened_at": now.isoformat(),
            }
            logger.info("Synced new position: %s", sym)

        # Positions that disappeared (closed outside engine)
        for sym in old_symbols - current_symbols:
            pos = self._open_positions.pop(sym, None)
            if pos:
                logger.info("Position removed during sync: %s (closed externally)", sym)

        # Update unrealized P&L for existing positions
        for sym in current_symbols & old_symbols:
            p = new_symbols[sym]
            self._open_positions[sym]["unrealized_pnl"] = p.get("unrealizedPnL", 0)
            self._open_positions[sym]["quantity"] = p.get("quantity", 0)

    # ------------------------------------------------------------------
    # Publishing to Redis
    # ------------------------------------------------------------------

    def publish_to_redis(self) -> bool:
        """Write risk status to Redis for data-service to serve.

        Writes to:
          - `engine:risk_status` — full JSON payload (TTL 60s)
          - Redis PubSub `dashboard:risk` — trigger for SSE push

        Returns True on success.
        """
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
        except ImportError:
            logger.error("Cannot import cache module")
            return False

        status = self.get_status()

        # Update Prometheus P&L and consecutive-loss gauges
        try:
            from lib.services.data.api.metrics import update_consecutive_losses, update_daily_pnl

            update_daily_pnl(status.get("daily_pnl", 0.0))
            update_consecutive_losses(status.get("consecutive_losses", 0))
        except Exception:
            pass

        try:
            payload_json = json.dumps(status, default=str, allow_nan=False)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialize risk status: %s", exc)
            return False

        try:
            cache_set("engine:risk_status", payload_json.encode(), ttl=60)

            if REDIS_AVAILABLE and _r is not None:
                try:
                    _r.publish("dashboard:risk", payload_json)

                    # Publish positions update trigger
                    if self._open_positions:
                        _r.publish("dashboard:live", payload_json)

                    # Publish risk warning if can't trade
                    if not status["can_trade"]:
                        _r.publish(
                            "dashboard:risk_warning",
                            json.dumps(
                                {
                                    "can_trade": False,
                                    "reason": status["block_reason"],
                                    "daily_pnl": status["daily_pnl"],
                                    "ts": status["last_check"],
                                }
                            ),
                        )

                    # Overnight warning
                    if status["is_overnight_warning"]:
                        _r.publish(
                            "dashboard:overnight_warning",
                            json.dumps(
                                {
                                    "warning": True,
                                    "open_positions": list(self._open_positions.keys()),
                                    "ts": status["last_check"],
                                }
                            ),
                        )
                except Exception as exc:
                    logger.debug("Redis PubSub publish failed (non-fatal): %s", exc)

            logger.debug("Risk status published to Redis")
            return True

        except Exception as exc:
            logger.error("Failed to publish risk to Redis: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_day_reset(self) -> None:
        """Reset daily counters if the trading date has changed."""
        now = self._now_fn()
        today = now.strftime("%Y-%m-%d")

        if self._trading_date != today:
            if self._trading_date is not None:
                logger.info(
                    "Day change: %s → %s — resetting daily risk counters",
                    self._trading_date,
                    today,
                )
            self._trading_date = today
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._consecutive_losses = 0
            self._closed_trades.clear()
            # Note: open positions are NOT cleared on day change —
            # overnight positions carry over until explicitly closed.

    @property
    def daily_pnl(self) -> float:
        """Current daily realized P&L."""
        self._ensure_day_reset()
        return self._daily_pnl

    @property
    def open_positions(self) -> dict[str, dict[str, Any]]:
        """Current open positions dict."""
        return dict(self._open_positions)

    @property
    def open_trade_count(self) -> int:
        """Number of currently open positions."""
        return len(self._open_positions)

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss streak."""
        self._ensure_day_reset()
        return self._consecutive_losses

    @property
    def closed_trades(self) -> list[TradeRecord]:
        """Closed trades for today."""
        self._ensure_day_reset()
        return list(self._closed_trades)

    def get_drawdown_status(self) -> dict[str, Any]:
        """Get current drawdown status relative to TPT EOD trailing drawdown."""
        return {
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "eod_trailing_drawdown": TPT_RULES.eod_trailing_drawdown,
            "pnl_vs_goal_min": self._daily_pnl - TPT_RULES.daily_profit_goal_min,
            "pnl_vs_goal_max": self._daily_pnl - TPT_RULES.daily_profit_goal_max,
            "at_daily_goal": self._daily_pnl >= TPT_RULES.daily_profit_goal_min,
            "at_stretch_goal": self._daily_pnl >= TPT_RULES.daily_profit_goal_max,
            "profit_target_progress": (self._daily_pnl / TPT_RULES.profit_target * 100)
            if TPT_RULES.profit_target > 0
            else 0,
            "risk_level": "green"
            if self._daily_pnl > 0
            else ("amber" if self._daily_pnl > self.max_daily_loss * 0.5 else "red"),
        }
