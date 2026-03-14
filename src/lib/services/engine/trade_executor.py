"""
Trade Executor — Staged Position Builder with Stop-Hunt Protection
===================================================================
Smart order execution for liquid micro futures. Instead of the simple
"market order + immediate hard stop" pattern, this executor:

    Phase 1 — SCOUT:
        Place initial market order (1 contract) to establish position.
        NO stop-loss yet. System monitors price action.

    Phase 2 — BUILD:
        Place limit orders at plan levels to accumulate position.
        Still no hard stop. Monitor for plan invalidation.

    Phase 3 — PROTECT:
        Once price breaks the first target level, set a stop at
        breakeven or a safe level behind the entry zone.

    Phase 4 — MANAGE:
        Take partial profits on spikes/extensions.
        Trail remaining position with plan-aware logic.
        Add more on quality pullbacks to entry zone.

Architecture:
    The trader reviews the morning plan, selects a trade, and hits
    "ENGAGE TRADE". The TradeExecutor then manages the staged entry
    using the plan's levels and the Rithmic MANUAL-flagged orders.

    Every order goes through CopyTrader.send_order_and_copy() with
    OrderPlacement.MANUAL — the human initiated the trade via WebUI.

TPT $150K Account Rules:
    - Max 15 contracts total
    - $3,300 daily loss limit
    - $4,500 EOD trailing drawdown
    - Target: $500-$1,800/day
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from lib.services.engine.copy_trader import stop_price_to_stop_ticks

logger = logging.getLogger("engine.trade_executor")


# ---------------------------------------------------------------------------
# Trade phases
# ---------------------------------------------------------------------------


class TradePhase(StrEnum):
    """Phases of a staged trade execution."""

    PLANNED = "planned"  # Trade identified from plan, not yet entered
    SCOUT = "scout"  # Initial small entry placed, monitoring
    BUILD = "build"  # Building position with limit orders
    PROTECT = "protect"  # Target broken, stop set, managing risk
    MANAGE = "manage"  # Taking profits, trailing, adding on pullbacks
    CLOSING = "closing"  # Winding down position
    CLOSED = "closed"  # Trade complete


class StopStrategy(StrEnum):
    """Stop-loss strategy for the current trade."""

    NONE = "none"  # No stop yet (scout/build phase)
    MENTAL = "mental"  # Plan invalidation level tracked but no order
    BREAKEVEN = "breakeven"  # Stop at entry after target break
    SAFE = "safe"  # Stop behind entry zone
    TRAILING = "trailing"  # Trailing stop (EMA9 or structure-based)


# ---------------------------------------------------------------------------
# Staged trade plan
# ---------------------------------------------------------------------------


@dataclass
class StagedTradePlan:
    """A trade plan with staged entry/exit levels from the morning pipeline."""

    # Identity
    trade_id: str = ""
    symbol: str = ""  # e.g. "MGC", "MES"
    product_code: str = ""  # Rithmic product code
    exchange: str = ""  # e.g. "NYMEX", "CME"
    direction: str = ""  # "LONG" or "SHORT"

    # Plan levels
    entry_zone_low: float = 0.0  # Bottom of entry zone
    entry_zone_high: float = 0.0  # Top of entry zone
    ideal_entry: float = 0.0  # Best entry price within zone
    invalidation_level: float = 0.0  # Plan is wrong below/above this

    # Targets
    target_1: float = 0.0  # First target (trigger protection)
    target_2: float = 0.0  # Second target (take partials)
    target_3: float = 0.0  # Full target / runner exit

    # Position sizing (from plan)
    max_contracts: int = 4  # Max for this trade
    initial_size: int = 1  # Scout entry size
    build_sizes: list[int] = field(default_factory=lambda: [1, 1, 1])  # Limit order sizes

    # Limit order levels for building
    build_levels: list[float] = field(default_factory=list)  # Prices for limit orders

    # Risk
    max_risk_dollars: float = 825.0  # Max loss on this trade
    reason: str = ""  # Why we're taking this trade
    plan_confidence: float = 0.0  # CNN/confluence score 0-100

    # Timestamps
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{self.direction}_{int(time.time())}"


# ---------------------------------------------------------------------------
# Active trade state
# ---------------------------------------------------------------------------


@dataclass
class ActiveTrade:
    """Runtime state of a trade being executed by the TradeExecutor."""

    plan: StagedTradePlan
    phase: TradePhase = TradePhase.PLANNED
    stop_strategy: StopStrategy = StopStrategy.NONE

    # Position tracking
    total_contracts: int = 0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Order tracking
    scout_order_id: str = ""
    limit_order_ids: list[str] = field(default_factory=list)
    stop_order_id: str = ""

    # Fill tracking
    fills: list[dict[str, Any]] = field(default_factory=list)
    partial_exits: list[dict[str, Any]] = field(default_factory=list)

    # Stop management
    current_stop_price: float = 0.0  # 0 = no stop placed
    mental_stop_price: float = 0.0  # Invalidation level (no order)

    # Phase transitions
    target_1_hit: bool = False
    target_2_hit: bool = False
    target_3_hit: bool = False

    # Timing
    entered_at: str = ""
    protected_at: str = ""
    closed_at: str = ""
    last_update: str = ""

    # Rithmic execution tracking
    rithmic_stop_order_placed: bool = False  # True once a real server-side stop exists
    last_rithmic_action: str = ""  # Last Rithmic action taken (e.g. "set_stop", "move_stop")

    # Events log
    events: list[str] = field(default_factory=list)

    def log_event(self, msg: str) -> None:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {msg}")
        if len(self.events) > 100:
            self.events = self.events[-100:]
        self.last_update = datetime.now(UTC).isoformat()

    @property
    def is_long(self) -> bool:
        return self.plan.direction == "LONG"

    @property
    def is_active(self) -> bool:
        return self.phase not in (TradePhase.PLANNED, TradePhase.CLOSED)

    @property
    def has_stop(self) -> bool:
        return self.current_stop_price > 0

    @property
    def contracts_remaining_to_build(self) -> int:
        return max(0, self.plan.max_contracts - self.total_contracts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.plan.trade_id,
            "symbol": self.plan.symbol,
            "direction": self.plan.direction,
            "phase": self.phase.value,
            "stop_strategy": self.stop_strategy.value,
            "total_contracts": self.total_contracts,
            "avg_entry_price": round(self.avg_entry_price, 6),
            "current_price": round(self.current_price, 6),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "current_stop_price": round(self.current_stop_price, 6),
            "mental_stop_price": round(self.mental_stop_price, 6),
            "rithmic_stop_order_placed": self.rithmic_stop_order_placed,
            "last_rithmic_action": self.last_rithmic_action,
            "target_1_hit": self.target_1_hit,
            "target_2_hit": self.target_2_hit,
            "target_3_hit": self.target_3_hit,
            "fills": self.fills,
            "partial_exits": self.partial_exits,
            "events": self.events[-20:],
            "phase_display": self._phase_display(),
            "contracts_to_build": self.contracts_remaining_to_build,
            "entered_at": self.entered_at,
            "last_update": self.last_update,
            "plan": {
                "entry_zone": f"{self.plan.entry_zone_low:.2f} - {self.plan.entry_zone_high:.2f}",
                "invalidation": self.plan.invalidation_level,
                "target_1": self.plan.target_1,
                "target_2": self.plan.target_2,
                "target_3": self.plan.target_3,
                "max_contracts": self.plan.max_contracts,
                "reason": self.plan.reason,
                "confidence": self.plan.plan_confidence,
            },
        }

    def _phase_display(self) -> str:
        displays = {
            TradePhase.PLANNED: "\U0001f4cb PLANNED \u2014 waiting for entry",
            TradePhase.SCOUT: "\U0001f50d SCOUT \u2014 initial entry, no stop",
            TradePhase.BUILD: "\U0001f3d7\ufe0f BUILD \u2014 accumulating via limits",
            TradePhase.PROTECT: "\U0001f6e1\ufe0f PROTECT \u2014 stop set, managing",
            TradePhase.MANAGE: "\U0001f4ca MANAGE \u2014 taking profits & trailing",
            TradePhase.CLOSING: "\U0001f504 CLOSING \u2014 winding down",
            TradePhase.CLOSED: "\u2705 CLOSED",
        }
        return displays.get(self.phase, self.phase.value)


# ---------------------------------------------------------------------------
# Trade Executor
# ---------------------------------------------------------------------------


class TradeExecutor:
    """Manages staged trade execution with stop-hunt protection.

    The executor works WITH the CopyTrader — every order goes through
    send_order_and_copy() with MANUAL flags for prop-firm compliance.
    The executor just decides WHAT to order and WHEN.

    Usage::

        executor = TradeExecutor()

        # From the morning plan, create a staged trade
        plan = StagedTradePlan(
            symbol="MGC", product_code="MGC", exchange="NYMEX",
            direction="LONG",
            entry_zone_low=2920.0, entry_zone_high=2935.0,
            ideal_entry=2925.0, invalidation_level=2910.0,
            target_1=2960.0, target_2=2990.0, target_3=3020.0,
            max_contracts=4, reason="H1 OB + VAL confluence"
        )

        # Trader clicks "ENGAGE TRADE" in WebUI
        trade = await executor.engage_trade(plan, copy_trader)

        # On each price tick, the executor checks phases
        actions = executor.on_tick(current_price, trade)
    """

    def __init__(self) -> None:
        self._active_trades: dict[str, ActiveTrade] = {}  # trade_id -> trade
        self._completed_trades: list[dict[str, Any]] = []

    @property
    def active_trade_count(self) -> int:
        return len(self._active_trades)

    @property
    def active_trades(self) -> list[ActiveTrade]:
        return list(self._active_trades.values())

    def get_active_trade(self, symbol: str) -> ActiveTrade | None:
        """Get active trade for a symbol."""
        for trade in self._active_trades.values():
            if trade.plan.symbol == symbol:
                return trade
        return None

    async def engage_trade(
        self,
        plan: StagedTradePlan,
        copy_trader: Any,  # CopyTrader instance
    ) -> ActiveTrade:
        """Initiate Phase 1 (SCOUT) — place initial market entry.

        This is triggered by the human clicking "ENGAGE TRADE" in the WebUI.
        Places a small market order to get into the trade, then monitors.

        Args:
            plan: The staged trade plan from the morning pipeline.
            copy_trader: CopyTrader instance for order execution.

        Returns:
            ActiveTrade with SCOUT phase.
        """
        trade = ActiveTrade(plan=plan)
        trade.mental_stop_price = plan.invalidation_level

        side = "BUY" if plan.direction == "LONG" else "SELL"

        trade.log_event(f"ENGAGE: {side} {plan.initial_size}x {plan.symbol} (SCOUT entry)")
        trade.log_event(f"Plan: Entry zone {plan.entry_zone_low:.2f}-{plan.entry_zone_high:.2f}")
        trade.log_event(f"Mental stop (invalidation): {plan.invalidation_level:.2f}")
        trade.log_event(f"Targets: T1={plan.target_1:.2f} T2={plan.target_2:.2f} T3={plan.target_3:.2f}")

        # Phase 1: Scout entry — market order, NO stop
        # We send stop_ticks=0 to NOT attach a server-side bracket stop.
        # The mental stop (invalidation level) is tracked but not ordered.
        try:
            result = await copy_trader.send_order_and_copy(
                security_code=plan.product_code,
                exchange=plan.exchange,
                side=side,
                qty=plan.initial_size,
                order_type="MARKET",
                stop_ticks=0,  # NO STOP on scout entry
                target_ticks=None,
                tag_prefix="SCOUT",
                reason=f"Scout entry: {plan.reason}",
            )

            if result.main_result and result.main_result.status.value in ("submitted", "filled"):
                trade.phase = TradePhase.SCOUT
                trade.scout_order_id = result.main_result.order_id
                trade.total_contracts = plan.initial_size
                trade.entered_at = datetime.now(UTC).isoformat()
                trade.log_event(f"\u2705 Scout entry submitted \u2014 {plan.initial_size}x {side} MARKET (no stop)")

                # Store in active trades
                self._active_trades[plan.trade_id] = trade

                # Queue build orders if plan has build levels
                if plan.build_levels:
                    trade.phase = TradePhase.BUILD
                    trade.log_event(f"Transitioning to BUILD \u2014 {len(plan.build_levels)} limit levels queued")

                    # Place limit orders to build position
                    await self._place_build_orders(trade, copy_trader)
            else:
                error = result.main_result.error if result.main_result else "no result"
                trade.log_event(f"\u274c Scout entry FAILED: {error}")
                trade.phase = TradePhase.CLOSED

        except Exception as exc:
            trade.log_event(f"\u274c Scout entry ERROR: {exc}")
            trade.phase = TradePhase.CLOSED
            logger.error("TradeExecutor.engage_trade failed: %s", exc)

        return trade

    async def _place_build_orders(
        self,
        trade: ActiveTrade,
        copy_trader: Any,
    ) -> None:
        """Place limit orders at plan levels to build position."""
        plan = trade.plan
        side = "BUY" if plan.direction == "LONG" else "SELL"

        for i, level in enumerate(plan.build_levels):
            size = plan.build_sizes[i] if i < len(plan.build_sizes) else 1

            if trade.total_contracts + size > plan.max_contracts:
                trade.log_event(f"Skipping build level {level:.2f} \u2014 would exceed max contracts")
                break

            try:
                result = await copy_trader.send_order_and_copy(
                    security_code=plan.product_code,
                    exchange=plan.exchange,
                    side=side,
                    qty=size,
                    order_type="LIMIT",
                    price=level,
                    stop_ticks=0,  # Still no stop during BUILD
                    target_ticks=None,
                    tag_prefix="BUILD",
                    reason=f"Build L{i + 1}: {plan.reason}",
                )

                if result.main_result:
                    trade.limit_order_ids.append(result.main_result.order_id)
                    trade.log_event(f"\U0001f4cb Build limit #{i + 1}: {side} {size}x @ {level:.2f}")

            except Exception as exc:
                trade.log_event(f"\u26a0\ufe0f Build limit #{i + 1} failed: {exc}")
                logger.warning("Build order failed: %s", exc)

    def on_tick(self, current_price: float, trade: ActiveTrade) -> list[dict[str, Any]]:
        """Process a price tick for an active trade.

        Returns a list of action dicts to be executed by the caller.
        Actions: set_stop, move_stop, take_partial, add_more, close_trade, alert
        """
        if not trade.is_active:
            return []

        trade.current_price = current_price
        trade.last_update = datetime.now(UTC).isoformat()
        actions: list[dict[str, Any]] = []

        # Update P&L
        if trade.total_contracts > 0 and trade.avg_entry_price > 0:
            if trade.is_long:
                trade.unrealized_pnl = (current_price - trade.avg_entry_price) * trade.total_contracts
            else:
                trade.unrealized_pnl = (trade.avg_entry_price - current_price) * trade.total_contracts

        # Check plan invalidation (mental stop)
        if self._check_invalidation(trade, current_price):
            actions.append(
                {
                    "action": "alert",
                    "level": "critical",
                    "message": (
                        f"\u26a0\ufe0f PLAN INVALIDATION \u2014 price {current_price:.2f} "
                        f"past invalidation {trade.mental_stop_price:.2f}"
                    ),
                }
            )
            actions.append(
                {
                    "action": "close_trade",
                    "reason": "Plan invalidation \u2014 price broke mental stop",
                    "price": current_price,
                }
            )
            return actions

        # Phase-specific logic
        if trade.phase == TradePhase.SCOUT:
            actions.extend(self._process_scout_phase(trade, current_price))
        elif trade.phase == TradePhase.BUILD:
            actions.extend(self._process_build_phase(trade, current_price))
        elif trade.phase == TradePhase.PROTECT:
            actions.extend(self._process_protect_phase(trade, current_price))
        elif trade.phase == TradePhase.MANAGE:
            actions.extend(self._process_manage_phase(trade, current_price))

        return actions

    def _check_invalidation(self, trade: ActiveTrade, price: float) -> bool:
        """Check if price has invalidated the trade plan."""
        if trade.mental_stop_price <= 0:
            return False
        if trade.has_stop:
            # Once we have a real stop, let it handle things
            return False
        if trade.is_long:
            return price <= trade.mental_stop_price
        else:
            return price >= trade.mental_stop_price

    def _process_scout_phase(
        self,
        trade: ActiveTrade,
        price: float,
    ) -> list[dict[str, Any]]:
        """SCOUT phase: Monitor initial entry, check for T1 break."""
        actions: list[dict[str, Any]] = []

        # Check if T1 is hit — transition to PROTECT
        t1_hit = (trade.is_long and price >= trade.plan.target_1) or (
            not trade.is_long and price <= trade.plan.target_1
        )

        if t1_hit and not trade.target_1_hit:
            trade.target_1_hit = True
            trade.phase = TradePhase.PROTECT
            trade.stop_strategy = StopStrategy.BREAKEVEN

            # Set stop at breakeven (entry price)
            stop_price = trade.avg_entry_price
            trade.current_stop_price = stop_price
            trade.protected_at = datetime.now(UTC).isoformat()

            trade.log_event(f"\U0001f3af T1 HIT @ {price:.2f} \u2014 setting breakeven stop @ {stop_price:.2f}")

            actions.append(
                {
                    "action": "set_stop",
                    "price": stop_price,
                    "reason": "T1 hit \u2014 breakeven protection",
                }
            )

        # Check if price is coming back into build zone
        in_zone = trade.plan.entry_zone_low <= price <= trade.plan.entry_zone_high
        if in_zone and trade.contracts_remaining_to_build > 0:
            actions.append(
                {
                    "action": "alert",
                    "level": "info",
                    "message": f"Price in entry zone ({price:.2f}) \u2014 build opportunity",
                }
            )

        return actions

    def _process_build_phase(
        self,
        trade: ActiveTrade,
        price: float,
    ) -> list[dict[str, Any]]:
        """BUILD phase: same as scout but also watching limit fills."""
        # Same T1 check as scout
        actions = self._process_scout_phase(trade, price)
        return actions

    def _process_protect_phase(
        self,
        trade: ActiveTrade,
        price: float,
    ) -> list[dict[str, Any]]:
        """PROTECT phase: stop is set, watching for T2."""
        actions: list[dict[str, Any]] = []

        # Check T2 hit — take partial profit
        t2_hit = (trade.is_long and price >= trade.plan.target_2) or (
            not trade.is_long and price <= trade.plan.target_2
        )

        if t2_hit and not trade.target_2_hit:
            trade.target_2_hit = True
            trade.phase = TradePhase.MANAGE
            trade.stop_strategy = StopStrategy.TRAILING

            # Take half off
            partial_qty = max(1, trade.total_contracts // 2)

            trade.log_event(f"\U0001f3af T2 HIT @ {price:.2f} \u2014 taking {partial_qty} off, trailing rest")

            actions.append(
                {
                    "action": "take_partial",
                    "qty": partial_qty,
                    "reason": f"T2 hit @ {price:.2f}",
                    "price": price,
                }
            )

            # Move stop to a safe level (midpoint between entry and T1)
            if trade.is_long:
                safe_stop = trade.avg_entry_price + (trade.plan.target_1 - trade.avg_entry_price) * 0.5
            else:
                safe_stop = trade.avg_entry_price - (trade.avg_entry_price - trade.plan.target_1) * 0.5

            trade.current_stop_price = safe_stop
            trade.stop_strategy = StopStrategy.SAFE

            actions.append(
                {
                    "action": "move_stop",
                    "price": safe_stop,
                    "reason": "T2 hit \u2014 stop moved to safe level",
                }
            )

        return actions

    def _process_manage_phase(
        self,
        trade: ActiveTrade,
        price: float,
    ) -> list[dict[str, Any]]:
        """MANAGE phase: trail stop, take profits on spikes, watch T3."""
        actions: list[dict[str, Any]] = []

        # Check T3 hit — close remaining
        t3_hit = (trade.is_long and price >= trade.plan.target_3) or (
            not trade.is_long and price <= trade.plan.target_3
        )

        if t3_hit and not trade.target_3_hit:
            trade.target_3_hit = True
            trade.log_event(f"\U0001f3c6 T3 HIT @ {price:.2f} \u2014 closing remaining position")

            actions.append(
                {
                    "action": "close_trade",
                    "reason": f"T3 full target hit @ {price:.2f}",
                    "price": price,
                }
            )
            return actions

        # Trail stop logic: only move in favorable direction
        if trade.stop_strategy == StopStrategy.TRAILING and trade.current_stop_price > 0:
            # Simple trailing: keep stop at 50% of distance from entry to current
            if trade.is_long:
                new_trail = trade.avg_entry_price + (price - trade.avg_entry_price) * 0.5
                if new_trail > trade.current_stop_price:
                    trade.current_stop_price = new_trail
                    actions.append(
                        {
                            "action": "move_stop",
                            "price": new_trail,
                            "reason": f"Trail updated: {new_trail:.2f}",
                        }
                    )
            else:
                new_trail = trade.avg_entry_price - (trade.avg_entry_price - price) * 0.5
                if new_trail < trade.current_stop_price:
                    trade.current_stop_price = new_trail
                    actions.append(
                        {
                            "action": "move_stop",
                            "price": new_trail,
                            "reason": f"Trail updated: {new_trail:.2f}",
                        }
                    )

        return actions

    async def close_trade(
        self,
        trade: ActiveTrade,
        copy_trader: Any,
        reason: str = "Manual close",
    ) -> None:
        """Close an active trade — market order to flatten."""
        if not trade.is_active:
            return

        side = "SELL" if trade.is_long else "BUY"

        trade.log_event(f"CLOSING: {side} {trade.total_contracts}x \u2014 {reason}")

        try:
            await copy_trader.send_order_and_copy(
                security_code=trade.plan.product_code,
                exchange=trade.plan.exchange,
                side=side,
                qty=trade.total_contracts,
                order_type="MARKET",
                stop_ticks=0,
                target_ticks=None,
                tag_prefix="CLOSE",
                reason=reason,
            )
            trade.phase = TradePhase.CLOSED
            trade.closed_at = datetime.now(UTC).isoformat()
            trade.log_event(f"\u2705 Position closed \u2014 {reason}")

            # Cancel any remaining limit orders
            # (CopyTrader would need a cancel method — TODO)
            if trade.limit_order_ids:
                trade.log_event(f"\u26a0\ufe0f {len(trade.limit_order_ids)} limit orders may need manual cancellation")

            # Move to completed
            self._completed_trades.append(trade.to_dict())
            if trade.plan.trade_id in self._active_trades:
                del self._active_trades[trade.plan.trade_id]

        except Exception as exc:
            trade.log_event(f"\u274c Close failed: {exc}")
            logger.error("TradeExecutor.close_trade failed: %s", exc)

    async def take_partial_profit(
        self,
        trade: ActiveTrade,
        copy_trader: Any,
        qty: int = 1,
        reason: str = "Partial profit",
    ) -> None:
        """Take partial profit on an active trade."""
        if not trade.is_active or trade.total_contracts < qty:
            return

        side = "SELL" if trade.is_long else "BUY"

        try:
            await copy_trader.send_order_and_copy(
                security_code=trade.plan.product_code,
                exchange=trade.plan.exchange,
                side=side,
                qty=qty,
                order_type="MARKET",
                stop_ticks=0,
                target_ticks=None,
                tag_prefix="PARTIAL",
                reason=reason,
            )
            trade.total_contracts -= qty
            trade.partial_exits.append(
                {
                    "qty": qty,
                    "price": trade.current_price,
                    "reason": reason,
                    "ts": datetime.now(UTC).isoformat(),
                }
            )
            trade.log_event(f"\U0001f4b0 Partial exit: {qty}x @ {trade.current_price:.2f} \u2014 {reason}")

            if trade.total_contracts <= 0:
                trade.phase = TradePhase.CLOSED
                trade.closed_at = datetime.now(UTC).isoformat()
                self._completed_trades.append(trade.to_dict())
                if trade.plan.trade_id in self._active_trades:
                    del self._active_trades[trade.plan.trade_id]

        except Exception as exc:
            trade.log_event(f"\u274c Partial exit failed: {exc}")

    def on_fill(self, trade: ActiveTrade, fill_price: float, fill_qty: int) -> None:
        """Record a fill (from limit order or market) and update average entry."""
        if not trade.is_active:
            return

        old_total = trade.total_contracts
        old_avg = trade.avg_entry_price

        if old_total == 0:
            trade.avg_entry_price = fill_price
        else:
            # Weighted average
            trade.avg_entry_price = (old_avg * old_total + fill_price * fill_qty) / (old_total + fill_qty)

        trade.total_contracts += fill_qty
        trade.fills.append(
            {
                "price": fill_price,
                "qty": fill_qty,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        trade.log_event(
            f"\U0001f4e5 Fill: +{fill_qty}x @ {fill_price:.2f} (avg now {trade.avg_entry_price:.2f}, total {trade.total_contracts}x)"
        )

    # ------------------------------------------------------------------
    # Live tick processing — wires on_tick() actions to real Rithmic
    # ------------------------------------------------------------------

    async def process_tick_actions(
        self,
        actions: list[dict[str, Any]],
        trade: ActiveTrade,
        copy_trader: Any,
    ) -> None:
        """Execute the action dicts returned by ``on_tick()`` via real Rithmic orders.

        This bridges the gap between the phase-based decision logic in
        ``on_tick()`` and actual server-side order management through CopyTrader.

        Args:
            actions: List of action dicts from ``on_tick()``.
            trade: The active trade these actions belong to.
            copy_trader: CopyTrader instance for Rithmic execution.
        """
        for action in actions:
            action_type = action.get("action", "")

            if action_type in ("set_stop", "move_stop"):
                stop_price = action.get("price", 0.0)
                reason = action.get("reason", action_type)

                if stop_price <= 0:
                    trade.log_event(f"⚠️ {action_type} skipped — invalid stop price {stop_price}")
                    continue

                # Convert absolute stop price → tick distance for Rithmic
                stop_ticks = stop_price_to_stop_ticks(
                    entry_price=trade.avg_entry_price,
                    stop_price=stop_price,
                    product_code=trade.plan.product_code,
                )

                trade.log_event(
                    f"🔧 Rithmic {action_type}: stop @ {stop_price:.2f} "
                    f"({stop_ticks} ticks from entry {trade.avg_entry_price:.2f}) — {reason}"
                )

                try:
                    result = await copy_trader.modify_stop_on_all(
                        security_code=trade.plan.product_code,
                        exchange=trade.plan.exchange,
                        new_stop_price=stop_price,
                        product_code=trade.plan.product_code,
                        entry_price=trade.avg_entry_price,
                        reason=f"TradeExecutor {action_type}: {reason}",
                    )

                    if result.get("ok"):
                        trade.rithmic_stop_order_placed = True
                        trade.last_rithmic_action = action_type
                        modified = result.get("accounts_modified", [])
                        trade.log_event(
                            f"✅ Rithmic {action_type} placed on {len(modified)} account(s) "
                            f"— {stop_ticks} ticks, stop @ {stop_price:.2f}"
                        )
                    else:
                        failed = result.get("accounts_failed", [])
                        fail_reason = result.get("reason", "unknown")
                        trade.log_event(
                            f"⚠️ Rithmic {action_type} partial/failed — reason: {fail_reason}, failed accounts: {failed}"
                        )
                        logger.warning(
                            "modify_stop_on_all incomplete for %s: %s",
                            trade.plan.trade_id,
                            result,
                        )

                except Exception as exc:
                    trade.log_event(f"❌ Rithmic {action_type} ERROR: {exc}")
                    logger.error(
                        "process_tick_actions %s failed for %s: %s",
                        action_type,
                        trade.plan.trade_id,
                        exc,
                    )

            elif action_type == "take_partial":
                qty = action.get("qty", 1)
                reason = action.get("reason", "Partial profit")
                trade.log_event(f"🔧 Rithmic take_partial: {qty}x — {reason}")
                await self.take_partial_profit(trade, copy_trader, qty=qty, reason=reason)

            elif action_type == "close_trade":
                reason = action.get("reason", "on_tick close signal")
                trade.log_event(f"🔧 Rithmic close_trade — {reason}")
                await self.close_trade(trade, copy_trader, reason=reason)

            elif action_type == "alert":
                level = action.get("level", "info")
                message = action.get("message", "")
                trade.log_event(f"🔔 Alert [{level}]: {message}")
                logger.log(
                    logging.WARNING if level == "critical" else logging.INFO,
                    "TradeExecutor alert [%s] %s: %s",
                    trade.plan.trade_id,
                    level,
                    message,
                )

            else:
                trade.log_event(f"⚠️ Unknown action type: {action_type}")
                logger.warning("process_tick_actions: unknown action %r", action_type)

    async def process_live_tick(
        self,
        symbol: str,
        price: float,
        copy_trader: Any,
    ) -> list[dict[str, Any]]:
        """End-to-end live tick handler: find trade → decide → execute.

        Called by the tick-stream consumer (e.g. SSE price feed handler).
        Finds the active trade for ``symbol``, runs phase logic via
        ``on_tick()``, then executes resulting actions through CopyTrader.

        Args:
            symbol: Instrument symbol (e.g. "MGC", "MES").
            price: Current market price.
            copy_trader: CopyTrader instance for Rithmic execution.

        Returns:
            The list of action dicts produced by ``on_tick()`` (for SSE broadcast).
            Empty list if no active trade for the symbol.
        """
        trade = self.get_active_trade(symbol)
        if trade is None:
            return []

        # Phase-based decision logic
        actions = self.on_tick(price, trade)

        # Execute actions through real Rithmic orders
        if actions:
            await self.process_tick_actions(actions, trade, copy_trader)

        return actions

    def status_summary(self) -> dict[str, Any]:
        """Return a summary of all active and recent trades."""
        return {
            "active_count": len(self._active_trades),
            "active_trades": [t.to_dict() for t in self._active_trades.values()],
            "completed_count": len(self._completed_trades),
            "recent_completed": self._completed_trades[-5:] if self._completed_trades else [],
        }
