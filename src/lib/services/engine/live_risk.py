"""
Live Risk State — Real-Time Risk Budget Integration
====================================================
Phase 5B: Wire RiskManager ↔ PositionManager into a unified live risk
state that is published to Redis every 5 seconds (or on every position
update) and pushed to the dashboard via SSE.

This module merges:
  - From RiskManager: account_size, daily_pnl, max_daily_loss, can_trade,
    block_reason, consecutive_losses
  - From PositionManager: all active MicroPosition objects with current P&L,
    bracket phase, R-multiple
  - Computed fields: remaining_risk_budget, total_unrealized_pnl,
    total_margin_used, margin_remaining

The LiveRiskState is the single source of truth for the dashboard's risk
strip (Phase 5E) and the dynamic position sizing on focus cards (Phase 5C).

Usage:
    from lib.services.engine.live_risk import LiveRiskState, compute_live_risk

    state = compute_live_risk(risk_manager, position_manager)
    state.publish_to_redis()
    payload = state.to_dict()  # for SSE push

    # Or use the background publisher:
    publisher = LiveRiskPublisher(risk_manager, position_manager)
    await publisher.start()  # publishes every 5s + on position changes
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from lib.services.engine.position_manager import PositionManager
    from lib.services.engine.risk import RiskManager

logger = logging.getLogger("engine.live_risk")

_EST = ZoneInfo("America/New_York")

# Redis keys
REDIS_KEY_LIVE_RISK = "engine:live_risk"
REDIS_CHANNEL_LIVE_RISK = "dashboard:live_risk"
REDIS_CHANNEL_RISK_WARNING = "dashboard:risk_warning"

# Default constants
DEFAULT_MAX_OPEN_TRADES = 2
DEFAULT_ACCOUNT_SIZE = 150_000
DEFAULT_MAX_RISK_PCT = 0.0075  # 0.75% per trade


# ---------------------------------------------------------------------------
# Position snapshot — serializable view of a live position
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PositionSnapshot:
    """Serializable snapshot of a single live position."""

    symbol: str = ""
    asset_name: str = ""
    side: str = ""  # "LONG" or "SHORT"
    quantity: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_price: float = 0.0
    unrealized_pnl: float = 0.0
    r_multiple: float = 0.0
    bracket_phase: str = "INITIAL"  # INITIAL, TP1_HIT, TP2_HIT, TRAILING, CLOSED
    hold_duration_seconds: int = 0
    risk_dollars: float = 0.0
    margin_used: float = 0.0
    source: str = "engine"  # "engine", "rithmic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_name": self.asset_name,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_price": self.stop_price,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "r_multiple": round(self.r_multiple, 2),
            "bracket_phase": self.bracket_phase,
            "hold_duration_seconds": self.hold_duration_seconds,
            "risk_dollars": round(self.risk_dollars, 2),
            "margin_used": round(self.margin_used, 2),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# LiveRiskState — the unified real-time risk picture
# ---------------------------------------------------------------------------
@dataclass
class LiveRiskState:
    """Unified live risk state merging RiskManager + PositionManager.

    This is the single payload that feeds:
      - Risk dashboard strip (Phase 5E)
      - Dynamic position sizing on focus cards (Phase 5C)
      - Live position overlay on focus cards (Phase 5D)
      - TradingView metrics push
    """

    # ── Account & limits ────────────────────────────────────────────────
    account_size: float = DEFAULT_ACCOUNT_SIZE
    max_risk_per_trade: float = 0.0
    max_daily_loss: float = -1500.0
    max_open_trades: int = DEFAULT_MAX_OPEN_TRADES
    risk_pct_per_trade: float = DEFAULT_MAX_RISK_PCT

    # ── Current state from RiskManager ──────────────────────────────────
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    can_trade: bool = True
    block_reason: str = ""
    is_past_entry_cutoff: bool = False
    is_daily_loss_exceeded: bool = False
    is_max_trades_reached: bool = False
    is_overnight_warning: bool = False
    trading_date: str = ""
    last_trade_time: str = ""

    # ── Live positions ──────────────────────────────────────────────────
    positions: list[PositionSnapshot] = field(default_factory=list)
    open_position_count: int = 0

    # ── Computed fields ─────────────────────────────────────────────────
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0  # = daily_pnl
    total_pnl: float = 0.0  # realized + unrealized
    total_margin_used: float = 0.0
    margin_remaining: float = 0.0
    total_risk_exposure: float = 0.0
    risk_pct_of_account: float = 0.0

    # Risk budget: how much room is left for new trades
    remaining_risk_budget: float = 0.0
    remaining_trade_slots: int = 0

    # ── Session info ────────────────────────────────────────────────────
    session_time_remaining: str = ""  # e.g. "2h 15m"
    session_active: bool = True

    # ── Timestamps ──────────────────────────────────────────────────────
    computed_at: str = ""
    computed_ts: float = 0.0

    # ── Status flags for dashboard coloring ─────────────────────────────
    health: str = "green"  # "green", "yellow", "red"
    health_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis / SSE / dashboard consumption."""
        return {
            # Account
            "account_size": self.account_size,
            "max_risk_per_trade": round(self.max_risk_per_trade, 2),
            "max_daily_loss": self.max_daily_loss,
            "max_open_trades": self.max_open_trades,
            "risk_pct_per_trade": self.risk_pct_per_trade,
            # Risk state
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trade_count": self.daily_trade_count,
            "consecutive_losses": self.consecutive_losses,
            "can_trade": self.can_trade,
            "block_reason": self.block_reason,
            "is_past_entry_cutoff": self.is_past_entry_cutoff,
            "is_daily_loss_exceeded": self.is_daily_loss_exceeded,
            "is_max_trades_reached": self.is_max_trades_reached,
            "is_overnight_warning": self.is_overnight_warning,
            "trading_date": self.trading_date,
            "last_trade_time": self.last_trade_time,
            # Positions
            "positions": [p.to_dict() for p in self.positions],
            "open_position_count": self.open_position_count,
            # Computed
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_margin_used": round(self.total_margin_used, 2),
            "margin_remaining": round(self.margin_remaining, 2),
            "total_risk_exposure": round(self.total_risk_exposure, 2),
            "risk_pct_of_account": round(self.risk_pct_of_account, 2),
            "remaining_risk_budget": round(self.remaining_risk_budget, 2),
            "remaining_trade_slots": self.remaining_trade_slots,
            # Session
            "session_time_remaining": self.session_time_remaining,
            "session_active": self.session_active,
            # Meta
            "computed_at": self.computed_at,
            "computed_ts": self.computed_ts,
            "health": self.health,
            "health_reason": self.health_reason,
        }

    def publish_to_redis(self) -> bool:
        """Write live risk state to Redis and publish to PubSub.

        Writes to:
          - REDIS_KEY_LIVE_RISK (TTL 30s) — for SSE catch-up reads
          - REDIS_CHANNEL_LIVE_RISK PubSub — for real-time SSE push
          - REDIS_CHANNEL_RISK_WARNING PubSub — when can_trade is False

        Returns True on success.
        """
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
        except ImportError:
            logger.error("Cannot import cache module for live risk publish")
            return False

        payload = self.to_dict()

        try:
            payload_json = json.dumps(payload, default=str, allow_nan=False)
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialize LiveRiskState: %s", exc)
            return False

        try:
            # Cache with TTL for SSE catch-up reads
            cache_set(REDIS_KEY_LIVE_RISK, payload_json.encode(), ttl=30)

            if REDIS_AVAILABLE and _r is not None:
                try:
                    # Main channel: dashboard risk strip + focus cards
                    _r.publish(REDIS_CHANNEL_LIVE_RISK, payload_json)

                    # Risk warning channel: when can't trade
                    if not self.can_trade:
                        warning = json.dumps(
                            {
                                "can_trade": False,
                                "reason": self.block_reason,
                                "daily_pnl": self.daily_pnl,
                                "health": self.health,
                                "ts": self.computed_at,
                            }
                        )
                        _r.publish(REDIS_CHANNEL_RISK_WARNING, warning)

                except Exception as exc:
                    logger.debug("Redis PubSub publish failed (non-fatal): %s", exc)

            return True

        except Exception as exc:
            logger.error("Failed to publish LiveRiskState to Redis: %s", exc)
            return False

    def get_position_for_asset(self, asset_name: str) -> PositionSnapshot | None:
        """Find the live position for a specific asset (by name or symbol)."""
        for pos in self.positions:
            if pos.asset_name == asset_name or pos.symbol == asset_name:
                return pos
        return None

    def has_position(self, asset_name: str) -> bool:
        """Check if there's a live position for a specific asset."""
        return self.get_position_for_asset(asset_name) is not None

    def compute_dynamic_size(
        self,
        entry_price: float,
        stop_price: float,
        point_value: float,
    ) -> tuple[int, float]:
        """Compute position size based on REMAINING risk budget.

        Instead of using the static max_risk_per_trade, this uses
        the remaining_risk_budget which accounts for current open
        positions and daily P&L.

        Args:
            entry_price: Planned entry price
            stop_price: Planned stop loss price
            point_value: Dollar value per 1.0 price point

        Returns:
            (num_contracts, risk_dollars)
        """
        if self.remaining_risk_budget <= 0 or not self.can_trade:
            return 0, 0.0

        stop_distance = abs(entry_price - stop_price)
        risk_per_contract = stop_distance * point_value
        if risk_per_contract <= 0:
            return 1, 0.0

        # Use the SMALLER of: remaining budget, or per-trade max
        effective_budget = min(self.remaining_risk_budget, self.max_risk_per_trade)
        num_contracts = max(1, int(effective_budget / risk_per_contract))
        actual_risk = num_contracts * risk_per_contract

        return num_contracts, round(actual_risk, 2)


# ---------------------------------------------------------------------------
# Computation: merge RiskManager + PositionManager → LiveRiskState
# ---------------------------------------------------------------------------
def compute_live_risk(
    risk_manager: RiskManager | None = None,
    position_manager: PositionManager | None = None,
) -> LiveRiskState:
    """Compute the unified live risk state from available sources.

    This is the main function called every ~5 seconds by the engine
    scheduler, and also triggered immediately on position changes.

    Args:
        risk_manager: Engine's RiskManager instance (may be None if unavailable)
        position_manager: Engine's PositionManager instance (may be None)

    Returns:
        LiveRiskState with all fields populated.
    """
    now = datetime.now(tz=_EST)
    state = LiveRiskState(
        computed_at=now.isoformat(),
        computed_ts=time.time(),
    )

    # ── Merge RiskManager state ─────────────────────────────────────────
    if risk_manager is not None:
        try:
            rm_status = risk_manager.get_status(now=now)

            state.account_size = rm_status.get("account_size", DEFAULT_ACCOUNT_SIZE)
            state.max_risk_per_trade = rm_status.get("max_risk_per_trade", 0.0)
            state.max_daily_loss = rm_status.get("max_daily_loss", -1500.0)
            state.max_open_trades = rm_status.get("max_open_trades", DEFAULT_MAX_OPEN_TRADES)
            state.risk_pct_per_trade = rm_status.get("rules", {}).get("risk_pct_per_trade", DEFAULT_MAX_RISK_PCT)
            state.daily_pnl = rm_status.get("daily_pnl", 0.0)
            state.daily_trade_count = rm_status.get("daily_trade_count", 0)
            state.consecutive_losses = rm_status.get("consecutive_losses", 0)
            state.can_trade = rm_status.get("can_trade", True)
            state.block_reason = rm_status.get("block_reason", "")
            state.is_past_entry_cutoff = rm_status.get("is_past_entry_cutoff", False)
            state.is_daily_loss_exceeded = rm_status.get("is_daily_loss_exceeded", False)
            state.is_max_trades_reached = rm_status.get("is_max_trades_reached", False)
            state.is_overnight_warning = rm_status.get("is_overnight_warning", False)
            state.trading_date = rm_status.get("trading_date", "")
            state.last_trade_time = rm_status.get("last_trade", "")
            state.total_risk_exposure = rm_status.get("total_risk_exposure", 0.0)
            state.risk_pct_of_account = rm_status.get("risk_pct_of_account", 0.0)

        except Exception as exc:
            logger.warning("Failed to read RiskManager state: %s", exc)

    else:
        # Defaults when no RiskManager
        state.max_risk_per_trade = state.account_size * state.risk_pct_per_trade

    # ── Merge PositionManager state ─────────────────────────────────────
    if position_manager is not None:
        try:
            all_positions = position_manager.get_all_positions()
            position_snapshots: list[PositionSnapshot] = []

            for _sym, micro_pos in all_positions.items():
                snap = _micro_position_to_snapshot(micro_pos)
                position_snapshots.append(snap)

            state.positions = position_snapshots
            state.open_position_count = len(position_snapshots)

        except Exception as exc:
            logger.warning("Failed to read PositionManager state: %s", exc)

    # ── Compute derived fields ──────────────────────────────────────────
    _compute_derived_fields(state)

    # ── Determine health status ─────────────────────────────────────────
    _compute_health(state)

    # ── Session time remaining ──────────────────────────────────────────
    _compute_session_time(state, now)

    logger.debug(
        "LiveRisk: PnL=$%.2f | Pos=%d/%d | Budget=$%.2f | Health=%s",
        state.total_pnl,
        state.open_position_count,
        state.max_open_trades,
        state.remaining_risk_budget,
        state.health,
    )

    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _micro_position_to_snapshot(micro_pos: Any) -> PositionSnapshot:
    """Convert a MicroPosition (or dict) to a PositionSnapshot."""
    # Handle both MicroPosition dataclass and dict representations
    if hasattr(micro_pos, "to_dict"):
        d = micro_pos.to_dict()
    elif isinstance(micro_pos, dict):
        d = micro_pos
    else:
        d = {}

    symbol = d.get("symbol", d.get("ticker", ""))
    asset_name = _resolve_asset_name(symbol)

    # Compute R-multiple
    r_mult = 0.0
    if not isinstance(micro_pos, dict) and hasattr(micro_pos, "r_multiple"):
        with contextlib.suppress(TypeError, ValueError):
            r_mult = float(micro_pos.r_multiple)  # type: ignore[union-attr]
    elif "r_multiple" in d:
        with contextlib.suppress(TypeError, ValueError):
            r_mult = float(d["r_multiple"])

    # Hold duration
    hold_secs = 0
    if not isinstance(micro_pos, dict) and hasattr(micro_pos, "hold_duration_seconds"):
        with contextlib.suppress(TypeError, ValueError):
            hold_secs = int(micro_pos.hold_duration_seconds)  # type: ignore[union-attr]

    # Bracket phase
    phase = "INITIAL"
    if not isinstance(micro_pos, dict) and hasattr(micro_pos, "phase"):
        phase_obj = micro_pos.phase  # type: ignore[union-attr]
        if hasattr(phase_obj, "name"):
            phase = phase_obj.name
        elif hasattr(phase_obj, "value"):
            phase = str(phase_obj.value)
        else:
            phase = str(phase_obj)
    elif "phase" in d:
        phase = str(d["phase"])
    elif "bracket_phase" in d:
        phase = str(d["bracket_phase"])

    return PositionSnapshot(
        symbol=symbol,
        asset_name=asset_name,
        side=d.get("side", d.get("direction", "UNKNOWN")).upper(),
        quantity=int(d.get("quantity", d.get("size", 1))),
        entry_price=float(d.get("entry_price", d.get("entry", 0))),
        current_price=float(d.get("current_price", d.get("price", 0))),
        stop_price=float(d.get("stop_price", d.get("stop_loss", 0))),
        unrealized_pnl=float(d.get("unrealized_pnl", d.get("pnl", 0))),
        r_multiple=r_mult,
        bracket_phase=phase,
        hold_duration_seconds=hold_secs,
        risk_dollars=float(d.get("risk_dollars", d.get("initial_risk", 0))),
        margin_used=float(d.get("margin_used", d.get("margin", 0))),
        source=d.get("source", "engine"),
    )


def _resolve_asset_name(symbol: str) -> str:
    """Resolve a ticker/symbol to its generalized asset name."""
    if not symbol:
        return ""
    try:
        from lib.core.asset_registry import get_asset_name_by_ticker

        return get_asset_name_by_ticker(symbol)
    except ImportError:
        return symbol


def _compute_derived_fields(state: LiveRiskState) -> None:
    """Compute all derived fields from the raw state."""
    # Total unrealized P&L across all positions
    state.total_unrealized_pnl = sum(p.unrealized_pnl for p in state.positions)

    # Total P&L = realized (daily) + unrealized
    state.total_realized_pnl = state.daily_pnl
    state.total_pnl = state.daily_pnl + state.total_unrealized_pnl

    # Total margin used
    state.total_margin_used = sum(p.margin_used for p in state.positions)

    # If margin_used is 0 for positions (not tracked), estimate from asset registry
    if state.total_margin_used == 0 and state.positions:
        try:
            from lib.core.asset_registry import get_asset_by_ticker

            for pos in state.positions:
                asset = get_asset_by_ticker(pos.symbol)
                if asset and asset.micro:
                    pos.margin_used = asset.micro.margin * pos.quantity
                    state.total_margin_used += pos.margin_used
        except ImportError:
            pass

    # Margin remaining
    state.margin_remaining = max(0, state.account_size - state.total_margin_used)

    # Risk exposure from open positions
    state.total_risk_exposure = sum(p.risk_dollars for p in state.positions)
    if state.account_size > 0:
        state.risk_pct_of_account = (state.total_risk_exposure / state.account_size) * 100

    # Remaining risk budget: how much $ risk is available for new trades
    # max_risk_per_trade × (max_open_trades − current_open)
    state.remaining_trade_slots = max(0, state.max_open_trades - state.open_position_count)

    if state.max_risk_per_trade <= 0:
        state.max_risk_per_trade = state.account_size * state.risk_pct_per_trade

    state.remaining_risk_budget = state.max_risk_per_trade * state.remaining_trade_slots

    # If daily loss is approaching limit, reduce budget further
    if state.max_daily_loss < 0:  # max_daily_loss is negative (e.g. -1500)
        pnl_room = abs(state.max_daily_loss) - abs(min(0, state.daily_pnl))
        if pnl_room < state.remaining_risk_budget:
            state.remaining_risk_budget = max(0, pnl_room)

    # If can't trade, zero out budget
    if not state.can_trade:
        state.remaining_risk_budget = 0.0
        state.remaining_trade_slots = 0


def _compute_health(state: LiveRiskState) -> None:
    """Determine the health color and reason for the dashboard strip."""
    reasons: list[str] = []

    # Red conditions
    if not state.can_trade:
        state.health = "red"
        reasons.append(state.block_reason or "Trading blocked")
    elif state.is_daily_loss_exceeded:
        state.health = "red"
        reasons.append("Daily loss limit exceeded")
    elif state.consecutive_losses >= 3:
        state.health = "red"
        reasons.append(f"{state.consecutive_losses} consecutive losses")

    # Yellow conditions
    elif state.is_max_trades_reached:
        state.health = "yellow"
        reasons.append("Max positions reached")
    elif state.is_past_entry_cutoff:
        state.health = "yellow"
        reasons.append("Past entry cutoff time")
    elif state.consecutive_losses >= 2:
        state.health = "yellow"
        reasons.append(f"{state.consecutive_losses} consecutive losses")
    elif state.max_daily_loss < 0 and state.daily_pnl < 0:
        # Check if approaching daily loss limit (within 30%)
        pnl_pct_of_limit = abs(state.daily_pnl) / abs(state.max_daily_loss)
        if pnl_pct_of_limit >= 0.70:
            state.health = "yellow"
            reasons.append(f"Approaching daily loss limit ({pnl_pct_of_limit:.0%})")
    elif state.is_overnight_warning:
        state.health = "yellow"
        reasons.append("Overnight risk warning")

    # Green
    else:
        state.health = "green"
        if state.daily_pnl > 0:
            reasons.append(f"Healthy — +${state.daily_pnl:,.0f} today")
        else:
            reasons.append("Ready to trade")

    state.health_reason = "; ".join(reasons)


def _compute_session_time(state: LiveRiskState, now: datetime) -> None:
    """Compute session time remaining string."""
    # TPT session ends at 4:00 PM ET, new session at 6:00 PM ET
    now_hour = now.hour
    now_min = now.minute

    # Session: 6:00 PM ET to 4:00 PM ET next day (22 hours)
    # For display purposes, show time until 4:00 PM ET cutoff
    if 6 <= now_hour < 16:
        # During main session (6 AM - 4 PM)
        remaining_hours = 15 - now_hour
        remaining_mins = 59 - now_min
        state.session_time_remaining = f"{remaining_hours}h {remaining_mins}m"
        state.session_active = True
    elif now_hour < 6:
        # Pre-market (midnight - 6 AM) — globex overnight
        state.session_time_remaining = f"{15 - now_hour}h {59 - now_min}m until close"
        state.session_active = True
    else:
        # After 4 PM — session closed
        state.session_time_remaining = "CLOSED"
        state.session_active = False


# ---------------------------------------------------------------------------
# Convenience: load from Redis (for API endpoints / SSE catch-up)
# ---------------------------------------------------------------------------
def load_from_redis() -> LiveRiskState | None:
    """Load the latest LiveRiskState from Redis cache.

    Returns None if not available.
    """
    try:
        from lib.core.cache import cache_get

        raw = cache_get(REDIS_KEY_LIVE_RISK)
        if not raw:
            return None

        data = json.loads(raw)
        return _dict_to_live_risk(data)
    except Exception as exc:
        logger.debug("Failed to load LiveRiskState from Redis: %s", exc)
        return None


def _dict_to_live_risk(data: dict[str, Any]) -> LiveRiskState:
    """Reconstruct a LiveRiskState from a serialized dict."""
    state = LiveRiskState()

    # Simple scalar fields
    for field_name in [
        "account_size",
        "max_risk_per_trade",
        "max_daily_loss",
        "max_open_trades",
        "risk_pct_per_trade",
        "daily_pnl",
        "daily_trade_count",
        "consecutive_losses",
        "can_trade",
        "block_reason",
        "is_past_entry_cutoff",
        "is_daily_loss_exceeded",
        "is_max_trades_reached",
        "is_overnight_warning",
        "trading_date",
        "last_trade_time",
        "open_position_count",
        "total_unrealized_pnl",
        "total_realized_pnl",
        "total_pnl",
        "total_margin_used",
        "margin_remaining",
        "total_risk_exposure",
        "risk_pct_of_account",
        "remaining_risk_budget",
        "remaining_trade_slots",
        "session_time_remaining",
        "session_active",
        "computed_at",
        "computed_ts",
        "health",
        "health_reason",
    ]:
        if field_name in data:
            setattr(state, field_name, data[field_name])

    # Position snapshots
    if "positions" in data and isinstance(data["positions"], list):
        for pos_dict in data["positions"]:
            snap = PositionSnapshot(**{k: v for k, v in pos_dict.items() if hasattr(PositionSnapshot, k)})
            state.positions.append(snap)

    return state


# ---------------------------------------------------------------------------
# Background publisher (for use in engine scheduler)
# ---------------------------------------------------------------------------
class LiveRiskPublisher:
    """Periodically computes and publishes LiveRiskState to Redis.

    Designed to be started as a background task in the engine scheduler.
    Publishes every `interval_seconds` and also on-demand when positions
    change.

    Usage:
        publisher = LiveRiskPublisher(risk_manager, position_manager)
        # In scheduler or asyncio loop:
        publisher.tick()  # Call every second; publishes every interval_seconds
        publisher.force_publish()  # Call on position change for immediate update
    """

    def __init__(
        self,
        risk_manager: RiskManager | None = None,
        position_manager: PositionManager | None = None,
        interval_seconds: float = 5.0,
    ):
        self._risk_manager = risk_manager
        self._position_manager = position_manager
        self._interval = interval_seconds
        self._last_publish: float = 0.0
        self._last_state: LiveRiskState | None = None

    @property
    def risk_manager(self) -> RiskManager | None:
        return self._risk_manager

    @risk_manager.setter
    def risk_manager(self, rm: RiskManager | None) -> None:
        self._risk_manager = rm

    @property
    def position_manager(self) -> PositionManager | None:
        return self._position_manager

    @position_manager.setter
    def position_manager(self, pm: PositionManager | None) -> None:
        self._position_manager = pm

    @property
    def last_state(self) -> LiveRiskState | None:
        """Most recently computed state (may be None before first tick)."""
        return self._last_state

    def tick(self) -> LiveRiskState | None:
        """Called every second by the engine scheduler.

        Returns the LiveRiskState if published (every interval_seconds),
        or None if it's not time yet.
        """
        now = time.time()
        if now - self._last_publish < self._interval:
            return None

        return self.force_publish()

    def force_publish(self) -> LiveRiskState:
        """Force an immediate computation and publish.

        Call this when a position opens/closes for instant dashboard update.
        """
        state = compute_live_risk(
            risk_manager=self._risk_manager,
            position_manager=self._position_manager,
        )

        state.publish_to_redis()
        self._last_publish = time.time()
        self._last_state = state

        return state
