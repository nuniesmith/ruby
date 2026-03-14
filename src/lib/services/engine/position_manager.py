"""
Position Manager — Stop-and-Reverse Micro Contract Strategy
=============================================================
Manages persistent 1-lot micro futures positions for core watchlist assets.
Each asset is always either LONG, SHORT, or FLAT. When a new breakout signal
fires in the opposite direction (and passes reversal gates), the position
is reversed: close current + open opposite.

3-Phase Bracket Walk:
    Phase 1 (INITIAL):  SL at entry ∓ SL×ATR.  When price hits TP1 → Phase 2.
    Phase 2 (BREAKEVEN): SL moved to entry (breakeven).  When TP2 hit → Phase 3.
    Phase 3 (TRAILING):  SL = EMA9 on 1m bars.  Exit at EMA9 touch or TP3.

State is persisted in Redis so positions survive engine restarts.

Usage::

    from lib.services.engine.position_manager import PositionManager

    pm = PositionManager(account_size=50_000)
    pm.load_state()  # restore from Redis

    # On breakout signal:
    orders = pm.process_signal(signal, bars_1m)

    # On every 1m bar close (position maintenance):
    orders = pm.update_all(bars_by_ticker)

    # Emit orders to Rithmic gateway
    for order in orders:
        gateway.send(order)

Environment Variables:
    PM_REVERSAL_MIN_CNN_PROB    — Min CNN prob to reverse (default 0.85)
    PM_REVERSAL_MIN_MTF_SCORE   — Min MTF score to reverse (default 0.60)
    PM_REVERSAL_COOLDOWN_SECS   — Cooldown between reversals (default 1800)
    PM_CHASE_MAX_ATR_FRACTION   — Max ATR fraction for market chase (default 0.50)
    PM_CHASE_MIN_CNN_PROB       — Min CNN prob for chase entry (default 0.90)
    PM_EMA_TRAIL_PERIOD         — EMA period for trailing (default 9)
    PM_EMA_TRAIL_USE_CLOSE      — Use bar close (not wick) for trail (default 1)
    PM_WINNING_REVERSAL_CNN     — CNN prob needed to reverse a winner (default 0.92)
    PM_MAX_POSITIONS            — Max simultaneous positions (default 5)
    PM_FOCUS_LOCK               — Enable one-asset focus lock (default 1)
    PM_PYRAMID_ENABLED          — Enable pyramiding into winning positions (default 1)
    PM_PYRAMID_MIN_CNN          — Min CNN prob to pyramid (default 0.65)
    PM_PYRAMID_MAX_LEVEL_LOW    — Max pyramid level for CNN 65–79% (default 2)
    PM_PYRAMID_MAX_LEVEL_HIGH   — Max pyramid level for CNN ≥ 80% (default 3)
    PM_PYRAMID_MAX_RISK_PCT     — Max total account risk for scaled position (default 0.015)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("engine.position_manager")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

REVERSAL_MIN_CNN_PROB = float(os.getenv("PM_REVERSAL_MIN_CNN_PROB", "0.85"))
REVERSAL_MIN_MTF_SCORE = float(os.getenv("PM_REVERSAL_MIN_MTF_SCORE", "0.60"))
REVERSAL_COOLDOWN_SECS = int(os.getenv("PM_REVERSAL_COOLDOWN_SECS", "1800"))
CHASE_MAX_ATR_FRACTION = float(os.getenv("PM_CHASE_MAX_ATR_FRACTION", "0.50"))
CHASE_MIN_CNN_PROB = float(os.getenv("PM_CHASE_MIN_CNN_PROB", "0.90"))
EMA_TRAIL_PERIOD = int(os.getenv("PM_EMA_TRAIL_PERIOD", "9"))
EMA_TRAIL_USE_CLOSE = os.getenv("PM_EMA_TRAIL_USE_CLOSE", "1") == "1"
WINNING_REVERSAL_CNN = float(os.getenv("PM_WINNING_REVERSAL_CNN", "0.92"))
MAX_POSITIONS = int(os.getenv("PM_MAX_POSITIONS", "5"))
FOCUS_LOCK_ENABLED = os.getenv("PM_FOCUS_LOCK", "1") == "1"
PYRAMID_ENABLED = os.getenv("PM_PYRAMID_ENABLED", "1") == "1"
PYRAMID_MIN_CNN_PROB = float(os.getenv("PM_PYRAMID_MIN_CNN", "0.65"))
PYRAMID_MAX_LEVEL_LOW = int(os.getenv("PM_PYRAMID_MAX_LEVEL_LOW", "2"))  # CNN 65–79%
PYRAMID_MAX_LEVEL_HIGH = int(os.getenv("PM_PYRAMID_MAX_LEVEL_HIGH", "3"))  # CNN ≥ 80%
PYRAMID_MAX_RISK_PCT = float(os.getenv("PM_PYRAMID_MAX_RISK_PCT", "0.015"))  # 1.5%

# Redis key prefix for persisting position state
REDIS_KEY_PREFIX = "engine:position:"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BracketPhase(StrEnum):
    """3-phase bracket progression."""

    INITIAL = "initial"
    """Phase 1: SL at original level. Waiting for TP1."""

    BREAKEVEN = "breakeven"
    """Phase 2: SL moved to entry (breakeven). TP1 was hit. Waiting for TP2."""

    TRAILING = "trailing"
    """Phase 3: SL = EMA9. TP2 was hit. Trailing toward TP3."""


class OrderAction(StrEnum):
    """Order action types emitted by the position manager."""

    BUY = "BUY"
    SELL = "SELL"
    MODIFY_STOP = "MODIFY_STOP"
    CANCEL = "CANCEL"


class OrderType(StrEnum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OrderCommand:
    """An order instruction to be sent to the Rithmic gateway.

    The gateway translates these into exchange order submissions.
    """

    symbol: str
    action: OrderAction
    order_type: OrderType
    quantity: int = 1
    price: float = 0.0
    stop_price: float = 0.0
    reason: str = ""
    position_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": round(self.price, 6),
            "stop_price": round(self.stop_price, 6),
            "reason": self.reason,
            "position_id": self.position_id,
            "timestamp": self.timestamp,
        }


@dataclass
class MicroPosition:
    """Tracks a single micro contract position for stop-and-reverse.

    Represents the complete state of a 1-lot micro futures position including
    bracket levels, phase tracking, EMA9 trailing state, and performance metrics.
    Serialisable to/from Redis for persistence across engine restarts.
    """

    # --- Identity ---
    symbol: str = ""
    ticker: str = ""
    position_id: str = ""

    # --- Core position ---
    direction: str = ""  # "LONG" or "SHORT"
    contracts: int = 1
    entry_price: float = 0.0
    entry_time: str = ""
    current_price: float = 0.0

    # --- Bracket levels ---
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0

    # --- ATR at entry (for reference) ---
    entry_atr: float = 0.0

    # --- Phase tracking ---
    phase: BracketPhase = BracketPhase.INITIAL
    tp1_hit: bool = False
    tp1_hit_time: str = ""
    tp2_hit: bool = False
    tp2_hit_time: str = ""
    tp3_hit: bool = False
    tp3_hit_time: str = ""
    breakeven_set: bool = False

    # --- EMA9 trailing state ---
    ema9_trail_active: bool = False
    ema9_trail_price: float = 0.0
    ema9_last_value: float = 0.0

    # --- Performance tracking ---
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # --- Source signal info ---
    breakout_type: str = ""
    cnn_prob: float = 0.0
    mtf_score: float = 0.0
    session_key: str = ""
    signal_quality: float = 0.0

    # --- Reversal tracking ---
    reversal_count: int = 0
    last_reversal_time: str = ""

    # --- Pyramid tracking ---
    pyramid_level: int = 0  # 0=base position, 1=first add, 2=second add, 3=third add
    pyramid_contracts: int = 0  # extra contracts added via pyramiding (total = contracts + pyramid_contracts)
    pyramid_stop: float = 0.0  # stop loss level set by last pyramid level
    last_pyramid_time: str = ""  # ISO timestamp of last pyramid add

    # --- Timestamps ---
    created_at: str = ""
    updated_at: str = ""
    closed_at: str = ""
    close_reason: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if not self.position_id:
            ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
            self.position_id = f"{self.symbol}_{self.direction}_{ts}"

    # --- Computed properties ---

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"

    @property
    def is_active(self) -> bool:
        return self.direction in ("LONG", "SHORT") and not self.closed_at

    @property
    def signed_pnl_ticks(self) -> float:
        """Unrealised P&L in price-distance units (not dollar-weighted)."""
        if not self.is_active or self.entry_price <= 0:
            return 0.0
        if self.is_long:
            return self.current_price - self.entry_price
        return self.entry_price - self.current_price

    @property
    def is_winning(self) -> bool:
        return self.signed_pnl_ticks > 0

    @property
    def is_losing(self) -> bool:
        return self.signed_pnl_ticks < 0

    @property
    def r_multiple(self) -> float:
        """Current R-multiple (P&L / initial risk)."""
        initial_risk = abs(self.entry_price - self.stop_loss) if self.phase == BracketPhase.INITIAL else self.entry_atr
        if initial_risk <= 0:
            return 0.0
        return self.signed_pnl_ticks / initial_risk

    @property
    def hold_duration_seconds(self) -> float:
        """How long the position has been open, in seconds."""
        if not self.entry_time:
            return 0.0
        try:
            entry_dt = datetime.fromisoformat(self.entry_time)
            now = datetime.now(UTC)
            if entry_dt.tzinfo is None:
                # Assume UTC if no timezone
                return (now.replace(tzinfo=None) - entry_dt).total_seconds()
            return (now - entry_dt).total_seconds()
        except (ValueError, TypeError):
            return 0.0

    # --- Serialisation ---

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict for Redis persistence."""
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MicroPosition:
        """Deserialise from a dict (e.g. loaded from Redis).

        Tolerates both old states (missing new fields — defaults apply) and
        future states (extra unknown fields — silently dropped) so that Redis
        upgrades never crash the engine on restart.
        """
        import dataclasses as _dc

        d = dict(d)  # shallow copy
        phase_str = d.pop("phase", "initial")
        try:
            phase = BracketPhase(phase_str)
        except ValueError:
            phase = BracketPhase.INITIAL

        # Drop keys that no longer exist as dataclass fields (forward-compat)
        known = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}

        return cls(phase=phase, **d)

    def update_price(self, price: float) -> None:
        """Update the current market price and excursion tracking."""
        self.current_price = price
        self.updated_at = datetime.now(UTC).isoformat()

        pnl_ticks = self.signed_pnl_ticks
        if pnl_ticks > self.max_favorable_excursion:
            self.max_favorable_excursion = pnl_ticks
        if pnl_ticks < 0 and abs(pnl_ticks) > self.max_adverse_excursion:
            self.max_adverse_excursion = abs(pnl_ticks)


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------


class PositionManager:
    """Manages persistent micro contract positions for the stop-and-reverse strategy.

    Responsibilities:
        1. Track active micro positions per asset (persisted in Redis).
        2. Process incoming breakout signals → decide: hold / reverse / enter / exit.
        3. Manage bracket phases (initial → breakeven → trailing).
        4. Compute EMA9 trail updates every bar.
        5. Emit ``OrderCommand`` objects for the Rithmic gateway.
        6. Persist position history for analysis.

    Thread safety:
        Not thread-safe.  Expected to be called from the engine's single
        scheduler thread.  If called from multiple threads, wrap in a lock.
    """

    def __init__(
        self,
        account_size: float = 50_000.0,
        core_tickers: frozenset[str] | None = None,
    ) -> None:
        self.account_size = account_size
        self._positions: dict[str, MicroPosition] = {}  # ticker → active position
        self._history: list[MicroPosition] = []  # closed positions (in-memory, flushed to DB)
        self._focus_asset: str | None = None  # ticker currently under focus lock

        # Import core tickers from models if not provided
        if core_tickers is None:
            try:
                from lib.core.models import CORE_TICKERS

                self._core_tickers = CORE_TICKERS
            except ImportError:
                self._core_tickers = frozenset()
        else:
            self._core_tickers = core_tickers

        logger.info(
            "PositionManager initialised: account=$%s, core_tickers=%s, max_positions=%d",
            f"{account_size:,.0f}",
            sorted(self._core_tickers),
            MAX_POSITIONS,
        )

    # ------------------------------------------------------------------
    # State persistence (Redis)
    # ------------------------------------------------------------------

    def load_state(self) -> int:
        """Load position state from Redis.  Returns count of positions loaded."""
        loaded = 0
        try:
            from lib.core.cache import cache_get

            # Restore the manager-level state (focus lock, etc.)
            mgr_key = f"{REDIS_KEY_PREFIX}_manager_state"
            mgr_raw = cache_get(mgr_key)
            if mgr_raw is not None:
                mgr_raw_str = mgr_raw.decode("utf-8") if isinstance(mgr_raw, bytes) else mgr_raw
                mgr_data = json.loads(mgr_raw_str)
                self._focus_asset = mgr_data.get("focus_asset")
                if self._focus_asset:
                    logger.info("Restored focus lock: %s", self._focus_asset)

            for ticker in self._core_tickers:
                key = f"{REDIS_KEY_PREFIX}{ticker}"
                raw = cache_get(key)
                if raw is None:
                    continue
                raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                data = json.loads(raw_str)
                pos = MicroPosition.from_dict(data)
                if pos.is_active:
                    self._positions[ticker] = pos
                    loaded += 1
                    logger.info(
                        "Restored position: %s %s %s @ %.4f (phase=%s)",
                        pos.symbol,
                        pos.direction,
                        ticker,
                        pos.entry_price,
                        pos.phase.value,
                    )
        except ImportError:
            logger.warning("Redis not available — starting with empty position state")
        except Exception as exc:
            logger.error("Failed to load position state from Redis: %s", exc)

        logger.info("Loaded %d active positions from Redis", loaded)
        return loaded

    def save_state(self) -> None:
        """Persist all active positions and manager state to Redis."""
        try:
            from lib.core.cache import cache_set

            # Persist manager-level state (focus lock, etc.)
            mgr_key = f"{REDIS_KEY_PREFIX}_manager_state"
            mgr_state = {"focus_asset": self._focus_asset}
            cache_set(mgr_key, json.dumps(mgr_state).encode(), ttl=86400)

            for ticker, pos in self._positions.items():
                key = f"{REDIS_KEY_PREFIX}{ticker}"
                cache_set(key, json.dumps(pos.to_dict()).encode(), ttl=86400)  # 24h TTL
        except ImportError:
            logger.debug("Redis not available — cannot persist position state")
        except Exception as exc:
            logger.error("Failed to save position state to Redis: %s", exc)

    def _clear_position_from_redis(self, ticker: str) -> None:
        """Remove a position's Redis key after it's closed."""
        try:
            from lib.core.cache import cache_set

            key = f"{REDIS_KEY_PREFIX}{ticker}"
            cache_set(key, b"", ttl=1)  # effectively delete by setting tiny TTL
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Position accessors
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> MicroPosition | None:
        """Get the active position for a ticker, or None."""
        pos = self._positions.get(ticker)
        if pos is not None and pos.is_active:
            return pos
        return None

    def get_all_positions(self) -> dict[str, MicroPosition]:
        """Return all active positions."""
        return {t: p for t, p in self._positions.items() if p.is_active}

    def get_position_count(self) -> int:
        """Number of currently active positions."""
        return sum(1 for p in self._positions.values() if p.is_active)

    def get_history(self) -> list[MicroPosition]:
        """Return closed position history (in-memory)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Focus lock — one asset at a time
    # ------------------------------------------------------------------

    @property
    def focus_asset(self) -> str | None:
        """The ticker currently locked in focus, or None if no lock."""
        return self._focus_asset

    def can_trade(self, ticker: str) -> bool:
        """Return True if the given ticker is allowed to trade right now.

        When focus lock is disabled (PM_FOCUS_LOCK=0), always returns True.
        When focus lock is enabled:
          - If no position is open, any ticker can start a position.
          - If a position is open, only the focused ticker is allowed
            (signals for other tickers are rejected silently).
        """
        if not FOCUS_LOCK_ENABLED:
            return True
        if self._focus_asset is None:
            return True
        return ticker == self._focus_asset

    def set_focus(self, ticker: str) -> None:
        """Lock focus onto *ticker* — called when a new position is opened."""
        if FOCUS_LOCK_ENABLED and self._focus_asset != ticker:
            logger.info("🔒 Focus lock: %s (was: %s)", ticker, self._focus_asset or "none")
            self._focus_asset = ticker

    def clear_focus(self) -> None:
        """Release focus lock — called when the focused position is closed."""
        if self._focus_asset is not None:
            logger.info("🔓 Focus lock released (was: %s)", self._focus_asset)
        self._focus_asset = None

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def process_signal(
        self,
        signal: Any,
        bars_1m: pd.DataFrame | None = None,
        *,
        range_config: Any = None,
    ) -> list[OrderCommand]:
        """Process a breakout signal and return order commands.

        The signal should be a BreakoutResult (or any object with the
        expected attributes: symbol, direction, trigger_price, breakout_type,
        cnn_prob, cnn_signal, filter_passed, mtf_score, atr_value, range_high,
        range_low).

        Returns a list of OrderCommand objects.  The caller is responsible
        for sending them to the Rithmic gateway.
        """
        orders: list[OrderCommand] = []

        ticker = getattr(signal, "symbol", "")
        direction = getattr(signal, "direction", "")
        trigger_price = getattr(signal, "trigger_price", 0.0)
        breakout_detected = getattr(signal, "breakout_detected", False)

        if not breakout_detected or not ticker or not direction or trigger_price <= 0:
            return orders

        # Only manage positions for core watchlist assets
        if ticker not in self._core_tickers:
            logger.debug("Signal for %s is not in core watchlist — ignoring for position management", ticker)
            return orders

        # Focus lock — reject signals for non-focused assets while a position is open
        if not self.can_trade(ticker):
            logger.debug(
                "Focus lock active (%s) — ignoring signal for %s",
                self._focus_asset,
                ticker,
            )
            return orders

        # Check if we already have a position for this ticker
        existing = self.get_position(ticker)

        if existing is not None:
            # Same direction — check for pyramid opportunity
            if existing.direction == direction:
                current_price = self._get_current_price(bars_1m, trigger_price)
                cnn_prob = getattr(signal, "cnn_prob", 0.0) or 0.0
                regime = getattr(signal, "regime", "") or ""
                wave_ratio = getattr(signal, "wave_ratio", 0.0) or 0.0

                pyramid_action = self.get_next_pyramid_level(
                    existing,
                    current_price=current_price,
                    cnn_prob=cnn_prob,
                    regime=regime,
                    wave_ratio=wave_ratio,
                )
                if pyramid_action is not None:
                    orders.extend(self.apply_pyramid(existing, pyramid_action, current_price))
                else:
                    logger.debug(
                        "Signal confirms existing %s position in %s — holding (pyramid gates not met)",
                        existing.direction,
                        ticker,
                    )
                return orders

            # Opposite direction — evaluate reversal
            if self._should_reverse(existing, signal):
                orders.extend(self._reverse_position(existing, signal, bars_1m, range_config))
            else:
                logger.info(
                    "Reversal gate rejected for %s: %s → %s (cnn=%.3f, mtf=%.2f)",
                    ticker,
                    existing.direction,
                    direction,
                    getattr(signal, "cnn_prob", 0.0) or 0.0,
                    getattr(signal, "mtf_score", 0.0) or 0.0,
                )
        else:
            # No existing position — evaluate new entry
            if self.get_position_count() >= MAX_POSITIONS:
                logger.info("Max positions (%d) reached — cannot open %s %s", MAX_POSITIONS, direction, ticker)
                return orders

            orders.extend(self._open_position(signal, bars_1m, range_config))

        return orders

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def _open_position(
        self,
        signal: Any,
        bars_1m: pd.DataFrame | None,
        range_config: Any,
    ) -> list[OrderCommand]:
        """Open a new micro position from a breakout signal."""
        ticker = getattr(signal, "symbol", "")
        direction = getattr(signal, "direction", "")
        trigger_price = getattr(signal, "trigger_price", 0.0)
        atr = getattr(signal, "atr_value", 0.0) or trigger_price * 0.005
        range_high = getattr(signal, "range_high", 0.0)
        range_low = getattr(signal, "range_low", 0.0)
        cnn_prob = getattr(signal, "cnn_prob", None) or 0.0
        mtf_score = getattr(signal, "mtf_score", None) or 0.0
        session_key = getattr(signal, "session_key", "") or ""
        btype = str(getattr(signal, "breakout_type", ""))

        # Extract bracket multipliers from range_config or use defaults
        sl_mult = 1.5
        tp1_mult = 2.0
        tp2_mult = 3.0
        tp3_mult = 4.5

        if range_config is not None:
            sl_mult = getattr(range_config, "sl_atr_mult", sl_mult)
            tp1_mult = getattr(range_config, "tp1_atr_mult", tp1_mult)
            tp2_mult = getattr(range_config, "tp2_atr_mult", tp2_mult)
            tp3_mult = getattr(range_config, "tp3_atr_mult", tp3_mult)

        # Compute bracket levels
        if direction == "LONG":
            entry_target = range_high  # limit at range high (just above breakout)
            sl = trigger_price - sl_mult * atr
            tp1 = trigger_price + tp1_mult * atr
            tp2 = trigger_price + tp2_mult * atr
            tp3 = trigger_price + tp3_mult * atr
        else:
            entry_target = range_low
            sl = trigger_price + sl_mult * atr
            tp1 = trigger_price - tp1_mult * atr
            tp2 = trigger_price - tp2_mult * atr
            tp3 = trigger_price - tp3_mult * atr

        # Decide order type: limit at range edge vs market chase
        current_price = self._get_current_price(bars_1m, trigger_price)
        order_type, order_price = self._decide_entry_type(
            direction=direction,
            entry_target=entry_target,
            trigger_price=trigger_price,
            current_price=current_price,
            atr=atr,
            cnn_prob=cnn_prob,
        )

        # Resolve the human-readable symbol name
        symbol_name = self._ticker_to_name(ticker)

        pos = MicroPosition(
            symbol=symbol_name,
            ticker=ticker,
            direction=direction,
            contracts=1,
            entry_price=order_price if order_type == OrderType.MARKET else entry_target,
            entry_time=datetime.now(UTC).isoformat(),
            current_price=current_price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            entry_atr=atr,
            phase=BracketPhase.INITIAL,
            breakout_type=btype,
            cnn_prob=cnn_prob,
            mtf_score=mtf_score,
            session_key=session_key,
        )

        self._positions[ticker] = pos
        self.set_focus(ticker)

        action = OrderAction.BUY if direction == "LONG" else OrderAction.SELL

        entry_order = OrderCommand(
            symbol=ticker,
            action=action,
            order_type=order_type,
            quantity=1,
            price=order_price if order_type == OrderType.LIMIT else 0.0,
            reason=f"New {direction} entry: {btype} breakout (CNN={cnn_prob:.3f}, MTF={mtf_score:.2f})",
            position_id=pos.position_id,
        )

        # Also submit the initial stop-loss order
        stop_action = OrderAction.SELL if direction == "LONG" else OrderAction.BUY
        stop_order = OrderCommand(
            symbol=ticker,
            action=stop_action,
            order_type=OrderType.STOP,
            quantity=1,
            stop_price=sl,
            reason=f"Initial SL @ {sl:.4f} ({sl_mult}×ATR from entry)",
            position_id=pos.position_id,
        )

        logger.info(
            "📈 OPEN %s %s: entry=%.4f SL=%.4f TP1=%.4f TP2=%.4f TP3=%.4f (type=%s, CNN=%.3f)",
            direction,
            ticker,
            pos.entry_price,
            sl,
            tp1,
            tp2,
            tp3,
            btype,
            cnn_prob,
        )

        self.save_state()
        return [entry_order, stop_order]

    def _reverse_position(
        self,
        existing: MicroPosition,
        signal: Any,
        bars_1m: pd.DataFrame | None,
        range_config: Any,
    ) -> list[OrderCommand]:
        """Reverse an existing position: close current + open opposite."""
        orders: list[OrderCommand] = []

        ticker = existing.ticker
        new_direction = getattr(signal, "direction", "")

        # Step 1: Close existing position with a market order
        close_action = OrderAction.SELL if existing.is_long else OrderAction.BUY
        close_order = OrderCommand(
            symbol=ticker,
            action=close_action,
            order_type=OrderType.MARKET,
            quantity=existing.contracts,
            reason=f"REVERSE: closing {existing.direction} to flip {new_direction}",
            position_id=existing.position_id,
        )
        orders.append(close_order)

        # Archive the closed position
        self._close_position(
            existing,
            reason=f"Reversed to {new_direction}",
            close_price=existing.current_price,
        )

        # Step 2: Open new position in opposite direction
        new_orders = self._open_position(signal, bars_1m, range_config)
        if new_orders:
            # Update reversal count on the new position
            new_pos_or_none = self.get_position(ticker)
            if new_pos_or_none is not None:
                new_pos_or_none.reversal_count = existing.reversal_count + 1
                new_pos_or_none.last_reversal_time = datetime.now(UTC).isoformat()

        orders.extend(new_orders)

        logger.info(
            "🔄 REVERSE %s: %s → %s (reversal #%d)",
            ticker,
            existing.direction,
            new_direction,
            existing.reversal_count + 1,
        )

        return orders

    def _close_position(
        self,
        pos: MicroPosition,
        reason: str,
        close_price: float = 0.0,
    ) -> None:
        """Close a position and move it to history."""
        now = datetime.now(UTC).isoformat()
        pos.closed_at = now
        pos.close_reason = reason
        pos.updated_at = now
        if close_price > 0:
            pos.current_price = close_price
            # Compute realised P&L in ticks
            if pos.is_long:
                pos.realized_pnl = close_price - pos.entry_price
            else:
                pos.realized_pnl = pos.entry_price - close_price

        self._history.append(pos)

        ticker = pos.ticker
        if ticker in self._positions:
            del self._positions[ticker]

        self._clear_position_from_redis(ticker)

        # Release focus lock when last position closes
        if not self._positions:
            self.clear_focus()

        logger.info(
            "📉 CLOSED %s %s: entry=%.4f exit=%.4f pnl_ticks=%.4f R=%.2f reason=%s",
            pos.direction,
            ticker,
            pos.entry_price,
            pos.current_price,
            pos.realized_pnl,
            pos.r_multiple,
            reason,
        )

    # ------------------------------------------------------------------
    # Bracket management & EMA9 trailing
    # ------------------------------------------------------------------

    def update_all(
        self,
        bars_by_ticker: dict[str, pd.DataFrame],
    ) -> list[OrderCommand]:
        """Update all active positions with latest 1m bar data.

        Call this on every 1-minute bar close for position maintenance.
        Returns any order commands generated by bracket transitions or
        EMA9 trailing stop triggers.
        """
        orders: list[OrderCommand] = []
        tickers_to_close: list[tuple[str, str, float]] = []  # (ticker, reason, price)

        for ticker, pos in list(self._positions.items()):
            if not pos.is_active:
                continue

            bars = bars_by_ticker.get(ticker)
            if bars is None or bars.empty:
                continue

            current_price = float(bars["Close"].iloc[-1])
            pos.update_price(current_price)

            # Check TP3 hit first — hard exit at full target takes priority over
            # any trailing stop, matching C# CheckPhase3Exits which checks tp3Hit
            # before ema9Stop.  Without this order a bar that simultaneously
            # touches the EMA9 trail and TP3 would close at the stop price
            # instead of booking the full profit.
            tp3_hit = self._check_tp3_hit(pos)
            if tp3_hit:
                tickers_to_close.append((ticker, "TP3 hit — full target achieved", current_price))
                continue

            # Check stop loss hit (EMA9 trail or original SL)
            stop_hit, stop_reason = self._check_stop_hit(pos, bars)
            if stop_hit:
                tickers_to_close.append((ticker, stop_reason, current_price))
                continue

            # Phase-specific updates
            phase_orders = self._update_bracket_phase(pos, bars)
            orders.extend(phase_orders)

        # Process closures
        for ticker, reason, price in tickers_to_close:
            closing_pos = self._positions.get(ticker)
            if closing_pos is None:
                continue
            pos = closing_pos

            close_action = OrderAction.SELL if pos.is_long else OrderAction.BUY
            orders.append(
                OrderCommand(
                    symbol=ticker,
                    action=close_action,
                    order_type=OrderType.MARKET,
                    quantity=pos.contracts,
                    reason=reason,
                    position_id=pos.position_id,
                )
            )
            self._close_position(pos, reason=reason, close_price=price)

        if orders:
            self.save_state()

        return orders

    def _update_bracket_phase(
        self,
        pos: MicroPosition,
        bars: pd.DataFrame,
    ) -> list[OrderCommand]:
        """Check for bracket phase transitions and return any resulting orders."""
        orders: list[OrderCommand] = []
        current_price = pos.current_price
        now_str = datetime.now(UTC).isoformat()

        if pos.phase == BracketPhase.INITIAL:
            # Check TP1 hit
            tp1_hit = (pos.is_long and current_price >= pos.tp1) or (pos.is_short and current_price <= pos.tp1)
            if tp1_hit:
                pos.tp1_hit = True
                pos.tp1_hit_time = now_str
                pos.phase = BracketPhase.BREAKEVEN
                pos.breakeven_set = True

                # Move stop to breakeven (entry price)
                pos.stop_loss = pos.entry_price

                orders.append(
                    OrderCommand(
                        symbol=pos.ticker,
                        action=OrderAction.MODIFY_STOP,
                        order_type=OrderType.STOP,
                        quantity=pos.contracts,
                        stop_price=pos.entry_price,
                        reason=f"TP1 hit @ {current_price:.4f} — stop moved to breakeven ({pos.entry_price:.4f})",
                        position_id=pos.position_id,
                    )
                )

                logger.info(
                    "✅ TP1 HIT %s %s @ %.4f → stop moved to breakeven (%.4f)",
                    pos.direction,
                    pos.ticker,
                    current_price,
                    pos.entry_price,
                )

        elif pos.phase == BracketPhase.BREAKEVEN:
            # Check TP2 hit → activate EMA9 trailing
            tp2_hit = (pos.is_long and current_price >= pos.tp2) or (pos.is_short and current_price <= pos.tp2)
            if tp2_hit:
                pos.tp2_hit = True
                pos.tp2_hit_time = now_str
                pos.phase = BracketPhase.TRAILING
                pos.ema9_trail_active = True

                # Compute initial EMA9 trail price
                ema9 = self._compute_ema9(bars)
                if ema9 is not None:
                    pos.ema9_last_value = ema9
                    pos.ema9_trail_price = ema9
                    pos.stop_loss = ema9

                    orders.append(
                        OrderCommand(
                            symbol=pos.ticker,
                            action=OrderAction.MODIFY_STOP,
                            order_type=OrderType.STOP,
                            quantity=pos.contracts,
                            stop_price=ema9,
                            reason=f"TP2 hit @ {current_price:.4f} — EMA9 trail activated (trail={ema9:.4f})",
                            position_id=pos.position_id,
                        )
                    )

                logger.info(
                    "✅ TP2 HIT %s %s @ %.4f → EMA9 trailing activated (EMA9=%.4f)",
                    pos.direction,
                    pos.ticker,
                    current_price,
                    ema9 or 0.0,
                )

        elif pos.phase == BracketPhase.TRAILING:
            # Update EMA9 trailing stop
            ema9 = self._compute_ema9(bars)
            if ema9 is not None and ema9 > 0:
                pos.ema9_last_value = ema9

                # Only move the trail in the favorable direction (ratchet)
                should_update = (pos.is_long and ema9 > pos.ema9_trail_price) or (
                    pos.is_short and ema9 < pos.ema9_trail_price
                )

                if should_update:
                    pos.ema9_trail_price = ema9
                    pos.stop_loss = ema9

                    orders.append(
                        OrderCommand(
                            symbol=pos.ticker,
                            action=OrderAction.MODIFY_STOP,
                            order_type=OrderType.STOP,
                            quantity=pos.contracts,
                            stop_price=ema9,
                            reason=f"EMA9 trail updated to {ema9:.4f} (R={pos.r_multiple:.2f})",
                            position_id=pos.position_id,
                        )
                    )

        pos.updated_at = now_str
        return orders

    def _check_stop_hit(self, pos: MicroPosition, bars: pd.DataFrame) -> tuple[bool, str]:
        """Check if the stop loss was hit on the latest bar."""
        if bars.empty:
            return False, ""

        last_bar = bars.iloc[-1]
        bar_low = float(last_bar.get("Low", pos.current_price))
        bar_high = float(last_bar.get("High", pos.current_price))

        if pos.is_long and bar_low <= pos.stop_loss:
            phase_label = pos.phase.value
            return True, f"Stop hit @ {pos.stop_loss:.4f} (phase={phase_label}, low={bar_low:.4f})"

        if pos.is_short and bar_high >= pos.stop_loss:
            phase_label = pos.phase.value
            return True, f"Stop hit @ {pos.stop_loss:.4f} (phase={phase_label}, high={bar_high:.4f})"

        return False, ""

    def _check_tp3_hit(self, pos: MicroPosition) -> bool:
        """Check if TP3 was hit (hard exit cap)."""
        if pos.tp3 <= 0:
            return False

        if pos.is_long and pos.current_price >= pos.tp3:
            pos.tp3_hit = True
            pos.tp3_hit_time = datetime.now(UTC).isoformat()
            return True

        if pos.is_short and pos.current_price <= pos.tp3:
            pos.tp3_hit = True
            pos.tp3_hit_time = datetime.now(UTC).isoformat()
            return True

        return False

    def _compute_ema9(self, bars: pd.DataFrame) -> float | None:
        """Compute EMA9 from 1-minute bar closes."""
        if bars is None or len(bars) < EMA_TRAIL_PERIOD:
            return None

        try:
            closes = bars["Close"].astype(float)
            ema = closes.ewm(span=EMA_TRAIL_PERIOD, adjust=False).mean()
            return float(ema.iloc[-1])
        except Exception as exc:
            logger.debug("EMA9 computation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Pyramiding
    # ------------------------------------------------------------------

    def get_next_pyramid_level(
        self,
        pos: MicroPosition,
        current_price: float,
        cnn_prob: float,
        *,
        regime: str = "",
        wave_ratio: float = 0.0,
    ) -> dict[str, Any] | None:
        """Determine whether and how to pyramid into an existing winning position.

        Called when a new confirming signal arrives for a ticker that already has
        an open position in the same direction.

        Args:
            pos: The existing MicroPosition.
            current_price: Latest price.
            cnn_prob: CNN breakout probability from the new signal (0–1).
            regime: Current regime string from RegimeDetector (e.g. "TRENDING_UP").
            wave_ratio: Wave dominance ratio from wave_analysis (> 1.5 = trending).

        Returns:
            A dict with pyramid action details, or ``None`` if pyramiding is
            not allowed right now.

            Dict keys:
                ``level``       — new pyramid level (1, 2, or 3)
                ``add_qty``     — contracts to add (always 1)
                ``new_stop``    — new stop loss price to set on all contracts
                ``reason``      — human-readable reason string
                ``r_at_entry``  — R-multiple at the time of the add

            Returns ``None`` if any gate fails (level cap, quality, risk, cooldown).
        """
        if not PYRAMID_ENABLED:
            return None

        # Gate 1: CNN quality
        if cnn_prob < PYRAMID_MIN_CNN_PROB:
            logger.debug(
                "Pyramid gate: CNN %.3f < %.3f (min required)",
                cnn_prob,
                PYRAMID_MIN_CNN_PROB,
            )
            return None

        # Gate 2: Position must be winning and moving in our direction
        r = pos.r_multiple
        next_level = pos.pyramid_level + 1

        # Minimum R needed for each pyramid level
        r_thresholds = {1: 1.0, 2: 2.0, 3: 3.0}
        min_r = r_thresholds.get(next_level, 99.0)
        if r < min_r:
            logger.debug(
                "Pyramid gate: R=%.2f < %.1f (needed for level %d)",
                r,
                min_r,
                next_level,
            )
            return None

        # Gate 3: Level cap based on CNN quality
        max_level = PYRAMID_MAX_LEVEL_HIGH if cnn_prob >= 0.80 else PYRAMID_MAX_LEVEL_LOW
        if next_level > max_level:
            logger.debug(
                "Pyramid gate: already at max level %d (CNN=%.3f caps at %d)",
                pos.pyramid_level,
                cnn_prob,
                max_level,
            )
            return None

        # Gate 4: Level 3 requires TRENDING regime AND wave_ratio > 1.5
        if next_level == 3:
            regime_upper = regime.upper()
            is_trending = "TRENDING" in regime_upper or "TREND" in regime_upper
            if not is_trending:
                logger.debug("Pyramid gate: L3 requires TRENDING regime (got '%s')", regime)
                return None
            if wave_ratio < 1.5:
                logger.debug("Pyramid gate: L3 requires wave_ratio > 1.5 (got %.2f)", wave_ratio)
                return None

        # Gate 5: Max risk — total position cannot exceed 1.5% of account
        # Each micro contract risk = distance_to_stop × tick_value
        # We approximate: 1 contract risk ≈ entry_atr × sl_mult × tick_multiplier
        # For simplicity, cap total contracts at 3
        total_contracts = pos.contracts + pos.pyramid_contracts + 1
        if total_contracts > 3:
            logger.debug("Pyramid gate: total contracts %d would exceed max 3", total_contracts)
            return None

        # Gate 6: Cooldown — don't add more than once per 15 minutes
        if pos.last_pyramid_time:
            try:
                last_dt = datetime.fromisoformat(pos.last_pyramid_time)
                now = datetime.now(UTC)
                if last_dt.tzinfo is None:
                    elapsed = (now.replace(tzinfo=None) - last_dt).total_seconds()
                else:
                    elapsed = (now - last_dt).total_seconds()
                if elapsed < 900:  # 15 minutes
                    logger.debug("Pyramid gate: cooldown %.0fs elapsed (need 900s)", elapsed)
                    return None
            except (ValueError, TypeError):
                pass

        # All gates passed — compute new stop level
        entry = pos.entry_price
        atr = pos.entry_atr or abs(current_price - entry) * 0.01

        if next_level == 1:
            # Move SL to breakeven
            new_stop = entry
            reason = f"Pyramid L1: +1 contract @ {current_price:.4f}, SL → breakeven ({entry:.4f})"
        elif next_level == 2:
            # SL to entry + 0.5R
            r_unit = abs(entry - pos.stop_loss) if pos.stop_loss else atr
            new_stop = entry + 0.5 * r_unit if pos.is_long else entry - 0.5 * r_unit
            reason = f"Pyramid L2: +1 contract @ {current_price:.4f}, SL → entry+0.5R ({new_stop:.4f})"
        else:  # level 3
            # SL to current − 1R
            r_unit = abs(entry - pos.stop_loss) if pos.stop_loss else atr
            new_stop = current_price - r_unit if pos.is_long else current_price + r_unit
            reason = f"Pyramid L3: +1 contract @ {current_price:.4f}, SL → price-1R ({new_stop:.4f})"

        logger.info(
            "✅ PYRAMID L%d %s %s: add 1 @ %.4f, new_stop=%.4f (R=%.2f, CNN=%.3f)",
            next_level,
            pos.direction,
            pos.ticker,
            current_price,
            new_stop,
            r,
            cnn_prob,
        )

        return {
            "level": next_level,
            "add_qty": 1,
            "new_stop": new_stop,
            "reason": reason,
            "r_at_entry": round(r, 2),
        }

    def apply_pyramid(
        self,
        pos: MicroPosition,
        pyramid_action: dict[str, Any],
        current_price: float,
    ) -> list[OrderCommand]:
        """Apply a pyramid action returned by ``get_next_pyramid_level()``.

        Updates the position's pyramid tracking fields and returns the
        OrderCommands needed: one entry order + one stop-modify order.

        Args:
            pos: The existing MicroPosition to pyramid into.
            pyramid_action: Dict returned by ``get_next_pyramid_level()``.
            current_price: Price to use for the market entry order.

        Returns:
            List of OrderCommands: [market_entry, modify_stop].
        """
        level = pyramid_action["level"]
        new_stop = pyramid_action["new_stop"]
        reason = pyramid_action["reason"]

        # Update position tracking
        pos.pyramid_level = level
        pos.pyramid_contracts += 1
        pos.pyramid_stop = new_stop
        pos.stop_loss = new_stop
        pos.last_pyramid_time = datetime.now(UTC).isoformat()

        # Entry order: add 1 contract at market
        entry_action = OrderAction.BUY if pos.is_long else OrderAction.SELL
        entry_order = OrderCommand(
            symbol=pos.ticker,
            action=entry_action,
            order_type=OrderType.MARKET,
            quantity=1,
            price=0.0,
            reason=reason,
            position_id=pos.position_id,
        )

        # Stop-modify order: move the stop on ALL contracts to new_stop
        stop_order = OrderCommand(
            symbol=pos.ticker,
            action=OrderAction.MODIFY_STOP,
            order_type=OrderType.STOP,
            quantity=pos.contracts + pos.pyramid_contracts,
            stop_price=new_stop,
            reason=f"Pyramid L{level} stop adjustment to {new_stop:.4f}",
            position_id=pos.position_id,
        )

        self.save_state()
        return [entry_order, stop_order]

    # ------------------------------------------------------------------
    # Reversal gate
    # ------------------------------------------------------------------

    def _should_reverse(self, existing: MicroPosition, signal: Any) -> bool:
        """Decide whether to reverse an existing position.

        Reversal is expensive (pay spread twice), so multiple gates
        must pass to prevent whipsawing.
        """
        new_direction = getattr(signal, "direction", "")

        # Gate 1: Must be opposite direction
        if new_direction == existing.direction:
            return False

        # Gate 2: CNN confidence must be high
        cnn_prob = getattr(signal, "cnn_prob", None) or 0.0
        min_prob = REVERSAL_MIN_CNN_PROB
        if existing.is_winning:
            # Need even higher conviction to flip a winning position
            min_prob = WINNING_REVERSAL_CNN
        if cnn_prob < min_prob:
            logger.debug(
                "Reversal gate: CNN prob %.3f < %.3f (winning=%s)",
                cnn_prob,
                min_prob,
                existing.is_winning,
            )
            return False

        # Gate 3: Quality filter must have passed
        filter_passed = getattr(signal, "filter_passed", None)
        if filter_passed is False:
            logger.debug("Reversal gate: filter not passed")
            return False

        # Gate 4: Cooldown — minimum time since last entry/reversal
        cooldown = REVERSAL_COOLDOWN_SECS
        entry_time_str = existing.last_reversal_time or existing.entry_time
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                now = datetime.now(UTC)
                if entry_dt.tzinfo is None:
                    elapsed = (now.replace(tzinfo=None) - entry_dt).total_seconds()
                else:
                    elapsed = (now - entry_dt).total_seconds()
                if elapsed < cooldown:
                    logger.debug(
                        "Reversal gate: cooldown — %.0fs since last entry (need %ds)",
                        elapsed,
                        cooldown,
                    )
                    return False
            except (ValueError, TypeError):
                pass

        # Gate 5: MTF alignment — higher TF must agree with new direction
        mtf_score = getattr(signal, "mtf_score", None) or 0.0
        if mtf_score < REVERSAL_MIN_MTF_SCORE:
            logger.debug(
                "Reversal gate: MTF score %.2f < %.2f",
                mtf_score,
                REVERSAL_MIN_MTF_SCORE,
            )
            return False

        # Gate 6: If current position is winning, be very selective
        if existing.is_winning and existing.r_multiple > 1.0 and cnn_prob < 0.95:
            # Don't flip a position that's already at 1R+ profit unless
            # the new signal is exceptional
            logger.debug(
                "Reversal gate: won't flip +%.2fR winner without CNN ≥ 0.95 (got %.3f)",
                existing.r_multiple,
                cnn_prob,
            )
            return False

        logger.info(
            "Reversal gate PASSED for %s: %s → %s (CNN=%.3f, MTF=%.2f, R=%.2f)",
            existing.ticker,
            existing.direction,
            new_direction,
            cnn_prob,
            mtf_score,
            existing.r_multiple,
        )
        return True

    # ------------------------------------------------------------------
    # Entry type decision
    # ------------------------------------------------------------------

    def _decide_entry_type(
        self,
        direction: str,
        entry_target: float,
        trigger_price: float,
        current_price: float,
        atr: float,
        cnn_prob: float,
    ) -> tuple[OrderType, float]:
        """Decide whether to use a limit order at range edge or market order.

        Returns (order_type, price) — price is the limit price for LIMIT
        orders or the current price for MARKET orders.
        """
        if current_price <= 0 or atr <= 0:
            return OrderType.MARKET, trigger_price

        # How far has price already moved past the trigger?
        overshoot = current_price - trigger_price if direction == "LONG" else trigger_price - current_price

        max_chase = CHASE_MAX_ATR_FRACTION * atr

        if overshoot <= 0:
            # Price hasn't reached trigger yet — use limit at entry target
            return OrderType.LIMIT, entry_target

        if overshoot <= max_chase and cnn_prob >= CHASE_MIN_CNN_PROB:
            # Price moved past trigger but within chase tolerance + high conviction
            logger.info(
                "Chase entry: %s overshoot=%.4f (%.1f%% ATR), CNN=%.3f — using MARKET",
                direction,
                overshoot,
                (overshoot / atr) * 100,
                cnn_prob,
            )
            return OrderType.MARKET, current_price

        if overshoot > max_chase:
            # Too far gone — still enter with limit at trigger (may not fill)
            logger.info(
                "Price moved too far (%.1f%% ATR) — placing limit at trigger %.4f",
                (overshoot / atr) * 100,
                trigger_price,
            )
            return OrderType.LIMIT, trigger_price

        # Default: limit at entry target
        return OrderType.LIMIT, entry_target

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def close_all(self, reason: str = "Manual close all") -> list[OrderCommand]:
        """Close all active positions.  Returns market order commands."""
        orders: list[OrderCommand] = []

        for ticker in list(self._positions.keys()):
            pos = self._positions.get(ticker)
            if pos is None or not pos.is_active:
                continue

            close_action = OrderAction.SELL if pos.is_long else OrderAction.BUY
            orders.append(
                OrderCommand(
                    symbol=ticker,
                    action=close_action,
                    order_type=OrderType.MARKET,
                    quantity=pos.contracts,
                    reason=reason,
                    position_id=pos.position_id,
                )
            )
            self._close_position(pos, reason=reason, close_price=pos.current_price)

        self.clear_focus()
        logger.info("Closed all positions: %d orders generated (%s)", len(orders), reason)
        return orders

    def close_for_session_end(self, session_key: str) -> list[OrderCommand]:
        """Close positions that belong to an intraday session that is ending.

        Only closes positions whose session_key matches.  Swing/overnight
        positions (Weekly, Monthly, Asian) are held through session boundaries.
        """
        # These breakout types are typically held across sessions
        _swing_types = {"WEEKLY", "MONTHLY", "ASIAN", "Weekly", "Monthly", "Asian"}

        orders: list[OrderCommand] = []

        for ticker in list(self._positions.keys()):
            pos = self._positions.get(ticker)
            if pos is None or not pos.is_active:
                continue

            # Skip swing/overnight types
            if pos.breakout_type in _swing_types:
                continue

            # Only close positions from the ending session
            if pos.session_key and pos.session_key != session_key:
                continue

            close_action = OrderAction.SELL if pos.is_long else OrderAction.BUY
            reason = f"Session end ({session_key}) — closing intraday {pos.breakout_type} position"
            orders.append(
                OrderCommand(
                    symbol=ticker,
                    action=close_action,
                    order_type=OrderType.MARKET,
                    quantity=pos.contracts,
                    reason=reason,
                    position_id=pos.position_id,
                )
            )
            self._close_position(pos, reason=reason, close_price=pos.current_price)

        if orders:
            logger.info("Session end (%s): closed %d intraday positions", session_key, len(orders))
            # Release focus if all positions are now closed
            if not self._positions:
                self.clear_focus()

        return orders

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_price(bars_1m: pd.DataFrame | None, fallback: float) -> float:
        """Get the most recent price from 1m bars, or use fallback."""
        if bars_1m is not None and not bars_1m.empty:
            try:
                return float(bars_1m["Close"].iloc[-1])
            except (IndexError, KeyError, TypeError):
                pass
        return fallback

    @staticmethod
    def _ticker_to_name(ticker: str) -> str:
        """Resolve a data ticker to a human-readable name."""
        try:
            from lib.core.models import TICKER_TO_NAME

            return TICKER_TO_NAME.get(ticker, ticker)
        except ImportError:
            return ticker

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, Any]:
        """Return a summary of all active positions for dashboard / logging."""
        positions = []
        total_unrealized = 0.0

        for ticker, pos in self._positions.items():
            if not pos.is_active:
                continue

            positions.append(
                {
                    "symbol": pos.symbol,
                    "ticker": ticker,
                    "direction": pos.direction,
                    "entry": round(pos.entry_price, 4),
                    "current": round(pos.current_price, 4),
                    "stop": round(pos.stop_loss, 4),
                    "tp1": round(pos.tp1, 4),
                    "tp2": round(pos.tp2, 4),
                    "tp3": round(pos.tp3, 4),
                    "phase": pos.phase.value,
                    "r_multiple": round(pos.r_multiple, 2),
                    "pnl_ticks": round(pos.signed_pnl_ticks, 4),
                    "mfe": round(pos.max_favorable_excursion, 4),
                    "mae": round(pos.max_adverse_excursion, 4),
                    "breakout_type": pos.breakout_type,
                    "cnn_prob": round(pos.cnn_prob, 3),
                    "session": pos.session_key,
                    "hold_secs": round(pos.hold_duration_seconds, 0),
                    "reversals": pos.reversal_count,
                    "pyramid_level": pos.pyramid_level,
                    "pyramid_contracts": pos.pyramid_contracts,
                    "total_contracts": pos.contracts + pos.pyramid_contracts,
                }
            )
            total_unrealized += pos.signed_pnl_ticks

        # Closed positions summary (today only)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        today_closed = [p for p in self._history if p.closed_at and p.closed_at.startswith(today)]
        today_realized = sum(p.realized_pnl for p in today_closed)
        today_wins = sum(1 for p in today_closed if p.realized_pnl > 0)
        today_losses = sum(1 for p in today_closed if p.realized_pnl <= 0)

        return {
            "active_positions": len(positions),
            "max_positions": MAX_POSITIONS,
            "positions": positions,
            "total_unrealized_ticks": round(total_unrealized, 4),
            "today_closed": len(today_closed),
            "today_realized_ticks": round(today_realized, 4),
            "today_wins": today_wins,
            "today_losses": today_losses,
            "today_win_rate": round(today_wins / max(len(today_closed), 1), 2),
            "updated_at": datetime.now(UTC).isoformat(),
            "focus_asset": self._focus_asset,
            "focus_lock_enabled": FOCUS_LOCK_ENABLED,
            "pyramid_enabled": PYRAMID_ENABLED,
        }

    def __repr__(self) -> str:
        active = self.get_position_count()
        return f"<PositionManager active={active}/{MAX_POSITIONS} core={len(self._core_tickers)} tickers>"
