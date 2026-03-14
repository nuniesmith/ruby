"""
Daily Trade Plan Generator — Phase 2B
======================================
Orchestrates the daily pre-market trade selection process:

1. Run bias_analyzer on all tracked assets → directional conviction per asset
2. Optionally call Grok for macro context (economic calendar, overnight news)
3. Score each asset for daily swing potential:
   bias confidence × ATR opportunity × volume regime × catalyst presence
4. Select 1-2 daily swing candidates (biggest expected move, highest conviction)
5. Select 3-4 scalp focus assets (best RB setup density + session fit)
6. Compute entry zone, stop, TP for daily swing (wider than scalp)
7. Position size: small (1 micro contract) — "big move, small risk" trades

Output:
  DailyPlan dataclass — swing_candidates (1-2 assets), scalp_focus (3-4 assets
  for RB system), market_context from Grok, no_trade_flags.

Persisted to Redis key ``engine:daily_plan`` for dashboard consumption.
Separate from the RB scalping system — daily trades run on different timeframe
and risk profile.

Usage:
    from lib.trading.strategies.daily.daily_plan import generate_daily_plan, DailyPlan

    plan = generate_daily_plan(account_size=50_000)
    print(plan.scalp_focus)       # ["Gold", "Nasdaq", "S&P", "Euro FX"]
    print(plan.swing_candidates)  # [SwingCandidate("Gold", ...)]
    print(plan.market_context)    # Grok macro brief or ""

    # Persist for dashboard
    plan.publish_to_redis(redis_client)

Pure computation (except optional Grok call and Redis publish).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from lib.core.utils import safe_float as _safe_float
from lib.trading.strategies.daily.bias_analyzer import (
    BiasDirection,
    DailyBias,
    compute_all_daily_biases,
)

if TYPE_CHECKING:
    from typing import Any as _Any  # noqa: F401 — keep block non-empty

    import pandas as pd

logger = logging.getLogger("strategies.daily.daily_plan")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Composite scoring weights for scalp focus selection (Phase 3A)
W_SIGNAL_QUALITY = 0.30
W_ATR_OPPORTUNITY = 0.25
W_RB_SETUP_DENSITY = 0.20
W_SESSION_FIT = 0.15
W_CATALYST_PRESENCE = 0.10

# Swing scoring weights — bias conviction matters most
W_SWING_CONVICTION = 0.35
W_SWING_ATR = 0.30
W_SWING_VOLUME = 0.15
W_SWING_CATALYST = 0.10
W_SWING_WEEKLY_POSITION = 0.10

# Selection counts
MAX_SCALP_FOCUS = 4
MAX_SWING_CANDIDATES = 2

# Minimum thresholds
MIN_SWING_SCORE = 30.0  # Out of 100 — below this, no swing candidate
MIN_SCALP_SCORE = 20.0  # Out of 100 — below this, not worth watching

# Swing trade risk profile (wider than scalp)
SWING_SL_ATR_MULT = 1.75  # Stop at 1.75× ATR from entry
SWING_TP1_ATR_MULT = 2.5  # TP1 at 2.5× ATR
SWING_TP2_ATR_MULT = 4.0  # TP2 at 4.0× ATR
SWING_TP3_ATR_MULT = 5.5  # TP3 at 5.5× ATR — the "big move" target
SWING_RISK_PCT = 0.005  # 0.5% of account per swing (smaller than scalp's 0.75%)

# Session definitions for session-fit scoring
# Maps asset names to their best session(s)
ASSET_SESSION_MAP: dict[str, tuple[str, ...]] = {
    "Gold": ("london", "us"),
    "Silver": ("london", "us"),
    "Copper": ("london", "us"),
    "Crude Oil": ("us",),
    "Natural Gas": ("us",),
    "S&P": ("us",),
    "Nasdaq": ("us",),
    "Russell 2000": ("us",),
    "Dow Jones": ("us",),
    "Euro FX": ("london", "us"),
    "British Pound": ("london",),
    "Japanese Yen": ("asian", "london"),
    "Australian Dollar": ("asian", "london"),
    "Canadian Dollar": ("us",),
    "Swiss Franc": ("london",),
    "10Y T-Note": ("us",),
    "30Y T-Bond": ("us",),
    "Corn": ("us",),
    "Soybeans": ("us",),
    "Wheat": ("us",),
    "Micro Bitcoin": ("us", "asian"),
    "Micro Ether": ("us", "asian"),
    # Kraken crypto — 24/7 but strongest in US + Asian overlap
    "BTC/USD": ("us", "asian"),
    "ETH/USD": ("us", "asian"),
    "SOL/USD": ("us", "asian"),
    "LINK/USD": ("us",),
    "AVAX/USD": ("us",),
    "DOT/USD": ("asian",),
    "ADA/USD": ("asian", "us"),
    "POL/USD": ("us",),
    "XRP/USD": ("us", "asian"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SwingCandidate:
    """A daily swing trade candidate with entry/stop/TP levels."""

    asset_name: str
    direction: str  # "LONG" or "SHORT"
    confidence: float  # 0.0–1.0 from bias analyzer
    swing_score: float  # 0–100 composite score
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    atr: float = 0.0
    last_price: float = 0.0
    risk_dollars: float = 0.0
    position_size: int = 1  # Micro contracts (small — swing = patience trade)
    reasoning: str = ""
    entry_styles: list[str] = field(default_factory=list)
    key_levels: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "swing_score": round(self.swing_score, 1),
            "entry_zone_low": self.entry_zone_low,
            "entry_zone_high": self.entry_zone_high,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "atr": round(self.atr, 6),
            "last_price": self.last_price,
            "risk_dollars": round(self.risk_dollars, 2),
            "position_size": self.position_size,
            "reasoning": self.reasoning,
            "entry_styles": self.entry_styles,
            "key_levels": self.key_levels,
        }


@dataclass
class ScalpFocusAsset:
    """A scalp-focus asset with composite ranking score."""

    asset_name: str
    composite_score: float  # 0–100
    signal_quality_score: float = 0.0
    atr_opportunity_score: float = 0.0
    rb_setup_density_score: float = 0.0
    session_fit_score: float = 0.0
    catalyst_score: float = 0.0
    bias_direction: str = "NEUTRAL"
    bias_confidence: float = 0.0
    last_price: float = 0.0
    atr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "composite_score": round(self.composite_score, 1),
            "signal_quality_score": round(self.signal_quality_score, 1),
            "atr_opportunity_score": round(self.atr_opportunity_score, 1),
            "rb_setup_density_score": round(self.rb_setup_density_score, 1),
            "session_fit_score": round(self.session_fit_score, 1),
            "catalyst_score": round(self.catalyst_score, 1),
            "bias_direction": self.bias_direction,
            "bias_confidence": round(self.bias_confidence, 3),
            "last_price": self.last_price,
            "atr": round(self.atr, 6),
        }


@dataclass
class DailyPlan:
    """Complete daily trade plan — the output of pre-market analysis."""

    # Core selections
    scalp_focus: list[ScalpFocusAsset] = field(default_factory=list)
    swing_candidates: list[SwingCandidate] = field(default_factory=list)

    # Context
    market_context: str = ""  # Grok macro brief (if available)
    grok_available: bool = False
    no_trade: bool = False
    no_trade_reason: str = ""

    # Phase 3C: Structured Grok analysis results (parsed JSON)
    # Contains: macro_bias, macro_summary, top_assets, risk_warnings,
    # economic_events, session_plan, correlation_notes, swing_insights
    grok_analysis: dict[str, Any] = field(default_factory=dict)

    # All biases (for dashboard display)
    all_biases: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    computed_at: str = ""
    account_size: int = 50_000
    session: str = ""  # "pre-market", "active", "off-hours"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scalp_focus": [a.to_dict() for a in self.scalp_focus],
            "swing_candidates": [s.to_dict() for s in self.swing_candidates],
            "scalp_focus_names": [a.asset_name for a in self.scalp_focus],
            "swing_candidate_names": [s.asset_name for s in self.swing_candidates],
            "market_context": self.market_context,
            "grok_available": self.grok_available,
            "grok_analysis": self.grok_analysis,
            "no_trade": self.no_trade,
            "no_trade_reason": self.no_trade_reason,
            "all_biases": self.all_biases,
            "computed_at": self.computed_at,
            "account_size": self.account_size,
            "session": self.session,
            "total_scalp_focus": len(self.scalp_focus),
            "total_swing_candidates": len(self.swing_candidates),
        }

    def publish_to_redis(self, redis_client: Any) -> bool:
        """Persist the daily plan to Redis for dashboard consumption.

        Keys written:
          - ``engine:daily_plan`` — full plan JSON
          - ``engine:focus_assets`` — list of scalp focus asset names
          - ``engine:swing_assets`` — list of swing candidate asset names

        Args:
            redis_client: Redis client instance (must have .set() and .publish()).

        Returns:
            True if published successfully, False on error.
        """
        try:
            payload = json.dumps(self.to_dict(), default=str)
            redis_client.set("engine:daily_plan", payload)
            redis_client.set(
                "engine:focus_assets",
                json.dumps([a.asset_name for a in self.scalp_focus]),
            )
            redis_client.set(
                "engine:swing_assets",
                json.dumps([s.asset_name for s in self.swing_candidates]),
            )
            # Expire in 18 hours — plan is regenerated daily pre-market
            redis_client.expire("engine:daily_plan", 18 * 3600)
            redis_client.expire("engine:focus_assets", 18 * 3600)
            redis_client.expire("engine:swing_assets", 18 * 3600)

            # Publish event for SSE listeners
            redis_client.publish("dashboard:daily_plan", payload)

            logger.info(
                "Daily plan published to Redis: %d scalp focus, %d swing candidates",
                len(self.scalp_focus),
                len(self.swing_candidates),
            )
            return True
        except Exception as exc:
            logger.error("Failed to publish daily plan to Redis: %s", exc)
            return False

    @staticmethod
    def load_from_redis(redis_client: Any) -> DailyPlan | None:
        """Load the most recent daily plan from Redis.

        Returns:
            DailyPlan reconstructed from Redis, or None if not found.
        """
        try:
            raw = redis_client.get("engine:daily_plan")
            if raw is None:
                return None
            data = json.loads(raw)

            plan = DailyPlan(
                market_context=data.get("market_context", ""),
                grok_available=data.get("grok_available", False),
                grok_analysis=data.get("grok_analysis", {}),
                no_trade=data.get("no_trade", False),
                no_trade_reason=data.get("no_trade_reason", ""),
                all_biases=data.get("all_biases", {}),
                computed_at=data.get("computed_at", ""),
                account_size=data.get("account_size", 50_000),
                session=data.get("session", ""),
            )

            # Reconstruct scalp focus
            for sf in data.get("scalp_focus", []):
                plan.scalp_focus.append(
                    ScalpFocusAsset(
                        asset_name=sf["asset_name"],
                        composite_score=sf.get("composite_score", 0.0),
                        signal_quality_score=sf.get("signal_quality_score", 0.0),
                        atr_opportunity_score=sf.get("atr_opportunity_score", 0.0),
                        rb_setup_density_score=sf.get("rb_setup_density_score", 0.0),
                        session_fit_score=sf.get("session_fit_score", 0.0),
                        catalyst_score=sf.get("catalyst_score", 0.0),
                        bias_direction=sf.get("bias_direction", "NEUTRAL"),
                        bias_confidence=sf.get("bias_confidence", 0.0),
                        last_price=sf.get("last_price", 0.0),
                        atr=sf.get("atr", 0.0),
                    )
                )

            # Reconstruct swing candidates
            for sc in data.get("swing_candidates", []):
                plan.swing_candidates.append(
                    SwingCandidate(
                        asset_name=sc["asset_name"],
                        direction=sc.get("direction", "NEUTRAL"),
                        confidence=sc.get("confidence", 0.0),
                        swing_score=sc.get("swing_score", 0.0),
                        entry_zone_low=sc.get("entry_zone_low", 0.0),
                        entry_zone_high=sc.get("entry_zone_high", 0.0),
                        stop_loss=sc.get("stop_loss", 0.0),
                        tp1=sc.get("tp1", 0.0),
                        tp2=sc.get("tp2", 0.0),
                        tp3=sc.get("tp3", 0.0),
                        atr=sc.get("atr", 0.0),
                        last_price=sc.get("last_price", 0.0),
                        risk_dollars=sc.get("risk_dollars", 0.0),
                        position_size=sc.get("position_size", 1),
                        reasoning=sc.get("reasoning", ""),
                        entry_styles=sc.get("entry_styles", []),
                        key_levels=sc.get("key_levels", {}),
                    )
                )

            return plan
        except Exception as exc:
            logger.error("Failed to load daily plan from Redis: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _get_current_session() -> str:
    """Determine the currently active trading session based on ET time.

    Returns one of: "asian", "london", "us", "off-hours"
    """
    now = datetime.now(tz=_EST)
    hour = now.hour

    # Asian: 19:00 – 02:00 ET (previous day 7pm to 2am)
    if hour >= 19 or hour < 2:
        return "asian"
    # London: 02:00 – 08:00 ET
    if 2 <= hour < 8:
        return "london"
    # US: 08:00 – 16:30 ET
    if 8 <= hour < 17:
        return "us"
    # Off-hours: 17:00 – 19:00 ET (CME maintenance / gap period)
    return "off-hours"


def _compute_session_fit_score(asset_name: str) -> float:
    """Score 0–100 for how well the current session fits this asset.

    100 = perfect match (asset's best session is active)
    50  = secondary match
    10  = asset is tradeable but session isn't ideal
    """
    current_session = _get_current_session()
    sessions = ASSET_SESSION_MAP.get(asset_name, ())

    if not sessions:
        return 30.0  # Unknown asset, give neutral score

    if current_session == "off-hours":
        return 10.0  # Nobody's session is great during maintenance

    if current_session in sessions:
        # Primary or secondary match
        if sessions[0] == current_session:
            return 100.0  # Primary session match
        return 70.0  # Secondary session match

    # Asset's best sessions don't include the current one
    return 20.0


def _compute_atr_opportunity_score(atr: float, last_price: float) -> float:
    """Score 0–100 for ATR as a percentage of price — higher = more opportunity.

    Normalized so that typical intraday movers score 50–80 and exceptional
    movers score 80–100. Very tight assets score 10–30.

    The metric: NATR (Normalized ATR) = (ATR / price) × 100
    """
    if last_price <= 0 or atr <= 0:
        return 10.0

    natr_pct = (atr / last_price) * 100.0

    # NATR ranges (approximate for 5m ATR over 5 days):
    # < 0.1%   — very tight (treasuries, some FX)  → 10–25
    # 0.1–0.3% — moderate (equity indices)          → 25–55
    # 0.3–0.6% — good opportunity (metals, oil)     → 55–80
    # 0.6–1.5% — excellent (volatile days, crypto)  → 80–95
    # > 1.5%   — extreme (could be dangerous)       → 90–100 (capped)
    if natr_pct < 0.05:
        return 10.0
    if natr_pct < 0.1:
        return 10.0 + (natr_pct - 0.05) / 0.05 * 15.0  # 10–25
    if natr_pct < 0.3:
        return 25.0 + (natr_pct - 0.1) / 0.2 * 30.0  # 25–55
    if natr_pct < 0.6:
        return 55.0 + (natr_pct - 0.3) / 0.3 * 25.0  # 55–80
    if natr_pct < 1.5:
        return 80.0 + (natr_pct - 0.6) / 0.9 * 15.0  # 80–95
    return min(100.0, 95.0 + (natr_pct - 1.5) / 1.0 * 5.0)  # 95–100 capped


def _compute_rb_setup_density_score(
    asset_name: str,
    redis_client: Any | None = None,
) -> float:
    """Score 0–100 for how many breakout types are forming ranges near price.

    Reads from Redis keys set by the engine's breakout detection loop.
    If no Redis client is available, returns a neutral 40.

    The engine stores recent breakout results per type per asset in Redis.
    We count how many distinct breakout types have an active/pending range.
    """
    if redis_client is None:
        return 40.0  # Neutral when we can't check

    try:
        # The engine stores breakout state per asset in Redis hash
        # Key: engine:breakout_state:{asset_name}
        # Fields: one per breakout type with last detection time/state
        key = f"engine:breakout_state:{asset_name}"
        state = redis_client.hgetall(key)

        if not state:
            return 30.0  # No data — not necessarily bad, might be quiet

        now = datetime.now(tz=_EST)
        active_count = 0.0
        _ = 13  # All breakout types (unused; count for reference only)

        for _type_name, raw_val in state.items():
            try:
                val = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                # Check if range is fresh (within last 2 hours)
                if isinstance(val, dict):
                    ts = val.get("timestamp") or val.get("detected_at", "")
                    if ts:
                        detected = datetime.fromisoformat(str(ts))
                        if hasattr(detected, "tzinfo") and detected.tzinfo is None:
                            detected = detected.replace(tzinfo=_EST)
                        age_secs = (now - detected).total_seconds()
                        if age_secs < 7200:  # Within 2 hours
                            active_count += 1
                    else:
                        # Has state but no timestamp — count it at half weight
                        active_count += 0.5
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        # Score: 0 types = 10, 1 = 25, 2 = 40, 3 = 55, 4+ = 70+
        if active_count <= 0:
            return 10.0
        # Logarithmic scaling: each additional type has diminishing returns
        # 1 type → ~30, 2 → ~50, 3 → ~65, 5 → ~80, 8+ → ~95
        score = 30.0 + 50.0 * (1.0 - math.exp(-0.5 * active_count))
        return min(100.0, score)

    except Exception as exc:
        logger.debug("RB setup density check failed for %s: %s", asset_name, exc)
        return 40.0


def _compute_catalyst_score(
    asset_name: str,
    redis_client: Any | None = None,
) -> float:
    """Score 0–100 for catalyst presence (economic events, earnings, etc.).

    Reads from the scorer.py economic event calendar persisted in Redis.
    If unavailable, returns neutral 30.
    """
    if redis_client is None:
        return 30.0

    try:
        # The scorer persists today's catalyst scores per asset
        key = "engine:catalyst_scores"
        raw = redis_client.hget(key, asset_name)
        if raw is None:
            return 30.0

        # Score is already 0–100 from scorer.py
        score = _safe_float(raw, 30.0)
        return max(0.0, min(100.0, score))
    except Exception:
        return 30.0


def _compute_signal_quality_for_asset(asset_name: str) -> float:
    """Fetch the latest signal quality score for an asset.

    Attempts to compute fresh signal quality from cached data.
    Returns 0–100 score.
    """
    try:
        from lib.analysis.signal_quality import compute_signal_quality
        from lib.analysis.volatility import kmeans_volatility_clusters
        from lib.analysis.wave_analysis import calculate_wave_analysis
        from lib.core.cache import get_data
        from lib.core.models import ASSETS

        ticker = ASSETS.get(asset_name)
        if not ticker:
            return 40.0

        df = get_data(ticker, "5m", "5d")
        if df is None or df.empty or len(df) < 30:
            return 30.0

        wave_result = calculate_wave_analysis(df, asset_name=asset_name)
        vol_result = kmeans_volatility_clusters(df)
        sq_result = compute_signal_quality(
            df,
            wave_result=wave_result,
            vol_result=vol_result,
        )

        quality_pct = _safe_float(sq_result.get("quality_pct", 0.0))
        return max(0.0, min(100.0, quality_pct))
    except Exception as exc:
        logger.debug("Signal quality compute failed for %s: %s", asset_name, exc)
        return 40.0


def _fetch_asset_data(asset_name: str) -> tuple[float, float]:
    """Fetch last price and ATR for an asset.

    Routing priority:
      1. EngineDataClient (GET /bars/{symbol}) — engine handles Redis →
         Postgres → Massive/Kraken internally; no direct API keys required.
      2. Engine snapshot endpoint (GET /market_data/snapshot/{symbol}) —
         provides a live price without needing bar history.
      3. Legacy fallback: ``cache.get_data`` (Redis/yfinance) — used when
         the engine is unreachable (e.g. local dev without Docker).

    Returns:
        (last_price, atr) — both 0.0 on failure.
    """
    # ── Resolve ticker from asset name ────────────────────────────────────
    ticker: str | None = None
    try:
        from lib.core.models import ASSETS

        ticker = ASSETS.get(asset_name)
    except Exception:
        pass

    symbol = ticker or asset_name  # fall back to using the name directly

    # ── Primary path: engine HTTP API ─────────────────────────────────────
    try:
        from lib.analysis.volatility import kmeans_volatility_clusters
        from lib.services.data.engine_data_client import get_client

        client = get_client()
        df = client.get_bars(symbol, interval="5m", days_back=5)
        if df is not None and not df.empty and len(df) >= 10:
            last_price = _safe_float(df["Close"].iloc[-1])
            try:
                vol = kmeans_volatility_clusters(df)
                atr = _safe_float(vol.get("raw_atr", 0.0))
            except Exception:
                if "High" in df.columns and "Low" in df.columns:
                    recent = df.tail(14)
                    tr = recent["High"] - recent["Low"]
                    atr = _safe_float(tr.mean())
                else:
                    atr = last_price * 0.005
            if last_price > 0:
                return last_price, atr

        # Engine has no bars yet — try snapshot for at least a live price
        snap = client.get_snapshot(symbol)
        if snap:
            snap_price = _safe_float(snap.get("price", 0.0))
            if snap_price > 0:
                logger.debug(
                    "Using snapshot price for %s (no bars from engine): %.4f",
                    asset_name,
                    snap_price,
                )
                return snap_price, snap_price * 0.005  # rough ATR estimate

    except Exception as exc:
        logger.debug("EngineDataClient fetch failed for %s: %s", asset_name, exc)

    # ── Legacy fallback: local cache / Redis ──────────────────────────────
    try:
        from lib.analysis.volatility import kmeans_volatility_clusters
        from lib.core.cache import get_data

        df = get_data(symbol, "5m", "5d")
        if df is None or df.empty or len(df) < 10:
            return 0.0, 0.0

        last_price = _safe_float(df["Close"].iloc[-1])
        try:
            vol = kmeans_volatility_clusters(df)
            atr = _safe_float(vol.get("raw_atr", 0.0))
        except Exception:
            if "High" in df.columns and "Low" in df.columns:
                recent = df.tail(14)
                tr = recent["High"] - recent["Low"]
                atr = _safe_float(tr.mean())
            else:
                atr = last_price * 0.005

        logger.debug(
            "Asset data for %s served from local cache (engine unavailable)",
            asset_name,
        )
        return last_price, atr
    except Exception as exc:
        logger.debug("Legacy cache fetch failed for %s: %s", asset_name, exc)
        return 0.0, 0.0


def _price_decimals_from_tick(tick_size: float) -> int:
    """Return the number of decimal places needed to represent one tick."""
    if tick_size <= 0:
        return 4
    s = f"{tick_size:.10f}".rstrip("0")
    if "." not in s:
        return 2
    decimals = len(s.split(".")[1])
    return max(2, min(decimals, 7))


# ---------------------------------------------------------------------------
# Swing candidate builder
# ---------------------------------------------------------------------------


def _build_swing_candidate(
    asset_name: str,
    bias: DailyBias,
    last_price: float,
    atr: float,
    swing_score: float,
    account_size: int,
) -> SwingCandidate:
    """Build a SwingCandidate with computed entry/stop/TP levels.

    Swing trades use wider stops and targets than scalps:
    - SL at 1.75× ATR
    - TP1 at 2.5× ATR, TP2 at 4× ATR, TP3 at 5.5× ATR
    - Position size: 1 micro contract (patience trade, not scalp)
    """
    direction = bias.direction.value  # "LONG" or "SHORT"

    # Get tick size for rounding
    tick_size = 0.01
    try:
        from lib.core.asset_registry import get_asset

        asset_obj = get_asset(asset_name)
        if asset_obj and asset_obj.micro:
            tick_size = asset_obj.micro.tick_size
    except ImportError:
        pass

    decimals = _price_decimals_from_tick(tick_size)

    # Entry zone: pullback area for the swing entry
    entry_width = atr * 0.6  # Wider than scalp entry zone

    if direction == "LONG":
        entry_low = last_price - entry_width
        entry_high = last_price + entry_width * 0.2  # Slightly above for breakout
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint - atr * SWING_SL_ATR_MULT
        tp1 = midpoint + atr * SWING_TP1_ATR_MULT
        tp2 = midpoint + atr * SWING_TP2_ATR_MULT
        tp3 = midpoint + atr * SWING_TP3_ATR_MULT
    else:  # SHORT
        entry_high = last_price + entry_width
        entry_low = last_price - entry_width * 0.2
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint + atr * SWING_SL_ATR_MULT
        tp1 = midpoint - atr * SWING_TP1_ATR_MULT
        tp2 = midpoint - atr * SWING_TP2_ATR_MULT
        tp3 = midpoint - atr * SWING_TP3_ATR_MULT

    # Position sizing: small, 1 micro for swing patience trades
    max_risk = account_size * SWING_RISK_PCT
    stop_distance = abs(midpoint - stop)
    position_size = 1  # Default 1 micro

    try:
        from lib.core.asset_registry import get_asset

        asset_obj = get_asset(asset_name)
        if asset_obj and asset_obj.micro:
            risk_per_contract = stop_distance * asset_obj.micro.point_value
            if risk_per_contract > 0:
                position_size = max(1, int(max_risk / risk_per_contract))
                # Cap swing size more conservatively — max 3 micros
                position_size = min(position_size, 3)
            risk_dollars = position_size * risk_per_contract
        else:
            risk_dollars = max_risk
    except ImportError:
        risk_dollars = max_risk

    # Determine entry styles based on bias components
    entry_styles = []
    if bias.candle_pattern and bias.candle_pattern.value in (
        "bullish_engulfing",
        "bearish_engulfing",
        "hammer",
        "shooting_star",
    ):
        entry_styles.append("pullback_to_key_level")
    if bias.overnight_gap_direction != 0.0:
        gap_aligns = (bias.overnight_gap_direction > 0 and direction == "LONG") or (
            bias.overnight_gap_direction < 0 and direction == "SHORT"
        )
        if gap_aligns:
            entry_styles.append("gap_continuation")
    if bias.atr_expanding:
        entry_styles.append("breakout_entry")
    if not entry_styles:
        entry_styles.append("pullback_to_key_level")  # Default

    # Key levels from bias analyzer
    kl = bias.key_levels
    key_levels_dict: dict[str, float] = {}
    if kl:
        key_levels_dict = {k: v for k, v in kl.to_dict().items() if v and v > 0}

    return SwingCandidate(
        asset_name=asset_name,
        direction=direction,
        confidence=bias.confidence,
        swing_score=swing_score,
        entry_zone_low=round(entry_low, decimals),
        entry_zone_high=round(entry_high, decimals),
        stop_loss=round(stop, decimals),
        tp1=round(tp1, decimals),
        tp2=round(tp2, decimals),
        tp3=round(tp3, decimals),
        atr=atr,
        last_price=round(last_price, decimals),
        risk_dollars=round(risk_dollars, 2),
        position_size=position_size,
        reasoning=bias.reasoning,
        entry_styles=entry_styles,
        key_levels=key_levels_dict,
    )


# ---------------------------------------------------------------------------
# Grok integration (optional)
# ---------------------------------------------------------------------------


def _fetch_grok_context(
    biases: dict[str, DailyBias],
    asset_names: list[str],
) -> str:
    """Optionally fetch macro context from Grok AI.

    Only called if XAI_API_KEY is set in environment. Returns empty string
    if Grok is unavailable or the call fails.

    Uses ``run_morning_briefing`` from ``grok_helper`` when possible (it
    already formats a comprehensive pre-market game plan). Falls back to
    the lower-level ``_call_grok`` with a custom prompt if the context dict
    cannot be assembled.
    """
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        from lib.integrations.grok_helper import _call_grok, run_morning_briefing
    except ImportError:
        logger.debug("grok_helper not available — skipping Grok context")
        return ""

    # Build a bias summary string for the prompt
    bias_lines: list[str] = []
    for name in asset_names[:8]:  # Limit to top 8
        b = biases.get(name)
        if b:
            bias_lines.append(f"  {name}: {b.direction.value} ({b.confidence:.0%}) — {b.reasoning[:80]}")

    now = datetime.now(tz=_EST)

    # Try using run_morning_briefing with a minimal context dict first
    try:
        context = {
            "time": now.strftime("%A %B %d, %Y %H:%M ET"),
            "account_size": 50_000,
            "risk_dollars": 375,
            "max_contracts": 10,
            "session_status": "pre-market",
        }
        result = run_morning_briefing(context, api_key)
        if result:
            return str(result)
    except Exception as exc:
        logger.debug("run_morning_briefing failed, falling back to raw call: %s", exc)

    # Fallback: direct Grok call with a focused prompt
    prompt = (
        f"Daily pre-market brief for {now.strftime('%A %B %d, %Y')} "
        f"({now.strftime('%H:%M')} ET).\n\n"
        "My system's directional bias for today:\n" + "\n".join(bias_lines) + "\n\n"
        "Give me a brief (3-5 sentences) macro context: "
        "1) Are we risk-on or risk-off? "
        "2) Any key economic releases today? "
        "3) Which 2-3 assets have the most potential for a big move? "
        "4) Any events that could cause sharp reversals? "
        "Keep it actionable — I'm a futures scalper/swing trader."
    )

    try:
        result = _call_grok(prompt, api_key=api_key, max_tokens=400)
        return str(result) if result else ""
    except Exception as exc:
        logger.warning("Grok analysis failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Phase 3A: Top-N Asset Selection
# ---------------------------------------------------------------------------


def select_daily_focus_assets(
    biases: dict[str, DailyBias],
    asset_names: list[str] | None = None,
    redis_client: Any | None = None,
    max_scalp: int = MAX_SCALP_FOCUS,
    max_swing: int = MAX_SWING_CANDIDATES,
    account_size: int = 50_000,
) -> tuple[list[ScalpFocusAsset], list[SwingCandidate]]:
    """Select the day's focused assets — 3-4 for scalping, 1-2 for swing.

    This is the Phase 3A implementation: composite ranking score per asset,
    selecting the best for each trading style.

    Args:
        biases: Daily bias per asset from compute_all_daily_biases().
        asset_names: List of asset names to consider. If None, uses all
                     assets from the ASSETS dict.
        redis_client: Optional Redis client for RB setup density + catalyst
                      score lookups.
        max_scalp: Maximum scalp focus assets to select (default 4).
        max_swing: Maximum swing candidates to select (default 2).
        account_size: Account size for swing position sizing.

    Returns:
        (scalp_focus_list, swing_candidates_list) — both sorted by score desc.
    """
    if asset_names is None:
        try:
            from lib.core.models import ASSETS

            asset_names = list(ASSETS.keys())
        except ImportError:
            asset_names = list(biases.keys())

    scalp_scored: list[ScalpFocusAsset] = []
    swing_scored: list[tuple[float, str]] = []  # (score, asset_name)

    for name in asset_names:
        bias = biases.get(name)
        last_price, atr = _fetch_asset_data(name)

        if last_price <= 0:
            logger.debug("Skipping %s — no price data", name)
            continue

        # ── Scalp composite score ───────────────────────────────────────
        sig_quality = _compute_signal_quality_for_asset(name)
        atr_opp = _compute_atr_opportunity_score(atr, last_price)
        rb_density = _compute_rb_setup_density_score(name, redis_client)
        session_fit = _compute_session_fit_score(name)
        catalyst = _compute_catalyst_score(name, redis_client)

        composite = (
            sig_quality * W_SIGNAL_QUALITY
            + atr_opp * W_ATR_OPPORTUNITY
            + rb_density * W_RB_SETUP_DENSITY
            + session_fit * W_SESSION_FIT
            + catalyst * W_CATALYST_PRESENCE
        )

        bias_dir = bias.direction.value if bias else "NEUTRAL"
        bias_conf = bias.confidence if bias else 0.0

        scalp_scored.append(
            ScalpFocusAsset(
                asset_name=name,
                composite_score=composite,
                signal_quality_score=sig_quality,
                atr_opportunity_score=atr_opp,
                rb_setup_density_score=rb_density,
                session_fit_score=session_fit,
                catalyst_score=catalyst,
                bias_direction=bias_dir,
                bias_confidence=bias_conf,
                last_price=last_price,
                atr=atr,
            )
        )

        # ── Swing score (only directional assets) ──────────────────────
        if bias and bias.direction != BiasDirection.NEUTRAL and bias.confidence > 0.15:
            vol_confirm = 1.0 if bias.volume_confirmation else 0.6
            weekly_pos = bias.weekly_range_position

            # Weekly position score: favor assets NOT at extremes
            # (room to run). 0.2–0.8 = good room, 0.0/1.0 = already at edge
            if bias.direction == BiasDirection.LONG:
                # Want price below weekly mid (room to run up)
                weekly_score = max(0.0, 1.0 - weekly_pos) * 100.0
            else:
                # Want price above weekly mid (room to run down)
                weekly_score = weekly_pos * 100.0

            sw_score = (
                (bias.confidence * 100.0) * W_SWING_CONVICTION
                + atr_opp * W_SWING_ATR
                + (vol_confirm * 100.0) * W_SWING_VOLUME
                + catalyst * W_SWING_CATALYST
                + weekly_score * W_SWING_WEEKLY_POSITION
            )
            swing_scored.append((sw_score, name))

    # ── Sort and select ─────────────────────────────────────────────────
    scalp_scored.sort(key=lambda x: x.composite_score, reverse=True)
    swing_scored.sort(key=lambda x: x[0], reverse=True)

    # Filter by minimum thresholds
    scalp_focus = [a for a in scalp_scored if a.composite_score >= MIN_SCALP_SCORE][:max_scalp]
    swing_names = [(score, name) for score, name in swing_scored if score >= MIN_SWING_SCORE][:max_swing]

    # Build swing candidates with full level computation
    swing_candidates = []
    for sw_score, name in swing_names:
        bias = biases.get(name)
        if bias is None:
            continue
        last_price, atr = _fetch_asset_data(name)
        if last_price <= 0 or atr <= 0:
            continue

        candidate = _build_swing_candidate(
            asset_name=name,
            bias=bias,
            last_price=last_price,
            atr=atr,
            swing_score=sw_score,
            account_size=account_size,
        )
        swing_candidates.append(candidate)

    logger.info(
        "Daily focus selection: %d/%d scalp assets, %d/%d swing candidates (from %d total)",
        len(scalp_focus),
        max_scalp,
        len(swing_candidates),
        max_swing,
        len(asset_names),
    )

    return scalp_focus, swing_candidates


# ---------------------------------------------------------------------------
# Main entry point — generate the complete daily plan
# ---------------------------------------------------------------------------


def generate_daily_plan(
    account_size: int = 50_000,
    asset_names: list[str] | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
    weekly_data: dict[str, pd.DataFrame] | None = None,
    current_opens: dict[str, float] | None = None,
    redis_client: Any | None = None,
    include_grok: bool = True,
) -> DailyPlan:
    """Generate the complete daily trade plan.

    This is the Phase 2B orchestrator that runs during pre-market:

    1. Fetch daily + weekly bar data for all tracked assets
    2. Run bias_analyzer on each → directional conviction
    3. Optionally call Grok for macro context
    4. Score assets for scalp focus (Phase 3A composite) and swing potential
    5. Select top 3-4 scalp + 1-2 swing candidates
    6. Compute entry/stop/TP for swing trades (wider than scalp)
    7. Package into DailyPlan for dashboard + Redis

    Args:
        account_size: Account size for risk/sizing calculations.
        asset_names: Override list of asset names. If None, uses all from ASSETS.
        daily_data: Pre-fetched daily bar data {name: df}. If None, fetches
                    from cache/data layer.
        weekly_data: Pre-fetched weekly bar data {name: df}. If None, fetches
                     from cache/data layer.
        current_opens: Today's open prices per asset. If None, gap analysis
                       is skipped in bias computation.
        redis_client: Redis client for breakout state + catalyst lookups and
                      optional plan persistence.
        include_grok: Whether to call Grok for macro context (default True,
                      requires XAI_API_KEY env var).

    Returns:
        DailyPlan with all selections, levels, and context.
    """
    now = datetime.now(tz=_EST)
    plan = DailyPlan(
        computed_at=now.isoformat(),
        account_size=account_size,
    )

    # Determine session
    hour = now.hour
    if 0 <= hour < 6:
        plan.session = "pre-market"
    elif 6 <= hour < 16:
        plan.session = "active"
    else:
        plan.session = "off-hours"

    # ── Step 1: Resolve asset names ─────────────────────────────────────
    if asset_names is None:
        try:
            from lib.core.models import ASSETS

            asset_names = list(ASSETS.keys())
        except ImportError:
            logger.error("Cannot determine asset list — no ASSETS dict available")
            plan.no_trade = True
            plan.no_trade_reason = "Asset list unavailable"
            return plan

    logger.info(
        "Generating daily plan for %d assets (account=$%s, session=%s)",
        len(asset_names),
        f"{account_size:,}",
        plan.session,
    )

    # ── Step 2: Fetch data if not provided ──────────────────────────────
    if daily_data is None:
        daily_data = {}
        for name in asset_names:
            try:
                from lib.core.cache import get_data
                from lib.core.models import ASSETS

                ticker = ASSETS.get(name)
                if ticker:
                    df = get_data(ticker, "1d", "60d")
                    if df is not None and not df.empty:
                        daily_data[name] = df
            except Exception as exc:
                logger.debug("Daily data fetch failed for %s: %s", name, exc)

    if weekly_data is None:
        weekly_data = {}
        for name in asset_names:
            try:
                from lib.core.cache import get_data
                from lib.core.models import ASSETS

                ticker = ASSETS.get(name)
                if ticker:
                    df = get_data(ticker, "1wk", "6mo")
                    if df is not None and not df.empty:
                        weekly_data[name] = df
            except Exception as exc:
                logger.debug("Weekly data fetch failed for %s: %s", name, exc)

    if not daily_data:
        logger.warning("No daily data available for any asset")
        plan.no_trade = True
        plan.no_trade_reason = "No market data available"
        return plan

    # ── Step 3: Run bias analyzer ───────────────────────────────────────
    logger.info("Running daily bias analysis on %d assets...", len(daily_data))
    biases = compute_all_daily_biases(
        daily_data=daily_data,
        weekly_data=weekly_data or None,
        current_opens=current_opens,
    )

    # Store all biases for dashboard
    plan.all_biases = {name: bias.to_dict() for name, bias in biases.items()}

    # Log bias summary
    for name, bias in biases.items():
        logger.info(
            "  %s: %s (%.0f%%) — %s",
            name,
            bias.direction.value,
            bias.confidence * 100,
            bias.reasoning[:60],
        )

    # ── Step 4: Select focus assets (Phase 3A) ──────────────────────────
    # Run BEFORE Grok so swing_candidates / scalp_focus names are available
    # for the structured Grok prompt (Phase 3C).
    scalp_focus, swing_candidates = select_daily_focus_assets(
        biases=biases,
        asset_names=asset_names,
        redis_client=redis_client,
        account_size=account_size,
    )

    plan.scalp_focus = scalp_focus
    plan.swing_candidates = swing_candidates

    # ── Step 5: Optional Grok context (Phase 3C: structured analysis) ───
    if include_grok and os.getenv("XAI_API_KEY", "").strip():
        logger.info("Fetching Grok macro context...")

        # Phase 3C: Try structured Grok analysis first
        try:
            from lib.integrations.grok_helper import (
                format_grok_daily_plan_for_display,
                run_daily_plan_grok_analysis,
            )

            grok_result = run_daily_plan_grok_analysis(
                biases=biases,
                asset_names=asset_names,
                swing_candidate_names=[s.asset_name for s in swing_candidates] if swing_candidates else None,
                scalp_focus_names=[a.asset_name for a in scalp_focus] if scalp_focus else None,
                account_size=account_size,
            )
            if grok_result:
                plan.grok_analysis = grok_result
                plan.market_context = format_grok_daily_plan_for_display(grok_result)
                plan.grok_available = True
                logger.info(
                    "Grok structured analysis received: macro=%s, %d top assets, %d warnings",
                    grok_result.get("macro_bias", "?"),
                    len(grok_result.get("top_assets", [])),
                    len(grok_result.get("risk_warnings", [])),
                )
        except ImportError:
            logger.debug("Structured Grok analysis not available — falling back to free-text")
            grok_result = None
        except Exception as exc:
            logger.warning("Structured Grok analysis failed: %s — falling back to free-text", exc)
            grok_result = None

        # Fallback to free-text Grok context if structured analysis failed
        if not plan.grok_available:
            plan.market_context = _fetch_grok_context(biases, asset_names)
            plan.grok_available = bool(plan.market_context)
            if plan.grok_available:
                logger.info("Grok free-text context received (%d chars)", len(plan.market_context))
    else:
        plan.grok_available = False

    # ── Step 6: Check no-trade conditions ────────────────────────────────
    if not scalp_focus and not swing_candidates:
        plan.no_trade = True
        plan.no_trade_reason = "No assets meet minimum quality thresholds"
    elif all(b.direction == BiasDirection.NEUTRAL for b in biases.values()):
        # All assets are neutral — unusual, might be a holiday/low-vol day
        plan.no_trade_reason = "All assets showing neutral bias — low conviction day"
        # Don't set no_trade=True — scalp focus can still work without directional bias

    # ── Log final plan ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DAILY PLAN — %s", now.strftime("%Y-%m-%d %H:%M ET"))
    logger.info("=" * 60)

    if plan.market_context:
        logger.info("📰 Grok: %s", plan.market_context[:120])

    logger.info(
        "🎯 Scalp Focus (%d): %s",
        len(scalp_focus),
        ", ".join(f"{a.asset_name} ({a.composite_score:.0f})" for a in scalp_focus) or "None",
    )

    for sc in swing_candidates:
        logger.info(
            "📈 Swing: %s %s (%.0f%% confidence, score=%.0f) Entry: %.4f–%.4f | SL: %.4f | TP1: %.4f | TP3: %.4f",
            sc.direction,
            sc.asset_name,
            sc.confidence * 100,
            sc.swing_score,
            sc.entry_zone_low,
            sc.entry_zone_high,
            sc.stop_loss,
            sc.tp1,
            sc.tp3,
        )

    if plan.no_trade:
        logger.info("🚫 NO TRADE: %s", plan.no_trade_reason)
    elif plan.no_trade_reason:
        logger.info("⚠️ Note: %s", plan.no_trade_reason)

    logger.info("=" * 60)

    return plan


# ---------------------------------------------------------------------------
# Convenience: generate + publish in one call
# ---------------------------------------------------------------------------


def generate_and_publish_daily_plan(
    redis_client: Any,
    account_size: int = 50_000,
    asset_names: list[str] | None = None,
    include_grok: bool = True,
) -> DailyPlan:
    """Generate the daily plan and immediately publish to Redis.

    Convenience wrapper for use in the engine scheduler's pre-market job.

    Args:
        redis_client: Redis client (required for publish + breakout state reads).
        account_size: Account size for sizing calculations.
        asset_names: Override asset list (None = all tracked).
        include_grok: Whether to include Grok macro context.

    Returns:
        The generated DailyPlan (also published to Redis).
    """
    plan = generate_daily_plan(
        account_size=account_size,
        asset_names=asset_names,
        redis_client=redis_client,
        include_grok=include_grok,
    )

    plan.publish_to_redis(redis_client)
    return plan
