"""
Daily Focus Computation
====================================
Computes the daily trading focus — the core "what should I trade today" payload.

For each tracked asset (MGC, MNQ, MES, MCL, SIL, HG), computes:
  - Bias: LONG / SHORT / NEUTRAL (from wave dominance + AO + confluence)
  - Entry zone (low–high), stop loss, TP1, TP2
  - Wave ratio, signal quality %, volatility percentile
  - Position size in micro contracts, risk in dollars
  - Should-not-trade flag with reason

The result is written to Redis key `engine:daily_focus` as JSON and served
by data-service via `GET /api/focus`.

Risk rules:
  - Risk per trade capped at 0.75% of account size (default $50k = $375)
  - Assets with quality < 55% flagged as NEUTRAL with "skip today" note
  - should_not_trade() returns True if ALL assets < 55% quality or
    max vol_percentile > 88%

Phase 3A: Top-4 Asset Selection for Live Trading
  - get_daily_plan_focus_assets() returns (scalp_names, swing_names) from
    the daily plan persisted in Redis, or generates a fresh plan if none exists
  - compute_daily_focus() accepts optional focus_assets filter to only
    compute full focus data for the selected assets (others returned as stubs)

Phase 5C: Dynamic Position Sizing on Focus Cards
  - compute_asset_focus() accepts an optional `live_risk: LiveRiskState`
  - When provided, remaining_risk_budget replaces static max_risk_per_trade
  - Dual micro/full contract sizing shown side by side
  - Live position overlay when there's an active position on this asset

Phase 5D: Live Position Overlay on Focus Cards
  - When in a trade, the focus card data includes position info
  - direction, entry, current P&L, bracket phase, R-multiple
  - Card flips from "setup" mode to "live position" mode

Usage:
    from lib.services.engine.focus import compute_daily_focus, should_not_trade

    focus = compute_daily_focus(account_size=50_000)
    if should_not_trade(focus):
        print("NO TRADE today")

    # With live risk state for dynamic sizing:
    from lib.services.engine.live_risk import compute_live_risk
    live_risk = compute_live_risk(risk_manager, position_manager)
    focus = compute_daily_focus(account_size=50_000, live_risk=live_risk)

    # With daily plan focus narrowing (Phase 3A):
    focus = compute_daily_focus(account_size=50_000, live_risk=live_risk, use_daily_plan=True)
    # Only the top 3-4 scalp focus assets get full computation; the rest are stubs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from lib.core.utils import safe_float as _safe_float

if TYPE_CHECKING:
    from lib.services.engine.live_risk import LiveRiskState

logger = logging.getLogger("engine.focus")

_EST = ZoneInfo("America/New_York")

# Default risk parameters
DEFAULT_ACCOUNT_SIZE = 50_000
DEFAULT_RISK_PCT = 0.0075  # 0.75% per trade
MIN_QUALITY_THRESHOLD = 0.55  # 55% — below this, flag as NEUTRAL/skip
EXTREME_VOL_THRESHOLD = 0.88  # 88th percentile — too volatile


def _price_decimals(tick_size: float) -> int:
    """Return the number of decimal places needed to represent one tick.

    Examples:
        0.25    → 2   (ES, NQ, MES, MNQ)
        0.10    → 2   (MGC, M2K)
        0.01    → 2   (MCL)
        0.0001  → 4   (M6B, 6A, 6C, 6S)
        0.00005 → 5   (6E full-size tick; micro uses 0.0001 but keep 5 as max)
        0.0000005 → 7 (6J)
    """
    if tick_size <= 0:
        return 4
    # Express tick as string and count decimal places
    s = f"{tick_size:.10f}".rstrip("0")
    if "." not in s:
        return 2
    decimals = len(s.split(".")[1])
    # Always show at least 2 decimal places, at most 7
    return max(2, min(decimals, 7))


def _compute_entry_zone(
    last_price: float,
    bias: str,
    atr: float,
    wave_ratio: float,
    tick_size: float = 0.01,
) -> dict[str, float]:
    """Compute entry zone, stop, TP1, TP2 based on bias and ATR.

    Entry zone is a range around the current price adjusted by ATR.
    Stop is placed at 1.5× ATR from entry midpoint.
    TP1 at 2× ATR, TP2 at 3.5× ATR (for scaling out).

    Rounding uses tick-aware precision so that forex pairs (e.g. 6E with
    tick=0.00005) are displayed with enough decimal places instead of
    collapsing to 4 decimals where entry_low == entry_high.
    """
    if atr <= 0:
        atr = last_price * 0.005  # fallback: 0.5% of price

    # Determine display precision from tick size
    decimals = _price_decimals(tick_size)

    # Tighter entries when wave is strong
    entry_width = atr * 0.5 if wave_ratio > 1.5 else atr * 0.75

    if bias == "LONG":
        entry_low = last_price - entry_width
        entry_high = last_price + entry_width * 0.3
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint - atr * 1.5
        tp1 = midpoint + atr * 2.0
        tp2 = midpoint + atr * 3.5
    elif bias == "SHORT":
        entry_high = last_price + entry_width
        entry_low = last_price - entry_width * 0.3
        midpoint = (entry_low + entry_high) / 2
        stop = midpoint + atr * 1.5
        tp1 = midpoint - atr * 2.0
        tp2 = midpoint - atr * 3.5
    else:
        # NEUTRAL — no real entry, just show reference levels
        entry_low = last_price - atr
        entry_high = last_price + atr
        midpoint = last_price
        stop = last_price - atr * 2.0
        tp1 = last_price + atr * 2.0
        tp2 = last_price + atr * 3.5

    return {
        "entry_low": round(entry_low, decimals),
        "entry_high": round(entry_high, decimals),
        "stop": round(stop, decimals),
        "tp1": round(tp1, decimals),
        "tp2": round(tp2, decimals),
        "price_decimals": decimals,
    }


def _compute_position_size(
    last_price: float,
    stop_price: float,
    tick_size: float,
    point_value: float,
    max_risk_dollars: float,
) -> tuple[int, float]:
    """Calculate position size in micro contracts from stop distance.

    Returns:
        (position_size, risk_dollars) — contracts and actual risk in $
    """
    stop_distance = abs(last_price - stop_price)
    if stop_distance <= 0 or tick_size <= 0 or point_value <= 0:
        return 1, max_risk_dollars

    # Number of ticks in the stop distance
    ticks = stop_distance / tick_size
    # Dollar risk per contract = ticks × (tick_size × point_value)
    # For micro contracts, point_value already reflects micro sizing
    dollar_per_tick = tick_size * point_value
    risk_per_contract = ticks * dollar_per_tick

    if risk_per_contract <= 0:
        return 1, max_risk_dollars

    # Max contracts within risk budget
    contracts = int(max_risk_dollars / risk_per_contract)
    contracts = max(1, contracts)

    actual_risk = contracts * risk_per_contract

    return contracts, round(actual_risk, 2)


def _derive_bias(
    wave_result: dict[str, Any],
    sq_result: dict[str, Any],
    quality: float,
) -> str:
    """Derive trading bias from wave analysis + signal quality context.

    Bias = wave dominance direction, confirmed by AO and quality threshold.
    """
    if quality < MIN_QUALITY_THRESHOLD:
        return "NEUTRAL"

    wave_bias = wave_result.get("bias", "NEUTRAL")
    dominance = _safe_float(wave_result.get("dominance", 0.0))
    ao = _safe_float(sq_result.get("ao", 0.0))
    market_context = sq_result.get("market_context", "RANGING")

    # Strong directional bias from waves
    if wave_bias == "BULLISH" and dominance > 0.05:
        # Confirm with AO or context
        if ao > 0 or market_context == "UPTREND":
            return "LONG"
        # Weaker confirmation — still LONG but less confident
        if dominance > 0.15:
            return "LONG"
    elif wave_bias == "BEARISH" and dominance < -0.05:
        if ao < 0 or market_context == "DOWNTREND":
            return "SHORT"
        if dominance < -0.15:
            return "SHORT"

    return "NEUTRAL"


def _compute_dual_sizing(
    name: str,
    entry_price: float,
    stop_price: float,
    max_risk_dollars: float,
) -> dict[str, Any]:
    """Compute position sizing for BOTH micro and full contracts side by side.

    Uses the asset registry to look up both contract variants and computes
    the number of contracts and dollar P&L estimates for each.

    Returns a dict with "micro" and "full" keys (either may be absent if
    the variant doesn't exist for this asset).
    """
    try:
        from lib.core.asset_registry import get_asset

        asset = get_asset(name)
        if asset is not None:
            return asset.dual_sizing(entry_price, stop_price, max_risk_dollars)
    except ImportError:
        pass

    # Fallback: return empty if asset registry not available
    return {}


def _build_live_position_overlay(
    name: str,
    live_risk: LiveRiskState,
) -> dict[str, Any] | None:
    """Build position overlay data if there's an active position on this asset.

    Returns a dict with live position info for the focus card, or None if
    no position is open.
    """
    pos = live_risk.get_position_for_asset(name)
    if pos is None:
        return None

    # Format hold duration
    hold_secs = pos.hold_duration_seconds
    if hold_secs >= 3600:
        hold_str = f"{hold_secs // 3600}h {(hold_secs % 3600) // 60}m"
    elif hold_secs >= 60:
        hold_str = f"{hold_secs // 60}m {hold_secs % 60}s"
    else:
        hold_str = f"{hold_secs}s"

    return {
        "has_live_position": True,
        "position_side": pos.side,
        "position_quantity": pos.quantity,
        "position_entry_price": pos.entry_price,
        "position_current_price": pos.current_price,
        "position_stop_price": pos.stop_price,
        "position_unrealized_pnl": round(pos.unrealized_pnl, 2),
        "position_r_multiple": round(pos.r_multiple, 2),
        "position_bracket_phase": pos.bracket_phase,
        "position_hold_duration": hold_str,
        "position_hold_seconds": hold_secs,
        "position_risk_dollars": round(pos.risk_dollars, 2),
        "position_source": pos.source,
        "position_symbol": pos.symbol,
    }


def compute_asset_focus(
    name: str,
    account_size: int = DEFAULT_ACCOUNT_SIZE,
    live_risk: LiveRiskState | None = None,
) -> dict[str, Any] | None:
    """Compute focus data for a single asset.

    Args:
        name: Asset name (e.g. "Gold", "S&P", "Nasdaq")
        account_size: Account size for risk calculations
        live_risk: Optional LiveRiskState for dynamic sizing + position overlay.
                   When provided:
                     - remaining_risk_budget replaces static max_risk_per_trade
                     - Dual micro/full sizing computed side by side
                     - Live position overlay included if in a trade
                     - "MAX POSITIONS" / "RISK BLOCKED" badges when applicable

    Returns dict with all focus fields, or None on failure.
    """
    try:
        from lib.analysis.signal_quality import compute_signal_quality
        from lib.analysis.volatility import kmeans_volatility_clusters
        from lib.analysis.wave_analysis import calculate_wave_analysis
        from lib.core.cache import get_data
        from lib.core.models import (  # noqa: F401
            ASSETS,
            CONTRACT_SPECS,
            MICRO_CONTRACT_SPECS,
        )
    except ImportError as exc:
        logger.error("Failed to import required modules: %s", exc)
        return None

    ticker = ASSETS.get(name)
    if not ticker:
        logger.warning("Unknown asset: %s", name)
        return None

    # Get micro contract specs for position sizing
    spec = MICRO_CONTRACT_SPECS.get(name, {})
    tick_size = _safe_float(spec.get("tick", 0.01))
    point_value = _safe_float(spec.get("point", 1.0))

    # Fetch data
    try:
        df = get_data(ticker, "5m", "5d")
        if df is None or df.empty or len(df) < 30:
            logger.warning(
                "Insufficient data for %s (%s): %d bars",
                name,
                ticker,
                len(df) if df is not None else 0,
            )
            return None
    except Exception as exc:
        logger.warning("Data fetch failed for %s: %s", name, exc)
        return None

    last_price = _safe_float(df["Close"].iloc[-1])
    if last_price <= 0:
        logger.warning("Invalid last price for %s: %s", name, last_price)
        return None

    # Run analysis modules
    try:
        wave_result = calculate_wave_analysis(df, asset_name=name)
    except Exception as exc:
        logger.warning("Wave analysis failed for %s: %s", name, exc)
        wave_result = {"wave_ratio": 1.0, "bias": "NEUTRAL", "dominance": 0.0}

    try:
        vol_result = kmeans_volatility_clusters(df)
    except Exception as exc:
        logger.warning("Volatility analysis failed for %s: %s", name, exc)
        vol_result = {
            "percentile": 0.5,
            "raw_atr": last_price * 0.005,
            "adaptive_atr": last_price * 0.005,
            "cluster": "MEDIUM",
        }

    try:
        sq_result = compute_signal_quality(
            df,
            wave_result=wave_result,
            vol_result=vol_result,
        )
    except Exception as exc:
        logger.warning("Signal quality failed for %s: %s", name, exc)
        sq_result = {
            "score": 0.0,
            "quality_pct": 0.0,
            "ao": 0.0,
            "market_context": "RANGING",
        }

    # Extract key metrics
    wave_ratio = _safe_float(wave_result.get("wave_ratio", 1.0))
    quality = _safe_float(sq_result.get("score", 0.0))
    quality_pct = _safe_float(sq_result.get("quality_pct", 0.0))
    vol_percentile = _safe_float(vol_result.get("percentile", 0.5))
    atr = _safe_float(vol_result.get("raw_atr", 0.0))

    # Derive bias
    bias = _derive_bias(wave_result, sq_result, quality)

    # Compute levels — pass tick_size so forex gets correct decimal precision
    levels = _compute_entry_zone(last_price, bias, atr, wave_ratio, tick_size=tick_size)

    # ── Risk budget: static or dynamic ──────────────────────────────────
    # Phase 5C: When live_risk is provided, use remaining_risk_budget
    # instead of the static max_risk_per_trade.  This accounts for current
    # open positions, daily P&L drawdown, and consecutive losses.
    static_max_risk = account_size * DEFAULT_RISK_PCT
    max_risk = static_max_risk

    risk_blocked = False
    risk_blocked_reason = ""
    max_positions_reached = False

    if live_risk is not None:
        if not live_risk.can_trade:
            risk_blocked = True
            risk_blocked_reason = live_risk.block_reason or "Trading blocked"
            max_risk = 0.0
        elif live_risk.remaining_trade_slots <= 0:
            max_positions_reached = True
            max_risk = 0.0
        elif live_risk.remaining_risk_budget > 0:
            # Use the smaller of: remaining budget or per-trade max
            effective_per_trade = live_risk.max_risk_per_trade or static_max_risk
            max_risk = min(live_risk.remaining_risk_budget, effective_per_trade)
        # Fallback: if remaining_risk_budget is 0 but can_trade and slots > 0,
        # use static max_risk (defensive)

    # Position sizing (primary: micro contracts)
    position_size, risk_dollars = _compute_position_size(
        last_price=last_price,
        stop_price=levels["stop"],
        tick_size=tick_size,
        point_value=point_value,
        max_risk_dollars=max_risk,
    )

    # Override to 0 when risk is blocked or max positions reached
    if risk_blocked or max_positions_reached:
        position_size = 0
        risk_dollars = 0.0

    # Estimate dollar value of TP targets for the display card.
    # dollar_per_tick = tick_size × point_value (already in micro-contract terms)
    dollar_per_tick = tick_size * point_value if tick_size > 0 and point_value > 0 else 0.0
    if dollar_per_tick > 0 and tick_size > 0:
        ticks_to_tp1 = abs(levels["tp1"] - last_price) / tick_size
        ticks_to_tp2 = abs(levels["tp2"] - last_price) / tick_size
        target1_dollars = float(int(position_size * ticks_to_tp1 * dollar_per_tick))
        target2_dollars = float(int(position_size * ticks_to_tp2 * dollar_per_tick))
    else:
        target1_dollars = 0.0
        target2_dollars = 0.0

    # ── Dual micro/full sizing (Phase 5C) ───────────────────────────────
    # Compute position sizes for BOTH micro and full contracts using the
    # asset registry, so the trader sees side-by-side values on the card.
    dual_sizing: dict[str, Any] = {}
    if not risk_blocked and not max_positions_reached and max_risk > 0:
        dual_sizing = _compute_dual_sizing(
            name=name,
            entry_price=last_price,
            stop_price=levels["stop"],
            max_risk_dollars=max_risk,
        )
        # Enrich with TP dollar estimates for each variant
        stop_distance = abs(last_price - levels["stop"])
        for variant_key, sizing in dual_sizing.items():
            if stop_distance > 0 and sizing.get("contracts", 0) > 0:
                try:
                    from lib.core.asset_registry import get_asset

                    asset_obj = get_asset(name)
                    if asset_obj:
                        v = asset_obj.variants.get(variant_key)
                        if v:
                            pv = v.point_value
                            c = sizing["contracts"]
                            sizing["tp1_dollars"] = round(c * abs(levels["tp1"] - last_price) * pv, 2)
                            sizing["tp2_dollars"] = round(c * abs(levels["tp2"] - last_price) * pv, 2)
                except ImportError:
                    pass

    # ── Live position overlay (Phase 5D) ────────────────────────────────
    # When in a trade on this asset, include position info so the card
    # can flip from "setup" mode to "live position" mode.
    position_overlay: dict[str, Any] | None = None
    if live_risk is not None:
        position_overlay = _build_live_position_overlay(name, live_risk)

    # Build skip note
    notes = []
    if risk_blocked:
        notes.append(f"🚫 RISK BLOCKED: {risk_blocked_reason}")
    elif max_positions_reached:
        notes.append("⚠️ MAX POSITIONS — no new entries")
    if quality < MIN_QUALITY_THRESHOLD:
        notes.append(f"Quality too low ({quality_pct:.0f}%) — skip today")
    if vol_percentile > EXTREME_VOL_THRESHOLD:
        notes.append(f"Extreme volatility ({vol_percentile:.0%}) — dangerous")

    price_decimals = int(levels.get("price_decimals", 4))

    result: dict[str, Any] = {
        "symbol": name,
        "ticker": ticker,
        "bias": bias,
        "bias_emoji": {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(bias, "⚪"),
        "last_price": round(last_price, price_decimals),
        "entry_low": levels["entry_low"],
        "entry_high": levels["entry_high"],
        "stop": levels["stop"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "price_decimals": price_decimals,
        "wave_ratio": round(wave_ratio, 2),
        "wave_ratio_text": wave_result.get("wave_ratio_text", f"{wave_ratio:.2f}x"),
        "quality": round(quality, 3),
        "quality_pct": round(quality_pct, 1),
        "high_quality": quality >= MIN_QUALITY_THRESHOLD,
        "vol_percentile": round(vol_percentile, 4),
        "vol_cluster": vol_result.get("cluster", "MEDIUM"),
        "market_phase": wave_result.get("market_phase", "UNKNOWN"),
        "trend_direction": wave_result.get("trend_direction", "NEUTRAL ↔️"),
        "momentum_state": wave_result.get("momentum_state", "NEUTRAL"),
        "dominance_text": wave_result.get("dominance_text", "Neutral"),
        "position_size": position_size,
        "risk_dollars": risk_dollars,
        "target1_dollars": target1_dollars,
        "target2_dollars": target2_dollars,
        "max_risk_allowed": round(max_risk, 2),
        "atr": round(atr, 6),
        "notes": "; ".join(notes) if notes else "",
        "skip": quality < MIN_QUALITY_THRESHOLD,
        # ── Phase 5C: Dual sizing ──────────────────────────────────────
        "dual_sizing": dual_sizing,
        "risk_blocked": risk_blocked,
        "risk_blocked_reason": risk_blocked_reason,
        "max_positions_reached": max_positions_reached,
        # ── Phase 5D: Live position overlay ────────────────────────────
        "has_live_position": position_overlay is not None,
        "live_position": position_overlay,
    }

    return result


def get_daily_plan_focus_assets(
    redis_client: Any | None = None,
    account_size: int = DEFAULT_ACCOUNT_SIZE,
    force_regenerate: bool = False,
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    """Get the focused asset lists from the daily plan.

    Checks Redis for a persisted daily plan first. If none exists (or
    ``force_regenerate`` is True), generates a fresh plan.

    Args:
        redis_client: Optional Redis client. If None, attempts to get one
                      from the cache module.
        account_size: Account size for plan generation.
        force_regenerate: If True, always generate a new plan even if one
                          exists in Redis.

    Returns:
        (scalp_names, swing_names, plan_dict) where:
          - scalp_names: list of 3-4 asset names for scalp focus
          - swing_names: list of 1-2 asset names for daily swing candidates
          - plan_dict: the full plan dict (for dashboard enrichment), or None
    """
    # Try to get Redis client if not provided
    if redis_client is None:
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                redis_client = _r
        except ImportError:
            pass

    # Attempt to load existing plan from Redis
    if redis_client is not None and not force_regenerate:
        try:
            from lib.trading.strategies.daily.daily_plan import DailyPlan

            plan = DailyPlan.load_from_redis(redis_client)
            if plan is not None:
                scalp_names = [a.asset_name for a in plan.scalp_focus]
                swing_names = [s.asset_name for s in plan.swing_candidates]
                logger.info(
                    "Loaded daily plan from Redis: scalp=%s, swing=%s",
                    scalp_names,
                    swing_names,
                )
                return scalp_names, swing_names, plan.to_dict()
        except Exception as exc:
            logger.debug("Could not load daily plan from Redis: %s", exc)

    # No cached plan — generate a fresh one
    try:
        from lib.trading.strategies.daily.daily_plan import generate_daily_plan

        plan = generate_daily_plan(
            account_size=account_size,
            redis_client=redis_client,
            include_grok=True,
        )

        # Publish if we have Redis
        if redis_client is not None:
            plan.publish_to_redis(redis_client)

        scalp_names = [a.asset_name for a in plan.scalp_focus]
        swing_names = [s.asset_name for s in plan.swing_candidates]
        return scalp_names, swing_names, plan.to_dict()

    except Exception as exc:
        logger.error("Failed to generate daily plan: %s", exc)
        return [], [], None


def compute_daily_focus(
    account_size: int = DEFAULT_ACCOUNT_SIZE,
    symbols: list[str] | None = None,
    live_risk: LiveRiskState | None = None,
    use_daily_plan: bool = False,
    redis_client: Any | None = None,
) -> dict[str, Any]:
    """Compute full daily focus payload for all tracked assets.

    Args:
        account_size: Account size for risk calculations
        symbols: List of asset names to compute (None = all tracked)
        live_risk: Optional LiveRiskState for dynamic sizing.
                   When provided, focus cards use remaining_risk_budget
                   and include live position overlays.
        use_daily_plan: If True, load or generate a daily plan (Phase 3A)
                        and tag each asset with its focus category
                        (scalp_focus, swing, or background).
        redis_client: Optional Redis client for daily plan lookups.

    Returns a dict with:
      - assets: list of per-asset focus dicts
      - no_trade: bool — True if should_not_trade()
      - no_trade_reason: str
      - computed_at: ISO timestamp
      - account_size: int
      - session_mode: current session
      - live_risk_active: bool — whether live_risk was used
      - daily_plan: dict or None — the daily plan data (when use_daily_plan=True)
      - scalp_focus_names: list[str] — focused scalp asset names
      - swing_candidate_names: list[str] — swing candidate asset names
    """
    try:
        from lib.core.models import ASSETS as _imported_assets

        assets_map: dict[str, str] = dict(_imported_assets)
    except ImportError:
        assets_map = {}

    if symbols is None:
        symbols = list(assets_map.keys())

    now = datetime.now(tz=_EST)

    # ── Phase 3A: Daily plan focus narrowing ────────────────────────────
    scalp_focus_names: list[str] = []
    swing_candidate_names: list[str] = []
    daily_plan_data: dict[str, Any] | None = None

    if use_daily_plan:
        scalp_focus_names, swing_candidate_names, daily_plan_data = get_daily_plan_focus_assets(
            redis_client=redis_client,
            account_size=account_size,
        )

    # Combined set of "important" assets — these get full computation
    important_assets = set(scalp_focus_names) | set(swing_candidate_names)

    logger.info(
        "Computing daily focus for %d assets (account=$%s%s%s)",
        len(symbols),
        f"{account_size:,}",
        ", live_risk=active" if live_risk else "",
        f", plan_focus={len(important_assets)}" if important_assets else "",
    )

    asset_results = []
    for name in symbols:
        try:
            result = compute_asset_focus(name, account_size=account_size, live_risk=live_risk)
            if result is not None:
                # Tag with focus category (Phase 3A)
                if name in scalp_focus_names:
                    result["focus_category"] = "scalp_focus"
                    result["focus_rank"] = scalp_focus_names.index(name) + 1
                elif name in swing_candidate_names:
                    result["focus_category"] = "swing"
                    result["focus_rank"] = swing_candidate_names.index(name) + 1
                else:
                    result["focus_category"] = "background"
                    result["focus_rank"] = 999

                asset_results.append(result)
                logger.info(
                    "  %s: %s %s | quality=%.0f%% | wave=%.2fx | price=%.2f%s",
                    name,
                    result["bias_emoji"],
                    result["bias"],
                    result["quality_pct"],
                    result["wave_ratio"],
                    result["last_price"],
                    f" [{result['focus_category']}]" if result["focus_category"] != "background" else "",
                )
            else:
                logger.warning("  %s: no data available", name)
        except Exception as exc:
            logger.error("  %s: focus computation failed: %s", name, exc)

    # Sort: focused assets first (by rank), then background by quality
    asset_results.sort(
        key=lambda x: (
            0 if x.get("focus_category") == "scalp_focus" else (1 if x.get("focus_category") == "swing" else 2),
            x.get("focus_rank", 999),
            -x.get("quality", 0),
            -x.get("wave_ratio", 0),
        ),
    )

    # Check no-trade conditions
    no_trade, no_trade_reason = should_not_trade(asset_results)

    # Determine session
    hour = now.hour
    if 0 <= hour < 5:
        session = "pre-market"
    elif 5 <= hour < 12:
        session = "active"
    else:
        session = "off-hours"

    # Count live positions from focus data
    live_positions = sum(1 for a in asset_results if a.get("has_live_position"))

    payload = {
        "assets": asset_results,
        "no_trade": no_trade,
        "no_trade_reason": no_trade_reason,
        "computed_at": now.isoformat(),
        "account_size": account_size,
        "session_mode": session,
        "total_assets": len(asset_results),
        "tradeable_assets": sum(1 for a in asset_results if not a.get("skip")),
        "live_risk_active": live_risk is not None,
        "live_positions": live_positions,
        "risk_blocked": live_risk.block_reason if live_risk and not live_risk.can_trade else "",
        # Phase 3A fields
        "daily_plan": daily_plan_data,
        "scalp_focus_names": scalp_focus_names,
        "swing_candidate_names": swing_candidate_names,
        "focus_mode_active": bool(important_assets),
    }

    logger.info(
        "Daily focus computed: %d assets, %d tradeable, no_trade=%s%s",
        len(asset_results),
        payload["tradeable_assets"],
        no_trade,
        f" ({no_trade_reason})" if no_trade else "",
    )

    return payload


def should_not_trade(
    focus_assets: list[dict[str, Any]],
    max_daily_loss: float = -250.0,
    max_consecutive_losses: int = 2,
) -> tuple[bool, str]:
    """Determine if today is a no-trade day.

    Conditions:
      1. ALL focus assets have quality < 55%
      2. Any focus asset has volatility percentile > 88%
      3. Daily loss already exceeds -$250 (placeholder — needs trade log)
      4. More than 2 consecutive losing trades today (placeholder)
      5. After 10:00 AM ET and no setups triggered (placeholder)

    Returns:
        (should_skip, reason) — True if should not trade.
    """
    if not focus_assets:
        return True, "No market data available"

    # Condition 1: All assets below quality threshold
    qualities = [_safe_float(a.get("quality", 0)) for a in focus_assets]
    if all(q < MIN_QUALITY_THRESHOLD for q in qualities):
        best_q = max(qualities) * 100
        return (
            True,
            f"All assets below {MIN_QUALITY_THRESHOLD * 100:.0f}% quality (best: {best_q:.0f}%)",
        )

    # Condition 2: Any asset has extreme volatility
    vol_percentiles = [_safe_float(a.get("vol_percentile", 0)) for a in focus_assets]
    extreme_vols = [
        a.get("symbol", "?") for a in focus_assets if _safe_float(a.get("vol_percentile", 0)) > EXTREME_VOL_THRESHOLD
    ]
    if extreme_vols:
        max_vol = max(vol_percentiles)
        return True, (
            f"Extreme volatility on {', '.join(extreme_vols)} ({max_vol:.0%} percentile) — high risk of stop hunts"
        )

    # Condition 5: Time-based (after 10 AM and no high-quality setups)
    now = datetime.now(tz=_EST)
    if now.hour >= 10 and now.hour < 12:
        tradeable = [a for a in focus_assets if not a.get("skip")]
        if not tradeable:
            return (
                True,
                "After 10:00 AM ET with no quality setups — session winding down",
            )

    return False, ""


def publish_focus_to_redis(focus_data: dict[str, Any]) -> bool:
    """Write focus payload to Redis for data-service to serve.

    Writes to:
      - `engine:daily_focus` — full JSON payload (TTL 5 min)
      - `engine:daily_focus:ts` — last update timestamp
      - Redis Stream `dashboard:stream:focus` — for SSE catch-up
      - Redis PubSub `dashboard:live` — trigger for SSE push

    Returns True on success.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module")
        return False

    try:
        # Serialize with safe float handling
        payload_json = json.dumps(focus_data, default=str, allow_nan=False)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize focus data: %s", exc)
        return False

    try:
        # Write main focus key
        cache_set("engine:daily_focus", payload_json.encode(), ttl=300)

        # Write timestamp
        ts = datetime.now(tz=_EST).isoformat()
        cache_set("engine:daily_focus:ts", ts.encode(), ttl=300)

        # Write to Redis Stream for SSE catch-up (if Redis is available)
        if REDIS_AVAILABLE and _r is not None:
            try:
                # Add to stream (keep last 100 entries, auto-trim)
                _r.xadd(
                    "dashboard:stream:focus",
                    {"data": payload_json, "ts": ts},
                    maxlen=100,
                    approximate=True,
                )
                # Publish trigger for SSE subscribers
                _r.publish("dashboard:live", payload_json)

                # Also publish per-asset events for granular SSE
                for asset in focus_data.get("assets", []):
                    symbol = asset.get("symbol", "").lower().replace(" ", "_")
                    if symbol:
                        asset_json = json.dumps(asset, default=str, allow_nan=False)
                        _r.publish(f"dashboard:asset:{symbol}", asset_json)

                # Publish no-trade event if applicable
                if focus_data.get("no_trade"):
                    _r.publish(
                        "dashboard:no_trade",
                        json.dumps(
                            {
                                "no_trade": True,
                                "reason": focus_data.get("no_trade_reason", ""),
                                "ts": ts,
                            }
                        ),
                    )

                # Phase 3B: Publish daily plan event when focus mode is active
                if focus_data.get("focus_mode_active") and focus_data.get("daily_plan"):
                    _r.publish(
                        "dashboard:daily_plan",
                        json.dumps(
                            {
                                "focus_mode_active": True,
                                "scalp_focus_names": focus_data.get("scalp_focus_names", []),
                                "swing_candidate_names": focus_data.get("swing_candidate_names", []),
                                "ts": ts,
                            }
                        ),
                    )

            except Exception as exc:
                logger.debug("Redis Stream/PubSub publish failed (non-fatal): %s", exc)

        logger.debug("Focus data published to Redis (key=engine:daily_focus)")
        return True

    except Exception as exc:
        logger.error("Failed to publish focus to Redis: %s", exc)
        return False
