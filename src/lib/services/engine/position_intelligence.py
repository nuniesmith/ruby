"""
Position Intelligence Engine — Phase POSINT
=============================================
Real-time per-position analysis ("live trading co-pilot").

Computes a rich intel overlay for each active position by wiring together
multiple analysis modules:

  - **ICT sweep zones** — nearby liquidity sweeps / FVGs / order blocks
  - **Confluence score** — multi-timeframe alignment strength (0–3)
  - **Volume profile levels** — POC / VAH / VAL for TP targeting
  - **CVD delta** — cumulative volume delta for buy/sell pressure
  - **Regime context** — trending / volatile / choppy classification

The primary entry point is ``compute_position_intelligence()``, which
returns a ``PositionIntel`` dataclass ready for dashboard rendering,
risk overlays, and automated bracket management.

Usage:
    from lib.services.engine.position_intelligence import (
        compute_position_intelligence,
        compute_multi_tp,
        assess_book_pressure,
        suggest_risk_actions,
    )

    intel = compute_position_intelligence(
        symbol="MES",
        entry_price=5420.0,
        direction="long",
        bars_1m=df_1m,
        bars_5m=df_5m,
    )
    actions = suggest_risk_actions(intel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "PositionIntel",
    "compute_position_intelligence",
    "compute_multi_tp",
    "assess_book_pressure",
    "suggest_risk_actions",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PositionIntel:
    """Rich intelligence overlay for a single active position.

    Attributes:
        sweep_zones: Nearby ICT liquidity sweep levels, FVGs, and order
            blocks that may act as support/resistance around the position.
        multi_tp_targets: Ordered list of take-profit price levels derived
            from ATR multiples and volume-profile levels.
        book_pressure: Level-2 order book pressure summary (bid/ask totals
            and imbalance ratio).  Empty dict when L2 data is unavailable.
        risk_actions: Suggested risk-management actions such as
            ``"trail_stop"``, ``"take_partial"``, or ``"hold"``.
        confluence_score: Multi-timeframe confluence score (0–3).
            3 = full alignment, 0 = no confluence.
        regime_context: Current market regime label
            (``"trending"`` | ``"volatile"`` | ``"choppy"``).
    """

    sweep_zones: list[dict[str, Any]] = field(default_factory=list)
    multi_tp_targets: list[float] = field(default_factory=list)
    book_pressure: dict[str, Any] = field(default_factory=dict)
    risk_actions: list[str] = field(default_factory=list)
    confluence_score: int = 0
    regime_context: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON / SSE transport."""
        return {
            "sweep_zones": self.sweep_zones,
            "multi_tp_targets": self.multi_tp_targets,
            "book_pressure": self.book_pressure,
            "risk_actions": self.risk_actions,
            "confluence_score": self.confluence_score,
            "regime_context": self.regime_context,
        }


# ---------------------------------------------------------------------------
# Lazy / guarded imports for analysis modules
# ---------------------------------------------------------------------------


def _import_ict_summary() -> Any:
    """Import ``ict_summary`` lazily so the module loads even if deps break."""
    try:
        from lib.analysis.ict import ict_summary

        return ict_summary
    except Exception:
        logger.warning("position_intelligence: could not import ict_summary")
        return None


def _import_check_confluence() -> Any:
    try:
        from lib.analysis.confluence import check_confluence

        return check_confluence
    except Exception:
        logger.warning("position_intelligence: could not import check_confluence")
        return None


def _import_compute_volume_profile() -> Any:
    try:
        from lib.analysis.volume_profile import compute_volume_profile

        return compute_volume_profile
    except Exception:
        logger.warning("position_intelligence: could not import compute_volume_profile")
        return None


def _import_compute_cvd() -> Any:
    try:
        from lib.analysis.cvd import compute_cvd

        return compute_cvd
    except Exception:
        logger.warning("position_intelligence: could not import compute_cvd")
        return None


def _import_regime_detector() -> Any:
    try:
        from lib.analysis.regime import RegimeDetector

        return RegimeDetector
    except Exception:
        logger.warning("position_intelligence: could not import RegimeDetector")
        return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_position_intelligence(
    symbol: str,
    entry_price: float,
    direction: str,
    bars_1m: pd.DataFrame,
    bars_5m: pd.DataFrame,
    l2_depth: dict[str, Any] | None = None,
) -> PositionIntel:
    """Compute a full intelligence overlay for an active position.

    Wires together ICT sweep zones, confluence scoring, volume-profile
    levels, CVD delta, and regime detection.  Each sub-call is wrapped in
    ``try/except`` so a single module failure never crashes the whole
    computation — the corresponding field simply falls back to a sensible
    default.

    Args:
        symbol: Instrument symbol (e.g. ``"MES"``).
        entry_price: Position entry price.
        direction: ``"long"`` or ``"short"``.
        bars_1m: 1-minute OHLCV DataFrame (recent history).
        bars_5m: 5-minute OHLCV DataFrame (recent history).
        l2_depth: Optional Level-2 order book snapshot with ``"bids"``
            and ``"asks"`` lists.

    Returns:
        Populated :class:`PositionIntel` dataclass.
    """
    # TODO: Wire real module — using mock/default for now
    logger.info(
        "compute_position_intelligence",
        symbol=symbol,
        entry_price=entry_price,
        direction=direction,
    )

    intel = PositionIntel()

    # --- ICT sweep zones ---------------------------------------------------
    try:
        ict_summary_fn = _import_ict_summary()
        if ict_summary_fn is not None and not bars_5m.empty:
            # TODO: Wire real module — using mock/default for now
            # summary = ict_summary_fn(bars_5m)
            # intel.sweep_zones = summary.get("nearest_levels", [])
            intel.sweep_zones = []
            logger.debug("ict sweep zones: placeholder active", symbol=symbol)
        else:
            intel.sweep_zones = []
    except Exception:
        logger.exception("Failed to compute ICT sweep zones", symbol=symbol)
        intel.sweep_zones = []

    # --- Confluence score --------------------------------------------------
    try:
        check_confluence_fn = _import_check_confluence()
        if check_confluence_fn is not None and not bars_1m.empty and not bars_5m.empty:
            # TODO: Wire real module — using mock/default for now
            # result = check_confluence_fn(
            #     htf_df=bars_5m,
            #     setup_df=bars_5m,
            #     entry_df=bars_1m,
            #     asset_name=symbol,
            # )
            # intel.confluence_score = result.get("score", 0)
            intel.confluence_score = 0
            logger.debug("confluence score: placeholder active", symbol=symbol)
        else:
            intel.confluence_score = 0
    except Exception:
        logger.exception("Failed to compute confluence score", symbol=symbol)
        intel.confluence_score = 0

    # --- Volume profile (POC / VAH / VAL for TP targeting) -----------------
    vp_levels: dict[str, float] = {}
    try:
        compute_vp = _import_compute_volume_profile()
        if compute_vp is not None and not bars_5m.empty:
            # TODO: Wire real module — using mock/default for now
            # profile = compute_vp(bars_5m, n_bins=50)
            # vp_levels = {
            #     "poc": profile.get("poc", entry_price),
            #     "vah": profile.get("vah", entry_price),
            #     "val": profile.get("val", entry_price),
            # }
            vp_levels = {"poc": entry_price, "vah": entry_price, "val": entry_price}
            logger.debug("volume profile: placeholder active", symbol=symbol)
    except Exception:
        logger.exception("Failed to compute volume profile", symbol=symbol)

    # --- CVD (cumulative volume delta) -------------------------------------
    try:
        compute_cvd_fn = _import_compute_cvd()
        if compute_cvd_fn is not None and not bars_1m.empty:
            # TODO: Wire real module — using mock/default for now
            # cvd_df = compute_cvd_fn(bars_1m)
            # latest_delta = float(cvd_df["cvd"].iloc[-1]) if "cvd" in cvd_df.columns else 0.0
            pass
            logger.debug("CVD: placeholder active", symbol=symbol)
    except Exception:
        logger.exception("Failed to compute CVD", symbol=symbol)

    # --- Regime detection --------------------------------------------------
    try:
        RegimeDetector = _import_regime_detector()
        if RegimeDetector is not None and not bars_5m.empty:
            # TODO: Wire real module — using mock/default for now
            # detector = RegimeDetector()
            # detector.fit(bars_5m)
            # info = detector.detect(bars_5m)
            # intel.regime_context = info.get("regime", "unknown")
            intel.regime_context = "unknown"
            logger.debug("regime: placeholder active", symbol=symbol)
        else:
            intel.regime_context = "unknown"
    except Exception:
        logger.exception("Failed to detect regime", symbol=symbol)
        intel.regime_context = "unknown"

    # --- Multi-TP targets --------------------------------------------------
    try:
        # ATR approximation from 5m bars (simple fallback)
        atr_value = _estimate_atr(bars_5m) if not bars_5m.empty else 0.0
        intel.multi_tp_targets = compute_multi_tp(
            entry=entry_price,
            direction=direction,
            atr=atr_value,
            vp_levels=vp_levels,
        )
    except Exception:
        logger.exception("Failed to compute multi-TP targets", symbol=symbol)
        intel.multi_tp_targets = []

    # --- L2 book pressure --------------------------------------------------
    try:
        intel.book_pressure = assess_book_pressure(l2_depth)
    except Exception:
        logger.exception("Failed to assess book pressure", symbol=symbol)
        intel.book_pressure = {}

    # --- Risk actions ------------------------------------------------------
    try:
        intel.risk_actions = suggest_risk_actions(intel)
    except Exception:
        logger.exception("Failed to suggest risk actions", symbol=symbol)
        intel.risk_actions = ["hold"]

    return intel


# ---------------------------------------------------------------------------
# Multi take-profit levels
# ---------------------------------------------------------------------------


def compute_multi_tp(
    entry: float,
    direction: str,
    atr: float,
    vp_levels: dict[str, float],
) -> list[float]:
    """Compute a list of take-profit price levels.

    Strategy:
        - TP1: 1× ATR from entry (quick scalp)
        - TP2: 2× ATR from entry (swing target)
        - TP3: nearest volume-profile level beyond 2× ATR (POC / VAH / VAL)

    Args:
        entry: Position entry price.
        direction: ``"long"`` or ``"short"``.
        atr: Current ATR value (e.g. from 5m bars).
        vp_levels: Dict with ``"poc"``, ``"vah"``, ``"val"`` prices.

    Returns:
        Ascending (long) or descending (short) list of TP prices.
    """
    # TODO: Wire real module — using mock/default for now
    if atr <= 0:
        return []

    sign = 1.0 if direction == "long" else -1.0

    tp1 = round(entry + sign * atr, 2)
    tp2 = round(entry + sign * 2.0 * atr, 2)

    # TP3: pick the best VP level beyond TP2
    tp3 = tp2  # fallback
    poc = vp_levels.get("poc", entry)
    vah = vp_levels.get("vah", entry)
    val = vp_levels.get("val", entry)

    candidates = [poc, vah, val]
    beyond = [lvl for lvl in candidates if (direction == "long" and lvl > tp2) or (direction == "short" and lvl < tp2)]
    if beyond:
        tp3 = round(min(beyond) if direction == "long" else max(beyond), 2)

    targets = sorted({tp1, tp2, tp3})
    if direction == "short":
        targets = sorted(targets, reverse=True)

    return targets


# ---------------------------------------------------------------------------
# Level-2 book pressure
# ---------------------------------------------------------------------------


def assess_book_pressure(
    l2_depth: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarise Level-2 order-book pressure.

    Args:
        l2_depth: Dict with ``"bids"`` and ``"asks"`` lists, where each
            entry is ``{"price": float, "size": int}``.  ``None`` if L2
            data is unavailable.

    Returns:
        Dict with ``bid_total``, ``ask_total``, ``imbalance_ratio``
        (bid_total / ask_total, clamped to [0.1, 10.0]).  Returns empty
        dict when no data is available.
    """
    # TODO: Wire real module — using mock/default for now
    if not l2_depth:
        return {}

    bids: list[dict[str, Any]] = l2_depth.get("bids", [])
    asks: list[dict[str, Any]] = l2_depth.get("asks", [])

    bid_total = sum(float(b.get("size", 0)) for b in bids)
    ask_total = sum(float(a.get("size", 0)) for a in asks)

    imbalance_ratio = 10.0 if ask_total == 0 else round(min(max(bid_total / ask_total, 0.1), 10.0), 3)

    return {
        "bid_total": bid_total,
        "ask_total": ask_total,
        "imbalance_ratio": imbalance_ratio,
    }


# ---------------------------------------------------------------------------
# Risk action suggestions
# ---------------------------------------------------------------------------


def suggest_risk_actions(intel: PositionIntel) -> list[str]:
    """Suggest risk-management actions based on the current intel snapshot.

    Potential actions:
        - ``"trail_stop"``   — conditions favour tightening the stop
        - ``"take_partial"`` — consider partial profit-taking
        - ``"hold"``         — no action needed; let the position run
        - ``"add_size"``     — regime + confluence support adding
        - ``"exit_full"``    — urgent exit signal

    Args:
        intel: A populated :class:`PositionIntel` instance.

    Returns:
        Ordered list of action strings (highest priority first).
    """
    # TODO: Wire real module — using mock/default for now
    actions: list[str] = []

    # Placeholder heuristic: suggest trailing if confluence is high
    if intel.confluence_score >= 3:
        actions.append("trail_stop")

    # Suggest partial TP if book is imbalanced against the position
    imbalance = intel.book_pressure.get("imbalance_ratio", 1.0)
    if imbalance < 0.5:
        actions.append("take_partial")

    # Choppy regime → conservative
    if intel.regime_context == "choppy":
        actions.append("take_partial")

    # Default: hold
    if not actions:
        actions.append("hold")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    return unique


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_atr(
    bars: pd.DataFrame,
    period: int = 14,
) -> float:
    """Quick ATR estimate from OHLCV bars (no external dependency).

    Uses the standard True Range formula:
        TR = max(H-L, |H-prev_C|, |L-prev_C|)
        ATR = EMA(TR, period)

    Returns 0.0 if the DataFrame is too small.
    """
    if bars.empty or len(bars) < period + 1:
        return 0.0

    try:
        high = bars["High"].astype(float)
        low = bars["Low"].astype(float)
        close = bars["Close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        logger.debug("_estimate_atr failed, returning 0.0")
        return 0.0
