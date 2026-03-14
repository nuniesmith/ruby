"""
Pre-session composite scorer for tradeable instrument selection.

Systematizes the morning (or pre-session) instrument selection process using
five weighted metrics. Works with any asset class — equities, futures, crypto,
forex, commodities — as long as OHLCV data is supplied.

Composite Score Formula (0-100):
  1. Normalized ATR (30%):  NATR = ATR_14 / close × 100, vs 20-day average
  2. Relative Volume (25%): RVOL = current_volume / 20-day avg volume
  3. Overnight Gap (15%):   |session_open - prior_close| / prior_close × 100
  4. Economic Catalyst (20%): tiered 0/33/66/100 based on event impact
  5. Momentum Score (10%):  |close - EMA_20| / ATR_14

Asset classes are identified by *tag strings* rather than instrument names.
Standard tags: "equity_index", "equity", "commodity_precious",
"commodity_energy", "commodity_metals", "forex", "crypto", "rates".

Output:
  - Traffic-light table sorted by composite score
  - Per-instrument detail cards with metric breakdowns
  - Focus recommendation (top 2-3 instruments)

Usage:
    from lib.analysis.scorer import PreMarketScorer, score_instruments, EVENT_CATALOG

    scorer = PreMarketScorer()
    results = scorer.score_all(data_dict, daily_dict)
    for r in results:
        print(f"{r['asset']}: {r['composite_score']:.1f} ({r['signal']})")
"""

import logging
import math
from datetime import time
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("scorer")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "natr": 0.30,
    "rvol": 0.25,
    "gap": 0.15,
    "catalyst": 0.20,
    "momentum": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Standard asset-class tag strings
# ---------------------------------------------------------------------------
#
# "equity_index"       – broad equity index products (e.g. S&P 500, Nasdaq, Dow)
# "equity"             – individual equities / sector ETFs
# "commodity_precious" – precious metals (e.g. Gold, Silver, Platinum)
# "commodity_energy"   – energy commodities (e.g. Crude Oil, Natural Gas, RBOB)
# "commodity_metals"   – industrial/base metals (e.g. Copper, Aluminum)
# "forex"              – foreign-exchange pairs (e.g. EUR/USD, DXY)
# "crypto"             – cryptocurrency assets (e.g. Bitcoin, Ethereum)
# "rates"              – interest-rate products (e.g. Treasuries, Eurodollar)


# ---------------------------------------------------------------------------
# Economic event catalog — maps events to affected asset-class tags
# ---------------------------------------------------------------------------


# Impact levels: 0 = none, 33 = low, 66 = medium, 100 = high
class _EventEntry(TypedDict):
    impact: int
    asset_tags: list[str]


EVENT_CATALOG: dict[str, _EventEntry] = {
    # Macro / monetary policy
    "FOMC": {
        "impact": 100,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex", "crypto"],
    },
    "Fed Minutes": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex"],
    },
    # Inflation
    "CPI": {
        "impact": 100,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex", "crypto"],
    },
    "PPI": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex"],
    },
    "PCE Price Index": {
        "impact": 100,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex"],
    },
    # Labor market
    "NFP": {
        "impact": 100,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates", "forex", "crypto"],
    },
    "Unemployment Claims": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "rates", "forex"],
    },
    "ADP Employment": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "rates", "forex"],
    },
    # Growth / activity
    "GDP": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "rates", "forex"],
    },
    "ISM Manufacturing": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity", "commodity_metals"],
    },
    "ISM Services": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity"],
    },
    "Retail Sales": {
        "impact": 66,
        "asset_tags": ["equity_index", "equity"],
    },
    "Durable Goods": {
        "impact": 33,
        "asset_tags": ["equity_index", "equity"],
    },
    "Consumer Confidence": {
        "impact": 33,
        "asset_tags": ["equity_index", "equity"],
    },
    "Housing Starts": {
        "impact": 33,
        "asset_tags": ["equity_index", "equity"],
    },
    # Energy
    "EIA Crude Inventory": {
        "impact": 100,
        "asset_tags": ["commodity_energy"],
    },
    "EIA Natural Gas": {
        "impact": 66,
        "asset_tags": ["commodity_energy"],
    },
    "OPEC Meeting": {
        "impact": 100,
        "asset_tags": ["commodity_energy"],
    },
    "OPEC+ Production": {
        "impact": 66,
        "asset_tags": ["commodity_energy"],
    },
    # FX / Dollar
    "DXY/USD Strength": {
        "impact": 66,
        "asset_tags": ["commodity_precious", "commodity_metals", "forex"],
    },
    # Rates / fixed income
    "Treasury Auction": {
        "impact": 33,
        "asset_tags": ["equity_index", "equity", "commodity_precious", "rates"],
    },
}


# ---------------------------------------------------------------------------
# Session hours for overnight/pre-session range calculation (EST)
#
# Keyed by asset-class tag.  Each entry may use None for "close" to indicate
# a 24/7 market with no defined close (e.g. crypto).
#
# Examples of instruments that fall under each tag:
#   "equity_index"       – ES, NQ, YM, RTY (CME Globex 6 PM – 9:30 AM ET)
#   "commodity_precious" – GC, SI, PL      (CME Globex 6 PM – 8:20 AM ET)
#   "commodity_energy"   – CL, NG, RB      (CME Globex 6 PM – 9:00 AM ET)
#   "commodity_metals"   – HG, ALI         (CME Globex 6 PM – 8:20 AM ET)
#   "equity"             – individual stocks/ETFs (regular session 9:30–16:00 ET)
#   "forex"              – FX pairs         (Sun 5 PM – Fri 5 PM ET, near-24/5)
#   "crypto"             – BTC, ETH, etc.   (24/7, no close)
#   "rates"              – ZN, ZB, GE       (CME Globex 6 PM – 9:30 AM ET)
# ---------------------------------------------------------------------------

SESSION_HOURS: dict[str, dict[str, time | None]] = {
    # CME Globex overnight session
    "equity_index": {"open": time(18, 0), "close": time(9, 30)},
    "rates": {"open": time(18, 0), "close": time(9, 30)},
    "commodity_precious": {"open": time(18, 0), "close": time(8, 20)},
    "commodity_metals": {"open": time(18, 0), "close": time(8, 20)},
    "commodity_energy": {"open": time(18, 0), "close": time(9, 0)},
    # Regular equity session
    "equity": {"open": time(9, 30), "close": time(16, 0)},
    # Near-24/5 FX (approximate; broker-dependent)
    "forex": {"open": time(17, 0), "close": time(17, 0)},
    # 24/7 crypto — no defined session close
    "crypto": {"open": None, "close": None},
}

# Asian session: 7 PM – 2 AM ET
ASIAN_SESSION = {"start": time(19, 0), "end": time(2, 0)}

# European session: 2 AM – 8 AM ET
EUROPEAN_SESSION = {"start": time(2, 0), "end": time(8, 0)}


# ---------------------------------------------------------------------------
# Tag inference helper (backward-compatibility shim)
# ---------------------------------------------------------------------------


def _infer_tags_from_name(name: str) -> list[str]:
    """Infer asset-class tags from a plain instrument name string.

    Used to maintain backward compatibility when callers pass a string
    ``asset_name`` rather than explicit ``asset_tags``.  The match is
    intentionally broad and case-insensitive.

    Args:
        name: Instrument name such as "Gold", "S&P", "Bitcoin", "EUR/USD".

    Returns:
        A non-empty list of tag strings.  Falls back to ``["equity"]`` when
        no keyword matches.
    """
    n = name.lower()

    # Equity index
    if any(
        kw in n
        for kw in (
            "s&p",
            "spx",
            " es",
            "/es",
            "e-mini s",
            "nq",
            "nasdaq",
            "dow",
            "ym ",
            "/ym",
            "rty",
            "russell",
            "dax",
            "ftse",
            "nikkei",
            "hang seng",
        )
    ):
        return ["equity_index"]

    # Precious metals
    if any(kw in n for kw in ("gold", "silver", "platinum", "palladium", "gc", " si ")):
        return ["commodity_precious"]

    # Energy
    if any(
        kw in n
        for kw in (
            "crude",
            "oil",
            "cl ",
            "/cl",
            "natural gas",
            "ng ",
            "/ng",
            "rbob",
            "gasoline",
            "heating oil",
            "brent",
        )
    ):
        return ["commodity_energy"]

    # Base / industrial metals
    if any(kw in n for kw in ("copper", "aluminum", "aluminium", "hg ", "/hg", "zinc", "nickel", "lead", "tin")):
        return ["commodity_metals"]

    # Crypto
    if any(
        kw in n
        for kw in (
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "crypto",
            "solana",
            "sol",
            "xrp",
            "ripple",
            "bnb",
            "doge",
            "litecoin",
            "ltc",
            "usdt",
            "usdc",
        )
    ):
        return ["crypto"]

    # Forex
    if any(
        kw in n
        for kw in ("eur", "usd", "gbp", "jpy", "chf", "aud", "cad", "nzd", "dxy", "forex", "fx ", "currency", "/")
    ):
        return ["forex"]

    # Rates / fixed income
    if any(
        kw in n
        for kw in ("treasury", "bond", "note", "t-bill", "zn", "zb", "eurodollar", "sofr", "libor", "gilt", "bund")
    ):
        return ["rates"]

    # Default fallback
    return ["equity"]


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------


def calc_natr_score(
    df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
    atr_period: int = 14,
    avg_lookback: int = 20,
) -> dict[str, Any]:
    """Compute Normalized ATR score (0-100).

    NATR = ATR_14 / close × 100, compared to its 20-day average.
    An instrument scoring 1.5× its norm gets a high volatility score.

    Returns:
        Dict with score (0-100), natr, natr_avg, ratio.
    """
    if df.empty or len(df) < atr_period + 5:
        return {"score": 0.0, "natr": 0.0, "natr_avg": 0.0, "ratio": 0.0}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    current_close = float(close.iloc[-1])
    current_atr = float(atr.iloc[-1])

    if current_close <= 0 or math.isnan(current_atr):
        return {"score": 0.0, "natr": 0.0, "natr_avg": 0.0, "ratio": 0.0}

    natr = current_atr / current_close * 100

    # Compare to historical average NATR.
    # Use daily data if available for a more stable average, otherwise use intraday.
    if daily_df is not None and len(daily_df) >= avg_lookback + atr_period:
        d_close = daily_df["Close"].astype(float)
        d_high = daily_df["High"].astype(float)
        d_low = daily_df["Low"].astype(float)
        d_tr1 = d_high - d_low
        d_tr2 = (d_high - d_close.shift(1)).abs()
        d_tr3 = (d_low - d_close.shift(1)).abs()
        d_tr = pd.concat([d_tr1, d_tr2, d_tr3], axis=1).max(axis=1)
        d_atr = d_tr.ewm(span=atr_period, adjust=False).mean()
        d_natr = d_atr / (d_close + 1e-10) * 100
        natr_avg = float(d_natr.iloc[-avg_lookback:].mean())
    else:
        # Fallback: use what we have
        natr_series = atr / (close + 1e-10) * 100
        lookback = min(avg_lookback * 78, len(natr_series))  # ~78 bars/day for 5-min
        natr_avg = float(natr_series.iloc[-lookback:].mean())

    if natr_avg <= 0:
        natr_avg = natr  # avoid division by zero

    ratio = natr / natr_avg

    # Score: 0 at ratio=0.5, 50 at ratio=1.0, 100 at ratio=2.0+
    # Linear interpolation clamped to [0, 100]
    score = _linear_scale(ratio, low_val=0.5, mid_val=1.0, high_val=2.0)

    return {
        "score": round(score, 1),
        "natr": round(natr, 4),
        "natr_avg": round(natr_avg, 4),
        "ratio": round(ratio, 2),
    }


def calc_rvol_score(
    df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
    avg_lookback: int = 20,
) -> dict[str, Any]:
    """Compute Relative Volume score (0-100).

    RVOL = current_volume / 20-day avg volume.
    Values above 1.5 indicate meaningful participation.

    Returns:
        Dict with score (0-100), current_vol, avg_vol, rvol.
    """
    if df.empty:
        return {"score": 0.0, "current_vol": 0, "avg_vol": 0, "rvol": 0.0}

    current_vol = float(df["Volume"].iloc[-1])

    if daily_df is not None and len(daily_df) >= avg_lookback:
        # Use daily volume average for a stable baseline
        daily_vols = daily_df["Volume"].astype(float).iloc[-avg_lookback:]
        avg_vol = float(daily_vols.mean())
    else:
        # Fallback: use rolling mean of intraday volume
        vol_series = df["Volume"].astype(float)
        lookback = min(avg_lookback * 78, len(vol_series))
        avg_vol = float(vol_series.iloc[-lookback:].mean())

    if avg_vol <= 0:
        return {
            "score": 0.0,
            "current_vol": int(current_vol),
            "avg_vol": 0,
            "rvol": 0.0,
        }

    rvol = current_vol / avg_vol

    # For intraday data, compare cumulative session volume to average.
    # This gives a more accurate RVOL than single-bar comparison.
    if daily_df is not None and len(daily_df) >= avg_lookback:
        try:
            idx = df.index.to_series()
            if hasattr(idx.dt, "date"):
                today = idx.dt.date.iloc[-1]
                today_mask = idx.dt.date == today
                today_vol = float(df.loc[today_mask, "Volume"].astype(float).sum())
                if today_vol > 0:
                    rvol = today_vol / avg_vol
        except Exception:
            pass

    # Score: 0 at rvol=0.3, 50 at rvol=1.0, 100 at rvol=2.5+
    score = _linear_scale(rvol, low_val=0.3, mid_val=1.0, high_val=2.5)

    return {
        "score": round(score, 1),
        "current_vol": int(current_vol),
        "avg_vol": int(avg_vol),
        "rvol": round(rvol, 2),
    }


def calc_gap_score(
    df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute overnight/pre-session gap magnitude score (0-100).

    Gap = |session_open - prior_close| / prior_close × 100.
    Larger gaps create tradeable scenarios.

    Returns:
        Dict with score (0-100), gap_pct, gap_direction.
    """
    if daily_df is not None and len(daily_df) >= 2:
        prior_close = float(daily_df["Close"].iloc[-2])
        current_open = float(daily_df["Open"].iloc[-1])
    elif not df.empty and len(df) >= 2:
        # Fallback: use first bar open vs last close of previous "day"
        prior_close = float(df["Close"].iloc[0])
        current_open = float(df["Open"].iloc[0])
    else:
        return {"score": 0.0, "gap_pct": 0.0, "gap_direction": "flat"}

    if prior_close <= 0:
        return {"score": 0.0, "gap_pct": 0.0, "gap_direction": "flat"}

    gap = current_open - prior_close
    gap_pct = abs(gap) / prior_close * 100

    direction = "up" if gap > 0 else "down" if gap < 0 else "flat"

    # Score: 0 at gap=0%, 50 at gap=0.3%, 100 at gap=1.0%+
    score = _linear_scale(gap_pct, low_val=0.0, mid_val=0.3, high_val=1.0)

    return {
        "score": round(score, 1),
        "gap_pct": round(gap_pct, 3),
        "gap_direction": direction,
        "gap_points": round(gap, 2),
    }


def calc_catalyst_score(
    asset_name: str | None = None,
    active_events: list[str] | None = None,
    *,
    asset_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Compute economic catalyst score (0-100) for an instrument.

    Uses the EVENT_CATALOG to match active events to the instrument's
    asset-class tags.  Score is the maximum impact of any matching event,
    with a small bonus when multiple events overlap.

    Tag resolution order (first wins):
      1. Explicit ``asset_tags`` keyword argument.
      2. Tags inferred from ``asset_name`` via :func:`_infer_tags_from_name`.
      3. Empty tag list → score of 0.

    Args:
        asset_name: Instrument name string (e.g. ``"Gold"``, ``"S&P"``).
                    Used for backward compatibility; ignored when
                    ``asset_tags`` is supplied.
        active_events: List of event names active today (from
                       :data:`EVENT_CATALOG`).  Pass ``None`` or ``[]``
                       to receive a zero score.
        asset_tags: Explicit list of asset-class tag strings for the
                    instrument (e.g. ``["equity_index"]``).  Takes
                    priority over ``asset_name`` when provided.

    Returns:
        Dict with keys ``score`` (0-100), ``matching_events`` (list),
        ``event_count`` (int).
    """
    if not active_events:
        return {"score": 0.0, "matching_events": [], "event_count": 0}

    # Resolve tags
    if asset_tags is not None:
        resolved_tags: list[str] = asset_tags
    elif asset_name is not None:
        resolved_tags = _infer_tags_from_name(asset_name)
    else:
        resolved_tags = []

    if not resolved_tags:
        return {"score": 0.0, "matching_events": [], "event_count": 0}

    resolved_set = set(resolved_tags)
    matching: list[str] = []
    max_impact = 0

    for event_name in active_events:
        if event_name not in EVENT_CATALOG:
            continue
        entry = EVENT_CATALOG[event_name]
        event_tags: list[str] = entry.get("asset_tags", [])  # type: ignore[assignment]
        entry_impact: int = entry.get("impact", 0)  # type: ignore[assignment]
        # Match when any of the instrument's tags overlap with the event's tags
        if resolved_set.intersection(event_tags):
            matching.append(event_name)
            max_impact = max(max_impact, entry_impact)

    # Small boost when multiple events affect this instrument
    score = float(max_impact)
    if len(matching) > 1:
        score = min(100.0, score + len(matching) * 5)

    return {
        "score": round(score, 1),
        "matching_events": matching,
        "event_count": len(matching),
    }


def calc_momentum_score(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
) -> dict[str, Any]:
    """Compute momentum score (0-100).

    Momentum = |close - EMA_20| / ATR_14, measuring displacement from
    equilibrium.  Higher values indicate strong directional moves.

    Returns:
        Dict with score (0-100), displacement, direction.
    """
    if df.empty or len(df) < max(ema_period, atr_period) + 5:
        return {"score": 0.0, "displacement": 0.0, "direction": "neutral"}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ema = close.ewm(span=ema_period, adjust=False).mean()

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    current_close = float(close.iloc[-1])
    current_ema = float(ema.iloc[-1])
    current_atr = float(atr.iloc[-1])

    if current_atr <= 0 or math.isnan(current_atr) or math.isnan(current_ema):
        return {"score": 0.0, "displacement": 0.0, "direction": "neutral"}

    displacement = abs(current_close - current_ema) / current_atr
    direction = "bullish" if current_close > current_ema else "bearish"

    # Score: 0 at displacement=0, 50 at displacement=1.0, 100 at displacement=3.0+
    score = _linear_scale(displacement, low_val=0.0, mid_val=1.0, high_val=3.0)

    return {
        "score": round(score, 1),
        "displacement": round(displacement, 2),
        "direction": direction,
        "ema_value": round(current_ema, 2),
    }


# ---------------------------------------------------------------------------
# Scaling utility
# ---------------------------------------------------------------------------


def _linear_scale(
    value: float,
    low_val: float = 0.0,
    mid_val: float = 1.0,
    high_val: float = 2.0,
) -> float:
    """Scale a value to 0-100 using a piecewise linear mapping.

    value <= low_val → 0
    value == mid_val → 50
    value >= high_val → 100
    Linear interpolation in between.
    """
    if math.isnan(value):
        return 0.0
    if value <= low_val:
        return 0.0
    if value >= high_val:
        return 100.0
    if value <= mid_val:
        # Scale from 0 to 50 over [low_val, mid_val]
        denom = mid_val - low_val
        if denom <= 0:
            return 50.0
        return (value - low_val) / denom * 50.0
    else:
        # Scale from 50 to 100 over [mid_val, high_val]
        denom = high_val - mid_val
        if denom <= 0:
            return 100.0
        return 50.0 + (value - mid_val) / denom * 50.0


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

# Traffic light thresholds
SIGNAL_THRESHOLDS = {
    "strong": 70,  # Green  — high priority, trade this
    "moderate": 45,  # Yellow — watchlist, secondary
    "weak": 0,  # Red    — skip today
}


def classify_signal(score: float) -> str:
    """Classify a composite score into a traffic-light signal.

    Returns "strong" (green), "moderate" (yellow), or "weak" (red).
    """
    if score >= SIGNAL_THRESHOLDS["strong"]:
        return "strong"
    elif score >= SIGNAL_THRESHOLDS["moderate"]:
        return "moderate"
    return "weak"


def signal_emoji(signal: str) -> str:
    """Return a traffic-light emoji for a signal classification."""
    return {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}.get(signal, "⚪")


def signal_color(signal: str) -> str:
    """Return a CSS color for a signal classification."""
    return {
        "strong": "#00D4AA",  # Teal/green — matches dashboard accent
        "moderate": "#FFD700",  # Gold
        "weak": "#FF6B6B",  # Coral red
    }.get(signal, "#888888")


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------


class PreMarketScorer:
    """Pre-session composite scorer for instrument selection.

    Computes a weighted composite score (0-100) for each instrument
    based on five metrics, then ranks them and provides focus
    recommendations.  Works with any asset class — equities, futures,
    crypto, forex, or commodities.

    Instruments are identified by name; asset-class tags are inferred
    automatically via :func:`_infer_tags_from_name` or may be supplied
    explicitly to :meth:`score_instrument`.

    Usage:
        scorer = PreMarketScorer()
        results = scorer.score_all(
            intraday_data={"Bitcoin": df_btc_5m, "EUR/USD": df_eurusd_5m, ...},
            daily_data={"Bitcoin": df_btc_daily, ...},
            active_events=["CPI", "FOMC"],
        )
        for r in results:
            print(f"{r['asset']}: {r['composite_score']:.1f}")
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize with optional custom weights.

        Args:
            weights: Dict mapping metric names to weights (must sum to 1.0).
                     Keys: "natr", "rvol", "gap", "catalyst", "momentum".
        """
        self.weights = weights or WEIGHTS.copy()
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning("Weights sum to %.3f, normalizing to 1.0", total)
            for k in self.weights:
                self.weights[k] /= total

    def score_instrument(
        self,
        asset_name: str,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame | None = None,
        active_events: list[str] | None = None,
        *,
        asset_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score a single instrument on all five metrics.

        Args:
            asset_name: Human-readable name for the instrument.
            intraday_df: Intraday OHLCV DataFrame.
            daily_df: Optional daily OHLCV DataFrame for stable baselines.
            active_events: List of economic event names active today.
            asset_tags: Explicit asset-class tags for catalyst matching.
                        When omitted, tags are inferred from ``asset_name``.

        Returns:
            Dict with individual metric scores, composite score,
            signal classification, and detailed metric breakdowns.
        """
        # Calculate each metric
        natr = calc_natr_score(intraday_df, daily_df)
        rvol = calc_rvol_score(intraday_df, daily_df)
        gap = calc_gap_score(intraday_df, daily_df)
        catalyst = calc_catalyst_score(
            asset_name=asset_name,
            active_events=active_events,
            asset_tags=asset_tags,
        )
        momentum = calc_momentum_score(intraday_df)

        # Weighted composite
        composite = (
            natr["score"] * self.weights["natr"]
            + rvol["score"] * self.weights["rvol"]
            + gap["score"] * self.weights["gap"]
            + catalyst["score"] * self.weights["catalyst"]
            + momentum["score"] * self.weights["momentum"]
        )

        signal = classify_signal(composite)

        return {
            "asset": asset_name,
            "composite_score": round(composite, 1),
            "signal": signal,
            "signal_emoji": signal_emoji(signal),
            "signal_color": signal_color(signal),
            # Individual metric scores (0-100)
            "natr_score": natr["score"],
            "rvol_score": rvol["score"],
            "gap_score": gap["score"],
            "catalyst_score": catalyst["score"],
            "momentum_score": momentum["score"],
            # Detailed breakdowns
            "natr_detail": natr,
            "rvol_detail": rvol,
            "gap_detail": gap,
            "catalyst_detail": catalyst,
            "momentum_detail": momentum,
        }

    def score_all(
        self,
        intraday_data: dict[str, pd.DataFrame],
        daily_data: dict[str, pd.DataFrame] | None = None,
        active_events: list[str] | None = None,
        asset_tags_map: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Score all instruments and return sorted results.

        Args:
            intraday_data: Dict of asset_name → intraday OHLCV DataFrame.
            daily_data: Optional dict of asset_name → daily OHLCV DataFrame.
            active_events: List of economic event names active today.
            asset_tags_map: Optional dict of asset_name → explicit tag list.
                            Tags for names absent from this map are inferred
                            automatically.

        Returns:
            List of score dicts, sorted by composite_score descending.
        """
        if daily_data is None:
            daily_data = {}
        if asset_tags_map is None:
            asset_tags_map = {}

        results = []
        for asset_name, df in intraday_data.items():
            daily_df = daily_data.get(asset_name)
            tags = asset_tags_map.get(asset_name)
            result = self.score_instrument(asset_name, df, daily_df, active_events, asset_tags=tags)
            results.append(result)

        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results

    def get_focus_assets(
        self,
        results: list[dict[str, Any]],
        max_focus: int = 3,
        min_score: float = 40.0,
    ) -> list[str]:
        """Return the top N assets to focus on today.

        Args:
            results: Scored results from score_all().
            max_focus: Maximum number of focus assets.
            min_score: Minimum composite score to be considered.

        Returns:
            List of asset names, ordered by score.
        """
        eligible = [r for r in results if r["composite_score"] >= min_score]
        return [r["asset"] for r in eligible[:max_focus]]


# ---------------------------------------------------------------------------
# Convenience function for direct use
# ---------------------------------------------------------------------------


def score_instruments(
    intraday_data: dict[str, pd.DataFrame],
    daily_data: dict[str, pd.DataFrame] | None = None,
    active_events: list[str] | None = None,
    weights: dict[str, float] | None = None,
    asset_tags_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Score all instruments and return sorted results.

    Convenience wrapper around :class:`PreMarketScorer` for one-shot use.

    Args:
        intraday_data: Dict of asset_name → intraday OHLCV DataFrame.
        daily_data: Optional dict of asset_name → daily OHLCV DataFrame.
        active_events: List of economic event names active today.
        weights: Optional custom metric weights (must sum to 1.0).
        asset_tags_map: Optional dict of asset_name → explicit tag list.

    Returns:
        List of score dicts, sorted by composite_score descending.
    """
    scorer = PreMarketScorer(weights=weights)
    return scorer.score_all(intraday_data, daily_data, active_events, asset_tags_map)


# ---------------------------------------------------------------------------
# DataFrame formatters for display
# ---------------------------------------------------------------------------


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert scorer results to a display-friendly DataFrame.

    Returns a DataFrame with columns suitable for ``st.dataframe()``
    with traffic-light color coding.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append(
            {
                "Signal": f"{r['signal_emoji']} {r['signal'].upper()}",
                "Asset": r["asset"],
                "Score": r["composite_score"],
                "NATR": r["natr_score"],
                "RVOL": r["rvol_score"],
                "Gap": r["gap_score"],
                "Catalyst": r["catalyst_score"],
                "Momentum": r["momentum_score"],
                "Events": ", ".join(r["catalyst_detail"]["matching_events"]) or "—",
                "Gap %": f"{r['gap_detail']['gap_pct']:.2f}%",
                "RVOL×": f"{r['rvol_detail']['rvol']:.1f}×",
                "Direction": r["momentum_detail"]["direction"],
            }
        )

    return pd.DataFrame(rows)


def results_to_summary(results: list[dict[str, Any]], max_focus: int = 3) -> str:
    """Generate a text summary of the scoring results.

    Suitable for inclusion in LLM prompts or display as markdown.
    """
    if not results:
        return "No instruments scored."

    lines = ["**Pre-Session Score Rankings:**\n"]
    for i, r in enumerate(results, 1):
        emoji = r["signal_emoji"]
        asset = r["asset"]
        score = r["composite_score"]
        signal = r["signal"].upper()

        detail_parts = []
        if r["catalyst_detail"]["matching_events"]:
            detail_parts.append(f"Events: {', '.join(r['catalyst_detail']['matching_events'])}")
        if r["gap_detail"]["gap_pct"] > 0.05:
            detail_parts.append(f"Gap: {r['gap_detail']['gap_direction']} {r['gap_detail']['gap_pct']:.2f}%")
        detail_parts.append(f"RVOL: {r['rvol_detail']['rvol']:.1f}×")
        detail_parts.append(f"Momentum: {r['momentum_detail']['direction']}")

        detail = " | ".join(detail_parts)
        focus = " ← **FOCUS**" if i <= max_focus and score >= 40 else ""
        lines.append(f"{i}. {emoji} **{asset}** — {score:.0f}/100 ({signal}){focus}")
        lines.append(f"   {detail}")

    return "\n".join(lines)
