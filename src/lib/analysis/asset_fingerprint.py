"""
Asset Fingerprint — Per-Asset Behavioral Profiling for CNN
=====================================================================
Profiles what makes each asset unique so the CNN can learn asset-specific
breakout behavior, and so the dashboard can display an "Asset DNA" radar
chart for each focused asset.

Per-asset fingerprint vector (computed daily, cacheable):

  - ``typical_daily_range_atr``      — how many ATR the asset typically moves
                                       in a day (Gold ~1.2, Nasdaq ~1.8, 6E ~0.7)
  - ``session_concentration``        — what fraction of the daily range happens
                                       in London vs US vs overnight
  - ``breakout_follow_through_rate`` — historically, what % of breakouts on
                                       this asset continue vs fade (per type)
  - ``mean_reversion_tendency``      — does this asset revert (choppy) or trend?
                                       Rolling Hurst exponent, normalised [0, 1]
  - ``volume_profile_shape``         — is volume U-shaped (equity open/close),
                                       L-shaped (London open), or flat (crypto)?
  - ``overnight_gap_tendency``       — how often does this asset gap overnight,
                                       and do gaps fill or continue?

These are NOT tabular features directly — they are used to:
  1. Create asset embedding training labels
  2. Detect when an asset is "acting like something else" (regime anomaly)
  3. Render the "Asset DNA" radar chart on the dashboard

The fingerprint analysis runs during off-hours and is persisted to
Redis (with a 24-hour TTL) so it doesn't impact live-session latency.

Public API::

    from lib.analysis.asset_fingerprint import (
        compute_asset_fingerprint,
        compute_all_fingerprints,
        AssetFingerprint,
        SessionConcentration,
        VolumeProfileShape,
    )

    fp = compute_asset_fingerprint("MGC", bars_daily, bars_1m)
    print(fp.typical_daily_range_atr)           # 1.18
    print(fp.session_concentration.london_pct)  # 0.42
    print(fp.mean_reversion_tendency)           # 0.38  (trending)
    print(fp.volume_profile_shape)              # VolumeProfileShape.U_SHAPED

    # Batch compute for all focused assets
    fingerprints = compute_all_fingerprints(bars_daily_by_ticker, bars_1m_by_ticker)
    for ticker, fp in fingerprints.items():
        print(ticker, fp.to_radar_dict())

Design:
  - Pure functions — no Redis, no side-effects, fully testable.
  - All inputs are pandas DataFrames; no external I/O.
  - Thread-safe: no shared mutable state.
  - Graceful degradation: returns neutral defaults when data is missing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.asset_fingerprint")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class VolumeProfileShape(StrEnum):
    """Classification of intraday volume distribution shape."""

    U_SHAPED = "U-shaped"
    """Equity-style: high volume at open and close, low midday."""

    L_SHAPED = "L-shaped"
    """London-open style: spike at open, gradual decline."""

    FLAT = "flat"
    """Crypto-style: relatively uniform volume across the 24-hour cycle."""

    FRONT_LOADED = "front-loaded"
    """Open spike then quick fade — typical of thin overnight sessions."""

    UNKNOWN = "unknown"
    """Insufficient data to classify."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SessionConcentration:
    """Fraction of daily range captured in each major session window.

    All values sum to ~1.0 (with floating-point tolerance).
    """

    overnight_pct: float = 0.0
    """18:00–03:00 ET (Globex overnight / Asian / Sydney)."""

    london_pct: float = 0.0
    """03:00–08:00 ET (London + Frankfurt)."""

    us_pct: float = 0.0
    """08:00–16:00 ET (US pre-market + RTH)."""

    settle_pct: float = 0.0
    """16:00–17:00 ET (settlement / post-RTH)."""

    def to_dict(self) -> dict[str, float]:
        return {
            "overnight_pct": round(self.overnight_pct, 4),
            "london_pct": round(self.london_pct, 4),
            "us_pct": round(self.us_pct, 4),
            "settle_pct": round(self.settle_pct, 4),
        }

    @property
    def dominant_session(self) -> str:
        """Return the session that captures the most range."""
        vals = {
            "overnight": self.overnight_pct,
            "london": self.london_pct,
            "us": self.us_pct,
            "settle": self.settle_pct,
        }
        return max(vals, key=vals.get)  # type: ignore[arg-type]


@dataclass
class BreakoutFollowThrough:
    """Historical breakout follow-through statistics for one asset.

    ``follow_through_rate`` is the fraction of breakouts that hit TP1
    before SL over the lookback period, aggregated across all types
    unless per-type rates are available.
    """

    follow_through_rate: float = 0.5
    """Overall rate [0, 1]: 1.0 = every breakout continues, 0 = every one fades."""

    sample_count: int = 0
    """Number of breakout events in the lookback window."""

    per_type: dict[str, float] = field(default_factory=dict)
    """Per-breakout-type follow-through rates (e.g. {"ORB": 0.62, "PDR": 0.55})."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "follow_through_rate": round(self.follow_through_rate, 4),
            "sample_count": self.sample_count,
            "per_type": {k: round(v, 4) for k, v in self.per_type.items()},
        }


@dataclass
class OvernightGapStats:
    """Statistics about overnight gap behavior for one asset."""

    gap_frequency: float = 0.0
    """Fraction of trading days that have a significant overnight gap."""

    avg_gap_atr_ratio: float = 0.0
    """Average gap size as a fraction of ATR."""

    gap_fill_rate: float = 0.0
    """Fraction of gaps that fill (retrace >50%) within the first session."""

    gap_continuation_rate: float = 0.0
    """Fraction of gaps that continue (price extends in gap direction)."""

    def to_dict(self) -> dict[str, float]:
        return {
            "gap_frequency": round(self.gap_frequency, 4),
            "avg_gap_atr_ratio": round(self.avg_gap_atr_ratio, 4),
            "gap_fill_rate": round(self.gap_fill_rate, 4),
            "gap_continuation_rate": round(self.gap_continuation_rate, 4),
        }


@dataclass
class AssetFingerprint:
    """Complete behavioral fingerprint for one asset.

    All scalar values are normalised to [0, 1] unless otherwise noted.
    """

    ticker: str = ""
    asset_name: str = ""

    # ── Core fingerprint dimensions ───────────────────────────────────────

    typical_daily_range_atr: float = 1.0
    """Median daily range / ATR(14).  Values >1.5 = very active, <0.8 = quiet."""

    session_concentration: SessionConcentration = field(default_factory=SessionConcentration)
    """Where in the 24-hour cycle does this asset move most?"""

    breakout_follow_through: BreakoutFollowThrough = field(default_factory=BreakoutFollowThrough)
    """Historical breakout continuation rate."""

    mean_reversion_tendency: float = 0.5
    """Rolling Hurst exponent, normalised to [0, 1].
    < 0.4 = mean-reverting (choppy), > 0.6 = trending (momentum).
    0.5 = random walk."""

    volume_profile_shape: VolumeProfileShape = VolumeProfileShape.UNKNOWN
    """Intraday volume distribution classification."""

    overnight_gap: OvernightGapStats = field(default_factory=OvernightGapStats)
    """Overnight gap behavior statistics."""

    # ── Metadata ──────────────────────────────────────────────────────────

    lookback_days: int = 0
    """Number of trading days used to compute the fingerprint."""

    computed_at: str = ""
    """ISO timestamp when the fingerprint was computed."""

    error: str = ""
    """Error message if computation failed partially."""

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation for Redis / API / persistence."""
        return {
            "ticker": self.ticker,
            "asset_name": self.asset_name,
            "typical_daily_range_atr": round(self.typical_daily_range_atr, 4),
            "session_concentration": self.session_concentration.to_dict(),
            "breakout_follow_through": self.breakout_follow_through.to_dict(),
            "mean_reversion_tendency": round(self.mean_reversion_tendency, 4),
            "volume_profile_shape": self.volume_profile_shape.value,
            "overnight_gap": self.overnight_gap.to_dict(),
            "lookback_days": self.lookback_days,
            "computed_at": self.computed_at,
            "error": self.error,
        }

    def to_radar_dict(self) -> dict[str, float]:
        """Flat dict of [0, 1] values for a radar/spider chart.

        All 6 dimensions normalised so they can be directly plotted.
        """
        # Normalise typical_daily_range_atr: clamp [0.5, 2.5] → [0, 1]
        range_norm = max(0.0, min(1.0, (self.typical_daily_range_atr - 0.5) / 2.0))

        # Session concentration: use the dominant session's fraction
        session_norm = max(
            self.session_concentration.overnight_pct,
            self.session_concentration.london_pct,
            self.session_concentration.us_pct,
            self.session_concentration.settle_pct,
        )

        return {
            "daily_range": round(range_norm, 3),
            "session_concentration": round(session_norm, 3),
            "follow_through": round(self.breakout_follow_through.follow_through_rate, 3),
            "trend_tendency": round(self.mean_reversion_tendency, 3),
            "volume_regularity": round(_volume_shape_to_score(self.volume_profile_shape), 3),
            "gap_tendency": round(self.overnight_gap.gap_frequency, 3),
        }


# ---------------------------------------------------------------------------
# Internal computation helpers
# ---------------------------------------------------------------------------


def _compute_atr(bars_daily: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR from daily bars.  Returns 0.0 if insufficient data."""
    if bars_daily is None or len(bars_daily) < period + 1:
        return 0.0
    try:
        high = bars_daily["High"].astype(float)
        low = bars_daily["Low"].astype(float)
        close = bars_daily["Close"].astype(float)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder smoothing
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        val = float(atr.iloc[-1])
        return val if not math.isnan(val) else 0.0
    except Exception:
        return 0.0


def _compute_typical_daily_range_atr(
    bars_daily: pd.DataFrame,
    lookback: int = 20,
) -> float:
    """Compute the median daily range / ATR(14) over *lookback* days.

    Returns 1.0 (neutral) if insufficient data.
    """
    if bars_daily is None or len(bars_daily) < lookback:
        return 1.0
    try:
        tail = bars_daily.tail(lookback)
        daily_ranges = (tail["High"].astype(float) - tail["Low"].astype(float)).values
        atr = _compute_atr(bars_daily, period=14)
        if atr <= 0:
            return 1.0
        ratios = daily_ranges / atr
        median = float(np.median(ratios[~np.isnan(ratios)]))
        return max(0.1, min(5.0, median)) if not math.isnan(median) else 1.0
    except Exception:
        return 1.0


def _compute_session_concentration(
    bars_1m: pd.DataFrame,
    lookback_days: int = 10,
) -> SessionConcentration:
    """Compute the fraction of daily range captured in each session window.

    Session boundaries (ET):
      - Overnight: 18:00–03:00
      - London:    03:00–08:00
      - US:        08:00–16:00
      - Settle:    16:00–17:00

    Uses 1-minute bars to measure the high-low range within each window,
    averaged over *lookback_days* trading days.
    """
    result = SessionConcentration()
    if bars_1m is None or bars_1m.empty:
        result.overnight_pct = 0.25
        result.london_pct = 0.25
        result.us_pct = 0.40
        result.settle_pct = 0.10
        return result

    try:
        from zoneinfo import ZoneInfo

        _ET = ZoneInfo("America/New_York")

        # Convert index to ET — work on a view, not a full copy of 50k rows.
        idx = bars_1m.index
        if idx.tz is None:  # type: ignore[union-attr]
            idx = idx.tz_localize("UTC")  # type: ignore[union-attr]
        idx_et = idx.tz_convert(_ET)  # type: ignore[union-attr]

        # Pre-slice to only the last (lookback_days) trading days worth of
        # bars before doing any column work — avoids copying the full 50k
        # bar dataframe just to discard 98% of it.
        dates_all = np.array(idx_et.date)  # type: ignore[union-attr]
        unique_dates = sorted(set(dates_all))
        if len(unique_dates) > lookback_days:
            cutoff_date = unique_dates[-lookback_days]
            mask = dates_all >= cutoff_date
            bars_slice = bars_1m.iloc[mask]
            idx_et = idx_et[mask]
        else:
            bars_slice = bars_1m

        high_col = "High" if "High" in bars_slice.columns else "high"
        low_col = "Low" if "Low" in bars_slice.columns else "low"

        # Build a minimal working DataFrame — only the columns we need.
        df = pd.DataFrame(
            {
                "high": bars_slice[high_col].values,
                "low": bars_slice[low_col].values,
                "hour": idx_et.hour,  # type: ignore[union-attr]
                "date": idx_et.date,  # type: ignore[union-attr]
            },
            index=idx_et,
        )

        # Assign session label vectorially — no per-date loop.
        hour = df["hour"]
        conditions = [
            (hour >= 18) | (hour < 3),  # overnight
            (hour >= 3) & (hour < 8),  # london
            (hour >= 8) & (hour < 16),  # us
            (hour >= 16) & (hour < 17),  # settle
        ]
        session_labels = ["overnight", "london", "us", "settle"]
        df["session"] = np.select(conditions, session_labels, default="other")

        # Per (date × session): compute high-low range, then average across dates.
        grp = df[df["session"] != "other"].groupby(["date", "session"])
        session_ranges = grp["high"].max() - grp["low"].min()  # type: ignore[operator]
        avg_by_session = session_ranges.groupby(level="session").mean()

        avg_overnight = float(avg_by_session.get("overnight", 0.0))
        avg_london = float(avg_by_session.get("london", 0.0))
        avg_us = float(avg_by_session.get("us", 0.0))
        avg_settle = float(avg_by_session.get("settle", 0.0))

        total = avg_overnight + avg_london + avg_us + avg_settle
        if total > 0:
            result.overnight_pct = avg_overnight / total
            result.london_pct = avg_london / total
            result.us_pct = avg_us / total
            result.settle_pct = avg_settle / total
        else:
            result.overnight_pct = 0.25
            result.london_pct = 0.25
            result.us_pct = 0.40
            result.settle_pct = 0.10

    except Exception as exc:
        logger.debug("Session concentration error: %s", exc)
        result.overnight_pct = 0.25
        result.london_pct = 0.25
        result.us_pct = 0.40
        result.settle_pct = 0.10

    return result


def _compute_hurst_exponent(
    prices: pd.Series,
    max_lag: int = 20,
) -> float:
    """Estimate the Hurst exponent using the rescaled range (R/S) method.

    Returns:
        Hurst exponent in [0, 1]:
          - H < 0.5 → mean-reverting / anti-persistent
          - H ≈ 0.5 → random walk
          - H > 0.5 → trending / persistent

    Falls back to 0.5 (random walk) if computation fails.
    """
    import warnings as _warnings

    if prices is None or len(prices) < max_lag * 2:
        return 0.5
    try:
        prices_arr = prices.dropna().values.astype(float)
        n = len(prices_arr)
        if n < max_lag * 2:
            return 0.5

        lags = range(2, max_lag + 1)
        rs_list = []

        # Suppress both numpy floating-point errors (np.errstate) AND Python
        # RuntimeWarnings (warnings.catch_warnings).  np.errstate only controls
        # the floating-point error *result* (e.g. returning NaN vs raising),
        # but numpy still emits Python-level RuntimeWarnings for "Degrees of
        # freedom <= 0" and "invalid value encountered" through the warnings
        # module.  Both must be silenced to avoid log spam during training.
        with _warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            for lag in lags:
                # Split into non-overlapping chunks of size lag
                chunks = n // lag
                if chunks < 1:
                    continue

                rs_values = []
                for i in range(chunks):
                    chunk = prices_arr[i * lag : (i + 1) * lag]
                    returns = np.diff(chunk)
                    if len(returns) < 2:
                        continue

                    mean_ret = np.mean(returns)
                    deviations = np.cumsum(returns - mean_ret)
                    r = np.max(deviations) - np.min(deviations)
                    s = np.std(returns, ddof=1)
                    if s > 1e-12:
                        rs_values.append(r / s)

                if rs_values:
                    rs_list.append((np.log(lag), np.log(np.mean(rs_values))))

        if len(rs_list) < 3:
            return 0.5

        log_lags = np.array([x[0] for x in rs_list])
        log_rs = np.array([x[1] for x in rs_list])

        # Linear regression: log(R/S) = H * log(lag) + c
        with _warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = float(coeffs[0])

        # Clamp to [0, 1]; NaN/inf from degenerate input → neutral 0.5
        if not np.isfinite(hurst):
            return 0.5
        return max(0.0, min(1.0, hurst))
    except Exception:
        return 0.5


def _classify_volume_profile(
    bars_1m: pd.DataFrame,
    lookback_days: int = 10,
) -> VolumeProfileShape:
    """Classify the intraday volume distribution shape.

    Examines 1-minute volume data across *lookback_days* to determine
    the volume shape:
      - U-shaped: high open + close, low midday (equity index)
      - L-shaped: high open, declining through day (London-centric)
      - Flat: uniform throughout (crypto 24/7)
      - Front-loaded: spike at open then quick fade
    """
    if bars_1m is None or bars_1m.empty:
        return VolumeProfileShape.UNKNOWN

    try:
        from zoneinfo import ZoneInfo

        _ET = ZoneInfo("America/New_York")

        vol_col = "Volume" if "Volume" in bars_1m.columns else "volume"
        if vol_col not in bars_1m.columns:
            return VolumeProfileShape.UNKNOWN

        # Convert index to ET without copying the full dataframe first.
        idx = bars_1m.index
        if idx.tz is None:  # type: ignore[union-attr]
            idx = idx.tz_localize("UTC")  # type: ignore[union-attr]
        idx_et = idx.tz_convert(_ET)  # type: ignore[union-attr]

        # Pre-slice to lookback_days before building any working frame.
        dates_all = np.array(idx_et.date)  # type: ignore[union-attr]
        unique_dates = sorted(set(dates_all))
        if len(unique_dates) > lookback_days:
            cutoff_date = unique_dates[-lookback_days]
            mask = dates_all >= cutoff_date
            vol_values = bars_1m[vol_col].values[mask]
            hours = idx_et.hour[mask]  # type: ignore[union-attr]
        else:
            vol_values = bars_1m[vol_col].values
            hours = idx_et.hour  # type: ignore[union-attr]

        # Build minimal frame — only volume and hour.
        df = pd.DataFrame({"vol": vol_values, "hour": hours})

        # Average volume per hour bucket
        hourly_vol = df.groupby("hour")["vol"].mean()
        if hourly_vol.empty or hourly_vol.sum() == 0:  # type: ignore[union-attr]
            return VolumeProfileShape.UNKNOWN

        # Normalise to [0, 1]
        vol_norm = hourly_vol / hourly_vol.max()  # type: ignore[operator]

        # Classify based on shape heuristics
        # RTH hours: 9–15 (09:00–15:59 ET)
        rth_hours = [h for h in range(9, 16) if h in vol_norm.index]
        eth_hours = [h for h in vol_norm.index if h not in range(9, 16)]

        if not rth_hours:
            return VolumeProfileShape.FLAT

        rth_vol = vol_norm.loc[rth_hours] if rth_hours else pd.Series(dtype=float)
        eth_vol = vol_norm.loc[eth_hours] if eth_hours else pd.Series(dtype=float)

        rth_mean = float(rth_vol.mean()) if not rth_vol.empty else 0
        eth_mean = float(eth_vol.mean()) if not eth_vol.empty else 0

        # Check for flatness (crypto-like)
        # Guard against single-element series where ddof=1 yields NaN/warning
        cv = float(vol_norm.std(ddof=0) / vol_norm.mean()) if (vol_norm.mean() > 0 and len(vol_norm) > 1) else 0
        if cv < 0.3:
            return VolumeProfileShape.FLAT

        # Check for U-shape: first and last RTH hours higher than middle
        if len(rth_hours) >= 4:
            first_hour_vol = float(vol_norm.get(rth_hours[0], 0))
            last_hour_vol = float(vol_norm.get(rth_hours[-1], 0))
            mid_hours = rth_hours[1:-1]
            mid_vol = float(vol_norm.loc[mid_hours].mean()) if mid_hours else 0

            if first_hour_vol > mid_vol * 1.3 and last_hour_vol > mid_vol * 1.2:
                return VolumeProfileShape.U_SHAPED

        # Check for L-shape: first hours high, monotonically declining
        if len(rth_hours) >= 3:
            first_vol = float(vol_norm.get(rth_hours[0], 0))
            second_vol = float(vol_norm.get(rth_hours[1], 0))
            last_third_mean = float(vol_norm.loc[rth_hours[-len(rth_hours) // 3 :]].mean())

            if first_vol > 0.7 and last_third_mean < first_vol * 0.5 and second_vol < first_vol:
                return VolumeProfileShape.L_SHAPED

        # Check for front-loaded: big spike at open, quick fade
        if len(rth_hours) >= 2:
            open_vol = float(vol_norm.get(rth_hours[0], 0))
            rest_mean = float(vol_norm.loc[rth_hours[1:]].mean())
            if open_vol > rest_mean * 2.0:
                return VolumeProfileShape.FRONT_LOADED

        # Default to U-shaped for RTH-dominant assets
        if rth_mean > eth_mean * 1.5:
            return VolumeProfileShape.U_SHAPED

        return VolumeProfileShape.FLAT

    except Exception as exc:
        logger.debug("Volume profile classification error: %s", exc)
        return VolumeProfileShape.UNKNOWN


def _compute_overnight_gap_stats(
    bars_daily: pd.DataFrame,
    lookback: int = 20,
    gap_threshold_atr_pct: float = 0.15,
) -> OvernightGapStats:
    """Compute overnight gap statistics from daily bars.

    A "gap" is defined as when today's Open is outside yesterday's
    High-Low range by at least *gap_threshold_atr_pct* × ATR.
    """
    result = OvernightGapStats()
    if bars_daily is None or len(bars_daily) < lookback + 1:
        return result

    try:
        tail = bars_daily.tail(lookback + 1)
        atr = _compute_atr(bars_daily, period=14)
        if atr <= 0:
            return result

        gap_threshold = gap_threshold_atr_pct * atr

        gap_count = 0
        gap_sizes = []
        fill_count = 0
        continuation_count = 0
        total_eligible = 0

        for i in range(1, len(tail)):
            prev_row = tail.iloc[i - 1]
            curr_row = tail.iloc[i]

            prev_high = float(prev_row["High"])
            prev_low = float(prev_row["Low"])
            float(prev_row["Close"])
            curr_open = float(curr_row["Open"])
            curr_high = float(curr_row["High"])
            curr_low = float(curr_row["Low"])
            curr_close = float(curr_row["Close"])

            total_eligible += 1

            # Gap up: open above prior high
            if curr_open > prev_high + gap_threshold:
                gap_size = curr_open - prev_high
                gap_count += 1
                gap_sizes.append(gap_size / atr)

                # Did the gap fill? (price came back below prior high)
                filled = curr_low <= prev_high + (gap_size * 0.5)
                if filled:
                    fill_count += 1

                # Did price continue higher?
                if curr_close > curr_open:
                    continuation_count += 1

            # Gap down: open below prior low
            elif curr_open < prev_low - gap_threshold:
                gap_size = prev_low - curr_open
                gap_count += 1
                gap_sizes.append(gap_size / atr)

                filled = curr_high >= prev_low - (gap_size * 0.5)
                if filled:
                    fill_count += 1

                if curr_close < curr_open:
                    continuation_count += 1

        if total_eligible > 0:
            result.gap_frequency = gap_count / total_eligible
        if gap_sizes:
            result.avg_gap_atr_ratio = float(np.mean(gap_sizes))
        if gap_count > 0:
            result.gap_fill_rate = fill_count / gap_count
            result.gap_continuation_rate = continuation_count / gap_count

    except Exception as exc:
        logger.debug("Overnight gap stats error: %s", exc)

    return result


def _volume_shape_to_score(shape: VolumeProfileShape) -> float:
    """Convert a volume shape classification to a [0, 1] regularity score.

    Higher score = more regular/predictable volume pattern.
    """
    return {
        VolumeProfileShape.U_SHAPED: 0.9,
        VolumeProfileShape.L_SHAPED: 0.7,
        VolumeProfileShape.FRONT_LOADED: 0.6,
        VolumeProfileShape.FLAT: 0.4,
        VolumeProfileShape.UNKNOWN: 0.5,
    }.get(shape, 0.5)


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def compute_asset_fingerprint(
    ticker: str,
    bars_daily: pd.DataFrame | None = None,
    bars_1m: pd.DataFrame | None = None,
    *,
    asset_name: str = "",
    lookback_days: int = 20,
) -> AssetFingerprint:
    """Compute a full behavioral fingerprint for one asset.

    All internal numpy/pandas computations suppress ``RuntimeWarning``
    to avoid flooding Docker logs during dataset generation.  The warnings
    are harmless — degenerate slices (e.g. ddof=1 on single-element arrays)
    produce NaN which the code already handles with neutral-default fallbacks.

    Args:
        ticker: Instrument ticker (e.g. ``"MGC"``, ``"MES"``, ``"6E"``).
        bars_daily: Daily OHLCV bars (≥ *lookback_days* + 14 rows ideal).
        bars_1m: 1-minute OHLCV bars (≥ *lookback_days* trading days ideal).
        asset_name: Human-readable name (optional, for display).
        lookback_days: Number of trading days to analyze.

    Returns:
        ``AssetFingerprint`` with all dimensions computed.  Missing data
        produces neutral defaults rather than errors.
    """
    import warnings as _warnings
    from datetime import datetime
    from zoneinfo import ZoneInfo

    fp = AssetFingerprint(
        ticker=ticker,
        asset_name=asset_name or ticker,
        computed_at=datetime.now(tz=ZoneInfo("America/New_York")).isoformat(),
    )

    errors: list[str] = []

    # Pre-slice bars_1m to only the rows needed for the lookback window.
    # Each sub-function only uses the last (lookback_days) trading days,
    # which is at most lookback_days * 1440 1-minute bars.  Passing the
    # full 50 000-bar series into functions that do df.copy() or per-date
    # groupby scans is the primary cause of the per-symbol hang.
    bars_1m_sliced: pd.DataFrame | None = bars_1m
    if bars_1m is not None and not bars_1m.empty:
        _max_1m_bars = lookback_days * 1440
        if len(bars_1m) > _max_1m_bars:
            bars_1m_sliced = bars_1m.iloc[-_max_1m_bars:]

    # Suppress numpy RuntimeWarnings for the entire fingerprint computation.
    # Individual sub-functions may encounter degenerate data (single-element
    # slices, zero-variance chunks) that triggers "Degrees of freedom <= 0"
    # and "invalid value encountered" warnings.  These are harmless — the
    # code handles NaN/inf with neutral defaults — but they produce thousands
    # of lines of log noise during dataset generation.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)

        # 1. Typical daily range / ATR
        try:
            fp.typical_daily_range_atr = _compute_typical_daily_range_atr(bars_daily, lookback_days)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"daily_range: {exc}")

        # 2. Session concentration
        try:
            fp.session_concentration = _compute_session_concentration(bars_1m_sliced, lookback_days)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"session_conc: {exc}")

        # 3. Mean reversion tendency (Hurst exponent)
        try:
            if bars_daily is not None and not bars_daily.empty:
                close_prices = bars_daily["Close"].astype(float)
                fp.mean_reversion_tendency = _compute_hurst_exponent(close_prices)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"hurst: {exc}")

        # 4. Volume profile shape
        try:
            fp.volume_profile_shape = _classify_volume_profile(bars_1m_sliced, lookback_days)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"vol_profile: {exc}")

        # 5. Overnight gap statistics
        try:
            fp.overnight_gap = _compute_overnight_gap_stats(bars_daily, lookback_days)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(f"gap_stats: {exc}")

    # 6. Breakout follow-through (requires historical breakout events —
    #    set to default here; callers with DB access can enrich afterward)
    fp.breakout_follow_through = BreakoutFollowThrough()

    # Count lookback days
    if bars_daily is not None and not bars_daily.empty:
        fp.lookback_days = min(lookback_days, len(bars_daily))
    elif bars_1m is not None and not bars_1m.empty:
        try:
            fp.lookback_days = bars_1m.index.normalize().nunique()  # type: ignore[union-attr]
        except Exception:
            fp.lookback_days = 0

    if errors:
        fp.error = "; ".join(errors)

    return fp


def compute_all_fingerprints(
    bars_daily_by_ticker: dict[str, pd.DataFrame],
    bars_1m_by_ticker: dict[str, pd.DataFrame] | None = None,
    *,
    asset_names: dict[str, str] | None = None,
    lookback_days: int = 20,
) -> dict[str, AssetFingerprint]:
    """Batch-compute fingerprints for all provided tickers.

    Args:
        bars_daily_by_ticker: Dict mapping ticker → daily OHLCV DataFrame.
        bars_1m_by_ticker: Dict mapping ticker → 1-minute OHLCV DataFrame.
                           Optional — session concentration and volume profile
                           will use defaults if missing.
        asset_names: Dict mapping ticker → human-readable name.
        lookback_days: Number of trading days to analyze.

    Returns:
        Dict mapping ticker → ``AssetFingerprint``.
    """
    if bars_1m_by_ticker is None:
        bars_1m_by_ticker = {}
    if asset_names is None:
        asset_names = {}

    fingerprints: dict[str, AssetFingerprint] = {}

    for ticker, bars_daily in bars_daily_by_ticker.items():
        try:
            fp = compute_asset_fingerprint(
                ticker,
                bars_daily=bars_daily,
                bars_1m=bars_1m_by_ticker.get(ticker),
                asset_name=asset_names.get(ticker, ticker),
                lookback_days=lookback_days,
            )
            fingerprints[ticker] = fp
        except Exception as exc:
            logger.warning("Fingerprint computation failed for %s: %s", ticker, exc)
            fingerprints[ticker] = AssetFingerprint(
                ticker=ticker,
                asset_name=asset_names.get(ticker, ticker),
                error=str(exc),
            )

    return fingerprints


def enrich_follow_through_from_db(
    fingerprint: AssetFingerprint,
    breakout_events: list[dict[str, Any]] | None = None,
) -> AssetFingerprint:
    """Enrich a fingerprint's breakout follow-through stats from historical DB data.

    Args:
        fingerprint: An existing ``AssetFingerprint`` to enrich.
        breakout_events: List of dicts from ``models.get_orb_events()`` or
                         similar, each with at least ``breakout_type``,
                         ``breakout_detected``, and an ``outcome`` field
                         (``"win"`` / ``"loss"`` / ``"timeout"``).

    Returns:
        The same fingerprint, mutated with updated follow-through stats.
    """
    if not breakout_events:
        return fingerprint

    try:
        total = 0
        wins = 0
        per_type_wins: dict[str, int] = {}
        per_type_total: dict[str, int] = {}

        for event in breakout_events:
            if not event.get("breakout_detected", False):
                continue
            total += 1
            bt = str(event.get("breakout_type", "ORB"))
            outcome = str(event.get("outcome", ""))

            per_type_total[bt] = per_type_total.get(bt, 0) + 1

            if outcome in ("win", "good_long", "good_short"):
                wins += 1
                per_type_wins[bt] = per_type_wins.get(bt, 0) + 1

        ft = BreakoutFollowThrough(
            follow_through_rate=wins / total if total > 0 else 0.5,
            sample_count=total,
            per_type={bt: per_type_wins.get(bt, 0) / cnt for bt, cnt in per_type_total.items() if cnt > 0},
        )
        fingerprint.breakout_follow_through = ft

    except Exception as exc:
        logger.debug("Follow-through enrichment error: %s", exc)

    return fingerprint
