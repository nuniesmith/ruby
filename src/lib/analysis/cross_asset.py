"""
Cross-Asset Correlation Features
============================================
Computes real-time cross-asset correlation signals for the CNN tabular
feature vector.  These features let the model see regime shifts that are
invisible from any single asset's price action alone.

For each breakout signal, we compute rolling correlations with related
assets and derive three CNN-ready features:

  - ``primary_peer_corr``   — correlation with the most-related peer asset
                              (Gold↔Silver, S&P↔Nasdaq, etc.), [-1, 1] → [0, 1]
  - ``cross_class_corr``    — correlation with the strongest cross-class mover
                              (e.g. Gold↔S&P divergence = risk-off), [-1, 1] → [0, 1]
  - ``correlation_regime``  — is the correlation structure normal (0.5),
                              elevated (1.0), or broken/inverted (0.0)?
                              Detected by comparing current 30-bar corr to
                              200-bar baseline.

Peer asset mapping is defined in ``PEER_MAP`` below, informed by the
``asset_registry.py`` asset class structure.

Public API::

    from lib.analysis.cross_asset import (
        compute_cross_asset_features,
        compute_correlation_matrix,
        detect_correlation_anomalies,
        CrossAssetFeatures,
        CorrelationAnomaly,
    )

    feats = compute_cross_asset_features("MGC", bars_by_ticker)
    print(feats.primary_peer_corr)     # 0.85
    print(feats.cross_class_corr)      # 0.32
    print(feats.correlation_regime)    # 0.5 (normal)

    # Full matrix + anomaly detection
    matrix = compute_correlation_matrix(bars_by_ticker, window=30)
    anomalies = detect_correlation_anomalies(bars_by_ticker)
    for a in anomalies:
        print(a.pair, a.current_corr, a.baseline_corr, a.z_score)

Design:
  - Pure functions — no Redis, no side-effects, fully testable.
  - All correlations use Pearson on log-returns (not raw prices).
  - Thread-safe: no shared mutable state.
  - Graceful degradation: returns neutral defaults (0.5) when data is
    missing or insufficient.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.cross_asset")


# ---------------------------------------------------------------------------
# Peer asset mapping
# ---------------------------------------------------------------------------
# For each ticker, define:
#   - primary_peer: the single most-correlated same-class asset
#   - cross_class_peers: 1–3 assets from other classes that reveal regime
#
# These are the 10 core micro/mini tickers tracked by the engine.

# Active training symbols: MGC, SIL, MES, MNQ, M2K, MYM, ZN, ZB, ZW
# All peer references are restricted to this set so the dataset generator
# never pre-loads dropped symbols (MCL, MNG, MHG, MBT, MET, BTC, ETH, SOL,
# 6E, 6B, 6J, 6A, 6C, 6S, M6E, M6B, ZC, ZS) during training.
PEER_MAP: dict[str, dict[str, Any]] = {
    # ── Metals ────────────────────────────────────────────────────────────
    "MGC": {
        "asset_name": "Gold",
        "primary_peer": "SIL",  # Gold↔Silver: tightest metals correlation
        "cross_class_peers": ["MES", "ZN"],  # Gold↔S&P (risk-off), Gold↔T-Note (safe haven)
    },
    "SIL": {
        "asset_name": "Silver",
        "primary_peer": "MGC",  # Silver↔Gold: primary metals peer
        "cross_class_peers": ["MES", "MNQ"],  # Silver↔S&P, Silver↔Nasdaq (risk appetite)
    },
    # ── Equity Index ──────────────────────────────────────────────────────
    "MES": {
        "asset_name": "S&P 500",
        "primary_peer": "MNQ",  # S&P↔Nasdaq: highest index correlation
        "cross_class_peers": ["MGC", "ZN"],  # S&P↔Gold (risk-off hedge), S&P↔T-Note (macro)
    },
    "MNQ": {
        "asset_name": "Nasdaq",
        "primary_peer": "MES",  # Nasdaq↔S&P: primary index peer
        "cross_class_peers": ["MGC", "ZN"],  # NQ↔Gold, NQ↔T-Note (rate sensitivity)
    },
    "M2K": {
        "asset_name": "Russell 2000",
        "primary_peer": "MES",  # Russell↔S&P: broad market correlation
        "cross_class_peers": ["MNQ", "ZN"],  # Russell↔Nasdaq, Russell↔T-Note
    },
    "MYM": {
        "asset_name": "Dow Jones",
        "primary_peer": "MES",  # Dow↔S&P: near-identical index correlation
        "cross_class_peers": ["MNQ", "MGC"],  # Dow↔Nasdaq, Dow↔Gold
    },
    # ── Interest Rate ─────────────────────────────────────────────────────
    "ZN": {
        "asset_name": "10Y T-Note",
        "primary_peer": "ZB",  # 10Y↔30Y: duration curve correlation
        "cross_class_peers": ["MGC", "MES"],  # T-Note↔Gold (safe haven), T-Note↔S&P (inverse)
    },
    "ZB": {
        "asset_name": "30Y T-Bond",
        "primary_peer": "ZN",  # 30Y↔10Y: primary rate peer
        "cross_class_peers": ["MGC", "MES"],  # T-Bond↔Gold, T-Bond↔S&P
    },
    # ── Agricultural ──────────────────────────────────────────────────────
    "ZW": {
        "asset_name": "Wheat",
        "primary_peer": "MGC",  # Wheat↔Gold: commodity complex (inflation/USD)
        "cross_class_peers": ["MES", "ZN"],  # Wheat↔S&P (risk appetite), Wheat↔T-Note (macro)
    },
}

# Known correlation pairs and their expected baseline correlation
# (approximate, based on historical analysis).  Used for anomaly detection.
# Restricted to active training symbols only: MGC, SIL, MES, MNQ, M2K, MYM, ZN, ZB, ZW
BASELINE_CORRELATIONS: dict[tuple[str, str], float] = {
    ("MGC", "SIL"): 0.75,  # Gold↔Silver: strongly correlated
    ("MGC", "MES"): 0.0,  # Gold↔S&P: near-zero (risk switch)
    ("MGC", "ZN"): 0.45,  # Gold↔T-Note: moderate (safe-haven flow)
    ("MES", "MNQ"): 0.92,  # S&P↔Nasdaq: very strongly correlated
    ("MES", "M2K"): 0.85,  # S&P↔Russell: strongly correlated
    ("MES", "MYM"): 0.97,  # S&P↔Dow: near-identical
    ("MES", "ZN"): -0.40,  # S&P↔T-Note: moderate inverse (risk-on/off)
    ("MNQ", "M2K"): 0.80,  # Nasdaq↔Russell: strong (broad risk-on)
    ("MNQ", "ZN"): -0.45,  # Nasdaq↔T-Note: rate sensitivity
    ("ZN", "ZB"): 0.92,  # 10Y↔30Y T-Bond: very strongly correlated
    ("ZN", "MES"): -0.40,  # T-Note↔S&P: (duplicate for lookup symmetry)
    ("SIL", "MES"): 0.35,  # Silver↔S&P: weak-moderate (risk appetite)
    ("ZW", "MGC"): 0.30,  # Wheat↔Gold: commodity/inflation complex
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CrossAssetFeatures:
    """CNN-ready cross-asset correlation features for a single signal.

    All values are normalised to [0, 1] for direct insertion into the
    tabular feature vector.
    """

    primary_peer_corr: float = 0.5
    """Correlation with the primary peer asset, [-1, 1] → [0, 1]."""

    cross_class_corr: float = 0.5
    """Strongest cross-class correlation magnitude, [-1, 1] → [0, 1]."""

    correlation_regime: float = 0.5
    """Correlation regime: 0.0 = broken/inverted, 0.5 = normal, 1.0 = elevated."""

    # Raw values for debugging / dashboard
    primary_peer_ticker: str = ""
    primary_peer_raw_corr: float = 0.0
    cross_class_ticker: str = ""
    cross_class_raw_corr: float = 0.0
    regime_z_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_peer_corr": round(self.primary_peer_corr, 4),
            "cross_class_corr": round(self.cross_class_corr, 4),
            "correlation_regime": round(self.correlation_regime, 4),
            "primary_peer_ticker": self.primary_peer_ticker,
            "primary_peer_raw_corr": round(self.primary_peer_raw_corr, 4),
            "cross_class_ticker": self.cross_class_ticker,
            "cross_class_raw_corr": round(self.cross_class_raw_corr, 4),
            "regime_z_score": round(self.regime_z_score, 4),
        }


@dataclass
class CorrelationAnomaly:
    """A detected anomaly in the cross-asset correlation structure.

    Anomalies occur when a correlation pair deviates by >2σ from its
    historical baseline, signalling a potential regime shift.
    """

    ticker_a: str
    ticker_b: str
    current_corr: float
    baseline_corr: float
    z_score: float
    regime_label: str = ""  # human-readable regime description
    severity: str = "moderate"  # "moderate" (2σ) or "extreme" (3σ)

    @property
    def pair(self) -> str:
        return f"{self.ticker_a}↔{self.ticker_b}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "current_corr": round(self.current_corr, 4),
            "baseline_corr": round(self.baseline_corr, 4),
            "z_score": round(self.z_score, 4),
            "regime_label": self.regime_label,
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series, dropping NaN/inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices / prices.shift(1))
    return lr.replace([np.inf, -np.inf], np.nan).dropna()


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation between two aligned series.

    Returns 0.0 if insufficient overlapping data.
    """
    import warnings as _warnings

    # Align on index
    combined = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(combined) < 5:
        return 0.0
    a = combined.iloc[:, 0].values
    b = combined.iloc[:, 1].values
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(corr) or math.isinf(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))


def _rolling_corr(
    x: pd.Series,
    y: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Rolling Pearson correlation between two series.

    Returns a Series of the same length as the shorter input, with NaNs
    where insufficient data exists.
    """
    combined = pd.concat(
        [x.rename("x"), y.rename("y")],
        axis=1,
        join="inner",
    ).dropna()
    if len(combined) < window:
        return pd.Series(dtype=float)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        return combined["x"].rolling(window).corr(combined["y"])  # type: ignore[return-value]


def _normalise_corr(raw: float) -> float:
    """Map a raw correlation [-1, 1] to [0, 1] for CNN consumption."""
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _extract_close(bars: pd.DataFrame) -> pd.Series:
    """Extract a Close price series from a bars DataFrame.

    Handles both 'Close' and 'close' column names, and ensures the index
    is a DatetimeIndex.
    """
    if "Close" in bars.columns:
        s = bars["Close"]
    elif "close" in bars.columns:
        s = bars["close"]
    else:
        raise ValueError(f"No 'Close' column in bars (columns: {list(bars.columns)})")
    return s.astype(float)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def compute_cross_asset_features(
    ticker: str,
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    short_window: int = 30,
    long_window: int = 200,
) -> CrossAssetFeatures:
    """Compute CNN-ready cross-asset correlation features for *ticker*.

    All internal numpy/pandas computations are wrapped in
    ``warnings.catch_warnings()`` to suppress ``RuntimeWarning`` from
    degenerate slices (ddof=1 on single-element arrays, NaN divides).
    These warnings are harmless — the code already handles NaN/inf
    fallbacks — but they flood Docker logs during dataset generation.

    Args:
        ticker: The ticker of the asset that generated the breakout signal.
        bars_by_ticker: Dict mapping ticker → 1-minute (or any frequency)
                        OHLCV DataFrame.  Must include *ticker* and ideally
                        its peer assets.
        short_window: Rolling window for current correlation (default 30 bars).
        long_window: Rolling window for baseline correlation (default 200 bars).

    Returns:
        ``CrossAssetFeatures`` with all values normalised to [0, 1].
        Falls back to neutral (0.5) defaults if data is insufficient.
    """
    result = CrossAssetFeatures()

    # Look up peer mapping
    peer_info = PEER_MAP.get(ticker)
    if peer_info is None:
        # Unknown ticker — try stripping suffix or prefix
        stripped = ticker.split("=")[0] if "=" in ticker else ticker
        peer_info = PEER_MAP.get(stripped)
    if peer_info is None:
        logger.debug("No peer mapping for ticker %s — returning neutral features", ticker)
        return result

    # Get the signal asset's bars
    signal_bars = bars_by_ticker.get(ticker)
    if signal_bars is None or signal_bars.empty:
        return result

    try:
        signal_close = _extract_close(signal_bars)
        signal_returns = _log_returns(signal_close)
    except Exception as exc:
        logger.debug("Cannot compute returns for %s: %s", ticker, exc)
        return result

    if len(signal_returns) < short_window:
        return result

    # ── Primary peer correlation ──────────────────────────────────────────
    primary_peer = peer_info.get("primary_peer", "")
    if primary_peer:
        peer_bars = bars_by_ticker.get(primary_peer)
        if peer_bars is not None and not peer_bars.empty:
            try:
                peer_returns = _log_returns(_extract_close(peer_bars))
                raw_corr = _pearson_corr(
                    signal_returns.tail(short_window),
                    peer_returns.tail(short_window),
                )
                result.primary_peer_ticker = primary_peer
                result.primary_peer_raw_corr = raw_corr
                result.primary_peer_corr = _normalise_corr(raw_corr)
            except Exception as exc:
                logger.debug("Primary peer corr %s↔%s error: %s", ticker, primary_peer, exc)

    # ── Cross-class correlation (strongest magnitude) ─────────────────────
    cross_peers = peer_info.get("cross_class_peers", [])
    best_cross_corr = 0.0
    best_cross_ticker = ""
    for cross_ticker in cross_peers:
        cross_bars = bars_by_ticker.get(cross_ticker)
        if cross_bars is None or cross_bars.empty:
            continue
        try:
            cross_returns = _log_returns(_extract_close(cross_bars))
            raw_corr = _pearson_corr(
                signal_returns.tail(short_window),
                cross_returns.tail(short_window),
            )
            if abs(raw_corr) > abs(best_cross_corr):
                best_cross_corr = raw_corr
                best_cross_ticker = cross_ticker
        except Exception:
            continue

    if best_cross_ticker:
        result.cross_class_ticker = best_cross_ticker
        result.cross_class_raw_corr = best_cross_corr
        result.cross_class_corr = _normalise_corr(best_cross_corr)

    # ── Correlation regime ────────────────────────────────────────────────
    # Compare the short-window primary peer correlation to the long-window
    # baseline.  If they diverge significantly, the correlation structure
    # has shifted → regime change.
    if primary_peer and len(signal_returns) >= long_window:
        peer_bars_long = bars_by_ticker.get(primary_peer)
        if peer_bars_long is not None and not peer_bars_long.empty:
            try:
                import warnings as _warnings

                peer_returns_long = _log_returns(_extract_close(peer_bars_long))
                rolling = _rolling_corr(signal_returns, peer_returns_long, window=short_window)
                rolling_clean = rolling.dropna()
                if len(rolling_clean) >= long_window:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore", category=RuntimeWarning)
                        baseline_mean = float(rolling_clean.iloc[-long_window:].mean())
                        baseline_std = float(rolling_clean.iloc[-long_window:].std())
                    current = float(rolling_clean.iloc[-1])

                    z = (current - baseline_mean) / baseline_std if baseline_std > 1e-06 else 0.0

                    result.regime_z_score = z

                    # Map z-score to regime:
                    #   z < -2  → broken/inverted → 0.0
                    #   -2 ≤ z ≤ 2 → normal → 0.5
                    #   z > 2   → elevated → 1.0
                    if z > 2.0:
                        result.correlation_regime = 1.0
                    elif z < -2.0:
                        result.correlation_regime = 0.0
                    else:
                        # Smooth mapping: scale z ∈ [-2, 2] to [0.25, 0.75]
                        result.correlation_regime = 0.5 + (z / 8.0)  # ±0.25 around 0.5
                        result.correlation_regime = max(0.0, min(1.0, result.correlation_regime))
            except Exception as exc:
                logger.debug("Correlation regime computation error: %s", exc)

    return result


def compute_correlation_matrix(
    bars_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str] | None = None,
    window: int = 30,
) -> pd.DataFrame:
    """Compute a rolling correlation matrix across all provided tickers.

    Args:
        bars_by_ticker: Dict mapping ticker → OHLCV DataFrame.
        tickers: Specific tickers to include.  Defaults to all keys in
                 *bars_by_ticker*.
        window: Rolling window for correlation (default 30 bars).

    Returns:
        Square DataFrame with tickers as both index and columns.
        Values are Pearson correlations of log-returns over *window* bars.
        Missing pairs are filled with NaN.
    """
    if tickers is None:
        tickers = sorted(bars_by_ticker.keys())

    # Build a returns DataFrame
    returns_dict: dict[str, pd.Series] = {}
    for t in tickers:
        bars = bars_by_ticker.get(t)
        if bars is None or bars.empty:
            continue
        try:
            returns_dict[t] = _log_returns(_extract_close(bars))
        except Exception:
            continue

    if len(returns_dict) < 2:
        return pd.DataFrame(index=tickers, columns=tickers, dtype=float)  # type: ignore[call-overload]

    returns_df = pd.DataFrame(returns_dict)

    # Use the tail *window* rows for correlation
    tail = returns_df.tail(window)
    if len(tail) < 5:
        return pd.DataFrame(index=tickers, columns=tickers, dtype=float)  # type: ignore[call-overload]

    corr_matrix = tail.corr(method="pearson")

    # Ensure all requested tickers are present (fill missing with NaN)
    corr_matrix = corr_matrix.reindex(index=tickers, columns=tickers)

    return corr_matrix


def detect_correlation_anomalies(
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    short_window: int = 30,
    long_window: int = 200,
    z_threshold: float = 2.0,
    pairs: list[tuple[str, str]] | None = None,
) -> list[CorrelationAnomaly]:
    """Detect correlation pairs that have deviated significantly from baseline.

    Compares the *short_window* rolling correlation to the *long_window*
    baseline for each monitored pair.  Pairs where the current correlation
    deviates by more than *z_threshold* standard deviations are flagged.

    Args:
        bars_by_ticker: Dict mapping ticker → OHLCV DataFrame.
        short_window: Current correlation window (default 30 bars).
        long_window: Baseline correlation window (default 200 bars).
        z_threshold: Z-score threshold for anomaly detection (default 2.0).
        pairs: Specific pairs to monitor.  Defaults to all pairs in
               ``BASELINE_CORRELATIONS``.

    Returns:
        List of ``CorrelationAnomaly`` objects, sorted by absolute z-score
        (most extreme first).
    """
    if pairs is None:
        pairs = list(BASELINE_CORRELATIONS.keys())

    anomalies: list[CorrelationAnomaly] = []

    for ticker_a, ticker_b in pairs:
        bars_a = bars_by_ticker.get(ticker_a)
        bars_b = bars_by_ticker.get(ticker_b)
        if bars_a is None or bars_a.empty or bars_b is None or bars_b.empty:
            continue

        try:
            returns_a = _log_returns(_extract_close(bars_a))
            returns_b = _log_returns(_extract_close(bars_b))

            rolling = _rolling_corr(returns_a, returns_b, window=short_window)
            rolling_clean = rolling.dropna()

            if len(rolling_clean) < long_window:
                continue

            baseline_mean = float(rolling_clean.iloc[-long_window:].mean())
            baseline_std = float(rolling_clean.iloc[-long_window:].std())
            current = float(rolling_clean.iloc[-1])

            if baseline_std < 1e-6:
                continue

            z = (current - baseline_mean) / baseline_std

            if abs(z) >= z_threshold:
                # Determine regime label
                expected = BASELINE_CORRELATIONS.get((ticker_a, ticker_b), baseline_mean)
                label = _infer_regime_label(ticker_a, ticker_b, current, expected)
                severity = "extreme" if abs(z) >= 3.0 else "moderate"

                anomalies.append(
                    CorrelationAnomaly(
                        ticker_a=ticker_a,
                        ticker_b=ticker_b,
                        current_corr=current,
                        baseline_corr=baseline_mean,
                        z_score=z,
                        regime_label=label,
                        severity=severity,
                    )
                )
        except Exception as exc:
            logger.debug("Anomaly detection error for %s↔%s: %s", ticker_a, ticker_b, exc)

    # Sort by absolute z-score (most extreme first)
    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
    return anomalies


def _infer_regime_label(
    ticker_a: str,
    ticker_b: str,
    current_corr: float,
    expected_corr: float,
) -> str:
    """Infer a human-readable regime label from a correlation anomaly.

    These labels capture the economic meaning of correlation shifts:
      - Gold↔S&P suddenly positive → "flight to safety"
      - S&P↔BTC suddenly very positive → "risk-on euphoria"
      - Crude↔S&P suddenly negative → "energy divergence"
      - Gold↔Silver decorrelating → "precious metals divergence"
    """
    pair = frozenset([ticker_a, ticker_b])
    shift = current_corr - expected_corr

    # Gold ↔ S&P
    if pair == frozenset(["MGC", "MES"]):
        if current_corr > 0.5:
            return "flight to safety — Gold and S&P rallying together"
        if current_corr < -0.5:
            return "risk-off rotation — Gold up, S&P down"
        return "Gold↔S&P correlation shift"

    # S&P ↔ Bitcoin
    if pair & {"MBT", "BTC"} and pair & {"MES", "MNQ"}:
        if current_corr > 0.8:
            return "risk-on euphoria — equities and crypto moving in lockstep"
        if current_corr < -0.3:
            return "crypto divergence — decoupling from equities"
        return "equity↔crypto correlation shift"

    # Crude ↔ S&P
    if pair == frozenset(["MCL", "MES"]):
        if current_corr < -0.3:
            return "energy divergence — crude falling while equities rise"
        if current_corr > 0.6:
            return "reflation trade — crude and equities rising together"
        return "crude↔equity correlation shift"

    # Gold ↔ Silver
    if pair == frozenset(["MGC", "SIL"]):
        if current_corr < 0.3:
            return "precious metals divergence — Gold/Silver ratio expanding"
        return "precious metals correlation shift"

    # S&P ↔ Nasdaq
    if pair == frozenset(["MES", "MNQ"]):
        if current_corr < 0.7:
            return "equity sector rotation — S&P and Nasdaq decorrelating"
        return "equity index correlation shift"

    # Generic
    if shift > 0.3:
        return f"{ticker_a}↔{ticker_b} correlation surge (+{shift:.2f})"
    if shift < -0.3:
        return f"{ticker_a}↔{ticker_b} correlation breakdown ({shift:.2f})"
    return f"{ticker_a}↔{ticker_b} correlation anomaly (z={current_corr:.2f})"


# ---------------------------------------------------------------------------
# Convenience: all-in-one for dashboard
# ---------------------------------------------------------------------------


def compute_cross_asset_summary(
    bars_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str] | None = None,
    short_window: int = 30,
    long_window: int = 200,
) -> dict[str, Any]:
    """Compute a full cross-asset summary suitable for dashboard rendering.

    Returns a dict with:
      - ``correlation_matrix``: current pairwise correlations (dict of dicts)
      - ``anomalies``: list of anomaly dicts
      - ``computed_at``: ISO timestamp
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if tickers is None:
        tickers = sorted(bars_by_ticker.keys())

    matrix = compute_correlation_matrix(bars_by_ticker, tickers=tickers, window=short_window)
    anomalies = detect_correlation_anomalies(
        bars_by_ticker,
        short_window=short_window,
        long_window=long_window,
    )

    # Convert matrix to nested dict (JSON-serialisable)
    matrix_dict: dict[str, dict[str, float | None]] = {}
    for t in matrix.index:
        row: dict[str, float | None] = {}
        for c in matrix.columns:
            val = matrix.loc[t, c]
            row[c] = round(float(val), 4) if pd.notna(val) else None
        matrix_dict[str(t)] = row

    return {
        "correlation_matrix": matrix_dict,
        "tickers": tickers,
        "anomalies": [a.to_dict() for a in anomalies],
        "anomaly_count": len(anomalies),
        "computed_at": datetime.now(tz=ZoneInfo("America/New_York")).isoformat(),
    }
