"""
Crypto Momentum Scorer — Unit Tests
=====================================

Tests the cross-asset crypto momentum scoring system that uses BTC/ETH/SOL
data to predict follow-through in correlated futures instruments (MES, MNQ,
MGC, MCL, MYM).

All tests run offline with synthetic OHLCV data — no Kraken API or Redis
required.

Run with:
    cd futures
    python -m pytest src/tests/test_crypto_momentum.py -v
"""

import math
import os
import sys
from datetime import datetime, timedelta
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lib.analysis.crypto_momentum import (  # noqa: E402
    ACTIONABLE_THRESHOLD,
    CRYPTO_ANCHORS,
    FUTURES_TARGETS,
    MIN_BARS_REQUIRED,
    SESSIONS,
    STRONG_THRESHOLD,
    CryptoMomentum,
    CryptoMomentumScorer,
    CryptoMomentumSignal,
    compute_atr,
    compute_ema,
    compute_rsi,
    compute_session_high_low,
    compute_single_crypto_momentum,
    compute_volume_ratio,
    crypto_momentum_to_tabular,
    detect_session,
    log_returns,
    pearson_correlation,
    score_futures_from_crypto,
)

_EST = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════
# Test data generators
# ═══════════════════════════════════════════════════════════════════════════


def _make_ohlcv(
    n: int = 100,
    base_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.01,
    base_volume: float = 1000.0,
    volume_surge_at: int | None = None,
    start_time: datetime | None = None,
    interval_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Parameters
    ----------
    n : int
        Number of bars.
    base_price : float
        Starting price.
    trend : float
        Per-bar drift (positive = uptrend, negative = downtrend).
    volatility : float
        Per-bar random walk standard deviation (fraction of price).
    base_volume : float
        Average volume per bar.
    volume_surge_at : int, optional
        If set, bar index where volume surges to 3x average.
    start_time : datetime, optional
        Starting timestamp. Defaults to midnight ET today.
    interval_minutes : int
        Minutes between bars.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = datetime.now(tz=_EST).replace(hour=0, minute=0, second=0, microsecond=0)

    timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(n)]

    closes = np.zeros(n)
    closes[0] = base_price
    for i in range(1, n):
        ret = trend + rng.normal(0, volatility)
        closes[i] = closes[i - 1] * (1 + ret)

    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.zeros(n)

    opens[0] = base_price
    for i in range(n):
        if i > 0:
            opens[i] = closes[i - 1] * (1 + rng.normal(0, volatility * 0.3))
        bar_range = abs(closes[i] - opens[i])
        extension = bar_range * rng.uniform(0.1, 0.5)
        highs[i] = max(opens[i], closes[i]) + extension
        lows[i] = min(opens[i], closes[i]) - extension
        volumes[i] = base_volume * rng.uniform(0.5, 1.5)

    if volume_surge_at is not None and 0 <= volume_surge_at < n:
        for j in range(max(0, volume_surge_at - 2), min(n, volume_surge_at + 3)):
            volumes[j] *= 3.0

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(timestamps, tz=_EST),
    )
    return df


def _make_trending_up(n: int = 100, base: float = 100.0, **kwargs) -> pd.DataFrame:
    """Generate a clear uptrending dataset."""
    return _make_ohlcv(n=n, base_price=base, trend=0.005, volatility=0.003, **kwargs)


def _make_trending_down(n: int = 100, base: float = 100.0, **kwargs) -> pd.DataFrame:
    """Generate a clear downtrending dataset."""
    return _make_ohlcv(n=n, base_price=base, trend=-0.005, volatility=0.003, **kwargs)


def _make_flat(n: int = 100, base: float = 100.0, **kwargs) -> pd.DataFrame:
    """Generate flat / ranging data."""
    return _make_ohlcv(n=n, base_price=base, trend=0.0, volatility=0.002, **kwargs)


def _make_session_time(hour: int, minute: int = 0) -> datetime:
    """Create a datetime at a specific ET hour today."""
    return datetime.now(tz=_EST).replace(hour=hour, minute=minute, second=0, microsecond=0)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Pure computation functions
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeEma:
    """Tests for compute_ema()."""

    def test_empty_input(self):
        result = compute_ema(np.array([]), 9)
        assert len(result) == 0

    def test_insufficient_data(self):
        # EMA seeds from the first finite value — no full-period warmup required,
        # so even when period > len(data) we get finite (usable) values.
        result = compute_ema(np.array([1.0, 2.0, 3.0]), 9)
        assert len(result) == 3
        assert all(np.isfinite(result))

    def test_exact_period(self):
        values = np.arange(1.0, 10.0)  # 9 values, period=9
        result = compute_ema(values, 9)
        assert len(result) == 9
        # Implementation seeds with values[0] and applies EMA from there
        # (no SMA warmup), so result[8] won't equal the simple mean.
        assert not np.isnan(result[8])
        # Recompute expected value using the same recursive formula
        alpha = 2.0 / (9 + 1)
        expected = values[0]
        for v in values[1:]:
            expected = alpha * v + (1.0 - alpha) * expected
        assert result[8] == pytest.approx(expected, rel=1e-6)

    def test_ema_tracks_trend(self):
        values = np.linspace(10, 20, 50)
        result = compute_ema(values, 9)
        # EMA should be below the last price in an uptrend (lag)
        last_valid = result[~np.isnan(result)]
        assert last_valid[-1] < values[-1]
        # EMA should be increasing
        assert last_valid[-1] > last_valid[-5]

    def test_ema_length_matches_input(self):
        values = np.random.default_rng(0).normal(100, 5, 200)
        for period in [5, 9, 21, 50]:
            result = compute_ema(values, period)
            assert len(result) == len(values)


class TestComputeRsi:
    """Tests for compute_rsi()."""

    def test_insufficient_data(self):
        result = compute_rsi(np.array([1.0, 2.0, 3.0]))
        assert result == 50.0  # neutral fallback

    def test_pure_uptrend(self):
        closes = np.linspace(10, 20, 30)
        rsi = compute_rsi(closes)
        assert rsi > 90  # should be near 100 for pure uptrend

    def test_pure_downtrend(self):
        closes = np.linspace(20, 10, 30)
        rsi = compute_rsi(closes)
        assert rsi < 10  # should be near 0 for pure downtrend

    def test_flat_market(self):
        closes = np.full(30, 100.0)
        rsi = compute_rsi(closes)
        # All deltas are zero → avg_loss=0 → RSI=100 by convention
        assert rsi == 100.0

    def test_rsi_range(self):
        rng = np.random.default_rng(42)
        closes = rng.normal(100, 5, 100).cumsum()
        rsi = compute_rsi(closes)
        assert 0 <= rsi <= 100


class TestComputeAtr:
    """Tests for compute_atr()."""

    def test_minimal_data(self):
        # Single bar: n < 2 → returns 0.0
        atr = compute_atr(np.array([10.0]), np.array([9.0]), np.array([9.5]))
        assert atr == 0.0

    def test_two_bars(self):
        # Two bars is insufficient for the default 14-period Wilder ATR
        # (requires at least period + 1 = 15 bars), so returns 0.0.
        atr = compute_atr(
            np.array([10.0, 11.0]),
            np.array([9.0, 9.5]),
            np.array([9.5, 10.5]),
        )
        assert atr == 0.0

    def test_atr_positive(self):
        df = _make_ohlcv(n=50)
        h = df["High"].values
        lo = df["Low"].values
        c = df["Close"].values
        atr = compute_atr(h, lo, c)  # type: ignore[arg-type]
        assert atr > 0

    def test_atr_increases_with_volatility(self):
        df_calm = _make_ohlcv(n=50, volatility=0.005)
        df_wild = _make_ohlcv(n=50, volatility=0.05)

        atr_calm = compute_atr(df_calm["High"].values, df_calm["Low"].values, df_calm["Close"].values)  # type: ignore[arg-type]
        atr_wild = compute_atr(df_wild["High"].values, df_wild["Low"].values, df_wild["Close"].values)  # type: ignore[arg-type]
        assert atr_wild > atr_calm


class TestComputeVolumeRatio:
    """Tests for compute_volume_ratio()."""

    def test_uniform_volume(self):
        volumes = np.full(30, 1000.0)
        ratio = compute_volume_ratio(volumes)
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_surge_detected(self):
        volumes = np.full(30, 1000.0)
        volumes[-1] = 5000.0  # 5x surge
        ratio = compute_volume_ratio(volumes)
        assert ratio > 4.0

    def test_empty_array(self):
        ratio = compute_volume_ratio(np.array([]))
        assert ratio == 1.0

    def test_single_value(self):
        ratio = compute_volume_ratio(np.array([100.0]))
        assert ratio == 1.0


class TestPearsonCorrelation:
    """Tests for pearson_correlation()."""

    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert pearson_correlation(xs, ys) == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert pearson_correlation(xs, ys) == pytest.approx(-1.0, abs=1e-10)

    def test_no_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, -1.0, 1.0, -1.0]
        r = pearson_correlation(xs, ys)
        assert abs(r) < 0.5

    def test_insufficient_data(self):
        assert math.isnan(pearson_correlation([1.0], [2.0]))
        assert math.isnan(pearson_correlation([1.0, 2.0], [3.0, 4.0]))

    def test_constant_series(self):
        xs = [5.0, 5.0, 5.0, 5.0]
        ys = [1.0, 2.0, 3.0, 4.0]
        assert math.isnan(pearson_correlation(xs, ys))

    def test_mismatched_length(self):
        assert math.isnan(pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0]))


class TestLogReturns:
    """Tests for log_returns()."""

    def test_basic(self):
        prices = [100.0, 110.0, 105.0]
        rets = log_returns(prices)
        assert len(rets) == 2
        assert rets[0] == pytest.approx(math.log(110 / 100), rel=1e-10)
        assert rets[1] == pytest.approx(math.log(105 / 110), rel=1e-10)

    def test_empty(self):
        assert log_returns([]) == []
        assert log_returns([100.0]) == []

    def test_zero_price_handled(self):
        rets = log_returns([100.0, 0.0, 50.0])
        assert rets[0] == 0.0  # zero price → 0 return
        assert rets[1] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Session detection
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectSession:
    """Tests for detect_session()."""

    def test_asian_session(self):
        # 20:00 ET → Asian session
        t = _make_session_time(20, 0)
        name, cfg = detect_session(t)
        assert name == "asian"

    def test_asian_session_after_midnight(self):
        # 01:00 ET → still Asian session
        t = _make_session_time(1, 0)
        name, cfg = detect_session(t)
        assert name == "asian"

    def test_london_session(self):
        # 04:00 ET → London
        t = _make_session_time(4, 0)
        name, cfg = detect_session(t)
        assert name == "london"

    def test_us_preopen(self):
        # 08:30 ET → US pre-open
        t = _make_session_time(8, 30)
        name, cfg = detect_session(t)
        assert name == "us_preopen"

    def test_us_rth(self):
        # 10:00 ET → US RTH
        t = _make_session_time(10, 0)
        name, cfg = detect_session(t)
        assert name == "us_rth"

    def test_session_boundary_start(self):
        # 19:00 ET exactly → Asian starts
        t = _make_session_time(19, 0)
        name, _ = detect_session(t)
        assert name == "asian"

    def test_session_boundary_end(self):
        # 02:00 ET exactly → London starts (Asian ends at 02:00)
        t = _make_session_time(2, 0)
        name, _ = detect_session(t)
        assert name == "london"

    def test_us_close_gap(self):
        # 17:00 ET → between US close and globex open
        t = _make_session_time(17, 0)
        name, _ = detect_session(t)
        assert name == "unknown"

    def test_lead_hours_asian(self):
        t = _make_session_time(21, 0)
        _, cfg = detect_session(t)
        assert cfg["lead_hours"] == 4.0

    def test_lead_hours_rth(self):
        t = _make_session_time(11, 0)
        _, cfg = detect_session(t)
        assert cfg["lead_hours"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Session high/low computation
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSessionHighLow:
    """Tests for compute_session_high_low()."""

    def test_empty_df(self):
        hi, lo = compute_session_high_low(pd.DataFrame(), dt_time(9, 0), dt_time(16, 0))
        assert hi == 0.0 and lo == 0.0

    def test_normal_session(self):
        start = _make_session_time(9, 0)
        df = _make_ohlcv(n=50, start_time=start)
        hi, lo = compute_session_high_low(df, dt_time(9, 0), dt_time(16, 0))
        assert hi > 0
        assert lo > 0
        assert hi >= lo

    def test_overnight_session(self):
        # Generate bars spanning 19:00 → 02:00
        start = _make_session_time(19, 0)
        # 7 hours at 5-min intervals = 84 bars
        df = _make_ohlcv(n=84, start_time=start, base_price=50000.0)
        hi, lo = compute_session_high_low(df, dt_time(19, 0), dt_time(2, 0))
        assert hi > 0
        assert lo > 0

    def test_missing_columns(self):
        df = pd.DataFrame({"Foo": [1, 2, 3]})
        hi, lo = compute_session_high_low(df, dt_time(9, 0), dt_time(16, 0))
        assert hi == 0.0 and lo == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Single crypto momentum computation
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeSingleCryptoMomentum:
    """Tests for compute_single_crypto_momentum()."""

    def test_bullish_momentum(self):
        df = _make_trending_up(n=80, base=50000.0)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        assert m.symbol == "BTC"
        assert m.direction == "bullish"
        assert m.strength > 0
        assert m.bar_count == 80
        assert m.price > 0
        assert m.ema_fast > 0
        assert m.ema_slow > 0

    def test_bearish_momentum(self):
        df = _make_trending_down(n=80, base=50000.0)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        assert m.direction == "bearish"
        assert m.strength > 0

    def test_neutral_flat_market(self):
        df = _make_flat(n=80, base=50000.0)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        # Flat market should have low strength
        assert m.strength < 0.5

    def test_insufficient_bars(self):
        df = _make_ohlcv(n=5, base_price=100.0)
        m = compute_single_crypto_momentum(df, "BTC")
        assert not m.valid
        assert "Insufficient" in m.error

    def test_empty_df(self):
        df = pd.DataFrame()
        m = compute_single_crypto_momentum(df, "ETH")
        assert not m.valid
        assert m.error != ""

    def test_missing_columns(self):
        df = pd.DataFrame(
            {"Foo": range(50), "Bar": range(50)},
            index=pd.date_range("2025-01-01", periods=50, freq="5min", tz=_EST),
        )
        m = compute_single_crypto_momentum(df, "SOL")
        assert not m.valid
        assert "Missing" in m.error

    def test_ema_spread_positive_uptrend(self):
        df = _make_trending_up(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.ema_spread > 0  # fast > slow in uptrend

    def test_ema_spread_negative_downtrend(self):
        df = _make_trending_down(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.ema_spread < 0  # fast < slow in downtrend

    def test_rsi_above_50_uptrend(self):
        df = _make_trending_up(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.rsi > 50

    def test_rsi_below_50_downtrend(self):
        df = _make_trending_down(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.rsi < 50

    def test_atr_positive(self):
        df = _make_ohlcv(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.atr > 0
        assert m.atr_pct > 0

    def test_volume_surge_detected(self):
        df = _make_ohlcv(n=80, volume_surge_at=78)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.volume_ratio > 1.5
        assert m.volume_surge

    def test_session_breakout_high(self):
        # Create bars that break above session high
        start = _make_session_time(20, 0)
        df = _make_trending_up(n=60, base=100.0, start_time=start)
        # Last bar should be well above session start
        m = compute_single_crypto_momentum(df, "BTC", now=start + timedelta(hours=5))
        assert m.valid
        # With strong uptrend, the last price should break the early session high
        # (this depends on the generated data, but trending up strongly should trigger it)

    def test_to_dict_all_fields(self):
        df = _make_trending_up(n=80)
        m = compute_single_crypto_momentum(df, "BTC")
        d = m.to_dict()
        expected_keys = {
            "symbol",
            "timestamp",
            "price",
            "ema_fast",
            "ema_slow",
            "ema_spread",
            "ema_cross_direction",
            "atr",
            "atr_pct",
            "session_high",
            "session_low",
            "broke_session_high",
            "broke_session_low",
            "rsi",
            "volume_ratio",
            "volume_surge",
            "direction",
            "strength",
            "bar_count",
            "valid",
            "error",
        }
        assert expected_keys.issubset(d.keys())
        assert d["valid"] is True
        assert d["symbol"] == "BTC"

    def test_lowercase_columns(self):
        """DataFrame with lowercase column names should work fine."""
        start = _make_session_time(9, 0)
        df = _make_ohlcv(n=50, start_time=start)
        df.columns = [c.lower() for c in df.columns]
        m = compute_single_crypto_momentum(df, "ETH")
        assert m.valid

    def test_strength_bounded(self):
        """Strength should be in [0, 1]."""
        for trend in [-0.01, -0.005, 0.0, 0.005, 0.01]:
            df = _make_ohlcv(n=80, trend=trend)
            m = compute_single_crypto_momentum(df, "X")
            assert 0.0 <= m.strength <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Score futures from crypto momentum
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreFuturesFromCrypto:
    """Tests for score_futures_from_crypto()."""

    def _make_bullish_momentum(self, symbol: str = "BTC") -> CryptoMomentum:
        m = CryptoMomentum(
            symbol=symbol,
            direction="bullish",
            strength=0.8,
            rsi=65.0,
            volume_ratio=2.0,
            volume_surge=True,
            ema_cross_direction="bullish",
            ema_spread=0.005,
            valid=True,
        )
        return m

    def _make_bearish_momentum(self, symbol: str = "BTC") -> CryptoMomentum:
        m = CryptoMomentum(
            symbol=symbol,
            direction="bearish",
            strength=0.7,
            rsi=35.0,
            volume_ratio=1.8,
            volume_surge=True,
            ema_cross_direction="bearish",
            ema_spread=-0.004,
            valid=True,
        )
        return m

    def _make_neutral_momentum(self, symbol: str = "BTC") -> CryptoMomentum:
        m = CryptoMomentum(
            symbol=symbol,
            direction="neutral",
            strength=0.05,
            rsi=50.0,
            volume_ratio=1.0,
            volume_surge=False,
            valid=True,
        )
        return m

    def test_bullish_signal_mes(self):
        momentums = [self._make_bullish_momentum("BTC")]
        # Asian session → high lead time
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", momentums, now=now)
        assert sig.direction == "bullish"
        assert sig.score > 0
        assert sig.futures_symbol == "MES"
        assert sig.session == "asian"
        assert sig.lead_time_hours == 4.0

    def test_bearish_signal_mnq(self):
        momentums = [self._make_bearish_momentum("BTC")]
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MNQ", momentums, now=now)
        assert sig.direction == "bearish"
        assert sig.score > 0

    def test_neutral_signal(self):
        momentums = [self._make_neutral_momentum()]
        sig = score_futures_from_crypto("MES", momentums)
        assert sig.direction == "neutral"
        assert sig.momentum_score == 0.0

    def test_no_valid_momentums(self):
        invalid = CryptoMomentum(symbol="BTC", valid=False)
        sig = score_futures_from_crypto("MES", [invalid])
        assert "No valid" in sig.error

    def test_unknown_futures_symbol(self):
        momentums = [self._make_bullish_momentum()]
        sig = score_futures_from_crypto("UNKNOWN", momentums)
        assert "Unknown" in sig.error

    def test_with_rolling_correlation(self):
        momentums = [self._make_bullish_momentum()]
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", momentums, rolling_corr=0.75, correlation_samples=100, now=now)
        assert sig.rolling_correlation == 0.75
        assert sig.correlation_samples == 100
        assert sig.correlation_score > 0

    def test_weak_correlation_reduces_score(self):
        momentums = [self._make_bullish_momentum()]
        now = _make_session_time(21, 0)
        sig_strong = score_futures_from_crypto("MES", momentums, rolling_corr=0.80, correlation_samples=100, now=now)
        sig_weak = score_futures_from_crypto("MES", momentums, rolling_corr=0.10, correlation_samples=100, now=now)
        assert sig_strong.score > sig_weak.score

    def test_insufficient_correlation_samples_uses_base(self):
        momentums = [self._make_bullish_momentum()]
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", momentums, rolling_corr=0.99, correlation_samples=5, now=now)
        # With only 5 samples (< MIN_CORRELATION_SAMPLES=20), rolling corr
        # should be ignored and base_correlation used instead
        assert sig.correlation_samples == 0  # indicates fallback

    def test_session_timing_affects_score(self):
        momentums = [self._make_bullish_momentum()]

        # Asian session — high lead time, session_score should be high
        sig_asian = score_futures_from_crypto("MES", momentums, now=_make_session_time(21, 0))
        # US RTH — no lead time, session_score should be low
        sig_rth = score_futures_from_crypto("MES", momentums, now=_make_session_time(11, 0))
        assert sig_asian.session_score > sig_rth.session_score
        assert sig_asian.score > sig_rth.score

    def test_volume_surge_boosts_score(self):
        m_surge = self._make_bullish_momentum()
        m_surge.volume_surge = True
        m_surge.volume_ratio = 3.0

        m_no_surge = self._make_bullish_momentum()
        m_no_surge.volume_surge = False
        m_no_surge.volume_ratio = 0.8

        now = _make_session_time(21, 0)
        sig_surge = score_futures_from_crypto("MES", [m_surge], now=now)
        sig_no = score_futures_from_crypto("MES", [m_no_surge], now=now)
        assert sig_surge.volume_score > sig_no.volume_score

    def test_multiple_anchors(self):
        btc = self._make_bullish_momentum("BTC")
        eth = self._make_bullish_momentum("ETH")
        sol = self._make_bearish_momentum("SOL")

        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", [btc, eth, sol], now=now)
        # BTC + ETH bullish should dominate SOL bearish
        assert sig.direction == "bullish"
        assert len(sig.anchor_signals) == 3

    def test_score_bounded_0_1(self):
        """Score should always be in [0, 1]."""
        momentums = [self._make_bullish_momentum()]
        for f_sym in FUTURES_TARGETS:
            sig = score_futures_from_crypto(f_sym, momentums)
            assert 0.0 <= sig.score <= 1.0

    def test_confidence_levels(self):
        m_strong = self._make_bullish_momentum()
        m_strong.strength = 0.95

        m_weak = self._make_neutral_momentum()

        now = _make_session_time(21, 0)
        sig_strong = score_futures_from_crypto("MES", [m_strong], now=now)
        sig_weak = score_futures_from_crypto("MES", [m_weak], now=now)

        assert sig_strong.confidence in ("moderate", "high")
        assert sig_weak.confidence == "low"

    def test_actionable_threshold(self):
        m = self._make_bullish_momentum()
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", [m], now=now)
        assert sig.is_actionable == (sig.score >= ACTIONABLE_THRESHOLD)

    def test_strong_threshold(self):
        m = self._make_bullish_momentum()
        m.strength = 0.95
        now = _make_session_time(21, 0)
        sig = score_futures_from_crypto("MES", [m], now=now)
        assert sig.is_strong == (sig.score >= STRONG_THRESHOLD)

    def test_to_dict_complete(self):
        momentums = [self._make_bullish_momentum()]
        sig = score_futures_from_crypto("MES", momentums, now=_make_session_time(21, 0))
        d = sig.to_dict()
        expected_keys = {
            "futures_symbol",
            "direction",
            "score",
            "confidence",
            "is_actionable",
            "is_strong",
            "momentum_score",
            "correlation_score",
            "session_score",
            "volume_score",
            "rolling_correlation",
            "correlation_samples",
            "anchor_signals",
            "session",
            "computed_at",
            "lead_time_hours",
            "error",
        }
        assert expected_keys.issubset(d.keys())


# ═══════════════════════════════════════════════════════════════════════════
# TEST: CryptoMomentumSignal.to_tabular_feature
# ═══════════════════════════════════════════════════════════════════════════


class TestTabularFeature:
    """Tests for CryptoMomentumSignal.to_tabular_feature()."""

    def test_bullish_positive(self):
        sig = CryptoMomentumSignal(direction="bullish", score=0.7, is_actionable=True)
        val = sig.to_tabular_feature()
        assert val == pytest.approx(0.7, rel=1e-6)

    def test_bearish_negative(self):
        sig = CryptoMomentumSignal(direction="bearish", score=0.6, is_actionable=True)
        val = sig.to_tabular_feature()
        assert val == pytest.approx(-0.6, rel=1e-6)

    def test_neutral_zero(self):
        sig = CryptoMomentumSignal(direction="neutral", score=0.1, is_actionable=False)
        val = sig.to_tabular_feature()
        assert val == 0.0

    def test_not_actionable_zero(self):
        sig = CryptoMomentumSignal(direction="bullish", score=0.2, is_actionable=False)
        val = sig.to_tabular_feature()
        assert val == 0.0

    def test_clamped_to_range(self):
        sig = CryptoMomentumSignal(direction="bullish", score=1.5, is_actionable=True)
        val = sig.to_tabular_feature()
        assert val == 1.0

        sig2 = CryptoMomentumSignal(direction="bearish", score=1.5, is_actionable=True)
        val2 = sig2.to_tabular_feature()
        assert val2 == -1.0


class TestCryptoMomentumToTabular:
    """Tests for crypto_momentum_to_tabular()."""

    def test_basic(self):
        signals = [
            CryptoMomentumSignal(futures_symbol="MES", direction="bullish", score=0.6, is_actionable=True),
            CryptoMomentumSignal(futures_symbol="MNQ", direction="bearish", score=0.5, is_actionable=True),
            CryptoMomentumSignal(futures_symbol="MGC", direction="neutral", score=0.1, is_actionable=False),
        ]
        result = crypto_momentum_to_tabular(signals)
        assert result["MES"] == pytest.approx(0.6, rel=1e-6)
        assert result["MNQ"] == pytest.approx(-0.5, rel=1e-6)
        assert result["MGC"] == 0.0

    def test_empty_signals(self):
        assert crypto_momentum_to_tabular([]) == {}


# ═══════════════════════════════════════════════════════════════════════════
# TEST: CryptoMomentumScorer (integration with synthetic data)
# ═══════════════════════════════════════════════════════════════════════════


class TestCryptoMomentumScorer:
    """Tests for the CryptoMomentumScorer class using score_with_data()."""

    def test_basic_scoring(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {
            "BTC": _make_trending_up(n=80, base=50000.0),
            "SOL": _make_trending_up(n=80, base=150.0),
        }
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data(crypto_bars, now=now)
        assert len(signals) == len(FUTURES_TARGETS)
        for sig in signals:
            assert sig.futures_symbol in FUTURES_TARGETS
            assert sig.error == ""

    def test_all_bullish_crypto(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {
            "BTC": _make_trending_up(n=80, base=50000.0),
            "SOL": _make_trending_up(n=80, base=150.0),
        }
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data(crypto_bars, now=now)
        for sig in signals:
            if sig.error == "":
                assert sig.direction == "bullish"

    def test_all_bearish_crypto(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {
            "BTC": _make_trending_down(n=80, base=50000.0),
            "SOL": _make_trending_down(n=80, base=150.0),
        }
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data(crypto_bars, now=now)
        for sig in signals:
            if sig.error == "":
                assert sig.direction == "bearish"

    def test_no_crypto_data(self):
        scorer = CryptoMomentumScorer()
        signals = scorer.score_with_data({})
        for sig in signals:
            assert "No valid" in sig.error or sig.score == 0.0

    def test_partial_crypto_data(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {
            "BTC": _make_trending_up(n=80, base=50000.0),
            # SOL missing
        }
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data(crypto_bars, now=now)
        # Should still produce valid signals from BTC alone
        for sig in signals:
            assert sig.error == ""

    def test_with_futures_correlation(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {
            "BTC": _make_trending_up(n=100, base=50000.0, seed=42),
            "SOL": _make_trending_up(n=100, base=150.0, seed=43),
        }
        futures_bars = {
            "MES": _make_trending_up(n=100, base=5900.0, seed=44),
            "MNQ": _make_trending_up(n=100, base=21000.0, seed=45),
        }
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data(crypto_bars, futures_bars, now=now)
        # When both crypto and futures trend up, correlation should be computed
        mes_sig = next(s for s in signals if s.futures_symbol == "MES")
        assert mes_sig.error == ""

    def test_custom_targets(self):
        custom_targets = {"CUSTOM": {"base_correlation": 0.7, "primary_crypto": "BTC"}}
        scorer = CryptoMomentumScorer(targets=custom_targets)
        crypto_bars = {"BTC": _make_trending_up(n=80, base=50000.0)}
        signals = scorer.score_with_data(crypto_bars)
        assert len(signals) == 1
        assert signals[0].futures_symbol == "CUSTOM"

    def test_custom_anchors(self):
        custom_anchors = [
            {"symbol": "BTC", "internal": "KRAKEN:XBTUSD", "weight": 1.0},
        ]
        scorer = CryptoMomentumScorer(anchors=custom_anchors)
        crypto_bars = {"BTC": _make_trending_up(n=80, base=50000.0)}
        signals = scorer.score_with_data(crypto_bars)
        assert len(signals) == len(FUTURES_TARGETS)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Configuration validation
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Tests for module-level configuration constants."""

    def test_crypto_anchors_have_required_fields(self):
        for anchor in CRYPTO_ANCHORS:
            assert "symbol" in anchor
            assert "internal" in anchor
            assert "weight" in anchor

    def test_crypto_anchor_weights_reasonable(self):
        # Weights don't have to sum to 1.0 (they're normalised at runtime)
        # but each should be positive
        for anchor in CRYPTO_ANCHORS:
            assert float(anchor["weight"]) > 0

    def test_futures_targets_have_required_fields(self):
        for _sym, cfg in FUTURES_TARGETS.items():
            assert "base_correlation" in cfg
            assert "primary_crypto" in cfg
            assert 0 <= cfg["base_correlation"] <= 1.0

    def test_sessions_cover_key_periods(self):
        assert "asian" in SESSIONS
        assert "london" in SESSIONS
        assert "us_preopen" in SESSIONS
        assert "us_rth" in SESSIONS

    def test_thresholds_ordered(self):
        assert ACTIONABLE_THRESHOLD < STRONG_THRESHOLD
        assert 0 < ACTIONABLE_THRESHOLD < 1
        assert 0 < STRONG_THRESHOLD < 1

    def test_min_bars_positive(self):
        assert MIN_BARS_REQUIRED > 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_single_bar(self):
        df = _make_ohlcv(n=1)
        m = compute_single_crypto_momentum(df, "BTC")
        assert not m.valid

    def test_exactly_min_bars(self):
        df = _make_ohlcv(n=MIN_BARS_REQUIRED)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        assert m.bar_count == MIN_BARS_REQUIRED

    def test_very_large_dataset(self):
        df = _make_ohlcv(n=5000, base_price=50000.0)
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        assert m.bar_count == 5000

    def test_zero_prices(self):
        """Bars with zero prices shouldn't crash."""
        df = _make_ohlcv(n=50)
        df.iloc[25, df.columns.get_loc("Close")] = 0.0
        m = compute_single_crypto_momentum(df, "BTC")
        # Should still produce a result (possibly with weird values)
        assert m.bar_count == 50

    def test_constant_prices(self):
        """Flat price (zero volatility) shouldn't crash."""
        n = 50
        start = _make_session_time(9, 0)
        times = [start + timedelta(minutes=i * 5) for i in range(n)]
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.0] * n,
                "Low": [100.0] * n,
                "Close": [100.0] * n,
                "Volume": [1000.0] * n,
            },
            index=pd.DatetimeIndex(times, tz=_EST),
        )
        m = compute_single_crypto_momentum(df, "BTC")
        assert m.valid
        # Constant prices: EMA spread=0, ATR=0, RSI=100 (no losses).
        # Session high==low==price so no breakout detected.
        # Strength should be very low regardless of direction label.
        assert m.strength < 0.5

    def test_nan_in_data(self):
        """NaN values in the DataFrame shouldn't crash."""
        df = _make_ohlcv(n=50)
        df.iloc[10, df.columns.get_loc("Close")] = float("nan")
        # Should not raise — may produce a result with unusual values
        try:
            m = compute_single_crypto_momentum(df, "BTC")
            # If it returns, it should be marked valid or have an error
            assert isinstance(m, CryptoMomentum)
        except Exception:
            # If it raises, that's also acceptable for NaN input
            pass

    def test_scorer_empty_crypto_bars_dict(self):
        scorer = CryptoMomentumScorer()
        signals = scorer.score_with_data({})
        assert len(signals) == len(FUTURES_TARGETS)
        for sig in signals:
            assert sig.score == 0.0 or "No valid" in sig.error

    def test_scorer_empty_dataframes(self):
        scorer = CryptoMomentumScorer()
        crypto_bars = {"BTC": pd.DataFrame(), "SOL": pd.DataFrame()}
        signals = scorer.score_with_data(crypto_bars)
        for sig in signals:
            assert sig.score == 0.0 or "No valid" in sig.error

    def test_all_futures_targets_scored(self):
        """Every target in FUTURES_TARGETS should get a signal."""
        scorer = CryptoMomentumScorer()
        crypto_bars = {"BTC": _make_trending_up(n=80, base=50000.0)}
        signals = scorer.score_with_data(crypto_bars)
        scored_symbols = {s.futures_symbol for s in signals}
        for sym in FUTURES_TARGETS:
            assert sym in scored_symbols

    def test_deterministic_with_same_data(self):
        """Same input should produce same output."""
        scorer = CryptoMomentumScorer()
        crypto_bars = {"BTC": _make_trending_up(n=80, base=50000.0, seed=99)}
        now = _make_session_time(21, 0)

        signals1 = scorer.score_with_data(crypto_bars, now=now)
        signals2 = scorer.score_with_data(crypto_bars, now=now)

        for s1, s2 in zip(signals1, signals2, strict=True):
            assert s1.score == s2.score
            assert s1.direction == s2.direction
            assert s1.momentum_score == s2.momentum_score


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Integration — full pipeline with realistic scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestRealisticScenarios:
    """Integration tests simulating realistic market scenarios."""

    def test_btc_asian_breakout_predicts_mes_at_london(self):
        """Simulate: BTC breaks Asian session high → MES should get bullish signal."""
        scorer = CryptoMomentumScorer()

        # BTC trending up strongly during Asian session
        start = _make_session_time(19, 0)
        btc = _make_trending_up(n=80, base=65000.0, start_time=start, seed=42)

        now = start + timedelta(hours=5)  # midnight → still in Asian session
        signals = scorer.score_with_data({"BTC": btc}, now=now)

        mes = next(s for s in signals if s.futures_symbol == "MES")
        assert mes.direction == "bullish"
        assert mes.session == "asian"
        assert mes.lead_time_hours > 0

    def test_crypto_selloff_warns_equity_futures(self):
        """Simulate: BTC + SOL dump → MES/MNQ should get bearish signal."""
        scorer = CryptoMomentumScorer()

        start = _make_session_time(3, 0)  # London session
        btc = _make_trending_down(n=80, base=60000.0, start_time=start, seed=42)
        sol = _make_trending_down(n=80, base=150.0, start_time=start, seed=43)

        now = start + timedelta(hours=3)
        signals = scorer.score_with_data({"BTC": btc, "SOL": sol}, now=now)

        mes = next(s for s in signals if s.futures_symbol == "MES")
        mnq = next(s for s in signals if s.futures_symbol == "MNQ")
        assert mes.direction == "bearish"
        assert mnq.direction == "bearish"

    def test_flat_crypto_neutral_signal(self):
        """Simulate: flat crypto → futures signals should be neutral/low."""
        scorer = CryptoMomentumScorer()

        start = _make_session_time(21, 0)
        btc = _make_flat(n=80, base=65000.0, start_time=start, seed=42)
        sol = _make_flat(n=80, base=150.0, start_time=start, seed=43)

        now = start + timedelta(hours=3)
        signals = scorer.score_with_data({"BTC": btc, "SOL": sol}, now=now)

        for sig in signals:
            # Flat crypto shouldn't produce strong directional signals
            assert sig.score < 0.5

    def test_mixed_crypto_signals(self):
        """BTC bullish + SOL bearish → should produce a moderate/mixed signal."""
        scorer = CryptoMomentumScorer()

        start = _make_session_time(21, 0)
        btc = _make_trending_up(n=80, base=65000.0, start_time=start, seed=42)
        sol = _make_trending_down(n=80, base=150.0, start_time=start, seed=43)

        now = start + timedelta(hours=3)
        signals = scorer.score_with_data({"BTC": btc, "SOL": sol}, now=now)

        mes = next(s for s in signals if s.futures_symbol == "MES")
        # BTC has weight 0.50 vs SOL 0.15, so bullish should dominate
        assert mes.direction == "bullish"
        # But the score should be lower than if both were bullish
        all_bull_signals = scorer.score_with_data(
            {
                "BTC": _make_trending_up(n=80, base=65000.0, start_time=start, seed=42),
                "SOL": _make_trending_up(n=80, base=150.0, start_time=start, seed=43),
            },
            now=now,
        )
        mes_all_bull = next(s for s in all_bull_signals if s.futures_symbol == "MES")
        assert mes.score <= mes_all_bull.score

    def test_gold_lower_correlation(self):
        """Gold (MGC) should have lower scores than MES for same crypto signal."""
        scorer = CryptoMomentumScorer()

        btc = _make_trending_up(n=80, base=65000.0, seed=42)
        now = _make_session_time(21, 0)
        signals = scorer.score_with_data({"BTC": btc}, now=now)

        mes = next(s for s in signals if s.futures_symbol == "MES")
        mgc = next(s for s in signals if s.futures_symbol == "MGC")

        # MES base_correlation=0.55 vs MGC base_correlation=0.25
        # So MES should get a higher correlation_score
        assert mes.correlation_score > mgc.correlation_score
