"""
Shared pytest fixtures for the futures test suite.

Provides synthetic OHLCV DataFrames that mirror real market data shapes
so every test module can exercise indicators, strategies, and detectors
without hitting the network.
"""

import atexit
import logging
import os
import threading

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Disable Redis connections during tests so alert/cache tests don't hang
# waiting for a Redis server that isn't running locally.
# ---------------------------------------------------------------------------
os.environ.setdefault("DISABLE_REDIS", "1")

# ---------------------------------------------------------------------------
# Prevent prometheus_client's C-extension (_mmap_dict) from crashing the
# interpreter during atexit/GC teardown.
#
# Root cause: prometheus_client >= 0.20 registers an atexit handler that
# tries to flush metrics to PROMETHEUS_MULTIPROC_DIR.  When that directory
# doesn't exist (test environment) the C extension aborts the process with
# "terminate called without an active exception" + core dump — even though
# all tests passed.  Setting this env var disables the _created timestamp
# series and the multiprocess flush path that triggers the crash.
# ---------------------------------------------------------------------------
os.environ.setdefault("PROMETHEUS_DISABLE_CREATED_SERIES", "true")


# ---------------------------------------------------------------------------
# Silence logging errors from daemon threads during interpreter shutdown.
#
# Root cause: DashboardEngine._loop() runs in a daemon thread.  When pytest
# finishes, Python tears down logging stream handlers *before* daemon threads
# are killed.  Any log call in those threads hits a closed file and raises
# "ValueError: I/O operation on closed file", which propagates through
# logging.Handler.handleError() → the C runtime abort handler →
# "terminate called without an active exception" + core dump.
#
# Setting logging.raiseExceptions = False tells the logging machinery to
# silently discard errors that occur inside handlers, preventing the crash.
# We register it as an atexit handler so it fires as early as possible in
# the teardown sequence — before stream handlers are closed.
# ---------------------------------------------------------------------------
def _silence_logging_on_shutdown() -> None:
    logging.raiseExceptions = False


atexit.register(_silence_logging_on_shutdown)


# ---------------------------------------------------------------------------
# Stop the DashboardEngine singleton before the interpreter exits.
#
# Root cause: DashboardEngine._loop() runs hmmlearn's HMM fitting inside a
# daemon thread.  hmmlearn uses Cython/C extensions that call std::terminate()
# if the thread is interrupted mid-computation during GC teardown — even with
# daemon=True.  The fix is to signal the engine to stop and join its thread
# with a short timeout *before* Python starts tearing down C extensions.
#
# We register this in two places:
#   1. atexit — fires early in the Python shutdown sequence.
#   2. A session-scoped autouse fixture — fires at pytest teardown time,
#      which is even earlier than atexit for the test process.
# ---------------------------------------------------------------------------
def _stop_engine_singleton() -> None:
    """Synchronously stop the DashboardEngine singleton if it was started."""
    try:
        from lib.trading.engine import _engine_instance  # noqa: PLC0415

        if _engine_instance is not None and _engine_instance._running:
            _engine_instance._running = False
            t = _engine_instance._thread
            if t is not None and t.is_alive():
                t.join(timeout=3)
    except Exception:
        pass


atexit.register(_stop_engine_singleton)

# ---------------------------------------------------------------------------
# Make sure the `src/` package is importable from tests regardless of how
# pytest is invoked (repo root, tests/ dir, or via CI).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_timestamps(n: int, freq: str = "5min", start: str = "2025-01-06 03:00") -> pd.DatetimeIndex:
    """Generate a tz-aware (US/Eastern) DatetimeIndex for *n* bars."""
    return pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")


def _random_walk_ohlcv(
    n: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.005,
    freq: str = "5min",
    seed: int = 42,
    volume_mean: int = 1000,
) -> pd.DataFrame:
    """Build a realistic-ish OHLCV DataFrame via geometric random walk.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a tz-aware DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    # Derive O/H/L from Close with small random spread
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.5, 0.5, n) * spread

    # Ensure H >= max(O, C) and L <= min(O, C)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(volume_mean, n).astype(float)
    volume = np.maximum(volume, 1)  # no zero-volume bars

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _trending_ohlcv(
    n: int = 300,
    start_price: float = 5000.0,
    trend: float = 0.001,
    volatility: float = 0.003,
    freq: str = "5min",
    seed: int = 123,
) -> pd.DataFrame:
    """Build a clearly trending OHLCV DataFrame (positive drift).

    Useful for testing trend-following indicators / strategies.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.005, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(800, n).astype(float)
    volume = np.maximum(volume, 1)

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _gappy_ohlcv(
    n: int = 300,
    start_price: float = 2700.0,
    gap_bars: list[int] | None = None,
    gap_size: float = 0.01,
    freq: str = "5min",
    seed: int = 77,
) -> pd.DataFrame:
    """Build OHLCV with intentional gaps (for FVG / sweep testing).

    At each bar index in *gap_bars*, an upward or downward gap is injected
    so that candle-1's high < candle-3's low (or vice-versa).
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.003, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    # Inject gaps
    if gap_bars is None:
        gap_bars = [50, 100, 150, 200]

    for gb in gap_bars:
        if gb + 2 >= n or gb < 2:
            continue
        direction = rng.choice([-1, 1])
        shift = close[gb] * gap_size * direction
        for k in range(gb, n):
            close[k] += shift
            high[k] += shift
            low[k] += shift
            opn[k] += shift

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(1200, n).astype(float)
    volume = np.maximum(volume, 1)

    # Inject volume spikes near gap bars (simulate event/impulsive bars)
    for gb in gap_bars:
        if gb < n:
            volume[gb] *= 4
            if gb + 1 < n:
                volume[gb + 1] *= 3

    idx = _make_timestamps(n, freq=freq)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _suppress_daemon_thread_logging_errors():
    """Session-scoped fixture that ensures logging errors from daemon threads
    (e.g. DashboardEngine._loop, regime HMM fitting) don't abort the process
    during interpreter teardown.

    The ``atexit`` registration above handles the common case, but this
    fixture also covers pytest's own teardown hooks which run before atexit.
    """
    yield
    # 1. Stop the engine background thread cleanly so hmmlearn's C extensions
    #    are not mid-computation when the interpreter tears down.
    _stop_engine_singleton()

    # 2. Also join any other lingering daemon threads with a short grace period
    #    so they can finish their current C-level work before GC runs.
    main_thread = threading.main_thread()
    for t in threading.enumerate():
        if t is not main_thread and t.daemon and t.is_alive():
            t.join(timeout=2)

    # 3. Disable logging exception propagation so any remaining threads that
    #    try to log against a closed stream fail silently instead of crashing.
    logging.raiseExceptions = False


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Generic 500-bar random-walk OHLCV DataFrame."""
    return _random_walk_ohlcv(n=500, seed=42)


@pytest.fixture()
def short_ohlcv_df() -> pd.DataFrame:
    """Short 50-bar DataFrame (edge-case / minimum-data tests)."""
    return _random_walk_ohlcv(n=50, seed=99, start_price=50.0)


@pytest.fixture()
def trending_df() -> pd.DataFrame:
    """300-bar trending DataFrame (positive drift)."""
    return _trending_ohlcv(n=300, seed=123)


@pytest.fixture()
def gappy_df() -> pd.DataFrame:
    """300-bar DataFrame with intentional price gaps for ICT tests."""
    return _gappy_ohlcv(n=300, seed=77)


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    """Empty OHLCV DataFrame for guard-clause tests."""
    return pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    """5-bar DataFrame (below most minimum-bar thresholds)."""
    return _random_walk_ohlcv(n=5, seed=11, start_price=20.0)


@pytest.fixture()
def gold_like_df() -> pd.DataFrame:
    """500-bar DataFrame at Gold-like price levels (~2700)."""
    return _random_walk_ohlcv(n=500, seed=55, start_price=2700.0, volatility=0.003)


@pytest.fixture()
def es_like_df() -> pd.DataFrame:
    """500-bar DataFrame at ES-like price levels (~5500)."""
    return _random_walk_ohlcv(n=500, seed=66, start_price=5500.0, volatility=0.002)
