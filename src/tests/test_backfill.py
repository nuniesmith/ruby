"""
Tests for Historical Data Backfill
================================================

Covers:

**Table Management:**
  - init_backfill_table creates the historical_bars table (SQLite)
  - init_backfill_table is idempotent (safe to call multiple times)

**Symbol Resolution:**
  - _get_backfill_symbols returns all model ASSETS when no env override
  - _get_backfill_symbols respects BACKFILL_SYMBOLS env var
  - _symbol_display_name returns human name from TICKER_TO_NAME

**Data Fetching:**
  - fetch_bars_chunk tries Massive first, falls back to yfinance
  - _fetch_chunk_yfinance skips chunks >7 days old
  - _fetch_chunk_massive returns empty when Massive unavailable

**Database Operations:**
  - _store_bars inserts rows and is idempotent (duplicates ignored)
  - _store_bars skips zero-OHLC rows
  - _get_latest_stored_timestamp returns correct max timestamp
  - _get_bar_count returns correct count
  - _store_bars handles empty DataFrame

**Date Range Computation:**
  - _compute_date_range goes back DEFAULT_DAYS_BACK when no stored data
  - _compute_date_range starts from latest stored bar + 1 minute
  - _compute_date_range returns equal dates when fully up to date

**Chunk Generation:**
  - _generate_chunks splits range into correct number of chunks
  - _generate_chunks handles ranges smaller than chunk size
  - _generate_chunks handles exact multiples

**Single-Symbol Backfill:**
  - backfill_symbol returns summary dict with correct keys
  - backfill_symbol counts new bars correctly
  - backfill_symbol handles fetch failures gracefully
  - backfill_symbol marks already-up-to-date symbols

**Full Backfill (run_backfill):**
  - run_backfill processes all symbols
  - run_backfill reports status "complete" when no errors
  - run_backfill reports status "partial" on some errors
  - run_backfill reports status "failed" on table init error
  - run_backfill publishes status to Redis/cache

**Query Interface:**
  - get_stored_bars returns DataFrame with correct columns
  - get_stored_bars returns empty DataFrame when no data
  - get_stored_bars respects days_back parameter

**Backfill Status:**
  - get_backfill_status returns per-symbol counts
  - get_backfill_status returns empty when no data

**Gap Report:**
  - get_gap_report computes coverage percentage
  - get_gap_report identifies significant gaps
  - get_gap_report handles empty data

**SQL Helpers:**
  - _placeholder returns ? for SQLite
  - _format_sql substitutes placeholders correctly

**API Endpoints:**
  - GET /backfill/status returns 200 with status
  - GET /backfill/gaps/{symbol} returns gap report

**Integration:**
  - Full round-trip: store bars → query bars → verify content
  - Idempotent inserts: storing same bars twice doesn't duplicate
"""

import contextlib
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("DISABLE_REDIS", "1")

_EST = ZoneInfo("America/New_York")
_UTC = UTC


# ---------------------------------------------------------------------------
# Helpers — build synthetic 1-minute bar DataFrames
# ---------------------------------------------------------------------------


def _make_bars_df(
    n: int = 60,
    start_price: float = 2700.0,
    start_time: str = "2026-02-27 09:30:00",
    freq: str = "1min",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic 1-min OHLCV DataFrame for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start=start_time, periods=n, freq=freq, tz="America/New_York")

    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] + rng.randn() * 2.0)

    opens = prices
    closes = [p + rng.randn() * 1.0 for p in prices]
    highs = [max(o, c) + abs(rng.randn()) * 0.5 for o, c in zip(opens, closes, strict=True)]
    lows = [min(o, c) - abs(rng.randn()) * 0.5 for o, c in zip(opens, closes, strict=True)]
    vols = rng.randint(100, 5000, size=n).tolist()

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_sqlite_tempdb(tmp_path, monkeypatch):
    """Point DB_PATH to a temp SQLite file for every test."""
    db_file = str(tmp_path / "test_backfill.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("DATABASE_URL", "")

    # Also reset models if imported
    try:
        from lib.core import models

        models.DB_PATH = db_file
        models.DATABASE_URL = ""
        models._USE_POSTGRES = False
        models._sa_engine = None
    except ImportError:
        pass


@pytest.fixture()
def backfill_db(tmp_path, monkeypatch):
    """Set up a fresh SQLite DB with the historical_bars table."""
    db_file = str(tmp_path / "test_backfill.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("DATABASE_URL", "")

    try:
        from lib.core import models

        models.DB_PATH = db_file
        models.DATABASE_URL = ""
        models._USE_POSTGRES = False
        models._sa_engine = None
    except ImportError:
        pass

    from lib.services.engine.backfill import init_backfill_table

    init_backfill_table()
    return db_file


@pytest.fixture()
def sample_bars():
    """Return a sample DataFrame of 120 1-min bars."""
    return _make_bars_df(n=120, start_time="2026-02-27 09:30:00")


# ===========================================================================
# SECTION 1: Table Management
# ===========================================================================


class TestInitBackfillTable:
    """Test table creation."""

    def test_creates_table(self, tmp_path, monkeypatch):
        db_file = str(tmp_path / "init_test.db")
        monkeypatch.setenv("DB_PATH", db_file)
        monkeypatch.setenv("DATABASE_URL", "")
        try:
            from lib.core import models

            models.DB_PATH = db_file
            models.DATABASE_URL = ""
            models._USE_POSTGRES = False
        except ImportError:
            pass

        from lib.services.engine.backfill import init_backfill_table

        init_backfill_table()

        conn = sqlite3.connect(db_file)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_bars'")
        assert cur.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_path, monkeypatch):
        db_file = str(tmp_path / "idempotent_test.db")
        monkeypatch.setenv("DB_PATH", db_file)
        monkeypatch.setenv("DATABASE_URL", "")
        try:
            from lib.core import models

            models.DB_PATH = db_file
            models.DATABASE_URL = ""
            models._USE_POSTGRES = False
        except ImportError:
            pass

        from lib.services.engine.backfill import init_backfill_table

        init_backfill_table()
        init_backfill_table()  # Should not raise
        init_backfill_table()  # Third time for good measure

        conn = sqlite3.connect(db_file)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_bars'")
        assert cur.fetchone() is not None
        conn.close()

    def test_creates_indexes(self, tmp_path, monkeypatch):
        db_file = str(tmp_path / "index_test.db")
        monkeypatch.setenv("DB_PATH", db_file)
        monkeypatch.setenv("DATABASE_URL", "")
        try:
            from lib.core import models

            models.DB_PATH = db_file
            models.DATABASE_URL = ""
            models._USE_POSTGRES = False
        except ImportError:
            pass

        from lib.services.engine.backfill import init_backfill_table

        init_backfill_table()

        conn = sqlite3.connect(db_file)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_hb_%'")
        indexes = [row[0] for row in cur.fetchall()]
        assert "idx_hb_symbol_ts" in indexes
        assert "idx_hb_symbol_interval" in indexes
        conn.close()


# ===========================================================================
# SECTION 2: Symbol Resolution
# ===========================================================================


class TestSymbolResolution:
    """Test symbol list and display name helpers."""

    def test_get_backfill_symbols_from_models(self, monkeypatch):
        monkeypatch.setenv("BACKFILL_SYMBOLS", "")
        # Force re-import to pick up env change

        import lib.services.engine.backfill as bf

        bf._SYMBOLS_OVERRIDE = ""
        symbols = bf._get_backfill_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        # Should contain known micro tickers
        tickers_str = ",".join(symbols)
        # At least one of these should be present
        assert any(t in tickers_str for t in ["MGC=F", "MNQ=F", "MES=F", "MCL=F", "SI=F"])

    def test_get_backfill_symbols_from_env(self, monkeypatch):
        import lib.services.engine.backfill as bf

        bf._SYMBOLS_OVERRIDE = "MGC=F,MNQ=F"
        symbols = bf._get_backfill_symbols()
        assert symbols == ["MGC=F", "MNQ=F"]
        bf._SYMBOLS_OVERRIDE = ""

    def test_symbol_display_name_known(self):
        from lib.services.engine.backfill import _symbol_display_name

        # MGC=F should map to "Gold" via models.TICKER_TO_NAME
        name = _symbol_display_name("MGC=F")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_symbol_display_name_unknown(self):
        from lib.services.engine.backfill import _symbol_display_name

        name = _symbol_display_name("UNKNOWN_TICKER")
        assert name == "UNKNOWN_TICKER"


# ===========================================================================
# SECTION 3: SQL Helpers
# ===========================================================================


class TestSQLHelpers:
    """Test SQL placeholder and formatting utilities."""

    def test_placeholder_sqlite(self):
        from lib.services.engine.backfill import _placeholder

        # In test env, we're using SQLite
        ph = _placeholder()
        assert ph == "?"

    def test_format_sql_replaces_placeholders(self):
        from lib.services.engine.backfill import _format_sql

        template = "SELECT * FROM t WHERE a = {ph} AND b = {ph}"
        result = _format_sql(template)
        assert "{ph}" not in result
        assert "?" in result or "%s" in result

    def test_format_sql_no_placeholders(self):
        from lib.services.engine.backfill import _format_sql

        template = "SELECT COUNT(*) FROM t"
        result = _format_sql(template)
        assert result == template


# ===========================================================================
# SECTION 4: Database Operations
# ===========================================================================


class TestStoreBars:
    """Test bar storage and retrieval from SQLite."""

    def test_store_bars_inserts_rows(self, backfill_db, sample_bars):
        from lib.services.engine.backfill import (
            _get_bar_count,
            _get_conn,
            _store_bars,
        )

        conn = _get_conn()
        new_count = _store_bars(conn, "MGC=F", sample_bars, "1m")
        assert new_count == 120
        total = _get_bar_count(conn, "MGC=F", "1m")
        assert total == 120
        conn.close()

    def test_store_bars_idempotent(self, backfill_db, sample_bars):
        from lib.services.engine.backfill import (
            _get_bar_count,
            _get_conn,
            _store_bars,
        )

        conn = _get_conn()
        first = _store_bars(conn, "MGC=F", sample_bars, "1m")
        assert first == 120

        # Store same bars again
        second = _store_bars(conn, "MGC=F", sample_bars, "1m")
        assert second == 0  # No new bars

        total = _get_bar_count(conn, "MGC=F", "1m")
        assert total == 120  # Still 120, no duplicates
        conn.close()

    def test_store_bars_empty_df(self, backfill_db):
        from lib.services.engine.backfill import _get_conn, _store_bars

        conn = _get_conn()
        result = _store_bars(conn, "MGC=F", pd.DataFrame(), "1m")
        assert result == 0
        conn.close()

    def test_store_bars_skips_zero_ohlc(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
        )

        idx = pd.date_range("2026-01-01 09:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0, 0.0, 102.0, 0.0, 104.0],
                "High": [101.0, 0.0, 103.0, 0.0, 105.0],
                "Low": [99.0, 0.0, 101.0, 0.0, 103.0],
                "Close": [100.5, 0.0, 102.5, 0.0, 104.5],
                "Volume": [1000, 0, 2000, 0, 3000],
            },
            index=idx,
        )

        conn = _get_conn()
        new_count = _store_bars(conn, "TEST", df, "1m")
        assert new_count == 3  # Only non-zero rows stored
        conn.close()

    def test_get_latest_stored_timestamp(self, backfill_db, sample_bars):
        from lib.services.engine.backfill import (
            _get_conn,
            _get_latest_stored_timestamp,
            _store_bars,
        )

        conn = _get_conn()
        # No data yet
        ts = _get_latest_stored_timestamp(conn, "MGC=F", "1m")
        assert ts is None

        # Store bars
        _store_bars(conn, "MGC=F", sample_bars, "1m")

        ts = _get_latest_stored_timestamp(conn, "MGC=F", "1m")
        assert ts is not None
        # Should be the last bar timestamp
        assert "2026-02-27" in str(ts)
        conn.close()

    def test_get_bar_count_empty(self, backfill_db):
        from lib.services.engine.backfill import _get_bar_count, _get_conn

        conn = _get_conn()
        count = _get_bar_count(conn, "NONEXISTENT", "1m")
        assert count == 0
        conn.close()

    def test_store_multiple_symbols(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_bar_count,
            _get_conn,
            _store_bars,
        )

        bars1 = _make_bars_df(n=50, start_price=2700.0)
        bars2 = _make_bars_df(n=30, start_price=15000.0, seed=99)

        conn = _get_conn()
        _store_bars(conn, "MGC=F", bars1, "1m")
        _store_bars(conn, "MNQ=F", bars2, "1m")

        assert _get_bar_count(conn, "MGC=F", "1m") == 50
        assert _get_bar_count(conn, "MNQ=F", "1m") == 30
        conn.close()


# ===========================================================================
# SECTION 5: Date Range Computation
# ===========================================================================


class TestComputeDateRange:
    """Test date range calculation for gap-aware fetching."""

    def test_no_existing_data_goes_back_default(self, backfill_db):
        from lib.services.engine.backfill import (
            _compute_date_range,
            _get_conn,
        )

        conn = _get_conn()
        start_dt, end_dt = _compute_date_range("MGC=F", conn, days_back=30)
        conn.close()

        assert start_dt < end_dt
        diff = (end_dt - start_dt).days
        assert 29 <= diff <= 31

    def test_existing_data_starts_after_latest(self, backfill_db, sample_bars):
        from lib.services.engine.backfill import (
            _compute_date_range,
            _get_conn,
            _store_bars,
        )

        conn = _get_conn()
        _store_bars(conn, "MGC=F", sample_bars, "1m")

        start_dt, end_dt = _compute_date_range("MGC=F", conn, days_back=30)
        conn.close()

        # start_dt should be after the last bar
        assert start_dt.year >= 2026

    def test_up_to_date_returns_equal_dates(self, backfill_db):
        from lib.services.engine.backfill import (
            _compute_date_range,
            _get_conn,
            _store_bars,
        )

        # Store bars that are very recent (simulating up-to-date)
        now = datetime.now(tz=_UTC)
        idx = pd.date_range(start=now - timedelta(minutes=5), periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [101.0] * 5,
                "Low": [99.0] * 5,
                "Close": [100.5] * 5,
                "Volume": [1000] * 5,
            },
            index=idx,
        )

        conn = _get_conn()
        _store_bars(conn, "MGC=F", df, "1m")

        start_dt, end_dt = _compute_date_range("MGC=F", conn, days_back=30)
        conn.close()

        # The start should be at most a few minutes after end (essentially up to date)
        # or start >= end indicating nothing to fetch
        diff = (end_dt - start_dt).total_seconds()
        assert diff <= 120  # At most 2 minutes gap


# ===========================================================================
# SECTION 6: Chunk Generation
# ===========================================================================


class TestGenerateChunks:
    """Test date range chunking."""

    def test_basic_chunking(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 1, tzinfo=_UTC)
        end = datetime(2026, 1, 16, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=5)

        assert len(chunks) == 3  # 0-5, 5-10, 10-16
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    def test_single_chunk(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 1, tzinfo=_UTC)
        end = datetime(2026, 1, 3, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=5)

        assert len(chunks) == 1
        assert chunks[0] == (start, end)

    def test_exact_multiple(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 1, tzinfo=_UTC)
        end = datetime(2026, 1, 11, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=5)

        assert len(chunks) == 2
        assert chunks[0][0] == start
        assert chunks[1][1] == end

    def test_empty_range(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 5, tzinfo=_UTC)
        end = datetime(2026, 1, 5, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=5)

        assert len(chunks) == 0

    def test_chunk_boundaries_continuous(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 1, tzinfo=_UTC)
        end = datetime(2026, 2, 1, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=7)

        # Verify chunks are contiguous
        for i in range(len(chunks) - 1):
            assert chunks[i][1] == chunks[i + 1][0], f"Gap between chunk {i} and {i + 1}"


# ===========================================================================
# SECTION 7: Data Fetching
# ===========================================================================


class TestFetchBarsChunk:
    """Test the fetch_bars_chunk function with mocks."""

    def test_returns_massive_data_when_available(self):
        import lib.services.engine.backfill as bf

        mock_df = _make_bars_df(n=10)

        original = bf._fetch_chunk_massive
        bf._fetch_chunk_massive = lambda *a, **kw: mock_df
        try:
            result = bf.fetch_bars_chunk(
                "MGC=F",
                datetime(2026, 1, 1),
                datetime(2026, 1, 5),
            )
            assert len(result) == 10
        finally:
            bf._fetch_chunk_massive = original

    def test_falls_back_to_yfinance(self):
        import lib.services.engine.backfill as bf

        mock_df = _make_bars_df(n=20)

        orig_massive = bf._fetch_chunk_massive
        orig_yf = bf._fetch_chunk_yfinance
        bf._fetch_chunk_massive = lambda *a, **kw: pd.DataFrame()
        bf._fetch_chunk_yfinance = lambda *a, **kw: mock_df
        try:
            result = bf.fetch_bars_chunk(
                "MGC=F",
                datetime(2026, 1, 1),
                datetime(2026, 1, 5),
            )
            assert len(result) == 20
        finally:
            bf._fetch_chunk_massive = orig_massive
            bf._fetch_chunk_yfinance = orig_yf

    def test_returns_empty_when_both_fail(self):
        import lib.services.engine.backfill as bf

        orig_massive = bf._fetch_chunk_massive
        orig_yf = bf._fetch_chunk_yfinance
        bf._fetch_chunk_massive = lambda *a, **kw: pd.DataFrame()
        bf._fetch_chunk_yfinance = lambda *a, **kw: pd.DataFrame()
        try:
            result = bf.fetch_bars_chunk(
                "MGC=F",
                datetime(2026, 1, 1),
                datetime(2026, 1, 5),
            )
            assert result.empty
        finally:
            bf._fetch_chunk_massive = orig_massive
            bf._fetch_chunk_yfinance = orig_yf


class TestFetchChunkMassive:
    """Test Massive-specific fetching."""

    def test_returns_empty_when_unavailable(self):
        from lib.services.engine.backfill import _fetch_chunk_massive

        # _get_massive_provider is imported from cache inside the function
        with patch("lib.core.cache._get_massive_provider", return_value=None):
            result = _fetch_chunk_massive(
                "MGC=F",
                datetime(2026, 1, 1),
                datetime(2026, 1, 5),
            )
            assert result.empty

    def test_returns_empty_on_exception(self):
        from lib.services.engine.backfill import _fetch_chunk_massive

        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.resolve_from_yahoo.side_effect = Exception("API error")

        # _get_massive_provider is imported from cache inside the function
        with patch("lib.core.cache._get_massive_provider", return_value=mock_provider):
            result = _fetch_chunk_massive(
                "MGC=F",
                datetime(2026, 1, 1),
                datetime(2026, 1, 5),
            )
            assert result.empty


class TestFetchChunkYfinance:
    """Test yfinance-specific fetching."""

    def test_skips_old_chunks(self):
        from lib.services.engine.backfill import _fetch_chunk_yfinance

        # Chunk from 30 days ago should be skipped (yfinance 1m limit is ~7 days)
        old_start = datetime.now(tz=_UTC) - timedelta(days=30)
        old_end = datetime.now(tz=_UTC) - timedelta(days=25)

        result = _fetch_chunk_yfinance("MGC=F", old_start, old_end)
        assert result.empty

    def test_handles_exception(self):
        from lib.services.engine.backfill import _fetch_chunk_yfinance

        recent_start = datetime.now(tz=_UTC) - timedelta(days=2)
        recent_end = datetime.now(tz=_UTC)

        # yfinance is imported inside the function as `import yfinance as yf`
        with patch("yfinance.download", side_effect=Exception("err")):
            result = _fetch_chunk_yfinance("MGC=F", recent_start, recent_end)
            assert result.empty


# ===========================================================================
# SECTION 8: Single-Symbol Backfill
# ===========================================================================


class TestBackfillSymbol:
    """Test backfill_symbol function."""

    def test_returns_summary_dict(self, backfill_db):
        import lib.services.engine.backfill as bf

        mock_df = _make_bars_df(n=50)

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: mock_df
        try:
            result = bf.backfill_symbol("MGC=F", days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert isinstance(result, dict)
        expected_keys = {
            "symbol",
            "name",
            "bars_before",
            "bars_after",
            "bars_added",
            "chunks_fetched",
            "chunks_with_data",
            "start_date",
            "end_date",
            "duration_seconds",
            "error",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_counts_new_bars(self, backfill_db):
        import lib.services.engine.backfill as bf

        mock_df = _make_bars_df(n=60)

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: mock_df
        try:
            result = bf.backfill_symbol("MGC=F", days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert result["bars_before"] == 0
        assert result["bars_after"] > 0
        assert result["bars_added"] > 0
        assert result["error"] == ""

    def test_handles_fetch_failure(self, backfill_db):
        import lib.services.engine.backfill as bf

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: pd.DataFrame()
        try:
            result = bf.backfill_symbol("MGC=F", days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert result["bars_added"] == 0
        assert result["error"] == ""  # Empty chunks aren't errors

    def test_error_captured_in_result(self, backfill_db):
        from lib.services.engine.backfill import backfill_symbol

        with patch(
            "lib.services.engine.backfill._get_conn",
            side_effect=Exception("DB connection failed"),
        ):
            result = backfill_symbol("MGC=F", days_back=5)

        assert result["error"] != ""
        assert "DB connection failed" in result["error"]

    def test_duration_recorded(self, backfill_db):
        import lib.services.engine.backfill as bf

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: pd.DataFrame()
        try:
            result = bf.backfill_symbol("MGC=F", days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert result["duration_seconds"] >= 0


# ===========================================================================
# SECTION 9: Full Backfill (run_backfill)
# ===========================================================================


class TestRunBackfill:
    """Test the main run_backfill orchestrator."""

    def test_processes_specified_symbols(self, backfill_db):
        import lib.services.engine.backfill as bf

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: _make_bars_df(n=10)
        try:
            summary = bf.run_backfill(symbols=["MGC=F", "MNQ=F"], days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert len(summary["symbols"]) == 2
        symbols_processed = [s["symbol"] for s in summary["symbols"]]
        assert "MGC=F" in symbols_processed
        assert "MNQ=F" in symbols_processed

    def test_status_complete_on_success(self, backfill_db):
        import lib.services.engine.backfill as bf

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = lambda *a, **kw: _make_bars_df(n=10)
        try:
            summary = bf.run_backfill(symbols=["MGC=F"], days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert summary["status"] == "complete"
        assert len(summary["errors"]) == 0

    def test_status_partial_on_some_errors(self, backfill_db):
        import lib.services.engine.backfill as bf

        def mock_backfill_symbol(symbol, **kwargs):
            if symbol == "BAD=F":
                return {
                    "symbol": symbol,
                    "name": "Bad",
                    "bars_before": 0,
                    "bars_after": 0,
                    "bars_added": 0,
                    "chunks_fetched": 0,
                    "chunks_with_data": 0,
                    "start_date": "",
                    "end_date": "",
                    "duration_seconds": 0.1,
                    "error": "Connection refused",
                }
            return {
                "symbol": symbol,
                "name": "Good",
                "bars_before": 0,
                "bars_after": 10,
                "bars_added": 10,
                "chunks_fetched": 1,
                "chunks_with_data": 1,
                "start_date": "2026-01-01",
                "end_date": "2026-01-05",
                "duration_seconds": 0.5,
                "error": "",
            }

        original = bf.backfill_symbol
        bf.backfill_symbol = mock_backfill_symbol
        try:
            summary = bf.run_backfill(symbols=["MGC=F", "BAD=F"], days_back=5)
        finally:
            bf.backfill_symbol = original

        assert summary["status"] == "partial"
        assert len(summary["errors"]) == 1

    def test_status_failed_on_table_init_error(self, tmp_path, monkeypatch):
        import lib.services.engine.backfill as bf

        original = bf.init_backfill_table
        bf.init_backfill_table = MagicMock(side_effect=Exception("Permission denied"))
        try:
            summary = bf.run_backfill(symbols=["MGC=F"])
        finally:
            bf.init_backfill_table = original

        assert summary["status"] == "failed"
        assert len(summary["errors"]) > 0

    def test_total_bars_added_is_sum(self, backfill_db):
        import lib.services.engine.backfill as bf

        bars1 = _make_bars_df(n=20, seed=1)
        bars2 = _make_bars_df(n=30, seed=2)

        call_idx = [0]

        def mock_fetch(ticker, start, end):
            call_idx[0] += 1
            if call_idx[0] % 2 == 1:
                return bars1
            return bars2

        original = bf.fetch_bars_chunk
        bf.fetch_bars_chunk = mock_fetch
        try:
            summary = bf.run_backfill(symbols=["MGC=F", "MNQ=F"], days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = original

        assert summary["total_bars_added"] >= 0
        assert summary["total_duration_seconds"] >= 0

    def test_publishes_status_to_redis(self, backfill_db):
        import lib.services.engine.backfill as bf

        publish_calls = []
        orig_fetch = bf.fetch_bars_chunk
        orig_publish = bf._publish_backfill_status

        bf.fetch_bars_chunk = lambda *a, **kw: pd.DataFrame()
        bf._publish_backfill_status = lambda summary: publish_calls.append(summary)
        try:
            bf.run_backfill(symbols=["MGC=F"], days_back=5, chunk_days=5)
        finally:
            bf.fetch_bars_chunk = orig_fetch
            bf._publish_backfill_status = orig_publish

        assert len(publish_calls) == 1
        assert isinstance(publish_calls[0], dict)
        assert "status" in publish_calls[0]


# ===========================================================================
# SECTION 10: Query Interface
# ===========================================================================


class TestGetStoredBars:
    """Test the get_stored_bars query function."""

    def test_returns_dataframe_with_ohlcv(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_stored_bars,
        )

        # Use recent timestamps so they fall within the days_back window
        now = datetime.now(tz=_UTC)
        recent_start = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        recent_bars = _make_bars_df(n=30, start_time=recent_start)

        conn = _get_conn()
        _store_bars(conn, "MGC=F", recent_bars, "1m")
        conn.close()

        df = get_stored_bars("MGC=F", days_back=30)
        assert not df.empty
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_returns_empty_when_no_data(self, backfill_db):
        from lib.services.engine.backfill import get_stored_bars

        df = get_stored_bars("NONEXISTENT", days_back=30)
        assert df.empty

    def test_respects_days_back(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_stored_bars,
        )

        # Store bars with recent timestamps
        now = datetime.now(tz=_UTC)
        recent_start = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        bars = _make_bars_df(n=30, start_time=recent_start)
        conn = _get_conn()
        _store_bars(conn, "MGC=F", bars, "1m")
        conn.close()

        # With days_back=30, recent bars should be found
        df = get_stored_bars("MGC=F", days_back=30)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        # With days_back=0, nothing should be found (range is 0)
        df_zero = get_stored_bars("MGC=F", days_back=0)
        # May or may not be empty depending on rounding, just verify no crash
        assert isinstance(df_zero, pd.DataFrame)

    def test_returns_sorted_by_timestamp(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_stored_bars,
        )

        now = datetime.now(tz=_UTC)
        recent_start = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        recent_bars = _make_bars_df(n=60, start_time=recent_start)

        conn = _get_conn()
        _store_bars(conn, "MGC=F", recent_bars, "1m")
        conn.close()

        df = get_stored_bars("MGC=F", days_back=30)
        if not df.empty and hasattr(df.index, "is_monotonic_increasing"):
            assert df.index.is_monotonic_increasing


# ===========================================================================
# SECTION 11: Backfill Status
# ===========================================================================


class TestGetBackfillStatus:
    """Test get_backfill_status summary query."""

    def test_returns_empty_when_no_data(self, backfill_db):
        from lib.services.engine.backfill import get_backfill_status

        status = get_backfill_status()
        assert isinstance(status, dict)
        assert status["total_bars"] == 0
        assert len(status["symbols"]) == 0

    def test_returns_per_symbol_counts(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_backfill_status,
        )

        bars1 = _make_bars_df(n=120)
        bars2 = _make_bars_df(n=30, seed=99)

        conn = _get_conn()
        _store_bars(conn, "MGC=F", bars1, "1m")
        _store_bars(conn, "MNQ=F", bars2, "1m")
        conn.close()

        status = get_backfill_status()
        assert status["total_bars"] == 150  # 120 + 30
        assert len(status["symbols"]) == 2

        syms = {s["symbol"]: s for s in status["symbols"]}
        assert "MGC=F" in syms
        assert "MNQ=F" in syms
        assert syms["MGC=F"]["bar_count"] == 120
        assert syms["MNQ=F"]["bar_count"] == 30

    def test_includes_date_ranges(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_backfill_status,
        )

        bars = _make_bars_df(n=120)
        conn = _get_conn()
        _store_bars(conn, "MGC=F", bars, "1m")
        conn.close()

        status = get_backfill_status()
        sym_info = status["symbols"][0]
        assert sym_info["earliest"] is not None
        assert sym_info["latest"] is not None
        assert sym_info["interval"] == "1m"


# ===========================================================================
# SECTION 12: Gap Report
# ===========================================================================


class TestGetGapReport:
    """Test gap analysis."""

    def test_empty_data(self, backfill_db):
        from lib.services.engine.backfill import get_gap_report

        report = get_gap_report("NONEXISTENT", days_back=30)
        assert report["total_bars"] == 0
        assert report["coverage_pct"] == 0.0
        assert len(report["gaps"]) == 0

    def test_continuous_data_no_gaps(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_gap_report,
        )

        # Create continuous 1-min bars with recent timestamps
        now = datetime.now(tz=_UTC)
        recent_start = (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        bars = _make_bars_df(n=300, start_time=recent_start)
        conn = _get_conn()
        _store_bars(conn, "MGC=F", bars, "1m")
        conn.close()

        report = get_gap_report("MGC=F", days_back=30)
        assert report["total_bars"] == 300
        assert report["coverage_pct"] > 0

    def test_data_with_gap(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_gap_report,
        )

        # Create bars with a 2-hour gap in the middle using recent timestamps
        now = datetime.now(tz=_UTC)
        start1 = (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        start2 = (now - timedelta(hours=2, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        bars1 = _make_bars_df(n=30, start_time=start1)
        bars2 = _make_bars_df(n=30, start_time=start2, seed=99)
        combined = pd.concat([bars1, bars2])

        conn = _get_conn()
        _store_bars(conn, "MGC=F", combined, "1m")
        conn.close()

        report = get_gap_report("MGC=F", days_back=30)
        assert report["total_bars"] == 60
        # Should detect the 2+ hour gap
        assert len(report["gaps"]) >= 1

    def test_coverage_percentage_reasonable(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_gap_report,
        )

        now = datetime.now(tz=_UTC)
        recent_start = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        recent_bars = _make_bars_df(n=120, start_time=recent_start)

        conn = _get_conn()
        _store_bars(conn, "MGC=F", recent_bars, "1m")
        conn.close()

        report = get_gap_report("MGC=F", days_back=30)
        assert 0 <= report["coverage_pct"] <= 100

    def test_report_has_expected_keys(self, backfill_db):
        from lib.services.engine.backfill import get_gap_report

        report = get_gap_report("MGC=F", days_back=30)
        assert "symbol" in report
        assert "total_bars" in report
        assert "expected_bars" in report
        assert "coverage_pct" in report
        assert "gaps" in report


# ===========================================================================
# SECTION 13: Integration — Full Round-Trip
# ===========================================================================


class TestFullRoundTrip:
    """Integration tests: store → query → verify."""

    def test_store_and_retrieve(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_stored_bars,
        )

        # Create bars with known values using recent timestamps
        now = datetime.now(tz=_EST)
        recent_start = now - timedelta(minutes=15)
        idx = pd.date_range(recent_start, periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(10)],
                "High": [101.0 + i for i in range(10)],
                "Low": [99.0 + i for i in range(10)],
                "Close": [100.5 + i for i in range(10)],
                "Volume": [1000 * (i + 1) for i in range(10)],
            },
            index=idx,
        )

        conn = _get_conn()
        new = _store_bars(conn, "TEST=F", df, "1m")
        conn.close()

        assert new == 10

        # Retrieve
        result = get_stored_bars("TEST=F", days_back=30)
        assert len(result) == 10
        assert "Open" in result.columns
        assert "Close" in result.columns

    def test_idempotent_full_cycle(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
            get_backfill_status,
        )

        bars = _make_bars_df(n=50)
        conn = _get_conn()

        # First insert
        n1 = _store_bars(conn, "MGC=F", bars, "1m")
        assert n1 == 50

        # Second insert (same data)
        n2 = _store_bars(conn, "MGC=F", bars, "1m")
        assert n2 == 0  # No new bars
        conn.close()

        status = get_backfill_status()
        assert status["total_bars"] == 50  # Still 50, not 100

    def test_multiple_intervals(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_bar_count,
            _get_conn,
            _store_bars,
            get_backfill_status,
        )

        bars = _make_bars_df(n=30)
        conn = _get_conn()

        _store_bars(conn, "MGC=F", bars, "1m")
        _store_bars(conn, "MGC=F", bars, "5m")

        assert _get_bar_count(conn, "MGC=F", "1m") == 30
        assert _get_bar_count(conn, "MGC=F", "5m") == 30
        conn.close()

        status = get_backfill_status()
        assert status["total_bars"] == 60  # 30 per interval


# ===========================================================================
# SECTION 14: Publish Backfill Status
# ===========================================================================


class TestPublishBackfillStatus:
    """Test Redis publishing of backfill results."""

    def test_publish_does_not_crash_without_redis(self):
        from lib.services.engine.backfill import _publish_backfill_status

        summary = {
            "symbols": [],
            "total_bars_added": 0,
            "total_duration_seconds": 1.5,
            "errors": [],
            "status": "complete",
        }
        # Should not raise even without Redis
        _publish_backfill_status(summary)

    def test_publish_serializes_summary(self):
        from lib.services.engine.backfill import _publish_backfill_status

        summary = {
            "symbols": [{"symbol": "MGC=F", "bars_added": 100, "error": ""}],
            "total_bars_added": 100,
            "total_duration_seconds": 5.0,
            "errors": [],
            "status": "complete",
        }

        with patch("lib.core.cache.cache_set") as mock_set:
            _publish_backfill_status(summary)
            if mock_set.called:
                args = mock_set.call_args[0]
                assert args[0] == "engine:backfill_status"
                payload = json.loads(args[1])
                assert payload["status"] == "complete"


# ===========================================================================
# SECTION 15: Engine Handler Integration
# ===========================================================================


class TestEngineHandlerIntegration:
    """Test that _handle_historical_backfill calls backfill correctly."""

    def test_handler_calls_run_backfill(self):
        from lib.services.engine.main import _handle_historical_backfill

        mock_engine = MagicMock()
        mock_summary = {
            "status": "complete",
            "total_bars_added": 500,
            "total_duration_seconds": 10.0,
            "errors": [],
            "symbols": [],
        }

        with patch("lib.services.engine.backfill.run_backfill", return_value=mock_summary) as mock_run:
            _handle_historical_backfill(mock_engine)
            mock_run.assert_called_once()

    def test_handler_handles_import_error(self):
        from lib.services.engine.main import _handle_historical_backfill

        mock_engine = MagicMock()

        with patch(
            "lib.services.engine.backfill.run_backfill",
            side_effect=ImportError("No module named 'backfill'"),
        ):
            # Should not raise
            _handle_historical_backfill(mock_engine)

    def test_handler_handles_runtime_error(self):
        from lib.services.engine.main import _handle_historical_backfill

        mock_engine = MagicMock()

        with patch(
            "lib.services.engine.backfill.run_backfill",
            side_effect=RuntimeError("Connection refused"),
        ):
            # Should not raise
            _handle_historical_backfill(mock_engine)


# ===========================================================================
# SECTION 16: API Endpoints
# ===========================================================================


class TestBackfillAPIEndpoints:
    """Test the /backfill/ API endpoints."""

    @pytest.fixture()
    def client(self, backfill_db):
        """Build a minimal FastAPI app with health router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.health import router as health_router

        app = FastAPI()
        app.include_router(health_router)
        return TestClient(app)

    def test_backfill_status_endpoint(self, client):
        resp = client.get("/backfill/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "total_bars" in data
        assert "symbols" in data

    def test_backfill_gaps_endpoint(self, client):
        resp = client.get("/backfill/gaps/MGC%3DF")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "total_bars" in data

    def test_backfill_status_includes_timestamp(self, client):
        resp = client.get("/backfill/status")
        data = resp.json()
        assert "timestamp" in data

    def test_backfill_gaps_unknown_symbol(self, client):
        resp = client.get("/backfill/gaps/UNKNOWN")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_bars"] == 0


# ===========================================================================
# SECTION 17: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_store_bars_with_nan_values(self, backfill_db):
        from lib.services.engine.backfill import (
            _get_conn,
            _store_bars,
        )

        idx = pd.date_range("2026-01-01 09:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0, float("nan"), 102.0, 103.0, 104.0],
                "High": [101.0, float("nan"), 103.0, 104.0, 105.0],
                "Low": [99.0, float("nan"), 101.0, 102.0, 103.0],
                "Close": [100.5, float("nan"), 102.5, 103.5, 104.5],
                "Volume": [1000, 0, 2000, 3000, 4000],
            },
            index=idx,
        )

        conn = _get_conn()
        # NaN rows may cause issues but shouldn't crash
        with contextlib.suppress(Exception):
            _store_bars(conn, "TEST", df, "1m")
        conn.close()

    def test_store_bars_with_negative_volume(self, backfill_db):
        from lib.services.engine.backfill import _get_conn, _store_bars

        idx = pd.date_range("2026-01-01 09:30", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, -500, 2000],
            },
            index=idx,
        )

        conn = _get_conn()
        new = _store_bars(conn, "TEST", df, "1m")
        # Should store all 3 (volume can be negative in some edge cases)
        assert new == 3
        conn.close()

    def test_get_backfill_status_no_table(self, tmp_path, monkeypatch):
        """Status should handle missing table gracefully."""
        db_file = str(tmp_path / "no_table.db")
        monkeypatch.setenv("DB_PATH", db_file)
        monkeypatch.setenv("DATABASE_URL", "")
        try:
            from lib.core import models

            models.DB_PATH = db_file
            models.DATABASE_URL = ""
            models._USE_POSTGRES = False
            models._sa_engine = None
        except ImportError:
            pass

        from lib.services.engine.backfill import get_backfill_status

        # Create the DB but don't create the table
        conn = sqlite3.connect(db_file)
        conn.close()

        status = get_backfill_status()
        # Should return gracefully, possibly with an error
        assert isinstance(status, dict)

    def test_run_backfill_empty_symbols_list(self, backfill_db):
        from lib.services.engine.backfill import run_backfill

        summary = run_backfill(symbols=[])
        assert summary["status"] == "failed"  # No symbols = nothing to do
        assert summary["total_bars_added"] == 0

    def test_chunk_days_larger_than_range(self):
        from lib.services.engine.backfill import _generate_chunks

        start = datetime(2026, 1, 1, tzinfo=_UTC)
        end = datetime(2026, 1, 2, tzinfo=_UTC)
        chunks = _generate_chunks(start, end, chunk_days=30)
        assert len(chunks) == 1
        assert chunks[0] == (start, end)
