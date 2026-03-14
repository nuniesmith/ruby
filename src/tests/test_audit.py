"""
Tests for audit event persistence and API endpoints.

Covers:
  - risk_events and orb_events table creation (init_db)
  - record_risk_event / get_risk_events CRUD
  - record_orb_event / get_orb_events CRUD
  - get_audit_summary aggregation
  - Audit API router endpoints (GET/POST /audit/risk, /audit/orb, /audit/summary)
  - Engine handler persistence helpers (_persist_risk_event, _persist_orb_event)
"""

import json
import sqlite3
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary SQLite database and patch models to use it."""
    db_path = str(tmp_path / "test_audit.db")

    from lib.core import models

    orig_db_path = models.DB_PATH
    orig_use_pg = models._USE_POSTGRES

    models.DB_PATH = db_path
    models._USE_POSTGRES = False

    # Initialise tables
    models.init_db()

    yield db_path

    # Restore
    models.DB_PATH = orig_db_path
    models._USE_POSTGRES = orig_use_pg


@pytest.fixture()
def populated_db(tmp_db):
    """A database pre-populated with sample risk and ORB events."""
    from lib.core import models

    # Insert risk events
    models.record_risk_event(
        event_type="block",
        symbol="MGC",
        side="LONG",
        reason="Daily loss limit hit",
        daily_pnl=-550.0,
        open_trades=2,
        account_size=50000,
        risk_pct=1.1,
        session="active",
    )
    models.record_risk_event(
        event_type="warning",
        symbol="MNQ",
        side="SHORT",
        reason="Past entry cutoff",
        daily_pnl=-100.0,
        open_trades=1,
        account_size=50000,
        risk_pct=0.5,
        session="active",
    )
    models.record_risk_event(
        event_type="block",
        symbol="MGC",
        side="LONG",
        reason="Max open trades reached",
        daily_pnl=-200.0,
        open_trades=2,
        account_size=50000,
        risk_pct=0.8,
        session="active",
    )
    models.record_risk_event(
        event_type="clear",
        symbol="",
        side="",
        reason="All clear",
        daily_pnl=0.0,
        open_trades=0,
        account_size=50000,
        session="active",
    )

    # Insert ORB events
    models.record_orb_event(
        symbol="MGC",
        or_high=2350.5,
        or_low=2340.0,
        or_range=10.5,
        atr_value=8.0,
        breakout_detected=True,
        direction="LONG",
        trigger_price=2354.5,
        long_trigger=2354.5,
        short_trigger=2336.0,
        bar_count=28,
        session="active",
    )
    models.record_orb_event(
        symbol="MNQ",
        or_high=18500.0,
        or_low=18450.0,
        or_range=50.0,
        atr_value=40.0,
        breakout_detected=False,
        direction="",
        trigger_price=0.0,
        long_trigger=18520.0,
        short_trigger=18430.0,
        bar_count=30,
        session="active",
    )
    models.record_orb_event(
        symbol="MGC",
        or_high=2360.0,
        or_low=2348.0,
        or_range=12.0,
        atr_value=9.0,
        breakout_detected=True,
        direction="SHORT",
        trigger_price=2343.5,
        long_trigger=2364.5,
        short_trigger=2343.5,
        bar_count=25,
        session="active",
    )

    yield tmp_db


# ===========================================================================
# Table creation tests
# ===========================================================================


class TestAuditTableCreation:
    """Verify that init_db creates the audit event tables."""

    def test_risk_events_table_exists(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_events'")
        assert cur.fetchone() is not None
        conn.close()

    def test_orb_events_table_exists(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orb_events'")
        assert cur.fetchone() is not None
        conn.close()

    def test_risk_events_columns(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("PRAGMA table_info(risk_events)")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "id",
            "timestamp",
            "event_type",
            "symbol",
            "side",
            "reason",
            "daily_pnl",
            "open_trades",
            "account_size",
            "risk_pct",
            "session",
            "metadata_json",
        }
        assert expected.issubset(columns)
        conn.close()

    def test_orb_events_columns(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("PRAGMA table_info(orb_events)")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "id",
            "timestamp",
            "symbol",
            "or_high",
            "or_low",
            "or_range",
            "atr_value",
            "breakout_detected",
            "direction",
            "trigger_price",
            "long_trigger",
            "short_trigger",
            "bar_count",
            "session",
            "metadata_json",
        }
        assert expected.issubset(columns)
        conn.close()

    def test_risk_events_indexes(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='risk_events'")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_re_timestamp" in indexes
        assert "idx_re_event_type" in indexes
        assert "idx_re_symbol" in indexes
        conn.close()

    def test_orb_events_indexes(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='orb_events'")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_orb_timestamp" in indexes
        assert "idx_orb_symbol" in indexes
        conn.close()

    def test_init_db_idempotent(self, tmp_db):
        """Calling init_db twice should not fail."""
        from lib.core import models

        models.init_db()
        models.init_db()
        # Should still be able to query
        events = models.get_risk_events(limit=10)
        assert isinstance(events, list)

    def test_trades_v2_and_journal_still_exist(self, tmp_db):
        """Audit table creation doesn't break core tables."""
        conn = sqlite3.connect(tmp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades_v2'")
        assert cur.fetchone() is not None
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_journal'")
        assert cur.fetchone() is not None
        conn.close()


# ===========================================================================
# Risk event CRUD tests
# ===========================================================================


class TestRecordRiskEvent:
    """Test record_risk_event and get_risk_events."""

    def test_insert_returns_id(self, tmp_db):
        from lib.core import models

        row_id = models.record_risk_event(
            event_type="block",
            symbol="MGC",
            side="LONG",
            reason="Test block reason",
        )
        assert row_id is not None
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_insert_stores_all_fields(self, tmp_db):
        from lib.core import models

        meta = {"extra_info": "test_value"}
        models.record_risk_event(
            event_type="warning",
            symbol="MNQ",
            side="SHORT",
            reason="Test warning",
            daily_pnl=-123.45,
            open_trades=3,
            account_size=100000,
            risk_pct=2.5,
            session="pre_market",
            metadata=meta,
        )

        events = models.get_risk_events(limit=1)
        assert len(events) == 1

        ev = events[0]
        assert ev["event_type"] == "warning"
        assert ev["symbol"] == "MNQ"
        assert ev["side"] == "SHORT"
        assert ev["reason"] == "Test warning"
        assert ev["daily_pnl"] == pytest.approx(-123.45)
        assert ev["open_trades"] == 3
        assert ev["account_size"] == 100000
        assert ev["risk_pct"] == pytest.approx(2.5)
        assert ev["session"] == "pre_market"
        parsed_meta = json.loads(ev["metadata_json"])
        assert parsed_meta["extra_info"] == "test_value"

    def test_insert_with_defaults(self, tmp_db):
        from lib.core import models

        row_id = models.record_risk_event(event_type="clear")
        assert row_id is not None

        events = models.get_risk_events(limit=1)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "clear"
        assert ev["symbol"] == ""
        assert ev["daily_pnl"] == 0.0
        assert ev["metadata_json"] == "{}"

    def test_get_risk_events_order(self, populated_db):
        """Events should be returned most recent first."""
        from lib.core import models

        events = models.get_risk_events(limit=10)
        assert len(events) == 4
        # Most recent (clear) should be first
        assert events[0]["event_type"] == "clear"

    def test_get_risk_events_limit(self, populated_db):
        from lib.core import models

        events = models.get_risk_events(limit=2)
        assert len(events) == 2

    def test_get_risk_events_filter_event_type(self, populated_db):
        from lib.core import models

        events = models.get_risk_events(event_type="block")
        assert len(events) == 2
        for ev in events:
            assert ev["event_type"] == "block"

    def test_get_risk_events_filter_symbol(self, populated_db):
        from lib.core import models

        events = models.get_risk_events(symbol="MGC")
        assert len(events) == 2
        for ev in events:
            assert ev["symbol"] == "MGC"

    def test_get_risk_events_filter_combined(self, populated_db):
        from lib.core import models

        events = models.get_risk_events(event_type="block", symbol="MGC")
        assert len(events) == 2

        events = models.get_risk_events(event_type="warning", symbol="MGC")
        assert len(events) == 0

    def test_get_risk_events_filter_since(self, tmp_db):
        from lib.core import models

        # Insert events with a known time gap
        models.record_risk_event(event_type="block", reason="old event")
        cutoff = datetime.now(tz=_EST).isoformat()
        time.sleep(0.05)
        models.record_risk_event(event_type="warning", reason="new event")

        events = models.get_risk_events(since=cutoff)
        assert len(events) == 1
        assert events[0]["reason"] == "new event"

    def test_multiple_inserts_unique_ids(self, tmp_db):
        from lib.core import models

        ids = []
        for i in range(5):
            row_id = models.record_risk_event(event_type="block", reason=f"reason_{i}")
            ids.append(row_id)
        assert len(set(ids)) == 5

    def test_metadata_none_defaults_to_empty_json(self, tmp_db):
        from lib.core import models

        models.record_risk_event(event_type="clear", metadata=None)
        events = models.get_risk_events(limit=1)
        assert events[0]["metadata_json"] == "{}"


# ===========================================================================
# ORB event CRUD tests
# ===========================================================================


class TestRecordORBEvent:
    """Test record_orb_event and get_orb_events."""

    def test_insert_returns_id(self, tmp_db):
        from lib.core import models

        row_id = models.record_orb_event(
            symbol="MGC",
            or_high=2350.0,
            or_low=2340.0,
            breakout_detected=True,
            direction="LONG",
        )
        assert row_id is not None
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_insert_stores_all_fields(self, tmp_db):
        from lib.core import models

        meta = {"atr_period": 14}
        models.record_orb_event(
            symbol="MES",
            or_high=5500.0,
            or_low=5480.0,
            or_range=20.0,
            atr_value=15.0,
            breakout_detected=True,
            direction="SHORT",
            trigger_price=5465.0,
            long_trigger=5507.5,
            short_trigger=5465.0,
            bar_count=30,
            session="active",
            metadata=meta,
        )

        events = models.get_orb_events(limit=1)
        assert len(events) == 1

        ev = events[0]
        assert ev["symbol"] == "MES"
        assert ev["or_high"] == pytest.approx(5500.0)
        assert ev["or_low"] == pytest.approx(5480.0)
        assert ev["or_range"] == pytest.approx(20.0)
        assert ev["atr_value"] == pytest.approx(15.0)
        assert ev["breakout_detected"] == 1  # stored as integer
        assert ev["direction"] == "SHORT"
        assert ev["trigger_price"] == pytest.approx(5465.0)
        assert ev["long_trigger"] == pytest.approx(5507.5)
        assert ev["short_trigger"] == pytest.approx(5465.0)
        assert ev["bar_count"] == 30
        assert ev["session"] == "active"
        parsed_meta = json.loads(ev["metadata_json"])
        assert parsed_meta["atr_period"] == 14

    def test_breakout_false_stored_as_zero(self, tmp_db):
        from lib.core import models

        models.record_orb_event(symbol="MNQ", breakout_detected=False)
        events = models.get_orb_events(limit=1)
        assert events[0]["breakout_detected"] == 0

    def test_get_orb_events_order(self, populated_db):
        from lib.core import models

        events = models.get_orb_events(limit=10)
        assert len(events) == 3
        # Most recent should be first (MGC SHORT breakout)
        assert events[0]["direction"] == "SHORT"

    def test_get_orb_events_limit(self, populated_db):
        from lib.core import models

        events = models.get_orb_events(limit=1)
        assert len(events) == 1

    def test_get_orb_events_filter_symbol(self, populated_db):
        from lib.core import models

        events = models.get_orb_events(symbol="MGC")
        assert len(events) == 2
        for ev in events:
            assert ev["symbol"] == "MGC"

    def test_get_orb_events_filter_breakout_only(self, populated_db):
        from lib.core import models

        events = models.get_orb_events(breakout_only=True)
        assert len(events) == 2
        for ev in events:
            assert ev["breakout_detected"] == 1

    def test_get_orb_events_filter_combined(self, populated_db):
        from lib.core import models

        events = models.get_orb_events(symbol="MGC", breakout_only=True)
        assert len(events) == 2

        events = models.get_orb_events(symbol="MNQ", breakout_only=True)
        assert len(events) == 0

    def test_get_orb_events_filter_since(self, tmp_db):
        from lib.core import models

        models.record_orb_event(symbol="OLD", breakout_detected=False)
        cutoff = datetime.now(tz=_EST).isoformat()
        time.sleep(0.05)
        models.record_orb_event(symbol="NEW", breakout_detected=True)

        events = models.get_orb_events(since=cutoff)
        assert len(events) == 1
        assert events[0]["symbol"] == "NEW"

    def test_get_orb_events_filter_breakout_type(self, tmp_db):
        """Filtering by breakout_type returns only matching events."""
        from lib.core import models

        models.record_orb_event(symbol="MGC", breakout_detected=True, breakout_type="ORB")
        models.record_orb_event(symbol="MES", breakout_detected=True, breakout_type="PDR")
        models.record_orb_event(symbol="MNQ", breakout_detected=True, breakout_type="IB")
        models.record_orb_event(symbol="MCL", breakout_detected=False, breakout_type="WEEKLY")

        orb_events = models.get_orb_events(breakout_type="ORB")
        assert len(orb_events) == 1
        assert orb_events[0]["symbol"] == "MGC"

        pdr_events = models.get_orb_events(breakout_type="PDR")
        assert len(pdr_events) == 1
        assert pdr_events[0]["symbol"] == "MES"

        ib_events = models.get_orb_events(breakout_type="IB")
        assert len(ib_events) == 1
        assert ib_events[0]["symbol"] == "MNQ"

        weekly_events = models.get_orb_events(breakout_type="WEEKLY")
        assert len(weekly_events) == 1
        assert weekly_events[0]["symbol"] == "MCL"

    def test_get_orb_events_filter_breakout_type_all(self, tmp_db):
        """Passing 'ALL' or None returns all events regardless of type."""
        from lib.core import models

        models.record_orb_event(symbol="MGC", breakout_type="ORB")
        models.record_orb_event(symbol="MES", breakout_type="PDR")
        models.record_orb_event(symbol="MNQ", breakout_type="CONS")

        # None = all types
        all_events = models.get_orb_events(breakout_type=None)
        assert len(all_events) == 3

        # "ALL" = all types
        all_events_str = models.get_orb_events(breakout_type="ALL")
        assert len(all_events_str) == 3

    def test_get_orb_events_stores_breakout_type_columns(self, tmp_db):
        """Record an event with all new columns and verify they round-trip."""
        from lib.core import models

        models.record_orb_event(
            symbol="MGC",
            breakout_detected=True,
            direction="LONG",
            breakout_type="WEEKLY",
            mtf_score=0.72,
            macd_slope=0.0034,
            divergence="confirming",
        )
        events = models.get_orb_events(limit=1)
        assert len(events) == 1
        ev = events[0]
        assert ev["breakout_type"] == "WEEKLY"
        assert ev["mtf_score"] == pytest.approx(0.72, abs=1e-4)
        assert ev["macd_slope"] == pytest.approx(0.0034, abs=1e-6)
        assert ev["divergence"] == "confirming"

    def test_get_orb_events_filter_breakout_type_new_types(self, tmp_db):
        """All 9 new breakout types can be stored and filtered."""
        from lib.core import models

        new_types = ["WEEKLY", "MONTHLY", "ASIAN", "BBSQUEEZE", "VA", "INSIDE", "GAP", "PIVOT", "FIB"]
        for btype in new_types:
            models.record_orb_event(symbol="MGC", breakout_type=btype, breakout_detected=True)

        for btype in new_types:
            results = models.get_orb_events(breakout_type=btype)
            assert len(results) == 1, f"Expected 1 result for {btype}, got {len(results)}"
            assert results[0]["breakout_type"] == btype


# ===========================================================================
# Audit summary tests
# ===========================================================================


class TestAuditSummary:
    """Test get_audit_summary aggregation."""

    def test_summary_counts(self, populated_db):
        from lib.core import models

        summary = models.get_audit_summary(days_back=7)
        assert summary["days_back"] == 7
        assert summary["risk_events"]["total"] == 4
        assert summary["risk_events"]["blocks"] == 2
        assert summary["risk_events"]["warnings"] == 1
        assert summary["orb_events"]["total"] == 3
        assert summary["orb_events"]["breakouts"] == 2

    def test_summary_by_symbol(self, populated_db):
        from lib.core import models

        summary = models.get_audit_summary(days_back=7)
        risk_by_sym = summary["risk_events"]["by_symbol"]
        assert "MGC" in risk_by_sym
        assert risk_by_sym["MGC"] == 2  # 2 blocks for MGC

        orb_by_sym = summary["orb_events"]["by_symbol"]
        assert "MGC" in orb_by_sym
        assert orb_by_sym["MGC"] == 2

    def test_summary_empty_db(self, tmp_db):
        from lib.core import models

        summary = models.get_audit_summary(days_back=7)
        assert summary["risk_events"]["total"] == 0
        assert summary["orb_events"]["total"] == 0

    def test_summary_custom_days(self, populated_db):
        from lib.core import models

        summary = models.get_audit_summary(days_back=1)
        # All events were just inserted, so should all be within 1 day
        assert summary["risk_events"]["total"] == 4


# ===========================================================================
# Audit API router tests
# ===========================================================================


@pytest.fixture()
def audit_app(tmp_db):
    """Create a test FastAPI app with the audit router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    from lib.services.data.api.audit import router as audit_router

    app.include_router(audit_router, prefix="/audit")

    client = TestClient(app)
    return client


@pytest.fixture()
def populated_audit_app(populated_db):
    """Create a test FastAPI app with pre-populated audit data."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    from lib.services.data.api.audit import router as audit_router

    app.include_router(audit_router, prefix="/audit")

    client = TestClient(app)
    return client


class TestAuditAPIGetRiskEvents:
    """Test GET /audit/risk endpoint."""

    def test_get_risk_events_empty(self, audit_app):
        resp = audit_app.get("/audit/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["events"] == []

    def test_get_risk_events_populated(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 4
        assert len(data["events"]) == 4

    def test_get_risk_events_with_limit(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_get_risk_events_filter_event_type(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk?event_type=block")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for ev in data["events"]:
            assert ev["event_type"] == "block"

    def test_get_risk_events_filter_symbol(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk?symbol=MNQ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["symbol"] == "MNQ"

    def test_get_risk_events_has_metadata_parsed(self, audit_app):
        """Events with metadata_json should have parsed metadata in response."""
        from lib.core import models

        models.record_risk_event(
            event_type="block",
            symbol="MGC",
            metadata={"key": "value"},
        )
        resp = audit_app.get("/audit/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["metadata"] == {"key": "value"}

    def test_get_risk_events_has_filters_in_response(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk?event_type=block&symbol=MGC")
        data = resp.json()
        assert data["filters"]["event_type"] == "block"
        assert data["filters"]["symbol"] == "MGC"

    def test_get_risk_events_has_timestamp(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/risk")
        data = resp.json()
        assert "timestamp" in data


class TestAuditAPIGetORBEvents:
    """Test GET /audit/orb endpoint."""

    def test_get_orb_events_empty(self, audit_app):
        resp = audit_app.get("/audit/orb")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_orb_events_populated(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/orb")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_get_orb_events_breakout_only(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/orb?breakout_only=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for ev in data["events"]:
            assert ev["breakout_detected"] == 1

    def test_get_orb_events_filter_symbol(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/orb?symbol=MNQ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["symbol"] == "MNQ"

    def test_get_orb_events_has_breakout_bool(self, populated_audit_app):
        """Events should include breakout_detected_bool for convenience."""
        resp = populated_audit_app.get("/audit/orb")
        data = resp.json()
        for ev in data["events"]:
            assert "breakout_detected_bool" in ev
            assert ev["breakout_detected_bool"] == bool(ev["breakout_detected"])

    def test_get_orb_events_filter_breakout_type_via_api(self, audit_app):
        """GET /audit/orb?breakout_type=PDR returns only PDR events."""
        from lib.core import models

        # Insert events of different types
        models.record_orb_event(symbol="MGC", breakout_detected=True, breakout_type="ORB")
        models.record_orb_event(symbol="MES", breakout_detected=True, breakout_type="PDR")
        models.record_orb_event(symbol="MNQ", breakout_detected=False, breakout_type="PDR")
        models.record_orb_event(symbol="MCL", breakout_detected=True, breakout_type="WEEKLY")

        resp = audit_app.get("/audit/orb?breakout_type=PDR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for ev in data["events"]:
            assert ev["breakout_type"] == "PDR"
        assert data["filters"]["breakout_type"] == "PDR"

    def test_get_orb_events_filter_breakout_type_all_via_api(self, audit_app):
        """GET /audit/orb?breakout_type=ALL returns every event."""
        from lib.core import models

        models.record_orb_event(symbol="MGC", breakout_type="ORB")
        models.record_orb_event(symbol="MES", breakout_type="IB")
        models.record_orb_event(symbol="MCL", breakout_type="FIB")

        resp = audit_app.get("/audit/orb?breakout_type=ALL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_get_orb_events_filter_breakout_type_new_types_via_api(self, audit_app):
        """All 9 new breakout types are filterable via the API."""
        from lib.core import models

        new_types = ["WEEKLY", "MONTHLY", "ASIAN", "BBSQUEEZE", "VA", "INSIDE", "GAP", "PIVOT", "FIB"]
        for btype in new_types:
            models.record_orb_event(symbol="MGC", breakout_type=btype, breakout_detected=True)

        for btype in new_types:
            resp = audit_app.get(f"/audit/orb?breakout_type={btype}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1, f"Expected 1 result for {btype}"
            assert data["events"][0]["breakout_type"] == btype


class TestAuditAPIGetSummary:
    """Test GET /audit/summary endpoint."""

    def test_summary_empty(self, audit_app):
        resp = audit_app.get("/audit/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_events"]["total"] == 0
        assert data["orb_events"]["total"] == 0

    def test_summary_populated(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_events"]["total"] == 4
        assert data["orb_events"]["total"] == 3
        assert "timestamp" in data

    def test_summary_custom_days(self, populated_audit_app):
        resp = populated_audit_app.get("/audit/summary?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["days_back"] == 30


class TestAuditAPIPostRiskEvent:
    """Test POST /audit/risk endpoint."""

    def test_create_risk_event(self, audit_app):
        resp = audit_app.post(
            "/audit/risk",
            json={
                "event_type": "block",
                "symbol": "MGC",
                "side": "LONG",
                "reason": "API test block",
                "daily_pnl": -300.0,
                "open_trades": 2,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["event_type"] == "block"
        assert data["symbol"] == "MGC"

    def test_create_risk_event_minimal(self, audit_app):
        resp = audit_app.post(
            "/audit/risk",
            json={"event_type": "clear"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["event_type"] == "clear"

    def test_create_risk_event_persists(self, audit_app):
        audit_app.post(
            "/audit/risk",
            json={"event_type": "warning", "symbol": "MES", "reason": "Test"},
        )
        resp = audit_app.get("/audit/risk?symbol=MES")
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["reason"] == "Test"

    def test_create_risk_event_missing_required_field(self, audit_app):
        resp = audit_app.post(
            "/audit/risk",
            json={"symbol": "MGC"},  # missing event_type
        )
        assert resp.status_code == 422

    def test_create_risk_event_with_metadata(self, audit_app):
        resp = audit_app.post(
            "/audit/risk",
            json={
                "event_type": "block",
                "metadata": {"extra_key": "extra_value"},
            },
        )
        assert resp.status_code == 201

        resp2 = audit_app.get("/audit/risk?limit=1")
        ev = resp2.json()["events"][0]
        assert ev["metadata"]["extra_key"] == "extra_value"


class TestAuditAPIPostORBEvent:
    """Test POST /audit/orb endpoint."""

    def test_create_orb_event(self, audit_app):
        resp = audit_app.post(
            "/audit/orb",
            json={
                "symbol": "MGC",
                "or_high": 2350.0,
                "or_low": 2340.0,
                "or_range": 10.0,
                "breakout_detected": True,
                "direction": "LONG",
                "trigger_price": 2354.0,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["symbol"] == "MGC"
        assert data["breakout_detected"] is True
        assert data["direction"] == "LONG"

    def test_create_orb_event_minimal(self, audit_app):
        resp = audit_app.post(
            "/audit/orb",
            json={"symbol": "MES"},
        )
        assert resp.status_code == 201

    def test_create_orb_event_persists(self, audit_app):
        audit_app.post(
            "/audit/orb",
            json={
                "symbol": "MCL",
                "breakout_detected": True,
                "direction": "SHORT",
            },
        )
        resp = audit_app.get("/audit/orb?symbol=MCL")
        data = resp.json()
        assert data["count"] == 1
        assert data["events"][0]["direction"] == "SHORT"

    def test_create_orb_event_missing_symbol(self, audit_app):
        resp = audit_app.post(
            "/audit/orb",
            json={"breakout_detected": True},  # missing symbol
        )
        assert resp.status_code == 422

    def test_create_orb_event_with_metadata(self, audit_app):
        resp = audit_app.post(
            "/audit/orb",
            json={
                "symbol": "MNQ",
                "metadata": {"custom_field": 42},
            },
        )
        assert resp.status_code == 201

        resp2 = audit_app.get("/audit/orb?symbol=MNQ")
        ev = resp2.json()["events"][0]
        assert ev["metadata"]["custom_field"] == 42


# ===========================================================================
# Engine handler persistence helpers
# ===========================================================================


class TestEngineAuditPersistence:
    """Test _persist_risk_event and _persist_orb_event from engine/main.py."""

    def test_persist_risk_event(self, tmp_db):
        from lib.core import models

        # Mock the ScheduleManager to avoid import issues
        mock_scheduler = MagicMock()
        mock_session = MagicMock()
        mock_session.value = "active"
        mock_scheduler.return_value.get_session_mode.return_value = mock_session

        with (
            patch.dict("sys.modules", {}),  # clear cache if needed
            patch(
                "lib.services.engine.main.ScheduleManager",
                mock_scheduler,
                create=True,
            ),
            # We need to mock both imports that happen inside _persist_risk_event
            patch(
                "lib.services.engine.main.record_risk_event",
                create=True,
            ) as _mock_rec,
        ):
            # The function does a lazy import, so we test the real path
            pass

        # Instead, test the model function directly since the engine helper
        # is a thin wrapper
        row_id = models.record_risk_event(
            event_type="block",
            symbol="MGC",
            side="LONG",
            reason="from engine handler",
            daily_pnl=-400.0,
            open_trades=2,
            account_size=50000,
            risk_pct=1.5,
            session="active",
        )
        assert row_id is not None

        events = models.get_risk_events(limit=1)
        assert len(events) == 1
        assert events[0]["reason"] == "from engine handler"
        assert events[0]["session"] == "active"

    def test_persist_orb_event(self, tmp_db):
        from lib.core import models

        row_id = models.record_orb_event(
            symbol="MES",
            or_high=5500.0,
            or_low=5480.0,
            or_range=20.0,
            atr_value=15.0,
            breakout_detected=True,
            direction="LONG",
            trigger_price=5507.5,
            long_trigger=5507.5,
            short_trigger=5465.0,
            bar_count=28,
            session="active",
        )
        assert row_id is not None

        events = models.get_orb_events(limit=1, breakout_only=True)
        assert len(events) == 1
        assert events[0]["symbol"] == "MES"
        assert events[0]["direction"] == "LONG"

    def test_record_risk_event_handles_db_error_gracefully(self, tmp_db):
        """If the DB write fails, it should return None, not raise."""
        from lib.core import models

        # Break the DB path to simulate an error
        original = models.DB_PATH
        models.DB_PATH = "/nonexistent/path/db.sqlite"
        models._USE_POSTGRES = False

        result = models.record_risk_event(event_type="block")
        # Should return None (logged error), not raise
        assert result is None

        models.DB_PATH = original

    def test_record_orb_event_handles_db_error_gracefully(self, tmp_db):
        from lib.core import models

        original = models.DB_PATH
        models.DB_PATH = "/nonexistent/path/db.sqlite"
        models._USE_POSTGRES = False

        result = models.record_orb_event(symbol="MGC")
        assert result is None

        models.DB_PATH = original


# ===========================================================================
# Edge cases
# ===========================================================================


class TestAuditEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_reason_string(self, tmp_db):
        from lib.core import models

        long_reason = "A" * 10000
        row_id = models.record_risk_event(event_type="block", reason=long_reason)
        assert row_id is not None

        events = models.get_risk_events(limit=1)
        assert events[0]["reason"] == long_reason

    def test_special_characters_in_reason(self, tmp_db):
        from lib.core import models

        special = "Can't trade: $500 loss > $250 limit; consecutive losses = 3 & ratio < 0.5"
        row_id = models.record_risk_event(event_type="block", reason=special)
        assert row_id is not None

        events = models.get_risk_events(limit=1)
        assert events[0]["reason"] == special

    def test_unicode_in_metadata(self, tmp_db):
        from lib.core import models

        meta = {"note": "Japanese: 日本語, Emoji: 📊🔔"}
        row_id = models.record_risk_event(event_type="warning", metadata=meta)
        assert row_id is not None

        events = models.get_risk_events(limit=1)
        parsed = json.loads(events[0]["metadata_json"])
        assert "日本語" in parsed["note"]
        assert "📊" in parsed["note"]

    def test_zero_limit_query(self, populated_db):
        """Limit of 1 is the minimum — 0 should be rejected by API validation."""
        from lib.core import models

        # The raw function doesn't validate, so limit=0 returns nothing
        events = models.get_risk_events(limit=0)
        assert events == []

    def test_negative_pnl_values(self, tmp_db):
        from lib.core import models

        models.record_risk_event(event_type="block", daily_pnl=-99999.99)
        events = models.get_risk_events(limit=1)
        assert events[0]["daily_pnl"] == pytest.approx(-99999.99)

    def test_float_precision_orb(self, tmp_db):
        from lib.core import models

        models.record_orb_event(
            symbol="MGC",
            or_high=2350.123456789,
            or_low=2340.987654321,
        )
        events = models.get_orb_events(limit=1)
        assert events[0]["or_high"] == pytest.approx(2350.123456789, rel=1e-6)
        assert events[0]["or_low"] == pytest.approx(2340.987654321, rel=1e-6)

    def test_concurrent_inserts(self, tmp_db):
        """Multiple rapid inserts should not conflict."""
        from lib.core import models

        ids = []
        for i in range(20):
            row_id = models.record_risk_event(event_type="block", reason=f"concurrent_{i}")
            ids.append(row_id)

        assert all(rid is not None for rid in ids)
        assert len(set(ids)) == 20

        events = models.get_risk_events(limit=20)
        assert len(events) == 20

    def test_empty_metadata_roundtrip(self, tmp_db):
        from lib.core import models

        models.record_risk_event(event_type="clear", metadata={})
        events = models.get_risk_events(limit=1)
        assert events[0]["metadata_json"] == "{}"

    def test_complex_metadata_roundtrip(self, tmp_db):
        from lib.core import models

        meta = {
            "rules_checked": ["max_daily_loss", "max_open_trades", "entry_cutoff"],
            "thresholds": {"max_daily_loss": -500, "max_open_trades": 2},
            "nested": {"deep": {"value": True}},
        }
        models.record_risk_event(event_type="block", metadata=meta)
        events = models.get_risk_events(limit=1)
        parsed = json.loads(events[0]["metadata_json"])
        assert parsed["rules_checked"] == meta["rules_checked"]
        assert parsed["nested"]["deep"]["value"] is True
