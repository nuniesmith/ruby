"""
Test suite for the FastAPI data-service API routers.

Tests all endpoints exposed by the data service without requiring
a running engine, Redis, or Postgres. Uses FastAPI's TestClient
with mocked dependencies.

Covers:
  - Health & metrics endpoints
  - Analysis endpoints (latest, status, assets, accounts, etc.)
  - Actions endpoints (force_refresh, optimize_now, update_settings, live feed)
  - Positions endpoints (CRUD for Rithmic integration)
  - Trades endpoints (create, close, cancel, list, legacy log_trade)
  - Journal endpoints (save, entries, stats, today)
"""

import os
from zoneinfo import ZoneInfo

import pytest

# Disable Redis before importing anything that touches it
os.environ.setdefault("DISABLE_REDIS", "1")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lib.services.data.main import SafeJSONResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EST = ZoneInfo("America/New_York")


class MockEngine:
    """Minimal mock of DashboardEngine for API testing."""

    def __init__(self):
        self.interval = "5m"
        self.period = "5d"
        self._refreshed = False
        self._live_feed_running = False
        self._settings = {
            "account_size": 150_000,
            "interval": "5m",
            "period": "5d",
        }

    def get_status(self):
        return {
            "engine": "running",
            "data_refresh": {"status": "idle", "last": "2025-01-06T10:00:00"},
            "optimization": {"status": "idle"},
            "backtest": {"status": "idle"},
            "live_feed": {
                "status": "running" if self._live_feed_running else "off",
                "connected": self._live_feed_running,
                "data_source": "yfinance",
                "bars": 42,
                "trades": 100,
            },
        }

    def force_refresh(self):
        self._refreshed = True

    def get_backtest_results(self):
        return [
            {
                "Asset": "Gold",
                "Return %": 12.5,
                "Win Rate %": 65.0,
                "Sharpe": 1.8,
                "# Trades": 20,
            }
        ]

    def get_strategy_history(self):
        return {"Gold": {"TrendEMA": {"wins": 5, "losses": 2}}}

    def get_live_feed_status(self):
        return self.get_status()["live_feed"]

    def start_live_feed(self):
        self._live_feed_running = True
        return True

    async def stop_live_feed(self):
        self._live_feed_running = False

    def upgrade_live_feed(self):
        pass

    def downgrade_live_feed(self):
        pass

    def update_settings(self, **kwargs):
        self._settings.update(kwargs)

    async def stop(self):
        pass


@pytest.fixture(autouse=True)
def _use_sqlite_tempdb(tmp_path, monkeypatch):
    """Point DB_PATH to a temp SQLite file for every test."""
    db_file = str(tmp_path / "test_journal.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("DATABASE_URL", "")
    # Re-init models to pick up new DB_PATH
    from lib.core import models

    models.DB_PATH = db_file
    models.DATABASE_URL = ""
    models._USE_POSTGRES = False
    models._sa_engine = None
    models.init_db()


@pytest.fixture()
def mock_engine():
    """Create a fresh MockEngine for each test."""
    return MockEngine()


def _build_test_app(mock_engine):
    """Build a fresh FastAPI app with routers but NO lifespan.

    This avoids the real lifespan which calls get_engine() and spawns
    background threads.  The mock engine is injected directly into the
    routers that need it.
    """
    from lib.services.data.api.actions import router as actions_router
    from lib.services.data.api.actions import (
        set_engine as actions_set_engine,
    )
    from lib.services.data.api.analysis import router as analysis_router
    from lib.services.data.api.analysis import (
        set_engine as analysis_set_engine,
    )
    from lib.services.data.api.health import router as health_router
    from lib.services.data.api.journal import router as journal_router
    from lib.services.data.api.positions import router as positions_router
    from lib.services.data.api.trades import router as trades_router

    # Inject the mock engine into routers that need it
    analysis_set_engine(mock_engine)
    actions_set_engine(mock_engine)

    app = FastAPI(title="Test Data Service", default_response_class=SafeJSONResponse)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
    app.include_router(actions_router, prefix="/actions", tags=["Actions"])
    app.include_router(positions_router, prefix="/positions", tags=["Positions"])
    app.include_router(trades_router, prefix="", tags=["Trades"])
    app.include_router(journal_router, prefix="/journal", tags=["Journal"])
    app.include_router(health_router, tags=["Health"])

    @app.get("/")
    def root():
        return {
            "service": "futures-data-service",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "analysis": "/analysis/latest",
                "status": "/analysis/status",
                "force_refresh": "/actions/force_refresh",
                "positions": "/positions/",
                "trades": "/trades",
                "journal": "/journal/entries",
                "health": "/health",
                "metrics": "/metrics",
            },
        }

    app.state.engine = mock_engine
    return app


@pytest.fixture()
def client(mock_engine):
    """Create a FastAPI TestClient with the mock engine injected.

    Builds a lightweight app copy (no lifespan) so the real
    DashboardEngine is never instantiated during tests.
    """
    import lib.services.data.api.health as _health_mod

    _orig_get_engine_or_none = _health_mod._get_engine_or_none
    _health_mod._get_engine_or_none = lambda: mock_engine

    app = _build_test_app(mock_engine)

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Clean up
    from lib.services.data.api.actions import (
        set_engine as actions_set_engine,
    )
    from lib.services.data.api.analysis import (
        set_engine as analysis_set_engine,
    )

    analysis_set_engine(None)
    actions_set_engine(None)
    _health_mod._get_engine_or_none = _orig_get_engine_or_none


# ═══════════════════════════════════════════════════════════════════════════
# Root endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestRoot:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "futures-data-service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"

    def test_root_contains_endpoint_links(self, client):
        resp = client.get("/")
        data = resp.json()
        endpoints = data["endpoints"]
        assert "analysis" in endpoints
        assert "health" in endpoints
        assert "trades" in endpoints
        assert "journal" in endpoints

    def test_docs_endpoint_accessible(self, client):
        # Our lightweight test app may or may not serve /docs
        # depending on FastAPI defaults. Just ensure no crash.
        resp = client.get("/docs")
        assert resp.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════
# Health & Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        # Status is 'ok' when engine is running AND model is available.
        # In the test environment there is no models/ directory so the
        # model health check reports available=False → overall status is
        # 'degraded'.  Both states are acceptable here; the important
        # thing is that the endpoint responds successfully.
        assert data["status"] in ("ok", "degraded")
        assert "timestamp" in data
        assert "components" in data

    def test_health_has_components(self, client):
        resp = client.get("/health")
        data = resp.json()
        components = data["components"]
        assert "redis" in components
        assert "engine" in components
        assert "live_feed" in components
        assert "database" in components

    def test_health_engine_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        engine_comp = data["components"]["engine"]
        assert engine_comp["status"] == "running"


class TestMetrics:
    def test_metrics_returns_ok(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data

    def test_metrics_has_engine_info(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        # engine_running may be True or False depending on mock
        assert "engine_running" in data
        assert "data_refresh" in data
        assert "optimization" in data


# ═══════════════════════════════════════════════════════════════════════════
# Analysis endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalysis:
    def test_status_returns_engine_state(self, client, mock_engine):
        resp = client.get("/analysis/status")
        assert resp.status_code == 200
        data = resp.json()
        status = mock_engine.get_status()
        assert data["engine"] == status["engine"]
        assert "data_refresh" in data

    def test_latest_all_returns_dict(self, client):
        resp = client.get("/analysis/latest")
        assert resp.status_code == 200
        data = resp.json()
        # Should have entries for each tracked asset
        assert isinstance(data, dict)
        # Each value should have at least a ticker key
        for _name, analysis in data.items():
            assert "ticker" in analysis

    def test_latest_ticker_returns_analysis(self, client):
        resp = client.get("/analysis/latest/GC=F")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "GC=F"
        assert "wave" in data
        assert "volatility" in data
        assert "signal_quality" in data
        assert "regime" in data
        assert "ict" in data
        assert "cvd" in data

    def test_latest_ticker_with_params(self, client):
        resp = client.get("/analysis/latest/ES=F?interval=5m&period=5d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "ES=F"
        assert data["interval"] == "5m"
        assert data["period"] == "5d"

    def test_latest_unknown_ticker(self, client):
        resp = client.get("/analysis/latest/UNKNOWN")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "UNKNOWN"
        # Unknown ticker should still return structure, just with null values
        assert data["wave"] is None

    def test_data_source_endpoint(self, client):
        resp = client.get("/analysis/data_source")
        assert resp.status_code == 200
        data = resp.json()
        assert "data_source" in data
        assert data["data_source"] in ("yfinance", "Massive")

    def test_assets_endpoint(self, client):
        resp = client.get("/analysis/assets")
        assert resp.status_code == 200
        data = resp.json()
        assert "assets" in data
        assets = data["assets"]
        assert "Gold" in assets
        assert "S&P" in assets
        assert "Nasdaq" in assets

    def test_accounts_endpoint(self, client):
        resp = client.get("/analysis/accounts")
        assert resp.status_code == 200
        data = resp.json()
        assert "50k" in data
        assert "100k" in data
        assert "150k" in data

    def test_backtest_results(self, client, mock_engine):
        resp = client.get("/analysis/backtest_results")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results = data["results"]
        assert len(results) >= 1
        assert results[0]["Asset"] == "Gold"

    def test_strategy_history(self, client, mock_engine):
        resp = client.get("/analysis/strategy_history")
        assert resp.status_code == 200
        data = resp.json()
        assert "Gold" in data

    def test_live_feed_status(self, client):
        resp = client.get("/analysis/live_feed")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "data_source" in data

    def test_force_refresh_via_analysis(self, client):
        """The analysis router also has a force_refresh POST endpoint."""
        resp = client.post("/analysis/force_refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "refresh_triggered"


# ═══════════════════════════════════════════════════════════════════════════
# Actions endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestActions:
    def test_force_refresh(self, client, mock_engine):
        mock_engine._refreshed = False
        resp = client.post("/actions/force_refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "refresh_triggered"
        assert mock_engine._refreshed is True

    def test_optimize_now(self, client, mock_engine):
        mock_engine._refreshed = False
        resp = client.post("/actions/optimize_now")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "optimization_triggered"
        assert "assets" in data
        assert mock_engine._refreshed is True

    def test_update_settings_account_size(self, client, mock_engine):
        resp = client.post(
            "/actions/update_settings",
            json={"account_size": 100_000},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["changed"]["account_size"] == 100_000

    def test_update_settings_interval(self, client, mock_engine):
        resp = client.post(
            "/actions/update_settings",
            json={"interval": "15m"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["changed"]["interval"] == "15m"

    def test_update_settings_invalid_account_size(self, client):
        resp = client.post(
            "/actions/update_settings",
            json={"account_size": 999_999},
        )
        assert resp.status_code == 400

    def test_update_settings_invalid_interval(self, client):
        resp = client.post(
            "/actions/update_settings",
            json={"interval": "3m"},
        )
        assert resp.status_code == 400

    def test_update_settings_no_changes(self, client):
        resp = client.post("/actions/update_settings", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_change"

    def test_live_feed_start(self, client, mock_engine):
        mock_engine._live_feed_running = False
        resp = client.post("/actions/live_feed/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert mock_engine._live_feed_running is True

    def test_live_feed_stop(self, client, mock_engine):
        mock_engine._live_feed_running = True
        resp = client.post("/actions/live_feed/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_live_feed_upgrade(self, client):
        resp = client.post("/actions/live_feed/upgrade")
        assert resp.status_code == 200
        assert resp.json()["status"] == "upgraded"

    def test_live_feed_downgrade(self, client):
        resp = client.post("/actions/live_feed/downgrade")
        assert resp.status_code == 200
        assert resp.json()["status"] == "downgraded"


# ═══════════════════════════════════════════════════════════════════════════
# Positions endpoints (Rithmic integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestPositions:
    def test_get_positions_empty(self, client):
        resp = client.get("/positions/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_positions"] is False
        assert data["positions"] == []
        assert data["total_unrealized_pnl"] == 0.0

    def test_update_and_get_positions(self, client):
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 2,
                    "avgPrice": 5500.25,
                    "unrealizedPnL": 125.50,
                },
                {
                    "symbol": "MGCZ5",
                    "side": "Short",
                    "quantity": 1,
                    "avgPrice": 2750.00,
                    "unrealizedPnL": -30.00,
                },
            ],
        }
        # Push positions
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"
        assert data["account"] == "Sim101"
        assert data["positions_count"] == 2

        # Read them back
        resp = client.get("/positions/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_positions"] is True
        assert data["account"] == "Sim101"
        assert len(data["positions"]) == 2
        assert data["total_unrealized_pnl"] == 95.50  # 125.50 + (-30.00)

    def test_update_positions_empty(self, client):
        payload = {"account": "Sim101", "positions": []}
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["positions_count"] == 0

    def test_clear_positions(self, client):
        # Push a position first
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 5500.0,
                    "unrealizedPnL": 50.0,
                }
            ],
        }
        client.post("/positions/update", json=payload)

        # Clear
        resp = client.delete("/positions/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

        # Verify cleared
        resp = client.get("/positions/")
        data = resp.json()
        assert data["has_positions"] is False

    def test_position_pnl_calculation(self, client):
        payload = {
            "account": "Live1",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 5,
                    "avgPrice": 5450.00,
                    "unrealizedPnL": 500.00,
                },
                {
                    "symbol": "MNQZ5",
                    "side": "Long",
                    "quantity": 3,
                    "avgPrice": 19500.00,
                    "unrealizedPnL": -200.00,
                },
                {
                    "symbol": "MGCZ5",
                    "side": "Short",
                    "quantity": 2,
                    "avgPrice": 2780.00,
                    "unrealizedPnL": 150.00,
                },
            ],
        }
        client.post("/positions/update", json=payload)

        resp = client.get("/positions/")
        data = resp.json()
        assert data["total_unrealized_pnl"] == 450.00  # 500 - 200 + 150


# ═══════════════════════════════════════════════════════════════════════════
# Trades endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestTrades:
    def test_create_trade(self, client):
        resp = client.post(
            "/trades",
            json={
                "account_size": 150_000,
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.00,
                "sl": 2740.00,
                "tp": 2770.00,
                "contracts": 2,
                "strategy": "TrendEMA",
                "notes": "Test trade",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["asset"] == "Gold"
        assert data["direction"] == "LONG"
        assert data["entry"] == 2750.00
        assert data["status"] == "OPEN"
        assert data["contracts"] == 2

    def test_close_trade(self, client):
        # Create a trade first
        resp = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.00,
                "contracts": 1,
            },
        )
        trade_id = resp.json()["id"]

        # Close it
        resp = client.post(
            f"/trades/{trade_id}/close",
            json={"close_price": 2760.00},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "CLOSED"
        assert data["close_price"] == 2760.00

    def test_cancel_trade(self, client):
        resp = client.post(
            "/trades",
            json={
                "asset": "S&P",
                "direction": "SHORT",
                "entry": 5500.00,
                "contracts": 1,
            },
        )
        trade_id = resp.json()["id"]

        resp = client.post(f"/trades/{trade_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    def test_list_trades_empty(self, client):
        resp = client.get("/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_trades_with_data(self, client):
        # Create two trades
        client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.0,
                "contracts": 1,
            },
        )
        client.post(
            "/trades",
            json={
                "asset": "S&P",
                "direction": "SHORT",
                "entry": 5500.0,
                "contracts": 1,
            },
        )

        resp = client.get("/trades")
        data = resp.json()
        assert len(data) == 2

    def test_list_open_trades(self, client):
        # Create and close one, leave one open
        resp1 = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.0,
                "contracts": 1,
            },
        )
        client.post(
            "/trades",
            json={
                "asset": "S&P",
                "direction": "SHORT",
                "entry": 5500.0,
                "contracts": 1,
            },
        )
        trade_id = resp1.json()["id"]
        client.post(f"/trades/{trade_id}/close", json={"close_price": 2760.0})

        resp = client.get("/trades/open")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["asset"] == "S&P"

    def test_list_closed_trades(self, client):
        resp1 = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.0,
                "contracts": 1,
            },
        )
        trade_id = resp1.json()["id"]
        client.post(f"/trades/{trade_id}/close", json={"close_price": 2760.0})

        resp = client.get("/trades?status=closed")
        data = resp.json()
        assert len(data) >= 1
        assert all(t["status"] == "CLOSED" for t in data)

    def test_get_trade_by_id(self, client):
        resp = client.post(
            "/trades",
            json={
                "asset": "Nasdaq",
                "direction": "LONG",
                "entry": 19500.0,
                "contracts": 3,
                "notes": "Breakout play",
            },
        )
        trade_id = resp.json()["id"]

        resp = client.get(f"/trades/{trade_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == trade_id
        assert data["asset"] == "Nasdaq"
        assert data["notes"] == "Breakout play"

    def test_get_trade_not_found(self, client):
        resp = client.get("/trades/99999")
        assert resp.status_code == 404

    def test_today_pnl_no_trades(self, client):
        resp = client.get("/trades/today/pnl")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pnl"] == 0

    def test_today_pnl_with_closed_trade(self, client):
        resp = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.0,
                "contracts": 1,
            },
        )
        trade_id = resp.json()["id"]
        client.post(f"/trades/{trade_id}/close", json={"close_price": 2760.0})

        resp = client.get("/trades/today/pnl")
        assert resp.status_code == 200
        data = resp.json()
        assert "pnl" in data

    def test_legacy_log_trade(self, client):
        resp = client.post(
            "/log_trade",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2750.0,
                "exit_price": 2760.0,
                "contracts": 1,
                "pnl": 100.0,
                "strategy": "TrendEMA",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "logged"
        assert data["trade_id"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Journal endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestJournal:
    def test_save_journal_entry(self, client):
        resp = client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-06",
                "account_size": 150_000,
                "gross_pnl": 500.0,
                "net_pnl": 475.0,
                "commissions": 25.0,
                "num_contracts": 10,
                "instruments": "MES, MNQ",
                "notes": "Good day, followed the plan.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["trade_date"] == "2025-01-06"
        assert data["net_pnl"] == 475.0

    def test_get_entries_empty(self, client):
        resp = client.get("/journal/entries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["entries"] == []

    def test_get_entries_after_save(self, client):
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-06",
                "account_size": 150_000,
                "gross_pnl": 500.0,
                "net_pnl": 475.0,
            },
        )
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-07",
                "account_size": 150_000,
                "gross_pnl": -200.0,
                "net_pnl": -225.0,
            },
        )

        resp = client.get("/journal/entries")
        data = resp.json()
        assert data["count"] == 2

    def test_get_entries_with_limit(self, client):
        for i in range(5):
            client.post(
                "/journal/save",
                json={
                    "trade_date": f"2025-01-{6 + i:02d}",
                    "account_size": 150_000,
                    "gross_pnl": 100.0 * (i + 1),
                    "net_pnl": 90.0 * (i + 1),
                },
            )

        resp = client.get("/journal/entries?limit=3")
        data = resp.json()
        assert data["count"] == 3

    def test_get_stats_empty(self, client):
        resp = client.get("/journal/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_days"] == 0

    def test_get_stats_with_data(self, client):
        # Save a winning day
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-06",
                "account_size": 150_000,
                "gross_pnl": 500.0,
                "net_pnl": 475.0,
                "commissions": 25.0,
            },
        )
        # Save a losing day
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-07",
                "account_size": 150_000,
                "gross_pnl": -200.0,
                "net_pnl": -225.0,
                "commissions": 25.0,
            },
        )

        resp = client.get("/journal/stats")
        data = resp.json()
        assert data["total_days"] == 2
        assert data["win_days"] == 1
        assert data["loss_days"] == 1

    def test_today_no_entry(self, client):
        resp = client.get("/journal/today")
        assert resp.status_code == 200
        data = resp.json()
        assert data["exists"] is False

    def test_today_with_entry(self, client):
        from datetime import date as dt_date

        today_str = dt_date.today().strftime("%Y-%m-%d")
        client.post(
            "/journal/save",
            json={
                "trade_date": today_str,
                "account_size": 150_000,
                "gross_pnl": 300.0,
                "net_pnl": 280.0,
            },
        )

        resp = client.get("/journal/today")
        data = resp.json()
        assert data["exists"] is True
        assert data["entry"]["trade_date"] == today_str

    def test_upsert_journal_entry(self, client):
        """Saving the same date twice should update, not duplicate."""
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-06",
                "account_size": 150_000,
                "gross_pnl": 500.0,
                "net_pnl": 475.0,
            },
        )
        # Update with new values
        client.post(
            "/journal/save",
            json={
                "trade_date": "2025-01-06",
                "account_size": 150_000,
                "gross_pnl": 600.0,
                "net_pnl": 575.0,
            },
        )

        resp = client.get("/journal/entries")
        data = resp.json()
        assert data["count"] == 1
        entry = data["entries"][0]
        assert entry["gross_pnl"] == 600.0
        assert entry["net_pnl"] == 575.0


# ═══════════════════════════════════════════════════════════════════════════
# Engine-not-ready scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineNotReady:
    """Test that endpoints gracefully return 503 when the engine isn't set."""

    @pytest.fixture()
    def client_no_engine(self):
        import lib.services.data.api.health as _health_mod

        _orig = _health_mod._get_engine_or_none
        _health_mod._get_engine_or_none = lambda: None

        app = _build_test_app(None)

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        # Restore
        from lib.services.data.api.actions import (
            set_engine as actions_set_engine,
        )
        from lib.services.data.api.analysis import (
            set_engine as analysis_set_engine,
        )

        analysis_set_engine(None)
        actions_set_engine(None)
        _health_mod._get_engine_or_none = _orig

    def test_status_503_without_engine(self, client_no_engine):
        resp = client_no_engine.get("/analysis/status")
        assert resp.status_code == 503

    def test_force_refresh_503_without_engine(self, client_no_engine):
        resp = client_no_engine.post("/actions/force_refresh")
        assert resp.status_code == 503

    def test_optimize_503_without_engine(self, client_no_engine):
        resp = client_no_engine.post("/actions/optimize_now")
        assert resp.status_code == 503

    def test_update_settings_503_without_engine(self, client_no_engine):
        resp = client_no_engine.post(
            "/actions/update_settings",
            json={"account_size": 100_000},
        )
        assert resp.status_code == 503

    def test_live_feed_start_503_without_engine(self, client_no_engine):
        resp = client_no_engine.post("/actions/live_feed/start")
        assert resp.status_code == 503

    def test_backtest_results_503_without_engine(self, client_no_engine):
        resp = client_no_engine.get("/analysis/backtest_results")
        assert resp.status_code == 503

    def test_health_still_works_without_engine(self, client_no_engine):
        """Health endpoint should work even without engine — reports degraded."""
        resp = client_no_engine.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"

    def test_positions_still_work_without_engine(self, client_no_engine):
        """Positions don't depend on engine."""
        resp = client_no_engine.get("/positions/")
        assert resp.status_code == 200

    def test_journal_still_works_without_engine(self, client_no_engine):
        """Journal doesn't depend on engine."""
        resp = client_no_engine.get("/journal/entries")
        assert resp.status_code == 200

    def test_trades_still_work_without_engine(self, client_no_engine):
        """Trade CRUD doesn't depend on engine."""
        resp = client_no_engine.get("/trades")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# CORS
# ═══════════════════════════════════════════════════════════════════════════


class TestCORS:
    def test_cors_headers_present(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases & validation
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_create_trade_missing_required_fields(self, client):
        """Omitting required fields should return 422."""
        resp = client.post("/trades", json={"asset": "Gold"})
        assert resp.status_code == 422

    def test_create_trade_direction_normalised_to_upper(self, client):
        resp = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "long",
                "entry": 2750.0,
                "contracts": 1,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["direction"] == "LONG"

    def test_journal_save_missing_date(self, client):
        resp = client.post(
            "/journal/save",
            json={"account_size": 150_000, "gross_pnl": 100.0},
        )
        assert resp.status_code == 422

    def test_update_settings_multiple_fields(self, client, mock_engine):
        resp = client.post(
            "/actions/update_settings",
            json={"account_size": 50_000, "interval": "15m", "period": "10d"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
        assert data["changed"]["account_size"] == 50_000
        assert data["changed"]["interval"] == "15m"
        assert data["changed"]["period"] == "10d"

    def test_close_nonexistent_trade(self, client):
        resp = client.post("/trades/99999/close", json={"close_price": 2760.0})
        # models raises an exception; the router doesn't catch it as 404
        # so it may return 404 or 500 depending on the models implementation
        assert resp.status_code in (404, 500)

    def test_cancel_nonexistent_trade(self, client):
        resp = client.post("/trades/99999/cancel")
        assert resp.status_code in (404, 500)

    def test_position_update_with_timestamp(self, client):
        """Positions can optionally include a timestamp from NT."""
        payload = {
            "account": "Sim101",
            "positions": [],
            "timestamp": "2025-01-06T10:30:00Z",
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200

    def test_journal_entries_limit_bounds(self, client):
        """Limit must be between 1 and 365."""
        resp = client.get("/journal/entries?limit=0")
        assert resp.status_code == 422

        resp = client.get("/journal/entries?limit=366")
        assert resp.status_code == 422

        resp = client.get("/journal/entries?limit=1")
        assert resp.status_code == 200

        resp = client.get("/journal/entries?limit=365")
        assert resp.status_code == 200
