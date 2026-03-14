"""
Tests for the Live Position API.

Covers:
  - POST /positions/update   — push positions from trading platform
  - GET  /positions/          — read current positions
  - DELETE /positions/        — clear stale positions
  - get_live_positions()      — direct cache read helper
  - Edge cases: empty positions, malformed payloads, cache expiry, etc.
  - Integration with Grok context builder (positions in market context)
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_position_cache():
    """Clear the position cache before and after each test.

    Also temporarily disables Redis so that cache_get/cache_set use the
    in-memory ``_mem_cache`` dict, which the test controls.
    """
    import lib.core.cache as _cache_mod

    # Save original state
    original_mem = dict(_cache_mod._mem_cache)
    original_redis_available = _cache_mod.REDIS_AVAILABLE
    original_r = _cache_mod._r

    # Force in-memory mode
    _cache_mod.REDIS_AVAILABLE = False
    _cache_mod._r = None
    _cache_mod._mem_cache.clear()

    yield

    # Restore
    _cache_mod._mem_cache.clear()
    _cache_mod._mem_cache.update(original_mem)
    _cache_mod.REDIS_AVAILABLE = original_redis_available
    _cache_mod._r = original_r


@pytest.fixture()
def client():
    """FastAPI test client with the positions router mounted at /positions."""
    from lib.services.data.api.positions import router as positions_router

    app = FastAPI()
    app.include_router(positions_router, prefix="/positions")
    return TestClient(app)


@pytest.fixture()
def full_client():
    """FastAPI test client with positions + health routers (for regression tests)."""
    from lib.services.data.api.health import router as health_router
    from lib.services.data.api.positions import router as positions_router

    app = FastAPI()
    app.include_router(positions_router, prefix="/positions")
    app.include_router(health_router)
    return TestClient(app)


@pytest.fixture()
def sample_payload():
    """A realistic position payload."""
    return {
        "account": "Sim101",
        "positions": [
            {
                "symbol": "MESZ5",
                "side": "Long",
                "quantity": 5,
                "avgPrice": 6045.25,
                "unrealizedPnL": 125.00,
                "lastUpdate": "2025-01-15T14:30:00Z",
            },
            {
                "symbol": "MNQZ5",
                "side": "Short",
                "quantity": 3,
                "avgPrice": 21580.00,
                "unrealizedPnL": -42.00,
                "lastUpdate": "2025-01-15T14:30:00Z",
            },
        ],
        "timestamp": "2025-01-15T14:30:00Z",
    }


@pytest.fixture()
def single_position_payload():
    """Payload with a single position."""
    return {
        "account": "Live001",
        "positions": [
            {
                "symbol": "MCLZ5",
                "side": "Long",
                "quantity": 2,
                "avgPrice": 71.45,
                "unrealizedPnL": 30.00,
            },
        ],
        "timestamp": "2025-01-15T15:00:00Z",
    }


@pytest.fixture()
def empty_payload():
    """Payload with no open positions (all closed)."""
    return {
        "account": "Sim101",
        "positions": [],
        "timestamp": "2025-01-15T16:00:00Z",
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST /positions/update
# ═══════════════════════════════════════════════════════════════════════════


class TestUpdatePositions:
    """Tests for the POST /positions/update endpoint."""

    def test_post_positions_success(self, client, sample_payload):
        """Basic successful position push."""
        resp = client.post("/positions/update", json=sample_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "received"
        assert body["positions_count"] == 2
        assert body["open_positions"] == 2
        assert body["total_unrealized_pnl"] == 83.00  # 125 + (-42)
        assert "received_at" in body

    def test_post_single_position(self, client, single_position_payload):
        """Push a single position."""
        resp = client.post("/positions/update", json=single_position_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_count"] == 1
        assert body["open_positions"] == 1
        assert body["total_unrealized_pnl"] == 30.00

    def test_post_empty_positions(self, client, empty_payload):
        """Push an empty positions list (all positions closed)."""
        resp = client.post("/positions/update", json=empty_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_count"] == 0
        assert body["open_positions"] == 0
        assert body["total_unrealized_pnl"] == 0.0

    def test_post_updates_cache(self, client, sample_payload):
        """Verify that POST actually writes to the cache."""
        from lib.services.data.api.positions import get_live_positions

        # Before: no positions
        before = get_live_positions()
        assert before["has_positions"] is False

        # Push
        client.post("/positions/update", json=sample_payload)

        # After: positions present
        after = get_live_positions()
        assert after["has_positions"] is True
        assert after["account"] == "Sim101"
        assert len(after["positions"]) == 2

    def test_post_overwrites_previous(self, client, sample_payload, single_position_payload):
        """A new POST replaces the previous positions snapshot."""
        from lib.services.data.api.positions import get_live_positions

        # First push: 2 positions
        client.post("/positions/update", json=sample_payload)
        assert len(get_live_positions()["positions"]) == 2

        # Second push: 1 position (replaces, not appends)
        client.post("/positions/update", json=single_position_payload)
        result = get_live_positions()
        assert len(result["positions"]) == 1
        assert result["account"] == "Live001"

    def test_post_negative_pnl(self, client):
        """Positions with negative unrealized PnL."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 10,
                    "avgPrice": 6100.00,
                    "unrealizedPnL": -500.00,
                },
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        assert resp.json()["total_unrealized_pnl"] == -500.00

    def test_post_zero_quantity_excluded(self, client):
        """Positions with quantity=0 are still accepted by API but counted correctly."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 0,
                    "avgPrice": 6045.25,
                    "unrealizedPnL": 0.0,
                },
                {
                    "symbol": "MNQZ5",
                    "side": "Short",
                    "quantity": 3,
                    "avgPrice": 21580.00,
                    "unrealizedPnL": -20.0,
                },
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        # Total received includes both, but open count only counts qty > 0
        assert body["positions_count"] == 2
        assert body["open_positions"] == 1

    def test_post_missing_account_fails(self, client):
        """Missing required 'account' field should fail validation."""
        payload = {
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 6000.00,
                    "unrealizedPnL": 0,
                }
            ],
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_post_missing_position_fields_fails(self, client):
        """Missing required position fields should fail validation."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    # missing side, quantity, avgPrice
                }
            ],
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 422

    def test_post_optional_fields_default(self, client):
        """Optional fields should default gracefully."""
        payload = {
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 6000.0,
                    # unrealizedPnL and lastUpdate are optional
                },
            ],
            # timestamp is optional
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions_count"] == 1
        assert body["total_unrealized_pnl"] == 0.0  # default

    def test_post_large_position_count(self, client):
        """Handle a large number of positions (e.g., 20 micro contracts)."""
        positions = []
        for i in range(20):
            positions.append(
                {
                    "symbol": f"MES{chr(65 + i % 6)}5",
                    "side": "Long" if i % 2 == 0 else "Short",
                    "quantity": i + 1,
                    "avgPrice": 6000.0 + i * 10,
                    "unrealizedPnL": (i - 10) * 25.0,
                }
            )
        payload = {
            "account": "Sim101",
            "positions": positions,
            "timestamp": "2025-01-15T14:30:00Z",
        }
        resp = client.post("/positions/update", json=payload)
        assert resp.status_code == 200
        assert resp.json()["positions_count"] == 20


# ═══════════════════════════════════════════════════════════════════════════
# GET /positions/
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPositions:
    """Tests for the GET /positions/ endpoint."""

    def test_get_no_positions(self, client):
        """GET with no cached positions returns empty response."""
        resp = client.get("/positions/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_positions"] is False
        assert body["positions"] == []
        assert body["account"] == ""
        assert body["total_unrealized_pnl"] == 0.0

    def test_get_after_push(self, client, sample_payload):
        """GET returns the positions pushed by POST."""
        client.post("/positions/update", json=sample_payload)

        resp = client.get("/positions/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_positions"] is True
        assert body["account"] == "Sim101"
        assert len(body["positions"]) == 2
        assert body["total_unrealized_pnl"] == 83.00
        assert body["received_at"] != ""

    def test_get_position_details(self, client, sample_payload):
        """Verify individual position fields are preserved."""
        client.post("/positions/update", json=sample_payload)

        resp = client.get("/positions/")
        positions = resp.json()["positions"]

        mes = next(p for p in positions if p["symbol"] == "MESZ5")
        assert mes["side"] == "Long"
        assert mes["quantity"] == 5
        assert mes["avgPrice"] == 6045.25
        assert mes["unrealizedPnL"] == 125.00

        mnq = next(p for p in positions if p["symbol"] == "MNQZ5")
        assert mnq["side"] == "Short"
        assert mnq["quantity"] == 3
        assert mnq["avgPrice"] == 21580.00
        assert mnq["unrealizedPnL"] == -42.00

    def test_get_after_clear(self, client, sample_payload):
        """GET returns empty after DELETE /positions/."""
        client.post("/positions/update", json=sample_payload)
        client.delete("/positions/")

        resp = client.get("/positions/")
        body = resp.json()
        assert body["has_positions"] is False
        assert body["positions"] == []


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /positions/
# ═══════════════════════════════════════════════════════════════════════════


class TestClearPositions:
    """Tests for the DELETE /positions/ endpoint."""

    def test_clear_when_empty(self, client):
        """DELETE on empty cache succeeds without error."""
        resp = client.delete("/positions/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

    def test_clear_removes_data(self, client, sample_payload):
        """DELETE removes cached positions."""
        from lib.services.data.api.positions import get_live_positions

        client.post("/positions/update", json=sample_payload)
        assert get_live_positions()["has_positions"] is True

        client.delete("/positions/")
        assert get_live_positions()["has_positions"] is False

    def test_clear_idempotent(self, client, sample_payload):
        """Multiple DELETEs don't cause errors."""
        client.post("/positions/update", json=sample_payload)
        client.delete("/positions/")
        client.delete("/positions/")
        client.delete("/positions/")

        resp = client.get("/positions/")
        assert resp.json()["has_positions"] is False


# ═══════════════════════════════════════════════════════════════════════════
# get_live_positions() helper
# ═══════════════════════════════════════════════════════════════════════════


class TestGetLivePositionsHelper:
    """Tests for the get_live_positions() direct cache reader."""

    def test_no_data(self):
        """Returns empty dict structure when cache is empty."""
        from lib.services.data.api.positions import get_live_positions

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []
        assert result["account"] == ""
        assert result["total_unrealized_pnl"] == 0.0
        assert result["timestamp"] == ""
        assert result["received_at"] == ""

    def test_after_push(self, client, sample_payload):
        """Returns correct data after a push."""
        from lib.services.data.api.positions import get_live_positions

        client.post("/positions/update", json=sample_payload)

        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["account"] == "Sim101"
        assert len(result["positions"]) == 2
        assert result["total_unrealized_pnl"] == 83.00

    def test_pnl_calculation(self, client):
        """Total unrealized PnL is computed correctly across positions."""
        from lib.services.data.api.positions import get_live_positions

        payload = {
            "account": "Test",
            "positions": [
                {
                    "symbol": "A",
                    "side": "Long",
                    "quantity": 1,
                    "avgPrice": 100,
                    "unrealizedPnL": 50.0,
                },
                {
                    "symbol": "B",
                    "side": "Short",
                    "quantity": 2,
                    "avgPrice": 200,
                    "unrealizedPnL": -30.0,
                },
                {
                    "symbol": "C",
                    "side": "Long",
                    "quantity": 3,
                    "avgPrice": 300,
                    "unrealizedPnL": 100.0,
                },
            ],
        }
        client.post("/positions/update", json=payload)

        result = get_live_positions()
        assert result["total_unrealized_pnl"] == 120.0  # 50 + (-30) + 100

    def test_corrupt_cache_data(self):
        """Handles corrupt cache data gracefully."""
        from lib.core.cache import cache_set
        from lib.services.data.api.positions import (
            _POSITIONS_CACHE_KEY,
            get_live_positions,
        )

        cache_set(_POSITIONS_CACHE_KEY, b"not valid json{{{", 60)

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []

    def test_partial_cache_data(self):
        """Handles cache data with missing fields gracefully."""
        from lib.core.cache import cache_set
        from lib.services.data.api.positions import (
            _POSITIONS_CACHE_KEY,
            get_live_positions,
        )

        data = json.dumps({"account": "Test"}).encode()  # missing positions key
        cache_set(_POSITIONS_CACHE_KEY, data, 60)

        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["positions"] == []
        assert result["account"] == "Test"


# ═══════════════════════════════════════════════════════════════════════════
# Health endpoint basic check
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Tests for the /health endpoint (mounted alongside positions)."""

    def test_health_returns_ok_or_degraded(self, full_client):
        """Health endpoint responds with a status field."""
        resp = full_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] in ("ok", "degraded")

    def test_health_has_components(self, full_client):
        """Health endpoint includes component status."""
        resp = full_client.get("/health")
        body = resp.json()
        assert "components" in body
        assert "redis" in body["components"]


# ═══════════════════════════════════════════════════════════════════════════
# Grok context integration
# ═══════════════════════════════════════════════════════════════════════════


class TestGrokContextIntegration:
    """Tests for live positions in Grok market context."""

    def test_context_without_positions(self):
        """Market context without positions shows 'No live positions'."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
        )
        assert ctx["positions_text"] == "No live positions"
        assert ctx["has_positions"] is False

    def test_context_with_positions(self):
        """Market context includes position details when available."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": True,
            "account": "Sim101",
            "positions": [
                {
                    "symbol": "MESZ5",
                    "side": "Long",
                    "quantity": 5,
                    "avgPrice": 6045.25,
                    "unrealizedPnL": 125.00,
                },
            ],
            "total_unrealized_pnl": 125.00,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is True
        assert "MESZ5" in ctx["positions_text"]
        assert "Long" in ctx["positions_text"]
        assert "Sim101" in ctx["positions_text"]
        assert "125" in ctx["positions_text"]

    def test_context_with_negative_pnl(self):
        """Market context formats negative PnL correctly."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": True,
            "account": "Live001",
            "positions": [
                {
                    "symbol": "MCLZ5",
                    "side": "Short",
                    "quantity": 2,
                    "avgPrice": 72.00,
                    "unrealizedPnL": -150.00,
                },
            ],
            "total_unrealized_pnl": -150.00,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is True
        assert "🔴" in ctx["positions_text"]
        assert "-150" in ctx["positions_text"]

    def test_context_with_empty_positions(self):
        """Empty positions list means no positions flag."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        live_pos = {
            "has_positions": False,
            "account": "Sim101",
            "positions": [],
            "total_unrealized_pnl": 0.0,
        }

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=live_pos,
        )
        assert ctx["has_positions"] is False
        assert ctx["positions_text"] == "No live positions"

    def test_context_positions_none(self):
        """Passing None for live_positions is handled gracefully."""
        from lib.integrations.grok_helper import format_market_context

        engine_mock = MagicMock()
        engine_mock.get_backtest_results.return_value = []

        ctx = format_market_context(
            engine=engine_mock,
            scanner_df=None,
            account_size=50000,
            risk_dollars=500,
            max_contracts=10,
            contract_specs={},
            selected_assets=[],
            live_positions=None,
        )
        assert ctx["has_positions"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic model validation
# ═══════════════════════════════════════════════════════════════════════════


class TestModelValidation:
    """Tests for Pydantic request/response models."""

    def test_nt_position_model(self):
        """NTPosition model accepts valid data."""
        from lib.services.data.api.positions import NTPosition

        pos = NTPosition(
            symbol="MESZ5",
            side="Long",
            quantity=5,
            avgPrice=6045.25,
            unrealizedPnL=125.0,
            instrument=None,
            tickSize=None,
            pointValue=None,
            lastUpdate="2025-01-15T14:30:00Z",
        )
        assert pos.symbol == "MESZ5"
        assert pos.side == "Long"
        assert pos.quantity == 5
        assert pos.avgPrice == 6045.25

    def test_nt_position_defaults(self):
        """NTPosition model defaults for optional fields."""
        from lib.services.data.api.positions import NTPosition

        pos = NTPosition(
            symbol="MESZ5",
            side="Long",
            quantity=1,
            avgPrice=6000.0,
            unrealizedPnL=0.0,
            instrument=None,
            tickSize=None,
            pointValue=None,
            lastUpdate=None,
        )
        assert pos.unrealizedPnL == 0.0
        assert pos.lastUpdate is None


# ═══════════════════════════════════════════════════════════════════════════
# Existing endpoints still work (regression)
# ═══════════════════════════════════════════════════════════════════════════


class TestExistingEndpointsRegression:
    """Verify that existing endpoints still respond after position API addition."""

    def test_health_still_works(self, full_client):
        resp = full_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    def test_positions_get_still_works(self, full_client):
        """GET /positions/ returns a valid response."""
        resp = full_client.get("/positions/")
        assert resp.status_code == 200
        body = resp.json()
        assert "has_positions" in body
        assert "positions" in body


# ═══════════════════════════════════════════════════════════════════════════
# Rapid-fire updates (simulating rapid platform behavior)
# ═══════════════════════════════════════════════════════════════════════════


class TestRapidUpdates:
    """Simulate rapid position updates like a trading platform would send."""

    def test_rapid_position_updates(self, client):
        """Multiple rapid POSTs don't corrupt data."""
        from lib.services.data.api.positions import get_live_positions

        for i in range(10):
            payload = {
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": float(i * 10),
                    },
                ],
                "timestamp": f"2025-01-15T14:30:{i:02d}Z",
            }
            resp = client.post("/positions/update", json=payload)
            assert resp.status_code == 200

        # Final state should reflect the last update
        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["total_unrealized_pnl"] == 90.0  # last: 9 * 10

    def test_position_lifecycle(self, client):
        """Simulate: no positions → open → update PnL → close → clear."""
        from lib.services.data.api.positions import get_live_positions

        # 1. No positions initially
        assert get_live_positions()["has_positions"] is False

        # 2. Open a position
        client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": 0.0,
                    },
                ],
            },
        )
        result = get_live_positions()
        assert result["has_positions"] is True
        assert result["total_unrealized_pnl"] == 0.0

        # 3. PnL updates
        client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MESZ5",
                        "side": "Long",
                        "quantity": 5,
                        "avgPrice": 6045.25,
                        "unrealizedPnL": 250.0,
                    },
                ],
            },
        )
        assert get_live_positions()["total_unrealized_pnl"] == 250.0

        # 4. Position closed (empty list)
        client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": [],
            },
        )
        result = get_live_positions()
        assert result["has_positions"] is False
        assert result["total_unrealized_pnl"] == 0.0

        # 5. Clean up
        client.delete("/positions/")
        assert get_live_positions()["has_positions"] is False
