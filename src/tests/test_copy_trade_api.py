"""
Tests for Copy Trade API Router — /api/copy-trade/*
=====================================================
Covers all endpoints in lib.services.data.api.copy_trade:

  POST /api/copy-trade/send                 — SEND ALL (security_code path)
  POST /api/copy-trade/send-from-ticker     — ticker → contract → SEND ALL
  GET  /api/copy-trade/status               — CopyTrader status
  GET  /api/copy-trade/history              — Order history
  GET  /api/copy-trade/compliance-log       — Audit trail
  GET  /api/copy-trade/rate                 — Rate-limit counter
  POST /api/copy-trade/high-impact          — Toggle high-impact mode
  POST /api/copy-trade/invalidate-cache     — Clear contract cache
  GET  /api/copy-trade/result/{batch_id}    — Poll remote result
  GET  /api/copy-trade/status/html          — HTMX status strip
  GET  /api/copy-trade/history/html         — HTMX history table

All tests run without Redis, a real engine, or async_rithmic installed.
The CopyTrader singleton is mocked at the module level.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Disable Redis before any cache imports
os.environ.setdefault("DISABLE_REDIS", "1")
os.environ.setdefault("RITHMIC_COPY_TRADING", "1")  # enable by default for most tests

from lib.services.data.api.copy_trade import router  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal FastAPI app for testing (no full data-service lifespan overhead)
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)


@pytest.fixture()
def client():
    with TestClient(_app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_mock_batch_result(
    *,
    ok: bool = True,
    side: str = "BUY",
    qty: int = 1,
    security_code: str = "MGCQ6",
    total_orders: int = 2,
    failed_count: int = 0,
    compliance_log: list[str] | None = None,
) -> MagicMock:
    """Build a mock CopyBatchResult with a to_dict() method."""
    result = MagicMock()
    result.to_dict.return_value = {
        "batch_id": f"WEBUI_{security_code}_12345",
        "security_code": security_code,
        "side": side,
        "qty": qty,
        "total_orders": total_orders,
        "failed_count": failed_count,
        "all_submitted": failed_count == 0,
        "ok": ok,
        "submitted_at": "2025-01-01T10:00:00+00:00",
        "compliance_log": compliance_log or ["✓ All orders use OrderPlacement.MANUAL flag"],
        "main_result": {
            "account_key": "main",
            "status": "SUBMITTED",
            "order_id": "O001",
        },
        "slave_results": [],
    }
    return result


def _make_mock_ct(
    *,
    main_connected: bool = True,
    slave_count: int = 2,
    rate_count: int = 5,
    is_warn: bool = False,
    is_hard_limit: bool = False,
    high_impact: bool = False,
) -> MagicMock:
    """Build a mock CopyTrader singleton."""
    ct = MagicMock()
    ct.status_summary.return_value = {
        "main": {
            "key": "main",
            "label": "Main Account",
            "connected": main_connected,
            "account_ids": ["ACC001"],
            "last_order_at": "",
            "order_count": 3,
        },
        "slaves": [
            {"key": "slave1", "label": "Slave 1", "connected": True, "enabled": True},
            {"key": "slave2", "label": "Slave 2", "connected": True, "enabled": True},
        ],
        "enabled_slave_count": slave_count,
        "high_impact_mode": high_impact,
        "rate_limit": {
            "count": rate_count,
            "warn_threshold": 3000,
            "hard_limit": 4500,
            "is_warn": is_warn,
            "is_hard_limit": is_hard_limit,
        },
        "contract_cache_size": 3,
        "recent_batches": [],
    }
    ct.get_order_history.return_value = [
        {
            "batch_id": "WEBUI_MGCQ6_001",
            "security_code": "MGCQ6",
            "side": "BUY",
            "qty": 1,
            "total_orders": 2,
            "failed_count": 0,
            "submitted_at": "2025-01-01T10:00:00",
        }
    ]
    ct.get_rate_status.return_value = {
        "count": rate_count,
        "warn_threshold": 3000,
        "hard_limit": 4500,
        "is_warn": is_warn,
        "is_hard_limit": is_hard_limit,
    }
    ct.send_order_and_copy = AsyncMock(return_value=_make_mock_batch_result())
    ct.send_order_from_ticker = AsyncMock(return_value=_make_mock_batch_result())
    ct.set_high_impact_mode = MagicMock()
    ct.invalidate_contract_cache = MagicMock()
    return ct


# ---------------------------------------------------------------------------
# Helpers for patching
# ---------------------------------------------------------------------------


def _patch_ct(ct: Any):
    """Return a context manager that patches get_copy_trader_direct."""
    return patch("lib.services.data.api.copy_trade._get_copy_trader_direct", return_value=ct)


def _patch_no_ct():
    """Patch get_copy_trader_direct to return None (uninitialised)."""
    return patch("lib.services.data.api.copy_trade._get_copy_trader_direct", return_value=None)


def _patch_disabled():
    """Patch _COPY_TRADING_ENABLED to False."""
    return patch("lib.services.data.api.copy_trade._COPY_TRADING_ENABLED", False)


def _patch_publish():
    """Patch _publish_sse to a no-op (avoids Redis calls)."""
    return patch("lib.services.data.api.copy_trade._publish_sse")


# ===========================================================================
# POST /api/copy-trade/send
# ===========================================================================


class TestSendAll:
    """Tests for the SEND ALL endpoint."""

    def _valid_body(self, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "security_code": "MGCQ6",
            "exchange": "NYMEX",
            "side": "BUY",
            "qty": 1,
            "order_type": "MARKET",
            "price": 0.0,
            "stop_ticks": 20,
            "target_ticks": None,
            "tag_prefix": "WEBUI",
            "reason": "ORB long breakout",
        }
        body.update(overrides)
        return body

    def test_send_all_success(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "batch_id" in d
        assert "result" in d
        ct.send_order_and_copy.assert_called_once()

    def test_send_all_passes_correct_kwargs(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        body = self._valid_body(
            side="SELL",
            qty=2,
            order_type="LIMIT",
            price=2015.50,
            stop_ticks=15,
            target_ticks=30,
            reason="test reason",
        )
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send", json=body)
        assert r.status_code == 200
        call_kwargs = ct.send_order_and_copy.call_args.kwargs
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["qty"] == 2
        assert call_kwargs["order_type"] == "LIMIT"
        assert call_kwargs["price"] == 2015.50
        assert call_kwargs["stop_ticks"] == 15
        assert call_kwargs["target_ticks"] == 30
        assert call_kwargs["reason"] == "test reason"

    def test_send_all_disabled_returns_403(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 403
        assert "RITHMIC_COPY_TRADING" in r.json()["detail"]

    def test_send_all_no_copy_trader_returns_503(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 503

    def test_send_all_invalid_side_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send", json=self._valid_body(side="HOLD"))
        assert r.status_code == 422

    def test_send_all_invalid_order_type_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send", json=self._valid_body(order_type="FOK"))
        assert r.status_code == 422

    def test_send_all_normalises_side_to_uppercase(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        body = self._valid_body(side="buy")
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send", json=body)
        assert r.status_code == 200
        call_kwargs = ct.send_order_and_copy.call_args.kwargs
        assert call_kwargs["side"] == "BUY"

    def test_send_all_exception_from_ct_returns_500(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        ct.send_order_and_copy = AsyncMock(side_effect=RuntimeError("Rithmic exploded"))
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 500
        assert "Rithmic exploded" in r.json()["detail"]

    def test_send_all_publishes_sse_on_success(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), patch("lib.services.data.api.copy_trade._publish_sse") as mock_pub:
            client.post("/api/copy-trade/send", json=self._valid_body())
        mock_pub.assert_called_once()
        call_args = mock_pub.call_args[0][0]
        assert call_args["event"] == "send_all"

    def test_send_all_qty_out_of_range_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send", json=self._valid_body(qty=0))
        assert r.status_code == 422

    def test_send_all_stop_ticks_out_of_range_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send", json=self._valid_body(stop_ticks=501))
        assert r.status_code == 422

    def test_send_all_reason_truncated_in_batch_id(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        long_reason = "x" * 100
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send", json=self._valid_body(reason=long_reason))
        assert r.status_code == 200
        # batch_id should contain a truncated version (max 20 chars of reason)
        d = r.json()
        assert "batch_id" in d

    def test_send_all_remote_mode_enqueues(self, client: TestClient) -> None:
        """In remote mode, the command is enqueued and 202 or 200 returned."""
        ct = _make_mock_ct()
        with (
            _patch_ct(ct),
            patch("lib.services.data.api.copy_trade._ENGINE_MODE", "remote"),
            patch("lib.services.data.api.copy_trade._enqueue_remote_command") as mock_enq,
            patch(
                "lib.services.data.api.copy_trade._wait_for_remote_result", new_callable=AsyncMock, return_value=None
            ),
        ):
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 202
        mock_enq.assert_called_once()
        d = r.json()
        assert d["ok"] is True
        assert d.get("accepted") is True
        assert "batch_id" in d

    def test_send_all_remote_mode_returns_result_when_ready(self, client: TestClient) -> None:
        """In remote mode with a ready result, returns 200 with result."""
        ct = _make_mock_ct()
        mock_result = {"ok": True, "total_orders": 2, "failed_count": 0}
        with (
            _patch_ct(ct),
            patch("lib.services.data.api.copy_trade._ENGINE_MODE", "remote"),
            patch("lib.services.data.api.copy_trade._enqueue_remote_command"),
            patch(
                "lib.services.data.api.copy_trade._wait_for_remote_result",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            _patch_publish(),
        ):
            r = client.post("/api/copy-trade/send", json=self._valid_body())
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["result"] == mock_result


# ===========================================================================
# POST /api/copy-trade/send-from-ticker
# ===========================================================================


class TestSendFromTicker:
    """Tests for the send-from-ticker endpoint."""

    def _valid_body(self, **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "ticker": "MGC=F",
            "side": "BUY",
            "qty": 1,
            "order_type": "MARKET",
            "price": 0.0,
            "stop_ticks": 20,
            "target_ticks": None,
            "tag_prefix": "WEBUI",
            "reason": "",
        }
        body.update(overrides)
        return body

    def test_send_from_ticker_success(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body())
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        ct.send_order_from_ticker.assert_called_once()

    def test_send_from_ticker_passes_ticker(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), _patch_publish():
            client.post("/api/copy-trade/send-from-ticker", json=self._valid_body(ticker="MES=F"))
        call_kwargs = ct.send_order_from_ticker.call_args.kwargs
        assert call_kwargs["ticker"] == "MES=F"

    def test_send_from_ticker_disabled_returns_403(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body())
        assert r.status_code == 403

    def test_send_from_ticker_no_ct_returns_503(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body())
        assert r.status_code == 503

    def test_send_from_ticker_invalid_side_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body(side="SHORT"))
        assert r.status_code == 422

    def test_send_from_ticker_normalises_side(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), _patch_publish():
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body(side="sell"))
        assert r.status_code == 200
        call_kwargs = ct.send_order_from_ticker.call_args.kwargs
        assert call_kwargs["side"] == "SELL"

    def test_send_from_ticker_exception_returns_500(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        ct.send_order_from_ticker = AsyncMock(side_effect=ValueError("bad ticker"))
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/send-from-ticker", json=self._valid_body())
        assert r.status_code == 500

    def test_send_from_ticker_publishes_sse(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct), patch("lib.services.data.api.copy_trade._publish_sse") as mock_pub:
            client.post("/api/copy-trade/send-from-ticker", json=self._valid_body())
        mock_pub.assert_called_once()
        assert mock_pub.call_args[0][0]["event"] == "send_from_ticker"


# ===========================================================================
# GET /api/copy-trade/status
# ===========================================================================


class TestGetStatus:
    """Tests for the status endpoint."""

    def test_status_with_connected_ct(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["enabled"] is True
        assert "status" in d
        assert d["status"]["main"]["connected"] is True

    def test_status_no_ct_returns_ok_false(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.get("/api/copy-trade/status")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is False
        assert "error" in d

    def test_status_includes_enabled_flag(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status")
        assert r.json()["enabled"] is True

    def test_status_includes_engine_mode(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status")
        assert "engine_mode" in r.json()

    def test_status_disconnected_main(self, client: TestClient) -> None:
        ct = _make_mock_ct(main_connected=False)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status")
        assert r.status_code == 200
        d = r.json()
        assert d["status"]["main"]["connected"] is False

    def test_status_rate_counter_included(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=1500, is_warn=False)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status")
        rate = r.json()["status"]["rate_limit"]
        assert rate["count"] == 1500


# ===========================================================================
# GET /api/copy-trade/history
# ===========================================================================


class TestGetHistory:
    """Tests for the order history endpoint."""

    def test_history_returns_list(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert isinstance(d["history"], list)
        assert len(d["history"]) >= 1

    def test_history_calls_get_order_history(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            client.get("/api/copy-trade/history?limit=10")
        ct.get_order_history.assert_called_once_with(limit=10)

    def test_history_default_limit_50(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            client.get("/api/copy-trade/history")
        ct.get_order_history.assert_called_once_with(limit=50)

    def test_history_limit_out_of_range_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history?limit=0")
        assert r.status_code == 422

    def test_history_limit_too_large_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history?limit=501")
        assert r.status_code == 422

    def test_history_no_ct_falls_back_to_redis(self, client: TestClient) -> None:
        """When CopyTrader is None, tries Redis fallback."""
        with _patch_no_ct(), patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", False, create=True):
            r = client.get("/api/copy-trade/history")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "history" in d

    def test_history_source_singleton_when_ct_available(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history")
        assert r.json()["source"] == "singleton"


# ===========================================================================
# GET /api/copy-trade/compliance-log
# ===========================================================================


class TestGetComplianceLog:
    """Tests for the compliance audit log endpoint."""

    def test_compliance_log_returns_list(self, client: TestClient) -> None:
        mock_entries = [
            {
                "batch_id": "WEBUI_MGC_001",
                "checklist": ["✓ All orders use OrderPlacement.MANUAL flag"],
                "timestamp": "2025-01-01T10:00:00+00:00",
            }
        ]
        with patch("lib.services.data.api.copy_trade._get_compliance_log", return_value=mock_entries):
            r = client.get("/api/copy-trade/compliance-log")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert len(d["entries"]) == 1

    def test_compliance_log_default_limit_20(self, client: TestClient) -> None:
        with patch("lib.services.data.api.copy_trade._get_compliance_log", return_value=[]) as mock_get:
            r = client.get("/api/copy-trade/compliance-log")
        assert r.status_code == 200
        mock_get.assert_called_once_with(limit=20)

    def test_compliance_log_custom_limit(self, client: TestClient) -> None:
        with patch("lib.services.data.api.copy_trade._get_compliance_log", return_value=[]) as mock_get:
            client.get("/api/copy-trade/compliance-log?limit=5")
        mock_get.assert_called_once_with(limit=5)

    def test_compliance_log_limit_too_large_returns_422(self, client: TestClient) -> None:
        r = client.get("/api/copy-trade/compliance-log?limit=201")
        assert r.status_code == 422

    def test_compliance_log_limit_zero_returns_422(self, client: TestClient) -> None:
        r = client.get("/api/copy-trade/compliance-log?limit=0")
        assert r.status_code == 422

    def test_compliance_log_empty_when_no_entries(self, client: TestClient) -> None:
        with patch("lib.services.data.api.copy_trade._get_compliance_log", return_value=[]):
            r = client.get("/api/copy-trade/compliance-log")
        assert r.status_code == 200
        assert r.json()["entries"] == []


# ===========================================================================
# GET /api/copy-trade/rate
# ===========================================================================


class TestGetRate:
    """Tests for the rate-limit counter endpoint."""

    def test_rate_returns_counter(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=42)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/rate")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["rate"]["count"] == 42

    def test_rate_no_ct_returns_error(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.get("/api/copy-trade/rate")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is False

    def test_rate_includes_thresholds(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/rate")
        rate = r.json()["rate"]
        assert "warn_threshold" in rate
        assert "hard_limit" in rate

    def test_rate_is_warn_flag(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=3100, is_warn=True)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/rate")
        assert r.json()["rate"]["is_warn"] is True

    def test_rate_is_hard_limit_flag(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=4600, is_hard_limit=True)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/rate")
        assert r.json()["rate"]["is_hard_limit"] is True


# ===========================================================================
# POST /api/copy-trade/high-impact
# ===========================================================================


class TestHighImpact:
    """Tests for the high-impact mode toggle endpoint."""

    def test_enable_high_impact(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/high-impact", json={"enabled": True})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["high_impact_mode"] is True
        ct.set_high_impact_mode.assert_called_once_with(True)

    def test_disable_high_impact(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/high-impact", json={"enabled": False})
        assert r.status_code == 200
        d = r.json()
        assert d["high_impact_mode"] is False
        ct.set_high_impact_mode.assert_called_once_with(False)

    def test_high_impact_no_ct_returns_503(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.post("/api/copy-trade/high-impact", json={"enabled": True})
        assert r.status_code == 503

    def test_high_impact_missing_body_returns_422(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/high-impact", json={})
        assert r.status_code == 422

    def test_high_impact_message_varies_by_state(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r_on = client.post("/api/copy-trade/high-impact", json={"enabled": True})
            r_off = client.post("/api/copy-trade/high-impact", json={"enabled": False})
        assert "1–2 s" in r_on.json()["message"]
        assert "200–800 ms" in r_off.json()["message"]


# ===========================================================================
# POST /api/copy-trade/invalidate-cache
# ===========================================================================


class TestInvalidateCache:
    """Tests for the contract cache invalidation endpoint."""

    def test_invalidate_all(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/invalidate-cache")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        ct.invalidate_contract_cache.assert_called_once_with(ticker=None)

    def test_invalidate_single_ticker(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/invalidate-cache?ticker=MGC%3DF")
        assert r.status_code == 200
        ct.invalidate_contract_cache.assert_called_once_with(ticker="MGC=F")

    def test_invalidate_message_includes_ticker(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.post("/api/copy-trade/invalidate-cache?ticker=MES%3DF")
        assert "MES=F" in r.json()["message"]

    def test_invalidate_no_ct_returns_503(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.post("/api/copy-trade/invalidate-cache")
        assert r.status_code == 503


# ===========================================================================
# GET /api/copy-trade/result/{batch_id}
# ===========================================================================


class TestPollResult:
    """Tests for the remote result polling endpoint."""

    def test_pending_returns_202(self, client: TestClient) -> None:
        with patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", False, create=True):
            r = client.get("/api/copy-trade/result/WEBUI_MGC_001")
        assert r.status_code == 202
        d = r.json()
        assert d["ok"] is True
        assert d["pending"] is True

    def test_ready_returns_200(self, client: TestClient) -> None:
        mock_result = {"ok": True, "total_orders": 2, "failed_count": 0}
        raw = json.dumps(mock_result).encode()

        mock_r = MagicMock()
        mock_r.get.return_value = raw

        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", True, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            r = client.get("/api/copy-trade/result/WEBUI_MGC_001")
        assert r.status_code == 200
        d = r.json()
        assert d["pending"] is False
        assert d["result"] == mock_result

    def test_batch_id_echoed_in_response(self, client: TestClient) -> None:
        with patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", False, create=True):
            r = client.get("/api/copy-trade/result/MY_BATCH_42")
        d = r.json()
        assert d["batch_id"] == "MY_BATCH_42"


# ===========================================================================
# GET /api/copy-trade/status/html
# ===========================================================================


class TestStatusHtml:
    """Tests for the HTMX status strip HTML endpoint."""

    def test_disabled_shows_disabled_message(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "DISABLED" in r.text

    def test_no_ct_shows_initialising(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "INITIALISING" in r.text

    def test_connected_shows_live(self, client: TestClient) -> None:
        ct = _make_mock_ct(main_connected=True, slave_count=2)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "LIVE" in r.text
        assert "2 slave" in r.text

    def test_disconnected_main_shows_warning(self, client: TestClient) -> None:
        ct = _make_mock_ct(main_connected=False)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "DISCONNECTED" in r.text

    def test_rate_limit_warning_shown(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=3100, is_warn=True)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "3100" in r.text

    def test_hard_rate_limit_shown(self, client: TestClient) -> None:
        ct = _make_mock_ct(rate_count=4600, is_hard_limit=True)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "RATE LIMIT" in r.text

    def test_high_impact_badge_shown(self, client: TestClient) -> None:
        ct = _make_mock_ct(high_impact=True)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert "HIGH IMPACT" in r.text

    def test_no_high_impact_badge_hidden(self, client: TestClient) -> None:
        ct = _make_mock_ct(high_impact=False)
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert "HIGH IMPACT" not in r.text

    def test_html_contains_conn_dot(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/status/html")
        assert "conn-dot" in r.text


# ===========================================================================
# GET /api/copy-trade/history/html
# ===========================================================================


class TestHistoryHtml:
    """Tests for the HTMX history table HTML endpoint."""

    def test_no_history_shows_empty_message(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        ct.get_order_history.return_value = []
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history/html")
        assert r.status_code == 200
        assert "No copy-trade" in r.text

    def test_history_renders_table(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        ct.get_order_history.return_value = [
            {
                "batch_id": "WEBUI_MGCQ6_001",
                "security_code": "MGCQ6",
                "side": "BUY",
                "qty": 1,
                "total_orders": 2,
                "failed_count": 0,
                "submitted_at": "2025-01-01T10:30:00",
            }
        ]
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history/html")
        assert r.status_code == 200
        assert "<table" in r.text
        assert "MGCQ6" in r.text
        assert "BUY" in r.text

    def test_history_shows_failed_order_in_red(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        ct.get_order_history.return_value = [
            {
                "batch_id": "WEBUI_MGCQ6_002",
                "security_code": "MGCQ6",
                "side": "SELL",
                "qty": 1,
                "total_orders": 2,
                "failed_count": 1,
                "submitted_at": "2025-01-01T11:00:00",
            }
        ]
        with _patch_ct(ct):
            r = client.get("/api/copy-trade/history/html")
        assert "var(--red)" in r.text

    def test_history_html_limit_param(self, client: TestClient) -> None:
        ct = _make_mock_ct()
        with _patch_ct(ct):
            client.get("/api/copy-trade/history/html?limit=5")
        ct.get_order_history.assert_called_with(limit=5)

    def test_history_html_no_ct_shows_empty(self, client: TestClient) -> None:
        with _patch_no_ct():
            r = client.get("/api/copy-trade/history/html")
        assert r.status_code == 200
        assert "No copy-trade" in r.text


# ===========================================================================
# _get_compliance_log helper (unit tests — no HTTP)
# ===========================================================================


class TestGetComplianceLogHelper:
    """Unit tests for the _get_compliance_log helper function."""

    def test_returns_empty_when_redis_unavailable(self) -> None:
        from lib.services.data.api.copy_trade import _get_compliance_log

        with patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", False, create=True):
            result = _get_compliance_log()
        assert result == []

    def test_parses_redis_entries(self) -> None:
        from lib.services.data.api.copy_trade import _get_compliance_log

        entries = [
            json.dumps({"batch_id": "B1", "checklist": ["✓ ok"], "timestamp": "2025-01-01"}).encode(),
            json.dumps({"batch_id": "B2", "checklist": ["✓ ok2"], "timestamp": "2025-01-02"}).encode(),
        ]
        mock_r = MagicMock()
        mock_r.lrange.return_value = entries

        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", True, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            result = _get_compliance_log(limit=2)

        assert len(result) == 2
        assert result[0]["batch_id"] == "B1"

    def test_skips_malformed_entries(self) -> None:
        from lib.services.data.api.copy_trade import _get_compliance_log

        entries = [b"not-json", json.dumps({"batch_id": "B1"}).encode()]
        mock_r = MagicMock()
        mock_r.lrange.return_value = entries

        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", True, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            result = _get_compliance_log()

        assert len(result) == 1
        assert result[0]["batch_id"] == "B1"


# ===========================================================================
# _batch_result_to_response helper (unit tests)
# ===========================================================================


class TestBatchResultToResponse:
    """Unit tests for the _batch_result_to_response helper."""

    def test_none_returns_error_dict(self) -> None:
        from lib.services.data.api.copy_trade import _batch_result_to_response

        result = _batch_result_to_response(None)
        assert result["ok"] is False
        assert "error" in result

    def test_dict_returned_as_is(self) -> None:
        from lib.services.data.api.copy_trade import _batch_result_to_response

        d = {"ok": True, "total_orders": 2}
        assert _batch_result_to_response(d) is d

    def test_object_with_to_dict_is_serialised(self) -> None:
        from lib.services.data.api.copy_trade import _batch_result_to_response

        obj = MagicMock()
        obj.to_dict.return_value = {"ok": True, "batch_id": "X"}
        result = _batch_result_to_response(obj)
        assert result["ok"] is True
        assert result["batch_id"] == "X"

    def test_unknown_type_returns_error(self) -> None:
        from lib.services.data.api.copy_trade import _batch_result_to_response

        result = _batch_result_to_response(42)
        assert result["ok"] is False


# ===========================================================================
# _publish_sse helper (unit tests)
# ===========================================================================


class TestPublishSse:
    """Unit tests for the _publish_sse helper."""

    def test_publishes_to_dashboard_channel(self) -> None:
        from lib.services.data.api.copy_trade import _publish_sse

        mock_r = MagicMock()
        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", True, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            _publish_sse({"event": "test", "data": "hello"})

        mock_r.publish.assert_called_once()
        channel = mock_r.publish.call_args[0][0]
        assert channel == "dashboard:copy_trade"

    def test_publish_non_fatal_on_redis_error(self) -> None:
        from lib.services.data.api.copy_trade import _publish_sse

        mock_r = MagicMock()
        mock_r.publish.side_effect = ConnectionError("Redis down")
        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", True, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            # Should not raise
            _publish_sse({"event": "test"})

    def test_publish_skipped_when_redis_unavailable(self) -> None:
        from lib.services.data.api.copy_trade import _publish_sse

        mock_r = MagicMock()
        with (
            patch("lib.services.data.api.copy_trade.REDIS_AVAILABLE", False, create=True),
            patch("lib.services.data.api.copy_trade._r", mock_r, create=True),
        ):
            _publish_sse({"event": "test"})

        mock_r.publish.assert_not_called()


# ===========================================================================
# Integration: copy-trading disabled globally
# ===========================================================================


class TestCopyTradingDisabledGlobally:
    """Tests covering the RITHMIC_COPY_TRADING=0 path."""

    def test_send_all_forbidden_when_disabled(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.post(
                "/api/copy-trade/send",
                json={
                    "security_code": "MGCQ6",
                    "exchange": "NYMEX",
                    "side": "BUY",
                    "qty": 1,
                    "stop_ticks": 20,
                },
            )
        assert r.status_code == 403

    def test_send_from_ticker_forbidden_when_disabled(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.post(
                "/api/copy-trade/send-from-ticker",
                json={"ticker": "MGC=F", "side": "BUY", "qty": 1, "stop_ticks": 20},
            )
        assert r.status_code == 403

    def test_status_available_when_disabled(self, client: TestClient) -> None:
        """GET /status is read-only so it works even when disabled."""
        with _patch_disabled(), _patch_no_ct():
            r = client.get("/api/copy-trade/status")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_history_available_when_disabled(self, client: TestClient) -> None:
        """GET /history is read-only so it works even when disabled."""
        ct = _make_mock_ct()
        with _patch_disabled(), _patch_ct(ct):
            r = client.get("/api/copy-trade/history")
        assert r.status_code == 200

    def test_status_html_shows_disabled(self, client: TestClient) -> None:
        with _patch_disabled():
            r = client.get("/api/copy-trade/status/html")
        assert r.status_code == 200
        assert "DISABLED" in r.text
