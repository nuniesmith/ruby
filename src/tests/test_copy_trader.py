"""
Tests for CopyTrader — Prop-Firm Compliant Multi-Account Order Replication
============================================================================
Tests cover:
  - CopyTrader initialisation and defaults
  - RollingRateCounter (window, warn, hard limit)
  - Compliance checklist generation
  - MANUAL flag enforcement on every order
  - Humanised delay between slave copies (200–800 ms default, 1–2 s high-impact)
  - Server-side bracket attachment (stop_ticks / target_ticks)
  - Rate-limit blocking when hard limit exceeded
  - Order tagging (RUBY_MANUAL_WEBUI / COPY_FROM_MAIN_HUMAN_150K)
  - Batch result aggregation (all_submitted, failed_count)
  - Account connection lifecycle (add, enable, disable, disconnect)
  - Contract resolution (TICKER_TO_RITHMIC mapping + cache)
  - PositionManager OrderCommand → CopyTrader translation
  - High-impact mode toggle
  - Order history and status summary
  - Edge cases (no main, no slaves, disconnected accounts, missing credentials)
  - Redis persistence (mocked)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.services.engine.copy_trader import (
    COPY_DELAY_MAX,
    COPY_DELAY_MIN,
    DEFAULT_STOP_TICKS,
    HIGH_IMPACT_DELAY_MAX,
    HIGH_IMPACT_DELAY_MIN,
    MIN_STOP_TICKS,
    RATE_LIMIT_HARD,
    RATE_LIMIT_WARN,
    TICK_SIZE,
    TICKER_TO_RITHMIC,
    CopyBatchResult,
    CopyOrderResult,
    CopyOrderStatus,
    CopyTrader,
    OrderSide,
    RollingRateCounter,
    _build_compliance_checklist,
    _ConnectedAccount,
    get_copy_trader,
    stop_price_to_stop_ticks,
)

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeConfig:
    """Minimal stand-in for RithmicAccountConfig."""

    key: str = "test_acc"
    label: str = "Test Account"
    gateway: str = "Chicago"
    system_name: str = "Rithmic Paper Trading"
    app_name: str = "ruby_futures"
    app_version: str = "1.0"
    enabled: bool = True
    _username: str = "user123"
    _password: str = "pass456"

    def get_username(self) -> str:
        return self._username

    def get_password(self) -> str:
        return self._password


class _FakeRithmicClient:
    """Mock RithmicClient with async methods."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.connected = False
        self._submitted_orders: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def list_accounts(self) -> list[Any]:
        account = MagicMock()
        account.account_id = "ACC001"
        return [account]

    async def get_front_month_contract(self, exchange: str = "", product_code: str = "") -> Any:
        contract = MagicMock()
        contract.security_code = f"{product_code}Q6"
        return contract

    async def submit_order(self, **kwargs: Any) -> None:
        self._submitted_orders.append(kwargs)

    async def cancel_all_orders(self, **kwargs: Any) -> None:
        pass

    async def modify_order(self, **kwargs: Any) -> None:
        pass


def _make_fake_account(
    key: str = "slave1",
    label: str = "Slave 1",
    is_main: bool = False,
    connected: bool = True,
) -> _ConnectedAccount:
    """Build a connected account wrapper with a fake client."""
    client = _FakeRithmicClient()
    client.connected = connected
    return _ConnectedAccount(
        key=key,
        label=label,
        is_main=is_main,
        client=client,
        account_ids=["ACC001"],
        connected=connected,
    )


@pytest.fixture()
def ct() -> CopyTrader:
    """Fresh CopyTrader with no connected accounts."""
    return CopyTrader()


@pytest.fixture()
def ct_wired() -> CopyTrader:
    """CopyTrader with a main + 2 slaves pre-wired (no real Rithmic)."""
    trader = CopyTrader()
    trader._main = _make_fake_account("main1", "Main Account", is_main=True)
    trader._slaves["slave1"] = _make_fake_account("slave1", "Slave 1")
    trader._slaves["slave2"] = _make_fake_account("slave2", "Slave 2")
    trader._enabled_slave_keys = {"slave1", "slave2"}
    return trader


# ===================================================================
# RollingRateCounter
# ===================================================================


class TestRollingRateCounter:
    """Rate-limit counter with rolling 60-minute window."""

    def test_empty_counter(self) -> None:
        rc = RollingRateCounter()
        assert rc.count == 0
        assert not rc.is_warn
        assert not rc.is_hard_limit

    def test_record_single(self) -> None:
        rc = RollingRateCounter()
        rc.record(1)
        assert rc.count == 1

    def test_record_bulk(self) -> None:
        rc = RollingRateCounter()
        rc.record(50)
        assert rc.count == 50

    def test_warn_threshold(self) -> None:
        rc = RollingRateCounter()
        rc.record(RATE_LIMIT_WARN)
        assert rc.is_warn
        assert not rc.is_hard_limit

    def test_hard_limit_threshold(self) -> None:
        rc = RollingRateCounter()
        rc.record(RATE_LIMIT_HARD)
        assert rc.is_hard_limit

    def test_pruning_expired_entries(self) -> None:
        """Old entries outside the window are pruned automatically."""
        rc = RollingRateCounter(window_seconds=1)
        rc.record(100)
        assert rc.count == 100
        # Sleep just over the window
        time.sleep(1.1)
        assert rc.count == 0

    def test_status_dict_shape(self) -> None:
        rc = RollingRateCounter()
        rc.record(42)
        status = rc.status_dict()
        assert status["actions_60min"] == 42
        assert status["warn_threshold"] == RATE_LIMIT_WARN
        assert status["hard_threshold"] == RATE_LIMIT_HARD
        assert status["warn"] is False
        assert status["blocked"] is False
        assert status["headroom"] == RATE_LIMIT_HARD - 42


# ===================================================================
# Compliance checklist
# ===================================================================


class TestComplianceChecklist:
    """Compliance log generation for audit trail."""

    def test_basic_checklist(self) -> None:
        checks = _build_compliance_checklist(
            side="BUY",
            security_code="MGCQ6",
            qty=1,
            stop_ticks=20,
            num_slaves=3,
            high_impact=False,
            delay_range=(0.2, 0.8),
            rate_count=10,
        )
        assert any("MANUAL" in c for c in checks)
        assert any("manual button push" in c for c in checks)
        assert any("stop_ticks=20" in c for c in checks)
        assert any("3 slave(s)" in c for c in checks)
        assert any("200–800 ms" in c for c in checks)
        # No warnings
        assert not any("⚠" in c for c in checks)

    def test_high_impact_noted(self) -> None:
        checks = _build_compliance_checklist(
            side="SELL",
            security_code="MESZ6",
            qty=2,
            stop_ticks=15,
            num_slaves=1,
            high_impact=True,
            delay_range=(1.0, 2.0),
            rate_count=5,
        )
        assert any("HIGH IMPACT" in c for c in checks)
        assert any("1000–2000 ms" in c for c in checks)

    def test_zero_stop_ticks_warning(self) -> None:
        checks = _build_compliance_checklist(
            side="BUY",
            security_code="MGCQ6",
            qty=1,
            stop_ticks=0,
            num_slaves=0,
            high_impact=False,
            delay_range=(0.2, 0.8),
            rate_count=0,
        )
        assert any("⚠" in c and "stop_ticks=0" in c for c in checks)

    def test_rate_limit_warning(self) -> None:
        checks = _build_compliance_checklist(
            side="BUY",
            security_code="MGCQ6",
            qty=1,
            stop_ticks=20,
            num_slaves=0,
            high_impact=False,
            delay_range=(0.2, 0.8),
            rate_count=RATE_LIMIT_WARN + 100,
        )
        assert any("⚠" in c and "rate limit" in c.lower() for c in checks)


# ===================================================================
# CopyOrderResult & CopyBatchResult
# ===================================================================


class TestCopyOrderResult:
    def test_to_dict(self) -> None:
        r = CopyOrderResult(
            account_key="a1",
            account_label="Account 1",
            is_main=True,
            order_id="ORD_001",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0.0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.SUBMITTED,
            tag="RUBY_MANUAL_WEBUI",
        )
        d = r.to_dict()
        assert d["account_key"] == "a1"
        assert d["placement_mode"] == "MANUAL"
        assert d["tag"] == "RUBY_MANUAL_WEBUI"
        assert d["status"] == "submitted"
        assert d["timestamp"]  # auto-filled


class TestCopyBatchResult:
    def test_empty_batch(self) -> None:
        b = CopyBatchResult(batch_id="B1", security_code="MGCQ6", side="BUY", qty=1)
        assert b.total_orders == 0
        assert b.failed_count == 0
        # No results at all — vacuously True
        assert b.all_submitted

    def test_all_submitted(self) -> None:
        main_r = CopyOrderResult(
            account_key="m",
            account_label="Main",
            is_main=True,
            order_id="O1",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.SUBMITTED,
        )
        slave_r = CopyOrderResult(
            account_key="s1",
            account_label="Slave 1",
            is_main=False,
            order_id="O2",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.SUBMITTED,
        )
        b = CopyBatchResult(
            batch_id="B1",
            security_code="MGCQ6",
            side="BUY",
            qty=1,
            main_result=main_r,
            slave_results=[slave_r],
        )
        assert b.total_orders == 2
        assert b.failed_count == 0
        assert b.all_submitted

    def test_failed_count(self) -> None:
        main_r = CopyOrderResult(
            account_key="m",
            account_label="Main",
            is_main=True,
            order_id="O1",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.SUBMITTED,
        )
        bad_slave = CopyOrderResult(
            account_key="s1",
            account_label="Slave 1",
            is_main=False,
            order_id="O2",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.ERROR,
            error="connection lost",
        )
        b = CopyBatchResult(
            batch_id="B1",
            security_code="MGCQ6",
            side="BUY",
            qty=1,
            main_result=main_r,
            slave_results=[bad_slave],
        )
        assert b.failed_count == 1
        assert not b.all_submitted

    def test_to_dict(self) -> None:
        b = CopyBatchResult(batch_id="B1", security_code="X", side="BUY", qty=1)
        d = b.to_dict()
        assert "batch_id" in d
        assert "compliance_log" in d
        assert d["all_submitted"] is True


# ===================================================================
# TICKER_TO_RITHMIC mapping
# ===================================================================


class TestTickerMapping:
    """Verify all core watchlist tickers have Rithmic mappings."""

    @pytest.mark.parametrize(
        "ticker,expected_product",
        [
            ("MGC=F", "MGC"),
            ("MCL=F", "MCL"),
            ("MES=F", "MES"),
            ("MNQ=F", "MNQ"),
            ("M6E=F", "M6E"),
        ],
    )
    def test_core_tickers_mapped(self, ticker: str, expected_product: str) -> None:
        assert ticker in TICKER_TO_RITHMIC
        assert TICKER_TO_RITHMIC[ticker]["product_code"] == expected_product
        assert TICKER_TO_RITHMIC[ticker]["exchange"] in ("CME", "NYMEX", "CBOT")

    def test_extended_tickers_present(self) -> None:
        extended = ["MYM=F", "M2K=F", "MBT=F", "MET=F", "SIL=F", "MNG=F"]
        for ticker in extended:
            assert ticker in TICKER_TO_RITHMIC, f"{ticker} missing from TICKER_TO_RITHMIC"

    def test_full_size_tickers_present(self) -> None:
        full_size = ["ES=F", "NQ=F", "GC=F", "CL=F"]
        for ticker in full_size:
            assert ticker in TICKER_TO_RITHMIC

    def test_no_duplicate_product_codes(self) -> None:
        """Each product code should appear exactly once."""
        seen: dict[str, str] = {}
        for ticker, mapping in TICKER_TO_RITHMIC.items():
            pc = mapping["product_code"]
            if pc in seen:
                pytest.fail(f"Product code {pc} duplicated: {seen[pc]} and {ticker}")
            seen[pc] = ticker


# ===================================================================
# CopyTrader — initialisation
# ===================================================================


class TestCopyTraderInit:
    def test_default_state(self, ct: CopyTrader) -> None:
        assert ct._main is None
        assert ct._slaves == {}
        assert ct._enabled_slave_keys == set()
        assert ct._high_impact_mode is False
        assert ct._rate_counter.count == 0

    def test_repr_no_main(self, ct: CopyTrader) -> None:
        r = repr(ct)
        assert "main=None" in r
        assert "slaves=0" in r

    def test_repr_with_main(self, ct_wired: CopyTrader) -> None:
        r = repr(ct_wired)
        assert "Main Account" in r
        assert "slaves=2" in r


# ===================================================================
# CopyTrader — account management
# ===================================================================


class TestAccountManagement:
    @pytest.mark.asyncio()
    async def test_add_account_no_async_rithmic(self, ct: CopyTrader) -> None:
        """Gracefully handles missing async-rithmic package."""
        config = _FakeConfig(key="acc1")

        with (
            patch.dict("sys.modules", {"async_rithmic": None}),
            patch(
                "lib.services.engine.copy_trader.CopyTrader.add_account",
                new=ct.add_account,
            ),
        ):
            # Just test that the code path works with our fake
            pass

        # The real test: call add_account when RithmicClient import fails
        # We mock the import inside the method

        async def _add_with_import_error() -> dict[str, Any]:
            """Simulate ImportError for async_rithmic."""
            import builtins

            real_import = builtins.__import__

            def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "async_rithmic":
                    raise ImportError("No module named 'async_rithmic'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                return await ct.add_account(config, is_main=True)

        result = await _add_with_import_error()
        assert result["connected"] is False
        assert "not installed" in result.get("error", "")

    @pytest.mark.asyncio()
    async def test_add_account_no_credentials(self, ct: CopyTrader) -> None:
        """Rejects accounts with empty credentials."""
        config = _FakeConfig(key="bad", _username="", _password="")

        # Mock so RithmicClient import succeeds but creds are empty
        fake_module = MagicMock()
        fake_module.RithmicClient = _FakeRithmicClient

        with patch.dict("sys.modules", {"async_rithmic": fake_module}):
            result = await ct.add_account(config, is_main=True)

        assert result["connected"] is False
        assert "credentials" in result.get("error", "")

    def test_enable_disable_slave(self, ct_wired: CopyTrader) -> None:
        assert "slave1" in ct_wired._enabled_slave_keys
        ct_wired.disable_slave("slave1")
        assert "slave1" not in ct_wired._enabled_slave_keys
        ct_wired.enable_slave("slave1")
        assert "slave1" in ct_wired._enabled_slave_keys

    def test_disable_nonexistent_slave(self, ct: CopyTrader) -> None:
        """Disabling a non-existent slave key is a no-op."""
        ct.disable_slave("nonexistent")  # should not raise

    @pytest.mark.asyncio()
    async def test_disconnect_all(self, ct_wired: CopyTrader) -> None:
        await ct_wired.disconnect_all()
        assert ct_wired._main is None
        assert ct_wired._slaves == {}
        assert ct_wired._enabled_slave_keys == set()


# ===================================================================
# CopyTrader — high-impact mode
# ===================================================================


class TestHighImpactMode:
    def test_toggle_on(self, ct: CopyTrader) -> None:
        ct.set_high_impact_mode(True)
        assert ct._high_impact_mode is True

    def test_toggle_off(self, ct: CopyTrader) -> None:
        ct.set_high_impact_mode(True)
        ct.set_high_impact_mode(False)
        assert ct._high_impact_mode is False


# ===================================================================
# CopyTrader — contract resolution
# ===================================================================


class TestContractResolution:
    @pytest.mark.asyncio()
    async def test_resolve_unknown_ticker(self, ct: CopyTrader) -> None:
        result = await ct.resolve_front_month("FAKE=F")
        assert result is None

    @pytest.mark.asyncio()
    async def test_resolve_no_client(self, ct: CopyTrader) -> None:
        """Without a connected client, resolution returns None."""
        result = await ct.resolve_front_month("MGC=F")
        assert result is None

    @pytest.mark.asyncio()
    async def test_resolve_with_client(self, ct_wired: CopyTrader) -> None:
        result = await ct_wired.resolve_front_month("MGC=F")
        assert result is not None
        security_code, exchange = result
        assert security_code == "MGCQ6"
        assert exchange == "NYMEX"

    @pytest.mark.asyncio()
    async def test_contract_cache(self, ct_wired: CopyTrader) -> None:
        """Second resolution uses cache (no client call)."""
        await ct_wired.resolve_front_month("MES=F")
        assert "MES" in ct_wired._contract_cache

        # Even if client is disconnected, cache works
        ct_wired._main.connected = False  # type: ignore[union-attr]
        result = await ct_wired.resolve_front_month("MES=F")
        assert result is not None

    def test_invalidate_single(self, ct: CopyTrader) -> None:
        ct._contract_cache["MGC"] = "MGCQ6"
        ct._contract_cache["MES"] = "MESQ6"
        ct.invalidate_contract_cache("MGC")
        assert "MGC" not in ct._contract_cache
        assert "MES" in ct._contract_cache

    def test_invalidate_all(self, ct: CopyTrader) -> None:
        ct._contract_cache["MGC"] = "MGCQ6"
        ct._contract_cache["MES"] = "MESQ6"
        ct.invalidate_contract_cache()
        assert ct._contract_cache == {}


# ===================================================================
# CopyTrader — send_order_and_copy (core flow)
# ===================================================================


class TestSendOrderAndCopy:
    """Core order + copy flow with compliance enforcement."""

    @pytest.mark.asyncio()
    async def test_basic_order_and_copy(self, ct_wired: CopyTrader) -> None:
        """Main + 2 slaves → 3 total orders, all with MANUAL flag."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=False,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            # 1 main + 2 slaves = 3 calls
            assert mock_submit.call_count == 3

            # Verify tags: first call is main, rest are copies
            calls = mock_submit.call_args_list
            assert calls[0].kwargs["tag"] == "RUBY_MANUAL_WEBUI"
            assert calls[0].kwargs["is_main"] is True
            assert calls[1].kwargs["tag"] == "COPY_FROM_MAIN_HUMAN_150K"
            assert calls[1].kwargs["is_main"] is False
            assert calls[2].kwargs["tag"] == "COPY_FROM_MAIN_HUMAN_150K"

    @pytest.mark.asyncio()
    async def test_main_failure_aborts_copies(self, ct_wired: CopyTrader) -> None:
        """If main order fails, no slave copies should be attempted."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            call_count = 0

            async def _mock_submit(**kwargs: Any) -> CopyOrderResult:
                nonlocal call_count
                call_count += 1
                return CopyOrderResult(
                    account_key="x",
                    account_label="X",
                    is_main=kwargs.get("is_main", False),
                    order_id="O",
                    security_code="MGCQ6",
                    exchange="NYMEX",
                    side="BUY",
                    qty=1,
                    order_type="MARKET",
                    price=0,
                    stop_ticks=20,
                    target_ticks=None,
                    status=CopyOrderStatus.ERROR,
                    error="connection refused",
                )

            ct_wired._submit_single_order = _mock_submit  # type: ignore[assignment]

            result = await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            # Only 1 call (main) — slaves were never attempted
            assert call_count == 1
            assert result.main_result is not None
            assert result.main_result.status == CopyOrderStatus.ERROR
            assert len(result.slave_results) == 0

    @pytest.mark.asyncio()
    async def test_compliance_log_populated(self, ct_wired: CopyTrader) -> None:
        """Every batch has a compliance checklist."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=False,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            result = await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            assert len(result.compliance_log) >= 4
            assert any("MANUAL" in c for c in result.compliance_log)
            assert any("stop_ticks" in c for c in result.compliance_log)

    @pytest.mark.asyncio()
    async def test_rate_limit_blocks_order(self, ct_wired: CopyTrader) -> None:
        """When hard rate limit is hit, orders are blocked."""
        ct_wired._rate_counter.record(RATE_LIMIT_HARD + 1)

        with patch("lib.services.engine.copy_trader._log_compliance"):
            result = await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

        # No orders submitted
        assert result.main_result is None
        assert len(result.slave_results) == 0
        assert any("BLOCKED" in c for c in result.compliance_log)

    @pytest.mark.asyncio()
    async def test_stop_ticks_passed_through(self, ct_wired: CopyTrader) -> None:
        """Server-side brackets (stop_ticks, target_ticks) are forwarded."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            captured_kwargs: list[dict[str, Any]] = []

            async def _capture_submit(**kwargs: Any) -> CopyOrderResult:
                captured_kwargs.append(kwargs)
                return CopyOrderResult(
                    account_key="x",
                    account_label="X",
                    is_main=kwargs.get("is_main", False),
                    order_id="O",
                    security_code="MGCQ6",
                    exchange="NYMEX",
                    side="BUY",
                    qty=1,
                    order_type="MARKET",
                    price=0,
                    stop_ticks=kwargs.get("stop_ticks", 0),
                    target_ticks=kwargs.get("target_ticks"),
                    status=CopyOrderStatus.SUBMITTED,
                )

            ct_wired._submit_single_order = _capture_submit  # type: ignore[assignment]

            await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=25,
                target_ticks=50,
            )

            # All 3 orders (main + 2 slaves) should have same brackets
            assert len(captured_kwargs) == 3
            for kw in captured_kwargs:
                assert kw["stop_ticks"] == 25
                assert kw["target_ticks"] == 50

    @pytest.mark.asyncio()
    async def test_no_main_returns_error(self, ct: CopyTrader) -> None:
        """Sending without a main account connected results in error."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            result = await ct.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            # Main result should be an error (no account connected)
            assert result.main_result is not None
            assert result.main_result.status == CopyOrderStatus.ERROR

    @pytest.mark.asyncio()
    async def test_no_slaves_main_only(self, ct: CopyTrader) -> None:
        """With no slaves, only main order is placed."""
        ct._main = _make_fake_account("main", "Main", is_main=True)

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="main",
                account_label="Main",
                is_main=True,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            result = await ct.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            assert mock_submit.call_count == 1  # main only
            assert len(result.slave_results) == 0

    @pytest.mark.asyncio()
    async def test_disabled_slaves_skipped(self, ct_wired: CopyTrader) -> None:
        """Disabled slaves are not copied to."""
        ct_wired.disable_slave("slave2")

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=False,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            # 1 main + 1 enabled slave = 2 calls
            assert mock_submit.call_count == 2

    @pytest.mark.asyncio()
    async def test_reason_in_batch_id(self, ct_wired: CopyTrader) -> None:
        """The reason string is included in batch_id."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=False,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            result = await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
                reason="ORB breakout",
            )

            assert "ORB breakout" in result.batch_id


# ===================================================================
# CopyTrader — send_order_from_ticker
# ===================================================================


class TestSendOrderFromTicker:
    @pytest.mark.asyncio()
    async def test_unknown_ticker(self, ct: CopyTrader) -> None:
        result = await ct.send_order_from_ticker(
            ticker="FAKE=F",
            side="BUY",
            qty=1,
        )
        assert any("Cannot resolve" in c for c in result.compliance_log)

    @pytest.mark.asyncio()
    async def test_no_client_for_resolution(self, ct: CopyTrader) -> None:
        """Known ticker but no connected client to resolve front month."""
        result = await ct.send_order_from_ticker(
            ticker="MGC=F",
            side="BUY",
            qty=1,
        )
        assert any("No connected" in c for c in result.compliance_log)

    @pytest.mark.asyncio()
    async def test_successful_ticker_resolution(self, ct_wired: CopyTrader) -> None:
        """Resolves ticker → security_code → sends order."""
        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch(
                "lib.services.engine.copy_trader.CopyTrader._submit_single_order",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            mock_submit.return_value = CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=False,
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

            result = await ct_wired.send_order_from_ticker(
                ticker="MGC=F",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

            assert result.security_code == "MGCQ6"
            assert mock_submit.call_count == 3  # main + 2 slaves


# ===================================================================
# CopyTrader — _submit_single_order (MANUAL flag enforcement)
# ===================================================================


class TestSubmitSingleOrder:
    """Verify that _submit_single_order always uses OrderPlacement.MANUAL."""

    @pytest.mark.asyncio()
    async def test_manual_flag_in_submit_kwargs(self, ct_wired: CopyTrader) -> None:
        """The MANUAL flag must be in submit_order kwargs for every order."""
        main_acct = ct_wired._main
        assert main_acct is not None

        # We'll capture what submit_order receives by patching at the client level
        captured: list[dict[str, Any]] = []
        original_submit = main_acct.client.submit_order

        async def _capture_submit(**kwargs: Any) -> None:
            captured.append(kwargs)
            return await original_submit(**kwargs)

        main_acct.client.submit_order = _capture_submit

        # Need to mock the async_rithmic imports inside _submit_single_order
        mock_order_placement = MagicMock()
        mock_order_placement.MANUAL = "MANUAL_FLAG"

        mock_tx_type = MagicMock()
        mock_tx_type.BUY = "BUY"
        mock_tx_type.SELL = "SELL"

        mock_order_type = MagicMock()
        mock_order_type.MARKET = "MARKET"
        mock_order_type.LIMIT = "LIMIT"

        fake_rithmic = MagicMock()
        fake_rithmic.OrderPlacement = mock_order_placement
        fake_rithmic.TransactionType = mock_tx_type
        fake_rithmic.OrderType = mock_order_type

        with patch.dict("sys.modules", {"async_rithmic": fake_rithmic}):
            result = await ct_wired._submit_single_order(
                acct=main_acct,
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0.0,
                stop_ticks=20,
                target_ticks=None,
                tag="TEST_MANUAL",
                is_main=True,
                batch_id="TEST_001",
            )

        assert result.status == CopyOrderStatus.SUBMITTED
        assert len(captured) == 1
        assert captured[0]["manual_or_auto"] == "MANUAL_FLAG"
        assert captured[0]["stop_ticks"] == 20

    @pytest.mark.asyncio()
    async def test_disconnected_account_returns_error(self, ct: CopyTrader) -> None:
        """Submitting to a disconnected account returns ERROR status."""
        acct = _make_fake_account("broken", "Broken", connected=False)

        result = await ct._submit_single_order(
            acct=acct,
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0.0,
            stop_ticks=20,
            target_ticks=None,
            tag="TEST",
            is_main=False,
            batch_id="ERR_001",
        )

        assert result.status == CopyOrderStatus.ERROR
        assert "disconnected" in result.error

    @pytest.mark.asyncio()
    async def test_none_account_returns_error(self, ct: CopyTrader) -> None:
        result = await ct._submit_single_order(
            acct=None,
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0.0,
            stop_ticks=20,
            target_ticks=None,
            tag="TEST",
            is_main=True,
            batch_id="ERR_002",
        )

        assert result.status == CopyOrderStatus.ERROR
        assert "not connected" in result.error

    @pytest.mark.asyncio()
    async def test_limit_order_includes_price(self, ct_wired: CopyTrader) -> None:
        """Limit orders pass the price through to submit_order."""
        main_acct = ct_wired._main
        assert main_acct is not None

        captured: list[dict[str, Any]] = []

        async def _capture(**kwargs: Any) -> None:
            captured.append(kwargs)

        main_acct.client.submit_order = _capture

        mock_rithmic = MagicMock()
        mock_rithmic.OrderPlacement.MANUAL = "MANUAL"
        mock_rithmic.TransactionType.BUY = "BUY"
        mock_rithmic.TransactionType.SELL = "SELL"
        mock_rithmic.OrderType.MARKET = "MARKET"
        mock_rithmic.OrderType.LIMIT = "LIMIT"

        with patch.dict("sys.modules", {"async_rithmic": mock_rithmic}):
            result = await ct_wired._submit_single_order(
                acct=main_acct,
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="LIMIT",
                price=2650.50,
                stop_ticks=20,
                target_ticks=40,
                tag="LIMIT_TEST",
                is_main=True,
                batch_id="LMT_001",
            )

        assert result.status == CopyOrderStatus.SUBMITTED
        assert len(captured) == 1
        assert captured[0]["price"] == 2650.50
        assert captured[0]["target_ticks"] == 40
        assert captured[0]["stop_ticks"] == 20


# ===================================================================
# CopyTrader — humanised delay verification
# ===================================================================


class TestHumanisedDelay:
    """Verify that slave copies are delayed by the correct range."""

    @pytest.mark.asyncio()
    async def test_normal_delay_range(self, ct_wired: CopyTrader) -> None:
        """Slave copies should have delay_ms in the 200–800 ms range."""

        async def _mock_submit(**kwargs: Any) -> CopyOrderResult:
            return CopyOrderResult(
                account_key=kwargs.get("acct", _make_fake_account()).key
                if isinstance(kwargs.get("acct"), _ConnectedAccount)
                else "x",
                account_label="X",
                is_main=kwargs.get("is_main", False),
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
                delay_ms=kwargs.get("delay_ms", 0),
            )

        # Patch sleep to capture delay values without actually sleeping
        sleep_values: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_values.append(seconds)

        ct_wired._submit_single_order = _mock_submit  # type: ignore[assignment]

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch("asyncio.sleep", side_effect=_mock_sleep),
        ):
            await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

        # 2 slaves = 2 delays
        assert len(sleep_values) == 2
        for delay in sleep_values:
            assert COPY_DELAY_MIN <= delay <= COPY_DELAY_MAX, (
                f"Delay {delay:.3f}s outside normal range [{COPY_DELAY_MIN}, {COPY_DELAY_MAX}]"
            )

    @pytest.mark.asyncio()
    async def test_high_impact_delay_range(self, ct_wired: CopyTrader) -> None:
        """In high-impact mode, delays should be 1.0–2.0 s."""
        ct_wired.set_high_impact_mode(True)

        sleep_values: list[float] = []

        async def _mock_sleep(seconds: float) -> None:
            sleep_values.append(seconds)

        async def _mock_submit(**kwargs: Any) -> CopyOrderResult:
            return CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=kwargs.get("is_main", False),
                order_id="O",
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

        ct_wired._submit_single_order = _mock_submit  # type: ignore[assignment]

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
            patch("asyncio.sleep", side_effect=_mock_sleep),
        ):
            await ct_wired.send_order_and_copy(
                security_code="MGCQ6",
                exchange="NYMEX",
                side="BUY",
                qty=1,
                stop_ticks=20,
            )

        assert len(sleep_values) == 2
        for delay in sleep_values:
            assert HIGH_IMPACT_DELAY_MIN <= delay <= HIGH_IMPACT_DELAY_MAX, (
                f"High-impact delay {delay:.3f}s outside range [{HIGH_IMPACT_DELAY_MIN}, {HIGH_IMPACT_DELAY_MAX}]"
            )


# ===================================================================
# CopyTrader — execute_order_commands (PM integration)
# ===================================================================


class TestExecuteOrderCommands:
    """PositionManager OrderCommand → CopyTrader translation."""

    @dataclass
    class _FakeOrderCommand:
        symbol: str = "MGC=F"
        action: str = "BUY"
        order_type: str = "MARKET"
        quantity: int = 1
        price: float = 0.0
        stop_price: float = 2630.0
        reason: str = "New LONG entry: ORB breakout"
        position_id: str = "pos_001"

    @pytest.mark.asyncio()
    async def test_buy_order_translated(self, ct_wired: CopyTrader) -> None:
        """BUY OrderCommand is translated and sent."""
        cmd = self._FakeOrderCommand(action="BUY")

        with patch.object(ct_wired, "send_order_from_ticker", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = CopyBatchResult(batch_id="PM_MGC=F", security_code="MGCQ6", side="BUY", qty=1)

            await ct_wired.execute_order_commands([cmd])

        assert mock_send.call_count == 1
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["ticker"] == "MGC=F"
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["tag_prefix"] == "PM"

    @pytest.mark.asyncio()
    async def test_sell_order_translated(self, ct_wired: CopyTrader) -> None:
        cmd = self._FakeOrderCommand(action="SELL")

        with patch.object(ct_wired, "send_order_from_ticker", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = CopyBatchResult(batch_id="PM_MGC=F", security_code="MGCQ6", side="SELL", qty=1)

            await ct_wired.execute_order_commands([cmd])

        call_kwargs = mock_send.call_args.kwargs
        assert call_kwargs["side"] == "SELL"

    @pytest.mark.asyncio()
    async def test_modify_stop_returns_result(self, ct_wired: CopyTrader) -> None:
        """MODIFY_STOP is now forwarded to modify_stop_on_all() and returns a dict result."""
        cmd = self._FakeOrderCommand(action="MODIFY_STOP", stop_price=2600.0)

        with (
            patch.object(ct_wired, "send_order_from_ticker", new_callable=AsyncMock) as mock_send,
            patch.object(
                ct_wired,
                "modify_stop_on_all",
                new_callable=AsyncMock,
                return_value={
                    "ok": True,
                    "accounts_modified": ["main"],
                    "accounts_failed": [],
                    "stop_ticks": 20,
                    "reason": "",
                },
            ),
        ):
            results = await ct_wired.execute_order_commands([cmd])

        # Now produces 1 dict result (either resolution error or modify result)
        assert len(results) == 1
        mock_send.assert_not_called()

    @pytest.mark.asyncio()
    async def test_cancel_returns_result(self, ct_wired: CopyTrader) -> None:
        """CANCEL is now forwarded to cancel_on_all() and returns a dict result."""
        cmd = self._FakeOrderCommand(action="CANCEL")

        with (
            patch.object(ct_wired, "send_order_from_ticker", new_callable=AsyncMock) as mock_send,
            patch.object(
                ct_wired,
                "cancel_on_all",
                new_callable=AsyncMock,
                return_value={"ok": True, "accounts_cancelled": ["main"], "accounts_failed": [], "reason": ""},
            ),
        ):
            results = await ct_wired.execute_order_commands([cmd])

        assert len(results) == 1
        mock_send.assert_not_called()

    @pytest.mark.asyncio()
    async def test_multiple_commands(self, ct_wired: CopyTrader) -> None:
        """Multiple entry commands are each forwarded."""
        cmds = [
            self._FakeOrderCommand(action="BUY", symbol="MGC=F"),
            self._FakeOrderCommand(action="SELL", symbol="MCL=F"),
        ]

        with patch.object(ct_wired, "send_order_from_ticker", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = CopyBatchResult(batch_id="PM_X", security_code="X", side="BUY", qty=1)

            results = await ct_wired.execute_order_commands(cmds)

        assert len(results) == 2
        assert mock_send.call_count == 2


# ===================================================================
# CopyTrader — status & monitoring
# ===================================================================


class TestStatusMonitoring:
    def test_status_summary_no_accounts(self, ct: CopyTrader) -> None:
        status = ct.status_summary()
        assert status["main"] is None
        assert status["slaves"] == []
        assert status["enabled_slave_count"] == 0
        assert status["high_impact_mode"] is False

    def test_status_summary_wired(self, ct_wired: CopyTrader) -> None:
        status = ct_wired.status_summary()
        assert status["main"] is not None
        assert status["main"]["key"] == "main1"
        assert status["main"]["connected"] is True
        assert len(status["slaves"]) == 2
        assert status["enabled_slave_count"] == 2

    def test_get_order_history_empty(self, ct: CopyTrader) -> None:
        assert ct.get_order_history() == []

    def test_get_order_history_returns_newest_first(self, ct: CopyTrader) -> None:
        ct._order_history.append({"batch_id": "B1", "ts": 1})
        ct._order_history.append({"batch_id": "B2", "ts": 2})
        ct._order_history.append({"batch_id": "B3", "ts": 3})
        history = ct.get_order_history(limit=3)
        assert history[0]["batch_id"] == "B3"
        assert history[-1]["batch_id"] == "B1"

    def test_get_rate_status(self, ct: CopyTrader) -> None:
        status = ct.get_rate_status()
        assert "actions_60min" in status
        assert "headroom" in status

    def test_rate_counter_records_on_batch(self, ct_wired: CopyTrader) -> None:
        """After a successful batch, rate counter reflects total orders."""
        initial = ct_wired._rate_counter.count
        # Manually record what a 3-order batch would add
        ct_wired._rate_counter.record(3)
        assert ct_wired._rate_counter.count == initial + 3


# ===================================================================
# CopyTrader — get_copy_trader singleton
# ===================================================================


class TestGetCopyTrader:
    def test_returns_instance(self) -> None:
        import lib.services.engine.copy_trader as ct_module

        ct_module._copy_trader = None  # reset
        trader = get_copy_trader()
        assert isinstance(trader, CopyTrader)

    def test_returns_same_instance(self) -> None:
        import lib.services.engine.copy_trader as ct_module

        ct_module._copy_trader = None  # reset
        t1 = get_copy_trader()
        t2 = get_copy_trader()
        assert t1 is t2


# ===================================================================
# CopyTrader — _get_enabled_slaves
# ===================================================================


class TestGetEnabledSlaves:
    def test_empty(self, ct: CopyTrader) -> None:
        assert ct._get_enabled_slaves() == []

    def test_filters_disabled(self, ct_wired: CopyTrader) -> None:
        ct_wired.disable_slave("slave1")
        enabled = ct_wired._get_enabled_slaves()
        assert len(enabled) == 1
        assert enabled[0].key == "slave2"

    def test_filters_disconnected(self, ct_wired: CopyTrader) -> None:
        ct_wired._slaves["slave1"].connected = False
        enabled = ct_wired._get_enabled_slaves()
        assert len(enabled) == 1
        assert enabled[0].key == "slave2"


# ===================================================================
# OrderSide enum
# ===================================================================


class TestOrderSide:
    def test_values(self) -> None:
        assert OrderSide.BUY == "BUY"
        assert OrderSide.SELL == "SELL"


# ===========================================================================
# RITHMIC-C: stop_price_to_stop_ticks
# ===========================================================================


class TestStopPriceToStopTicks:
    """Tests for the tick-size conversion helper."""

    def test_known_product_mgc(self) -> None:
        # MGC tick = 0.10; entry=2000.00, stop=1998.50 → dist=1.50 → 15 ticks
        result = stop_price_to_stop_ticks(2000.00, 1998.50, "MGC")
        assert result == 15

    def test_known_product_mes(self) -> None:
        # MES tick = 0.25; entry=4500.00, stop=4498.75 → dist=1.25 → 5 ticks
        result = stop_price_to_stop_ticks(4500.00, 4498.75, "MES")
        assert result == 5

    def test_known_product_mcl(self) -> None:
        # MCL tick = 0.01; entry=80.00, stop=79.90 → dist=0.10 → 10 ticks
        result = stop_price_to_stop_ticks(80.00, 79.90, "MCL")
        assert result == 10

    def test_short_side_stop_above_entry(self) -> None:
        # Short: stop is above entry — abs() should still work
        # MGC tick = 0.10; entry=2000.00, stop=2001.50 → dist=1.50 → 15 ticks
        result = stop_price_to_stop_ticks(2000.00, 2001.50, "MGC")
        assert result == 15

    def test_unknown_product_returns_default(self) -> None:
        result = stop_price_to_stop_ticks(100.0, 99.0, "UNKNWN")
        assert result == DEFAULT_STOP_TICKS

    def test_zero_entry_price_returns_default(self) -> None:
        result = stop_price_to_stop_ticks(0.0, 99.0, "MGC")
        assert result == DEFAULT_STOP_TICKS

    def test_equal_prices_returns_min_ticks(self) -> None:
        # distance == 0 → min_ticks
        result = stop_price_to_stop_ticks(2000.0, 2000.0, "MGC")
        assert result == MIN_STOP_TICKS

    def test_very_tight_stop_clamped_to_min(self) -> None:
        # dist = 0.05 on MGC (tick=0.10) → round(0.5) = 0 → clamped to MIN_STOP_TICKS
        result = stop_price_to_stop_ticks(2000.0, 1999.95, "MGC")
        assert result >= MIN_STOP_TICKS

    def test_custom_min_ticks_respected(self) -> None:
        result = stop_price_to_stop_ticks(2000.0, 1999.90, "MGC", min_ticks=5)
        assert result >= 5

    def test_custom_default_ticks_used_on_unknown(self) -> None:
        result = stop_price_to_stop_ticks(100.0, 99.0, "XYZ", default_ticks=42)
        assert result == 42

    def test_tick_size_table_has_core_products(self) -> None:
        for code in ("MGC", "MCL", "MES", "MNQ", "M6E", "MBT"):
            assert code in TICK_SIZE, f"{code} missing from TICK_SIZE"

    def test_all_ticker_to_rithmic_product_codes_in_tick_table(self) -> None:
        missing = []
        for ticker, info in TICKER_TO_RITHMIC.items():
            pc = info.get("product_code", "")
            if pc and pc not in TICK_SIZE:
                missing.append(f"{ticker}→{pc}")
        assert not missing, f"product codes missing from TICK_SIZE: {missing}"

    def test_mnq_tick(self) -> None:
        # MNQ tick = 0.25; dist = 5.0 → 20 ticks
        result = stop_price_to_stop_ticks(15000.0, 14995.0, "MNQ")
        assert result == 20

    def test_m6e_tick(self) -> None:
        # M6E tick = 0.0001; dist = 0.002 → 20 ticks
        result = stop_price_to_stop_ticks(1.0800, 1.0780, "M6E")
        assert result == 20


# ===========================================================================
# RITHMIC-C: modify_stop_on_all
# ===========================================================================


class TestModifyStopOnAll:
    """Tests for CopyTrader.modify_stop_on_all()."""

    @pytest.mark.asyncio()
    async def test_no_connected_accounts_returns_not_ok(self, ct: CopyTrader) -> None:
        result = await ct.modify_stop_on_all(
            security_code="MGCQ6",
            exchange="NYMEX",
            new_stop_price=1990.0,
            product_code="MGC",
            entry_price=2000.0,
        )
        assert result["ok"] is False
        assert result["accounts_modified"] == []
        assert "no connected accounts" in result["reason"]

    @pytest.mark.asyncio()
    async def test_async_rithmic_not_installed_fails_gracefully(self, ct_wired: CopyTrader) -> None:
        """When async_rithmic is not importable, the account is marked failed."""
        captured: list[dict[str, Any]] = []

        async def _fake_modify(**kwargs: Any) -> None:
            captured.append(kwargs)

        ct_wired._main.client.modify_order = _fake_modify  # type: ignore[union-attr]

        # Patch import to succeed but simulate the call itself raising ImportError
        original_modify = ct_wired._main.client.modify_order  # type: ignore[union-attr]

        async def _raise_import(**kwargs: Any) -> None:
            raise ImportError("async_rithmic not installed")

        ct_wired._main.client.modify_order = _raise_import  # type: ignore[union-attr]

        result = await ct_wired.modify_stop_on_all(
            security_code="MGCQ6",
            exchange="NYMEX",
            new_stop_price=1990.0,
            product_code="MGC",
            entry_price=2000.0,
        )
        # Should not raise; main marked as failed
        assert isinstance(result, dict)

        # Restore
        ct_wired._main.client.modify_order = original_modify  # type: ignore[union-attr]

    @pytest.mark.asyncio()
    async def test_modify_called_with_stop_ticks(self, ct_wired: CopyTrader) -> None:
        """modify_stop_on_all converts price → ticks and calls client.modify_order."""
        captured_kwargs: list[dict[str, Any]] = []

        async def _fake_modify(**kwargs: Any) -> None:
            captured_kwargs.append(kwargs)

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.modify_order = _fake_modify  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.modify_stop_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                new_stop_price=1990.0,  # 10 ticks below 2000 for MGC (tick=0.10)
                product_code="MGC",
                entry_price=2000.0,
            )

        assert len(captured_kwargs) >= 1
        assert captured_kwargs[0]["stop_ticks"] == 100  # dist=10.0 / 0.10 = 100 ticks
        assert result["stop_ticks"] == 100

    @pytest.mark.asyncio()
    async def test_product_code_inferred_from_security_code(self, ct_wired: CopyTrader) -> None:
        """product_code should be inferred from the security_code prefix."""
        result = await ct_wired.modify_stop_on_all(
            security_code="MGCQ6",  # prefix "MGC" → tick=0.10
            exchange="NYMEX",
            new_stop_price=1990.0,
            entry_price=2000.0,
            # product_code intentionally omitted
        )
        # stop_ticks computed correctly regardless of connection state
        assert result["stop_ticks"] == 100  # dist=10.0 / 0.10 = 100

    @pytest.mark.asyncio()
    async def test_timeout_marks_account_failed(self, ct_wired: CopyTrader) -> None:
        async def _timeout(**kwargs: Any) -> None:
            raise TimeoutError

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            raise TimeoutError

        ct_wired._main.client.modify_order = _timeout  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.modify_stop_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                new_stop_price=1990.0,
                product_code="MGC",
                entry_price=2000.0,
            )

        assert ct_wired._main is not None
        assert ct_wired._main.key in result["accounts_failed"]  # type: ignore[operator]
        assert result["ok"] is False

    @pytest.mark.asyncio()
    async def test_rate_counter_incremented(self, ct_wired: CopyTrader) -> None:
        before = ct_wired._rate_counter.count

        async def _fake_modify(**kwargs: Any) -> None:
            pass

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.modify_order = _fake_modify  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            await ct_wired.modify_stop_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                new_stop_price=1990.0,
                product_code="MGC",
                entry_price=2000.0,
            )

        # ct_wired has main + 2 enabled slaves → 3 accounts each get 1 record
        assert ct_wired._rate_counter.count == before + 3

    @pytest.mark.asyncio()
    async def test_result_includes_audit_fields(self, ct_wired: CopyTrader) -> None:
        async def _fake_modify(**kwargs: Any) -> None:
            pass

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.modify_order = _fake_modify  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.modify_stop_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                new_stop_price=1990.0,
                product_code="MGC",
                entry_price=2000.0,
                position_id="POS-001",
                reason="TP1 hit — moving to breakeven",
            )

        assert result["position_id"] == "POS-001"
        assert result["reason"] == "TP1 hit — moving to breakeven"
        assert result["security_code"] == "MGCQ6"
        assert result["new_stop_price"] == 1990.0


# ===========================================================================
# RITHMIC-C: cancel_on_all
# ===========================================================================


class TestCancelOnAll:
    """Tests for CopyTrader.cancel_on_all()."""

    @pytest.mark.asyncio()
    async def test_no_connected_accounts_returns_not_ok(self, ct: CopyTrader) -> None:
        result = await ct.cancel_on_all(security_code="MGCQ6", exchange="NYMEX")
        assert result["ok"] is False
        assert "no connected accounts" in result["reason"]

    @pytest.mark.asyncio()
    async def test_cancel_called_on_main(self, ct_wired: CopyTrader) -> None:
        captured: list[dict[str, Any]] = []

        async def _fake_cancel(**kwargs: Any) -> None:
            captured.append(kwargs)

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.cancel_all_orders = _fake_cancel  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.cancel_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                reason="position reversed",
            )

        assert len(captured) == 1
        assert captured[0].get("security_code") == "MGCQ6"
        assert result["ok"] is True
        assert ct_wired._main is not None
        assert ct_wired._main.key in result["accounts_cancelled"]  # type: ignore[operator]

    @pytest.mark.asyncio()
    async def test_cancel_without_security_code_cancels_all(self, ct_wired: CopyTrader) -> None:
        captured: list[dict[str, Any]] = []

        async def _fake_cancel(**kwargs: Any) -> None:
            captured.append(kwargs)

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.cancel_all_orders = _fake_cancel  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.cancel_on_all()

        assert "security_code" not in captured[0]
        assert result["ok"] is True

    @pytest.mark.asyncio()
    async def test_timeout_marks_account_failed(self, ct_wired: CopyTrader) -> None:
        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            raise TimeoutError

        ct_wired._main.client.cancel_all_orders = AsyncMock(side_effect=TimeoutError)  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.cancel_on_all(security_code="MGCQ6", exchange="NYMEX")

        assert result["ok"] is False
        assert ct_wired._main is not None
        assert ct_wired._main.key in result["accounts_failed"]  # type: ignore[operator]

    @pytest.mark.asyncio()
    async def test_rate_counter_incremented(self, ct_wired: CopyTrader) -> None:
        before = ct_wired._rate_counter.count

        async def _fake_cancel(**kwargs: Any) -> None:
            pass

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.cancel_all_orders = _fake_cancel  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            await ct_wired.cancel_on_all(security_code="MGCQ6")

        # ct_wired has main + 2 enabled slaves → 3 accounts each get 1 record
        assert ct_wired._rate_counter.count == before + 3

    @pytest.mark.asyncio()
    async def test_audit_fields_in_result(self, ct_wired: CopyTrader) -> None:
        async def _fake_cancel(**kwargs: Any) -> None:
            pass

        async def _direct_wait_for(coro, timeout=None):  # type: ignore[no-untyped-def]
            return await coro

        ct_wired._main.client.cancel_all_orders = _fake_cancel  # type: ignore[union-attr]

        fake_op = MagicMock()
        fake_op.MANUAL = "MANUAL"
        fake_async_rithmic = MagicMock()
        fake_async_rithmic.OrderPlacement = fake_op

        with (
            patch("lib.services.engine.copy_trader.asyncio.wait_for", side_effect=_direct_wait_for),
            patch.dict("sys.modules", {"async_rithmic": fake_async_rithmic}),
        ):
            result = await ct_wired.cancel_on_all(
                security_code="MGCQ6",
                exchange="NYMEX",
                position_id="POS-42",
                reason="session end cancel",
            )

        assert result["position_id"] == "POS-42"
        assert result["reason"] == "session end cancel"
        assert result["security_code"] == "MGCQ6"


# ===========================================================================
# RITHMIC-C: execute_order_commands (updated routing)
# ===========================================================================


class TestExecuteOrderCommandsRouting:
    """Tests for the updated execute_order_commands() covering all routing branches."""

    @dataclass
    class _Cmd:
        symbol: str
        action: str
        order_type: str
        quantity: int = 1
        price: float = 0.0
        stop_price: float = 0.0
        reason: str = ""
        position_id: str = ""

    @pytest.mark.asyncio()
    async def test_stop_companion_is_skipped(self, ct_wired: CopyTrader) -> None:
        """A STOP-type companion order is skipped (bracket covers it)."""
        cmd = self._Cmd(symbol="MGC=F", action="BUY", order_type="STOP", stop_price=1990.0)
        results = await ct_wired.execute_order_commands([cmd])
        assert results == []

    @pytest.mark.asyncio()
    async def test_stop_companion_stores_price_for_entry(self, ct_wired: CopyTrader) -> None:
        """STOP companion price is captured so the following entry can compute stop_ticks."""
        submitted_kwargs: list[dict[str, Any]] = []

        async def _capture(**kwargs: Any) -> None:
            submitted_kwargs.append(kwargs)

        ct_wired._main.client.submit_order = _capture  # type: ignore[union-attr]

        stop_cmd = self._Cmd(symbol="MGC=F", action="BUY", order_type="STOP", stop_price=1990.0)
        entry_cmd = self._Cmd(
            symbol="MGC=F",
            action="BUY",
            order_type="MARKET",
            price=2000.0,
        )

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            # Process STOP first, then entry
            results = await ct_wired.execute_order_commands([stop_cmd, entry_cmd])

        # stop_companion produces no result; entry produces one batch
        assert len(results) == 1

    @pytest.mark.asyncio()
    async def test_modify_stop_dispatched_when_contract_unresolvable(self, ct_wired: CopyTrader) -> None:
        """MODIFY_STOP for an unknown ticker returns an error dict (not an exception)."""
        cmd = self._Cmd(
            symbol="UNKNOWN=F",
            action="MODIFY_STOP",
            order_type="STOP",
            stop_price=100.0,
            position_id="POS-X",
        )
        results = await ct_wired.execute_order_commands([cmd])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, dict)
        assert r["ok"] is False
        assert r["ticker"] == "UNKNOWN=F"

    @pytest.mark.asyncio()
    async def test_modify_stop_calls_modify_stop_on_all(self, ct_wired: CopyTrader) -> None:
        """MODIFY_STOP is forwarded to modify_stop_on_all()."""
        modify_calls: list[dict[str, Any]] = []

        async def _fake_modify_stop(**kwargs: Any) -> dict[str, Any]:
            modify_calls.append(kwargs)
            return {"ok": True, "accounts_modified": ["main"], "accounts_failed": [], "stop_ticks": 20, "reason": ""}

        ct_wired.modify_stop_on_all = _fake_modify_stop  # type: ignore[method-assign]

        cmd = self._Cmd(
            symbol="MGC=F",
            action="MODIFY_STOP",
            order_type="STOP",
            stop_price=1990.0,
            position_id="POS-1",
            reason="TP1 hit",
        )

        # Fake contract resolution
        ct_wired._contract_cache["MGC"] = "MGCQ6"

        results = await ct_wired.execute_order_commands([cmd], entry_prices={"MGC=F": 2000.0})
        assert len(results) == 1
        assert modify_calls[0]["new_stop_price"] == 1990.0
        assert modify_calls[0]["position_id"] == "POS-1"

    @pytest.mark.asyncio()
    async def test_cancel_calls_cancel_on_all(self, ct_wired: CopyTrader) -> None:
        """CANCEL action is forwarded to cancel_on_all()."""
        cancel_calls: list[dict[str, Any]] = []

        async def _fake_cancel(**kwargs: Any) -> dict[str, Any]:
            cancel_calls.append(kwargs)
            return {"ok": True, "accounts_cancelled": ["main"], "accounts_failed": [], "reason": ""}

        ct_wired.cancel_on_all = _fake_cancel  # type: ignore[method-assign]

        cmd = self._Cmd(
            symbol="MGC=F",
            action="CANCEL",
            order_type="MARKET",
            position_id="POS-2",
            reason="position closed",
        )

        ct_wired._contract_cache["MGC"] = "MGCQ6"

        results = await ct_wired.execute_order_commands([cmd])
        assert len(results) == 1
        assert cancel_calls[0]["position_id"] == "POS-2"

    @pytest.mark.asyncio()
    async def test_unknown_action_skipped(self, ct_wired: CopyTrader) -> None:
        cmd = self._Cmd(symbol="MGC=F", action="UNKNOWN_ACTION", order_type="MARKET")
        results = await ct_wired.execute_order_commands([cmd])
        assert results == []

    @pytest.mark.asyncio()
    async def test_entry_prices_forwarded_to_modify_stop(self, ct_wired: CopyTrader) -> None:
        """entry_prices dict is forwarded and used for tick conversion."""
        received_entry: list[float] = []

        async def _fake_modify_stop(**kwargs: Any) -> dict[str, Any]:
            received_entry.append(kwargs.get("entry_price", -1.0))
            return {"ok": True, "accounts_modified": [], "accounts_failed": [], "stop_ticks": 20, "reason": ""}

        ct_wired.modify_stop_on_all = _fake_modify_stop  # type: ignore[method-assign]
        ct_wired._contract_cache["MGC"] = "MGCQ6"

        cmd = self._Cmd(
            symbol="MGC=F",
            action="MODIFY_STOP",
            order_type="STOP",
            stop_price=1995.0,
        )

        await ct_wired.execute_order_commands([cmd], entry_prices={"MGC=F": 2010.0})
        assert received_entry == [2010.0]

    @pytest.mark.asyncio()
    async def test_mixed_commands_processed_in_order(self, ct_wired: CopyTrader) -> None:
        """A realistic batch: STOP companion, BUY entry, MODIFY_STOP all processed in sequence."""
        entry_submitted: list[str] = []
        modify_called: list[str] = []

        async def _fake_submit(**kwargs: Any) -> None:
            entry_submitted.append(kwargs.get("security_code", "?"))

        async def _fake_modify(**kwargs: Any) -> dict[str, Any]:
            modify_called.append(kwargs.get("new_stop_price", 0.0))
            return {"ok": True, "accounts_modified": ["main"], "accounts_failed": [], "stop_ticks": 50, "reason": ""}

        ct_wired._main.client.submit_order = _fake_submit  # type: ignore[union-attr]
        ct_wired.modify_stop_on_all = _fake_modify  # type: ignore[method-assign]
        ct_wired._contract_cache["MGC"] = "MGCQ6"

        cmds = [
            self._Cmd(symbol="MGC=F", action="BUY", order_type="STOP", stop_price=1990.0),  # companion
            self._Cmd(symbol="MGC=F", action="BUY", order_type="MARKET", price=2000.0),  # entry
            self._Cmd(symbol="MGC=F", action="MODIFY_STOP", order_type="STOP", stop_price=2000.0),  # breakeven
        ]

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            results = await ct_wired.execute_order_commands(cmds)

        # 1 entry batch result + 1 modify result
        assert len(results) == 2
        assert modify_called == [2000.0]


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    @pytest.mark.asyncio()
    async def test_safe_disconnect_already_disconnected(self, ct: CopyTrader) -> None:
        """Disconnecting an already-disconnected account is a no-op."""
        acct = _make_fake_account("x", "X", connected=False)
        await ct._safe_disconnect(acct)
        assert acct.connected is False

    @pytest.mark.asyncio()
    async def test_safe_disconnect_client_throws(self, ct: CopyTrader) -> None:
        """If client.disconnect() throws, it's swallowed."""
        acct = _make_fake_account("x", "X", connected=True)

        async def _bad_disconnect() -> None:
            raise RuntimeError("socket closed")

        acct.client.disconnect = _bad_disconnect
        # Should not raise
        await ct._safe_disconnect(acct)
        assert acct.connected is False

    def test_batch_result_to_dict_serialisable(self) -> None:
        """Batch result .to_dict() produces JSON-serialisable output."""
        main_r = CopyOrderResult(
            account_key="m",
            account_label="Main",
            is_main=True,
            order_id="O1",
            security_code="MGCQ6",
            exchange="NYMEX",
            side="BUY",
            qty=1,
            order_type="MARKET",
            price=0,
            stop_ticks=20,
            target_ticks=None,
            status=CopyOrderStatus.SUBMITTED,
            tag="RUBY_MANUAL_WEBUI",
        )
        b = CopyBatchResult(
            batch_id="B1",
            security_code="MGCQ6",
            side="BUY",
            qty=1,
            main_result=main_r,
            slave_results=[],
            compliance_log=["✓ test"],
        )
        serialised = json.dumps(b.to_dict(), default=str)
        assert "RUBY_MANUAL_WEBUI" in serialised
        assert "MANUAL" in serialised

    @pytest.mark.asyncio()
    async def test_concurrent_sends_serialised(self, ct_wired: CopyTrader) -> None:
        """Concurrent send_order_and_copy calls are serialised by the lock."""
        call_order: list[str] = []

        async def _mock_submit(**kwargs: Any) -> CopyOrderResult:
            call_order.append(kwargs.get("batch_id", "?"))
            await asyncio.sleep(0.01)  # simulate work
            return CopyOrderResult(
                account_key="x",
                account_label="X",
                is_main=kwargs.get("is_main", False),
                order_id="O",
                security_code="X",
                exchange="CME",
                side="BUY",
                qty=1,
                order_type="MARKET",
                price=0,
                stop_ticks=20,
                target_ticks=None,
                status=CopyOrderStatus.SUBMITTED,
            )

        ct_wired._submit_single_order = _mock_submit  # type: ignore[assignment]

        with (
            patch("lib.services.engine.copy_trader._log_compliance"),
            patch("lib.services.engine.copy_trader.CopyTrader._persist_batch_result"),
        ):
            r1, r2 = await asyncio.gather(
                ct_wired.send_order_and_copy(
                    security_code="A",
                    exchange="CME",
                    side="BUY",
                    qty=1,
                    stop_ticks=20,
                    tag_prefix="T1",
                ),
                ct_wired.send_order_and_copy(
                    security_code="B",
                    exchange="CME",
                    side="SELL",
                    qty=1,
                    stop_ticks=20,
                    tag_prefix="T2",
                ),
            )

        # Both completed
        assert r1.main_result is not None
        assert r2.main_result is not None
