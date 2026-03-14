"""
Tests for the Massive.com futures data provider.

Covers:
  - MassiveDataProvider initialisation and availability detection
  - Yahoo → Massive ticker mapping
  - Front-month contract resolution (mocked)
  - Aggregate (OHLCV) fetching and resampling (mocked)
  - Daily bar fetching (mocked)
  - Snapshot fetching (mocked)
  - Recent trades fetching (mocked)
  - Active contract listing (mocked)
  - Contract cache invalidation
  - Timestamp parsing helper
  - Resampling helper (_resample_to_interval)
  - MassiveFeedManager lifecycle, subscription building, message handling
  - Raw JSON dict messages (wire protocol: ev/sym/o/h/l/c/v/s/e/p/t/bp/ap)
  - SDK object messages (backwards-compat: event_type/symbol/open/high/…)
  - get_data_source() in cache.py
  - Singleton provider management
  - Graceful error handling / fallback paths
"""

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from lib.integrations.massive_client import (
    INTERVAL_TO_RESOLUTION,
    MASSIVE_PRODUCT_TO_YAHOO,
    PERIOD_TO_DAYS,
    YAHOO_TO_MASSIVE_PRODUCT,
    MassiveDataProvider,
    MassiveFeedManager,
    _parse_timestamp_index,
    _resample_to_interval,
    get_massive_aggs,
    get_massive_daily,
    get_massive_provider,
    get_massive_snapshot,
    is_massive_available,
    reset_provider,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the singleton provider is reset between tests."""
    reset_provider()
    yield
    reset_provider()


@pytest.fixture()
def mock_rest_client():
    """Create a mock RESTClient."""
    client = MagicMock()
    return client


def _make_provider_no_key():
    """Create a provider with no API key."""
    with patch.dict(os.environ, {}, clear=False):
        env = os.environ.copy()
        env.pop("MASSIVE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            return MassiveDataProvider(api_key="")


def _make_provider_with_mock(mock_client):
    """Create a provider with a mocked REST client."""
    with patch("massive.RESTClient", return_value=mock_client):
        provider = MassiveDataProvider(api_key="test_key_12345")
    return provider


def _make_futures_agg(
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 99.0,
    close: float = 103.0,
    volume: int = 1000,
    window_start: str | int = "2025-01-06T08:00:00-03:00",
    ticker: str = "ESZ5",
) -> SimpleNamespace:
    """Create a mock FuturesAgg object."""
    agg = SimpleNamespace(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        window_start=window_start,
        ticker=ticker,
        transactions=500,
        dollar_volume=100000,
        session_end_date=None,
        settlement_price=None,
    )
    return agg


def _make_futures_contract(
    ticker: str | None = "ESZ5",
    product_code: str = "ES",
    active: bool = True,
    first_trade_date: str = "2025-06-01",
    last_trade_date: str = "2025-12-19",
    name: str = "E-mini S&P 500 Dec 2025",
) -> SimpleNamespace:
    """Create a mock FuturesContract object."""
    return SimpleNamespace(
        ticker=ticker,
        product_code=product_code,
        active=active,
        first_trade_date=first_trade_date,
        last_trade_date=last_trade_date,
        name=name,
        settlement_date=None,
        days_to_maturity=30,
        trading_venue="CME",
        type="single",
        group_code=None,
        max_order_quantity=None,
        min_order_quantity=None,
        settlement_tick_size=None,
        spread_tick_size=None,
        trade_tick_size=None,
    )


def _make_futures_snapshot(
    ticker="ESZ5",
    product_code="ES",
    last_price=5500.0,
    bid=5499.75,
    ask=5500.25,
    session_open=5480.0,
    session_high=5510.0,
    session_low=5475.0,
    session_volume=500000,
):
    """Create a mock FuturesSnapshot object."""
    return SimpleNamespace(
        ticker=ticker,
        product_code=product_code,
        last_trade=SimpleNamespace(price=last_price, size=5, last_updated=None, timeframe=None),
        last_quote=SimpleNamespace(
            bid=bid,
            bid_size=100,
            ask=ask,
            ask_size=80,
            bid_timestamp=None,
            ask_timestamp=None,
            last_updated=None,
            timeframe=None,
        ),
        session=SimpleNamespace(
            open=session_open,
            high=session_high,
            low=session_low,
            close=last_price,
            volume=session_volume,
            change=20.0,
            change_percent=0.36,
            settlement_price=5498.0,
            previous_settlement=5480.0,
        ),
        last_minute=SimpleNamespace(
            open=5499.0,
            high=5500.5,
            low=5498.5,
            close=5500.0,
            volume=1200.0,
            last_updated=None,
            timeframe=None,
        ),
        details=SimpleNamespace(open_interest=2500000, settlement_date=None),
    )


def _make_futures_trade(price=5500.0, size=3, timestamp=1736150400000000000):
    """Create a mock FuturesTrade object."""
    return SimpleNamespace(
        price=price,
        size=size,
        timestamp=timestamp,
        sequence_number=12345,
        ticker="ESZ5",
        session_end_date=None,
        report_sequence=None,
    )


def _make_ohlcv_df(n=100, freq="1min", start="2025-01-06 03:00"):
    """Create a simple OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz=_EST)
    close = 5500.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": close + rng.uniform(-0.5, 0.5, n),
            "High": close + rng.uniform(0.5, 1.5, n),
            "Low": close - rng.uniform(0.5, 1.5, n),
            "Close": close,
            "Volume": rng.poisson(1000, n).astype(float),
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Mapping constants
# ═══════════════════════════════════════════════════════════════════════════


class TestMappingConstants:
    """Verify the ticker mapping dictionaries are consistent."""

    def test_yahoo_to_massive_has_all_main_tickers(self):
        main = ["ES=F", "NQ=F", "GC=F", "SI=F", "HG=F", "CL=F"]
        for t in main:
            assert t in YAHOO_TO_MASSIVE_PRODUCT, f"Missing {t}"

    def test_yahoo_to_massive_micro_tickers(self):
        micro = ["MES=F", "MNQ=F", "MGC=F", "SIL=F", "MHG=F", "MCL=F"]
        for t in micro:
            assert t in YAHOO_TO_MASSIVE_PRODUCT, f"Missing micro {t}"

    def test_reverse_mapping_is_inverse(self):
        for yahoo, product in YAHOO_TO_MASSIVE_PRODUCT.items():
            assert product in MASSIVE_PRODUCT_TO_YAHOO
            assert MASSIVE_PRODUCT_TO_YAHOO[product] == yahoo

    def test_interval_to_resolution_coverage(self):
        expected = ["1m", "5m", "15m", "30m", "1h", "60m", "1d"]
        for i in expected:
            assert i in INTERVAL_TO_RESOLUTION, f"Missing interval {i}"

    def test_period_to_days_coverage(self):
        expected = ["1d", "5d", "10d", "15d", "1mo"]
        for p in expected:
            assert p in PERIOD_TO_DAYS, f"Missing period {p}"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — MassiveDataProvider init & availability
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderInit:
    """Test MassiveDataProvider initialisation and availability."""

    def test_no_api_key_is_unavailable(self):
        provider = _make_provider_no_key()
        assert not provider.is_available
        assert provider.api_key == ""

    def test_with_api_key_tries_to_init_client(self, mock_rest_client):
        provider = _make_provider_with_mock(mock_rest_client)
        assert provider.is_available
        assert provider.api_key == "test_key_12345"

    def test_client_init_error_makes_unavailable(self):
        with patch("massive.RESTClient", side_effect=Exception("auth fail")):
            provider = MassiveDataProvider(api_key="bad_key")
        assert not provider.is_available

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"MASSIVE_API_KEY": "env_key_123"}), patch("massive.RESTClient"):
            provider = MassiveDataProvider()
            assert provider.api_key == "env_key_123"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Contract resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestContractResolution:
    """Test front-month contract resolution."""

    def test_resolve_front_month_success(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5", product_code="ES")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_front_month("ES")
        assert result == "ESZ5"

    def test_resolve_front_month_picks_nearest_expiry(self, mock_rest_client):
        c1 = _make_futures_contract(ticker="ESH6", last_trade_date="2026-03-20")
        c2 = _make_futures_contract(ticker="ESZ5", last_trade_date="2025-12-19")
        mock_rest_client.list_futures_contracts.return_value = [c1, c2]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_front_month("ES")
        assert result == "ESZ5"  # nearest expiry

    def test_resolve_front_month_no_contracts(self, mock_rest_client):
        mock_rest_client.list_futures_contracts.return_value = []
        provider = _make_provider_with_mock(mock_rest_client)

        # Tier 3 fallback: returns root symbol itself when no contracts found
        result = provider.resolve_front_month("UNKNOWN")
        assert result == "UNKNOWN"

    def test_resolve_front_month_api_error(self, mock_rest_client):
        mock_rest_client.list_futures_contracts.side_effect = Exception("timeout")
        provider = _make_provider_with_mock(mock_rest_client)

        # Tier 3 fallback: returns root symbol itself when API errors out
        result = provider.resolve_front_month("ES")
        assert result == "ES"

    def test_resolve_front_month_caches_result(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        r1 = provider.resolve_front_month("ES")
        r2 = provider.resolve_front_month("ES")
        assert r1 == r2 == "ESZ5"
        # Should only call the API once
        assert mock_rest_client.list_futures_contracts.call_count == 1

    def test_resolve_from_yahoo(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_from_yahoo("ES=F")
        assert result == "ESZ5"

    def test_resolve_from_yahoo_unknown_ticker(self, mock_rest_client):
        provider = _make_provider_with_mock(mock_rest_client)
        result = provider.resolve_from_yahoo("FAKE=F")
        assert result is None

    def test_resolve_unavailable_provider(self):
        provider = _make_provider_no_key()
        assert provider.resolve_front_month("ES") is None

    def test_get_all_front_months(self, mock_rest_client):
        es_contract = _make_futures_contract(ticker="ESZ5", product_code="ES")
        nq_contract = _make_futures_contract(ticker="NQZ5", product_code="NQ")

        def mock_list(**kwargs):
            pc = kwargs.get("product_code", "")
            if pc == "ES":
                return [es_contract]
            elif pc == "NQ":
                return [nq_contract]
            return []

        mock_rest_client.list_futures_contracts.side_effect = mock_list
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_all_front_months(["ES=F", "NQ=F", "FAKE=F"])
        assert "ES=F" in result
        assert "NQ=F" in result
        assert "FAKE=F" not in result

    def test_invalidate_contract_cache(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        provider.resolve_front_month("ES")
        assert mock_rest_client.list_futures_contracts.call_count == 1

        provider.invalidate_contract_cache("ES")
        provider.resolve_front_month("ES")
        assert mock_rest_client.list_futures_contracts.call_count == 2

    def test_invalidate_all_contract_cache(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        provider.resolve_front_month("ES")
        provider.invalidate_contract_cache()  # clear all
        provider.resolve_front_month("ES")
        assert mock_rest_client.list_futures_contracts.call_count == 2

    def test_contract_with_no_ticker_skipped(self, mock_rest_client):
        c1 = _make_futures_contract(ticker=None, product_code="ES")
        c2 = _make_futures_contract(ticker="ESZ5", product_code="ES")
        mock_rest_client.list_futures_contracts.return_value = [c1, c2]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_front_month("ES")
        assert result == "ESZ5"

    def test_combo_contracts_filtered_out(self, mock_rest_client):
        """Products like CL return many combo/spread contracts that must be
        excluded so the outright front-month is found (regression test)."""
        combos = [
            _make_futures_contract(
                ticker=f"CL:BF F6-G6-H{i}",
                product_code="CL",
                last_trade_date="2026-12-19",
            )
            for i in range(15)
        ]
        # Override type to 'combo' and keep the dash/colon in ticker
        for c in combos:
            c.type = "combo"
        outright = _make_futures_contract(
            ticker="CLJ6",
            product_code="CL",
            last_trade_date="2026-03-20",
        )
        mock_rest_client.list_futures_contracts.return_value = combos + [outright]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_front_month("CL")
        assert result == "CLJ6"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Aggregates (OHLCV) fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregates:
    """Test aggregate data fetching and processing."""

    def test_get_aggs_basic(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]

        aggs = [
            _make_futures_agg(
                open_=5500,
                high=5510,
                low=5495,
                close=5505,
                volume=1000,
                window_start=1736150400000000000,
            ),
            _make_futures_agg(
                open_=5505,
                high=5515,
                low=5500,
                close=5512,
                volume=800,
                window_start=1736150460000000000,
            ),
        ]
        mock_rest_client.list_futures_aggregates.return_value = aggs
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="1m", period="1d")
        assert not df.empty
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) == 2

    def test_get_aggs_empty_response(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.return_value = []
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="5m", period="5d")
        assert df.empty

    def test_get_aggs_api_error(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.side_effect = Exception("API error")
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="5m", period="5d")
        assert df.empty

    def test_get_aggs_unknown_ticker(self, mock_rest_client):
        provider = _make_provider_with_mock(mock_rest_client)
        df = provider.get_aggs("FAKE=F", interval="5m", period="5d")
        assert df.empty

    def test_get_aggs_provider_unavailable(self):
        provider = _make_provider_no_key()
        df = provider.get_aggs("ES=F", interval="5m", period="5d")
        assert df.empty

    def test_get_aggs_resamples_5m(self, mock_rest_client):
        """Verify that 1-minute bars are resampled to 5-minute bars."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]

        # Create 10 one-minute bars with nanosecond timestamps
        base_ns = 1736150400000000000  # some base time
        aggs = []
        for i in range(10):
            aggs.append(
                _make_futures_agg(
                    open_=5500 + i,
                    high=5510 + i,
                    low=5495 + i,
                    close=5505 + i,
                    volume=100,
                    window_start=base_ns + i * 60 * 1_000_000_000,
                )
            )
        mock_rest_client.list_futures_aggregates.return_value = aggs
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="5m", period="1d")
        # 10 x 1m bars → 2 x 5m bars
        assert len(df) == 2

    def test_get_aggs_custom_period(self, mock_rest_client):
        """Test with non-standard period like '10d'."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.return_value = [
            _make_futures_agg(window_start=1736150400000000000),
        ]
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="1m", period="10d")
        assert not df.empty

    def test_get_daily(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.return_value = [
            _make_futures_agg(window_start="2025-01-06"),
            _make_futures_agg(window_start="2025-01-07"),
        ]
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_daily("ES=F", period="10d")
        assert not df.empty

    def test_get_aggs_with_iso_timestamps(self, mock_rest_client):
        """Test handling of ISO 8601 timestamp strings."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.return_value = [
            _make_futures_agg(window_start="2025-01-06T08:00:00-03:00"),
            _make_futures_agg(window_start="2025-01-06T08:01:00-03:00"),
        ]
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="1m", period="1d")
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_aggs_drops_nan_ohlc(self, mock_rest_client):
        """Bars with missing OHLC values should be dropped."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]

        good_agg = _make_futures_agg(
            open_=5500,
            high=5510,
            low=5495,
            close=5505,
            window_start=1736150400000000000,
        )
        bad_agg = SimpleNamespace(
            open=None,
            high=None,
            low=None,
            close=None,
            volume=0,
            window_start=1736150460000000000,
            ticker="ESZ5",
        )
        mock_rest_client.list_futures_aggregates.return_value = [good_agg, bad_agg]
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="1m", period="1d")
        assert len(df) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Snapshots
# ═══════════════════════════════════════════════════════════════════════════


class TestSnapshots:
    """Test real-time snapshot fetching."""

    def test_get_snapshot_basic(self, mock_rest_client):
        snap = _make_futures_snapshot()
        mock_rest_client.get_futures_snapshot.return_value = [snap]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is not None
        assert result["ticker"] == "ESZ5"
        assert result["last_price"] == 5500.0
        assert result["bid"] == 5499.75
        assert result["ask"] == 5500.25
        assert result["session_open"] == 5480.0
        assert result["session_high"] == 5510.0
        assert result["session_volume"] == 500000
        assert result["open_interest"] == 2500000
        assert result["minute_close"] == 5500.0

    def test_get_snapshot_empty(self, mock_rest_client):
        mock_rest_client.get_futures_snapshot.return_value = []
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is None

    def test_get_snapshot_unknown_ticker(self, mock_rest_client):
        provider = _make_provider_with_mock(mock_rest_client)
        result = provider.get_snapshot(yahoo_ticker="FAKE=F")
        assert result is None

    def test_get_snapshot_provider_unavailable(self):
        provider = _make_provider_no_key()
        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is None

    def test_get_snapshot_api_error(self, mock_rest_client):
        mock_rest_client.get_futures_snapshot.side_effect = Exception("err")
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is None

    def test_get_snapshot_by_product_code(self, mock_rest_client):
        snap = _make_futures_snapshot(product_code="NQ")
        mock_rest_client.get_futures_snapshot.return_value = [snap]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(product_code="NQ")
        assert result is not None
        assert result["product_code"] == "NQ"

    def test_get_all_snapshots(self, mock_rest_client):
        es_snap = _make_futures_snapshot(ticker="ESZ5", product_code="ES")
        nq_snap = _make_futures_snapshot(ticker="NQZ5", product_code="NQ")
        mock_rest_client.get_futures_snapshot.return_value = [es_snap, nq_snap]
        provider = _make_provider_with_mock(mock_rest_client)

        results = provider.get_all_snapshots(["ES=F", "NQ=F"])
        assert "ES=F" in results
        assert "NQ=F" in results

    def test_get_all_snapshots_api_error(self, mock_rest_client):
        mock_rest_client.get_futures_snapshot.side_effect = Exception("err")
        provider = _make_provider_with_mock(mock_rest_client)

        results = provider.get_all_snapshots(["ES=F"])
        assert results == {}

    def test_snapshot_missing_subfields(self, mock_rest_client):
        """Snapshot with None sub-objects should still work."""
        snap = SimpleNamespace(
            ticker="ESZ5",
            product_code="ES",
            last_trade=None,
            last_quote=None,
            session=None,
            last_minute=None,
            details=None,
        )
        mock_rest_client.get_futures_snapshot.return_value = [snap]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is not None
        assert result["ticker"] == "ESZ5"
        assert "last_price" not in result  # no last_trade


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Recent trades
# ═══════════════════════════════════════════════════════════════════════════


class TestRecentTrades:
    """Test recent trade fetching for CVD."""

    def test_get_recent_trades(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        trades = [
            _make_futures_trade(price=5500.0, size=3),
            _make_futures_trade(price=5500.25, size=5),
            _make_futures_trade(price=5499.75, size=2),
        ]
        mock_rest_client.list_futures_trades.return_value = trades
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_recent_trades("ES=F", minutes_back=5)
        assert not df.empty
        assert len(df) == 3
        assert "price" in df.columns
        assert "size" in df.columns

    def test_get_recent_trades_empty(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_trades.return_value = []
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_recent_trades("ES=F")
        assert df.empty

    def test_get_recent_trades_error(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_trades.side_effect = Exception("err")
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_recent_trades("ES=F")
        assert df.empty

    def test_get_recent_trades_unavailable(self):
        provider = _make_provider_no_key()
        df = provider.get_recent_trades("ES=F")
        assert df.empty


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Active contract listing
# ═══════════════════════════════════════════════════════════════════════════


class TestActiveContracts:
    """Test active contract listing."""

    def test_get_active_contracts(self, mock_rest_client):
        contracts = [
            _make_futures_contract(ticker="ESZ5", last_trade_date="2025-12-19"),
            _make_futures_contract(ticker="ESH6", last_trade_date="2026-03-20"),
        ]
        mock_rest_client.list_futures_contracts.return_value = contracts
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_active_contracts("ES")
        assert len(result) == 2
        assert result[0]["ticker"] == "ESZ5"
        assert result[1]["ticker"] == "ESH6"

    def test_get_active_contracts_empty(self, mock_rest_client):
        mock_rest_client.list_futures_contracts.return_value = []
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_active_contracts("UNKNOWN")
        assert result == []

    def test_get_active_contracts_error(self, mock_rest_client):
        mock_rest_client.list_futures_contracts.side_effect = Exception("err")
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_active_contracts("ES")
        assert result == []

    def test_get_active_contracts_unavailable(self):
        provider = _make_provider_no_key()
        result = provider.get_active_contracts("ES")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Timestamp parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestTimestampParsing:
    """Test the _parse_timestamp_index helper."""

    def test_nanosecond_timestamps(self):
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "Close": [102, 103],
                "timestamp": [1736150400000000000, 1736150460000000000],
            }
        )
        result = _parse_timestamp_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "timestamp" not in result.columns

    def test_millisecond_timestamps(self):
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "Close": [102, 103],
                "timestamp": [1736150400000, 1736150460000],
            }
        )
        result = _parse_timestamp_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_second_timestamps(self):
        df = pd.DataFrame(
            {
                "Open": [100],
                "Close": [102],
                "timestamp": [1736150400],
            }
        )
        result = _parse_timestamp_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_iso_string_timestamps(self):
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "Close": [102, 103],
                "timestamp": ["2025-01-06T08:00:00-03:00", "2025-01-06T08:01:00-03:00"],
            }
        )
        result = _parse_timestamp_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_timestamp_column(self):
        df = pd.DataFrame({"Open": [100], "Close": [102]})
        result = _parse_timestamp_index(df)
        assert "timestamp" not in result.columns

    def test_empty_df(self):
        df = pd.DataFrame({"Open": [], "Close": [], "timestamp": []})
        result = _parse_timestamp_index(df)
        assert result.empty

    def test_timezone_conversion_to_eastern(self):
        df = pd.DataFrame(
            {
                "Open": [100],
                "Close": [102],
                "timestamp": [1736150400000000000],
            }
        )
        result = _parse_timestamp_index(df)
        # Index should be in Eastern time
        assert getattr(result.index, "tz", None) is not None
        tz_name = str(getattr(result.index, "tz", ""))
        assert "Eastern" in tz_name or "US/Eastern" in tz_name or "America/New_York" in tz_name


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Resampling
# ═══════════════════════════════════════════════════════════════════════════


class TestResampling:
    """Test the _resample_to_interval helper."""

    def test_resample_1m_to_5m(self):
        df = _make_ohlcv_df(n=10, freq="1min")
        result = _resample_to_interval(df, "5m")
        assert len(result) == 2
        assert result["Volume"].iloc[0] > 0

    def test_resample_1m_to_15m(self):
        df = _make_ohlcv_df(n=30, freq="1min")
        result = _resample_to_interval(df, "15m")
        assert len(result) == 2

    def test_resample_1m_to_30m(self):
        df = _make_ohlcv_df(n=60, freq="1min")
        result = _resample_to_interval(df, "30m")
        assert len(result) == 2

    def test_no_resample_for_1m(self):
        df = _make_ohlcv_df(n=10, freq="1min")
        result = _resample_to_interval(df, "1m")
        assert len(result) == 10  # unchanged

    def test_resample_1m_to_1h(self):
        df = _make_ohlcv_df(n=120, freq="1min")
        result = _resample_to_interval(df, "1h")
        assert len(result) == 2

    def test_no_resample_for_1d(self):
        df = _make_ohlcv_df(n=5, freq="1D")
        result = _resample_to_interval(df, "1d")
        assert len(result) == 5

    def test_empty_df_resample(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # type: ignore[call-overload]
        result = _resample_to_interval(df, "5m")
        assert result.empty

    def test_resample_preserves_ohlcv_semantics(self):
        """Open should be first, High should be max, Low should be min,
        Close should be last, Volume should be summed."""
        idx = pd.date_range("2025-01-06 03:00", periods=5, freq="1min", tz=_EST)
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [110.0, 111.0, 112.0, 113.0, 114.0],
                "Low": [90.0, 91.0, 92.0, 93.0, 94.0],
                "Close": [105.0, 106.0, 107.0, 108.0, 109.0],
                "Volume": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            index=idx,
        )

        result = _resample_to_interval(df, "5m")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["Open"] == 100.0  # first
        assert row["High"] == 114.0  # max
        assert row["Low"] == 90.0  # min
        assert row["Close"] == 109.0  # last
        assert row["Volume"] == 150.0  # sum


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — MassiveFeedManager
# ═══════════════════════════════════════════════════════════════════════════


class TestFeedManager:
    """Test WebSocket feed manager lifecycle and message handling."""

    def test_init_defaults(self):
        feed = MassiveFeedManager()
        assert not feed.is_connected
        assert not feed.is_running
        assert feed.msg_count == 0
        assert feed.bar_count == 0
        assert feed.trade_count == 0

    def test_start_without_api_key(self):
        feed = MassiveFeedManager(api_key="")
        result = feed.start()
        assert not result
        assert not feed.is_running

    def test_start_without_resolved_tickers(self, mock_rest_client):
        """Start succeeds with broad subscriptions even if no tickers resolve.

        Broad subscriptions (AM.*, T.*) auto-discover contracts via reverse
        mapping, so the feed manager returns True and relies on runtime
        discovery instead of upfront resolution.
        """
        provider = _make_provider_with_mock(mock_rest_client)
        mock_rest_client.list_futures_contracts.return_value = []

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["FAKE=F"],
            provider=provider,
        )
        result = feed.start()
        assert result

    def test_get_status(self):
        feed = MassiveFeedManager(api_key="test")
        status = feed.get_status()
        assert "connected" in status
        assert "running" in status
        assert "bar_count" in status
        assert "trade_count" in status
        assert "msg_count" in status
        assert status["connected"] is False

    def test_build_subscriptions(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_trades=True,
            subscribe_quotes=False,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        # Default is broad subscriptions (AM.*, T.*)
        assert any("AM." in s for s in subs)  # minute aggs
        assert any("T." in s for s in subs)  # trades
        assert not any("Q." in s for s in subs)  # no quotes

    def test_build_subscriptions_per_ticker(self, mock_rest_client):
        """Legacy per-ticker subscriptions when broad mode is disabled."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_trades=True,
            subscribe_quotes=False,
            use_broad_subscriptions=False,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        assert any("AM.ESZ5" in s for s in subs)  # minute aggs
        assert any("T.ESZ5" in s for s in subs)  # trades
        assert not any("Q." in s for s in subs)  # no quotes

    def test_build_subscriptions_with_quotes(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_trades=True,
            subscribe_quotes=True,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        # Broad mode: Q.*
        assert any("Q." in s for s in subs)

    def test_build_subscriptions_with_quotes_per_ticker(self, mock_rest_client):
        """Per-ticker quote subscriptions when broad mode is disabled."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_trades=True,
            subscribe_quotes=True,
            use_broad_subscriptions=False,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        assert any("Q.ESZ5" in s for s in subs)

    def test_build_subscriptions_with_second_aggs(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_second_aggs=True,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        # Broad mode: A.*
        assert any("A." in s for s in subs)

    def test_build_subscriptions_with_second_aggs_per_ticker(self, mock_rest_client):
        """Per-ticker second-agg subscriptions when broad mode is disabled."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_second_aggs=True,
            use_broad_subscriptions=False,
        )
        feed._resolve_tickers()
        subs = feed._build_subscriptions()

        assert any("A.ESZ5" in s for s in subs)

    def test_handle_bar_message_raw_dict(self, mock_rest_client):
        """Raw JSON dict bar message (wire protocol: ev/sym/o/h/l/c/v/s/e)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "AM",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5510.0,
            "l": 5495.0,
            "c": 5505.0,
            "v": 1000,
            "n": 500,
            "s": 1736150400000,
            "e": 1736150460000,
            "vw": 5502.5,
        }

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        assert feed.msg_count == 1
        assert feed.bar_count == 1
        bars = feed.latest_bars
        assert "ESZ5" in bars
        assert bars["ESZ5"]["close"] == 5505.0
        assert bars["ESZ5"]["open"] == 5500.0
        assert bars["ESZ5"]["volume"] == 1000
        assert bars["ESZ5"]["transactions"] == 500

    def test_handle_bar_message_sdk_object(self, mock_rest_client):
        """SDK model object bar message (backwards-compat path)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = SimpleNamespace(
            event_type="AM",
            symbol="ESZ5",
            open=5500.0,
            high=5510.0,
            low=5495.0,
            close=5505.0,
            volume=1000,
            transactions=500,
            start_timestamp=1736150400000,
            end_timestamp=1736150460000,
            total_value=5500000.0,
        )

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        assert feed.msg_count == 1
        assert feed.bar_count == 1
        bars = feed.latest_bars
        assert "ESZ5" in bars
        assert bars["ESZ5"]["close"] == 5505.0

    def test_handle_second_agg_raw_dict(self, mock_rest_client):
        """Per-second aggregate (A channel) via raw dict."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "A",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5501.0,
            "l": 5499.5,
            "c": 5500.5,
            "v": 42,
            "s": 1736150400000,
            "e": 1736150401000,
        }

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        assert feed.bar_count == 1
        bars = feed.latest_bars
        assert bars["ESZ5"]["close"] == 5500.5

    def test_handle_trade_message_raw_dict(self, mock_rest_client):
        """Raw JSON dict trade message (wire protocol: ev/sym/p/s/t/q)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "T",
            "sym": "ESZ5",
            "p": 5500.25,
            "s": 3,
            "t": 1736150400000,
            "q": 12345,
        }

        feed._handle_messages([msg])

        assert feed.trade_count == 1
        trades = feed.latest_trades
        assert "ESZ5" in trades
        assert trades["ESZ5"]["price"] == 5500.25
        assert trades["ESZ5"]["size"] == 3

    def test_handle_trade_message_sdk_object(self, mock_rest_client):
        """SDK model object trade message (backwards-compat path)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = SimpleNamespace(
            event_type="T",
            symbol="ESZ5",
            price=5500.25,
            size=3,
            timestamp=1736150400000,
            sequence_number=12345,
        )

        feed._handle_messages([msg])

        assert feed.trade_count == 1
        trades = feed.latest_trades
        assert "ESZ5" in trades
        assert trades["ESZ5"]["price"] == 5500.25

    def test_handle_quote_message_raw_dict(self, mock_rest_client):
        """Raw JSON dict quote message (wire protocol: ev/sym/bp/bs/ap/as/t)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "Q",
            "sym": "ESZ5",
            "bp": 5499.75,
            "bs": 100,
            "ap": 5500.25,
            "as": 80,
            "t": 1736150400000,
        }

        feed._handle_messages([msg])

        quotes = feed._latest_quotes
        assert "ESZ5" in quotes
        assert quotes["ESZ5"]["bid"] == 5499.75
        assert quotes["ESZ5"]["ask"] == 5500.25
        assert quotes["ESZ5"]["bid_size"] == 100
        assert quotes["ESZ5"]["ask_size"] == 80

    def test_handle_quote_message_sdk_object(self, mock_rest_client):
        """SDK model object quote message (backwards-compat path)."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = SimpleNamespace(
            event_type="Q",
            symbol="ESZ5",
            bid_price=5499.75,
            bid_size=100,
            ask_price=5500.25,
            ask_size=80,
            bid_timestamp=None,
            ask_timestamp=None,
            sip_timestamp=None,
        )

        feed._handle_messages([msg])

        quotes = feed._latest_quotes
        assert "ESZ5" in quotes
        assert quotes["ESZ5"]["bid"] == 5499.75
        assert quotes["ESZ5"]["ask"] == 5500.25

    def test_trade_buffer(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        # Push 5 trade messages using raw dict format
        for i in range(5):
            msg = {
                "ev": "T",
                "sym": "ESZ5",
                "p": 5500.0 + i * 0.25,
                "s": 1 + i,
                "t": 1736150400000 + i,
                "q": i,
            }
            feed._handle_messages([msg])

        # Read buffer
        buffer = feed.get_trade_buffer("ES=F", clear=False)
        assert len(buffer) == 5
        assert buffer[0]["price"] == 5500.0

        # Read and clear
        buffer2 = feed.get_trade_buffer("ES=F", clear=True)
        assert len(buffer2) == 5

        # Now buffer should be empty
        buffer3 = feed.get_trade_buffer("ES=F")
        assert len(buffer3) == 0

    def test_trade_buffer_unknown_ticker(self):
        feed = MassiveFeedManager(api_key="test")
        result = feed.get_trade_buffer("FAKE=F")
        assert result == []

    def test_trade_buffer_cap(self, mock_rest_client):
        """Buffer should be capped at 10000 trades then trimmed to 5000."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        # Push 10001 trades using raw dict format
        for i in range(10001):
            msg = {
                "ev": "T",
                "sym": "ESZ5",
                "p": 5500.0,
                "s": 1,
                "t": i,
                "q": i,
            }
            feed._handle_messages([msg])

        buffer = feed.get_trade_buffer("ES=F", clear=False)
        assert len(buffer) <= 10000

    def test_on_bar_callback_raw_dict(self, mock_rest_client):
        """Bar callback fires with yahoo_ticker and parsed bar from raw dict."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        received = []
        feed.on_bar(lambda ticker, bar: received.append((ticker, bar)))

        msg = {
            "ev": "AM",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5510.0,
            "l": 5495.0,
            "c": 5505.0,
            "v": 1000,
            "n": 500,
            "s": 1736150400000,
            "e": 1736150460000,
            "vw": 5502.5,
        }
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        assert len(received) == 1
        assert received[0][0] == "ES=F"
        assert received[0][1]["close"] == 5505.0

    def test_on_bar_callback_sdk_object(self, mock_rest_client):
        """Bar callback fires from SDK model object."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        received = []
        feed.on_bar(lambda ticker, bar: received.append((ticker, bar)))

        msg = SimpleNamespace(
            event_type="AM",
            symbol="ESZ5",
            open=5500.0,
            high=5510.0,
            low=5495.0,
            close=5505.0,
            volume=1000,
            transactions=500,
            start_timestamp=1736150400000,
            end_timestamp=1736150460000,
            total_value=5500000.0,
        )
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        assert len(received) == 1
        assert received[0][0] == "ES=F"
        assert received[0][1]["close"] == 5505.0

    def test_on_trade_callback(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        received = []
        feed.on_trade(lambda ticker, trade: received.append((ticker, trade)))

        msg = {
            "ev": "T",
            "sym": "ESZ5",
            "p": 5500.25,
            "s": 3,
            "t": 1736150400000,
            "q": 12345,
        }
        feed._handle_messages([msg])

        assert len(received) == 1
        assert received[0][0] == "ES=F"
        assert received[0][1]["price"] == 5500.25

    def test_callback_error_doesnt_crash(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        feed.on_bar(lambda t, b: 1 / 0)  # will raise ZeroDivisionError

        msg = {
            "ev": "AM",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5510.0,
            "l": 5495.0,
            "c": 5505.0,
            "v": 1000,
            "s": 1736150400000,
            "e": 1736150460000,
        }
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])  # should not raise

        assert feed.bar_count == 1

    def test_handle_empty_messages(self):
        feed = MassiveFeedManager(api_key="test")
        feed._handle_messages([])
        feed._handle_messages(None)
        assert feed.msg_count == 0

    def test_handle_status_message_dict(self, mock_rest_client):
        """Status/auth messages (ev=status) should be silently ignored."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {"ev": "status", "status": "auth_success", "message": "authenticated"}
        feed._handle_messages([msg])
        assert feed.bar_count == 0
        assert feed.trade_count == 0

    def test_handle_unknown_event_type_sdk(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = SimpleNamespace(event_type="status", symbol=None, message="connected")
        feed._handle_messages([msg])
        assert feed.bar_count == 0
        assert feed.trade_count == 0

    def test_handle_bar_no_symbol_raw_dict(self, mock_rest_client):
        feed = MassiveFeedManager(api_key="test_key")
        msg = {
            "ev": "AM",
            "sym": None,
            "o": 100.0,
            "h": 105.0,
            "l": 99.0,
            "c": 103.0,
            "v": 100,
        }
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_bar(msg, None)
        assert feed.bar_count == 0

    def test_handle_bar_no_symbol_sdk_object(self, mock_rest_client):
        feed = MassiveFeedManager(api_key="test_key")
        msg = SimpleNamespace(
            event_type="AM",
            symbol=None,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=100,
            transactions=10,
            start_timestamp=1736150400000,
            end_timestamp=1736150460000,
            total_value=0,
        )
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_bar(msg, None)
        assert feed.bar_count == 0

    def test_stop_idempotent(self):
        feed = MassiveFeedManager(api_key="test")
        asyncio.run(feed.stop())  # should not crash
        asyncio.run(feed.stop())
        assert not feed.is_running

    def test_push_bar_to_cache(self, mock_rest_client):
        """Test that bars get pushed to the cache layer."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        with patch("lib.core.cache.cache_set") as mock_cache:
            feed._push_bar_to_cache(
                "ESZ5",
                {
                    "open": 5500.0,
                    "high": 5510.0,
                    "low": 5495.0,
                    "close": 5505.0,
                    "volume": 1000,
                },
            )
            mock_cache.assert_called_once()

    def test_channel_prefix_constants(self):
        """Verify the channel prefix constants match the Massive futures WS docs."""
        assert MassiveFeedManager.PREFIX_MINUTE_AGG == "AM"
        assert MassiveFeedManager.PREFIX_SECOND_AGG == "A"
        assert MassiveFeedManager.PREFIX_TRADE == "T"
        assert MassiveFeedManager.PREFIX_QUOTE == "Q"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Singleton & convenience functions
# ═══════════════════════════════════════════════════════════════════════════


class TestSingletonAndConvenience:
    """Test singleton provider and convenience functions."""

    def test_singleton_returns_same_instance(self):
        with patch("lib.integrations.massive_client.MassiveDataProvider") as mock_cls:
            mock_cls.return_value = MagicMock(is_available=False, api_key="")
            p1 = get_massive_provider()
            p2 = get_massive_provider()
            assert p1 is p2
            assert mock_cls.call_count == 1

    def test_reset_provider_clears_singleton(self):
        with patch("lib.integrations.massive_client.MassiveDataProvider") as mock_cls:
            mock_cls.side_effect = [
                MagicMock(is_available=False, api_key=""),
                MagicMock(is_available=False, api_key=""),
            ]
            p1 = get_massive_provider()
            reset_provider()
            p2 = get_massive_provider()
            assert p1 is not p2
            assert mock_cls.call_count == 2

    def test_is_massive_available_false(self):
        with patch("lib.integrations.massive_client.MassiveDataProvider") as mock_cls:
            mock_cls.return_value = MagicMock(is_available=False, api_key="")
            result = is_massive_available()
            assert result is False

    def test_is_massive_available_true(self):
        with patch("lib.integrations.massive_client.MassiveDataProvider") as mock_cls:
            mock_cls.return_value = MagicMock(is_available=True, api_key="test")
            result = is_massive_available()
            assert result is True

    def test_get_massive_aggs_delegates(self):
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.get_aggs.return_value = pd.DataFrame({"Close": [100]})
        with patch("lib.integrations.massive_client.MassiveDataProvider", return_value=mock_provider):
            _df = get_massive_aggs("ES=F", "5m", "5d")  # noqa: F841
            mock_provider.get_aggs.assert_called_once_with("ES=F", "5m", "5d")

    def test_get_massive_daily_delegates(self):
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.get_daily.return_value = pd.DataFrame({"Close": [100]})
        with patch("lib.integrations.massive_client.MassiveDataProvider", return_value=mock_provider):
            _df = get_massive_daily("ES=F", "10d")  # noqa: F841
            mock_provider.get_daily.assert_called_once_with("ES=F", "10d")

    def test_get_massive_snapshot_delegates(self):
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.get_snapshot.return_value = {"last_price": 5500}
        with patch("lib.integrations.massive_client.MassiveDataProvider", return_value=mock_provider):
            _result = get_massive_snapshot("ES=F")  # noqa: F841
            mock_provider.get_snapshot.assert_called_once_with(yahoo_ticker="ES=F")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — cache.py integration (get_data_source)
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheIntegration:
    """Test that cache.py correctly integrates with Massive."""

    def test_get_data_source_yfinance_default(self):
        """Without Massive API key, data source should be yfinance."""
        import lib.core.cache as cache
        from lib.core.cache import get_data_source

        # Directly inject a mock provider that reports unavailable
        mock_provider = MagicMock()
        mock_provider.is_available = False
        cache._massive_checked = True
        cache._massive_provider = mock_provider

        result = get_data_source()
        assert result == "yfinance"

        # Cleanup
        cache._massive_checked = False
        cache._massive_provider = None

    def test_get_data_source_massive_when_available(self):
        import lib.core.cache as cache
        from lib.core.cache import get_data_source

        # Directly inject a mock provider that reports available
        mock_provider = MagicMock()
        mock_provider.is_available = True
        cache._massive_checked = True
        cache._massive_provider = mock_provider

        result = get_data_source()
        assert result == "Massive"

        # Cleanup
        cache._massive_checked = False
        cache._massive_provider = None


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Thread safety
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Verify that concurrent access patterns don't crash."""

    def test_concurrent_contract_resolution(self, mock_rest_client):
        """Multiple threads resolving contracts simultaneously."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        results = []
        errors = []

        def resolve():
            try:
                r = provider.resolve_front_month("ES")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert all(r == "ESZ5" for r in results)

    def test_concurrent_bar_reads(self, mock_rest_client):
        """Multiple threads reading latest_bars simultaneously."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        errors = []

        def write_bars():
            for i in range(100):
                msg = {
                    "ev": "AM",
                    "sym": "ESZ5",
                    "o": 5500.0 + i,
                    "h": 5510.0,
                    "l": 5495.0,
                    "c": 5505.0,
                    "v": 1000,
                    "n": 500,
                    "s": 1736150400000 + i * 60000,
                    "e": 1736150460000 + i * 60000,
                }
                with patch.object(feed, "_push_bar_to_cache"):
                    feed._handle_messages([msg])

        def read_bars():
            for _ in range(100):
                try:
                    _ = feed.latest_bars
                except Exception as e:
                    errors.append(e)

        writer = threading.Thread(target=write_bars)
        readers = [threading.Thread(target=read_bars) for _ in range(5)]

        writer.start()
        for r in readers:
            r.start()
        writer.join(timeout=5)
        for r in readers:
            r.join(timeout=5)

        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Various edge cases and error recovery."""

    def test_microsecond_timestamps(self):
        """Timestamps in microseconds (between millis and nanos)."""
        df = pd.DataFrame(
            {
                "Open": [100],
                "Close": [102],
                "timestamp": [1736150400000000],  # microseconds
            }
        )
        result = _parse_timestamp_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_snapshot_with_partial_data(self, mock_rest_client):
        """Snapshot where some fields are None."""
        snap = SimpleNamespace(
            ticker="ESZ5",
            product_code="ES",
            last_trade=SimpleNamespace(price=5500.0, size=None, last_updated=None, timeframe=None),
            last_quote=None,
            session=SimpleNamespace(
                open=None,
                high=None,
                low=None,
                close=None,
                volume=None,
                change=None,
                change_percent=None,
                settlement_price=None,
                previous_settlement=None,
            ),
            last_minute=None,
            details=None,
        )
        mock_rest_client.get_futures_snapshot.return_value = [snap]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.get_snapshot(yahoo_ticker="ES=F")
        assert result is not None
        assert result["last_price"] == 5500.0
        assert result.get("session_open") is None

    def test_zero_volume_bars(self, mock_rest_client):
        """Bars with zero or None volume."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]

        agg = _make_futures_agg(volume=0, window_start=1736150400000000000)
        agg_none = SimpleNamespace(
            open=100,
            high=105,
            low=99,
            close=103,
            volume=None,
            window_start=1736150460000000000,
            ticker="ESZ5",
        )
        mock_rest_client.list_futures_aggregates.return_value = [agg, agg_none]
        provider = _make_provider_with_mock(mock_rest_client)

        df = provider.get_aggs("ES=F", interval="1m", period="1d")
        assert not df.empty
        # Volume should be 0, not NaN for both
        assert (df["Volume"] >= 0).all()

    def test_contract_with_no_last_trade_date(self, mock_rest_client):
        """Contract missing last_trade_date should still resolve."""
        contract = SimpleNamespace(
            ticker="ESZ5",
            product_code="ES",
            active=True,
            first_trade_date=None,
            last_trade_date=None,
            name="Test",
            settlement_date=None,
            days_to_maturity=None,
            trading_venue="CME",
            type="future",
            group_code=None,
            max_order_quantity=None,
            min_order_quantity=None,
            settlement_tick_size=None,
            spread_tick_size=None,
            trade_tick_size=None,
        )
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        result = provider.resolve_front_month("ES")
        assert result == "ESZ5"

    def test_multiple_message_types_in_single_batch_raw(self, mock_rest_client):
        """Handle a mix of bar + trade + quote raw dict messages in one batch."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        messages = [
            {
                "ev": "AM",
                "sym": "ESZ5",
                "o": 5500.0,
                "h": 5510.0,
                "l": 5495.0,
                "c": 5505.0,
                "v": 1000,
                "n": 500,
                "s": 1736150400000,
                "e": 1736150460000,
            },
            {
                "ev": "T",
                "sym": "ESZ5",
                "p": 5505.25,
                "s": 2,
                "t": 1736150401000,
                "q": 100,
            },
            {
                "ev": "Q",
                "sym": "ESZ5",
                "bp": 5504.75,
                "bs": 50,
                "ap": 5505.25,
                "as": 30,
                "t": 1736150401000,
            },
        ]

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages(messages)

        assert feed.msg_count == 1  # one batch
        assert feed.bar_count == 1
        assert feed.trade_count == 1
        assert "ESZ5" in feed._latest_quotes
        assert feed._latest_quotes["ESZ5"]["bid"] == 5504.75
        assert feed._latest_quotes["ESZ5"]["ask"] == 5505.25

    def test_multiple_message_types_in_single_batch_sdk(self, mock_rest_client):
        """Handle a mix of bar + trade + quote SDK objects in one batch."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        messages = [
            SimpleNamespace(
                event_type="AM",
                symbol="ESZ5",
                open=5500.0,
                high=5510.0,
                low=5495.0,
                close=5505.0,
                volume=1000,
                transactions=500,
                start_timestamp=1736150400000,
                end_timestamp=1736150460000,
                total_value=5500000.0,
            ),
            SimpleNamespace(
                event_type="T",
                symbol="ESZ5",
                price=5505.25,
                size=2,
                timestamp=1736150401000,
                sequence_number=100,
            ),
            SimpleNamespace(
                event_type="Q",
                symbol="ESZ5",
                bid_price=5504.75,
                bid_size=50,
                ask_price=5505.25,
                ask_size=30,
                bid_timestamp=None,
                ask_timestamp=None,
                sip_timestamp=None,
            ),
        ]

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages(messages)

        assert feed.msg_count == 1
        assert feed.bar_count == 1
        assert feed.trade_count == 1
        assert "ESZ5" in feed._latest_quotes

    def test_period_fallback_unknown(self, mock_rest_client):
        """Unknown period string should default to 5 days."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        mock_rest_client.list_futures_aggregates.return_value = [_make_futures_agg(window_start=1736150400000000000)]
        provider = _make_provider_with_mock(mock_rest_client)

        # "weird_period" can't be parsed, should default to 5 days
        df = provider.get_aggs("ES=F", interval="1m", period="weird_period")
        assert not df.empty

    def test_add_and_remove_tickers(self, mock_rest_client):
        """Test dynamic ticker add/remove on feed manager."""
        contract_es = _make_futures_contract(ticker="ESZ5", product_code="ES")
        contract_nq = _make_futures_contract(ticker="NQZ5", product_code="NQ")

        def mock_list(**kwargs):
            pc = kwargs.get("product_code", "")
            if pc == "ES":
                return [contract_es]
            elif pc == "NQ":
                return [contract_nq]
            return []

        mock_rest_client.list_futures_contracts.side_effect = mock_list
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()
        assert "ES=F" in feed._yahoo_to_massive

        # Add NQ=F
        feed.add_tickers(["NQ=F"])
        feed._resolve_tickers()
        assert "NQ=F" in feed._yahoo_to_massive

        # Remove ES=F
        feed.remove_tickers(["ES=F"])
        assert "ES=F" not in feed._yahoo_to_massive


# ═══════════════════════════════════════════════════════════════════════════
# TESTS — Raw WebSocket protocol specifics
# ═══════════════════════════════════════════════════════════════════════════


class TestRawWebSocketProtocol:
    """Verify raw JSON wire-protocol handling matches Massive docs."""

    def test_bar_dict_keys_match_wire_format(self, mock_rest_client):
        """Ensure the raw dict parser reads the correct short keys
        (o, h, l, c, v, s, e, n, vw) documented at
        https://massive.com/docs/websocket/futures/aggregates-per-minute"""
        contract = _make_futures_contract(ticker="MESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["MES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        wire_msg = {
            "ev": "AM",
            "sym": "MESZ5",
            "o": 5601.25,
            "h": 5605.00,
            "l": 5600.00,
            "c": 5604.75,
            "v": 423,
            "n": 87,
            "s": 1736150400000,
            "e": 1736150460000,
            "vw": 5602.30,
        }

        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([wire_msg])

        bar = feed.latest_bars["MESZ5"]
        assert bar["open"] == 5601.25
        assert bar["high"] == 5605.00
        assert bar["low"] == 5600.00
        assert bar["close"] == 5604.75
        assert bar["volume"] == 423
        assert bar["transactions"] == 87
        assert bar["start_timestamp"] == 1736150400000
        assert bar["end_timestamp"] == 1736150460000
        assert bar["total_value"] == 5602.30

    def test_trade_dict_keys_match_wire_format(self, mock_rest_client):
        """Ensure raw dict parser reads p (price), s (size), t (timestamp),
        q (sequence) as documented at
        https://massive.com/docs/websocket/futures/trades"""
        contract = _make_futures_contract(ticker="MESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["MES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        wire_msg = {
            "ev": "T",
            "sym": "MESZ5",
            "p": 5604.75,
            "s": 2,
            "t": 1736150401234,
            "q": 99887,
        }

        feed._handle_messages([wire_msg])

        trade = feed.latest_trades["MESZ5"]
        assert trade["price"] == 5604.75
        assert trade["size"] == 2
        assert trade["timestamp"] == 1736150401234
        assert trade["sequence"] == 99887

    def test_quote_dict_keys_match_wire_format(self, mock_rest_client):
        """Ensure raw dict parser reads bp, bs, ap, as, t as documented at
        https://massive.com/docs/websocket/futures/quotes"""
        contract = _make_futures_contract(ticker="MESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["MES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        wire_msg = {
            "ev": "Q",
            "sym": "MESZ5",
            "bp": 5604.50,
            "bs": 200,
            "ap": 5604.75,
            "as": 150,
            "t": 1736150401234,
        }

        feed._handle_messages([wire_msg])

        q = feed._latest_quotes["MESZ5"]
        assert q["bid"] == 5604.50
        assert q["bid_size"] == 200
        assert q["ask"] == 5604.75
        assert q["ask_size"] == 150
        assert q["timestamp"] == 1736150401234

    def test_auth_status_message_ignored(self, mock_rest_client):
        """Auth/status messages must not be counted as bars or trades."""
        feed = MassiveFeedManager(api_key="test_key")
        feed._handle_messages(
            [
                {"ev": "status", "status": "auth_success", "message": "authenticated"},
            ]
        )
        assert feed.bar_count == 0
        assert feed.trade_count == 0

    def test_subscribe_status_message_ignored(self, mock_rest_client):
        feed = MassiveFeedManager(api_key="test_key")
        feed._handle_messages(
            [
                {
                    "ev": "status",
                    "status": "success",
                    "message": "subscribed to: AM.MESZ5",
                },
            ]
        )
        assert feed.bar_count == 0
        assert feed.trade_count == 0

    def test_zero_volume_bar_wire(self, mock_rest_client):
        """No-trade windows may send bars with v=0; volume should be 0, not None."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "AM",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5500.0,
            "l": 5500.0,
            "c": 5500.0,
            "v": 0,
            "s": 1736150400000,
            "e": 1736150460000,
        }
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        bar = feed.latest_bars["ESZ5"]
        assert bar["volume"] == 0

    def test_missing_volume_key_defaults_zero(self, mock_rest_client):
        """If 'v' key is absent the parser should default to 0."""
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()

        msg = {
            "ev": "AM",
            "sym": "ESZ5",
            "o": 5500.0,
            "h": 5500.0,
            "l": 5500.0,
            "c": 5500.0,
            "s": 1736150400000,
            "e": 1736150460000,
        }
        with patch.object(feed, "_push_bar_to_cache"):
            feed._handle_messages([msg])

        bar = feed.latest_bars["ESZ5"]
        assert bar["volume"] == 0

    def test_upgrade_queues_second_agg_subscriptions(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
        )
        feed._resolve_tickers()
        feed._running = True

        feed.upgrade_to_second_aggs()

        assert feed._subscribe_second_aggs is True
        assert any("A.ESZ5" in s for s in feed._pending_subscribe)

    def test_downgrade_queues_unsubscribe(self, mock_rest_client):
        contract = _make_futures_contract(ticker="ESZ5")
        mock_rest_client.list_futures_contracts.return_value = [contract]
        provider = _make_provider_with_mock(mock_rest_client)

        feed = MassiveFeedManager(
            api_key="test_key",
            yahoo_tickers=["ES=F"],
            provider=provider,
            subscribe_second_aggs=True,
        )
        feed._resolve_tickers()
        feed._running = True

        feed.downgrade_to_minute_aggs()

        assert feed._subscribe_second_aggs is False
        assert any("A.ESZ5" in s for s in feed._pending_unsubscribe)
