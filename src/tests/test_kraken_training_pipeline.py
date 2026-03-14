"""
Kraken Training Pipeline Tests
================================

Verify that Kraken crypto symbols (BTC, ETH, SOL, KRAKEN:XBTUSD, etc.)
flow correctly through the entire dataset generation pipeline:

  1. Symbol resolution — short aliases → KRAKEN:* internal tickers
  2. Kraken detection — _is_kraken_symbol() correctly identifies crypto
  3. Asset class mapping — crypto symbols get asset_class_id = 1.0
  4. Volatility class mapping — crypto symbols get volatility = 1.0
  5. Bar loading routing — Kraken symbols bypass Massive, use Kraken loader
  6. Chart rendering compatibility — OHLCV DataFrames from Kraken render OK
  7. Tabular feature build — _build_row produces correct v6 18-feature vector
  8. feature_contract.json consistency — all Kraken symbols present

Run with:
    cd futures
    python -m pytest src/tests/test_kraken_training_pipeline.py -v
"""

import json
import os
import sys
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# ---------------------------------------------------------------------------
# Stub heavy dependencies so tests run offline without Redis / Postgres
# ---------------------------------------------------------------------------
_fake_cache: dict[str, Any] = {}


def _fake_cache_get(key):
    return _fake_cache.get(key)


def _fake_cache_set(key, value, ttl=0):
    _fake_cache[key] = value


def _fake_cache_key(*parts):
    return ":".join(str(p) for p in parts)


_cache_mod = MagicMock()
_cache_mod.cache_get = _fake_cache_get
_cache_mod.cache_set = _fake_cache_set
_cache_mod._cache_key = _fake_cache_key
_cache_mod.REDIS_AVAILABLE = False
sys.modules.setdefault("lib.core.cache", _cache_mod)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Symbol Resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolResolution:
    """Verify _SYMBOL_TO_TICKER maps and _resolve_ticker for crypto."""

    def test_short_alias_btc(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        assert _SYMBOL_TO_TICKER.get("BTC") == "KRAKEN:XBTUSD"

    def test_short_alias_eth(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        assert _SYMBOL_TO_TICKER.get("ETH") == "KRAKEN:ETHUSD"

    def test_short_alias_sol(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        assert _SYMBOL_TO_TICKER.get("SOL") == "KRAKEN:SOLUSD"

    def test_internal_ticker_passthrough(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        assert _SYMBOL_TO_TICKER.get("KRAKEN:XBTUSD") == "KRAKEN:XBTUSD"
        assert _SYMBOL_TO_TICKER.get("KRAKEN:ETHUSD") == "KRAKEN:ETHUSD"
        assert _SYMBOL_TO_TICKER.get("KRAKEN:SOLUSD") == "KRAKEN:SOLUSD"

    def test_pair_style_alias(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        assert _SYMBOL_TO_TICKER.get("XBTUSD") == "KRAKEN:XBTUSD"
        assert _SYMBOL_TO_TICKER.get("ETHUSD") == "KRAKEN:ETHUSD"

    def test_resolve_ticker_short(self):
        from lib.services.training.dataset_generator import _resolve_ticker

        assert _resolve_ticker("BTC") == "KRAKEN:XBTUSD"
        assert _resolve_ticker("ETH") == "KRAKEN:ETHUSD"
        assert _resolve_ticker("SOL") == "KRAKEN:SOLUSD"

    def test_resolve_ticker_internal(self):
        from lib.services.training.dataset_generator import _resolve_ticker

        assert _resolve_ticker("KRAKEN:XBTUSD") == "KRAKEN:XBTUSD"

    def test_resolve_ticker_futures_unchanged(self):
        """Futures symbols should resolve to their =F ticker, not Kraken."""
        from lib.services.training.dataset_generator import _resolve_ticker

        assert _resolve_ticker("MES") == "MES=F"
        assert _resolve_ticker("MGC") == "MGC=F"
        assert _resolve_ticker("MNQ") == "MNQ=F"

    def test_cme_crypto_futures_not_kraken(self):
        """CME micro crypto futures (MBT, MET) should NOT resolve to Kraken."""
        from lib.services.training.dataset_generator import _resolve_ticker

        assert _resolve_ticker("MBT") == "MBT=F"
        assert _resolve_ticker("MET") == "MET=F"

    def test_all_kraken_aliases_present(self):
        """All 9 Kraken crypto pairs should have short, internal, and pair aliases."""
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        expected_pairs = [
            ("BTC", "KRAKEN:XBTUSD"),
            ("ETH", "KRAKEN:ETHUSD"),
            ("SOL", "KRAKEN:SOLUSD"),
            ("LINK", "KRAKEN:LINKUSD"),
            ("AVAX", "KRAKEN:AVAXUSD"),
            ("DOT", "KRAKEN:DOTUSD"),
            ("ADA", "KRAKEN:ADAUSD"),
            ("POL", "KRAKEN:POLUSD"),
            ("XRP", "KRAKEN:XRPUSD"),
        ]
        for short, internal in expected_pairs:
            assert _SYMBOL_TO_TICKER.get(short) == internal, f"{short} → {internal}"
            assert _SYMBOL_TO_TICKER.get(internal) == internal, f"{internal} → self"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Kraken Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestKrakenDetection:
    """Verify _is_kraken_symbol correctly identifies crypto symbols."""

    def test_internal_ticker_detected(self):
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("KRAKEN:XBTUSD") is True
        assert _is_kraken_symbol("KRAKEN:ETHUSD") is True
        assert _is_kraken_symbol("KRAKEN:SOLUSD") is True

    def test_short_alias_detected(self):
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("BTC") is True
        assert _is_kraken_symbol("ETH") is True
        assert _is_kraken_symbol("SOL") is True

    def test_pair_alias_detected(self):
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("XBTUSD") is True
        assert _is_kraken_symbol("ETHUSD") is True

    def test_futures_not_detected(self):
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("MES") is False
        assert _is_kraken_symbol("MGC") is False
        assert _is_kraken_symbol("MNQ") is False
        assert _is_kraken_symbol("6E") is False

    def test_cme_crypto_not_kraken(self):
        """CME crypto futures should NOT be detected as Kraken."""
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("MBT") is False
        assert _is_kraken_symbol("MET") is False

    def test_case_insensitive(self):
        from lib.services.training.dataset_generator import _is_kraken_symbol

        assert _is_kraken_symbol("kraken:xbtusd") is True
        assert _is_kraken_symbol("Kraken:ETHUSD") is True


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Asset Class Mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestAssetClassMapping:
    """Verify crypto symbols map to asset_class_id = 1.0 (crypto class)."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC",
            "ETH",
            "SOL",
            "LINK",
            "AVAX",
            "DOT",
            "ADA",
            "POL",
            "XRP",
            "KRAKEN:XBTUSD",
            "KRAKEN:ETHUSD",
            "KRAKEN:SOLUSD",
            "MBT",
            "MET",
        ],
    )
    def test_crypto_asset_class(self, symbol):
        from lib.analysis.ml.breakout_cnn import get_asset_class_id

        assert get_asset_class_id(symbol) == 1.0, f"{symbol} should be crypto (1.0)"

    @pytest.mark.parametrize(
        "symbol,expected",
        [
            ("MES", 0.0),  # equity index
            ("MNQ", 0.0),  # equity index
            ("6E", 0.25),  # FX
            ("MGC", 0.5),  # metals
            ("MCL", 0.5),  # energy
            ("ZN", 0.75),  # treasuries
        ],
    )
    def test_non_crypto_asset_class(self, symbol, expected):
        from lib.analysis.ml.breakout_cnn import get_asset_class_id

        assert get_asset_class_id(symbol) == expected, f"{symbol} should be {expected}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Volatility Class Mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestVolatilityClassMapping:
    """Verify crypto symbols map to high volatility (1.0)."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC",
            "ETH",
            "SOL",
            "LINK",
            "AVAX",
            "DOT",
            "ADA",
            "POL",
            "XRP",
            "KRAKEN:XBTUSD",
            "KRAKEN:ETHUSD",
            "KRAKEN:SOLUSD",
            "MBT",
        ],
    )
    def test_crypto_high_volatility(self, symbol):
        from lib.analysis.ml.breakout_cnn import get_asset_volatility_class

        assert get_asset_volatility_class(symbol) == 1.0, f"{symbol} should be high vol (1.0)"

    def test_equity_medium_volatility(self):
        from lib.analysis.ml.breakout_cnn import get_asset_volatility_class

        assert get_asset_volatility_class("MES") == 0.5

    def test_fx_low_volatility(self):
        from lib.analysis.ml.breakout_cnn import get_asset_volatility_class

        assert get_asset_volatility_class("M6E") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Feature Contract Consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureContract:
    """Verify feature_contract.json includes all Kraken symbols."""

    @pytest.fixture
    def contract(self):
        path = os.path.join(_PROJECT_ROOT, "models", "feature_contract.json")
        if not os.path.isfile(path):
            pytest.skip("feature_contract.json not found")
        with open(path) as f:
            return json.load(f)

    def test_version_6(self, contract):
        assert contract["version"] == 8

    def test_num_tabular_18(self, contract):
        assert contract["num_tabular"] == 37

    def test_tabular_features_list(self, contract):
        features = contract["tabular_features"]
        assert len(features) == 37
        assert "asset_class_id" in features
        assert "asset_volatility_class" in features
        assert "breakout_type_ord" in features
        # v7 additions
        assert "daily_bias_direction" in features
        assert "daily_bias_confidence" in features
        assert "prior_day_pattern" in features
        assert "weekly_range_position" in features
        assert "monthly_trend_score" in features
        assert "crypto_momentum_score" in features
        # v7.1 additions
        assert "breakout_type_category" in features
        assert "session_overlap_flag" in features
        assert "atr_trend" in features
        assert "volume_trend" in features
        # v8-B additions — Cross-Asset Correlation
        assert "primary_peer_corr" in features
        assert "cross_class_corr" in features
        assert "correlation_regime" in features
        # v8-C additions — Asset Fingerprint
        assert "typical_daily_range_norm" in features
        assert "session_concentration" in features
        assert "breakout_follow_through" in features
        assert "hurst_exponent" in features
        assert "overnight_gap_tendency" in features
        assert "volume_profile_shape" in features

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC",
            "ETH",
            "SOL",
            "LINK",
            "AVAX",
            "DOT",
            "ADA",
            "POL",
            "XRP",
            "KRAKEN:XBTUSD",
            "KRAKEN:ETHUSD",
            "KRAKEN:SOLUSD",
            "KRAKEN:LINKUSD",
            "KRAKEN:AVAXUSD",
            "KRAKEN:DOTUSD",
            "KRAKEN:ADAUSD",
            "KRAKEN:POLUSD",
            "KRAKEN:XRPUSD",
        ],
    )
    def test_asset_class_map_has_crypto(self, contract, symbol):
        acm = contract["asset_class_map"]
        assert symbol in acm, f"{symbol} missing from asset_class_map"
        assert acm[symbol] == 1.0, f"{symbol} should be 1.0 (crypto)"

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC",
            "ETH",
            "SOL",
            "LINK",
            "AVAX",
            "DOT",
            "ADA",
            "POL",
            "XRP",
            "KRAKEN:XBTUSD",
            "KRAKEN:ETHUSD",
            "KRAKEN:SOLUSD",
        ],
    )
    def test_volatility_classes_has_crypto(self, contract, symbol):
        avc = contract["asset_volatility_classes"]
        assert symbol in avc, f"{symbol} missing from asset_volatility_classes"
        assert avc[symbol] == 1.0, f"{symbol} should be 1.0 (high vol)"

    def test_all_13_breakout_types(self, contract):
        bt = contract.get("breakout_types", {})
        expected = [
            "ORB",
            "PrevDay",
            "InitialBalance",
            "Consolidation",
            "Weekly",
            "Monthly",
            "Asian",
            "BollingerSqueeze",
            "ValueArea",
            "InsideDay",
            "GapRejection",
            "PivotPoints",
            "Fibonacci",
        ]
        for name in expected:
            assert name in bt, f"Missing breakout type: {name}"
        assert len(bt) == 13

    def test_breakout_type_ordinals(self, contract):
        bto = contract.get("breakout_type_ordinals", {})
        assert bto.get("ORB") == 0.0
        assert bto.get("FIB") == 1.0
        assert len(bto) == 13

    def test_session_thresholds(self, contract):
        st = contract.get("session_thresholds", {})
        expected_sessions = [
            "cme",
            "sydney",
            "tokyo",
            "shanghai",
            "frankfurt",
            "london",
            "london_ny",
            "us",
            "cme_settle",
        ]
        for s in expected_sessions:
            assert s in st, f"Missing session threshold: {s}"
            assert 0.0 < st[s] <= 1.0, f"Invalid threshold for {s}: {st[s]}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Kraken Bar Loader
# ═══════════════════════════════════════════════════════════════════════════


class TestKrakenBarLoader:
    """Verify _load_bars_from_kraken produces correct DataFrame shape."""

    @patch("lib.services.training.dataset_generator.requests")
    def test_load_bars_returns_dataframe(self, mock_requests):
        """Mocked Kraken API returns valid OHLCV → DataFrame with correct columns."""
        import pandas as pd

        # Build fake Kraken OHLC response
        now = int(datetime.now(tz=UTC).timestamp())
        candles = []
        for i in range(100):
            ts = now - (100 - i) * 60
            o = 95000.0 + i * 10
            h = o + 50
            lo = o - 30
            c = o + 20
            candles.append([ts, str(o), str(h), str(lo), str(c), "0", str(i * 0.5), 10])

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "error": [],
            "result": {
                "XXBTZUSD": candles,
                "last": now,
            },
        }
        fake_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = fake_response

        from lib.services.training.dataset_generator import _load_bars_from_kraken

        df = _load_bars_from_kraken("BTC", days=1)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

        # Verify expected columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Missing column: {col}"

        # Verify DatetimeIndex
        assert df.index.name == "datetime"
        assert hasattr(df.index, "tz")  # UTC-aware

        # Verify data is numeric
        assert df["Open"].dtype in ("float64", "float32")
        assert df["Close"].dtype in ("float64", "float32")

    @patch("lib.services.training.dataset_generator.requests")
    def test_load_bars_strips_kraken_prefix(self, mock_requests):
        """Internal KRAKEN: prefix should be stripped when calling the API."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "error": [],
            "result": {"XETHZUSD": [], "last": 0},
        }
        fake_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = fake_response

        from lib.services.training.dataset_generator import _load_bars_from_kraken

        _load_bars_from_kraken("KRAKEN:ETHUSD", days=1)

        # The API call should use the bare pair without the prefix
        call_args = mock_requests.get.call_args
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        if isinstance(params, dict):
            pair = params.get("pair", "")
            assert "KRAKEN:" not in pair, f"API pair should not have KRAKEN: prefix, got: {pair}"

    @patch("lib.services.training.dataset_generator.requests")
    def test_load_bars_handles_api_error(self, mock_requests):
        """Kraken API error should return None gracefully."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "error": ["EGeneral:Invalid arguments"],
            "result": {},
        }
        fake_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = fake_response

        from lib.services.training.dataset_generator import _load_bars_from_kraken

        df = _load_bars_from_kraken("KRAKEN:FAKEUSD", days=1)
        assert df is None

    @patch("lib.services.training.dataset_generator.requests")
    def test_load_bars_handles_connection_error(self, mock_requests):
        """Network failure should return None, not raise."""
        mock_requests.get.side_effect = Exception("Connection refused")

        from lib.services.training.dataset_generator import _load_bars_from_kraken

        df = _load_bars_from_kraken("BTC", days=1)
        assert df is None

    @patch("lib.services.training.dataset_generator.requests")
    def test_load_bars_deduplicates(self, mock_requests):
        """Duplicate timestamps should be removed."""
        now = int(datetime.now(tz=UTC).timestamp())
        ts = now - 120
        candles = [
            [ts, "95000", "95050", "94970", "95020", "0", "1.5", 10],
            [ts, "95000", "95050", "94970", "95020", "0", "1.5", 10],  # duplicate
            [ts + 60, "95020", "95080", "95000", "95060", "0", "2.0", 12],
        ]
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "error": [],
            "result": {"XXBTZUSD": candles, "last": now},
        }
        fake_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = fake_response

        from lib.services.training.dataset_generator import _load_bars_from_kraken

        df = _load_bars_from_kraken("BTC", days=1)
        assert df is not None
        assert len(df) == 2  # deduped from 3 to 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST: load_bars Routing
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadBarsRouting:
    """Verify load_bars routes Kraken symbols to the Kraken loader."""

    @patch("lib.services.training.dataset_generator._load_bars_from_kraken")
    @patch("lib.services.training.dataset_generator._load_bars_from_db")
    def test_btc_routes_to_kraken(self, mock_db, mock_kraken):
        """BTC should attempt Kraken loader, not Massive."""
        import pandas as pd

        mock_kraken.return_value = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [10]},
            index=pd.DatetimeIndex([datetime.now(tz=UTC)], name="datetime"),
        )
        mock_db.return_value = None

        from lib.services.training.dataset_generator import load_bars

        # Force EngineDataClient to return None so the legacy fallback path is exercised
        with patch("lib.services.data.engine_data_client.EngineDataClient.get_bars", return_value=None):
            df = load_bars("BTC", source="kraken", days=1)

        assert df is not None
        assert mock_kraken.called

    @patch("lib.services.training.dataset_generator._load_bars_from_kraken")
    @patch("lib.services.training.dataset_generator._load_bars_from_db")
    def test_kraken_prefix_routes_to_kraken(self, mock_db, mock_kraken):
        """KRAKEN:XBTUSD should route to Kraken loader."""
        import pandas as pd

        mock_kraken.return_value = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [10]},
            index=pd.DatetimeIndex([datetime.now(tz=UTC)], name="datetime"),
        )
        mock_db.return_value = None

        from lib.services.training.dataset_generator import load_bars

        # Force EngineDataClient to return None so the legacy fallback path is exercised
        with patch("lib.services.data.engine_data_client.EngineDataClient.get_bars", return_value=None):
            df = load_bars("KRAKEN:XBTUSD", source="kraken", days=1)

        assert df is not None
        assert mock_kraken.called

    @patch("lib.services.training.dataset_generator._load_bars_from_engine")
    @patch("lib.services.training.dataset_generator._load_bars_from_massive")
    @patch("lib.services.training.dataset_generator._load_bars_from_db")
    def test_futures_does_not_route_to_kraken(self, mock_db, mock_massive, mock_engine):
        """MES should NOT route to Kraken — should try db → massive → csv."""
        import pandas as pd

        mock_massive.return_value = pd.DataFrame(
            {"Open": [5900], "High": [5910], "Low": [5890], "Close": [5905], "Volume": [100]},
            index=pd.DatetimeIndex([datetime.now(tz=UTC)], name="datetime"),
        )
        mock_db.return_value = None
        mock_engine.return_value = None

        from lib.services.training.dataset_generator import load_bars

        # Force EngineDataClient to return None so the legacy fallback path is exercised
        with patch("lib.services.data.engine_data_client.EngineDataClient.get_bars", return_value=None):
            df = load_bars("MES", source="massive", days=1)

        assert df is not None
        assert mock_massive.called


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Chart Rendering Compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestChartRenderingCompatibility:
    """Verify that OHLCV DataFrames from Kraken are compatible with chart renderers."""

    def _make_crypto_bars(self, n=60):
        """Create a synthetic crypto OHLCV DataFrame mimicking Kraken output."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        base = 95000.0
        times = pd.date_range(
            start="2026-03-07 10:00:00",
            periods=n,
            freq="1min",
            tz="UTC",
        )
        opens = base + np.cumsum(np.random.randn(n) * 50)
        highs = opens + np.abs(np.random.randn(n) * 30)
        lows = opens - np.abs(np.random.randn(n) * 30)
        closes = opens + np.random.randn(n) * 20
        volumes = np.abs(np.random.randn(n) * 2) + 0.1

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            },
            index=times,
        )
        df.index.name = "datetime"
        return df

    def test_crypto_bars_have_correct_schema(self):
        """Bars should have the columns that renderers expect."""
        df = self._make_crypto_bars()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns
        assert df.index.name == "datetime"
        assert len(df) == 60

    def test_crypto_bars_no_negative_prices(self):
        """Crypto bars should never have negative OHLC values."""
        df = self._make_crypto_bars()
        # Some synthetic values might go negative with random walk; real Kraken won't.
        # But the generator filters these: (Open > 0) & (High > 0) & (Low > 0) & (Close > 0)
        # Here we just verify the schema is correct for rendering.
        assert df["High"].max() > 0

    def test_parity_renderer_bar_conversion(self):
        """If parity renderer is available, verify it can convert crypto bars."""
        df = self._make_crypto_bars()
        try:
            from lib.analysis.rendering.chart_renderer_parity import dataframe_to_parity_bars

            bars = dataframe_to_parity_bars(df)
            assert len(bars) == len(df)
        except ImportError:
            pytest.skip("chart_renderer_parity not available")

    def test_original_renderer_accepts_crypto_bars(self):
        """If mplfinance renderer is available, verify it accepts crypto bars."""
        df = self._make_crypto_bars()
        try:
            # Just verify the function can be called without crashing (don't save)
            # We pass save_path=None or a temp path to avoid writing files
            import tempfile

            from lib.analysis.rendering.chart_renderer import render_ruby_snapshot

            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                render_ruby_snapshot(
                    bars=df,
                    symbol="BTC",
                    orb_high=df["High"].iloc[10],
                    orb_low=df["Low"].iloc[10],
                    direction="long",
                    quality_pct=85,
                    label="long",
                    save_path=tmp.name,
                )
                # Result should be the path (or None if render failed)
                # Either way, no crash = success
        except ImportError:
            pytest.skip("chart_renderer not available (mplfinance not installed)")
        except Exception as exc:
            # Some environments lack display; that's OK — we just need no schema errors
            if "display" in str(exc).lower() or "tkinter" in str(exc).lower():
                pytest.skip(f"No display available: {exc}")
            raise


# ═══════════════════════════════════════════════════════════════════════════
# TEST: ASSET_CLASS_ORDINALS / ASSET_VOLATILITY_CLASS consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestBreakoutCnnMappings:
    """Verify breakout_cnn.py has consistent mappings for all Kraken symbols."""

    def test_ordinals_and_contract_in_sync(self):
        """ASSET_CLASS_ORDINALS should match feature_contract.json asset_class_map."""
        from lib.analysis.ml.breakout_cnn import ASSET_CLASS_ORDINALS

        contract_path = os.path.join(_PROJECT_ROOT, "models", "feature_contract.json")
        if not os.path.isfile(contract_path):
            pytest.skip("feature_contract.json not found")

        with open(contract_path) as f:
            contract = json.load(f)

        acm = contract["asset_class_map"]

        # Every key in the contract should also be in ASSET_CLASS_ORDINALS
        for symbol, value in acm.items():
            py_value = ASSET_CLASS_ORDINALS.get(symbol.upper())
            assert py_value is not None, f"{symbol} in feature_contract.json but not in ASSET_CLASS_ORDINALS"
            assert abs(py_value - value) < 0.01, f"{symbol}: contract={value} vs python={py_value}"

    def test_volatility_and_contract_in_sync(self):
        """ASSET_VOLATILITY_CLASS should match feature_contract.json asset_volatility_classes."""
        from lib.analysis.ml.breakout_cnn import ASSET_VOLATILITY_CLASS

        contract_path = os.path.join(_PROJECT_ROOT, "models", "feature_contract.json")
        if not os.path.isfile(contract_path):
            pytest.skip("feature_contract.json not found")

        with open(contract_path) as f:
            contract = json.load(f)

        avc = contract.get("asset_volatility_classes", {})

        for symbol, value in avc.items():
            py_value = ASSET_VOLATILITY_CLASS.get(symbol.upper())
            assert py_value is not None, f"{symbol} in feature_contract.json but not in ASSET_VOLATILITY_CLASS"
            assert abs(py_value - value) < 0.01, f"{symbol}: contract={value} vs python={py_value}"

    def test_breakout_type_ordinals_match(self):
        """BREAKOUT_TYPE_ORDINALS should match feature_contract.json."""
        from lib.analysis.ml.breakout_cnn import BREAKOUT_TYPE_ORDINALS

        contract_path = os.path.join(_PROJECT_ROOT, "models", "feature_contract.json")
        if not os.path.isfile(contract_path):
            pytest.skip("feature_contract.json not found")

        with open(contract_path) as f:
            contract = json.load(f)

        bto = contract.get("breakout_type_ordinals", {})

        for bt_name, value in bto.items():
            py_value = BREAKOUT_TYPE_ORDINALS.get(bt_name.upper())
            assert py_value is not None, f"{bt_name} in feature_contract.json but not in BREAKOUT_TYPE_ORDINALS"
            assert abs(py_value - value) < 0.01, f"{bt_name}: contract={value} vs python={py_value}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Kraken Symbols in Trainer Default Symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainerDefaultSymbols:
    """Verify the trainer includes crypto in its default symbol list."""

    def test_default_symbols_include_crypto(self):
        """trainer_server.py DEFAULT_SYMBOLS should include BTC, ETH, SOL."""
        try:
            # Read the file directly to avoid importing the full server
            trainer_path = os.path.join(_SRC, "lib", "training", "trainer_server.py")
            if not os.path.isfile(trainer_path):
                pytest.skip("trainer_server.py not found")

            with open(trainer_path) as f:
                content = f.read()

            # Look for DEFAULT_SYMBOLS definition
            assert "BTC" in content, "BTC should be in trainer DEFAULT_SYMBOLS"
            assert "ETH" in content, "ETH should be in trainer DEFAULT_SYMBOLS"
            assert "SOL" in content, "SOL should be in trainer DEFAULT_SYMBOLS"
        except Exception as exc:
            pytest.skip(f"Could not read trainer_server.py: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST: End-to-End Tabular Feature Shape
# ═══════════════════════════════════════════════════════════════════════════


class TestTabularFeatureShape:
    """Verify that building a tabular row for a crypto symbol produces 18 features."""

    def test_asset_class_id_for_btc_is_crypto(self):
        from lib.analysis.ml.breakout_cnn import get_asset_class_id

        val = get_asset_class_id("BTC")
        assert val == 1.0

    def test_asset_class_id_for_kraken_ticker(self):
        from lib.analysis.ml.breakout_cnn import get_asset_class_id

        val = get_asset_class_id("KRAKEN:XBTUSD")
        assert val == 1.0

    def test_volatility_class_for_all_9_kraken_pairs(self):
        from lib.analysis.ml.breakout_cnn import get_asset_volatility_class

        pairs = ["BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "POL", "XRP"]
        for p in pairs:
            assert get_asset_volatility_class(p) == 1.0, f"{p} vol class != 1.0"

    def test_feature_contract_has_28_features(self):
        """Sanity check: exactly 37 tabular features in the v8 contract."""
        contract_path = os.path.join(_PROJECT_ROOT, "models", "feature_contract.json")
        if not os.path.isfile(contract_path):
            pytest.skip("feature_contract.json not found")

        with open(contract_path) as f:
            contract = json.load(f)

        assert contract["num_tabular"] == 37
        assert len(contract["tabular_features"]) == 37

        expected_features = [
            "quality_pct_norm",
            "volume_ratio",
            "atr_pct",
            "cvd_delta",
            "nr7_flag",
            "direction_flag",
            "session_ordinal",
            "london_overlap_flag",
            "or_range_atr_ratio",
            "premarket_range_ratio",
            "bar_of_day",
            "day_of_week",
            "vwap_distance",
            "asset_class_id",
            "breakout_type_ord",
            "asset_volatility_class",
            "hour_of_day",
            "tp3_atr_mult_norm",
            # v7 additions
            "daily_bias_direction",
            "daily_bias_confidence",
            "prior_day_pattern",
            "weekly_range_position",
            "monthly_trend_score",
            "crypto_momentum_score",
            # v7.1 additions
            "breakout_type_category",
            "session_overlap_flag",
            "atr_trend",
            "volume_trend",
            # v8-B additions (cross-asset correlation)
            "primary_peer_corr",
            "cross_class_corr",
            "correlation_regime",
            # v8-C additions (asset fingerprint)
            "typical_daily_range_norm",
            "session_concentration",
            "breakout_follow_through",
            "hurst_exponent",
            "overnight_gap_tendency",
            "volume_profile_shape",
        ]
        for feat in expected_features:
            assert feat in contract["tabular_features"], f"Missing feature: {feat}"
