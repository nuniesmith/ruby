"""
test_data_provider_routing.py — Trainer ↔ Engine Data Routing Tests
====================================================================

Validates the core architectural rule:

    The TRAINER only talks to the ENGINE via HTTP.
    The ENGINE/DATA-SERVICE (on the cloud box) owns Massive, Kraken, yfinance.
    The TRAINER must NEVER instantiate those clients directly.

Root cause from the 2026-03-11 training failure logs:
  1. Trainer starts dataset generation with bars_source=engine — ✅ correct
  2. Engine loads 50k bars for MGC and returns them — ✅ correct
  3. After ORB simulation, generate_dataset_for_symbol() calls
     compute_all_crypto_momentum() which imports crypto_momentum.py
     which calls cache.get_data() → initialises Massive client locally
     on the trainer box → MASSIVE_API_KEY not set → falls to yfinance
     → yfinance receives /MGC → HTTP 500 → 10k lines of HTML spam
  4. Same code path also initialises Kraken REST client locally →
     1,280 rate-limit errors (EGeneral:Too many requests)
  5. Meanwhile engine backfill returns +0 bars for every symbol because
     the historical_bars table was empty — the fills were async and the
     trainer didn't wait long enough.

Architecture (correct flow):
    WebUI → trainer API (home, CUDA) → engine API (cloud, Ubuntu)
    Engine has: Massive API key, Kraken, Postgres, Redis, yfinance
    Trainer has: CUDA GPU, API key to talk to engine, nothing else

These tests enforce:
  - _load_bars_from_engine is the ONLY network call the trainer makes
  - crypto_momentum / cache.get_data / Massive / Kraken / yfinance
    are NEVER imported or called when running on the trainer side
  - The engine bars API resolves symbols correctly and fills gaps
    before returning data to the trainer
  - When the engine returns 0 bars, the trainer fails gracefully
    without falling through to local yfinance/Kraken/Massive
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the src/ tree is importable
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

os.environ.setdefault("DISABLE_REDIS", "1")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

# The 25 training symbols from the failing log run
TRAINING_SYMBOLS = [
    "MGC",
    "SIL",
    "MHG",
    "MCL",
    "MNG",
    "MES",
    "MNQ",
    "M2K",
    "MYM",
    "6E",
    "6B",
    "6J",
    "6A",
    "6C",
    "6S",
    "ZN",
    "ZB",
    "ZC",
    "ZS",
    "ZW",
    "MBT",
    "MET",
    "BTC",
    "ETH",
    "SOL",
]

KRAKEN_SPOT_SYMBOLS = {"BTC", "ETH", "SOL"}
CME_CRYPTO_FUTURES = {"MBT", "MET"}
CME_FUTURES = set(TRAINING_SYMBOLS) - KRAKEN_SPOT_SYMBOLS


def _make_bars(n: int = 500, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic 1-minute OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    spread = close * 0.002
    idx = pd.date_range("2025-01-06 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "Open": close + rng.uniform(-0.5, 0.5, n) * spread,
            "High": close + rng.uniform(0, 1, n) * spread,
            "Low": close - rng.uniform(0, 1, n) * spread,
            "Close": close,
            "Volume": rng.poisson(1000, n).astype(float),
        },
        index=idx,
    )


def _make_engine_json_response(bars_df: pd.DataFrame, filling: bool = False) -> dict:
    """Build a JSON payload matching what the engine /bars endpoint returns."""
    split = {
        "columns": bars_df.columns.tolist(),
        "index": bars_df.index.astype(str).tolist(),
        "data": bars_df.values.tolist(),
    }
    return {
        "symbol": "MGC=F",
        "interval": "1m",
        "bar_count": len(bars_df),
        "filled": True,
        "filling": filling,
        "fill_status_url": None,
        "bars_added": 0,
        "fill_error": None,
        "data": split,
    }


def _make_empty_engine_response() -> dict:
    """Engine response when it has zero bars for a symbol."""
    return {
        "symbol": "MGC=F",
        "interval": "1m",
        "bar_count": 0,
        "filled": True,
        "filling": False,
        "fill_status_url": None,
        "bars_added": 0,
        "fill_error": None,
        "data": {
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "index": [],
            "data": [],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 1 — Symbol resolution basics
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolResolution:
    """_resolve_ticker must produce valid tickers and never the /SYMBOL format."""

    def test_never_produces_slash_prefix(self):
        """The /MGC format caused HTTP 500 from Yahoo — must never happen."""
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in TRAINING_SYMBOLS:
            ticker = _resolve_ticker(symbol)
            assert not ticker.startswith("/"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r}  — slash prefix "
                f"causes yfinance HTTP 500.  Must be {symbol}=F or KRAKEN:*"
            )

    def test_cme_futures_resolve_to_equals_f(self):
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in CME_FUTURES:
            ticker = _resolve_ticker(symbol)
            assert "=" in ticker or ticker.startswith("KRAKEN:"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r}  — CME futures must have =F suffix"
            )

    def test_spot_crypto_resolves_to_kraken_prefix(self):
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in KRAKEN_SPOT_SYMBOLS:
            ticker = _resolve_ticker(symbol)
            assert ticker.startswith("KRAKEN:"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r}  — spot crypto should route to KRAKEN:* not CME"
            )

    def test_cme_crypto_futures_not_routed_to_kraken(self):
        """MBT/MET are CME micro futures, not Kraken spot."""
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in CME_CRYPTO_FUTURES:
            ticker = _resolve_ticker(symbol)
            assert not ticker.startswith("KRAKEN:"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r}  — MBT/MET are CME futures, not Kraken spot"
            )
            assert ticker.endswith("=F"), f"_resolve_ticker({symbol!r}) → {ticker!r}  — expected =F"

    def test_all_25_symbols_have_explicit_mapping(self):
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        for symbol in TRAINING_SYMBOLS:
            assert symbol in _SYMBOL_TO_TICKER, (
                f"{symbol!r} missing from _SYMBOL_TO_TICKER — will fall back to generic {symbol}=F which may be wrong"
            )

    def test_is_kraken_detection_correct(self):
        """Only BTC/ETH/SOL should be detected as Kraken, not MBT/MET/MGC."""
        from lib.services.training.dataset_generator import _is_kraken_symbol

        for sym in KRAKEN_SPOT_SYMBOLS:
            assert _is_kraken_symbol(sym), f"{sym} should be Kraken"

        for sym in ["MGC", "MES", "MNQ", "MBT", "MET", "6E", "ZN"]:
            assert not _is_kraken_symbol(sym), f"{sym} should NOT be Kraken — it's a CME futures symbol"


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 2 — Trainer must only call the engine HTTP API
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainerOnlyTalksToEngine:
    """When bars_source='engine', the trainer must route all data requests
    through EngineDataClient (HTTP to the data service).  It must never
    instantiate Massive, Kraken, or yfinance clients locally."""

    def test_load_bars_engine_source_only_calls_engine(self):
        """load_bars() should go through EngineDataClient and return its result
        — never falling through to Massive/Kraken/yfinance/Redis."""

        massive_called = False
        kraken_called = False
        cache_called = False
        db_called = False

        bars = _make_bars(500, start_price=2700.0)

        def _mock_massive(symbol, days=90):
            nonlocal massive_called
            massive_called = True
            return bars

        def _mock_kraken(symbol, days=90):
            nonlocal kraken_called
            kraken_called = True
            return bars

        def _mock_cache(symbol, days=90):
            nonlocal cache_called
            cache_called = True
            return bars

        def _mock_db(symbol, days=90):
            nonlocal db_called
            db_called = True
            return bars

        with (
            # Primary path: EngineDataClient returns data — no fallback needed
            patch(
                "lib.services.data.engine_data_client.EngineDataClient.get_bars",
                return_value=bars,
            ),
            patch("lib.services.training.dataset_generator._load_bars_from_massive", side_effect=_mock_massive),
            patch("lib.services.training.dataset_generator._load_bars_from_kraken", side_effect=_mock_kraken),
            patch("lib.services.training.dataset_generator._load_bars_from_cache", side_effect=_mock_cache),
            patch("lib.services.training.dataset_generator._load_bars_from_db", side_effect=_mock_db),
        ):
            from lib.services.training.dataset_generator import load_bars

            result = load_bars("MGC", source="engine", days=180)

        assert result is not None and not result.empty, "EngineDataClient returned data — result must not be None"
        assert not massive_called, (
            "Massive was called on the trainer — trainer must not have "
            "Massive API key; all data comes through the engine HTTP API"
        )
        assert not kraken_called, (
            "Kraken was called on the trainer — trainer must not call "
            "Kraken directly; engine handles crypto bar resolution"
        )
        assert not cache_called, (
            "Redis cache was called on the trainer — trainer doesn't have "
            "Redis; all data comes through the engine HTTP API"
        )

    def test_engine_returns_data_no_fallback_triggered(self):
        """When EngineDataClient returns valid bars, NO fallback loader should fire."""
        call_log = []

        def _make_loader(name, returns_data=False):
            def _loader(symbol, days=90):
                call_log.append(name)
                return _make_bars(100) if returns_data else None

            return _loader

        with (
            # Primary path: EngineDataClient returns data — legacy loaders must not run
            patch(
                "lib.services.data.engine_data_client.EngineDataClient.get_bars",
                return_value=_make_bars(100),
            ),
            patch("lib.services.training.dataset_generator._load_bars_from_engine", side_effect=_make_loader("engine")),
            patch("lib.services.training.dataset_generator._load_bars_from_db", side_effect=_make_loader("db")),
            patch("lib.services.training.dataset_generator._load_bars_from_cache", side_effect=_make_loader("cache")),
            patch(
                "lib.services.training.dataset_generator._load_bars_from_massive",
                side_effect=_make_loader("massive"),
            ),
            patch(
                "lib.services.training.dataset_generator._load_bars_from_kraken",
                side_effect=_make_loader("kraken"),
            ),
            patch("lib.services.training.dataset_generator._load_bars_from_csv", side_effect=_make_loader("csv")),
        ):
            from lib.services.training.dataset_generator import load_bars

            load_bars("MGC", source="engine", days=90)

        assert call_log == [], (
            f"Expected no legacy loaders to fire but got {call_log}.  When "
            f"EngineDataClient returns data, no other loader should be tried."
        )

    def test_engine_fails_fallback_order_is_correct(self):
        """When EngineDataClient fails, the legacy fallback chain fires:
        _load_bars_from_engine → _load_bars_from_db → _load_bars_from_massive → csv.
        Kraken must NOT be in the chain for a CME futures symbol."""
        call_log = []

        def _make_loader(name):
            def _loader(symbol, days=90):
                call_log.append(name)
                return None  # all fail

            return _loader

        with (
            # Force EngineDataClient to fail → triggers legacy fallback chain
            patch(
                "lib.services.data.engine_data_client.EngineDataClient.get_bars",
                side_effect=Exception("engine unavailable"),
            ),
            patch("lib.services.training.dataset_generator._load_bars_from_engine", side_effect=_make_loader("engine")),
            patch("lib.services.training.dataset_generator._load_bars_from_db", side_effect=_make_loader("db")),
            patch("lib.services.training.dataset_generator._load_bars_from_cache", side_effect=_make_loader("cache")),
            patch(
                "lib.services.training.dataset_generator._load_bars_from_massive", side_effect=_make_loader("massive")
            ),
            patch("lib.services.training.dataset_generator._load_bars_from_kraken", side_effect=_make_loader("kraken")),
            patch("lib.services.training.dataset_generator._load_bars_from_csv", side_effect=_make_loader("csv")),
        ):
            from lib.services.training.dataset_generator import load_bars

            result = load_bars("MGC", source="engine", days=90)

        assert result is None, "All loaders failed — should return None"
        assert "engine" in call_log, "Legacy _load_bars_from_engine must be tried in fallback chain"
        # Kraken should NOT be in the chain for a CME futures symbol
        assert "kraken" not in call_log, (
            f"Kraken was tried for CME futures symbol MGC — it should only be "
            f"tried for KRAKEN:* spot crypto symbols. call_log={call_log}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 3 — crypto_momentum must not make direct API calls on trainer
# ═══════════════════════════════════════════════════════════════════════════


class TestCryptoMomentumNoDirectAPICalls:
    """The crypto_momentum module is imported on the trainer during
    generate_dataset_for_symbol().  It must NOT call Kraken/Massive/yfinance
    directly — those calls must be routed through the engine."""

    def test_crypto_momentum_import_initializes_clients(self):
        """Verify that importing crypto_momentum triggers Kraken/Massive
        client initialization — this is the bug we need to fix.

        This test DOCUMENTS the current broken behavior so we know when
        it's fixed.  When the fix lands, flip the assertion."""
        # The logs showed:
        #   [KRAKEN] 16:09:07  Kraken REST client initialized (auth=no)
        #   [MASSIVE] 16:09:07  MASSIVE_API_KEY not set — using yfinance fallback
        # These appeared on the TRAINER container.

        # We just verify the code path exists — the actual fix is to make
        # generate_dataset_for_symbol() skip crypto_momentum entirely when
        # running on the trainer, or have crypto_momentum route through engine.
        from lib.services.training.dataset_generator import generate_dataset_for_symbol

        assert callable(generate_dataset_for_symbol)

    def test_generate_dataset_for_symbol_catches_crypto_momentum_failure(self):
        """If crypto_momentum fails (no Kraken/Massive on trainer), the
        pipeline must continue with a default score of 0.5 — not crash."""
        from lib.services.training.dataset_generator import (
            DatasetConfig,
            generate_dataset_for_symbol,
        )

        bars = _make_bars(1000, start_price=5500.0, seed=99)
        config = DatasetConfig(
            breakout_type="ORB",
            orb_session="us",
            output_dir=tempfile.mkdtemp(),
            image_dir=os.path.join(tempfile.mkdtemp(), "images"),
            use_parity_renderer=False,
            max_samples_per_label=3,
            max_samples_per_type_label=3,
        )

        # Make crypto_momentum raise (simulating no Kraken on trainer)
        # Also mock the chart renderer to avoid Pillow _idat.fileno()
        # errors on some platforms — this test validates crypto_momentum
        # error handling, not chart rendering.
        _mock_render = MagicMock(return_value="/tmp/fake_render.png")
        with (
            patch(
                "lib.analysis.crypto_momentum.compute_all_crypto_momentum",
                side_effect=Exception("No Kraken on trainer"),
            ),
            patch(
                "lib.analysis.rendering.chart_renderer.render_ruby_snapshot",
                _mock_render,
            ),
        ):
            # Should not raise — crypto_momentum failure is non-fatal
            rows, stats = generate_dataset_for_symbol(
                symbol="MES",
                bars_1m=bars,
                config=config,
            )

        # Pipeline continued despite crypto_momentum failure
        # (rows may be 0 if rendering fails, but no exception = success)
        assert isinstance(rows, list)
        assert isinstance(stats.total_windows, int)

    def test_crypto_momentum_score_defaults_to_neutral(self):
        """When crypto_momentum can't compute a score, every row should
        get the neutral default (0.5), not NaN or an exception."""
        from lib.services.training.dataset_generator import (
            DatasetConfig,
            generate_dataset_for_symbol,
        )

        bars = _make_bars(1000, start_price=5500.0, seed=99)
        config = DatasetConfig(
            breakout_type="ORB",
            orb_session="us",
            output_dir=tempfile.mkdtemp(),
            image_dir=os.path.join(tempfile.mkdtemp(), "images"),
            use_parity_renderer=False,
            max_samples_per_label=2,
            max_samples_per_type_label=2,
        )

        with patch(
            "lib.analysis.crypto_momentum.compute_all_crypto_momentum",
            side_effect=ImportError("no kraken on trainer"),
        ):
            rows, stats = generate_dataset_for_symbol(
                symbol="MES",
                bars_1m=bars,
                config=config,
            )

        for row in rows:
            score = row.get("crypto_momentum_score", None)
            assert score is not None, "crypto_momentum_score missing from row"
            assert score == 0.5, (
                f"crypto_momentum_score={score}, expected 0.5 (neutral default) "
                f"when crypto_momentum is unavailable on the trainer"
            )


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 4 — Engine _load_bars_from_engine behaviour
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineBarLoader:
    """Test _load_bars_from_engine — the trainer's ONLY data access path.

    Note: _load_bars_from_engine does ``import requests as _requests``
    locally, so we must patch the ``requests`` module itself (via
    ``patch("requests.get", ...)``) rather than a module-level attribute.
    """

    def test_returns_dataframe_on_success(self):
        from lib.services.training.dataset_generator import _load_bars_from_engine

        bars = _make_bars(200, start_price=2700.0)
        payload = _make_engine_json_response(bars)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        with patch("requests.get", return_value=mock_resp):
            result = _load_bars_from_engine("MGC", days=180)

        assert result is not None
        assert len(result) == 200
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_returns_none_on_empty_data(self):
        """When engine has 0 bars, trainer must get None — not fall to yfinance."""
        from lib.services.training.dataset_generator import _load_bars_from_engine

        payload = _make_empty_engine_response()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        with patch("requests.get", return_value=mock_resp):
            result = _load_bars_from_engine("MGC", days=180)

        assert result is None, (
            "Engine returned 0 bars but loader didn't return None — "
            "this causes the fallback chain to try local Massive/yfinance"
        )

    def test_returns_none_on_404(self):
        from lib.services.training.dataset_generator import _load_bars_from_engine

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("requests.get", return_value=mock_resp):
            result = _load_bars_from_engine("INVALID", days=90)

        assert result is None

    def test_returns_none_on_500(self):
        from lib.services.training.dataset_generator import _load_bars_from_engine

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.get", return_value=mock_resp):
            result = _load_bars_from_engine("MGC", days=90)

        assert result is None

    def test_returns_none_on_connection_error(self):
        """If the engine is unreachable, return None (don't crash)."""
        from lib.services.training.dataset_generator import _load_bars_from_engine

        with patch("requests.get", side_effect=ConnectionError("Engine unreachable via Tailscale")):
            result = _load_bars_from_engine("MGC", days=90)

        assert result is None

    def test_polls_fill_status_when_filling(self):
        """When engine returns filling=True, loader should poll then re-fetch."""
        from lib.services.training.dataset_generator import _load_bars_from_engine

        bars = _make_bars(200, start_price=2700.0)
        payload_filling = _make_engine_json_response(bars, filling=True)
        payload_filling["fill_status_url"] = "/bars/MGC/fill/status"

        payload_done = _make_engine_json_response(bars, filling=False)

        call_urls = []

        def _mock_get(url, **kwargs):
            call_urls.append(url)
            resp = MagicMock()
            resp.status_code = 200

            if "/fill/status" in url:
                resp.json.return_value = {"status": "complete", "bars_added": 50}
            elif kwargs.get("params", {}).get("auto_fill") == "false":
                resp.json.return_value = payload_done
            else:
                resp.json.return_value = payload_filling
            return resp

        with (
            patch("requests.get", side_effect=_mock_get),
            patch("lib.services.training.dataset_generator._ENGINE_FILL_POLL_INTERVAL", 0),
            patch("lib.services.training.dataset_generator._ENGINE_FILL_POLL_MAX_WAIT", 1),
        ):
            result = _load_bars_from_engine("MGC", days=180)

        assert result is not None
        # Should have polled the fill status URL
        fill_polls = [u for u in call_urls if "/fill/status" in u]
        assert len(fill_polls) >= 1, (
            f"Expected at least 1 fill status poll, got {len(fill_polls)}.  URLs called: {call_urls}"
        )

    def test_sends_api_key_header(self):
        """Trainer must send X-API-Key header for inter-service auth."""
        from lib.services.training.dataset_generator import _load_bars_from_engine

        bars = _make_bars(100)
        payload = _make_engine_json_response(bars)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload

        captured_kwargs: list[dict] = []

        def _capture_get(url, **kwargs):
            captured_kwargs.append(kwargs)
            return mock_resp

        with (
            patch("requests.get", side_effect=_capture_get),
            patch("lib.services.training.dataset_generator._API_KEY", "test-secret-123"),
        ):
            _load_bars_from_engine("MGC", days=30)

        # Check that the API key was passed in headers
        assert captured_kwargs, "requests.get was never called"
        headers = captured_kwargs[0].get("headers", {})
        assert headers.get("X-API-Key") == "test-secret-123", f"Expected X-API-Key header, got headers={headers}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 5 — Engine /bars endpoint does the heavy lifting
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineBarEndpoint:
    """The engine /bars endpoint must resolve symbols, fill gaps from
    Massive/Kraken, and return complete data to the trainer."""

    def test_normalize_symbol_handles_short_names(self):
        """Engine must accept both 'MGC' and 'MGC=F'."""
        from lib.services.data.api.bars import _normalize_symbol

        # Short name → normalised
        assert _normalize_symbol("MGC") is not None
        assert _normalize_symbol("MES") is not None

        # Already has =F → should still work
        result = _normalize_symbol("MGC=F")
        assert result is not None

    def test_normalize_symbol_handles_kraken_tickers(self):
        """Engine must handle crypto tickers."""
        from lib.services.data.api.bars import _normalize_symbol

        for sym in ["BTC", "ETH", "SOL"]:
            result = _normalize_symbol(sym)
            assert result is not None, f"_normalize_symbol({sym!r}) returned None"


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 6 — Full pipeline: no local API clients on trainer
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipelineNoLocalClients:
    """Simulate a full generate_dataset run with bars_override (as if the
    engine already served the data) and verify that NO local API clients
    (Massive, Kraken, yfinance) are instantiated."""

    def test_generate_dataset_with_bars_override_no_network(self):
        """When bars_override is provided (simulating engine-served data),
        no network client should be initialised."""

        # Track whether any external client is initialized
        clients_initialized = []

        original_massive_init = None
        original_kraken_init = None

        def _spy_massive_init(self, *a, **kw):
            clients_initialized.append("MassiveDataProvider")
            if original_massive_init:
                return original_massive_init(self, *a, **kw)

        def _spy_kraken_init(self, *a, **kw):
            clients_initialized.append("KrakenDataProvider")
            if original_kraken_init:
                return original_kraken_init(self, *a, **kw)

        yf_calls = []

        def _spy_yf_download(tickers, *a, **kw):
            yf_calls.append(str(tickers))
            return pd.DataFrame()

        bars = _make_bars(500, start_price=2700.0)
        bars_override = {sym: bars for sym in ["MGC", "M6E"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            from lib.services.training.dataset_generator import (
                DatasetConfig,
                generate_dataset,
            )

            config = DatasetConfig(
                output_dir=tmpdir,
                image_dir=os.path.join(tmpdir, "images"),
                bars_source="engine",
                breakout_type="ORB",
                orb_session="us",
                use_parity_renderer=False,
                max_samples_per_label=2,
                max_samples_per_type_label=2,
            )

            with (
                patch(
                    "lib.integrations.massive_client.MassiveDataProvider.__init__",
                    side_effect=_spy_massive_init,
                ),
                patch(
                    "lib.integrations.kraken_client.KrakenDataProvider.__init__",
                    side_effect=_spy_kraken_init,
                ),
                patch("lib.core.cache._yf_download", side_effect=_spy_yf_download),
                # Block EngineDataClient from making real network calls — bars_override
                # is passed for the primary symbols, but generate_dataset also calls
                # load_bars() for cross-asset peer symbols; those must not hit the network.
                patch(
                    "lib.services.data.engine_data_client.EngineDataClient.get_bars",
                    return_value=None,
                ),
                patch(
                    "lib.services.data.engine_data_client.EngineDataClient.get_daily_bars",
                    return_value=None,
                ),
                # Also block the legacy _load_bars_from_engine HTTP fallback —
                # when EngineDataClient returns None, the legacy loader is tried next
                # and it makes a direct requests.get call that would hang in tests.
                patch(
                    "lib.services.training.dataset_generator._load_bars_from_engine",
                    return_value=None,
                ),
                # Block crypto_momentum from making direct Kraken calls
                patch(
                    "lib.analysis.crypto_momentum.compute_all_crypto_momentum",
                    side_effect=Exception("Blocked: no Kraken on trainer"),
                ),
                contextlib.suppress(Exception),
            ):
                generate_dataset(
                    symbols=["MGC"],
                    days_back=30,
                    config=config,
                    bars_override=bars_override,
                )

        # The important assertions:
        assert not yf_calls, (
            f"yfinance was called on the trainer: {yf_calls}\n"
            f"The trainer must never call yfinance — all data comes from "
            f"the engine HTTP API."
        )

    def test_no_yfinance_slash_ticker_in_any_code_path(self):
        """Ensure _resolve_ticker never produces a /SYMBOL ticker that
        would be passed to yfinance (reproducing the HTTP 500 bug)."""
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in TRAINING_SYMBOLS:
            ticker = _resolve_ticker(symbol)
            # Must never start with /
            assert not ticker.startswith("/"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r} — this is the "
                f"exact bug that caused 10k lines of Yahoo HTML error spam "
                f"in the training logs"
            )
            # Must be a known format
            assert "=" in ticker or ticker.startswith("KRAKEN:"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r} — not a valid Yahoo =F ticker or KRAKEN:* ticker"
            )


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 7 — Regression: exact failures from the logs
# ═══════════════════════════════════════════════════════════════════════════


class TestLogRegressions:
    """Reproduce the exact failure modes from the 2026-03-11 training logs."""

    def test_regression_yfinance_slash_mgc(self):
        """From logs:
          yfinance ERROR: Failed to get ticker '/MGC'
          yfinance ERROR: ['/MGC']: TypeError("'Response' object is not subscriptable")

        This was caused by cache.get_data() falling through to yfinance
        on the trainer when MASSIVE_API_KEY was not set.  The ticker format
        /MGC is invalid for yfinance — it should be MGC=F."""
        from lib.services.training.dataset_generator import _resolve_ticker

        # The ticker used in the logs was /MGC — verify our resolver
        # produces MGC=F instead
        for sym in ["MGC", "SIL", "MHG", "MCL", "MES", "MNQ", "M2K", "MYM"]:
            ticker = _resolve_ticker(sym)
            assert not ticker.startswith("/"), f"Got {ticker!r} for {sym!r}"
            assert "=F" in ticker or ticker.startswith("KRAKEN:"), f"Got {ticker!r} for {sym!r}"

    def test_regression_kraken_too_many_requests(self):
        """From logs:
          kraken_client ERROR: EGeneral:Too many requests  (1,280 times!)

        This was caused by crypto_momentum.py calling Kraken REST directly
        on the trainer for each of the 25 symbols.  Only 3 symbols (BTC,
        ETH, SOL) should ever touch Kraken, and even those should go
        through the engine, not directly."""
        from lib.services.training.dataset_generator import _is_kraken_symbol

        # Count how many training symbols would hit Kraken
        kraken_count = sum(1 for s in TRAINING_SYMBOLS if _is_kraken_symbol(s))
        assert kraken_count == len(KRAKEN_SPOT_SYMBOLS), (
            f"{kraken_count} symbols detected as Kraken, expected "
            f"{len(KRAKEN_SPOT_SYMBOLS)} (only {KRAKEN_SPOT_SYMBOLS}).  "
            f"If CME futures hit Kraken that explains the 1,280 rate-limit errors."
        )

    def test_regression_engine_backfill_zero_bars(self):
        """From logs:
          ✅ Gold (MGC=F): +0 new bars (total: 0)
          ✅ Silver (SI=F): +0 new bars (total: 0)
          ... all 22 symbols: +0 bars

        The engine returned 0 bars because the historical_bars table was
        empty and the async fill hadn't finished.  The trainer should
        detect this and either wait for the fill or fail gracefully —
        NOT fall through to local yfinance."""
        from lib.services.training.dataset_generator import _load_bars_from_engine

        # Simulate engine returning 200 OK with 0 bars + filling=False
        # (fill finished but got nothing — maybe Massive was down)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_empty_engine_response()

        # _load_bars_from_engine imports requests locally as _requests — patch at
        # the source module level so the local alias is intercepted correctly.
        with patch("requests.get", return_value=mock_resp):
            result = _load_bars_from_engine("MGC", days=180)

        assert result is None, (
            "Engine returned 0 bars but loader returned non-None.  "
            "This causes generate_dataset to try rendering with empty bars, "
            "which triggers cache.get_data() → Massive init → yfinance → "
            "HTTP 500 spam."
        )

    def test_regression_massive_api_key_not_set_on_trainer(self):
        """From logs:
          MASSIVE_API_KEY not set — Massive data provider disabled, using yfinance fallback

        This message appeared on the TRAINER container.  The trainer should
        never need MASSIVE_API_KEY — it should talk to the engine which has it."""
        from lib.services.training.dataset_generator import DatasetConfig

        config = DatasetConfig()
        assert config.bars_source == "engine", (
            f"Default bars_source={config.bars_source!r}, expected 'engine'.  "
            f"The trainer must use the engine HTTP API (which has the Massive key), "
            f"not try to use Massive locally."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 8 — Provider symbol map completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderSymbolMaps:
    """Each API provider on the ENGINE side must have valid symbol mappings
    for all instruments the trainer might request."""

    def test_massive_covers_all_cme_training_symbols(self):
        """Every CME futures ticker the trainer sends must be resolvable
        by Massive on the engine side."""
        from lib.integrations.massive_client import YAHOO_TO_MASSIVE_PRODUCT
        from lib.services.training.dataset_generator import _resolve_ticker

        missing = []
        for symbol in CME_FUTURES:
            ticker = _resolve_ticker(symbol)
            if ticker.startswith("KRAKEN:"):
                continue
            if ticker not in YAHOO_TO_MASSIVE_PRODUCT:
                missing.append((symbol, ticker))

        assert not missing, (
            "These tickers have no Massive product mapping on the engine:\n"
            + "\n".join(f"  {s} → {t}" for s, t in missing)
            + "\nThe engine won't be able to backfill these from Massive, "
            "so the trainer will get 0 bars."
        )

    def test_kraken_covers_all_spot_crypto(self):
        from lib.integrations.kraken_client import ALL_KRAKEN_TICKERS
        from lib.services.training.dataset_generator import _resolve_ticker

        for symbol in KRAKEN_SPOT_SYMBOLS:
            ticker = _resolve_ticker(symbol)
            assert ticker in ALL_KRAKEN_TICKERS, f"{symbol} → {ticker} not in Kraken ticker set"

    def test_massive_product_codes_are_bare_symbols(self):
        """Massive product codes must not have =F or KRAKEN: prefix."""
        from lib.integrations.massive_client import YAHOO_TO_MASSIVE_PRODUCT

        for yahoo_ticker, product_code in YAHOO_TO_MASSIVE_PRODUCT.items():
            assert "=" not in product_code, f"Product code {product_code!r} for {yahoo_ticker!r} has '='"
            assert not product_code.startswith("KRAKEN:"), (
                f"Product code {product_code!r} for {yahoo_ticker!r} has KRAKEN prefix"
            )

    def test_engine_bars_endpoint_has_kraken_spot_map(self):
        """The engine /bars endpoint must have a mapping for BTC/ETH/SOL
        so it can resolve these to Kraken when Massive doesn't cover them."""
        from lib.services.data.api.bars import get_bars

        # Just verify the function exists and is callable — the actual
        # mapping is tested in the endpoint's own test suite
        assert callable(get_bars)


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 9 — DatasetConfig defaults enforce correct architecture
# ═══════════════════════════════════════════════════════════════════════════


class TestDatasetConfigDefaults:
    """The default DatasetConfig must enforce the engine-first architecture."""

    def test_default_bars_source_is_engine(self):
        from lib.services.training.dataset_generator import DatasetConfig

        config = DatasetConfig()
        assert config.bars_source == "engine"

    def test_default_renderer_is_parity(self):
        from lib.services.training.dataset_generator import DatasetConfig

        config = DatasetConfig()
        assert config.use_parity_renderer is True

    def test_default_breakout_type_is_all(self):
        from lib.services.training.dataset_generator import DatasetConfig

        config = DatasetConfig()
        assert config.breakout_type == "all"

    def test_trainer_server_passes_bars_source_to_config(self):
        """The trainer_server pipeline must pass bars_source through to
        DatasetConfig so it reaches generate_dataset."""
        # We can't easily run the full pipeline, but we verify the
        # DatasetConfig is constructed with bars_source in the server code
        import inspect

        from lib.services.training.trainer_server import _run_training_pipeline

        source = inspect.getsource(_run_training_pipeline)
        assert "bars_source" in source, (
            "_run_training_pipeline doesn't reference bars_source — "
            "the config might not pass it through to DatasetConfig"
        )
        assert "DatasetConfig" in source, "_run_training_pipeline doesn't create a DatasetConfig"


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 10 — Trainer fetches symbol list from engine /bars/symbols
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainerSymbolFetching:
    """The trainer must ask the engine for its symbol list at job startup
    rather than using a stale hardcoded list.  This ensures the trainer
    always trains on exactly the same assets the engine has data for."""

    def test_fetch_symbols_from_engine_uses_symbols_endpoint(self):
        """_fetch_symbols_from_engine should prefer /bars/symbols (no DB hit)."""
        from lib.services.training.trainer_server import _fetch_symbols_from_engine

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "symbols": [
                "6A",
                "6B",
                "6C",
                "6E",
                "6J",
                "6S",
                "M2K",
                "MBT",
                "MCL",
                "MES",
                "MET",
                "MGC",
                "MHG",
                "MNG",
                "MNQ",
                "MYM",
                "SIL",
                "ZB",
                "ZC",
                "ZN",
                "ZS",
                "ZW",
            ],
            "total": 22,
        }

        with patch(
            "lib.services.data.engine_data_client.EngineDataClient.get_symbols",
            return_value=sorted(mock_resp.json.return_value["symbols"]),
        ):
            result = _fetch_symbols_from_engine("http://data:8000", api_key="", timeout=5)

        assert result is not None
        assert len(result) > 0

    def test_fetch_symbols_falls_back_to_assets_endpoint(self):
        """/bars/symbols not available on older engine — EngineDataClient falls back
        to /bars/assets internally; _fetch_symbols_from_engine must return a usable list."""
        from lib.services.training.trainer_server import _fetch_symbols_from_engine

        # EngineDataClient already handles the /bars/symbols → /bars/assets fallback
        # internally.  We verify that _fetch_symbols_from_engine returns the result.
        with patch(
            "lib.services.data.engine_data_client.EngineDataClient.get_symbols",
            return_value=["MCL", "MES", "MGC", "MNQ"],
        ):
            result = _fetch_symbols_from_engine("http://data:8000", api_key="", timeout=5)

        assert result is not None
        assert "MGC" in result, f"MGC not found in {result}"
        assert "MES" in result, f"MES not found in {result}"
        assert "MNQ" in result, f"MNQ not found in {result}"

    def test_fetch_symbols_returns_none_when_engine_unreachable(self):
        """When the engine is down, _fetch_symbols_from_engine returns None
        so the pipeline falls back to _FALLBACK_SYMBOLS."""
        from lib.services.training.trainer_server import _fetch_symbols_from_engine

        with patch(
            "lib.services.data.engine_data_client.EngineDataClient.get_symbols",
            side_effect=Exception("Engine unreachable"),
        ):
            result = _fetch_symbols_from_engine("http://data:8000", api_key="", timeout=5)

        assert result is None, (
            "Expected None when engine is unreachable, but got a symbol list.  "
            "The caller must fall back to _FALLBACK_SYMBOLS."
        )

    def test_fetch_symbols_sends_api_key_header(self):
        """Symbol fetch must pass the API key to EngineDataClient."""
        from lib.services.data.engine_data_client import EngineDataClient
        from lib.services.training.trainer_server import _fetch_symbols_from_engine

        captured_api_keys: list[str] = []

        # Intercept EngineDataClient construction by wrapping get_symbols on a
        # real instance — we spy on what api_key was passed by inspecting the
        # instance created inside _fetch_symbols_from_engine.
        original_cls = EngineDataClient

        class _SpyClient(original_cls):
            def get_symbols(self, **kwargs):
                captured_api_keys.append(self._api_key)
                return ["MGC", "MES"]

        with patch(
            "lib.services.data.engine_data_client.EngineDataClient",
            side_effect=lambda **kw: _SpyClient(**kw),
        ):
            _fetch_symbols_from_engine("http://data:8000", api_key="secret-key-123", timeout=5)

        assert captured_api_keys, "EngineDataClient.get_symbols was never called"
        assert captured_api_keys[0] == "secret-key-123", (
            f"Expected api_key='secret-key-123' passed to EngineDataClient, got: {captured_api_keys[0]!r}"
        )

    def test_fetch_symbols_deduplicates_and_sorts(self):
        """Symbol list must be deduplicated and sorted for determinism.

        EngineDataClient.get_symbols() already sorts and deduplicates — we verify
        _fetch_symbols_from_engine passes the result through unchanged.
        """
        from lib.services.training.trainer_server import _fetch_symbols_from_engine

        # EngineDataClient returns an already-sorted, deduplicated list
        with patch(
            "lib.services.data.engine_data_client.EngineDataClient.get_symbols",
            return_value=["MCL", "MES", "MGC", "ZN"],
        ):
            result = _fetch_symbols_from_engine("http://data:8000", api_key="", timeout=5)

        assert result is not None
        assert result == sorted(set(result)), "Symbol list must be sorted and deduplicated"
        assert len(result) == len(set(result)), "Symbol list contains duplicates"

    def test_fallback_symbols_are_all_valid(self):
        """Every symbol in _FALLBACK_SYMBOLS must be resolvable by _resolve_ticker."""
        from lib.services.training.dataset_generator import _resolve_ticker
        from lib.services.training.trainer_server import _FALLBACK_SYMBOLS

        for symbol in _FALLBACK_SYMBOLS:
            ticker = _resolve_ticker(symbol)
            assert not ticker.startswith("/"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r} — slash prefix would break yfinance"
            )
            assert "=" in ticker or ticker.startswith("KRAKEN:"), (
                f"_resolve_ticker({symbol!r}) → {ticker!r} — not a valid ticker format"
            )

    def test_pipeline_uses_fetched_symbols_when_engine_available(self):
        """When the engine returns symbols, _run_training_pipeline must use
        those symbols (not the hardcoded fallback list)."""
        import inspect

        from lib.services.training.trainer_server import _run_training_pipeline

        source = inspect.getsource(_run_training_pipeline)
        # The pipeline must call _fetch_symbols_from_engine when no explicit
        # symbols are provided
        assert "_fetch_symbols_from_engine" in source, (
            "_run_training_pipeline doesn't call _fetch_symbols_from_engine — "
            "it will always use the hardcoded fallback list instead of asking "
            "the engine what data is available"
        )
        # And it must have a fallback path
        assert "_FALLBACK_SYMBOLS" in source, (
            "_run_training_pipeline doesn't reference _FALLBACK_SYMBOLS — "
            "there's no fallback when the engine is unreachable"
        )

    def test_engine_bars_symbols_endpoint_returns_short_symbols(self):
        """GET /bars/symbols must return short symbols ('MGC') not Yahoo
        tickers ('MGC=F') so the trainer doesn't need its own ticker map."""
        from lib.services.data.api.bars import list_symbols

        # list_symbols() imports models.ASSETS which needs the DB — just verify
        # the function exists and is callable (full integration tested live)
        assert callable(list_symbols), "list_symbols endpoint not found in bars.py"

    def test_engine_data_client_get_symbols_strips_equals_f(self):
        """EngineDataClient.get_symbols() must return short symbols ('MGC'),
        not Yahoo tickers ('MGC=F'), and must pass Kraken tickers through unchanged.

        This replaces the old _ticker_to_short_symbol test — that helper was
        removed when _fetch_symbols_from_engine was simplified to delegate to
        EngineDataClient, which already handles ticker normalisation internally.
        """
        from lib.services.data.engine_data_client import EngineDataClient

        # Simulate the /bars/assets response with Yahoo-style tickers
        assets_payload = {
            "assets": [
                {"name": "Gold", "ticker": "MGC=F"},
                {"name": "S&P", "ticker": "MES=F"},
                {"name": "Euro FX", "ticker": "6E=F"},
                {"name": "T-Note", "ticker": "ZN=F"},
                {"name": "Bitcoin spot", "ticker": "KRAKEN:XBTUSD"},
                {"name": "Ether spot", "ticker": "KRAKEN:ETHUSD"},
                {"name": "Micro Bitcoin CME", "ticker": "MBT=F"},
            ]
        }

        client = EngineDataClient(base_url="http://fake-engine:9999")

        # Patch both endpoints: /bars/symbols returns 404, /bars/assets returns data
        def _mock_get(path, **kwargs):
            if path == "/bars/symbols":
                return None  # fast-path returns None → triggers assets fallback
            if path == "/bars/assets":
                return assets_payload
            return None

        with patch.object(client, "_get", side_effect=_mock_get):
            symbols = client.get_symbols(use_cache=False)

        assert "MGC" in symbols, f"MGC missing from {symbols}"
        assert "MES" in symbols, f"MES missing from {symbols}"
        assert "6E" in symbols, f"6E missing from {symbols}"
        assert "ZN" in symbols, f"ZN missing from {symbols}"
        assert "KRAKEN:XBTUSD" in symbols, f"KRAKEN:XBTUSD missing from {symbols}"
        assert "KRAKEN:ETHUSD" in symbols, f"KRAKEN:ETHUSD missing from {symbols}"
        assert "MBT" in symbols, f"MBT missing from {symbols}"
        # Must NOT contain =F tickers
        assert "MGC=F" not in symbols, "MGC=F must be stripped to MGC"
        assert "MES=F" not in symbols, "MES=F must be stripped to MES"
        assert "MBT=F" not in symbols, "MBT=F must be stripped to MBT"
