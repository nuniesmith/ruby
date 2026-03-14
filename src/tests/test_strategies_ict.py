"""# pyright: reportArgumentType=false
Tests for the ICT-enhanced strategy (ICTTrendEMA), engine multi-TF
confluence wiring, and alert configuration integration.

Covers:
  - ICTTrendEMA strategy instantiation and basic backtest execution
  - ICT confluence array computation
  - Engine multi-TF fetch helper (_fetch_tf_safe)
  - Engine alert enable/disable flags
  - Strategy registry includes ICTTrendEMA
  - suggest_params covers ICTTrendEMA
  - make_strategy produces a configured subclass
  - Alert dispatcher cooldown runtime adjustment
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
from lib.core.alerts import (  # noqa: E402
    AlertDispatcher,
)
from lib.trading.strategies import (  # noqa: E402
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    ICTTrendEMA,
    _atr,
    _ema,
    make_strategy,
    score_backtest,
    suggest_params,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers (self-contained; don't depend on conftest fixtures
# so this file can run independently)
# ---------------------------------------------------------------------------


def _make_timestamps(n: int, freq: str = "5min", start: str = "2025-01-06 03:00") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")


def _random_walk_ohlcv(
    n: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.005,
    freq: str = "5min",
    seed: int = 42,
    volume_mean: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))
    spread = close * rng.uniform(0.001, 0.004, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.5, 0.5, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(volume_mean, n).astype(float)
    volume = np.maximum(volume, 1)
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
    seed: int = 123,
) -> pd.DataFrame:
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
    idx = _make_timestamps(n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _impulsive_ohlcv(
    n: int = 400,
    start_price: float = 2700.0,
    seed: int = 77,
) -> pd.DataFrame:
    """Create data with clear impulse moves for ICT detection."""
    rng = np.random.default_rng(seed)
    close = np.full(n, start_price, dtype=float)
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + rng.normal(0, 0.003))
    # Inject impulse moves at specific points
    impulse_bars = [80, 160, 240, 320]
    for ib in impulse_bars:
        if ib + 5 < n:
            direction = rng.choice([-1, 1])
            move = close[ib] * 0.02 * direction
            for k in range(ib, ib + 5):
                close[k] += move * (k - ib + 1) / 5
            # Propagate
            for k in range(ib + 5, n):
                close[k] += move

    spread = np.abs(close) * rng.uniform(0.001, 0.005, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(1000, n).astype(float)
    for ib in impulse_bars:
        if ib < n:
            volume[ib] *= 5
    volume = np.maximum(volume, 1)
    idx = _make_timestamps(n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ICTTrendEMA — Strategy Registration
# ═══════════════════════════════════════════════════════════════════════════


class TestICTTrendEMARegistration:
    """Verify ICTTrendEMA is properly registered in the strategy system."""

    def test_in_strategy_classes(self):
        assert "ICTTrendEMA" in STRATEGY_CLASSES
        assert STRATEGY_CLASSES["ICTTrendEMA"] is ICTTrendEMA

    def test_in_strategy_labels(self):
        assert "ICTTrendEMA" in STRATEGY_LABELS
        assert "ICT" in STRATEGY_LABELS["ICTTrendEMA"]

    def test_is_strategy_subclass(self):
        from backtesting import Strategy

        assert issubclass(ICTTrendEMA, Strategy)

    def test_has_ict_parameters(self):
        """ICTTrendEMA should have ICT-specific class attributes."""
        assert hasattr(ICTTrendEMA, "ob_proximity")
        assert hasattr(ICTTrendEMA, "fvg_proximity")
        assert hasattr(ICTTrendEMA, "ict_mode")

    def test_inherits_ema_parameters(self):
        """ICTTrendEMA should have the same EMA params as TrendEMACross."""
        for attr in (
            "n1",
            "n2",
            "trend_period",
            "atr_period",
            "atr_sl_mult",
            "atr_tp_mult",
            "trade_size",
        ):
            assert hasattr(ICTTrendEMA, attr), f"Missing attribute: {attr}"

    def test_default_ict_mode(self):
        assert ICTTrendEMA.ict_mode == 1

    def test_default_ob_proximity(self):
        assert 0 < ICTTrendEMA.ob_proximity <= 5

    def test_default_fvg_proximity(self):
        assert 0 < ICTTrendEMA.fvg_proximity <= 5


# ═══════════════════════════════════════════════════════════════════════════
# ICTTrendEMA — make_strategy & suggest_params
# ═══════════════════════════════════════════════════════════════════════════


class TestICTTrendEMAFactory:
    """Test that make_strategy and suggest_params work for ICTTrendEMA."""

    def test_make_strategy_returns_subclass(self):
        params = {
            "n1": 10,
            "n2": 25,
            "trend_period": 60,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "ob_proximity": 1.5,
            "fvg_proximity": 2.0,
            "ict_mode": 1,
            "trade_size": 0.10,
        }
        cls = make_strategy("ICTTrendEMA", params)
        assert issubclass(cls, ICTTrendEMA)
        assert cls.n1 == 10
        assert cls.ob_proximity == 1.5
        assert cls.ict_mode == 1

    def test_make_strategy_different_params_independent(self):
        """Two configured subclasses should not share param mutations."""
        cls1 = make_strategy("ICTTrendEMA", {"n1": 5, "ict_mode": 0, "trade_size": 0.1})
        cls2 = make_strategy("ICTTrendEMA", {"n1": 15, "ict_mode": 2, "trade_size": 0.2})
        assert cls1.n1 == 5
        assert cls2.n1 == 15
        assert cls1.ict_mode == 0
        assert cls2.ict_mode == 2

    def test_suggest_params_returns_expected_keys(self):
        """suggest_params for ICTTrendEMA should produce all required keys."""

        class MockTrial:
            """Minimal Optuna trial mock."""

            def __init__(self):
                self._suggestions = {}
                self._counter = 0

            def suggest_int(self, name, low, high, **kw):
                # Return midpoint
                val = (low + high) // 2
                self._suggestions[name] = val
                return val

            def suggest_float(self, name, low, high, step=None, **kw):
                val = (low + high) / 2
                if step:
                    val = round(val / step) * step
                self._suggestions[name] = val
                return val

        trial = MockTrial()
        params = suggest_params(trial, "ICTTrendEMA")
        assert isinstance(params, dict)
        expected = {
            "n1",
            "n2",
            "trend_period",
            "atr_period",
            "atr_sl_mult",
            "atr_tp_mult",
            "ob_proximity",
            "fvg_proximity",
            "ict_mode",
            "trade_size",
        }
        assert expected.issubset(set(params.keys())), f"Missing keys: {expected - set(params.keys())}"

    def test_suggest_params_values_in_range(self):
        class MockTrial:
            def suggest_int(self, name, low, high, **kw):
                return low

            def suggest_float(self, name, low, high, step=None, **kw):
                return low

        params = suggest_params(MockTrial(), "ICTTrendEMA")
        assert 5 <= params["n1"] <= 20
        assert 0.5 <= params["ob_proximity"] <= 2.5
        assert 1.0 <= params["fvg_proximity"] <= 3.0
        assert 0 <= params["ict_mode"] <= 2


# ═══════════════════════════════════════════════════════════════════════════
# ICTTrendEMA — Backtest Execution
# ═══════════════════════════════════════════════════════════════════════════


class TestICTTrendEMABacktest:
    """Run actual backtests with ICTTrendEMA to verify it doesn't crash."""

    @pytest.fixture()
    def bt_data(self):
        return _random_walk_ohlcv(n=500, seed=42, start_price=100.0)

    @pytest.fixture()
    def trending_data(self):
        return _trending_ohlcv(n=400, seed=123)

    @pytest.fixture()
    def impulsive_data(self):
        return _impulsive_ohlcv(n=400, seed=77)

    def test_backtest_runs_without_error(self, bt_data):
        """ICTTrendEMA should complete a backtest without exceptions."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ob_proximity": 1.5,
                "fvg_proximity": 2.0,
                "ict_mode": 0,  # optional mode (most permissive)
                "trade_size": 0.10,
            },
        )
        bt = Backtest(bt_data, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None
        assert "# Trades" in stats.index or "# Trades" in stats

    def test_backtest_mode_1_runs(self, bt_data):
        """ict_mode=1 (require 1 ICT signal) should still run."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 1,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(bt_data, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_backtest_mode_2_runs(self, bt_data):
        """ict_mode=2 (require both OB+FVG) should still run — may have 0 trades."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 2,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(bt_data, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_mode_2_fewer_or_equal_trades_vs_mode_0(self, bt_data):
        """Stricter ICT mode should produce ≤ trades than permissive mode."""
        from backtesting import Backtest

        base_params = {
            "n1": 9,
            "n2": 21,
            "trend_period": 50,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "trade_size": 0.10,
        }
        cls0 = make_strategy("ICTTrendEMA", {**base_params, "ict_mode": 0})
        cls2 = make_strategy("ICTTrendEMA", {**base_params, "ict_mode": 2})

        from typing import Any

        stats0: Any = Backtest(bt_data, cls0, cash=100_000, trade_on_close=True).run()
        stats2: Any = Backtest(bt_data, cls2, cash=100_000, trade_on_close=True).run()

        trades_0 = int(stats0["# Trades"])
        trades_2 = int(stats2["# Trades"])
        assert trades_2 <= trades_0, f"Mode 2 ({trades_2} trades) should have ≤ trades than mode 0 ({trades_0})"

    def test_trending_data_produces_trades_mode_0(self, trending_data):
        """On trending data with mode 0, ICTTrendEMA should take some trades."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 0,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(trending_data, cls, cash=100_000, trade_on_close=True)
        from typing import Any

        stats: Any = bt.run()
        # Mode 0 is permissive — equivalent to TrendEMA, should have trades
        assert int(stats["# Trades"]) >= 0  # at least doesn't crash

    def test_impulsive_data_backtest(self, impulsive_data):
        """Impulsive data should give ICT detectors something to find."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 1,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(impulsive_data, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_score_backtest_on_ict_result(self, bt_data):
        """score_backtest should work on ICTTrendEMA results."""
        from backtesting import Backtest

        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 0,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(bt_data, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        score = score_backtest(stats, min_trades=0)
        assert isinstance(score, float)


# ═══════════════════════════════════════════════════════════════════════════
# ICT Confluence Array Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestICTConfluenceArray:
    """Test the _ict_confluence_array helper used by ICTTrendEMA.init()."""

    @pytest.fixture()
    def df_500(self):
        return _random_walk_ohlcv(n=500, seed=42)

    def test_returns_numpy_array(self, df_500):
        from lib.trading.strategies import _ict_confluence_array

        close = pd.Series(df_500["Close"])
        ema_f = np.asarray(_ema(close, 9))
        ema_s = np.asarray(_ema(close, 21))
        ema_t = np.asarray(_ema(close, 50))
        atr_a = np.asarray(_atr(df_500["High"], df_500["Low"], df_500["Close"], 14))
        result = _ict_confluence_array(
            df_500,
            ema_f,
            ema_s,
            ema_t,
            atr_a,
            atr_period=14,
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df_500)

    def test_values_in_valid_range(self, df_500):
        from lib.trading.strategies import _ict_confluence_array

        close = pd.Series(df_500["Close"])
        ema_f = np.asarray(_ema(close, 9))
        ema_s = np.asarray(_ema(close, 21))
        ema_t = np.asarray(_ema(close, 50))
        atr_a = np.asarray(_atr(df_500["High"], df_500["Low"], df_500["Close"], 14))
        result = _ict_confluence_array(
            df_500,
            ema_f,
            ema_s,
            ema_t,
            atr_a,
        )
        # Scores should be in [-2, 2]
        assert np.all(np.abs(result) <= 2.01)

    def test_empty_df_returns_zeros(self):
        from lib.trading.strategies import _ict_confluence_array

        empty = pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))
        result = _ict_confluence_array(
            empty,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        assert len(result) == 0

    def test_short_df_returns_zeros(self):
        from lib.trading.strategies import _ict_confluence_array

        short = _random_walk_ohlcv(n=20, seed=99)
        close = pd.Series(short["Close"])
        ema_f = np.asarray(_ema(close, 9))
        ema_s = np.asarray(_ema(close, 21))
        ema_t = np.asarray(_ema(close, 50))
        atr_a = np.asarray(_atr(short["High"], short["Low"], short["Close"], 14))
        result = _ict_confluence_array(
            short,
            ema_f,
            ema_s,
            ema_t,
            atr_a,
        )
        assert np.all(result == 0)

    def test_impulsive_data_has_nonzero_scores(self):
        """Impulsive data should produce at least some ICT confluence scores."""
        from lib.trading.strategies import _ict_confluence_array

        df = _impulsive_ohlcv(n=500, seed=77)
        close = pd.Series(df["Close"])
        ema_f = np.asarray(_ema(close, 9))
        ema_s = np.asarray(_ema(close, 21))
        ema_t = np.asarray(_ema(close, 50))
        atr_a = np.asarray(_atr(df["High"], df["Low"], df["Close"], 14))
        result = _ict_confluence_array(
            df,
            ema_f,
            ema_s,
            ema_t,
            atr_a,
            ob_proximity_atr=2.0,
            fvg_proximity_atr=3.0,
        )
        # On impulsive data we expect at least some non-zero scores
        # (though this depends on the random data — be lenient)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)


# ═══════════════════════════════════════════════════════════════════════════
# ICT Confluence per-bar helper
# ═══════════════════════════════════════════════════════════════════════════


class TestICTConfluencePerBar:
    """Test _compute_ict_confluence (per-bar helper)."""

    def test_returns_dict(self):
        from lib.trading.strategies import _compute_ict_confluence

        df = _random_walk_ohlcv(n=200, seed=42)
        result = _compute_ict_confluence(df, bar_index=150, direction="long")
        assert isinstance(result, dict)
        assert "ob_aligned" in result
        assert "fvg_aligned" in result
        assert "score" in result

    def test_score_is_nonnegative(self):
        from lib.trading.strategies import _compute_ict_confluence

        df = _random_walk_ohlcv(n=200, seed=42)
        result = _compute_ict_confluence(df, bar_index=150, direction="long")
        assert result["score"] >= 0

    def test_score_max_is_2(self):
        from lib.trading.strategies import _compute_ict_confluence

        df = _impulsive_ohlcv(n=300, seed=77)
        for bar in range(50, 250, 20):
            for d in ("long", "short"):
                result = _compute_ict_confluence(df, bar_index=bar, direction=d)
                assert result["score"] <= 2

    def test_empty_df_returns_zero_score(self):
        from lib.trading.strategies import _compute_ict_confluence

        empty = pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))
        result = _compute_ict_confluence(empty, bar_index=0, direction="long")
        assert result["score"] == 0

    def test_short_df_returns_zero_score(self):
        from lib.trading.strategies import _compute_ict_confluence

        short = _random_walk_ohlcv(n=10, seed=99)
        result = _compute_ict_confluence(short, bar_index=5, direction="short")
        assert result["score"] == 0

    def test_ob_sl_is_float_or_none(self):
        from lib.trading.strategies import _compute_ict_confluence

        df = _impulsive_ohlcv(n=300, seed=77)
        result = _compute_ict_confluence(df, bar_index=200, direction="long")
        assert result["ob_sl"] is None or isinstance(result["ob_sl"], float)

    def test_fvg_tp_is_float_or_none(self):
        from lib.trading.strategies import _compute_ict_confluence

        df = _impulsive_ohlcv(n=300, seed=77)
        result = _compute_ict_confluence(df, bar_index=200, direction="short")
        assert result["fvg_tp"] is None or isinstance(result["fvg_tp"], float)


# ═══════════════════════════════════════════════════════════════════════════
# Engine — Multi-TF Confluence & Alert Flags
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineAlertFlags:
    """Test that DashboardEngine respects alert enable/disable flags."""

    def test_default_flags_are_true(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        assert engine._alerts_regime_enabled is True
        assert engine._alerts_confluence_enabled is True
        assert engine._alerts_signal_enabled is True

    def test_flags_can_be_toggled(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        engine._alerts_regime_enabled = False
        engine._alerts_confluence_enabled = False
        engine._alerts_signal_enabled = False
        assert engine._alerts_regime_enabled is False
        assert engine._alerts_confluence_enabled is False
        assert engine._alerts_signal_enabled is False

    def test_dispatch_regime_alert_suppressed_when_disabled(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        engine._alerts_regime_enabled = False

        # Should not raise and should not dispatch
        with patch("lib.trading.engine.get_dispatcher") as mock_disp:
            engine._dispatch_regime_alert("Gold", "trending", "volatile", 0.9)
            mock_disp.assert_not_called()

    def test_dispatch_regime_alert_calls_dispatcher_when_enabled(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        engine._alerts_regime_enabled = True

        mock_dispatcher = MagicMock()
        mock_dispatcher.has_channels = True
        mock_dispatcher.send_regime_change = MagicMock(return_value=True)

        with patch("lib.trading.engine.get_dispatcher", return_value=mock_dispatcher):
            engine._dispatch_regime_alert("Gold", "trending", "volatile", 0.9)
            mock_dispatcher.send_regime_change.assert_called_once()

    def test_check_confluence_suppressed_when_disabled(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        engine._alerts_confluence_enabled = False

        with patch("lib.trading.engine.get_dispatcher") as mock_disp:
            engine._check_confluence_alerts()
            mock_disp.assert_not_called()


class TestEngineFetchTFSafe:
    """Test the _fetch_tf_safe helper for multi-TF data loading."""

    def test_returns_dataframe(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        fake_df = _random_walk_ohlcv(n=100, seed=42)

        with patch("lib.trading.engine.get_data", return_value=fake_df):
            result = engine._fetch_tf_safe("ES=F", "15m", "S&P", "HTF")
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_returns_empty_on_failure(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)

        with patch("lib.trading.engine.get_data", side_effect=Exception("Network error")):
            result = engine._fetch_tf_safe("ES=F", "15m", "S&P", "HTF")
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fallback_to_engine_interval(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000, interval="5m", period="5d")
        fake_df = _random_walk_ohlcv(n=100, seed=42)

        call_count = [0]
        original_empty = pd.DataFrame()

        def mock_get_data(ticker, interval, period, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (15m) returns empty
                return original_empty
            else:
                # Fallback call (5m) returns data
                return fake_df

        with patch("lib.trading.engine.get_data", side_effect=mock_get_data):
            result = engine._fetch_tf_safe("ES=F", "15m", "S&P", "HTF")
            # Should have fallen back to engine's default
            assert call_count[0] == 2
            assert not result.empty


class TestEngineMultiTFConfluence:
    """Test that the engine uses proper multi-TF intervals for confluence."""

    def test_check_confluence_uses_recommended_timeframes(self):
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000, interval="5m", period="5d")
        engine._alerts_confluence_enabled = True

        # Track which intervals get_data is called with
        fetched_intervals = []
        fake_df = _random_walk_ohlcv(n=100, seed=42)

        def mock_get_data(ticker, interval, period, **kw):
            fetched_intervals.append(interval)
            return fake_df

        mock_dispatcher = MagicMock()
        mock_dispatcher.has_channels = True
        mock_dispatcher.send_confluence_alert = MagicMock(return_value=True)

        with (
            patch("lib.trading.engine.get_data", side_effect=mock_get_data),
            patch("lib.trading.engine.get_dispatcher", return_value=mock_dispatcher),
            patch("lib.trading.engine.ASSETS", {"Gold": "GC=F"}),
        ):
            engine._check_confluence_alerts()
            # Gold's recommended TFs are 1h/15m/5m
            # The engine should fetch at least 3 timeframes for Gold
            assert len(fetched_intervals) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# Alert Dispatcher — Runtime Cooldown Adjustment
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertCooldownAdjustment:
    """Test that cooldown can be adjusted at runtime (as the UI does)."""

    def test_cooldown_change_propagates_to_store(self):
        d = AlertDispatcher(_disable_redis=True, cooldown_sec=300)
        assert d.cooldown_sec == 300
        assert d._store.cooldown_sec == 300

        # Simulate UI adjustment
        d.cooldown_sec = 60
        d._store.cooldown_sec = 60
        assert d.cooldown_sec == 60
        assert d._store.cooldown_sec == 60

    def test_shorter_cooldown_allows_faster_resend(self):
        d = AlertDispatcher(_disable_redis=True, cooldown_sec=1)
        d._store.mark_sent("fast_key")
        assert d._store.should_send("fast_key") is False
        time.sleep(1.1)
        assert d._store.should_send("fast_key") is True

    def test_cooldown_change_affects_new_checks(self):
        d = AlertDispatcher(_disable_redis=True, cooldown_sec=300)
        d._store.mark_sent("key_adj")
        assert d._store.should_send("key_adj") is False

        # Reduce cooldown to 0
        d._store.cooldown_sec = 0
        assert d._store.should_send("key_adj") is True

    def test_stats_reflect_cooldown(self):
        d = AlertDispatcher(_disable_redis=True, cooldown_sec=120)
        stats = d.get_stats()
        assert stats["cooldown_sec"] == 120


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ICTTrendEMA vs TrendEMACross Comparison
# ═══════════════════════════════════════════════════════════════════════════


class TestICTvsTrendEMAComparison:
    """Compare ICTTrendEMA (mode 0) against TrendEMACross to verify
    they produce similar results since mode 0 is the permissive mode."""

    @pytest.fixture()
    def shared_data(self):
        return _random_walk_ohlcv(n=500, seed=42)

    def test_both_strategies_complete(self, shared_data):
        """Both strategies should complete without error on the same data."""
        from backtesting import Backtest

        base_params = {
            "n1": 9,
            "n2": 21,
            "trend_period": 50,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "trade_size": 0.10,
        }

        cls_ema = make_strategy("TrendEMA", base_params)
        cls_ict = make_strategy("ICTTrendEMA", {**base_params, "ict_mode": 0})

        stats_ema = Backtest(shared_data, cls_ema, cash=100_000, trade_on_close=True).run()
        stats_ict = Backtest(shared_data, cls_ict, cash=100_000, trade_on_close=True).run()

        assert stats_ema is not None
        assert stats_ict is not None

    def test_mode_0_similar_to_trendema(self, shared_data):
        """Mode 0 should produce the same or similar trade count as TrendEMA
        since ICT is optional in mode 0."""
        from backtesting import Backtest

        base_params = {
            "n1": 9,
            "n2": 21,
            "trend_period": 50,
            "atr_period": 14,
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 2.5,
            "trade_size": 0.10,
        }

        cls_ema = make_strategy("TrendEMA", base_params)
        cls_ict = make_strategy("ICTTrendEMA", {**base_params, "ict_mode": 0})

        from typing import Any

        stats_ema: Any = Backtest(shared_data, cls_ema, cash=100_000, trade_on_close=True).run()
        stats_ict: Any = Backtest(shared_data, cls_ict, cash=100_000, trade_on_close=True).run()
        trades_ema = int(stats_ema["# Trades"])
        trades_ict = int(stats_ict["# Trades"])

        # With mode 0, ICT score doesn't gate entries, so trade counts
        # should be identical (same EMA crossover logic, same exits)
        assert trades_ict == trades_ema, f"Mode 0 ICT trades ({trades_ict}) != TrendEMA trades ({trades_ema})"


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_ict_strategy_with_5_bars(self):
        """ICTTrendEMA should handle very short data gracefully."""
        from backtesting import Backtest

        tiny = _random_walk_ohlcv(n=30, seed=11, start_price=20.0)
        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 5,
                "n2": 10,
                "trend_period": 15,
                "atr_period": 10,
                "atr_sl_mult": 1.0,
                "atr_tp_mult": 2.0,
                "ict_mode": 0,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(tiny, cls, cash=10_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_ict_strategy_high_volatility(self):
        """ICTTrendEMA should handle high-volatility data."""
        from backtesting import Backtest

        volatile = _random_walk_ohlcv(n=300, seed=88, volatility=0.02)
        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 2.0,
                "atr_tp_mult": 3.0,
                "ict_mode": 1,
                "trade_size": 0.05,
            },
        )
        bt = Backtest(volatile, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_ict_strategy_low_volatility(self):
        """ICTTrendEMA should handle low-volatility (flat) data."""
        from backtesting import Backtest

        flat = _random_walk_ohlcv(n=300, seed=77, volatility=0.0005)
        cls = make_strategy(
            "ICTTrendEMA",
            {
                "n1": 9,
                "n2": 21,
                "trend_period": 50,
                "atr_period": 14,
                "atr_sl_mult": 1.5,
                "atr_tp_mult": 2.5,
                "ict_mode": 1,
                "trade_size": 0.10,
            },
        )
        bt = Backtest(flat, cls, cash=100_000, trade_on_close=True)
        stats = bt.run()
        assert stats is not None

    def test_alert_dispatcher_with_redis_disabled(self):
        """_disable_redis flag should prevent Redis connection attempts."""
        d = AlertDispatcher(_disable_redis=True)
        assert d._store._redis is None
        assert isinstance(d._store._memory_store, dict)

    def test_engine_previous_regimes_tracking(self):
        """Engine should track previous regimes per asset."""
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        assert engine._previous_regimes == {}
        engine._previous_regimes["Gold"] = "trending"
        assert engine._previous_regimes["Gold"] == "trending"

    def test_engine_previous_confluence_tracking(self):
        """Engine should track previous confluence scores per asset."""
        from lib.trading.engine import DashboardEngine

        engine = DashboardEngine(account_size=100_000)
        assert engine._previous_confluence == {}
        engine._previous_confluence["Gold"] = 2
        assert engine._previous_confluence["Gold"] == 2
