"""
Background engine that keeps market data fresh, runs Optuna optimizations
when cached results expire, and backtests all assets with optimal parameters.

# pyright: reportArgumentType=false

Improvements over the original:
  - Walk-forward validation (train/test split) to prevent overfitting
  - Session time filtering (only backtest during the 3 AM–noon EST window)
  - Volatility regime detection (low/normal/high) per asset
  - Strategy confidence scoring based on out-of-sample consistency
  - Eight strategies optimized: TrendEMA, RSI, Breakout, VWAP, ORB, MACD, PullbackEMA, EventReaction
  - Engine-triggered alerts: auto-dispatch on regime changes and full confluence

Usage:
    from lib.engine import get_engine
    engine = get_engine(account_size=150_000, interval="5m", period="5d")
    # engine auto-starts a daemon thread on first call

The engine is a singleton — repeated calls return the same instance.
All results are stored in the Redis/memory cache layer so the dashboard
simply reads from cache without blocking on heavy computation.
"""

import asyncio
import contextlib
import json
import logging
import threading
import time
import warnings
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from backtesting import Backtest  # noqa: E402

from lib.analysis.confluence import (  # noqa: E402
    check_confluence,
    get_recommended_timeframes,
)
from lib.analysis.regime import detect_regime_hmm, fit_detector  # noqa: E402
from lib.analysis.signal_quality import compute_signal_quality  # noqa: E402
from lib.analysis.volatility import kmeans_volatility_clusters  # noqa: E402
from lib.analysis.wave_analysis import calculate_wave_analysis  # noqa: E402
from lib.core.alerts import get_dispatcher  # noqa: E402
from lib.core.cache import (  # noqa: E402
    _cache_key,
    cache_get,
    cache_set,
    get_cached_optimization,
    get_daily,
    get_data,
    get_data_source,
    set_cached_optimization,
)
from lib.core.models import ASSETS, CONTRACT_MODE  # noqa: E402
from lib.integrations.massive_client import (  # noqa: E402
    MassiveFeedManager,
    get_massive_provider,
    is_massive_available,
)
from lib.trading.strategies import (  # noqa: E402
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    _safe_float,
    make_strategy,
    score_backtest,
    suggest_params,
)
from lib.trading.strategies.costs import slippage_commission_rate  # noqa: E402

_EST = ZoneInfo("America/New_York")

logger = logging.getLogger("engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[ENGINE] %(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)

# Strategy keys the optimizer will explore (all active strategies)
OPTIMIZER_STRATEGIES = [
    "TrendEMA",
    "RSI",
    "Breakout",
    "VWAP",
    "ORB",
    "MACD",
    "PullbackEMA",
    "EventReaction",
    "ICTTrendEMA",
    "VolumeProfile",
]

# Number of Optuna trials per strategy during optimization
TRIALS_PER_STRATEGY = 30

# Walk-forward split: train on first portion, validate on the rest
TRAIN_RATIO = 0.70
MIN_TRAIN_BARS = 150  # minimum bars needed for training split
MIN_TEST_BARS = 50  # minimum bars for validation to be meaningful

# Session hours (EST) — only trade during this window
SESSION_START_HOUR = 3  # 3 AM EST (futures pre-market)
SESSION_END_HOUR = 12  # 12 PM EST (noon — hard close)


# ---------------------------------------------------------------------------
# Session time filtering
# ---------------------------------------------------------------------------


def filter_session_hours(
    df: pd.DataFrame,
    start_hour: int = SESSION_START_HOUR,
    end_hour: int = SESSION_END_HOUR,
) -> pd.DataFrame:
    """Filter DataFrame to only include bars within the trading session.

    Default window: 3 AM – 12 PM EST, matching the morning trading playbook.
    Converts the index to US/Eastern before filtering so that the hour check
    is correct regardless of the server's system clock timezone.
    Returns the full DataFrame if the index is not datetime-based.
    """
    if df.empty:
        return df
    try:
        idx = df.index.to_series()
        # Convert to US/Eastern so hour filtering matches EST session window
        if hasattr(idx.dt, "tz") and idx.dt.tz is not None:
            # Index is already tz-aware → convert to Eastern
            est_idx = idx.dt.tz_convert(_EST)
        else:
            # Index is tz-naive → assume UTC (yfinance default) and localize
            est_idx = idx.dt.tz_localize("UTC").dt.tz_convert(_EST)
        hours = est_idx.dt.hour
        mask = (hours >= start_hour) & (hours < end_hour)
        filtered = df.loc[mask]
        return filtered if len(filtered) >= 20 else df
    except (AttributeError, TypeError):
        return df


# ---------------------------------------------------------------------------
# Volatility regime detection (HMM-based with ATR fallback)
# ---------------------------------------------------------------------------


def _detect_regime_atr(df: pd.DataFrame) -> str:
    """Fallback: classify volatility regime using short-vs-long ATR ratio.

    Returns "low_vol", "normal", or "high_vol".
    Used when HMM fitting fails (insufficient data, missing dependencies).
    """
    if df.empty or len(df) < 50:
        return "normal"
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_short = tr.iloc[-14:].mean()
    atr_long = tr.iloc[-50:].mean()
    ratio = atr_short / (atr_long + 1e-10)

    if ratio < 0.7:
        return "low_vol"
    elif ratio > 1.5:
        return "high_vol"
    return "normal"


def detect_regime(df: pd.DataFrame, ticker: str | None = None) -> dict:
    """Detect market regime using HMM + K-Means volatility clustering.

    Combines HMM state (trending/volatile/choppy) with K-Means volatility
    cluster (LOW/MEDIUM/HIGH) to produce nuanced combined regime labels
    like "HMM_TRENDING_LOW_VOL" or "HMM_CHOPPY_HIGH_VOL".

    The final position multiplier is HMM multiplier × vol cluster multiplier,
    giving smarter risk scaling that respects both market state and volatility.

    Returns a dict with:
      - regime: str (HMM label or ATR fallback)
      - combined_regime: str (e.g. "HMM_TRENDING_LOW_VOL")
      - vol_cluster: str ("LOW", "MEDIUM", "HIGH")
      - vol_percentile: float
      - vol_multiplier: float
      - adaptive_atr: float
      - probabilities: dict
      - position_multiplier: float (HMM × vol multiplier)
      - method: "hmm+kmeans", "hmm", or "atr_fallback+kmeans"
    """
    # --- K-Means volatility clustering (always computed) ---
    vol_info = kmeans_volatility_clusters(df)
    vol_cluster = vol_info.get("cluster", "MEDIUM")
    vol_mult = vol_info.get("position_multiplier", 1.0)

    # --- HMM regime detection ---
    hmm_result = None
    if ticker:
        try:
            hmm_result = detect_regime_hmm(ticker, df)
            if hmm_result.get("regime") != "choppy" or hmm_result.get("confidence", 0) > 0:
                hmm_result["method"] = "hmm"
            else:
                hmm_result = None
        except Exception as exc:
            logger.debug("HMM regime detection failed for %s: %s", ticker, exc)

    if hmm_result is not None:
        # Combine HMM state with K-Means vol cluster
        hmm_state = hmm_result.get("regime", "choppy").upper()
        combined_regime = f"HMM_{hmm_state}_{vol_cluster}_VOL"
        hmm_mult = hmm_result.get("position_multiplier", 1.0)
        final_multiplier = round(hmm_mult * vol_mult, 4)

        hmm_result["combined_regime"] = combined_regime
        hmm_result["vol_cluster"] = vol_cluster
        hmm_result["vol_percentile"] = vol_info.get("percentile", 0.5)
        hmm_result["vol_multiplier"] = vol_mult
        hmm_result["adaptive_atr"] = vol_info.get("adaptive_atr", 0.0)
        hmm_result["sl_multiplier"] = vol_info.get("sl_multiplier", 1.0)
        hmm_result["vol_strategy_hint"] = vol_info.get("strategy_hint", "NORMAL STOPS")
        hmm_result["position_multiplier"] = final_multiplier
        hmm_result["method"] = "hmm+kmeans"
        return hmm_result

    # Fallback to simple ATR-based detection + K-Means
    atr_regime = _detect_regime_atr(df)
    multiplier_map = {"low_vol": 0.5, "normal": 1.0, "high_vol": 0.5}
    atr_mult = multiplier_map.get(atr_regime, 1.0)
    combined_regime = f"ATR_{atr_regime.upper()}_{vol_cluster}_VOL"
    final_multiplier = round(atr_mult * vol_mult, 4)

    return {
        "regime": atr_regime,
        "combined_regime": combined_regime,
        "vol_cluster": vol_cluster,
        "vol_percentile": vol_info.get("percentile", 0.5),
        "vol_multiplier": vol_mult,
        "adaptive_atr": vol_info.get("adaptive_atr", 0.0),
        "sl_multiplier": vol_info.get("sl_multiplier", 1.0),
        "vol_strategy_hint": vol_info.get("strategy_hint", "NORMAL STOPS"),
        "probabilities": {},
        "confidence": 0.0,
        "confident": False,
        "position_multiplier": final_multiplier,
        "persistence": 0,
        "method": "atr_fallback+kmeans",
    }


# ---------------------------------------------------------------------------
# Optimization runner — multi-strategy with walk-forward validation
# ---------------------------------------------------------------------------


def run_optimization(ticker: str, interval: str, period: str, account_size: int) -> dict | None:
    """Run Optuna optimization across all strategy types for one asset.

    Walk-forward approach:
      1. Split data into train (70%) and test (30%)
      2. Optimize each strategy on the training set
      3. Validate the winner on the test set
      4. Combined score = 40% train + 60% test (prioritise OOS performance)
      5. Assign confidence based on train→test score degradation

    Returns the best result dict (with walk-forward metrics) or None.
    """
    cached = get_cached_optimization(ticker, interval, period)
    if cached is not None:
        return cached

    df = get_data(ticker, interval, period)
    if df.empty:
        return None

    # Compute realistic commission rate for this instrument
    from lib.core.models import TICKER_TO_NAME

    asset_name = TICKER_TO_NAME.get(ticker, "S&P")
    comm_rate = slippage_commission_rate(asset_name, CONTRACT_MODE)

    # Apply session filter for more realistic optimisation
    df_session = filter_session_hours(df)

    # Fit HMM on the full dataset (more bars = better model)
    fit_detector(ticker, df)

    # Detect regime on the session-filtered data
    regime_info = detect_regime(df_session, ticker=ticker)
    regime = regime_info["regime"]

    # Walk-forward split
    use_walk_forward = len(df_session) >= (MIN_TRAIN_BARS + MIN_TEST_BARS)
    if use_walk_forward:
        split_idx = int(len(df_session) * TRAIN_RATIO)
        df_train = df_session.iloc[:split_idx].copy()
        df_test = df_session.iloc[split_idx:].copy()
    else:
        df_train = df_session.copy()
        df_test = None

    best_score = -1e9
    best_result: dict | None = None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Filter to only strategies that are actually registered/available
    active_strategies = [s for s in OPTIMIZER_STRATEGIES if s in STRATEGY_CLASSES]

    for strat_key in active_strategies:

        def _make_objective(sk, data):
            """Closure so each strategy key and dataset is captured properly."""

            def objective(trial):
                params = suggest_params(trial, sk)
                strat_cls = make_strategy(sk, params)
                try:
                    bt = Backtest(
                        data,
                        strat_cls,
                        cash=account_size,
                        commission=comm_rate,
                        exclusive_orders=True,
                        finalize_trades=True,
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
                        stats: Any = bt.run()
                except Exception:
                    return -100.0
                return score_backtest(stats, min_trades=3)

            return objective

        study = optuna.create_study(direction="maximize")
        study.optimize(
            _make_objective(strat_key, df_train),
            n_trials=TRIALS_PER_STRATEGY,
            show_progress_bar=False,
        )

        train_score = study.best_value
        bp = dict(study.best_params)

        # Walk-forward validation on test set
        test_score = train_score  # default if no test set
        if df_test is not None and len(df_test) >= MIN_TEST_BARS:
            winning_cls = make_strategy(strat_key, bp)
            try:
                bt_test = Backtest(
                    df_test,
                    winning_cls,
                    cash=account_size,
                    commission=comm_rate,
                    exclusive_orders=True,
                    finalize_trades=True,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
                    test_stats = bt_test.run()
                test_score = score_backtest(test_stats, min_trades=1)
            except Exception:
                test_score = -100.0

        # Combined score: prioritise out-of-sample performance
        combined = 0.4 * train_score + 0.6 * test_score if use_walk_forward else train_score

        if combined > best_score:
            best_score = combined
            bp["strategy"] = strat_key
            bp["score"] = round(combined, 4)

            # Run the winning trial on FULL session data for final stats
            winning_cls = make_strategy(strat_key, bp)
            try:
                bt = Backtest(
                    df_session,
                    winning_cls,
                    cash=account_size,
                    commission=comm_rate,
                    exclusive_orders=True,
                    finalize_trades=True,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
                    stats: Any = bt.run()
                return_pct = round(float(stats["Return [%]"]), 2)
                sharpe = _safe_float(stats["Sharpe Ratio"])
                sortino = _safe_float(stats.get("Sortino Ratio", 0))
                profit_factor = _safe_float(stats.get("Profit Factor", 0))
                win_rate = _safe_float(stats["Win Rate [%]"])
                max_dd = _safe_float(stats["Max. Drawdown [%]"])
                n_trades = int(stats["# Trades"])
            except Exception:
                return_pct = 0.0
                sharpe = 0.0
                sortino = 0.0
                profit_factor = 0.0
                win_rate = 0.0
                max_dd = 0.0
                n_trades = 0

            # Confidence assessment based on train→test degradation
            if use_walk_forward and train_score > -50:
                degradation = (train_score - test_score) / (abs(train_score) + 1e-10) if train_score > 0 else 0
                if degradation < 0.2:
                    confidence = "high"
                elif degradation < 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "medium"  # no walk-forward data available

            best_result = {
                "ticker": ticker,
                "strategy": strat_key,
                "strategy_label": STRATEGY_LABELS.get(strat_key, strat_key),
                "params": {k: v for k, v in bp.items() if k not in ("strategy", "score")},
                "return_pct": return_pct,
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2),
                "profit_factor": round(profit_factor, 2),
                "win_rate": round(win_rate, 1),
                "max_dd": round(max_dd, 2),
                "n_trades": n_trades,
                "score": round(best_score, 4),
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4),
                "walk_forward": use_walk_forward,
                "confidence": confidence,
                "regime": regime,
                "regime_probabilities": regime_info.get("probabilities", {}),
                "regime_confidence": regime_info.get("confidence", 0.0),
                "regime_method": regime_info.get("method", "unknown"),
                "position_multiplier": regime_info.get("position_multiplier", 1.0),
                "updated": datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M"),
                # Legacy fields for backward compatibility with app.py
                "n1": bp.get("n1", bp.get("macd_fast", 9)),
                "n2": bp.get("n2", bp.get("macd_slow", 21)),
                "size": bp.get("trade_size", 0.10),
            }

    if best_result is not None:
        set_cached_optimization(ticker, interval, period, best_result)

    return best_result


# ---------------------------------------------------------------------------
# Backtester runner
# ---------------------------------------------------------------------------

TTL_BACKTEST = 600  # 10 minutes


def run_backtest(
    ticker: str,
    name: str,
    interval: str,
    period: str,
    account_size: int,
    use_optimized: bool = True,
    strategy_key: str | None = None,
    strategy_params: dict | None = None,
    session_filter: bool = True,
) -> dict | None:
    """Run a backtest for a single asset.  Returns stats dict or None.

    If use_optimized is True, uses cached optimization results (strategy
    type + params).  Otherwise falls back to TrendEMACross defaults.
    When session_filter is True, restricts data to trading hours (3AM-noon).
    """
    cache_key = _cache_key("bt", ticker, interval, period, str(account_size))
    cached = cache_get(cache_key)
    if cached is not None:
        try:
            return json.loads(cached.decode())
        except Exception:
            pass

    df = get_data(ticker, interval, period)
    if df.empty:
        return None

    df_bt = filter_session_hours(df) if session_filter else df.copy()
    if df_bt.empty:
        return None

    # Determine strategy class and params
    opt = get_cached_optimization(ticker, interval, period) if use_optimized else None

    if strategy_key and strategy_params:
        # Caller explicitly provided strategy
        strat_cls = make_strategy(strategy_key, strategy_params)
        params_label = "Custom"
        used_strategy = strategy_key
    elif opt and "strategy" in opt:
        # Use the optimizer's best strategy
        used_strategy = opt["strategy"]
        p = dict(opt.get("params", {}))
        strat_cls = make_strategy(used_strategy, p)
        params_label = "Optimized"
    else:
        # Fallback to TrendEMACross defaults
        used_strategy = "TrendEMA"
        fallback_params = {
            "n1": opt["n1"] if opt else 9,
            "n2": opt["n2"] if opt else 21,
            "trade_size": opt.get("size", 0.10) if opt else 0.10,
        }
        strat_cls = make_strategy(used_strategy, fallback_params)
        params_label = "Optimized" if opt else "Default"

    # Compute realistic commission rate for this instrument
    from lib.core.models import CONTRACT_MODE, TICKER_TO_NAME

    bt_asset_name = TICKER_TO_NAME.get(ticker, name)
    bt_comm_rate = slippage_commission_rate(bt_asset_name, CONTRACT_MODE)

    try:
        bt = Backtest(
            df_bt,
            strat_cls,
            cash=account_size,
            commission=bt_comm_rate,
            exclusive_orders=True,
            finalize_trades=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")
            stats: Any = bt.run()
    except Exception as exc:
        logger.warning("Backtest error for %s (%s): %s", name, used_strategy, exc)
        return None

    # Extract key params for display
    display_params = {}
    if opt and "params" in opt:
        display_params = opt["params"]
    elif opt:
        display_params = {"n1": opt.get("n1"), "n2": opt.get("n2")}

    row = {
        "Asset": name,
        "Ticker": ticker,
        "Strategy": STRATEGY_LABELS.get(used_strategy, used_strategy),
        "Strategy Key": used_strategy,
        "n1": display_params.get("n1"),
        "n2": display_params.get("n2"),
        "Size": display_params.get("trade_size", opt.get("size", 0.10) if opt else 0.10),
        "Params": params_label,
        "Return %": round(float(stats["Return [%]"]), 2),
        "Buy & Hold %": round(float(stats["Buy & Hold Return [%]"]), 2),
        "# Trades": int(stats["# Trades"]),
        "Win Rate %": round(float(stats["Win Rate [%]"]), 1) if int(stats["# Trades"]) > 0 else 0.0,
        "Max DD %": round(float(stats["Max. Drawdown [%]"]), 2),
        "Sharpe": round(_safe_float(stats["Sharpe Ratio"]), 2),
        "Sortino": round(_safe_float(stats.get("Sortino Ratio", 0)), 2),
        "Profit Factor": round(_safe_float(stats.get("Profit Factor", 0)), 2),
        "Expectancy %": round(_safe_float(stats.get("Expectancy [%]", 0)), 3),
        "Final Equity $": round(float(stats["Equity Final [$]"]), 2),
        "Confidence": opt.get("confidence", "—") if opt else "—",
        "Regime": opt.get("regime", "—") if opt else "—",
        "Walk-Forward": opt.get("walk_forward", False) if opt else False,
        "updated": datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M"),
    }

    # Add strategy-specific params for transparency
    for k, v in display_params.items():
        if k not in row:
            row[k] = v

    cache_set(cache_key, json.dumps(row).encode(), TTL_BACKTEST)
    return row


# ---------------------------------------------------------------------------
# Dashboard Engine (singleton background worker)
# ---------------------------------------------------------------------------


class DashboardEngine:
    """Daemon thread that auto-refreshes data, optimises, and backtests.

    Lifecycle:
        engine = DashboardEngine(...)
        engine.start()          # spawns daemon thread
        engine.get_status()     # read from UI
        engine.force_refresh()  # manual trigger from UI
    """

    DATA_REFRESH_INTERVAL = 60  # refresh OHLCV every minute
    OPTIMIZATION_INTERVAL = 3600  # re-optimise hourly
    BACKTEST_INTERVAL = 600  # re-backtest every 10 min

    # Minimum 1m bars needed before signal quality can be computed
    MIN_BARS_FOR_SQ = 25
    # Maximum 1m bars to keep in the rolling buffer per ticker
    MAX_BAR_BUFFER = 300

    def __init__(
        self,
        account_size: int = 150_000,
        interval: str = "5m",
        period: str = "5d",
    ):
        self.account_size = account_size
        self.interval = interval
        self.period = period

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self.status: dict = {
            "engine": "stopped",
            "data_refresh": {"last": None, "status": "idle", "error": None},
            "optimization": {
                "last": None,
                "status": "idle",
                "progress": "",
                "error": None,
            },
            "backtest": {
                "last": None,
                "status": "idle",
                "progress": "",
                "error": None,
            },
            "live_feed": {
                "status": "off",
                "connected": False,
                "bars": 0,
                "trades": 0,
                "data_source": "yfinance",
                "error": None,
            },
        }

        self.backtest_results: list[dict] = []
        # Track per-asset strategy confidence across runs
        self.strategy_history: dict[str, list[str]] = {}
        # Track previous regimes for change detection (alert dispatch)
        self._previous_regimes: dict[str, str] = {}
        # Track previous confluence scores for change detection
        self._previous_confluence: dict[str, int] = {}

        # Alert enable/disable flags (controlled from UI via session state)
        self._alerts_regime_enabled: bool = True
        self._alerts_confluence_enabled: bool = True
        self._alerts_signal_enabled: bool = True

        # Massive WebSocket live feed manager
        self._live_feed: MassiveFeedManager | None = None
        self._live_feed_enabled: bool = False

        # Rolling 1m bar buffer for WS-triggered signal quality computation.
        # Keyed by Yahoo ticker → list of bar dicts (OHLCV).
        # Filled by the _on_bar callback; trimmed to MAX_BAR_BUFFER.
        self._bar_buffer: dict[str, list[dict]] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self._thread.start()
        with self._lock:
            self.status["engine"] = "running"
            self.status["live_feed"]["data_source"] = get_data_source()
        logger.info(
            "Engine started  account=%s  interval=%s  period=%s  data_source=%s",
            self.account_size,
            self.interval,
            self.period,
            get_data_source(),
        )

        # Auto-start the live WebSocket feed if Massive API is available
        if is_massive_available():
            self.start_live_feed()

    async def stop(self) -> None:
        self._running = False
        await self.stop_live_feed()
        with self._lock:
            self.status["engine"] = "stopped"
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Engine stopped")

    # -- live feed management ------------------------------------------------

    def _compute_sq_from_buffer(self, yahoo_ticker: str) -> None:
        """Compute signal quality from the rolling 1m bar buffer and cache it.

        Called by the ``_on_bar`` callback after every confirmed 1m bar.
        Builds a lightweight DataFrame from the buffer, reuses any cached
        wave/vol analysis from the 5m engine cycle, and writes the result
        under a ``fks_sq_1m`` cache key with a short TTL.
        """
        buf = self._bar_buffer.get(yahoo_ticker)
        if not buf or len(buf) < self.MIN_BARS_FOR_SQ:
            return

        try:
            df_1m = pd.DataFrame(buf)
            # Ensure standard column names
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in df_1m.columns:
                    return

            # Reuse cached wave & vol from the 5m optimisation cycle (if available)
            wave_raw = cache_get(_cache_key("fks_wave", yahoo_ticker, self.interval, self.period))
            vol_raw = cache_get(_cache_key("fks_vol", yahoo_ticker, self.interval, self.period))
            wave_result = json.loads(wave_raw) if wave_raw else None
            vol_result = json.loads(vol_raw) if vol_raw else None

            sq_result = compute_signal_quality(
                df_1m,
                wave_result=wave_result,
                vol_result=vol_result,
            )

            # Cache under a 1m-specific key with a short TTL (90s)
            cache_set(
                _cache_key("fks_sq_1m", yahoo_ticker),
                json.dumps(sq_result).encode(),
                90,
            )

            logger.debug(
                "1m signal quality %s: %s%% (%s)",
                yahoo_ticker,
                sq_result.get("quality_pct", "?"),
                sq_result.get("market_context", "?"),
            )
        except Exception as exc:
            logger.debug("1m signal quality failed for %s: %s", yahoo_ticker, exc)

    def start_live_feed(self) -> bool:
        """Start the Massive WebSocket live feed for all tracked assets.

        Streams real-time minute bars and trades into the cache so the
        dashboard stays current without polling.  Uses broad wildcard
        subscriptions (``AM.*``, ``T.*``) for robustness — no per-ticker
        contract resolution needed for the WebSocket channel.

        Returns True on success.
        """
        if self._live_feed is not None and self._live_feed.is_running:
            logger.info("Live feed already running")
            return True

        if not is_massive_available():
            logger.info("Massive API not available — live feed skipped")
            with self._lock:
                self.status["live_feed"]["status"] = "unavailable"
                self.status["live_feed"]["error"] = "MASSIVE_API_KEY not set"
            return False

        try:
            yahoo_tickers = list(ASSETS.values())
            self._live_feed = MassiveFeedManager(
                yahoo_tickers=yahoo_tickers,
                provider=get_massive_provider(),
                subscribe_trades=True,
                subscribe_quotes=False,
                use_broad_subscriptions=False,
            )

            # Register a callback that logs, buffers, and computes 1m signal quality
            def _on_bar(yahoo_ticker: str, bar: dict) -> None:
                logger.debug(
                    "Live bar: %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%s",
                    yahoo_ticker,
                    bar.get("open", 0),
                    bar.get("high", 0),
                    bar.get("low", 0),
                    bar.get("close", 0),
                    bar.get("volume", 0),
                )

                # --- Append to rolling 1m bar buffer ---
                row = {
                    "Open": bar.get("open"),
                    "High": bar.get("high"),
                    "Low": bar.get("low"),
                    "Close": bar.get("close"),
                    "Volume": bar.get("volume", 0) or 0,
                }
                if yahoo_ticker not in self._bar_buffer:
                    self._bar_buffer[yahoo_ticker] = []
                self._bar_buffer[yahoo_ticker].append(row)
                # Trim to keep memory bounded
                if len(self._bar_buffer[yahoo_ticker]) > self.MAX_BAR_BUFFER:
                    self._bar_buffer[yahoo_ticker] = self._bar_buffer[yahoo_ticker][-self.MAX_BAR_BUFFER :]

                # --- Compute signal quality on this 1m bar ---
                self._compute_sq_from_buffer(yahoo_ticker)

            self._live_feed.on_bar(_on_bar)
            started = self._live_feed.start()

            with self._lock:
                if started:
                    self.status["live_feed"]["status"] = "running"
                    self.status["live_feed"]["connected"] = True
                    self.status["live_feed"]["error"] = None
                    self._live_feed_enabled = True
                    logger.info("Live feed started for %d assets", len(yahoo_tickers))
                else:
                    self.status["live_feed"]["status"] = "failed"
                    self.status["live_feed"]["error"] = "Could not start WebSocket"

            return started

        except Exception as exc:
            logger.error("Failed to start live feed: %s", exc)
            with self._lock:
                self.status["live_feed"]["status"] = "error"
                self.status["live_feed"]["error"] = str(exc)
            return False

    async def stop_live_feed(self) -> None:
        """Stop the Massive WebSocket live feed."""
        if self._live_feed is not None:
            await self._live_feed.stop()
            with self._lock:
                self.status["live_feed"]["status"] = "off"
                self.status["live_feed"]["connected"] = False
            self._live_feed_enabled = False
            logger.info("Live feed stopped")

    def upgrade_live_feed(self) -> None:
        """Switch the live feed to per-second aggregates (positions open)."""
        if self._live_feed is not None and self._live_feed.is_running:
            self._live_feed.upgrade_to_second_aggs()
            logger.info("Live feed upgraded to per-second aggregates")

    def downgrade_live_feed(self) -> None:
        """Switch the live feed back to per-minute aggregates."""
        if self._live_feed is not None and self._live_feed.is_running:
            self._live_feed.downgrade_to_minute_aggs()
            logger.info("Live feed downgraded to per-minute aggregates")

    def get_live_feed_status(self) -> dict:
        """Return current live feed status for the UI."""
        if self._live_feed is None:
            return {
                "status": "off",
                "connected": False,
                "data_source": get_data_source(),
            }
        feed_status = self._live_feed.get_status()
        return {
            "status": "running" if feed_status["running"] else "off",
            "connected": feed_status["connected"],
            "tickers": feed_status["tickers"],
            "bars": feed_status["bar_count"],
            "trades": feed_status["trade_count"],
            "msgs": feed_status["msg_count"],
            "uptime": feed_status["uptime_seconds"],
            "errors": feed_status["errors"],
            "data_source": get_data_source(),
        }

    def update_settings(
        self,
        account_size: int | None = None,
        interval: str | None = None,
        period: str | None = None,
    ) -> None:
        """Update engine settings (takes effect next cycle)."""
        if account_size is not None:
            self.account_size = account_size
        if interval is not None:
            self.interval = interval
        if period is not None:
            self.period = period

    def get_status(self) -> dict:
        with self._lock:
            status_copy = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.status.items()}
        # Merge in live feed stats if running
        if self._live_feed is not None:
            feed_info = self._live_feed.get_status()
            status_copy["live_feed"] = {
                "status": "running" if feed_info["running"] else "off",
                "connected": feed_info["connected"],
                "bars": feed_info["bar_count"],
                "trades": feed_info["trade_count"],
                "data_source": get_data_source(),
                "error": (feed_info["errors"][-1] if feed_info["errors"] else None),
            }
        else:
            status_copy.setdefault("live_feed", {})
            status_copy["live_feed"]["data_source"] = get_data_source()
        return status_copy

    def force_refresh(self) -> None:
        """Trigger an immediate data refresh in a separate thread."""
        # Invalidate Massive contract cache so we pick up any rolls
        try:
            provider = get_massive_provider()
            if provider.is_available:
                provider.invalidate_contract_cache()
        except Exception:
            pass
        threading.Thread(target=self._refresh_data, daemon=True).start()

    def get_strategy_history(self) -> dict[str, list[str]]:
        """Return per-asset strategy selection history (thread-safe)."""
        with self._lock:
            return dict(self.strategy_history)

    # -- alert dispatch ------------------------------------------------------

    def _dispatch_regime_alert(self, asset_name: str, old_regime: str, new_regime: str, confidence: float) -> None:
        """Fire a regime-change alert via the alert dispatcher."""
        if not self._alerts_regime_enabled:
            logger.debug("Regime alert suppressed (disabled): %s", asset_name)
            return
        try:
            dispatcher = get_dispatcher()
            if dispatcher.has_channels:
                dispatcher.send_regime_change(
                    asset=asset_name,
                    old_regime=old_regime,
                    new_regime=new_regime,
                    confidence=confidence,
                )
                logger.info(
                    "Alert dispatched: %s regime %s → %s (%.0f%%)",
                    asset_name,
                    old_regime,
                    new_regime,
                    confidence * 100,
                )
        except Exception as exc:
            logger.debug("Regime alert dispatch failed for %s: %s", asset_name, exc)

    def _check_confluence_alerts(self) -> None:
        """Check confluence across all assets and dispatch alerts on score == 3.

        Uses per-asset recommended timeframes (HTF / Setup / Entry) so the
        confluence evaluation mirrors what the Signals tab shows in the UI.
        Falls back to the engine's default interval for any timeframe that
        fails to fetch.
        """
        if not self._alerts_confluence_enabled:
            logger.debug("Confluence alerts suppressed (disabled)")
            return
        try:
            dispatcher = get_dispatcher()
            if not dispatcher.has_channels:
                return

            for name, ticker in ASSETS.items():
                try:
                    # Get recommended multi-TF intervals for this asset
                    htf_interval, setup_interval, entry_interval = get_recommended_timeframes(name)

                    # Fetch each timeframe independently; fall back to
                    # the engine's default interval on failure.
                    htf_df = self._fetch_tf_safe(ticker, htf_interval, name, "HTF")
                    setup_df = self._fetch_tf_safe(ticker, setup_interval, name, "Setup")
                    entry_df = self._fetch_tf_safe(ticker, entry_interval, name, "Entry")

                    # Need at least the entry frame to evaluate
                    if entry_df.empty or len(entry_df) < 30:
                        continue

                    # If HTF or Setup failed, fall back to entry_df
                    if htf_df.empty or len(htf_df) < 30:
                        htf_df = entry_df
                    if setup_df.empty or len(setup_df) < 30:
                        setup_df = entry_df

                    result = check_confluence(
                        htf_df=htf_df,
                        setup_df=setup_df,
                        entry_df=entry_df,
                        asset_name=name,
                    )
                    score = result.get("score", 0)
                    direction = result.get("direction", "neutral")
                    prev_score = self._previous_confluence.get(name, 0)

                    # Only alert when confluence *transitions* to 3/3
                    if score == 3 and prev_score < 3:
                        tf_label = f"HTF={htf_interval} Setup={setup_interval} Entry={entry_interval}"
                        details = result.get("summary", "")
                        details = f"{details} [{tf_label}]" if details else tf_label

                        dispatcher.send_confluence_alert(
                            asset=name,
                            score=score,
                            direction=direction,
                            details=details,
                        )
                        logger.info(
                            "Confluence alert: %s → %d/3 %s (%s)",
                            name,
                            score,
                            direction,
                            tf_label,
                        )

                    with self._lock:
                        self._previous_confluence[name] = score

                except Exception as exc:
                    logger.debug("Confluence check failed for %s: %s", name, exc)
        except Exception as exc:
            logger.debug("Confluence alert sweep failed: %s", exc)

    def _fetch_tf_safe(self, ticker: str, interval: str, name: str, label: str) -> pd.DataFrame:
        """Fetch data for a specific timeframe, returning empty DF on failure.

        Uses a sensible period for each interval type:
          - 1m  → 5d  (Yahoo max for 1m)
          - 5m  → 5d
          - 15m → 15d
          - 1h  → 1mo
        Falls back to the engine's configured interval/period on error.
        """
        _interval_to_period = {
            "1m": "5d",
            "2m": "5d",
            "5m": "5d",
            "15m": "15d",
            "30m": "1mo",
            "60m": "3mo",
            "1h": "3mo",
        }
        period = _interval_to_period.get(interval, self.period)
        try:
            df = get_data(ticker, interval, period)
            if not df.empty:
                return df
        except Exception as exc:
            logger.debug(
                "Multi-TF fetch failed for %s %s (%s): %s",
                name,
                label,
                interval,
                exc,
            )

        # Fallback: use engine's default interval
        try:
            return get_data(ticker, self.interval, self.period)
        except Exception:
            return pd.DataFrame()

    # -- main loop -----------------------------------------------------------

    def _loop(self) -> None:
        last_data = 0.0
        last_opt = 0.0
        last_bt = 0.0
        last_confluence = 0.0

        # EOD action tracking — store the calendar date on which each action
        # last fired so we fire at most once per trading day.
        _eod_warning_fired_date: date | None = None
        _eod_close_fired_date: date | None = None

        # Initial data load immediately
        self._refresh_data()

        while self._running:
            now = time.time()

            if now - last_data >= self.DATA_REFRESH_INTERVAL:
                self._refresh_data()
                last_data = now

            if now - last_opt >= self.OPTIMIZATION_INTERVAL:
                self._run_optimizations()
                last_opt = now

            if now - last_bt >= self.BACKTEST_INTERVAL:
                self._run_backtests()
                last_bt = now

            # Check confluence for alerts every 2 minutes
            if now - last_confluence >= 120:
                self._check_confluence_alerts()
                last_confluence = now

            # ------------------------------------------------------------------
            # EOD safety actions — time-of-day checks in ET (DST-aware)
            # ------------------------------------------------------------------
            _now_et = datetime.now(tz=_EST)
            _today = _now_et.date()
            _hm = (_now_et.hour, _now_et.minute)

            # 15:45 ET — position-close warning (fires once per calendar day,
            # in the window 15:45–15:59 so a restart at 15:50 still fires).
            if _hm >= (15, 45) and _hm < (16, 0) and _eod_warning_fired_date != _today:
                _eod_warning_fired_date = _today
                try:
                    self._eod_warning()
                except Exception as _exc:
                    logger.error("EOD warning raised: %s", _exc)

            # 16:00 ET — hard cancel + flatten (fires once per calendar day,
            # catch-up window extends to 16:14 so a restart shortly after
            # 16:00 still triggers the close).
            if _hm >= (16, 0) and _hm < (16, 15) and _eod_close_fired_date != _today:
                _eod_close_fired_date = _today
                try:
                    self._eod_close_positions()
                except Exception as _exc:
                    logger.error("EOD close raised: %s", _exc)

            # Sleep in short increments so stop() is responsive
            for _ in range(10):
                if not self._running:
                    break
                time.sleep(1)

    # -- EOD safety actions --------------------------------------------------

    def _eod_warning(self) -> None:
        """Fire the 15:45 ET position-close warning alert.

        Dispatched once per calendar day via the alert channel (Discord)
        and logged at WARNING level so it surfaces in
        any log aggregator.  Does NOT touch orders or positions.
        """
        try:
            dispatcher = get_dispatcher()
            msg = (
                "⚠️  *15-minute warning*: all open positions should be manually "
                "flattened before 16:00 ET.  The automated EOD close will fire at "
                "16:00 and will market-flatten any remaining positions across all "
                "connected Rithmic accounts."
            )
            logger.warning("EOD WARNING (15:45 ET): automated close fires in 15 minutes")
            if dispatcher.has_channels:
                dispatcher.send_risk_alert(
                    title="EOD Position Close — 15-min Warning",
                    message=msg,
                    extra_fields={"Scheduled hard-close": "16:00 ET"},
                )
        except Exception as exc:
            logger.debug("EOD warning dispatch failed: %s", exc)

    def _eod_close_positions(self) -> None:
        """Cancel all working orders then flatten all positions at market (16:00 ET).

        Runs the async ``RithmicAccountManager.eod_close_all_positions()`` from
        inside the synchronous engine thread by spinning up a fresh event loop.
        If ``async-rithmic`` is not installed, or no accounts are configured,
        the method is a safe no-op.

        A risk alert is dispatched afterwards summarising which accounts were
        closed and whether any errors occurred.
        """
        logger.warning("EOD HARD CLOSE (16:00 ET): cancelling all orders and flattening all positions")

        results: list[dict[str, Any]] = []
        try:
            from lib.integrations.rithmic_client import get_manager as _get_rithmic_manager

            mgr = _get_rithmic_manager()
            # Run the async coroutine in a fresh event loop on this engine thread.
            loop = asyncio.new_event_loop()
            try:
                results = loop.run_until_complete(mgr.eod_close_all_positions(dry_run=False))
            finally:
                loop.close()
        except ImportError:
            logger.debug("EOD close skipped — rithmic_client not importable")
            return
        except Exception as exc:
            logger.error("EOD close encountered an unexpected error: %s", exc)

        # Build a summary for the alert channels
        ok_accounts = [r for r in results if not r.get("error") and not r.get("skipped")]
        err_accounts = [r for r in results if r.get("error")]
        skipped = [r for r in results if r.get("skipped")]

        summary_lines = []
        for r in ok_accounts:
            label = r.get("label", r.get("key", "?"))
            summary_lines.append(f"✅ {label}: cancelled={r.get('cancelled')} flattened={r.get('flattened')}")
        for r in err_accounts:
            label = r.get("label", r.get("key", "?"))
            summary_lines.append(f"❌ {label}: {r.get('error')}")
        for r in skipped:
            label = r.get("label", r.get("key", "?"))
            summary_lines.append(f"⏭  {label}: {r.get('skipped')}")

        summary = "\n".join(summary_lines) if summary_lines else "No Rithmic accounts configured."
        any_error = bool(err_accounts)

        title = "EOD Position Close — " + ("ERRORS DETECTED ⚠️" if any_error else "Complete ✅")
        try:
            dispatcher = get_dispatcher()
            if dispatcher.has_channels:
                dispatcher.send_risk_alert(
                    title=title,
                    message=summary,
                    extra_fields={"Time": datetime.now(tz=_EST).strftime("%H:%M:%S ET")},
                )
        except Exception as exc:
            logger.debug("EOD close alert dispatch failed: %s", exc)

        logger.info(
            "EOD close complete: %d ok  %d errors  %d skipped",
            len(ok_accounts),
            len(err_accounts),
            len(skipped),
        )

    # -- data refresh --------------------------------------------------------

    def _refresh_data(self) -> None:
        with self._lock:
            self.status["data_refresh"]["status"] = "running"
            self.status["data_refresh"]["error"] = None
        errors: list[str] = []
        for name, ticker in ASSETS.items():
            try:
                # Primary interval (5m)
                get_data(ticker, self.interval, self.period)
                get_daily(ticker)

                # 1-minute data — the live heartbeat
                try:
                    get_data(ticker, "1m", "1d")
                except Exception as exc_1m:
                    logger.debug("1m fetch failed for %s: %s", name, exc_1m)

                # Multi-TF prefetch for confluence (HTF / Setup / Entry)
                htf_iv, setup_iv, entry_iv = get_recommended_timeframes(name)
                for iv, label in [
                    (htf_iv, "HTF"),
                    (setup_iv, "Setup"),
                    (entry_iv, "Entry"),
                ]:
                    if iv != self.interval and iv != "1m":
                        with contextlib.suppress(Exception):
                            self._fetch_tf_safe(ticker, iv, name, label)
            except Exception as exc:
                msg = f"{name}: {exc}"
                errors.append(msg)
                logger.warning("Data refresh failed for %s: %s", name, exc)
        with self._lock:
            self.status["data_refresh"]["last"] = datetime.now(tz=_EST).strftime("%H:%M:%S")
            self.status["data_refresh"]["status"] = "idle"
            self.status["data_refresh"]["error"] = "; ".join(errors) if errors else None
        if not errors:
            logger.info(
                "Data refresh complete for %d assets (1m + %s + multi-TF)",
                len(ASSETS),
                self.interval,
            )

    # -- optimization --------------------------------------------------------

    def _run_optimizations(self) -> None:
        with self._lock:
            self.status["optimization"]["status"] = "running"
            self.status["optimization"]["error"] = None
        errors: list[str] = []
        total = len(ASSETS)

        # Refit HMM detectors per instrument at the start of each optimization cycle
        for name, ticker in ASSETS.items():
            try:
                df = get_data(ticker, self.interval, self.period)
                if not df.empty:
                    fit_detector(ticker, df)
            except Exception as exc:
                logger.debug("HMM refit failed for %s: %s", name, exc)

        # --- Ruby Wave & Volatility Analysis (cached per asset) ---
        for name, ticker in ASSETS.items():
            try:
                df = get_data(ticker, self.interval, self.period)
                if df.empty:
                    continue
                df_session = filter_session_hours(df)
                if df_session.empty:
                    df_session = df

                # Wave analysis
                wave_result = calculate_wave_analysis(df_session, asset_name=name)
                cache_set(
                    _cache_key("fks_wave", ticker, self.interval, self.period),
                    json.dumps(wave_result).encode(),
                    3600,
                )

                # K-Means volatility clustering
                vol_result = kmeans_volatility_clusters(df_session)
                cache_set(
                    _cache_key("fks_vol", ticker, self.interval, self.period),
                    json.dumps(vol_result).encode(),
                    3600,
                )

                # Signal quality score (port of Pine's signal_quality_score)
                sq_result = compute_signal_quality(
                    df_session,
                    wave_result=wave_result,
                    vol_result=vol_result,
                )
                cache_set(
                    _cache_key("fks_sq", ticker, self.interval, self.period),
                    json.dumps(sq_result).encode(),
                    3600,
                )

                logger.info(
                    "Ruby analysis %s: wave=%s (ratio=%s), vol=%s (%s%%), quality=%s%%",
                    name,
                    wave_result.get("bias", "?"),
                    wave_result.get("wave_ratio_text", "?"),
                    vol_result.get("cluster", "?"),
                    round(vol_result.get("percentile", 0) * 100),
                    sq_result.get("quality_pct", "?"),
                )
            except Exception as exc:
                logger.debug("Ruby analysis failed for %s: %s", name, exc)

        for i, (name, ticker) in enumerate(ASSETS.items()):
            with self._lock:
                self.status["optimization"]["progress"] = f"{name} ({i + 1}/{total})"
            try:
                result = run_optimization(ticker, self.interval, self.period, self.account_size)
                if result:
                    strat_label = result.get("strategy_label", "?")
                    confidence = result.get("confidence", "?")
                    regime = result.get("regime", "?")
                    # Enrich optimization result with Ruby data
                    fks_wave_raw = cache_get(_cache_key("fks_wave", ticker, self.interval, self.period))
                    fks_vol_raw = cache_get(_cache_key("fks_vol", ticker, self.interval, self.period))
                    if fks_wave_raw:
                        fks_wave = json.loads(fks_wave_raw)
                        result["wave_bias"] = fks_wave.get("bias", "NEUTRAL")
                        result["wave_ratio"] = fks_wave.get("wave_ratio", 1.0)
                        result["wave_dominance"] = fks_wave.get("dominance", 0.0)
                        result["trend_speed"] = fks_wave.get("trend_speed", 0.0)
                        result["market_phase"] = fks_wave.get("market_phase", "?")
                    if fks_vol_raw:
                        fks_vol = json.loads(fks_vol_raw)
                        result["vol_cluster"] = fks_vol.get("cluster", "MEDIUM")
                        result["vol_percentile"] = fks_vol.get("percentile", 0.5)
                        # Apply vol cluster multiplier to position sizing
                        vol_mult = fks_vol.get("position_multiplier", 1.0)
                        result["position_multiplier"] = round(result.get("position_multiplier", 1.0) * vol_mult, 4)
                    # Signal quality enrichment
                    fks_sq_raw = cache_get(_cache_key("fks_sq", ticker, self.interval, self.period))
                    if fks_sq_raw:
                        fks_sq = json.loads(fks_sq_raw)
                        result["signal_quality"] = fks_sq.get("score", 0.0)
                        result["signal_quality_pct"] = fks_sq.get("quality_pct", 0.0)
                        result["high_quality_signal"] = fks_sq.get("high_quality", False)
                        result["signal_market_context"] = fks_sq.get("market_context", "?")
                    # Re-cache with enriched data
                    set_cached_optimization(ticker, self.interval, self.period, result)

                    logger.info(
                        "Optimized %s → %s (sharpe=%.2f, return=%.2f%%, "
                        "confidence=%s, regime=%s, wave=%s, vol=%s, quality=%s%%)",
                        name,
                        strat_label,
                        result.get("sharpe", 0),
                        result.get("return_pct", 0),
                        confidence,
                        regime,
                        result.get("wave_bias", "?"),
                        result.get("vol_cluster", "?"),
                        result.get("signal_quality_pct", "?"),
                    )
                    # --- Engine-triggered regime-change alerts ---
                    with self._lock:
                        prev_regime = self._previous_regimes.get(name)
                    if prev_regime is not None and prev_regime != regime:
                        regime_conf = result.get("regime_confidence", 0.0)
                        self._dispatch_regime_alert(name, prev_regime, regime, regime_conf)
                    with self._lock:
                        self._previous_regimes[name] = regime
                    # Track strategy selection history
                    with self._lock:
                        history = self.strategy_history.setdefault(name, [])
                        history.append(result["strategy"])
                        # Keep last 10 selections
                        if len(history) > 10:
                            self.strategy_history[name] = history[-10:]
            except Exception as exc:
                msg = f"{name}: {exc}"
                errors.append(msg)
                logger.warning("Optimization failed for %s: %s", name, exc)
        with self._lock:
            self.status["optimization"]["last"] = datetime.now(tz=_EST).strftime("%H:%M:%S")
            self.status["optimization"]["status"] = "idle"
            self.status["optimization"]["progress"] = ""
            self.status["optimization"]["error"] = "; ".join(errors) if errors else None
        if not errors:
            logger.info("Optimization complete for %d assets", total)

    # -- backtesting ---------------------------------------------------------

    def _run_backtests(self) -> None:
        with self._lock:
            self.status["backtest"]["status"] = "running"
            self.status["backtest"]["error"] = None
        results: list[dict] = []
        errors: list[str] = []
        total = len(ASSETS)
        for i, (name, ticker) in enumerate(ASSETS.items()):
            with self._lock:
                self.status["backtest"]["progress"] = f"{name} ({i + 1}/{total})"
            try:
                row = run_backtest(ticker, name, self.interval, self.period, self.account_size)
                if row is not None:
                    results.append(row)
            except Exception as exc:
                msg = f"{name}: {exc}"
                errors.append(msg)
                logger.warning("Backtest failed for %s: %s", name, exc)
        with self._lock:
            self.backtest_results = results
            self.status["backtest"]["last"] = datetime.now(tz=_EST).strftime("%H:%M:%S")
            self.status["backtest"]["status"] = "idle"
            self.status["backtest"]["progress"] = ""
            self.status["backtest"]["error"] = "; ".join(errors) if errors else None
        if not errors:
            logger.info("Backtest complete for %d assets (%d results)", total, len(results))

    def get_backtest_results(self) -> list[dict]:
        """Return latest backtest results (thread-safe)."""
        with self._lock:
            return list(self.backtest_results)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_engine_instance: DashboardEngine | None = None
_engine_lock = threading.Lock()


def get_engine(
    account_size: int = 150_000,
    interval: str = "5m",
    period: str = "5d",
) -> DashboardEngine:
    """Return (and auto-start) the singleton engine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = DashboardEngine(account_size, interval, period)
            _engine_instance.start()
        else:
            _engine_instance.update_settings(account_size, interval, period)
        return _engine_instance
