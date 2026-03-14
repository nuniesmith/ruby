"""
Strategies Package
==================
Trading strategy modules for Ruby Futures.

Sub-packages:
  - daily/   — Daily bias analysis, trade plan generation, swing detection
  - rb/      — Range Breakout scalping system (all 13 breakout types)

The ``rb`` package is the canonical public façade for the Range Breakout
system.  It re-exports the most commonly used symbols from the core type
definitions, the detection layer, range builders, and the publisher pipeline::

    from lib.trading.strategies.rb import (
        # Core types
        BreakoutType,
        RangeConfig,
        get_range_config,
        # Detection
        detect_range_breakout,
        detect_breakout_for_type,
        detect_all_breakout_types,
        detect_breakouts_filtered,
        BreakoutResult,
        # Range builders (pure, no side-effects)
        compute_atr,
        localize_bars,
        build_orb_range,
        build_pdr_range,
        build_ib_range,
        build_consolidation_range,
        build_weekly_range,
        build_monthly_range,
        build_asian_range,
        build_bbsqueeze_range,
        build_va_range,
        build_inside_day_range,
        build_gap_rejection_range,
        build_pivot_range,
        build_fibonacci_range,
        # Publisher / orchestration
        publish_breakout_result,
        persist_breakout_result,
        dispatch_to_position_manager,
        send_breakout_alert,
        publish_pipeline,
        # Handler pipeline
        handle_breakout_check,
    )

The ``daily`` package provides higher-timeframe analysis that feeds into
the CNN feature vector (v7+) and the dashboard daily plan view::

    from lib.trading.strategies.daily import (
        compute_daily_bias,
        DailyBias,
        BiasDirection,
    )

Phase 1G additions (RB System Refactor):
  - ``rb.range_builders`` — all 13 ``build_*_range()`` functions extracted
    from ``engine/breakout.py``, plus canonical ``compute_atr()`` and
    ``localize_bars()`` shared helpers.
  - ``rb.detector`` — unified detection façade with ``detect_breakout_for_type()``
    and ``detect_breakouts_filtered()`` convenience wrappers.
  - ``rb.publisher`` — Redis pub + alerting with ``publish_pipeline()``
    convenience and ``get_alert_template()`` for alert formatting.

Phase 4B additions (CNN Sub-Feature Decomposition):
  - ``breakout_type_category`` — coarse grouping: time=0, range=0.5, squeeze=1.0
  - ``session_overlap_flag`` — 1.0 if London+NY overlap window
  - ``atr_trend`` — ATR expanding=1.0, contracting=0.0 (10-bar lookback)
  - ``volume_trend`` — 5-bar volume slope normalised [0, 1]
  These sub-features are computed in ``lib.analysis.breakout_cnn`` and wired
  into ``dataset_generator._build_row()`` and ``feature_contract.json`` v7.1.
"""

from lib.trading.strategies.strategy_defs import (
    STRATEGY_CLASSES,
    STRATEGY_LABELS,
    BreakoutStrategy,
    EventReaction,
    ICTTrendEMA,
    MACDMomentum,
    ORBStrategy,
    PlainEMACross,
    PullbackEMA,
    RSIReversal,
    TrendEMACross,
    VWAPReversion,
    _atr,
    _compute_ict_confluence,
    _ema,
    _ict_confluence_array,
    _is_bearish_engulfing,
    _is_bullish_engulfing,
    _is_hammer,
    _is_nan,
    _is_shooting_star,
    _macd_histogram,
    _macd_line,
    _macd_signal,
    _passthrough,
    _rolling_max,
    _rolling_min,
    _rsi,
    _safe_float,
    _sma,
    make_strategy,
    score_backtest,
    suggest_params,
)

__all__ = [
    # Dicts
    "STRATEGY_CLASSES",
    "STRATEGY_LABELS",
    # Strategy classes
    "TrendEMACross",
    "RSIReversal",
    "BreakoutStrategy",
    "VWAPReversion",
    "ORBStrategy",
    "MACDMomentum",
    "PullbackEMA",
    "EventReaction",
    "PlainEMACross",
    "ICTTrendEMA",
    # Public helpers
    "make_strategy",
    "score_backtest",
    "suggest_params",
    "_safe_float",
    # Indicator helpers (used by tests and backtesting)
    "_ema",
    "_sma",
    "_atr",
    "_rsi",
    "_macd_line",
    "_macd_signal",
    "_macd_histogram",
    "_rolling_max",
    "_rolling_min",
    "_passthrough",
    # ICT helpers
    "_compute_ict_confluence",
    "_ict_confluence_array",
    # Candle pattern helpers
    "_is_bullish_engulfing",
    "_is_bearish_engulfing",
    "_is_hammer",
    "_is_shooting_star",
    "_is_nan",
]
