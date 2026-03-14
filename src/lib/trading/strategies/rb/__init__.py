"""
Range Breakout (RB) Scalping System
====================================
Unified detection, publishing, and alerting for all 13 range-breakout types.

This package is the canonical home for the RB strategy layer.  It re-exports
the most commonly used symbols so callers can write::

    from lib.trading.strategies.rb import (
        BreakoutType,
        RangeConfig,
        get_range_config,
        detect_range_breakout,
        detect_breakout_for_type,
        detect_all_breakout_types,
        BreakoutResult,
        handle_breakout_check,
        compute_atr,
    )

Sub-modules:
    detector        — unified ``detect_range_breakout()`` facade
    range_builders  — per-type range construction functions (pure, no side-effects)
    publisher       — Redis pub + alerting helpers (orchestration, no detection logic)

The heavy detection logic currently lives in ``lib.services.engine.breakout``
and will migrate here incrementally (Phase 1C/1G).  For now this package acts
as a clean public façade that delegates to the engine layer.

Phase 1G additions:
  - ``range_builders`` module with all 13 ``build_*_range()`` functions +
    canonical ``compute_atr()`` and ``localize_bars()`` helpers.
  - ``detector`` module with ``detect_breakout_for_type()`` and
    ``detect_breakouts_filtered()`` convenience wrappers.
  - ``publisher`` module with ``publish_pipeline()`` convenience and
    ``get_alert_template()`` for alert formatting.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports from core (canonical type definitions)
# ---------------------------------------------------------------------------
from lib.core.breakout_types import (
    BreakoutType,
    RangeConfig,
    all_range_configs,
    breakout_type_from_name,
    breakout_type_from_ord,
    breakout_type_ord,
    get_range_config,
)

# ---------------------------------------------------------------------------
# Re-exports from handlers (orchestration pipeline)
# ---------------------------------------------------------------------------
from lib.services.engine.handlers import (
    fetch_bars_1m,
    get_assets_for_session_key,
    get_htf_bars,
    handle_breakout_check,
    handle_breakout_multi,
    handle_orb_check,
    run_mtf_on_result,
)

# ---------------------------------------------------------------------------
# Re-exports from engine (detection + result — kept for backward compat)
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.breakout import (
    DEFAULT_CONFIGS,
)

# ---------------------------------------------------------------------------
# Re-exports from detector (unified detection facade)
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.detector import (
    BreakoutResult,
    detect_all_breakout_types,
    detect_breakout_for_type,
    detect_breakouts_filtered,
    detect_consolidation_breakout,
    detect_ib_breakout,
    detect_pdr_breakout,
    detect_range_breakout,
)

# ---------------------------------------------------------------------------
# Re-exports from publisher (Redis pub + alerting)
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.publisher import (
    dispatch_to_position_manager,
    get_alert_template,
    persist_breakout_result,
    publish_breakout_result,
    publish_pipeline,
    send_breakout_alert,
)

# ---------------------------------------------------------------------------
# Re-exports from range_builders (pure range construction + shared helpers)
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.range_builders import (
    build_asian_range,
    build_bbsqueeze_range,
    build_consolidation_range,
    build_fibonacci_range,
    build_gap_rejection_range,
    build_ib_range,
    build_inside_day_range,
    build_monthly_range,
    build_orb_range,
    build_pdr_range,
    build_pivot_range,
    build_va_range,
    build_weekly_range,
    check_bar_quality,
    compute_atr,
    get_range_builder,
    localize_bars,
)

__all__ = [
    # Core types
    "BreakoutType",
    "RangeConfig",
    "get_range_config",
    "all_range_configs",
    "breakout_type_ord",
    "breakout_type_from_ord",
    "breakout_type_from_name",
    # Range builders (pure functions)
    "compute_atr",
    "localize_bars",
    "check_bar_quality",
    "get_range_builder",
    "build_orb_range",
    "build_pdr_range",
    "build_ib_range",
    "build_consolidation_range",
    "build_weekly_range",
    "build_monthly_range",
    "build_asian_range",
    "build_bbsqueeze_range",
    "build_va_range",
    "build_inside_day_range",
    "build_gap_rejection_range",
    "build_pivot_range",
    "build_fibonacci_range",
    # Detection facade
    "BreakoutResult",
    "DEFAULT_CONFIGS",
    "detect_range_breakout",
    "detect_breakout_for_type",
    "detect_all_breakout_types",
    "detect_breakouts_filtered",
    "detect_pdr_breakout",
    "detect_ib_breakout",
    "detect_consolidation_breakout",
    # Publisher / orchestration
    "get_alert_template",
    "publish_breakout_result",
    "persist_breakout_result",
    "dispatch_to_position_manager",
    "send_breakout_alert",
    "publish_pipeline",
    # Handler pipeline (from engine)
    "handle_breakout_check",
    "handle_breakout_multi",
    "handle_orb_check",
    "get_assets_for_session_key",
    "fetch_bars_1m",
    "get_htf_bars",
    "run_mtf_on_result",
]
