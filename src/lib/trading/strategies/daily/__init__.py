"""Daily strategy layer — bias analysis, trade plans, and swing detection.

Public API:
    from lib.trading.strategies.daily import (
        # Bias analyzer (Phase 2A)
        compute_daily_bias,
        compute_all_daily_biases,
        rank_assets_by_conviction,
        DailyBias,
        BiasDirection,
        CandlePattern,
        KeyLevels,

        # Daily plan generator (Phase 2B)
        generate_daily_plan,
        generate_and_publish_daily_plan,
        select_daily_focus_assets,
        DailyPlan,
        SwingCandidate,
        ScalpFocusAsset,

        # Swing detector (Phase 2C)
        detect_swing_entries,
        detect_pullback_entry,
        detect_breakout_entry,
        detect_gap_continuation,
        evaluate_swing_exits,
        scan_swing_entries_all_assets,
        enrich_swing_candidates,
        create_swing_state,
        update_swing_state,
        SwingSignal,
        SwingExitSignal,
        SwingState,
        SwingEntryStyle,
        SwingExitReason,
        SwingPhase,
    )
"""

from lib.trading.strategies.daily.bias_analyzer import (
    BiasDirection,
    CandlePattern,
    DailyBias,
    KeyLevels,
    compute_all_daily_biases,
    compute_daily_bias,
    rank_assets_by_conviction,
)
from lib.trading.strategies.daily.daily_plan import (
    DailyPlan,
    ScalpFocusAsset,
    SwingCandidate,
    generate_and_publish_daily_plan,
    generate_daily_plan,
    select_daily_focus_assets,
)
from lib.trading.strategies.daily.swing_detector import (
    SwingEntryStyle,
    SwingExitReason,
    SwingExitSignal,
    SwingPhase,
    SwingSignal,
    SwingState,
    create_swing_state,
    detect_breakout_entry,
    detect_gap_continuation,
    detect_pullback_entry,
    detect_swing_entries,
    enrich_swing_candidates,
    evaluate_swing_exits,
    scan_swing_entries_all_assets,
    update_swing_state,
)

__all__ = [
    # Bias analyzer (Phase 2A)
    "BiasDirection",
    "CandlePattern",
    "DailyBias",
    "KeyLevels",
    "compute_all_daily_biases",
    "compute_daily_bias",
    "rank_assets_by_conviction",
    # Daily plan generator (Phase 2B)
    "DailyPlan",
    "ScalpFocusAsset",
    "SwingCandidate",
    "generate_and_publish_daily_plan",
    "generate_daily_plan",
    "select_daily_focus_assets",
    # Swing detector (Phase 2C)
    "SwingEntryStyle",
    "SwingExitReason",
    "SwingExitSignal",
    "SwingPhase",
    "SwingSignal",
    "SwingState",
    "create_swing_state",
    "detect_breakout_entry",
    "detect_gap_continuation",
    "detect_pullback_entry",
    "detect_swing_entries",
    "enrich_swing_candidates",
    "evaluate_swing_exits",
    "scan_swing_entries_all_assets",
    "update_swing_state",
]
