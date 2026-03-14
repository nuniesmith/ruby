"""
lib.services.engine — Engine service shared modules.

Core business-logic modules for the engine worker:

    from lib.services.engine.focus import compute_daily_focus, should_not_trade
    from lib.services.engine.scheduler import ScheduleManager, SessionMode
    from lib.services.engine.risk import RiskManager
    from lib.services.engine.patterns import evaluate_no_trade
    from lib.services.engine.position_manager import PositionManager
    from lib.services.engine.copy_trader import CopyTrader, get_copy_trader
    from lib.services.engine.backfill import run_backfill, get_backfill_status
    from lib.services.engine.model_watcher import ModelWatcher
    from lib.services.engine import focus, scheduler, risk, patterns, position_manager, copy_trader, backfill, model_watcher
"""
