"""
lib.core — Core infrastructure modules.

Re-exports the public API from each sub-module so callers can do:

    from lib.core import cache_get, cache_set, ASSETS, init_db
    from lib.core import BreakoutType, get_range_config, RBSession, get_session

.. note::

   ``RBSession`` is the preferred name (Phase 1G rename).
   ``ORBSession`` is kept as a backward-compatible alias.
"""

from lib.core.alerts import AlertDispatcher, get_dispatcher, send_risk_alert
from lib.core.breakout_types import (
    DETECTED_BREAKOUT_TYPES,
    EXCHANGE_BREAKOUT_TYPES,
    HTF_BREAKOUT_TYPES,
    RESEARCHED_BREAKOUT_TYPES,
    BreakoutType,
    RangeConfig,
    all_range_configs,
    breakout_type_from_name,
    breakout_type_from_ord,
    breakout_type_ord,
    get_range_config,
    types_with_ema_trailing,
    types_with_tp3,
)
from lib.core.cache import (
    REDIS_AVAILABLE,
    TTL_DAILY,
    TTL_INTRADAY,
    _cache_key,
    _df_to_bytes,
    cache_get,
    cache_set,
    clear_cached_optimization,
    flush_all,
    get_cached_indicator,
    get_cached_optimization,
    get_daily,
    get_data,
    get_data_source,
    set_cached_indicator,
    set_cached_optimization,
)
from lib.core.logging_config import get_logger, setup_logging
from lib.core.models import (
    ACCOUNT_PROFILES,
    ACTIVE_TICKERS,
    ACTIVE_WATCHLIST,
    ASSETS,
    CONTRACT_MODE,
    CONTRACT_SPECS,
    CORE_TICKERS,
    CORE_WATCHLIST,
    DB_PATH,
    EXTENDED_TICKERS,
    EXTENDED_WATCHLIST,
    STATUS_CLOSED,
    STATUS_OPEN,
    TICKER_TO_NAME,
    cancel_trade,
    close_trade,
    create_trade,
    get_all_trades,
    get_closed_trades,
    get_daily_journal,
    get_journal_stats,
    get_max_contracts_for_profile,
    get_open_trades,
    get_today_pnl,
    init_db,
    save_daily_journal,
)
from lib.core.multi_session import (
    ALL_SESSION_KEYS,
    SESSION_BY_KEY,
    ORBSession,
    RBSession,
    all_sessions,
    daytime_sessions,
    get_session,
    overnight_sessions,
    session_keys,
    sessions_for_asset_class,
)

__all__ = [
    # breakout_types
    "BreakoutType",
    "DETECTED_BREAKOUT_TYPES",
    "EXCHANGE_BREAKOUT_TYPES",
    "HTF_BREAKOUT_TYPES",
    "RESEARCHED_BREAKOUT_TYPES",
    "RangeConfig",
    "all_range_configs",
    "breakout_type_from_name",
    "breakout_type_from_ord",
    "breakout_type_ord",
    "get_range_config",
    "types_with_ema_trailing",
    "types_with_tp3",
    # multi_session
    "ALL_SESSION_KEYS",
    "SESSION_BY_KEY",
    "ORBSession",
    "RBSession",
    "all_sessions",
    "daytime_sessions",
    "get_session",
    "overnight_sessions",
    "session_keys",
    "sessions_for_asset_class",
    # alerts
    "AlertDispatcher",
    "get_dispatcher",
    "send_risk_alert",
    # cache
    "REDIS_AVAILABLE",
    "TTL_DAILY",
    "TTL_INTRADAY",
    "_cache_key",
    "_df_to_bytes",
    "cache_get",
    "cache_set",
    "clear_cached_optimization",
    "flush_all",
    "get_cached_indicator",
    "get_cached_optimization",
    "get_daily",
    "get_data",
    "get_data_source",
    "set_cached_indicator",
    "set_cached_optimization",
    # logging
    "get_logger",
    "setup_logging",
    # models
    "ACCOUNT_PROFILES",
    "ACTIVE_TICKERS",
    "ACTIVE_WATCHLIST",
    "ASSETS",
    "CONTRACT_MODE",
    "CONTRACT_SPECS",
    "CORE_TICKERS",
    "CORE_WATCHLIST",
    "DB_PATH",
    "EXTENDED_TICKERS",
    "EXTENDED_WATCHLIST",
    "STATUS_CLOSED",
    "STATUS_OPEN",
    "TICKER_TO_NAME",
    "cancel_trade",
    "close_trade",
    "create_trade",
    "get_all_trades",
    "get_closed_trades",
    "get_daily_journal",
    "get_journal_stats",
    "get_max_contracts_for_profile",
    "get_open_trades",
    "get_today_pnl",
    "init_db",
    "save_daily_journal",
]
