"""
lib — Shared library for Ruby Futures.

All business logic modules live under organised sub-packages:

    # Core infrastructure
    from lib.core.cache import cache_get, cache_set
    from lib.core.models import init_db, ASSETS
    from lib.core.alerts import get_dispatcher
    from lib.core.logging_config import setup_logging, get_logger
    from lib.core.breakout_types import BreakoutType, get_range_config
    from lib.core.multi_session import get_session, all_sessions, RBSession, ORBSession

    # Analysis modules
    from lib.analysis.volatility import kmeans_volatility_clusters
    from lib.analysis.wave_analysis import calculate_wave_analysis
    from lib.analysis.ict import detect_fvgs, detect_order_blocks
    from lib.analysis.cvd import compute_cvd
    from lib.analysis.confluence import check_confluence
    from lib.analysis.regime import RegimeDetector
    from lib.analysis.scorer import PreMarketScorer
    from lib.analysis.signal_quality import compute_signal_quality
    from lib.analysis.volume_profile import compute_volume_profile
    from lib.analysis.rendering.chart_renderer import render_ruby_snapshot
    from lib.analysis.rendering.chart_renderer_parity import render_parity_snapshot

    # Trading modules
    from lib.trading.engine import get_engine, DashboardEngine

    # External integrations
    from lib.integrations.grok_helper import GrokSession
    from lib.integrations.massive_client import get_massive_provider

Services (engine, data, web, training) are sub-packages:

    from lib.services.engine.focus import compute_daily_focus
    from lib.services.data.main import app
    from lib.services.web.main import app
    from lib.services.training.trainer_server import app
    from lib.services.training.dataset_generator import generate_dataset, DatasetConfig
    from lib.services.training.rb_simulator import simulate_batch, BracketConfig

Install in editable mode for development:

    pip install -e .          # CPU-only (engine, web, dashboard)
    pip install -e ".[gpu]"   # + PyTorch/CUDA for training
"""
