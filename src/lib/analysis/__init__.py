"""
lib.analysis — Market analysis and signal modules.

Re-exports the public API from each sub-module so callers can do:

    from lib.analysis import compute_cvd, detect_fvgs, kmeans_volatility_clusters
    from lib.analysis import apply_all_filters, predict_breakout
"""

from lib.analysis.breakout_filters import (
    BreakoutFilterResult,
    FilterVerdict,
    ORBFilterResult,  # backward-compat alias
    apply_all_filters,
    check_lunch_filter,
    check_mtf_analyzer,
    check_multi_tf_bias,
    check_nr7,
    check_premarket_range,
    check_session_window,
    check_vwap_confluence,
    compute_session_vwap,
    extract_premarket_range,
)
from lib.analysis.confluence import (
    MultiTimeframeFilter,
    check_confluence,
    get_recommended_timeframes,
)
from lib.analysis.cvd import (
    compute_cvd,
    cvd_summary,
    detect_absorption_candles,
    detect_cvd_divergences,
)
from lib.analysis.ict import (
    detect_breaker_blocks,
    detect_fvgs,
    detect_order_blocks,
    ict_summary,
)
from lib.analysis.regime import (
    RegimeDetector,
    detect_regime_hmm,
    fit_detector,
)
from lib.analysis.scorer import (
    EVENT_CATALOG,
    PreMarketScorer,
    results_to_dataframe,
    score_instruments,
)
from lib.analysis.signal_quality import compute_signal_quality
from lib.analysis.volatility import kmeans_volatility_clusters
from lib.analysis.volume_profile import (
    compute_session_profiles,
    compute_volume_profile,
    find_naked_pocs,
    format_profile_summary,
    profile_to_dataframe,
)
from lib.analysis.wave_analysis import calculate_wave_analysis

# Optional import — chart renderers require mplfinance / Pillow which may
# not be installed in all environments (e.g. slim web container).
try:
    from lib.analysis.rendering.chart_renderer import (
        RenderConfig,
        cleanup_inference_images,
        render_batch_snapshots,
        render_ruby_snapshot,
        render_snapshot_for_inference,
    )
except ImportError:
    render_ruby_snapshot = None  # type: ignore[assignment,misc]
    render_batch_snapshots = None  # type: ignore[assignment,misc]
    render_snapshot_for_inference = None  # type: ignore[assignment,misc]
    cleanup_inference_images = None  # type: ignore[assignment,misc]
    RenderConfig = None  # type: ignore[assignment,misc]

try:
    from lib.analysis.rendering.chart_renderer_parity import (
        ParityBar,
        compute_vwap_from_bars,
        dataframe_to_parity_bars,
        render_parity_batch,
        render_parity_snapshot,
        render_parity_to_file,
        render_parity_to_temp,
    )
except ImportError:
    render_parity_snapshot = None  # type: ignore[assignment,misc]
    render_parity_to_file = None  # type: ignore[assignment,misc]
    render_parity_to_temp = None  # type: ignore[assignment,misc]
    render_parity_batch = None  # type: ignore[assignment,misc]
    dataframe_to_parity_bars = None  # type: ignore[assignment,misc]
    compute_vwap_from_bars = None  # type: ignore[assignment,misc]
    ParityBar = None  # type: ignore[assignment,misc]

# Optional import — sentiment modules require optional deps (vaderSentiment,
# praw) that may not be installed in all environments.
try:
    from lib.analysis.sentiment.news_sentiment import (
        REDIS_KEY_TEMPLATE,
        REDIS_SPIKE_KEY,
        REDIS_TTL_SECONDS,
        NewsSentiment,
        ScoredArticle,
        cache_sentiments,
        compute_all_news_sentiments,
        compute_news_sentiment,
        get_cached_sentiments,
        load_sentiment_from_cache,
        run_news_sentiment_pipeline,
    )
except ImportError:
    NewsSentiment = None  # type: ignore[assignment,misc]
    ScoredArticle = None  # type: ignore[assignment,misc]
    REDIS_KEY_TEMPLATE = None  # type: ignore[assignment]
    REDIS_SPIKE_KEY = None  # type: ignore[assignment]
    REDIS_TTL_SECONDS = None  # type: ignore[assignment]
    cache_sentiments = None  # type: ignore[assignment,misc]
    compute_all_news_sentiments = None  # type: ignore[assignment,misc]
    compute_news_sentiment = None  # type: ignore[assignment,misc]
    get_cached_sentiments = None  # type: ignore[assignment,misc]
    load_sentiment_from_cache = None  # type: ignore[assignment,misc]
    run_news_sentiment_pipeline = None  # type: ignore[assignment,misc]

try:
    from lib.analysis.sentiment.reddit_sentiment import (
        WINDOWS_MINUTES,
        RedditSignal,
        RedditSnapshot,
        aggregate_asset,
        get_asset_signal,
        get_full_snapshot,
    )
except ImportError:
    WINDOWS_MINUTES = None  # type: ignore[assignment]
    RedditSignal = None  # type: ignore[assignment,misc]
    RedditSnapshot = None  # type: ignore[assignment,misc]
    aggregate_asset = None  # type: ignore[assignment,misc]
    get_asset_signal = None  # type: ignore[assignment,misc]
    get_full_snapshot = None  # type: ignore[assignment,misc]

# Optional import — breakout_cnn requires torch which may not be installed
# in all environments.  Only inference functions are re-exported here;
# training lives in the lib.training package.
_breakout_cnn_default_threshold: float = 0.82

try:
    from lib.analysis.ml.breakout_cnn import DEFAULT_THRESHOLD as _cnn_threshold
    from lib.analysis.ml.breakout_cnn import (
        BreakoutDataset,
        HybridBreakoutCNN,
        predict_breakout,
        predict_breakout_batch,
    )

    _breakout_cnn_default_threshold = _cnn_threshold
except ImportError:
    BreakoutDataset = None  # type: ignore[assignment,misc]
    HybridBreakoutCNN = None  # type: ignore[assignment,misc]
    predict_breakout = None  # type: ignore[assignment,misc]
    predict_breakout_batch = None  # type: ignore[assignment,misc]

DEFAULT_THRESHOLD: float = _breakout_cnn_default_threshold

__all__ = [
    # chart_renderer (optional)
    "RenderConfig",
    "render_ruby_snapshot",
    "render_batch_snapshots",
    "render_snapshot_for_inference",
    "cleanup_inference_images",
    # chart_renderer_parity (optional)
    "ParityBar",
    "render_parity_snapshot",
    "render_parity_to_file",
    "render_parity_to_temp",
    "render_parity_batch",
    "dataframe_to_parity_bars",
    "compute_vwap_from_bars",
    # confluence
    "MultiTimeframeFilter",
    "check_confluence",
    "get_recommended_timeframes",
    # cvd
    "compute_cvd",
    "cvd_summary",
    "detect_absorption_candles",
    "detect_cvd_divergences",
    # ict
    "detect_breaker_blocks",
    "detect_fvgs",
    "detect_order_blocks",
    "ict_summary",
    # breakout_filters (new canonical name)
    "BreakoutFilterResult",
    "FilterVerdict",
    "apply_all_filters",
    "check_lunch_filter",
    "check_mtf_analyzer",
    "check_multi_tf_bias",
    "check_nr7",
    "check_premarket_range",
    "check_session_window",
    "check_vwap_confluence",
    "compute_session_vwap",
    "extract_premarket_range",
    # orb_filters backward-compat alias
    "ORBFilterResult",
    # regime
    "RegimeDetector",
    "detect_regime_hmm",
    "fit_detector",
    # scorer
    "EVENT_CATALOG",
    "PreMarketScorer",
    "results_to_dataframe",
    "score_instruments",
    # signal_quality
    "compute_signal_quality",
    # volatility
    "kmeans_volatility_clusters",
    # volume_profile
    "compute_session_profiles",
    "compute_volume_profile",
    "find_naked_pocs",
    "format_profile_summary",
    "profile_to_dataframe",
    # wave_analysis
    "calculate_wave_analysis",
    # news_sentiment (optional)
    "NewsSentiment",
    "ScoredArticle",
    "REDIS_KEY_TEMPLATE",
    "REDIS_SPIKE_KEY",
    "REDIS_TTL_SECONDS",
    "cache_sentiments",
    "compute_all_news_sentiments",
    "compute_news_sentiment",
    "get_cached_sentiments",
    "load_sentiment_from_cache",
    "run_news_sentiment_pipeline",
    # reddit_sentiment (optional)
    "WINDOWS_MINUTES",
    "RedditSignal",
    "RedditSnapshot",
    "aggregate_asset",
    "get_asset_signal",
    "get_full_snapshot",
    # breakout_cnn (optional — inference + model classes)
    "BreakoutDataset",
    "HybridBreakoutCNN",
    "predict_breakout",
    "predict_breakout_batch",
    "DEFAULT_THRESHOLD",
]
