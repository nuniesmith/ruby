"""
lib.analysis.ml — Machine learning sub-package.

Re-exports the public API from the breakout CNN module so callers can use either:

    from lib.analysis.ml import HybridBreakoutCNN, predict_breakout
    from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN, predict_breakout
"""

# breakout_cnn — hybrid CNN + tabular model for breakout prediction
try:
    from lib.analysis.ml.breakout_cnn import (
        ASSET_VOLATILITY_CLASS,
        DEFAULT_THRESHOLD,
        FEATURE_CONTRACT_VERSION,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        NUM_TABULAR,
        TABULAR_FEATURES,
        BreakoutDataset,
        HybridBreakoutCNN,
        TrainResult,
        evaluate_model,
        generate_feature_contract,
        get_asset_class_id,
        get_asset_volatility_class,
        get_atr_trend,
        get_breakout_type_category,
        get_breakout_type_ordinal,
        get_daily_bias_confidence,
        get_daily_bias_direction,
        get_device,
        get_inference_transform,
        get_monthly_trend_score,
        get_prior_day_pattern,
        get_session_ordinal,
        get_session_overlap_flag,
        get_session_threshold,
        get_training_transform,
        get_type_embedding_weights,
        get_volume_trend,
        get_weekly_range_position,
        invalidate_model_cache,
        model_info,
        predict_breakout,
        predict_breakout_batch,
        train_model,
    )
except ImportError:
    # torch or other heavy deps not installed — provide None stubs so that
    # environments without PyTorch can still import the package.
    ASSET_VOLATILITY_CLASS = None  # type: ignore[assignment]
    DEFAULT_THRESHOLD = 0.82  # type: ignore[assignment]
    FEATURE_CONTRACT_VERSION = None  # type: ignore[assignment]
    IMAGE_SIZE = None  # type: ignore[assignment]
    IMAGENET_MEAN = None  # type: ignore[assignment]
    IMAGENET_STD = None  # type: ignore[assignment]
    NUM_TABULAR = None  # type: ignore[assignment]
    TABULAR_FEATURES = None  # type: ignore[assignment]
    BreakoutDataset = None  # type: ignore[assignment,misc]
    HybridBreakoutCNN = None  # type: ignore[assignment,misc]
    TrainResult = None  # type: ignore[assignment,misc]
    evaluate_model = None  # type: ignore[assignment,misc]
    generate_feature_contract = None  # type: ignore[assignment,misc]
    get_asset_class_id = None  # type: ignore[assignment,misc]
    get_asset_volatility_class = None  # type: ignore[assignment,misc]
    get_atr_trend = None  # type: ignore[assignment,misc]
    get_breakout_type_category = None  # type: ignore[assignment,misc]
    get_breakout_type_ordinal = None  # type: ignore[assignment,misc]
    get_daily_bias_confidence = None  # type: ignore[assignment,misc]
    get_daily_bias_direction = None  # type: ignore[assignment,misc]
    get_device = None  # type: ignore[assignment,misc]
    get_inference_transform = None  # type: ignore[assignment,misc]
    get_monthly_trend_score = None  # type: ignore[assignment,misc]
    get_prior_day_pattern = None  # type: ignore[assignment,misc]
    get_session_ordinal = None  # type: ignore[assignment,misc]
    get_session_overlap_flag = None  # type: ignore[assignment,misc]
    get_session_threshold = None  # type: ignore[assignment,misc]
    get_training_transform = None  # type: ignore[assignment,misc]
    get_type_embedding_weights = None  # type: ignore[assignment,misc]
    get_volume_trend = None  # type: ignore[assignment,misc]
    get_weekly_range_position = None  # type: ignore[assignment,misc]
    invalidate_model_cache = None  # type: ignore[assignment,misc]
    model_info = None  # type: ignore[assignment,misc]
    predict_breakout = None  # type: ignore[assignment,misc]
    predict_breakout_batch = None  # type: ignore[assignment,misc]
    train_model = None  # type: ignore[assignment,misc]

__all__ = [
    # constants
    "ASSET_VOLATILITY_CLASS",
    "DEFAULT_THRESHOLD",
    "FEATURE_CONTRACT_VERSION",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_TABULAR",
    "TABULAR_FEATURES",
    # classes
    "BreakoutDataset",
    "HybridBreakoutCNN",
    "TrainResult",
    # training / evaluation
    "evaluate_model",
    "train_model",
    # feature helpers
    "generate_feature_contract",
    "get_asset_class_id",
    "get_asset_volatility_class",
    "get_atr_trend",
    "get_breakout_type_category",
    "get_breakout_type_ordinal",
    "get_daily_bias_confidence",
    "get_daily_bias_direction",
    "get_device",
    "get_inference_transform",
    "get_monthly_trend_score",
    "get_prior_day_pattern",
    "get_session_ordinal",
    "get_session_overlap_flag",
    "get_session_threshold",
    "get_training_transform",
    "get_type_embedding_weights",
    "get_volume_trend",
    "get_weekly_range_position",
    # model lifecycle
    "invalidate_model_cache",
    "model_info",
    # inference
    "predict_breakout",
    "predict_breakout_batch",
]
