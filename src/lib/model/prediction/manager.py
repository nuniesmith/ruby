"""
Prediction Model Manager

This module handles the creation, loading, and management of
prediction models for different assets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lib.model._shims import logger

# --- Stubs for modules not present in this project ---
AssetDataManager = None  # stub: data.manager.AssetDataManager  # type: ignore[assignment]

if TYPE_CHECKING:
    from lib.model.ml.gaussian import GaussianModel as GaussianPredictor
    from lib.model.ml.polynomial import PolynomialRegression as PolynomialPredictor
    from lib.model.statistical.bayesian import BayesianLinearRegression as BayesianPredictor
else:
    try:
        from lib.model.statistical.bayesian import BayesianLinearRegression as BayesianPredictor
    except ImportError:
        BayesianPredictor = None  # type: ignore[assignment,misc]

    try:
        from lib.model.ml.gaussian import GaussianModel as GaussianPredictor
    except ImportError:
        GaussianPredictor = None  # type: ignore[assignment,misc]

    try:
        from lib.model.ml.polynomial import PolynomialRegression as PolynomialPredictor
    except ImportError:
        PolynomialPredictor = None  # type: ignore[assignment,misc]

get_config = None  # stub: core.constants.manager.get_config


class ModelManager:
    """
    Manages prediction models for multiple assets, including
    creation, initialization, and persistence.
    """

    def __init__(self, data_fetcher: AssetDataManager | None = None, model_dir: str | None = None):  # type: ignore[valid-type]
        """
        Initialize the model manager.

        Args:
            data_fetcher: Data fetching component for retrieving market data
            model_dir: Directory to store/load trained models
        """
        self.data_fetcher = data_fetcher
        self.model_dir = model_dir or "models"
        self.models: dict[str, Any] = {}
        self.model_types: dict[str, str] = {}
        self.constants = get_config() if get_config is not None else None  # type: ignore[call-non-callable]

    def _initialize_models(self) -> None:
        """Initialize prediction models for each asset."""
        logger.info("Initializing prediction models")

        if self.constants is None:
            logger.warning("No constants/config available — skipping model initialization")
            return

        for asset in self.constants.SUPPORTED_ASSETS:
            model_type = self.model_types.get(asset, "bayesian")
            self.models[asset] = self._create_model(asset, model_type)

    def _create_model(
        self, asset: str, model_type: str
    ) -> BayesianPredictor | GaussianPredictor | PolynomialPredictor | None:  # type: ignore[valid-type]
        """
        Create a predictor model of the specified type.

        Args:
            asset: Asset the model will predict
            model_type: Type of prediction model to create

        Returns:
            Instantiated predictor model or None if the type is unknown / not available
        """
        if model_type == "bayesian":
            if BayesianPredictor is None:
                logger.warning("BayesianPredictor not available — falling back to None")
                return None
            return BayesianPredictor(asset=asset)  # type: ignore[call-arg]
        if model_type == "gaussian":
            if GaussianPredictor is None:
                logger.warning("GaussianPredictor not available — falling back to None")
                return None
            return GaussianPredictor(asset=asset)  # type: ignore[call-arg]
        if model_type == "polynomial":
            if PolynomialPredictor is None:
                logger.warning("PolynomialPredictor not available — falling back to None")
                return None
            return PolynomialPredictor(asset=asset)  # type: ignore[call-arg]

        logger.warning("Unknown model type %r for asset %s — returning None", model_type, asset)
        return None

    def get_model(self, asset: str) -> BayesianPredictor | GaussianPredictor | PolynomialPredictor | None:  # type: ignore[valid-type]
        """
        Get the prediction model for a specific asset.

        Args:
            asset: Asset identifier

        Returns:
            The predictor model for the asset, or None if not available
        """
        return self.models.get(asset)

    def load_models(self, model_dir: str | None = None) -> dict[str, bool]:
        """
        Load all trained models from disk.

        Args:
            model_dir: Directory to load models from (overrides instance default)

        Returns:
            Dict mapping asset name → load success bool
        """
        import pathlib

        load_dir = pathlib.Path(model_dir or self.model_dir)
        results: dict[str, bool] = {}

        if self.constants is None:
            logger.warning("No constants/config — skipping model load")
            return results

        for asset in self.constants.SUPPORTED_ASSETS:
            model_path = load_dir / f"{asset}_model.pkl"
            if model_path.exists():
                try:
                    import pickle

                    with open(model_path, "rb") as f:
                        self.models[asset] = pickle.load(f)  # noqa: S301
                    logger.info("Loaded model for %s from %s", asset, model_path)
                    results[asset] = True
                except Exception as exc:
                    logger.error("Error loading model for %s: %s", asset, exc)
                    results[asset] = False
            else:
                logger.warning("No model file found for %s at %s", asset, model_path)
                results[asset] = False

        return results

    def save_models(self, model_dir: str | None = None) -> dict[str, bool]:
        """
        Persist all trained models to disk.

        Args:
            model_dir: Directory to save models (overrides instance default)

        Returns:
            Dict mapping asset name → save success bool
        """
        import pathlib
        import pickle

        save_dir = pathlib.Path(model_dir or self.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, bool] = {}

        for asset, model in self.models.items():
            if model is None:
                results[asset] = False
                continue
            model_path = save_dir / f"{asset}_model.pkl"
            try:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info("Saved model for %s to %s", asset, model_path)
                results[asset] = True
            except Exception as exc:
                logger.error("Error saving model for %s: %s", asset, exc)
                results[asset] = False

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Return a summary of all managed models."""
        return {
            "model_dir": self.model_dir,
            "model_count": len(self.models),
            "assets": list(self.models.keys()),
            "model_types": self.model_types,
            "available": {
                "bayesian": BayesianPredictor is not None,
                "gaussian": GaussianPredictor is not None,
                "polynomial": PolynomialPredictor is not None,
            },
        }

    def __repr__(self) -> str:
        return (
            f"ModelManager(model_dir={self.model_dir!r}, "
            f"models={list(self.models.keys())}, "
            f"model_types={self.model_types})"
        )


def get_model_manager(
    data_fetcher: Any = None,
    model_dir: str | None = None,
) -> ModelManager:
    """Convenience factory — returns a ready-to-use ModelManager instance."""
    manager = ModelManager(data_fetcher=data_fetcher, model_dir=model_dir)
    return manager


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_manager: ModelManager | None = None


def get_default_manager() -> ModelManager:
    """Return the process-wide ModelManager singleton, creating it on first call."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
