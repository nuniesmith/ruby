"""
HMM-based regime detection for futures trading.

Uses a 3-state GaussianHMM (hmmlearn) to classify market conditions into:
  - trending   (state 0): low-vol directional moves, full position sizing
  - volatile   (state 1): high-vol trending, reduced sizing
  - choppy     (state 2): range-bound/mean-reverting, minimal sizing

Design decisions (per docs/todo.md blueprint):
  - Multi-feature: log returns, normalized ATR, volume ratio
  - StandardScaler preprocessing (features must be stationary)
  - Forward algorithm for filtered probabilities (NO Viterbi — avoids look-ahead)
  - Confidence threshold (default 0.6) before acting on regime signal
  - Persistence filter: regime must hold for N consecutive bars before switching
  - Per-instrument fitting (GC, CL, ES, NQ have different volatility structures)
  - Retrain at session open; update probabilities bar-by-bar during session

Usage:
    from lib.regime import RegimeDetector
    detector = RegimeDetector()
    detector.fit(df)                    # fit on historical OHLCV
    info = detector.detect(df)          # get current regime + probabilities
    mult = info["position_multiplier"]  # scale position size by this
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("regime")

# Regime labels indexed by state
REGIME_LABELS = {0: "trending", 1: "volatile", 2: "choppy"}

# Position sizing multipliers per regime
REGIME_MULTIPLIERS = {
    "trending": 1.0,  # full size — favorable conditions
    "volatile": 0.5,  # half size — high vol, wider stops needed
    "choppy": 0.25,  # quarter size — low edge, preserve capital
}

# Default config
DEFAULT_N_STATES = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_PERSISTENCE_BARS = 3
DEFAULT_LOOKBACK_DAYS = 30  # rolling window for training (in calendar days)
MIN_BARS_FOR_FIT = 200  # minimum bars needed to fit HMM


def _compute_features(df: pd.DataFrame) -> np.ndarray | None:
    """Compute the 3-feature matrix from OHLCV data.

    Features (all stationary):
      1. Log returns: log(close_t / close_{t-1})
      2. Normalized ATR: ATR_14 / close (volatility relative to price level)
      3. Volume ratio: volume / 20-bar rolling mean volume

    Returns an (N, 3) ndarray or None if insufficient data.
    """
    if df.empty or len(df) < 50:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # 1. Log returns
    log_ret = np.log(close / close.shift(1))

    # 2. Normalized ATR (14-period)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    norm_atr = atr14 / (close + 1e-10)

    # 3. Volume ratio
    vol_ma20 = volume.rolling(20).mean()
    vol_ratio = volume / (vol_ma20 + 1e-10)

    features = pd.DataFrame(
        {
            "log_ret": log_ret,
            "norm_atr": norm_atr,
            "vol_ratio": vol_ratio,
        }
    ).dropna()

    if len(features) < MIN_BARS_FOR_FIT:
        return None

    return features.values


def _label_states(model, features: np.ndarray) -> dict[int, str]:
    """Map HMM state indices to regime labels based on learned means.

    The state with the lowest norm_atr mean → "trending"
    The state with the highest norm_atr mean → "volatile"
    The remaining state → "choppy"
    """
    means = model.means_  # shape (n_states, n_features)
    # norm_atr is feature index 1
    atr_means = means[:, 1]
    sorted_states = np.argsort(atr_means)

    mapping = {}
    mapping[int(sorted_states[0])] = "trending"
    mapping[int(sorted_states[-1])] = "volatile"
    for s in sorted_states[1:-1]:
        mapping[int(s)] = "choppy"

    return mapping


class RegimeDetector:
    """Per-instrument HMM regime detector.

    Fits a GaussianHMM on historical OHLCV data and provides real-time
    regime classification using forward-algorithm filtered probabilities.
    """

    def __init__(
        self,
        n_states: int = DEFAULT_N_STATES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        persistence_bars: int = DEFAULT_PERSISTENCE_BARS,
        n_seeds: int = 5,
    ):
        self.n_states = n_states
        self.confidence_threshold = confidence_threshold
        self.persistence_bars = persistence_bars
        self.n_seeds = n_seeds

        self.model: Any = None
        self.scaler: Any = None
        self.state_mapping: dict[int, str] = {}
        self.is_fitted = False
        self._persistence_count = 0
        self._last_regime = "choppy"  # conservative default

    def fit(self, df: pd.DataFrame) -> bool:
        """Fit the HMM on historical OHLCV data.

        Tries multiple random seeds and selects the model with best BIC.
        Returns True if fitting succeeded, False otherwise.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("hmmlearn or scikit-learn not installed. Install with: pip install hmmlearn scikit-learn")
            return False

        features = _compute_features(df)
        if features is None:
            logger.info("Insufficient data for HMM fit (%d bars)", len(df))
            return False

        # StandardScaler — features must be zero-mean, unit-variance
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features)

        # Try multiple seeds, pick best BIC
        best_model = None
        best_score = -np.inf

        for seed in range(self.n_seeds):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = GaussianHMM(
                        n_components=self.n_states,
                        covariance_type="diag",
                        n_iter=100,
                        random_state=seed * 42,
                        verbose=False,
                    )
                    model.fit(X)

                if not model.monitor_.converged:
                    continue

                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as exc:
                logger.debug("HMM fit failed with seed %d: %s", seed, exc)
                continue

        if best_model is None:
            logger.warning("HMM fit failed across all seeds")
            return False

        self.model = best_model
        self.state_mapping = _label_states(best_model, X)
        self.is_fitted = True
        self._persistence_count = 0
        self._last_regime = "choppy"

        logger.info(
            "HMM fit complete: %d bars, mapping=%s, log-likelihood=%.1f",
            len(X),
            self.state_mapping,
            best_score,
        )
        return True

    def predict_proba(self, df: pd.DataFrame) -> dict[str, float] | None:
        """Get filtered (forward-algorithm) regime probabilities for the latest bar.

        Returns a dict like {"trending": 0.75, "volatile": 0.15, "choppy": 0.10}
        or None if prediction is not possible.
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            return None

        features = _compute_features(df)
        if features is None:
            return None

        try:
            X = self.scaler.transform(features)
            # Forward algorithm: predict_proba gives filtered probabilities
            # (causal, no look-ahead — unlike Viterbi)
            proba = self.model.predict_proba(X)
            latest = proba[-1]  # probabilities at the most recent bar
        except Exception as exc:
            logger.debug("HMM predict_proba failed: %s", exc)
            return None

        result: dict[str, float] = {}
        for state_idx, prob in enumerate(latest):
            label = self.state_mapping.get(state_idx, "choppy")
            result[label] = result.get(label, 0.0) + float(prob)

        return result

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        """Full regime detection with confidence and persistence filtering.

        Returns a dict with:
          - regime: str ("trending", "volatile", "choppy")
          - probabilities: dict of regime → probability
          - confidence: float (probability of the detected regime)
          - confident: bool (whether confidence exceeds threshold)
          - position_multiplier: float (sizing multiplier for this regime)
          - persistence: int (consecutive bars in current regime)
        """
        default = {
            "regime": "choppy",
            "probabilities": {"trending": 0.0, "volatile": 0.0, "choppy": 1.0},
            "confidence": 0.0,
            "confident": False,
            "position_multiplier": REGIME_MULTIPLIERS["choppy"],
            "persistence": 0,
        }

        proba = self.predict_proba(df)
        if proba is None:
            return default

        # Determine the dominant regime
        raw_regime = max(proba, key=proba.get)  # type: ignore[arg-type]
        raw_confidence = proba[raw_regime]

        # Persistence filter: only switch regime after N consecutive bars
        if raw_regime == self._last_regime:
            self._persistence_count += 1
        else:
            self._persistence_count = 1

        effective_regime = raw_regime if self._persistence_count >= self.persistence_bars else self._last_regime

        self._last_regime = raw_regime

        confident = raw_confidence >= self.confidence_threshold

        # Position multiplier: blend by probability if confident,
        # else use conservative choppy multiplier
        if confident:
            multiplier = sum(proba.get(r, 0.0) * REGIME_MULTIPLIERS[r] for r in REGIME_MULTIPLIERS)
        else:
            multiplier = REGIME_MULTIPLIERS["choppy"]

        return {
            "regime": effective_regime,
            "probabilities": {k: round(v, 4) for k, v in proba.items()},
            "confidence": round(raw_confidence, 4),
            "confident": confident,
            "position_multiplier": round(multiplier, 4),
            "persistence": self._persistence_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize current state for caching / display."""
        return {
            "is_fitted": self.is_fitted,
            "n_states": self.n_states,
            "state_mapping": self.state_mapping,
            "confidence_threshold": self.confidence_threshold,
            "persistence_bars": self.persistence_bars,
            "last_regime": self._last_regime,
            "persistence_count": self._persistence_count,
        }


# ---------------------------------------------------------------------------
# Module-level registry: one detector per instrument ticker
# ---------------------------------------------------------------------------

_detectors: dict[str, RegimeDetector] = {}


def get_detector(ticker: str) -> RegimeDetector:
    """Return (or create) the RegimeDetector for a given instrument."""
    if ticker not in _detectors:
        _detectors[ticker] = RegimeDetector()
    return _detectors[ticker]


def fit_detector(ticker: str, df: pd.DataFrame) -> bool:
    """Fit (or refit) the HMM for a given instrument on OHLCV data."""
    detector = get_detector(ticker)
    return detector.fit(df)


def detect_regime_hmm(ticker: str, df: pd.DataFrame) -> dict[str, Any]:
    """Get the current HMM regime for a given instrument.

    If the HMM hasn't been fitted yet, attempts to fit first.
    Falls back to a conservative default if fitting fails.
    """
    detector = get_detector(ticker)
    if not detector.is_fitted:
        detector.fit(df)
    return detector.detect(df)
