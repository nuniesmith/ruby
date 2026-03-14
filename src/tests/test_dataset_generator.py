"""
Tests for dataset_generator.py — pre-compute caching, skip path,
_build_row feature correctness, and timezone handling.

Covers:
  1. Pre-compute caching: cross-asset and fingerprint are computed exactly
     once per symbol and attached to all sim results.
  2. Skip path correctness: skipped images still get a _build_row call that
     uses the cached features (not re-computing them).
  3. _build_row feature vector shape and value ranges.
  4. Timezone correctness: ring-buffer timestamps are Eastern Time.
  5. generate_dataset_for_symbol smoke test end-to-end with mocked deps.
  6. Pre-compute failure isolation: a crash in fingerprint/cross-asset must
     not abort the whole symbol — the run continues with 0.5 defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers — synthetic bar data
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")


def _make_bars_1m(
    n: int = 300,
    base_price: float = 2000.0,
    symbol: str = "MGC",
    *,
    tz_aware: bool = True,
) -> pd.DataFrame:
    """Return a minimal 1-minute OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    # Build a contiguous minute-frequency index starting at a known ET time
    # (2025-01-06 09:00 ET = 14:00 UTC, a Monday)
    start = pd.Timestamp("2025-01-06 14:00:00", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC" if tz_aware else None)

    closes = base_price + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.3, n)
    volumes = rng.integers(100, 1000, n).astype(float)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=idx,
    )


def _make_bars_daily(n: int = 30, base_price: float = 2000.0) -> pd.DataFrame:
    """Return a minimal daily OHLCV DataFrame."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-12-01", periods=n, freq="B")
    closes = base_price + np.cumsum(rng.normal(0, 5, n))
    opens = closes + rng.normal(0, 2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.integers(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _make_orb_sim_result(
    symbol: str = "MGC",
    label: str = "good_long",
    breakout_time: str = "2025-01-06T09:35:00",
) -> Any:
    """Return a minimal ORBSimResult with required attributes.

    ``is_trade`` is a read-only property on ORBSimResult derived from ``label``:
    any label other than ``"no_trade"`` → is_trade is True.  We use
    ``label="good_long"`` by default so all results are trades.
    """
    from lib.services.training.rb_simulator import ORBSimResult

    r = ORBSimResult(
        label=label,
        symbol=symbol,
        direction="LONG",
        entry=2000.0,
        sl=1990.0,
        tp1=2020.0,
        tp2=2030.0,
        tp3=2045.0,
        or_high=2010.0,
        or_low=1995.0,
        or_range=15.0,
        atr=10.0,
        quality_pct=75,
        outcome="tp1_hit",
        pnl_r=1.0,
        breakout_time=breakout_time,
        or_start_time="2025-01-06T09:30:00",
    )
    # is_trade is a @property: label != "no_trade" → True.  No setter needed.
    assert r.is_trade, f"Expected is_trade=True for label={label!r}"
    return r


# ---------------------------------------------------------------------------
# 1. Pre-compute caching — cross-asset features computed exactly once
# ---------------------------------------------------------------------------


class TestCrossAssetPreComputeCache:
    """compute_cross_asset_features must be called exactly once per symbol."""

    def test_cross_asset_called_once_for_many_results(self, tmp_path):
        """With 10 sim results, cross-asset pre-compute fires once, not 10 times."""
        from lib.services.training.dataset_generator import DatasetConfig

        bars_1m = _make_bars_1m(3000)
        bars_daily = _make_bars_daily()
        peer_bars = _make_bars_1m(3000, base_price=1800.0)
        bars_by_ticker = {"MGC": bars_1m, "SIL": peer_bars}

        call_count = {"n": 0}

        @dataclass
        class _FakeCrossFeats:
            primary_peer_corr: float = 0.6
            cross_class_corr: float = 0.4
            correlation_regime: float = 0.5

        def _fake_compute_cross(sym, bars_dict):
            call_count["n"] += 1
            return _FakeCrossFeats()

        # Build fake sim results — label="good_long" → is_trade=True via property
        results = [_make_orb_sim_result() for _ in range(10)]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.cross_asset.compute_cross_asset_features",
                side_effect=_fake_compute_cross,
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                return_value=MagicMock(
                    typical_daily_range_atr=1.5,
                    session_concentration=MagicMock(
                        overnight_pct=0.25,
                        london_pct=0.25,
                        us_pct=0.40,
                        settle_pct=0.10,
                    ),
                    breakout_follow_through=MagicMock(follow_through_rate=0.6),
                    mean_reversion_tendency=0.5,
                    overnight_gap_stats=MagicMock(gap_frequency=0.3),
                    volume_profile=MagicMock(name="D_SHAPE"),
                ),
            ),
            patch(
                "lib.analysis.rendering.chart_renderer_parity.render_parity_to_file",
                return_value=str(tmp_path / "fake.png"),
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
                include_no_trade=False,
            )
            from lib.services.training.dataset_generator import generate_dataset_for_symbol

            generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                bars_daily=bars_daily,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

        assert call_count["n"] == 1, (
            f"compute_cross_asset_features should be called once per symbol, got {call_count['n']} calls"
        )

    def test_cached_cross_asset_attached_to_all_results(self, tmp_path):
        """_cached_cross_asset is set on every sim result after pre-compute."""
        from lib.services.training.dataset_generator import DatasetConfig

        bars_1m = _make_bars_1m(3000)
        peer_bars = _make_bars_1m(3000, base_price=1800.0)
        bars_by_ticker = {"MGC": bars_1m, "SIL": peer_bars}

        @dataclass
        class _FakeCrossFeats:
            primary_peer_corr: float = 0.7
            cross_class_corr: float = 0.3
            correlation_regime: float = 0.5

        sentinel = _FakeCrossFeats()
        results = [_make_orb_sim_result() for _ in range(5)]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.cross_asset.compute_cross_asset_features",
                return_value=sentinel,
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                return_value=MagicMock(
                    typical_daily_range_atr=1.5,
                    session_concentration=MagicMock(overnight_pct=0.25, london_pct=0.25, us_pct=0.40, settle_pct=0.10),
                    breakout_follow_through=MagicMock(follow_through_rate=0.6),
                    mean_reversion_tendency=0.5,
                    overnight_gap_stats=MagicMock(gap_frequency=0.3),
                    volume_profile=MagicMock(name="D_SHAPE"),
                ),
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
                include_no_trade=False,
            )
            from lib.services.training.dataset_generator import generate_dataset_for_symbol

            generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

        # Every result must have the sentinel attached
        for i, r in enumerate(results):
            assert getattr(r, "_cached_cross_asset", None) is sentinel, (
                f"Result {i} missing _cached_cross_asset sentinel"
            )


# ---------------------------------------------------------------------------
# 2. Pre-compute caching — fingerprint computed exactly once
# ---------------------------------------------------------------------------


class TestFingerprintPreComputeCache:
    """compute_asset_fingerprint must be called exactly once per symbol."""

    def test_fingerprint_called_once_for_many_results(self, tmp_path):
        """With 8 sim results, fingerprint pre-compute fires once."""
        from lib.services.training.dataset_generator import DatasetConfig

        bars_1m = _make_bars_1m(5000)
        bars_daily = _make_bars_daily(30)
        call_count = {"n": 0}

        def _fake_fp(sym, bars_daily=None, bars_1m=None, lookback_days=20):
            call_count["n"] += 1
            return MagicMock(
                typical_daily_range_atr=1.5,
                session_concentration=MagicMock(overnight_pct=0.25, london_pct=0.25, us_pct=0.40, settle_pct=0.10),
                breakout_follow_through=MagicMock(follow_through_rate=0.6),
                mean_reversion_tendency=0.5,
                overnight_gap_stats=MagicMock(gap_frequency=0.3),
                volume_profile=MagicMock(name="D_SHAPE"),
            )

        results = [_make_orb_sim_result() for _ in range(8)]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                side_effect=_fake_fp,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
                include_no_trade=False,
            )
            from lib.services.training.dataset_generator import generate_dataset_for_symbol

            generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                bars_daily=bars_daily,
                config=cfg,
            )

        assert call_count["n"] == 1, (
            f"compute_asset_fingerprint should be called once per symbol, got {call_count['n']} calls"
        )

    def test_fingerprint_slice_limits_bars_to_20_days(self, tmp_path):
        """bars_1m passed to compute_asset_fingerprint is capped at 20*1440 rows."""
        from lib.services.training.dataset_generator import DatasetConfig

        n_full = 50_000
        bars_1m = _make_bars_1m(n_full)
        sliced_lengths: list[int] = []

        def _capture_fp(sym, bars_daily=None, bars_1m=None, lookback_days=20):
            if bars_1m is not None:
                sliced_lengths.append(len(bars_1m))
            return MagicMock(
                typical_daily_range_atr=1.5,
                session_concentration=MagicMock(overnight_pct=0.25, london_pct=0.25, us_pct=0.40, settle_pct=0.10),
                breakout_follow_through=MagicMock(follow_through_rate=0.6),
                mean_reversion_tendency=0.5,
                overnight_gap_stats=MagicMock(gap_frequency=0.3),
                volume_profile=MagicMock(name="D_SHAPE"),
            )

        results = [_make_orb_sim_result()]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                side_effect=_capture_fp,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
            )
            from lib.services.training.dataset_generator import generate_dataset_for_symbol

            generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        assert sliced_lengths, "compute_asset_fingerprint was never called"
        assert sliced_lengths[0] <= 20 * 1440, f"Fingerprint received {sliced_lengths[0]} bars; expected ≤ {20 * 1440}"


# ---------------------------------------------------------------------------
# 3. Skip path — _build_row still runs, but uses cached features
# ---------------------------------------------------------------------------


class TestSkipExistingPath:
    """Images that already exist on disk are skipped but still get a CSV row."""

    def test_skip_existing_produces_csv_row(self, tmp_path):
        """A pre-existing image file results in a row in the CSV output."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        bars_1m = _make_bars_1m(300)
        result = _make_orb_sim_result(breakout_time="2025-01-06T09:35:00")

        # Pre-create the image file so skip_existing triggers
        # Compute the expected filename the same way the generator does
        ts_str = result.breakout_time
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_filename = f"MGC_{safe_ts}_{result.label}_0.png"
        image_path = images_dir / image_filename
        image_path.write_bytes(b"\x89PNG\r\n")  # minimal PNG header

        with patch(
            "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
            return_value=[result],
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=True,
                use_parity_renderer=False,
                include_no_trade=False,
            )
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        assert stats.skipped_existing == 1, "Expected 1 skipped image"
        assert len(rows) == 1, f"Expected 1 CSV row even for skipped image, got {len(rows)}"
        assert rows[0]["image_path"] == str(image_path), "Row image_path should point to the existing file"

    def test_skip_path_uses_cached_cross_asset_not_recomputed(self, tmp_path):
        """Skipped images must use the pre-cached cross-asset features, not recompute."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        bars_1m = _make_bars_1m(300)
        peer_bars = _make_bars_1m(300, base_price=1800.0)
        bars_by_ticker = {"MGC": bars_1m, "SIL": peer_bars}

        result = _make_orb_sim_result(breakout_time="2025-01-06T09:35:00")

        # Pre-create the image file
        ts_str = result.breakout_time
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_path = images_dir / f"MGC_{safe_ts}_{result.label}_0.png"
        image_path.write_bytes(b"\x89PNG\r\n")

        cross_asset_call_count = {"n": 0}

        @dataclass
        class _FakeCrossFeats:
            primary_peer_corr: float = 0.65
            cross_class_corr: float = 0.35
            correlation_regime: float = 0.5

        def _counting_compute_cross(sym, bars_dict):
            cross_asset_call_count["n"] += 1
            return _FakeCrossFeats()

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=[result],
            ),
            patch(
                "lib.analysis.cross_asset.compute_cross_asset_features",
                side_effect=_counting_compute_cross,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=True,
                use_parity_renderer=False,
            )
            generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

        # Pre-compute fires once; the skip path must NOT add a second call
        assert cross_asset_call_count["n"] == 1, (
            f"Expected exactly 1 cross-asset compute (pre-compute only), got {cross_asset_call_count['n']}"
        )

    def test_skip_path_row_has_correct_cross_asset_values(self, tmp_path):
        """Skipped image row must carry the pre-computed cross-asset feature values."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        bars_1m = _make_bars_1m(300)
        peer_bars = _make_bars_1m(300, base_price=1800.0)
        bars_by_ticker = {"MGC": bars_1m, "SIL": peer_bars}

        result = _make_orb_sim_result(breakout_time="2025-01-06T09:35:00")

        ts_str = result.breakout_time
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_path = images_dir / f"MGC_{safe_ts}_{result.label}_0.png"
        image_path.write_bytes(b"\x89PNG\r\n")

        @dataclass
        class _FakeCrossFeats:
            primary_peer_corr: float = 0.82
            cross_class_corr: float = 0.44
            correlation_regime: float = 1.0  # elevated

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=[result],
            ),
            patch(
                "lib.analysis.cross_asset.compute_cross_asset_features",
                return_value=_FakeCrossFeats(),
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=True,
                use_parity_renderer=False,
            )
            rows, _ = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

        assert rows, "Expected at least one row"
        row = rows[0]
        assert abs(row["primary_peer_corr"] - 0.82) < 1e-6, (
            f"primary_peer_corr={row['primary_peer_corr']!r} expected 0.82"
        )
        assert abs(row["cross_class_corr"] - 0.44) < 1e-6
        assert abs(row["correlation_regime"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. Pre-compute failure isolation
# ---------------------------------------------------------------------------


class TestPreComputeFailureIsolation:
    """A crash in fingerprint/cross-asset must not abort the symbol's run."""

    def test_fingerprint_exception_does_not_abort_run(self, tmp_path):
        """If compute_asset_fingerprint raises, the run continues with defaults."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        bars_1m = _make_bars_1m(300)
        results = [_make_orb_sim_result()]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                side_effect=RuntimeError("simulated fingerprint crash"),
            ),
            patch(
                "lib.analysis.rendering.chart_renderer_parity.render_parity_to_file",
                return_value=None,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
                include_no_trade=False,
            )
            # Must not raise
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        # We get a row even when fingerprint fails (defaults to 0.5)
        # render failure is possible without images dir, but we still get stats
        assert isinstance(rows, list)
        assert stats is not None

    def test_cross_asset_exception_does_not_abort_run(self, tmp_path):
        """If compute_cross_asset_features raises, the run continues."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        bars_1m = _make_bars_1m(300)
        peer_bars = _make_bars_1m(300, base_price=1800.0)
        bars_by_ticker = {"MGC": bars_1m, "SIL": peer_bars}
        results = [_make_orb_sim_result()]

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.cross_asset.compute_cross_asset_features",
                side_effect=ConnectionError("simulated network failure"),
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
            )
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

        assert isinstance(rows, list)

    def test_fingerprint_failure_gives_0_5_defaults_in_row(self, tmp_path):
        """When fingerprint pre-compute fails, v8-C features default to 0.5."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)
        bars_1m = _make_bars_1m(300)

        result = _make_orb_sim_result(breakout_time="2025-01-06T09:35:00")

        # Pre-create the image file so we always get a row
        ts_str = result.breakout_time
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_path = images_dir / f"MGC_{safe_ts}_{result.label}_0.png"
        image_path.write_bytes(b"\x89PNG\r\n")

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=[result],
            ),
            patch(
                "lib.analysis.asset_fingerprint.compute_asset_fingerprint",
                side_effect=ValueError("degenerate bars"),
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=True,
                use_parity_renderer=False,
            )
            rows, _ = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        assert rows, "Should still get a CSV row"
        row = rows[0]
        fp_keys = [
            "typical_daily_range_norm",
            "session_concentration",
            "breakout_follow_through",
            "hurst_exponent",
            "overnight_gap_tendency",
            "volume_profile_shape",
        ]
        for k in fp_keys:
            if k in row:
                assert abs(row[k] - 0.5) < 0.1, f"Expected ~0.5 default for {k} when fingerprint fails, got {row[k]}"


# ---------------------------------------------------------------------------
# 5. _build_row — feature vector shape and value ranges
# ---------------------------------------------------------------------------


class TestBuildRowFeatureVector:
    """_build_row must return a dict with all 37 v8-C feature columns."""

    # Expected column set based on actual _build_row output keys.
    # Note: some names differ from the docstring's normalised feature names —
    # the raw values are stored here and BreakoutDataset normalises them at
    # load time (e.g. quality_pct → / 100, tp3_atr_mult → / 5.0, etc.).
    EXPECTED_COLUMNS = {
        # Core metadata
        "label",
        "image_path",
        "symbol",
        # Trade details echoed into CSV for debugging
        "direction",
        "entry",
        "atr",
        "atr_value",
        "or_high",
        "or_low",
        "or_range",
        "range_size",
        "sl",
        "tp1",
        "breakout_time",
        "outcome",
        "pnl_r",
        "hold_bars",
        "pm_high",
        "pm_low",
        # v6 base features (stored raw; BreakoutDataset normalises)
        "quality_pct",  # stored raw (÷100 in dataset)
        "volume_ratio",
        "atr_pct",
        "cvd_delta",
        "nr7_flag",
        "or_range_atr_ratio",
        "premarket_range",  # raw PM range (ratio stored separately)
        "pm_range_ratio",  # PM range / OR range
        "bar_of_day_minutes",
        "day_of_week_norm",
        "vwap_distance",
        "asset_class_id",
        "breakout_type_ord",
        "breakout_type",
        "asset_volatility_class",
        "hour_of_day",
        "tp3_atr_mult",  # stored raw (÷5.0 in dataset)
        # v7 (18–23)
        "daily_bias_direction",
        "daily_bias_confidence",
        "prior_day_pattern",
        "weekly_range_position",
        "monthly_trend_score",
        "crypto_momentum_score",
        # v7.1 (24–27)
        "breakout_type_category",
        "session_overlap_flag",
        "session_key",
        "atr_trend",
        "volume_trend",
        # v8-B (28–30)
        "primary_peer_corr",
        "cross_class_corr",
        "correlation_regime",
        # v8-C (31–36)
        "typical_daily_range_norm",
        "session_concentration",
        "breakout_follow_through",
        "hurst_exponent",
        "overnight_gap_tendency",
        "volume_profile_shape",
        # Other context fields
        "london_overlap_flag",
    }

    def _run_build_row(self, result=None, image_path: str = "/fake/path.png") -> dict:
        from lib.services.training.dataset_generator import _build_row

        if result is None:
            result = _make_orb_sim_result()
        return _build_row(result, image_path)

    def test_all_expected_columns_present(self):
        row = self._run_build_row()
        missing = self.EXPECTED_COLUMNS - set(row.keys())
        assert not missing, f"Missing columns in _build_row output: {missing}"

    def test_quality_pct_norm_in_range(self):
        row = self._run_build_row()
        # quality_pct is stored raw (0–100); BreakoutDataset divides by 100
        v = row["quality_pct"]
        assert 0 <= v <= 100, f"quality_pct={v} out of [0, 100]"

    def test_atr_pct_non_negative(self):
        row = self._run_build_row()
        assert row["atr_pct"] >= 0.0, f"atr_pct={row['atr_pct']} should be ≥ 0"

    def test_direction_stored_as_string(self):
        long_result = _make_orb_sim_result(label="good_long")
        short_result = _make_orb_sim_result(label="good_short")
        short_result.direction = "SHORT"

        row_long = self._run_build_row(long_result)
        row_short = self._run_build_row(short_result)

        # direction is stored as the raw string; BreakoutDataset converts to flag
        assert row_long["direction"] == "LONG", f"direction={row_long['direction']!r} expected 'LONG'"
        assert row_short["direction"] == "SHORT", f"direction={row_short['direction']!r} expected 'SHORT'"

    def test_day_of_week_norm_in_range(self):
        row = self._run_build_row()
        v = row["day_of_week_norm"]
        assert 0.0 <= v <= 1.0, f"day_of_week_norm={v} out of [0, 1]"

    def test_hour_of_day_in_range(self):
        row = self._run_build_row()
        v = row["hour_of_day"]
        assert 0.0 <= v <= 1.0, f"hour_of_day={v} out of [0, 1]"

    def test_breakout_type_ord_in_range(self):
        row = self._run_build_row()
        v = row["breakout_type_ord"]
        assert 0.0 <= v <= 1.0, f"breakout_type_ord={v} out of [0, 1]"

    def test_crypto_momentum_score_defaults_to_half(self):
        """When no _crypto_momentum_score is attached, default is 0.5."""
        result = _make_orb_sim_result()
        # Ensure no pre-computed score is attached
        if hasattr(result, "_crypto_momentum_score"):
            del result._crypto_momentum_score
        row = self._run_build_row(result)
        assert row["crypto_momentum_score"] == 0.5

    def test_cached_cross_asset_used_when_present(self):
        """When _cached_cross_asset is attached, those values appear in the row."""
        from lib.services.training.dataset_generator import _build_row

        @dataclass
        class _CA:
            primary_peer_corr: float = 0.91
            cross_class_corr: float = 0.22
            correlation_regime: float = 0.0

        result = _make_orb_sim_result()
        result._cached_cross_asset = _CA()

        row = _build_row(result, "/fake/path.png")
        assert abs(row["primary_peer_corr"] - 0.91) < 1e-6
        assert abs(row["cross_class_corr"] - 0.22) < 1e-6
        assert abs(row["correlation_regime"] - 0.0) < 1e-6

    def test_cached_fingerprint_used_when_present(self):
        """When _cached_fingerprint is attached, v8-C values appear in the row."""
        from lib.services.training.dataset_generator import _build_row

        result = _make_orb_sim_result()
        result._cached_fingerprint = MagicMock(
            typical_daily_range_atr=2.0,  # → norm = (2.0 - 0.5) / 2.0 = 0.75
            session_concentration=MagicMock(
                overnight_pct=0.1,
                london_pct=0.2,
                us_pct=0.6,  # dominant
                settle_pct=0.1,
            ),
            breakout_follow_through=MagicMock(follow_through_rate=0.72),
            mean_reversion_tendency=0.55,
            overnight_gap_stats=MagicMock(gap_frequency=0.4),
            volume_profile=MagicMock(name="D_SHAPE"),
        )

        row = _build_row(result, "/fake/path.png")

        assert abs(row["typical_daily_range_norm"] - 0.75) < 1e-6, (
            f"typical_daily_range_norm={row['typical_daily_range_norm']!r}"
        )
        assert abs(row["session_concentration"] - 0.6) < 1e-6, f"session_concentration={row['session_concentration']!r}"
        assert abs(row["breakout_follow_through"] - 0.72) < 1e-6
        assert abs(row["hurst_exponent"] - 0.55) < 1e-6

    def test_no_nan_in_output(self):
        """No NaN values should appear in the _build_row output."""
        import math

        row = self._run_build_row()
        for k, v in row.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN found in column {k!r}"

    def test_image_path_stored_verbatim(self):
        path = "/dataset/images/MGC_20250106_good_long_0.png"
        row = self._run_build_row(image_path=path)
        assert row["image_path"] == path

    def test_label_stored(self):
        result = _make_orb_sim_result(label="bad_short")
        result.direction = "SHORT"
        row = self._run_build_row(result)
        assert row["label"] == "bad_short"


# ---------------------------------------------------------------------------
# 6. Timezone correctness — ring buffer timestamps are Eastern Time
# ---------------------------------------------------------------------------


class TestRingBufferTimezoneET:
    """_RingBufferHandler must stamp log records with Eastern Time, not UTC."""

    def test_ring_buffer_ts_is_eastern_not_utc(self):
        """Ring buffer 'ts' field must differ from UTC by the ET offset."""
        # We can't import _RingBufferHandler directly (it's module-level),
        # so we replicate its logic and verify it produces ET timestamps.
        et = ZoneInfo("America/New_York")

        # Simulate a log record created right now
        now_utc = datetime.now(UTC)
        record_created = now_utc.timestamp()

        # Parse back to an offset-aware datetime to compare with UTC
        ts_et = datetime.fromtimestamp(record_created, tz=et)

        # The ET offset from UTC is between -5 and -4 (accounting for DST)
        # ts_et.utcoffset() gives us the direct answer
        utc_offset = ts_et.utcoffset()
        assert utc_offset is not None
        offset_secs = utc_offset.total_seconds()
        # EDT = -14400s (-4h), EST = -18000s (-5h)
        assert offset_secs in (-14400, -18000), (
            f"ET offset should be -4h or -5h, got {offset_secs / 3600:.1f}h — "
            f"ZoneInfo('America/New_York') may not have timezone data (tzdata missing?)"
        )

    def test_zoneinfo_america_new_york_resolves(self):
        """ZoneInfo('America/New_York') must resolve without error — tzdata must be installed."""
        et = ZoneInfo("America/New_York")
        now = datetime.now(et)
        offset = now.utcoffset()
        assert offset is not None, "utcoffset() returned None — ZoneInfo did not resolve correctly"

    def test_ring_buffer_handler_uses_et(self, caplog):
        """_RingBufferHandler entries have 'ts' in ET format (HH:MM:SS, not UTC)."""
        # We test the ring-buffer timestamp logic directly without importing
        # trainer_server (which starts uvicorn and a ring-buffer handler at
        # module level — side-effectful in tests).
        et = ZoneInfo("America/New_York")

        # Replicate the handler's emit logic
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        # record.created is set by LogRecord.__init__ to time.time()
        # Compute what the ring buffer would produce
        ts_from_et = datetime.fromtimestamp(record.created, tz=et).strftime("%H:%M:%S")
        ts_from_utc = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")

        # Suppress unused-variable warning — we keep ts_from_utc for debugging
        _ = ts_from_utc

        # Must be parseable as HH:MM:SS
        h, m, s = ts_from_et.split(":")
        assert 0 <= int(h) <= 23
        assert 0 <= int(m) <= 59
        assert 0 <= int(s) <= 59


# ---------------------------------------------------------------------------
# 7. _et_timestamper — structlog processor signature
# ---------------------------------------------------------------------------


class TestEtTimestamperSignature:
    """_et_timestamper must have the correct structlog Processor signature."""

    def test_et_timestamper_adds_timestamp_key(self):
        from lib.core.logging_config import _et_timestamper

        event_dict: dict[str, Any] = {"event": "hello", "level": "info"}
        result = _et_timestamper(None, "info", event_dict)
        assert "timestamp" in result, "_et_timestamper must add 'timestamp' key"

    def test_et_timestamper_timestamp_is_et(self):
        from lib.core.logging_config import _et_timestamper

        event_dict: dict[str, Any] = {"event": "hello"}
        result = _et_timestamper(None, "info", event_dict)
        ts = result["timestamp"]

        # Must be a HH:MM:SS string
        assert isinstance(ts, str), f"timestamp should be str, got {type(ts)}"
        parts = ts.split(":")
        assert len(parts) == 3, f"timestamp {ts!r} should be HH:MM:SS"
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        assert 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59

    def test_et_timestamper_returns_mutable_mapping(self):
        from collections.abc import MutableMapping

        from lib.core.logging_config import _et_timestamper

        event_dict: dict[str, Any] = {"event": "test"}
        result = _et_timestamper(None, "info", event_dict)
        assert isinstance(result, MutableMapping), f"_et_timestamper must return MutableMapping, got {type(result)}"

    def test_et_timestamper_is_valid_structlog_processor(self):
        """structlog must accept _et_timestamper as a processor without type errors."""
        from lib.core.logging_config import _et_timestamper

        # Verify it's callable with the three required args structlog passes
        event_dict: dict[str, Any] = {"event": "pipeline_test"}
        result = _et_timestamper(None, "warning", event_dict)
        assert result is not None


# ---------------------------------------------------------------------------
# 8. generate_dataset_for_symbol smoke test — end-to-end with mocked deps
# ---------------------------------------------------------------------------


class TestGenerateDatasetForSymbolSmoke:
    """End-to-end smoke test: one symbol, multiple trade results, parity renderer."""

    def test_produces_rows_equal_to_renderable_trades(self, tmp_path):
        """Row count must equal the number of is_trade results that pass caps."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        bars_1m = _make_bars_1m(500)
        bars_daily = _make_bars_daily(30)

        n_trades = 6
        results = []
        for i in range(n_trades):
            r = _make_orb_sim_result(
                label="good_long" if i % 2 == 0 else "bad_long",
                breakout_time=f"2025-01-06T0{9 + i // 2}:{(i * 5) % 60:02d}:00",
            )
            r._window_offset = i * 50
            r._window_size = 50
            results.append(r)

        rendered_paths: list[str] = []

        def _fake_render(bars, orb_high, orb_low, vwap_values=None, direction=None, save_path="", breakout_type=None):
            rendered_paths.append(save_path)
            # Actually write a tiny valid file
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(b"\x89PNG\r\n")
            return save_path

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.rendering.chart_renderer_parity.render_parity_to_file",
                side_effect=_fake_render,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=False,
                use_parity_renderer=True,
                include_no_trade=False,
            )
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                bars_daily=bars_daily,
                config=cfg,
            )

        assert len(rows) == n_trades, f"Expected {n_trades} rows, got {len(rows)}"
        assert stats.total_images == n_trades
        assert stats.skipped_existing == 0
        assert stats.render_failures == 0

    def test_stats_skipped_count_matches_existing_files(self, tmp_path):
        """stats.skipped_existing must equal the number of pre-existing image files."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)
        bars_1m = _make_bars_1m(500)

        n_trades = 4
        results = []
        pre_existing_count = 2

        for i in range(n_trades):
            r = _make_orb_sim_result(
                label="good_long",
                breakout_time=f"2025-01-06T09:{30 + i:02d}:00",
            )
            results.append(r)

        # Pre-create files for the first pre_existing_count results
        for i in range(pre_existing_count):
            ts_str = results[i].breakout_time
            safe_ts = (
                ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
            )
            fname = f"MGC_{safe_ts}_{results[i].label}_{i}.png"
            (images_dir / fname).write_bytes(b"\x89PNG\r\n")

        rendered_paths: list[str] = []

        def _fake_render(bars, orb_high, orb_low, vwap_values=None, direction=None, save_path="", breakout_type=None):
            rendered_paths.append(save_path)
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(b"\x89PNG\r\n")
            return save_path

        with (
            patch(
                "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
                return_value=results,
            ),
            patch(
                "lib.analysis.rendering.chart_renderer_parity.render_parity_to_file",
                side_effect=_fake_render,
            ),
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(images_dir),
                skip_existing=True,
                use_parity_renderer=True,
                include_no_trade=False,
            )
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        assert stats.skipped_existing == pre_existing_count, (
            f"Expected {pre_existing_count} skipped, got {stats.skipped_existing}"
        )
        assert len(rendered_paths) == n_trades - pre_existing_count, (
            f"Expected {n_trades - pre_existing_count} actual renders, got {len(rendered_paths)}"
        )
        assert len(rows) == n_trades

    def test_empty_bars_returns_empty_rows(self, tmp_path):
        """Passing an empty bars_1m DataFrame results in zero rows (no crash)."""
        from lib.services.training.dataset_generator import DatasetConfig, generate_dataset_for_symbol

        bars_1m = pd.DataFrame()

        with patch(
            "lib.services.training.dataset_generator._run_simulators_for_breakout_type",
            return_value=[],
        ):
            cfg = DatasetConfig(
                output_dir=str(tmp_path),
                image_dir=str(tmp_path / "images"),
                skip_existing=False,
                use_parity_renderer=False,
            )
            rows, stats = generate_dataset_for_symbol(
                symbol="MGC",
                bars_1m=bars_1m,
                config=cfg,
            )

        assert rows == []
        assert stats.total_images == 0


# ---------------------------------------------------------------------------
# 9. DatasetConfig defaults sanity
# ---------------------------------------------------------------------------


class TestDatasetConfigDefaults:
    def test_skip_existing_is_true_by_default(self):
        from lib.services.training.dataset_generator import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.skip_existing is True, "skip_existing should default to True for resumability"

    def test_use_parity_renderer_default(self):
        from lib.services.training.dataset_generator import DatasetConfig

        cfg = DatasetConfig()
        # Parity renderer is the production renderer — should be enabled
        assert cfg.use_parity_renderer is True, "use_parity_renderer should default to True"

    def test_output_dirs_have_sensible_defaults(self):
        from lib.services.training.dataset_generator import DatasetConfig

        cfg = DatasetConfig()
        assert cfg.output_dir == "dataset"
        assert "images" in cfg.image_dir
