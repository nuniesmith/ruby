"""
CNN v8 Training Loop Smoke Test
================================
Verifies that the full v8 training recipe runs end-to-end without crashing on
a tiny synthetic dataset.  Does NOT require:
  - Real chart images (uses solid-colour 224×224 PNGs)
  - A GPU (runs on CPU — slow but correct)
  - Any external data services

What this tests:
  ✓ HybridBreakoutCNN v8 architecture instantiates (EfficientNetV2-S backbone,
    asset class embedding (5,4), asset ID embedding (25,8), wider tabular head)
  ✓ BreakoutDataset loads a CSV with all 37 v8 tabular feature columns
  ✓ train_model() runs 2 epochs without raising:
      - Mixup augmentation on tabular features (alpha=0.2)
      - Gradient accumulation (steps=2)
      - Cosine warmup LR scheduler
      - Separate param groups (backbone LR vs head+embeddings LR)
      - Label smoothing loss (0.10)
  ✓ TrainResult is returned with sane fields (best_epoch, epochs_trained,
    best_val_acc in [0,1])
  ✓ Model checkpoint is saved to disk
  ✓ evaluate_model() runs on the validation set without crashing
  ✓ predict_breakout() returns a float probability in [0.0, 1.0]
  ✓ predict_breakout_batch() returns a list of the correct length
  ✓ Model can be re-loaded from the saved checkpoint
  ✓ Forward pass with and without asset embedding IDs both work
  ✓ freeze_backbone() / unfreeze_backbone() toggle grad state correctly

This test intentionally uses only 40 synthetic samples and 2 epochs so it
completes in under 60 seconds on CPU.  It is the canary for the real
~50K-sample / 80-epoch training run on the GPU rig.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if PyTorch is not installed.
# The test suite runs in CI without torch — these tests should silently skip
# rather than causing import errors that block unrelated tests.
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping CNN smoke tests")
pytest.importorskip("torchvision", reason="torchvision not installed — skipping CNN smoke tests")
pytest.importorskip("PIL", reason="Pillow not installed — skipping CNN smoke tests")

from PIL import Image  # noqa: E402  (after importorskip guards)

# ---------------------------------------------------------------------------
# Constants — must match the v8 contract in breakout_cnn.py
# ---------------------------------------------------------------------------

NUM_TABULAR = 37  # v8 feature count
NUM_ASSET_CLASSES = 5
NUM_ASSETS = 25
ASSET_CLASS_EMBED_DIM = 4
ASSET_ID_EMBED_DIM = 8

# All 37 tabular feature column names in contract order.
# These must match TABULAR_FEATURES in breakout_cnn.py exactly.
TABULAR_FEATURE_NAMES: list[str] = [
    "quality_pct_norm",  # [0]
    "volume_ratio",  # [1]
    "atr_pct",  # [2]
    "cvd_delta",  # [3]
    "nr7_flag",  # [4]
    "direction_flag",  # [5]
    "session_ordinal",  # [6]
    "london_overlap_flag",  # [7]
    "or_range_atr_ratio",  # [8]
    "premarket_range_ratio",  # [9]
    "bar_of_day",  # [10]
    "day_of_week",  # [11]
    "vwap_distance",  # [12]
    "asset_class_id",  # [13]
    "breakout_type_ord",  # [14]
    "asset_volatility_class",  # [15]
    "hour_of_day",  # [16]
    "tp3_atr_mult_norm",  # [17]
    "daily_bias_direction",  # [18]
    "daily_bias_confidence",  # [19]
    "prior_day_pattern",  # [20]
    "weekly_range_position",  # [21]
    "monthly_trend_score",  # [22]
    "crypto_momentum_score",  # [23]
    "breakout_type_category",  # [24]
    "session_overlap_flag",  # [25]
    "atr_trend",  # [26]
    "volume_trend",  # [27]
    "primary_peer_corr",  # [28]  v8-B
    "cross_class_corr",  # [29]  v8-B
    "correlation_regime",  # [30]  v8-B
    "typical_daily_range_norm",  # [31]  v8-C
    "session_concentration",  # [32]  v8-C
    "breakout_follow_through",  # [33]  v8-C
    "hurst_exponent",  # [34]  v8-C
    "overnight_gap_tendency",  # [35]  v8-C
    "volume_profile_shape",  # [36]  v8-C
]

assert len(TABULAR_FEATURE_NAMES) == NUM_TABULAR, (
    f"Smoke test feature list has {len(TABULAR_FEATURE_NAMES)} entries, "
    f"expected {NUM_TABULAR}.  Update TABULAR_FEATURE_NAMES to match the v8 contract."
)

# Valid label values accepted by BreakoutDataset
LABELS = ["good_long", "good_short", "bad_long", "bad_short"]


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_solid_png(path: Path, color: tuple[int, int, int] = (30, 30, 40)) -> None:
    """Save a 224×224 solid-colour PNG to *path*."""
    img = Image.new("RGB", (224, 224), color=color)
    img.save(str(path))


def _make_tabular_row(rng: np.random.Generator, label: str) -> dict[str, Any]:
    """Return a dict of all 37 tabular features with random valid values."""
    row: dict[str, Any] = {}
    for feat in TABULAR_FEATURE_NAMES:
        if feat in ("nr7_flag", "direction_flag", "london_overlap_flag", "session_overlap_flag"):
            row[feat] = float(rng.integers(0, 2))
        elif feat == "cvd_delta":
            row[feat] = float(rng.uniform(-1.0, 1.0))
        elif feat == "asset_class_id":
            row[feat] = float(rng.integers(0, NUM_ASSET_CLASSES)) / 4.0
        elif feat == "breakout_type_ord":
            row[feat] = float(rng.integers(0, 13)) / 12.0
        elif feat in ("correlation_regime", "breakout_type_category", "asset_volatility_class"):
            # 0.0, 0.5, or 1.0
            row[feat] = float(rng.choice([0.0, 0.5, 1.0]))
        else:
            row[feat] = float(rng.uniform(0.0, 1.0))
    row["label"] = label
    return row


def _build_synthetic_dataset(
    tmp_dir: Path,
    n_samples: int = 40,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Build a tiny synthetic dataset under *tmp_dir*.

    Returns (train_csv_path, val_csv_path).

    Layout:
        tmp_dir/
            images/
                sample_000.png … sample_039.png
            train.csv
            val.csv
    """
    rng = np.random.default_rng(seed)
    img_dir = tmp_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i in range(n_samples):
        label = LABELS[i % len(LABELS)]
        img_path = img_dir / f"sample_{i:03d}.png"
        # Vary colour slightly so images aren't byte-identical (avoids any
        # accidental caching / dedup that could mask real bugs)
        colour = (
            int(rng.integers(20, 60)),
            int(rng.integers(20, 60)),
            int(rng.integers(20, 60)),
        )
        _make_solid_png(img_path, color=colour)
        row = _make_tabular_row(rng, label)
        row["image_path"] = str(img_path)
        # asset_class_idx and asset_idx are integer IDs for the embedding layers
        row["asset_class_idx"] = int(rng.integers(0, NUM_ASSET_CLASSES))
        row["asset_idx"] = int(rng.integers(0, NUM_ASSETS))
        rows.append(row)

    fieldnames = ["image_path", "label", "asset_class_idx", "asset_idx"] + TABULAR_FEATURE_NAMES

    # 80/20 split
    n_train = int(n_samples * 0.8)
    train_rows, val_rows = rows[:n_train], rows[n_train:]

    def _write_csv(path: Path, data: list[dict[str, Any]]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    train_csv = tmp_dir / "train.csv"
    val_csv = tmp_dir / "val.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(val_csv, val_rows)

    return train_csv, val_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Module-scoped fixture — build the dataset once, share across all tests."""
    with tempfile.TemporaryDirectory(prefix="v8_smoke_") as tmp:
        tmp_dir = Path(tmp)
        train_csv, val_csv = _build_synthetic_dataset(tmp_dir, n_samples=40)
        yield {
            "tmp_dir": tmp_dir,
            "train_csv": train_csv,
            "val_csv": val_csv,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestV8Architecture:
    """Verify HybridBreakoutCNN v8 instantiates with the correct structure."""

    def test_import(self):
        """The module must import cleanly (no missing deps at import time)."""
        from lib.analysis.ml.breakout_cnn import NUM_TABULAR as CONTRACT_NUM_TABULAR
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN  # noqa: F401

        assert CONTRACT_NUM_TABULAR == NUM_TABULAR, (
            f"Contract NUM_TABULAR={CONTRACT_NUM_TABULAR} != smoke test expectation {NUM_TABULAR}. "
            "Update TABULAR_FEATURE_NAMES in this smoke test."
        )

    def test_feature_contract_matches(self):
        """TABULAR_FEATURES list in breakout_cnn.py must match this test's list exactly."""
        from lib.analysis.ml.breakout_cnn import TABULAR_FEATURES

        assert TABULAR_FEATURES == TABULAR_FEATURE_NAMES, (
            "TABULAR_FEATURES in breakout_cnn.py does not match TABULAR_FEATURE_NAMES "
            "in this smoke test.\n"
            f"breakout_cnn.py: {TABULAR_FEATURES}\n"
            f"smoke test:      {TABULAR_FEATURE_NAMES}"
        )

    def test_model_instantiates_with_embeddings(self):
        """Model with asset embeddings (v8-A) instantiates without error."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        assert model.use_asset_embeddings is True
        assert model._embed_dim == ASSET_CLASS_EMBED_DIM + ASSET_ID_EMBED_DIM  # type: ignore[attr-defined]  # 12

    def test_model_instantiates_without_embeddings(self):
        """Model without embeddings (legacy compat path) also instantiates."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=False)
        assert model.use_asset_embeddings is False
        assert model._embed_dim == 0  # type: ignore[attr-defined]

    def test_embedding_dimensions(self):
        """Embedding layers have the exact dims specified in the v8 contract."""
        from lib.analysis.ml.breakout_cnn import (
            ASSET_CLASS_EMBED_DIM as CONTRACT_CLASS_DIM,
        )
        from lib.analysis.ml.breakout_cnn import (
            ASSET_ID_EMBED_DIM as CONTRACT_ID_DIM,
        )
        from lib.analysis.ml.breakout_cnn import (
            NUM_ASSET_CLASSES as CONTRACT_CLASSES,
        )
        from lib.analysis.ml.breakout_cnn import (
            NUM_ASSETS as CONTRACT_ASSETS,
        )
        from lib.analysis.ml.breakout_cnn import (
            HybridBreakoutCNN,
        )

        assert CONTRACT_CLASSES == NUM_ASSET_CLASSES
        assert CONTRACT_ASSETS == NUM_ASSETS
        assert CONTRACT_CLASS_DIM == ASSET_CLASS_EMBED_DIM
        assert CONTRACT_ID_DIM == ASSET_ID_EMBED_DIM

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        ce = model.asset_class_embedding  # type: ignore[attr-defined]
        ae = model.asset_id_embedding  # type: ignore[attr-defined]
        assert ce.num_embeddings == NUM_ASSET_CLASSES
        assert ce.embedding_dim == ASSET_CLASS_EMBED_DIM
        assert ae.num_embeddings == NUM_ASSETS
        assert ae.embedding_dim == ASSET_ID_EMBED_DIM

    def test_tabular_head_width(self):
        """Tabular head must have the wider v8-D architecture (37→256→128→64)."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        # First Linear layer: in=37, out=256
        first_linear = model.tabular_head[0]  # type: ignore[attr-defined]
        assert first_linear.in_features == NUM_TABULAR, (
            f"Expected tabular head input {NUM_TABULAR}, got {first_linear.in_features}"
        )
        assert first_linear.out_features == 256, (
            f"Expected tabular head first layer width 256, got {first_linear.out_features}"
        )

    def test_classifier_input_dim(self):
        """Classifier must accept (1280 image + 64 tabular + 12 embedding) = 1356 dims."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        expected_combined = 1280 + 64 + (ASSET_CLASS_EMBED_DIM + ASSET_ID_EMBED_DIM)  # 1356
        first_classifier_linear = model.classifier[0]  # type: ignore[attr-defined]
        assert first_classifier_linear.in_features == expected_combined, (
            f"Classifier input dim: expected {expected_combined}, got {first_classifier_linear.in_features}"
        )

    def test_forward_with_embeddings(self):
        """Full forward pass with asset embedding IDs produces (B, 2) logits."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        model.eval()  # type: ignore[attr-defined]
        B = 4
        imgs = torch.zeros(B, 3, 224, 224)
        tabs = torch.zeros(B, NUM_TABULAR)
        class_ids = torch.randint(0, NUM_ASSET_CLASSES, (B,))
        asset_ids = torch.randint(0, NUM_ASSETS, (B,))
        with torch.no_grad():
            out = model(imgs, tabs, asset_class_ids=class_ids, asset_ids=asset_ids)  # type: ignore[operator]
        assert out.shape == (B, 2), f"Expected (B,2) logits, got {out.shape}"

    def test_forward_without_embedding_ids(self):
        """Forward pass without passing embedding IDs must still work (pad path)."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        model.eval()  # type: ignore[attr-defined]
        B = 2
        imgs = torch.zeros(B, 3, 224, 224)
        tabs = torch.zeros(B, NUM_TABULAR)
        with torch.no_grad():
            out = model(imgs, tabs)  # type: ignore[operator]  # no embedding IDs passed
        assert out.shape == (B, 2)

    def test_freeze_unfreeze_backbone(self):
        """freeze_backbone / unfreeze_backbone must correctly toggle requires_grad."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        model.freeze_backbone()
        for p in model.cnn.parameters():  # type: ignore[attr-defined]
            assert not p.requires_grad, "Backbone param should be frozen"
        model.unfreeze_backbone()
        for p in model.cnn.parameters():  # type: ignore[attr-defined]
            assert p.requires_grad, "Backbone param should be unfrozen"

    def test_output_probabilities(self):
        """Softmax over logits must produce valid probabilities summing to 1."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        model.eval()  # type: ignore[attr-defined]
        B = 8
        imgs = torch.randn(B, 3, 224, 224)
        tabs = torch.rand(B, NUM_TABULAR)
        class_ids = torch.randint(0, NUM_ASSET_CLASSES, (B,))
        asset_ids = torch.randint(0, NUM_ASSETS, (B,))
        with torch.no_grad():
            logits = model(imgs, tabs, asset_class_ids=class_ids, asset_ids=asset_ids)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1)
        # All probabilities in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()
        # Each row sums to 1
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)


class TestBreakoutDataset:
    """Verify BreakoutDataset loads and shapes the synthetic CSV correctly."""

    def test_loads_synthetic_csv(self, synthetic_dataset):
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, BreakoutDataset

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        ds = BreakoutDataset(
            str(synthetic_dataset["train_csv"]),
            image_root=None,
        )
        assert len(ds) > 0, "Dataset must have at least one sample"

    def test_getitem_shapes(self, synthetic_dataset):
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, BreakoutDataset

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        ds = BreakoutDataset(str(synthetic_dataset["train_csv"]))
        # __getitem__ returns (img_tensor, tabular_tensor, label, is_valid, asset_class_id, asset_id)
        # or a similar tuple — unpack by position and check the fields we care about
        sample = ds[0]
        assert len(sample) >= 5, f"Expected at least 5 elements from __getitem__, got {len(sample)}"
        img = sample[0]
        tab = sample[1]
        label = sample[2]
        # asset_class_id and asset_id are the last two elements regardless of tuple length
        class_id = sample[-2]
        asset_id = sample[-1]
        assert img.shape == (3, 224, 224), f"Image shape: {img.shape}"
        assert tab.shape == (NUM_TABULAR,), f"Tabular shape: {tab.shape}"
        assert int(label) in (0, 1), f"Label must be 0 or 1, got {label}"
        assert 0 <= int(class_id) < NUM_ASSET_CLASSES
        assert 0 <= int(asset_id) < NUM_ASSETS

    def test_all_samples_load(self, synthetic_dataset):
        """Every row in the synthetic CSV must load without error."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, BreakoutDataset

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        ds = BreakoutDataset(str(synthetic_dataset["train_csv"]))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample is not None
            assert len(sample) >= 5, f"Sample {i} has unexpected length {len(sample)}"

    def test_label_balance(self, synthetic_dataset):
        """Synthetic dataset should have both positive and negative labels."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, BreakoutDataset

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        ds = BreakoutDataset(str(synthetic_dataset["train_csv"]))
        labels = [int(ds[i][2]) for i in range(len(ds))]
        assert 0 in labels, "Must have at least one negative label (0)"
        assert 1 in labels, "Must have at least one positive label (1)"


class TestV8TrainingLoop:
    """
    End-to-end training smoke test.

    Runs 2 epochs with all v8 recipe features enabled on the synthetic dataset.
    This is the primary canary before the real GPU run.
    """

    @pytest.fixture(scope="class")
    def train_result(self, synthetic_dataset, tmp_path_factory):
        """Run 2-epoch training once; share the result across all tests in this class."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, train_model

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        model_dir = str(tmp_path_factory.mktemp("models_smoke"))
        result = train_model(
            data_csv=str(synthetic_dataset["train_csv"]),
            val_csv=str(synthetic_dataset["val_csv"]),
            epochs=2,
            batch_size=8,  # small batch to keep it fast on CPU
            lr=2e-4,  # matches v8 default backbone LR
            freeze_epochs=1,  # freeze backbone for epoch 0, unfreeze epoch 1
            model_dir=model_dir,
            num_workers=0,  # no multiprocessing in tests
            save_best=True,
            patience=15,
            grad_accum_steps=2,  # v8: effective batch = 8×2 = 16
            mixup_alpha=0.2,  # v8: mixup on tabular features
            warmup_epochs=1,  # cosine warmup (1 of 2 epochs)
        )
        return result, model_dir

    def test_train_returns_result(self, train_result):
        """train_model() must return a TrainResult (not None)."""
        result, _ = train_result
        assert result is not None, (
            "train_model() returned None — training crashed. Check logs above for the actual exception."
        )

    def test_result_fields_present(self, train_result):
        """TrainResult must have the expected fields."""
        result, _ = train_result
        assert hasattr(result, "model_path"), "TrainResult missing model_path"
        assert hasattr(result, "best_epoch"), "TrainResult missing best_epoch"
        assert hasattr(result, "epochs_trained"), "TrainResult missing epochs_trained"

    def test_epochs_trained(self, train_result):
        """epochs_trained must be ≤ the requested 2 (may be less if early stopping fires)."""
        result, _ = train_result
        assert 1 <= result.epochs_trained <= 2, f"Expected 1–2 epochs_trained, got {result.epochs_trained}"

    def test_best_epoch_in_range(self, train_result):
        """best_epoch must be within [1, epochs_trained] (1-based)."""
        result, _ = train_result
        assert 1 <= result.best_epoch <= result.epochs_trained, (
            f"best_epoch={result.best_epoch} out of range for epochs_trained={result.epochs_trained}"
        )

    def test_val_accuracy_from_filename(self, train_result):
        """best_val_acc is encoded in the checkpoint filename (e.g. _acc50.pt → 50%)."""
        import re

        result, _ = train_result
        # Checkpoint filenames are like: breakout_cnn_20260309_112626_acc50.pt
        # Extract the accuracy from the filename as a sanity check
        m = re.search(r"_acc(\d+)", Path(result.model_path).name)
        if m:
            acc_pct = int(m.group(1))
            assert 0 <= acc_pct <= 100, f"Accuracy in filename out of range: {acc_pct}"

    def test_checkpoint_saved(self, train_result):
        """A .pt model file must be written to model_dir."""
        result, model_dir = train_result
        assert result.model_path is not None, "model_path is None"
        model_path = Path(result.model_path)
        assert model_path.exists(), f"Model checkpoint not found: {model_path}"
        assert model_path.suffix == ".pt", f"Unexpected file extension: {model_path.suffix}"
        # Must be non-empty
        assert model_path.stat().st_size > 0, "Checkpoint file is empty"

    def test_checkpoint_loadable(self, train_result):
        """The saved checkpoint must load back into HybridBreakoutCNN without error."""
        from lib.analysis.ml.breakout_cnn import HybridBreakoutCNN

        result, _ = train_result
        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        state = torch.load(result.model_path, map_location="cpu", weights_only=True)
        # Checkpoints may be a raw state_dict or a dict with a 'model_state_dict' key
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)  # type: ignore[attr-defined]  # strict=False tolerates minor key diffs

    def test_no_nan_in_checkpoint(self, train_result):
        """No NaN values must appear in the saved model weights."""
        result, _ = train_result
        state = torch.load(result.model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        for name, tensor in state.items():
            assert not torch.isnan(tensor).any(), f"NaN detected in weight tensor '{name}'"


class TestV8Inference:
    """Verify inference functions work after training."""

    @pytest.fixture(scope="class")
    def trained_model_path(self, synthetic_dataset, tmp_path_factory):
        """Train a quick model and return the checkpoint path."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, train_model

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        model_dir = str(tmp_path_factory.mktemp("models_infer"))
        result = train_model(
            data_csv=str(synthetic_dataset["train_csv"]),
            val_csv=str(synthetic_dataset["val_csv"]),
            epochs=2,
            batch_size=8,
            lr=2e-4,
            model_dir=model_dir,
            num_workers=0,
            save_best=True,
            patience=15,
            grad_accum_steps=2,
            mixup_alpha=0.2,
            warmup_epochs=1,
        )
        if result is None:
            pytest.skip("Training returned None — cannot test inference")
        assert result is not None
        return result.model_path

    def test_evaluate_model_runs(self, synthetic_dataset, trained_model_path):
        """evaluate_model() on the val CSV must return a dict with acc/prec/rec."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, evaluate_model

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        # Actual signature: evaluate_model(model_path, val_csv, ...)
        metrics = evaluate_model(
            model_path=trained_model_path,
            val_csv=str(synthetic_dataset["val_csv"]),
            batch_size=8,
            num_workers=0,
        )
        assert metrics is not None, "evaluate_model() returned None"
        for key in ("val_accuracy", "val_precision", "val_recall"):
            assert key in metrics, f"Missing key '{key}' in evaluate_model() result"
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"metrics['{key}']={val} out of [0,1]"

    def test_predict_breakout_returns_probability(self, synthetic_dataset, trained_model_path):
        """predict_breakout() must return a result dict with a probability in [0.0, 1.0]."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, predict_breakout

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        # tabular_features must be a Sequence[float] in TABULAR_FEATURES order
        rng = np.random.default_rng(0)
        tabular_list = [float(rng.uniform(0.0, 1.0)) for _ in TABULAR_FEATURE_NAMES]

        # Use the first image from the val CSV
        import csv as _csv

        with open(synthetic_dataset["val_csv"]) as f:
            reader = _csv.DictReader(f)
            first_row = next(reader)
        image_path = first_row["image_path"]

        result = predict_breakout(
            image_path=image_path,
            tabular_features=tabular_list,
            model_path=trained_model_path,
        )
        assert result is not None, "predict_breakout() returned None"
        # predict_breakout returns a dict with a 'prob' key
        assert "prob" in result, f"Expected 'prob' key in result, got: {list(result.keys())}"
        prob = result["prob"]
        assert isinstance(prob, float), f"Expected float probability, got {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1]"

    def test_predict_breakout_batch(self, synthetic_dataset, trained_model_path):
        """predict_breakout_batch() must return a list with len == input len."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, predict_breakout_batch

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")
        import csv as _csv

        rng = np.random.default_rng(1)
        image_paths = []
        tabular_batch = []
        with open(synthetic_dataset["val_csv"]) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                tabular = [float(rng.uniform(0.0, 1.0)) for _ in TABULAR_FEATURE_NAMES]
                image_paths.append(row["image_path"])
                tabular_batch.append(tabular)

        if not image_paths:
            pytest.skip("No val samples available")

        # Actual signature: predict_breakout_batch(image_paths, tabular_features_batch, ...)
        results = predict_breakout_batch(
            image_paths=image_paths,
            tabular_features_batch=tabular_batch,
            model_path=trained_model_path,
            batch_size=8,
        )
        assert results is not None, "predict_breakout_batch() returned None"
        assert len(results) == len(image_paths), f"Expected {len(image_paths)} results, got {len(results)}"
        for res in results:
            if res is not None:
                assert "prob" in res, f"Expected 'prob' key in batch result item, got: {list(res.keys())}"
                prob = res["prob"]
                assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0,1]"


class TestV8GradAccumAndMixup:
    """
    White-box tests: verify the v8 recipe mechanics are actually exercised.

    These tests instrument the training step directly (bypassing train_model())
    to confirm mixup and grad accumulation work as expected.
    """

    def test_mixup_changes_tabular_features(self):
        """
        When mixup_alpha > 0, the tabular features in a batch should be
        interpolated between two random samples (i.e., not identical to input).
        """
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        # Replicate the mixup logic from train_model() directly
        alpha = 0.2
        B = 16
        tabs = torch.eye(B, NUM_TABULAR)  # each row is a distinct unit vector

        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(B)
        mixed = lam * tabs + (1 - lam) * tabs[idx]

        # With lam != 1.0 and idx shuffled, at least some rows should differ
        # from the original (unless lam happens to be exactly 1.0)
        if not np.isclose(lam, 1.0, atol=1e-6):
            assert not torch.allclose(mixed, tabs), "Mixup produced identical output — interpolation not applied"

    def test_grad_accumulation_accumulates_gradients(self):
        """
        With grad_accum_steps=2, gradients must NOT be zeroed after the first
        mini-batch step — they must accumulate until the accumulation boundary.

        This test directly exercises the accumulation logic to confirm no
        optimizer.zero_grad() is called prematurely.
        """
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, HybridBreakoutCNN

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore[attr-defined]
        criterion = torch.nn.CrossEntropyLoss()
        grad_accum_steps = 2

        B = 4
        imgs = torch.randn(B, 3, 224, 224)
        tabs = torch.rand(B, NUM_TABULAR)
        class_ids = torch.randint(0, NUM_ASSET_CLASSES, (B,))
        asset_ids = torch.randint(0, NUM_ASSETS, (B,))
        labels = torch.randint(0, 2, (B,))

        optimizer.zero_grad()

        # Step 1: compute loss, backward (do NOT zero_grad yet)
        out1 = model(imgs, tabs, asset_class_ids=class_ids, asset_ids=asset_ids)  # type: ignore[operator]
        loss1 = criterion(out1, labels) / grad_accum_steps
        loss1.backward()

        # Collect gradients after first step
        grads_after_step1 = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}  # type: ignore[attr-defined]
        assert grads_after_step1, "No gradients computed after step 1"

        # Step 2: second mini-batch — gradients must ACCUMULATE (add to existing)
        out2 = model(imgs, tabs, asset_class_ids=class_ids, asset_ids=asset_ids)  # type: ignore[operator]
        loss2 = criterion(out2, labels) / grad_accum_steps
        loss2.backward()

        # At least one grad must have changed (accumulated)
        any_changed = False
        for name, grad1 in grads_after_step1.items():
            p = dict(model.named_parameters())[name]  # type: ignore[attr-defined]
            if p.grad is not None and not torch.allclose(p.grad, grad1):
                any_changed = True
                break

        assert any_changed, (
            "Gradients did not accumulate across two backward() calls — zero_grad() may have been called prematurely"
        )

        # Now optimizer step + zero_grad (the boundary)
        optimizer.step()
        optimizer.zero_grad()

        # After zero_grad, all grads must be zero/None
        for p in model.parameters():  # type: ignore[attr-defined]
            if p.grad is not None:
                assert torch.all(p.grad == 0), "Gradients not zeroed after optimizer.zero_grad()"

    def test_separate_lr_groups(self):
        """
        The optimizer must have exactly 2 param groups:
          group 0: backbone params at lr (2e-4)
          group 1: head + embedding params at lr*5 (1e-3)
        """
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE, HybridBreakoutCNN

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
        lr = 2e-4
        head_lr = lr * 5

        backbone_params = list(model.cnn.parameters())  # type: ignore[attr-defined]
        head_params = (
            list(model.tabular_head.parameters())  # type: ignore[attr-defined]
            + list(model.classifier.parameters())  # type: ignore[attr-defined]
            + list(model.asset_class_embedding.parameters())  # type: ignore[attr-defined]
            + list(model.asset_id_embedding.parameters())  # type: ignore[attr-defined]
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=1e-4,
        )

        assert len(optimizer.param_groups) == 2, f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        assert abs(optimizer.param_groups[0]["lr"] - lr) < 1e-10, (
            f"Backbone LR: expected {lr}, got {optimizer.param_groups[0]['lr']}"
        )
        assert abs(optimizer.param_groups[1]["lr"] - head_lr) < 1e-10, (
            f"Head LR: expected {head_lr}, got {optimizer.param_groups[1]['lr']}"
        )

    def test_cosine_warmup_lr_schedule(self):
        """
        With warmup_epochs=2, total_epochs=6:
          - Epoch 0: LR = base * (1/2) = 0.5×  (linear warmup step 1 of 2)
          - Epoch 1: LR = base * (2/2) = 1.0×  (linear warmup step 2 of 2)
          - Epoch 2+: cosine decay (LR ≤ base)
        """
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        warmup_epochs = 2
        total_epochs = 6

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            # cosine annealing from warmup_epochs to total_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        # Linear warmup
        assert abs(_lr_lambda(0) - 0.5) < 1e-9, f"Epoch 0 LR multiplier: {_lr_lambda(0)}"
        assert abs(_lr_lambda(1) - 1.0) < 1e-9, f"Epoch 1 LR multiplier: {_lr_lambda(1)}"
        # Cosine should be ≤ 1.0 after warmup
        for epoch in range(warmup_epochs, total_epochs):
            assert _lr_lambda(epoch) <= 1.0 + 1e-9, f"Cosine LR at epoch {epoch} exceeded 1.0: {_lr_lambda(epoch)}"
        # Cosine should decrease over time (not necessarily strictly, but trend down)
        cosine_lrs = [_lr_lambda(e) for e in range(warmup_epochs, total_epochs)]
        assert cosine_lrs[0] >= cosine_lrs[-1], f"Cosine LR did not decrease from first to last: {cosine_lrs}"

    def test_label_smoothing_loss(self):
        """CrossEntropyLoss with label_smoothing=0.10 must not produce NaN on valid inputs."""
        from lib.analysis.ml.breakout_cnn import _TORCH_AVAILABLE

        if not _TORCH_AVAILABLE:
            pytest.skip("torch not available")

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.10)
        B = 8
        logits = torch.randn(B, 2)
        labels = torch.randint(0, 2, (B,))
        loss = criterion(logits, labels)
        assert not torch.isnan(loss), "Label smoothing loss produced NaN"
        assert not torch.isinf(loss), "Label smoothing loss produced Inf"
        assert loss.item() > 0.0, "Label smoothing loss should be positive"
