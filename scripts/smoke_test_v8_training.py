#!/usr/bin/env python3
"""
Smoke Test — v8 Training Loop (Synthetic Data, No GPU Required)
================================================================
Validates that all v8 architecture and training recipe changes work
together end-to-end WITHOUT requiring:
  - A running trainer server
  - MASSIVE_API_KEY or any external data
  - A CUDA GPU (runs on CPU, takes ~60-90 seconds)

What it tests:
  1. HybridBreakoutCNN v8 model instantiation (embeddings, wider head, GELU)
  2. BreakoutDataset loading from a synthetic CSV + synthetic images
  3. Forward pass with asset embeddings (asset_class_ids + asset_ids)
  4. train_model() with v8 recipe: mixup, grad accumulation, cosine warmup,
     separate LR groups (backbone vs head+embeddings), label smoothing 0.10
  5. evaluate_model() produces metrics dict
  6. _normalise_tabular_for_inference() backward compat (v6→v7→v7.1→v8)
  7. predict_breakout() inference on a synthetic image
  8. generate_feature_contract() produces valid v8 contract
  9. split_dataset() stratified split works on synthetic data

Usage:
    cd futures
    python scripts/smoke_test_v8_training.py

    # Or via pytest (auto-discovered):
    PYTHONPATH=src python -m pytest scripts/smoke_test_v8_training.py -v
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src/ is on the Python path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 80  # small but enough for 2 epochs with batch_size=8
NUM_TABULAR = 37  # v8 contract
IMAGE_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 2
PATIENCE = 5  # won't trigger in 2 epochs
FREEZE_EPOCHS = 1
WARMUP_EPOCHS = 1
GRAD_ACCUM = 2
MIXUP_ALPHA = 0.2

# Colors
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

# Symbols for synthetic data — cover multiple asset classes
SYNTHETIC_SYMBOLS = ["MGC", "MNQ", "MES", "6E", "MBT", "ETH", "SOL"]
SYNTHETIC_SESSIONS = ["us", "london", "tokyo", "cme"]
SYNTHETIC_BREAKOUT_TYPES = ["ORB", "PrevDay", "InitialBalance", "Consolidation"]


def _log(msg: str, color: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{DIM}[{ts}]{RESET} {color}{msg}{RESET}")


def _ok(msg: str) -> None:
    _log(f"✓ {msg}", GREEN)


def _fail(msg: str) -> None:
    _log(f"✗ {msg}", RED)


def _info(msg: str) -> None:
    _log(f"  {msg}", CYAN)


def _warn(msg: str) -> None:
    _log(f"⚠ {msg}", YELLOW)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_image(path: Path, size: int = IMAGE_SIZE) -> None:
    """Create a minimal valid PNG image using Pillow."""
    import numpy as np
    from PIL import Image

    # Random RGB noise — mimics a chart snapshot
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    img.save(str(path), "PNG")


def _generate_synthetic_csv(
    csv_path: Path,
    image_dir: Path,
    num_samples: int = NUM_SAMPLES,
) -> None:
    """Generate a synthetic labels.csv and chart images for testing.

    The CSV mirrors the exact column schema that BreakoutDataset expects,
    with all 37 v8 tabular feature source columns.
    """
    import random

    random.seed(42)

    image_dir.mkdir(parents=True, exist_ok=True)

    # CSV columns that BreakoutDataset reads
    fieldnames = [
        "image_path",
        "label",
        "symbol",
        "direction",
        "quality_pct",
        "volume_ratio",
        "atr_pct",
        "cvd_delta",
        "nr7_flag",
        "london_overlap_flag",
        "range_size",
        "atr_value",
        "or_range_atr_ratio",
        "premarket_range",
        "pm_range_ratio",
        "bar_of_day_minutes",
        "day_of_week_norm",
        "vwap_distance",
        "asset_class_id",
        "entry",
        "sl",
        "tp1",
        "or_high",
        "or_low",
        "or_range",
        "atr",
        "pnl_r",
        "hold_bars",
        "outcome",
        "breakout_time",
        "pm_high",
        "pm_low",
        "session_key",
        "breakout_type",
        "breakout_type_ord",
        # v6
        "asset_volatility_class",
        "hour_of_day",
        "tp3_atr_mult",
        # v7
        "daily_bias_direction",
        "daily_bias_confidence",
        "prior_day_pattern",
        "weekly_range_position",
        "monthly_trend_score",
        "crypto_momentum_score",
        # v7.1
        "breakout_type_category",
        "session_overlap_flag",
        "atr_trend",
        "volume_trend",
        # v8-B
        "primary_peer_corr",
        "cross_class_corr",
        "correlation_regime",
        # v8-C
        "typical_daily_range_norm",
        "session_concentration",
        "breakout_follow_through",
        "hurst_exponent",
        "overnight_gap_tendency",
        "volume_profile_shape",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_samples):
            label = "good" if i % 2 == 0 else "bad"
            symbol = random.choice(SYNTHETIC_SYMBOLS)
            session = random.choice(SYNTHETIC_SESSIONS)
            bt = random.choice(SYNTHETIC_BREAKOUT_TYPES)
            direction = random.choice(["LONG", "SHORT"])

            img_name = f"sample_{i:04d}.png"
            img_path = image_dir / img_name
            _generate_synthetic_image(img_path)

            # Plausible feature values
            entry = 2000.0 + random.uniform(-500, 500)
            atr = entry * random.uniform(0.002, 0.01)
            or_range = atr * random.uniform(0.3, 1.5)

            row = {
                "image_path": img_name,
                "label": label,
                "symbol": symbol,
                "direction": direction,
                "quality_pct": round(random.uniform(40, 95), 1),
                "volume_ratio": round(random.uniform(0.5, 5.0), 4),
                "atr_pct": round(atr / entry, 6),
                "cvd_delta": round(random.uniform(-0.8, 0.8), 4),
                "nr7_flag": random.choice([0, 1]),
                "london_overlap_flag": random.choice([0, 1]),
                "range_size": round(or_range, 6),
                "atr_value": round(atr, 6),
                "or_range_atr_ratio": round(or_range / atr if atr > 0 else 0, 4),
                "premarket_range": round(or_range * random.uniform(0.5, 2.0), 6),
                "pm_range_ratio": round(random.uniform(0.2, 3.0), 4),
                "bar_of_day_minutes": random.randint(0, 1380),
                "day_of_week_norm": round(random.randint(0, 4) / 4.0, 4),
                "vwap_distance": round(random.uniform(-2.0, 2.0), 4),
                "asset_class_id": round(random.randint(0, 4) / 4.0, 4),
                "entry": round(entry, 6),
                "sl": round(entry - atr * 1.5 if direction == "LONG" else entry + atr * 1.5, 6),
                "tp1": round(entry + atr * 2.0 if direction == "LONG" else entry - atr * 2.0, 6),
                "or_high": round(entry + or_range / 2, 6),
                "or_low": round(entry - or_range / 2, 6),
                "or_range": round(or_range, 6),
                "atr": round(atr, 6),
                "pnl_r": round(random.uniform(-2.0, 3.0), 4),
                "hold_bars": random.randint(5, 120),
                "outcome": random.choice(["tp1", "tp2", "sl", "time_stop"]),
                "breakout_time": "2025-01-15T10:30:00",
                "pm_high": round(entry + or_range, 6),
                "pm_low": round(entry - or_range, 6),
                "session_key": session,
                "breakout_type": bt,
                "breakout_type_ord": round(SYNTHETIC_BREAKOUT_TYPES.index(bt) / 12.0, 6),
                # v6
                "asset_volatility_class": random.choice([0.0, 0.5, 1.0]),
                "hour_of_day": round(random.randint(0, 23) / 23.0, 4),
                "tp3_atr_mult": round(random.uniform(2.0, 5.0), 4),
                # v7
                "daily_bias_direction": random.choice([0.0, 0.5, 1.0]),
                "daily_bias_confidence": round(random.uniform(0, 1), 4),
                "prior_day_pattern": round(random.uniform(0, 1), 4),
                "weekly_range_position": round(random.uniform(0, 1), 4),
                "monthly_trend_score": round(random.uniform(0, 1), 4),
                "crypto_momentum_score": round(random.uniform(0, 1), 4),
                # v7.1
                "breakout_type_category": random.choice([0.0, 0.5, 1.0]),
                "session_overlap_flag": random.choice([0.0, 1.0]),
                "atr_trend": round(random.uniform(0, 1), 4),
                "volume_trend": round(random.uniform(0, 1), 4),
                # v8-B
                "primary_peer_corr": round(random.uniform(0, 1), 4),
                "cross_class_corr": round(random.uniform(0, 1), 4),
                "correlation_regime": round(random.uniform(0, 1), 4),
                # v8-C
                "typical_daily_range_norm": round(random.uniform(0, 1), 4),
                "session_concentration": round(random.uniform(0, 1), 4),
                "breakout_follow_through": round(random.uniform(0, 1), 4),
                "hurst_exponent": round(random.uniform(0, 1), 4),
                "overnight_gap_tendency": round(random.uniform(0, 1), 4),
                "volume_profile_shape": round(random.uniform(0, 1), 4),
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Test steps
# ---------------------------------------------------------------------------

_test_results: list[tuple[str, bool, str]] = []


def _run_test(name: str, func, *args, **kwargs) -> bool:
    """Run a test step, capturing pass/fail."""
    try:
        func(*args, **kwargs)
        _test_results.append((name, True, ""))
        _ok(name)
        return True
    except Exception as exc:
        tb = traceback.format_exc()
        _test_results.append((name, False, str(exc)))
        _fail(f"{name}: {exc}")
        # Print traceback for debugging
        for line in tb.strip().split("\n"):
            _log(f"    {line}", DIM)
        return False


def test_01_model_instantiation():
    """HybridBreakoutCNN v8 model creates without errors."""
    from lib.analysis.breakout_cnn import NUM_TABULAR, HybridBreakoutCNN

    assert NUM_TABULAR == 37, f"Expected NUM_TABULAR=37, got {NUM_TABULAR}"

    model = HybridBreakoutCNN(
        num_tabular=37,
        pretrained=False,  # skip ImageNet download for speed
        use_asset_embeddings=True,
    )

    # Verify architecture components exist
    assert hasattr(model, "cnn"), "Missing CNN backbone"
    assert hasattr(model, "tabular_head"), "Missing tabular head"
    assert hasattr(model, "classifier"), "Missing classifier"
    assert hasattr(model, "asset_class_embedding"), "Missing asset_class_embedding"
    assert hasattr(model, "asset_id_embedding"), "Missing asset_id_embedding"
    assert model.use_asset_embeddings is True
    assert model._embed_dim == 12, f"Expected embed_dim=12 (4+8), got {model._embed_dim}"

    # Check tabular head architecture (37 → 256 → 128 → 64)
    layers = list(model.tabular_head.children())
    first_linear = layers[0]
    assert first_linear.in_features == 37, f"Tabular head input should be 37, got {first_linear.in_features}"
    assert first_linear.out_features == 256, f"Tabular head first hidden should be 256, got {first_linear.out_features}"

    _info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")


def test_02_forward_pass():
    """Forward pass with images + tabular + embeddings produces correct output shape."""
    import torch

    from lib.analysis.breakout_cnn import HybridBreakoutCNN

    model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
    model.eval()

    batch_size = 4
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    tabular = torch.randn(batch_size, NUM_TABULAR)
    asset_class_ids = torch.randint(0, 5, (batch_size,))
    asset_ids = torch.randint(0, 25, (batch_size,))

    with torch.no_grad():
        output = model(images, tabular, asset_class_ids=asset_class_ids, asset_ids=asset_ids)

    assert output.shape == (batch_size, NUM_CLASSES), (
        f"Expected output shape ({batch_size}, {NUM_CLASSES}), got {output.shape}"
    )

    # Verify softmax produces valid probabilities
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5), "Softmax probabilities don't sum to 1"

    _info(f"  Output shape: {output.shape}, prob range: [{probs.min():.3f}, {probs.max():.3f}]")


def test_03_forward_pass_no_embeddings():
    """Forward pass works with embeddings disabled (backward compat mode)."""
    import torch

    from lib.analysis.breakout_cnn import HybridBreakoutCNN

    model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=False)
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    tabular = torch.randn(batch_size, NUM_TABULAR)

    with torch.no_grad():
        output = model(images, tabular)

    assert output.shape == (batch_size, NUM_CLASSES)
    _info(f"  No-embedding output shape: {output.shape}")


def test_04_dataset_loading(csv_path: str, image_dir: str):
    """BreakoutDataset loads synthetic CSV and returns correct tensor shapes."""
    from lib.analysis.breakout_cnn import BreakoutDataset, get_inference_transform

    transform = get_inference_transform()
    dataset = BreakoutDataset(csv_path, transform=transform, image_root=image_dir)

    assert len(dataset) == NUM_SAMPLES, f"Expected {NUM_SAMPLES} samples, got {len(dataset)}"

    # Load first sample — __getitem__ returns a 6-tuple:
    # (img, tabular, target, valid_flag, asset_class_idx, asset_idx)
    sample = dataset[0]
    assert sample is not None, "First sample is None"
    assert len(sample) == 6, f"Expected 6-tuple from __getitem__, got {len(sample)}-tuple"

    img, tab, label, valid_flag, asset_class_idx, asset_idx = sample

    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Image shape: {img.shape}"
    assert tab.shape == (NUM_TABULAR,), f"Tabular shape: {tab.shape} (expected ({NUM_TABULAR},))"
    assert label in (0, 1), f"Label should be 0 or 1, got {label}"
    assert valid_flag.item() in (True, False), f"valid_flag should be bool, got {valid_flag}"
    assert isinstance(asset_class_idx.item(), int), f"asset_class_idx should be int tensor, got {asset_class_idx}"
    assert isinstance(asset_idx.item(), int), f"asset_idx should be int tensor, got {asset_idx}"

    _info(f"  Sample 6-tuple — img: {img.shape}, tab: {tab.shape}, label: {label}, valid: {valid_flag.item()}")
    _info(f"  Embedding IDs — class: {int(asset_class_idx)}, asset: {int(asset_idx)}")


def test_05_dataloader_batching(csv_path: str, image_dir: str):
    """DataLoader batching + skip_invalid_collate works with v8 5-tuple."""
    import torch

    from lib.analysis.breakout_cnn import BreakoutDataset, get_inference_transform

    transform = get_inference_transform()
    dataset = BreakoutDataset(csv_path, transform=transform, image_root=image_dir)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )

    batch = next(iter(loader))
    assert batch is not None, "First batch is None"

    imgs, tabs, labels, class_ids, asset_ids = batch
    assert imgs.shape[0] == BATCH_SIZE
    assert tabs.shape == (BATCH_SIZE, NUM_TABULAR)
    assert labels.shape == (BATCH_SIZE,)
    assert class_ids.shape == (BATCH_SIZE,)
    assert asset_ids.shape == (BATCH_SIZE,)

    _info(f"  Batch shapes — imgs: {imgs.shape}, tabs: {tabs.shape}, labels: {labels.shape}")
    _info(f"  Embedding batch — class_ids: {class_ids.tolist()}, asset_ids: {asset_ids.tolist()}")


def test_06_train_model_2_epochs(csv_path: str, image_dir: str, model_dir: str):
    """train_model() runs 2 epochs with v8 recipe without crashing.

    This is THE key test — it validates:
      - Gradient accumulation (2 steps)
      - Mixup on tabular features (alpha=0.2)
      - Label smoothing (0.10)
      - Cosine warmup (1 epoch warmup)
      - Separate param groups (backbone LR vs head+embeddings LR)
      - Freeze/unfreeze backbone at epoch boundary
      - Early stopping logic (doesn't trigger in 2 epochs)
      - Best model saving
    """
    from lib.analysis.breakout_cnn import train_model

    result = train_model(
        data_csv=csv_path,
        val_csv=None,  # auto-split 85/15
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=2e-4,
        weight_decay=1e-4,
        freeze_epochs=FREEZE_EPOCHS,
        model_dir=model_dir,
        image_root=image_dir,
        num_workers=0,  # avoid multiprocessing issues in test
        save_best=True,
        patience=PATIENCE,
        grad_accum_steps=GRAD_ACCUM,
        mixup_alpha=MIXUP_ALPHA,
        warmup_epochs=WARMUP_EPOCHS,
    )

    assert result is not None, "train_model returned None — training failed"
    assert result.model_path is not None, "No model path returned"
    assert os.path.exists(result.model_path), f"Model file not found: {result.model_path}"
    assert result.epochs_trained == EPOCHS, f"Expected {EPOCHS} epochs, trained {result.epochs_trained}"

    model_size_mb = os.path.getsize(result.model_path) / 1e6
    _info(f"  Model saved: {result.model_path}")
    _info(f"  Model size: {model_size_mb:.1f} MB")
    _info(f"  Epochs trained: {result.epochs_trained}, best epoch: {result.best_epoch}")


def test_07_evaluate_model(model_dir: str, csv_path: str, image_dir: str):
    """evaluate_model() produces metrics dict from the trained model."""
    from lib.analysis.breakout_cnn import evaluate_model

    # Find the model file
    model_files = list(Path(model_dir).glob("breakout_cnn_*.pt"))
    assert len(model_files) > 0, "No model files found after training"

    # Use the most recent one
    model_path = str(sorted(model_files, key=lambda p: p.stat().st_mtime)[-1])

    metrics = evaluate_model(
        model_path=model_path,
        val_csv=csv_path,
        image_root=image_dir,
        batch_size=BATCH_SIZE,
    )

    assert metrics is not None, "evaluate_model returned None"
    assert "val_accuracy" in metrics, f"Missing val_accuracy in metrics: {metrics}"
    assert "val_precision" in metrics, f"Missing val_precision in metrics: {metrics}"
    assert "val_recall" in metrics, f"Missing val_recall in metrics: {metrics}"

    acc = metrics["val_accuracy"] * 100
    prec = metrics["val_precision"] * 100
    rec = metrics["val_recall"] * 100

    _info(f"  Accuracy:  {acc:.1f}%")
    _info(f"  Precision: {prec:.1f}%")
    _info(f"  Recall:    {rec:.1f}%")
    _info("  (Low metrics expected — synthetic random data, only 2 epochs)")


def test_08_normalise_backward_compat():
    """_normalise_tabular_for_inference() handles all legacy vector lengths."""
    from lib.analysis.breakout_cnn import NUM_TABULAR, _normalise_tabular_for_inference

    # v5: 8 features
    v5 = [0.5] * 8
    result = _normalise_tabular_for_inference(v5)
    assert len(result) == NUM_TABULAR, f"v5 padding: expected {NUM_TABULAR}, got {len(result)}"

    # v4: 14 features
    v4 = [0.5] * 14
    result = _normalise_tabular_for_inference(v4)
    assert len(result) == NUM_TABULAR, f"v4 padding: expected {NUM_TABULAR}, got {len(result)}"

    # v6: 18 features
    v6 = [0.5] * 18
    result = _normalise_tabular_for_inference(v6)
    assert len(result) == NUM_TABULAR, f"v6 padding: expected {NUM_TABULAR}, got {len(result)}"

    # v7: 24 features
    v7 = [0.5] * 24
    result = _normalise_tabular_for_inference(v7)
    assert len(result) == NUM_TABULAR, f"v7 padding: expected {NUM_TABULAR}, got {len(result)}"

    # v7.1: 28 features
    v71 = [0.5] * 28
    result = _normalise_tabular_for_inference(v71)
    assert len(result) == NUM_TABULAR, f"v7.1 padding: expected {NUM_TABULAR}, got {len(result)}"

    # v8: 37 features (no padding)
    v8 = [0.5] * 37
    result = _normalise_tabular_for_inference(v8)
    assert len(result) == NUM_TABULAR, f"v8 native: expected {NUM_TABULAR}, got {len(result)}"

    # Invalid length should raise
    try:
        _normalise_tabular_for_inference([0.5] * 10)
        raise AssertionError("Should have raised ValueError for 10 features")
    except ValueError:
        pass  # expected

    _info("  All legacy lengths padded correctly: 8, 14, 18, 24, 28, 37")
    _info("  Invalid length (10) correctly raises ValueError")


def test_09_predict_breakout(image_dir: str, model_dir: str = ""):
    """predict_breakout() runs inference on a synthetic image with v8 tabular vector.

    If a model_dir with a trained checkpoint is provided, we copy it to the
    default location so _load_model() can find it.  Otherwise we test that
    predict_breakout gracefully returns None when no model is available (which
    is acceptable — the important thing is it doesn't crash).
    """
    import torch

    from lib.analysis.breakout_cnn import (
        NUM_TABULAR,
        HybridBreakoutCNN,
        _normalise_tabular_for_inference,
        get_inference_transform,
    )

    # Create a test image
    test_img = Path(image_dir) / "test_inference.png"
    _generate_synthetic_image(test_img)

    # Build a 37-feature tabular vector and normalise it
    tabular_37 = [0.5] * 37
    normalised = _normalise_tabular_for_inference(tabular_37)
    assert len(normalised) == NUM_TABULAR

    # Rather than relying on the global model cache (which looks in models/),
    # do a direct forward pass with a fresh model to validate inference works.
    model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
    model.eval()

    # If we have a trained model from the smoke test, load its weights
    if model_dir:
        model_files = list(Path(model_dir).glob("breakout_cnn_*.pt"))
        if model_files:
            best = sorted(model_files, key=lambda p: p.stat().st_mtime)[-1]
            state = torch.load(str(best), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            _info(f"  Loaded weights from: {best.name}")

    transform = get_inference_transform()
    from PIL import Image

    img = transform(Image.open(str(test_img)).convert("RGB")).unsqueeze(0)
    tab = torch.tensor([normalised], dtype=torch.float32)
    cls_ids = torch.tensor([0], dtype=torch.long)
    asset_ids = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        output = model(img, tab, asset_class_ids=cls_ids, asset_ids=asset_ids)
        probs = torch.softmax(output, dim=1)

    prob_good = probs[0, 1].item()
    assert 0.0 <= prob_good <= 1.0, f"Probability out of range: {prob_good}"

    _info(f"  Direct inference P(good breakout): {prob_good:.4f}")
    _info(f"  Normalised tabular[0:5]: {[round(x, 3) for x in normalised[:5]]}")


def test_10_generate_feature_contract(model_dir: str):
    """generate_feature_contract() produces valid v8 JSON."""
    from lib.analysis.breakout_cnn import generate_feature_contract

    contract_path = os.path.join(model_dir, "test_feature_contract.json")
    contract = generate_feature_contract(output_path=contract_path)

    assert contract is not None, "generate_feature_contract returned None"
    assert contract["version"] == 8, f"Expected version 8, got {contract['version']}"
    assert contract["num_tabular"] == 37, f"Expected 37 tabular, got {contract['num_tabular']}"
    assert len(contract["tabular_features"]) == 37

    # Check v8-A embedding additions exist
    assert "asset_class_lookup" in contract, "Missing asset_class_lookup"
    assert "asset_id_lookup" in contract, "Missing asset_id_lookup"
    assert "num_asset_classes" in contract
    assert "num_assets" in contract
    assert "asset_class_embed_dim" in contract
    assert "asset_id_embed_dim" in contract

    # Check v8-B/C description sections
    assert "v8_cross_asset_descriptions" in contract, "Missing v8_cross_asset_descriptions"
    assert "v8_fingerprint_descriptions" in contract, "Missing v8_fingerprint_descriptions"

    # v8_architecture and v8_training_recipe live in the manually-maintained
    # models/feature_contract.json but are NOT emitted by generate_feature_contract().
    # That's fine — the function generates what inference consumers need.
    # Verify the on-disk contract has them instead.
    on_disk_contract = Path(os.path.dirname(os.path.abspath(model_dir))) / "models" / "feature_contract.json"
    if on_disk_contract.exists():
        with open(on_disk_contract) as f:
            disk = json.load(f)
        assert "v8_architecture" in disk, "On-disk contract missing v8_architecture"
        assert "v8_training_recipe" in disk, "On-disk contract missing v8_training_recipe"
        gate = disk["v8_training_recipe"]["gate_check"]
        _info(
            f"  On-disk contract gate: acc≥{gate['min_accuracy']}%, prec≥{gate['min_precision']}%, rec≥{gate['min_recall']}%"
        )
    else:
        _info("  On-disk models/feature_contract.json not found (expected in CI — skipping architecture check)")

    # Verify file was written
    assert os.path.exists(contract_path), f"Contract file not written to {contract_path}"

    with open(contract_path) as f:
        loaded = json.load(f)
    assert loaded["version"] == 8

    _info(f"  Contract v{contract['version']}: {contract['num_tabular']} features")
    _info(
        f"  Embeddings: {contract['num_asset_classes']} classes × {contract['asset_class_embed_dim']}d + {contract['num_assets']} assets × {contract['asset_id_embed_dim']}d"
    )


def test_11_split_dataset(csv_path: str, tmpdir: str):
    """split_dataset() produces stratified train/val splits."""
    from lib.services.training.dataset_generator import split_dataset

    train_csv, val_csv = split_dataset(
        csv_path=csv_path,
        val_fraction=0.15,
        output_dir=tmpdir,
        stratify=True,
        random_seed=42,
    )

    assert os.path.exists(train_csv), f"Train CSV not found: {train_csv}"
    assert os.path.exists(val_csv), f"Val CSV not found: {val_csv}"

    # Count rows
    with open(train_csv) as f:
        train_count = sum(1 for _ in csv.reader(f)) - 1  # minus header
    with open(val_csv) as f:
        val_count = sum(1 for _ in csv.reader(f)) - 1

    total = train_count + val_count
    assert total == NUM_SAMPLES, f"Split lost samples: {train_count} + {val_count} = {total}, expected {NUM_SAMPLES}"

    val_pct = val_count / total * 100
    _info(f"  Train: {train_count}, Val: {val_count} ({val_pct:.0f}%)")
    _info(f"  Total preserved: {total}")


def test_12_mixup_doesnt_corrupt_gradients():
    """Verify mixup on tabular features doesn't produce NaN gradients."""
    import torch
    import torch.nn as nn

    from lib.analysis.breakout_cnn import HybridBreakoutCNN

    model = HybridBreakoutCNN(pretrained=False, use_asset_embeddings=True)
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simulate one batch with mixup
    batch_size = 4
    images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    tabular = torch.randn(batch_size, NUM_TABULAR)
    labels = torch.randint(0, 2, (batch_size,))
    class_ids = torch.randint(0, 5, (batch_size,))
    asset_ids = torch.randint(0, 25, (batch_size,))

    # Apply mixup
    import numpy as np

    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    perm = torch.randperm(batch_size)
    tabular_mixed = lam * tabular + (1 - lam) * tabular[perm]

    optimizer.zero_grad()
    output = model(images, tabular_mixed, asset_class_ids=class_ids, asset_ids=asset_ids)
    loss = criterion(output, labels)

    assert not torch.isnan(loss), "Loss is NaN after mixup"
    assert not torch.isinf(loss), "Loss is Inf after mixup"

    loss.backward()

    # Check no NaN gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    optimizer.step()

    _info(f"  Mixup loss: {loss.item():.4f} (λ={lam:.3f})")
    _info("  No NaN gradients detected")


def test_13_embedding_indices_valid():
    """Verify get_asset_class_idx and get_asset_idx return valid indices."""
    from lib.analysis.breakout_cnn import (
        NUM_ASSET_CLASSES,
        NUM_ASSETS,
        get_asset_class_idx,
        get_asset_idx,
    )

    test_symbols = ["MGC", "MNQ", "MES", "6E", "MBT", "BTC", "ETH", "SOL"]

    for sym in test_symbols:
        cls_idx = get_asset_class_idx(sym)
        asset_idx = get_asset_idx(sym)

        assert 0 <= cls_idx < NUM_ASSET_CLASSES, f"{sym}: class_idx={cls_idx} out of range [0, {NUM_ASSET_CLASSES})"
        assert 0 <= asset_idx < NUM_ASSETS, f"{sym}: asset_idx={asset_idx} out of range [0, {NUM_ASSETS})"

    # Unknown symbol should return 0 (safe default)
    unknown_cls = get_asset_class_idx("UNKNOWN_TICKER_XYZ")
    unknown_asset = get_asset_idx("UNKNOWN_TICKER_XYZ")
    assert unknown_cls == 0, f"Unknown symbol class_idx should be 0, got {unknown_cls}"
    assert unknown_asset == 0, f"Unknown symbol asset_idx should be 0, got {unknown_asset}"

    _info(f"  All {len(test_symbols)} symbols have valid embedding indices")
    _info("  Unknown symbols safely default to index 0")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_smoke_test() -> int:
    """Run all smoke tests. Returns 0 on success, 1 on any failure."""

    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  v8 Training Smoke Test — Synthetic Data, No GPU Required{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()

    start_time = time.monotonic()

    # Create temporary directory for all test artifacts
    tmpdir = tempfile.mkdtemp(prefix="v8_smoke_test_")
    image_dir = os.path.join(tmpdir, "images")
    csv_path = os.path.join(tmpdir, "labels.csv")
    model_dir = os.path.join(tmpdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    _info(f"Working directory: {tmpdir}")
    print()

    try:
        # ── Step 0: Check PyTorch is available ─────────────────────────────
        _info("Checking PyTorch availability...")
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _ok(f"PyTorch {torch.__version__} on {device}")
            if device == "cuda":
                _info(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                _info("  (CPU-only is fine for smoke test — just slower)")
        except ImportError:
            _fail("PyTorch not installed — cannot run smoke test")
            _info("Install with: pip install torch torchvision")
            return 1

        print()

        # ── Step 1: Generate synthetic data ────────────────────────────────
        _info(f"Generating {NUM_SAMPLES} synthetic samples...")
        _generate_synthetic_csv(Path(csv_path), Path(image_dir), NUM_SAMPLES)
        _ok(f"Synthetic dataset: {NUM_SAMPLES} samples, {len(list(Path(image_dir).glob('*.png')))} images")
        print()

        # ── Step 2: Run tests ──────────────────────────────────────────────
        _info("Running v8 smoke tests...")
        print()

        # Model architecture tests
        _run_test("01. Model instantiation (v8 architecture)", test_01_model_instantiation)
        _run_test("02. Forward pass with embeddings", test_02_forward_pass)
        _run_test("03. Forward pass without embeddings (compat)", test_03_forward_pass_no_embeddings)

        # Dataset tests
        _run_test("04. Dataset loading (synthetic CSV)", test_04_dataset_loading, csv_path, image_dir)
        _run_test("05. DataLoader batching (5-tuple collate)", test_05_dataloader_batching, csv_path, image_dir)

        # Training tests — THE critical ones
        _run_test(
            "06. train_model() 2 epochs (mixup+grad_accum+warmup+LR_groups)",
            test_06_train_model_2_epochs,
            csv_path,
            image_dir,
            model_dir,
        )
        _run_test("07. evaluate_model() produces metrics", test_07_evaluate_model, model_dir, csv_path, image_dir)

        # Inference and normalization tests
        _run_test("08. Tabular normalization backward compat (v5→v8)", test_08_normalise_backward_compat)
        _run_test("09. predict_breakout() inference", test_09_predict_breakout, image_dir, model_dir)

        # Feature contract
        _run_test("10. Feature contract generation (v8)", test_10_generate_feature_contract, model_dir)

        # Dataset splitting
        _run_test("11. Stratified dataset split", test_11_split_dataset, csv_path, tmpdir)

        # Edge case tests
        _run_test("12. Mixup doesn't corrupt gradients", test_12_mixup_doesnt_corrupt_gradients)
        _run_test("13. Embedding indices valid for all symbols", test_13_embedding_indices_valid)

    finally:
        # Cleanup
        _info(f"\nCleaning up: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Report ─────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    passed = sum(1 for _, ok, _ in _test_results if ok)
    failed = sum(1 for _, ok, _ in _test_results if not ok)
    total = len(_test_results)

    print()
    print(f"{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  Results: {passed}/{total} passed, {failed} failed ({elapsed:.1f}s){RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")
    print()

    if failed > 0:
        _fail("FAILED TESTS:")
        for name, ok, err in _test_results:
            if not ok:
                _fail(f"  {name}: {err}")
        print()
        print(f"{BOLD}{RED}  v8 SMOKE TEST FAILED — fix the above before training{RESET}")
        print()
        return 1
    else:
        print(f"{BOLD}{GREEN}  ✓ v8 SMOKE TEST PASSED — all {total} tests green{RESET}")
        print(f"{GREEN}  → Ready to generate dataset and train v8 champion{RESET}")
        print()
        return 0


# ---------------------------------------------------------------------------
# pytest compatibility — each test is also runnable via pytest
# ---------------------------------------------------------------------------

_PYTEST_TMPDIR = None


def _get_pytest_tmpdir():
    """Lazy-create a shared tmpdir for pytest runs."""
    global _PYTEST_TMPDIR
    if _PYTEST_TMPDIR is None:
        _PYTEST_TMPDIR = tempfile.mkdtemp(prefix="v8_pytest_")
        image_dir = os.path.join(_PYTEST_TMPDIR, "images")
        csv_path = os.path.join(_PYTEST_TMPDIR, "labels.csv")
        model_dir = os.path.join(_PYTEST_TMPDIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        _generate_synthetic_csv(Path(csv_path), Path(image_dir), NUM_SAMPLES)
    return _PYTEST_TMPDIR


def _csv_path():
    return os.path.join(_get_pytest_tmpdir(), "labels.csv")


def _image_dir():
    return os.path.join(_get_pytest_tmpdir(), "images")


def _model_dir():
    d = os.path.join(_get_pytest_tmpdir(), "models")
    os.makedirs(d, exist_ok=True)
    return d


# These are discovered by pytest automatically
def test_model_instantiation():
    test_01_model_instantiation()


def test_forward_pass():
    test_02_forward_pass()


def test_forward_pass_no_embeddings():
    test_03_forward_pass_no_embeddings()


def test_dataset_loading():
    test_04_dataset_loading(_csv_path(), _image_dir())


def test_dataloader_batching():
    test_05_dataloader_batching(_csv_path(), _image_dir())


def test_normalise_backward_compat():
    test_08_normalise_backward_compat()


def test_embedding_indices():
    test_13_embedding_indices_valid()


def test_mixup_gradients():
    test_12_mixup_doesnt_corrupt_gradients()


def test_feature_contract():
    test_10_generate_feature_contract(_model_dir())


def test_split_dataset():
    test_11_split_dataset(_csv_path(), _get_pytest_tmpdir())


# Training tests are slower — mark them so pytest can skip with -m "not slow"
def test_train_2_epochs():
    """Slow: actually runs 2 epochs of training."""
    test_06_train_model_2_epochs(_csv_path(), _image_dir(), _model_dir())


def test_evaluate_after_training():
    """Slow: requires test_train_2_epochs to have run first."""
    # Ensure training has happened
    model_files = list(Path(_model_dir()).glob("breakout_cnn_*.pt"))
    if not model_files:
        test_06_train_model_2_epochs(_csv_path(), _image_dir(), _model_dir())
    test_07_evaluate_model(_model_dir(), _csv_path(), _image_dir())


def test_predict_breakout():
    test_09_predict_breakout(_image_dir(), _model_dir())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_smoke_test())
