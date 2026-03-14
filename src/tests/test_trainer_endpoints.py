"""
Tests for Trainer Server HTTP endpoints in ``lib.services.training.trainer_server``.

Covers:
  1. GET  /health          → 200 with expected shape
  2. GET  /status          → valid JSON with expected fields
  3. POST /train           → 202 with valid params (training mocked)
  4. POST /train           → 409 when already busy
  5. GET  /train/validate  → dataset validation report
  6. GET  /models          → lists model files
  7. POST /train/cancel    → sets cancel flag
  8. GET  /logs            → returns log lines

All tests use ``httpx.AsyncClient`` against the FastAPI ``app`` from
trainer_server.  Heavy imports (torch, dataset_generator) are mocked so
tests run without GPU or real data.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure DISABLE_REDIS is set before any transitive imports try to connect
# ---------------------------------------------------------------------------
os.environ.setdefault("DISABLE_REDIS", "1")

# ---------------------------------------------------------------------------
# Stub out heavy / GPU-only dependencies that trainer_server transitively
# imports.  We must do this *before* importing the module under test.
# ---------------------------------------------------------------------------

# Provide a minimal torch stub so the module body doesn't crash on import
_torch_stub = types.ModuleType("torch")
_torch_stub.cuda = MagicMock()  # type: ignore[attr-defined]
_torch_stub.cuda.is_available = MagicMock(return_value=False)
_torch_stub.cuda.device_count = MagicMock(return_value=0)

_torchvision_stub = types.ModuleType("torchvision")
_torchaudio_stub = types.ModuleType("torchaudio")

# Only inject stubs if the real packages aren't installed
if "torch" not in sys.modules:
    sys.modules.setdefault("torch", _torch_stub)
if "torchvision" not in sys.modules:
    sys.modules.setdefault("torchvision", _torchvision_stub)
if "torchaudio" not in sys.modules:
    sys.modules.setdefault("torchaudio", _torchaudio_stub)

from typing import TYPE_CHECKING  # noqa: E402

import httpx  # noqa: E402
import pytest_asyncio  # noqa: E402

from lib.services.training.trainer_server import (  # noqa: E402
    TrainStatus,
    _state,
    app,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def client():
    """Async HTTP client bound to the trainer FastAPI app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset the global TrainState before each test so tests are isolated."""
    _state.reset()
    _state.last_result = None
    _state.error = None
    yield
    _state.reset()


# ---------------------------------------------------------------------------
# 1. GET /health → 200
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_shape(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "trainer"
        assert "uptime_seconds" in body
        assert isinstance(body["uptime_seconds"], (int, float))


# ---------------------------------------------------------------------------
# 2. GET /status → valid JSON with expected fields
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_200(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_has_expected_fields(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/status")
        body = resp.json()

        # Core state fields
        assert "status" in body
        assert "started_at" in body
        assert "finished_at" in body
        assert "progress" in body
        assert "cancel_requested" in body
        assert "last_result" in body
        assert "error" in body

        # Enriched fields added by the endpoint
        assert "gpu" in body
        assert "champion" in body
        assert "config" in body

    @pytest.mark.asyncio
    async def test_status_idle_by_default(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/status")
        body = resp.json()
        assert body["status"] == "idle"

    @pytest.mark.asyncio
    async def test_status_config_has_defaults(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/status")
        config = resp.json()["config"]
        assert "default_symbols" in config
        assert "default_days_back" in config
        assert "default_epochs" in config
        assert "default_batch_size" in config
        assert "default_session" in config

    @pytest.mark.asyncio
    async def test_status_gpu_info_present(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/status")
        gpu = resp.json()["gpu"]
        # At minimum, the "available" key must be present
        assert "available" in gpu


# ---------------------------------------------------------------------------
# 3. POST /train → 202 with valid params (mock actual training)
# ---------------------------------------------------------------------------


class TestTrainEndpoint:
    @pytest.mark.asyncio
    async def test_train_returns_202(self, client: httpx.AsyncClient) -> None:
        with patch(
            "lib.services.training.trainer_server._run_training_pipeline",
            side_effect=lambda params: _state.finish(result={"ok": True}),
        ):
            resp = await client.post("/train", json={"symbols": ["MGC"], "epochs": 5})
            assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_train_response_contains_message(self, client: httpx.AsyncClient) -> None:
        with patch(
            "lib.services.training.trainer_server._run_training_pipeline",
            side_effect=lambda params: _state.finish(result={"ok": True}),
        ):
            resp = await client.post("/train", json={"symbols": ["MGC"]})
            body = resp.json()
            assert "message" in body
            assert "Training started" in body["message"]

    @pytest.mark.asyncio
    async def test_train_response_echoes_params(self, client: httpx.AsyncClient) -> None:
        with patch(
            "lib.services.training.trainer_server._run_training_pipeline",
            side_effect=lambda params: _state.finish(result={"ok": True}),
        ):
            resp = await client.post("/train", json={"symbols": ["MGC", "MES"], "epochs": 10})
            body = resp.json()
            assert "params" in body
            assert body["params"]["symbols"] == ["MGC", "MES"]

    @pytest.mark.asyncio
    async def test_train_default_params_accepted(self, client: httpx.AsyncClient) -> None:
        """POST /train with empty body should use defaults and return 202."""
        with patch(
            "lib.services.training.trainer_server._run_training_pipeline",
            side_effect=lambda params: _state.finish(result={"ok": True}),
        ):
            resp = await client.post("/train")
            assert resp.status_code == 202


# ---------------------------------------------------------------------------
# 4. POST /train while busy → 409
# ---------------------------------------------------------------------------


class TestTrainBusy:
    @pytest.mark.asyncio
    async def test_train_while_busy_returns_409(self, client: httpx.AsyncClient) -> None:
        # Simulate an in-progress training run
        _state.set(TrainStatus.TRAINING, progress="epoch 3/60")

        resp = await client.post("/train", json={"symbols": ["MGC"]})
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_train_busy_error_message(self, client: httpx.AsyncClient) -> None:
        _state.set(TrainStatus.GENERATING, progress="loading bars")

        resp = await client.post("/train", json={})
        body = resp.json()
        assert "error" in body
        assert "already in progress" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_train_busy_includes_status(self, client: httpx.AsyncClient) -> None:
        _state.set(TrainStatus.EVALUATING)

        resp = await client.post("/train", json={})
        body = resp.json()
        assert "status" in body


# ---------------------------------------------------------------------------
# 5. GET /train/validate → dataset validation report
# ---------------------------------------------------------------------------


class TestValidateEndpoint:
    @pytest.mark.asyncio
    async def test_validate_returns_report(self, client: httpx.AsyncClient) -> None:
        mock_report = {
            "valid": True,
            "total_rows": 100,
            "columns": ["image_path", "label", "symbol"],
            "label_distribution": {1: 50, 0: 50},
            "symbols": ["MGC", "MES"],
            "missing_images": 0,
            "empty_image_paths": 0,
        }

        with patch(
            "lib.services.training.dataset_generator.validate_dataset",
            return_value=mock_report,
        ):
            resp = await client.get("/train/validate")
            assert resp.status_code == 200
            body = resp.json()
            assert body["valid"] is True
            assert body["total_rows"] == 100
            assert "coverage_pct" in body

    @pytest.mark.asyncio
    async def test_validate_missing_csv_returns_404(self, client: httpx.AsyncClient) -> None:
        mock_report = {
            "valid": False,
            "error": "CSV not found: /app/dataset/labels.csv",
        }

        with patch(
            "lib.services.training.dataset_generator.validate_dataset",
            return_value=mock_report,
        ):
            resp = await client.get("/train/validate")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_validate_partial_returns_206(self, client: httpx.AsyncClient) -> None:
        mock_report = {
            "valid": False,
            "total_rows": 50,
            "columns": ["image_path", "label"],
            "label_distribution": {1: 25, 0: 25},
            "symbols": [],
            "missing_images": 10,
            "empty_image_paths": 0,
            "error": "10 images not found on disk",
        }

        with patch(
            "lib.services.training.dataset_generator.validate_dataset",
            return_value=mock_report,
        ):
            resp = await client.get("/train/validate")
            assert resp.status_code == 206
            body = resp.json()
            assert body["valid"] is False
            assert body["coverage_pct"] == 80.0

    @pytest.mark.asyncio
    async def test_validate_coverage_pct_calculated(self, client: httpx.AsyncClient) -> None:
        mock_report = {
            "valid": True,
            "total_rows": 200,
            "columns": ["image_path", "label", "symbol"],
            "label_distribution": {},
            "symbols": [],
            "missing_images": 0,
            "empty_image_paths": 0,
        }

        with patch(
            "lib.services.training.dataset_generator.validate_dataset",
            return_value=mock_report,
        ):
            resp = await client.get("/train/validate")
            body = resp.json()
            assert body["coverage_pct"] == 100.0


# ---------------------------------------------------------------------------
# 6. GET /models → lists model files
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_list(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "breakout_cnn_best.pt").write_bytes(b"\x00" * 100)
        (models_dir / "breakout_cnn_20250101.pt").write_bytes(b"\x00" * 50)

        with (
            patch("lib.services.training.trainer_server.MODELS_DIR", models_dir),
            patch("lib.services.training.trainer_server.ARCHIVE_DIR", models_dir / "archive"),
            patch("lib.services.training.trainer_server.CHAMPION_META", models_dir / "breakout_cnn_best_meta.json"),
        ):
            resp = await client.get("/models")
            assert resp.status_code == 200
            body = resp.json()
            assert "models" in body
            assert isinstance(body["models"], list)
            assert len(body["models"]) == 2

    @pytest.mark.asyncio
    async def test_models_empty_dir(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with (
            patch("lib.services.training.trainer_server.MODELS_DIR", models_dir),
            patch("lib.services.training.trainer_server.ARCHIVE_DIR", models_dir / "archive"),
            patch("lib.services.training.trainer_server.CHAMPION_META", models_dir / "breakout_cnn_best_meta.json"),
        ):
            resp = await client.get("/models")
            body = resp.json()
            assert body["models"] == []

    @pytest.mark.asyncio
    async def test_models_includes_archive(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        archive_dir = models_dir / "archive"
        archive_dir.mkdir()
        (models_dir / "breakout_cnn_best.pt").write_bytes(b"\x00" * 100)
        (archive_dir / "breakout_cnn_20250601_120000.pt").write_bytes(b"\x00" * 80)

        with (
            patch("lib.services.training.trainer_server.MODELS_DIR", models_dir),
            patch("lib.services.training.trainer_server.ARCHIVE_DIR", archive_dir),
            patch("lib.services.training.trainer_server.CHAMPION_META", models_dir / "breakout_cnn_best_meta.json"),
        ):
            resp = await client.get("/models")
            body = resp.json()
            names = [m["name"] for m in body["models"]]
            assert "breakout_cnn_best.pt" in names
            assert "breakout_cnn_20250601_120000.pt" in names

            # The archive model should be marked as archive
            archive_entry = [m for m in body["models"] if m["name"] == "breakout_cnn_20250601_120000.pt"][0]
            assert archive_entry["archive"] is True

    @pytest.mark.asyncio
    async def test_models_champion_sorted_first(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "breakout_cnn_best.pt").write_bytes(b"\x00" * 100)
        (models_dir / "other_model.pt").write_bytes(b"\x00" * 50)

        with (
            patch("lib.services.training.trainer_server.MODELS_DIR", models_dir),
            patch("lib.services.training.trainer_server.ARCHIVE_DIR", models_dir / "archive"),
            patch("lib.services.training.trainer_server.CHAMPION_META", models_dir / "breakout_cnn_best_meta.json"),
        ):
            resp = await client.get("/models")
            body = resp.json()
            assert body["models"][0]["name"] == "breakout_cnn_best.pt"

    @pytest.mark.asyncio
    async def test_models_nonexistent_dir(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        models_dir = tmp_path / "nonexistent_models"

        with (
            patch("lib.services.training.trainer_server.MODELS_DIR", models_dir),
            patch("lib.services.training.trainer_server.ARCHIVE_DIR", models_dir / "archive"),
            patch("lib.services.training.trainer_server.CHAMPION_META", models_dir / "breakout_cnn_best_meta.json"),
        ):
            resp = await client.get("/models")
            assert resp.status_code == 200
            body = resp.json()
            assert body["models"] == []


# ---------------------------------------------------------------------------
# 7. POST /train/cancel → sets cancel flag
# ---------------------------------------------------------------------------


class TestCancelEndpoint:
    @pytest.mark.asyncio
    async def test_cancel_active_run(self, client: httpx.AsyncClient) -> None:
        _state.set(TrainStatus.TRAINING, progress="epoch 10/60")

        resp = await client.post("/train/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert "Cancellation requested" in body["message"]

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self, client: httpx.AsyncClient) -> None:
        _state.set(TrainStatus.GENERATING)

        await client.post("/train/cancel")
        assert _state.cancel_requested is True

    @pytest.mark.asyncio
    async def test_cancel_no_active_run_returns_409(self, client: httpx.AsyncClient) -> None:
        # State is idle (from autouse fixture)
        resp = await client.post("/train/cancel")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_already_done_returns_409(self, client: httpx.AsyncClient) -> None:
        _state.finish(result={"ok": True})

        resp = await client.post("/train/cancel")
        assert resp.status_code == 409
        body = resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# 8. GET /logs → returns log lines
# ---------------------------------------------------------------------------


class TestLogsEndpoint:
    @pytest.mark.asyncio
    async def test_logs_returns_200(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/logs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_logs_response_shape(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/logs")
        body = resp.json()
        assert "lines" in body
        assert "next_offset" in body
        assert "total" in body
        assert isinstance(body["lines"], list)

    @pytest.mark.asyncio
    async def test_logs_offset_parameter(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/logs", params={"offset": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["next_offset"], int)

    @pytest.mark.asyncio
    async def test_logs_next_offset_advances(self, client: httpx.AsyncClient) -> None:
        resp1 = await client.get("/logs", params={"offset": 0})
        body1 = resp1.json()
        offset1 = body1["next_offset"]

        # Second call with the returned offset should not error
        resp2 = await client.get("/logs", params={"offset": offset1})
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["next_offset"] >= offset1
