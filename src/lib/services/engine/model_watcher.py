"""
Model Watcher — filesystem-based hot-reload for CNN model files
================================================================
Uses the ``watchdog`` library for instant inotify/FSEvents/ReadDirectoryChanges
notifications when the champion model file changes on disk.  Falls back to
polling (stat-based) if watchdog is not installed or if the filesystem does
not support native events.

Usage (from the engine main loop)::

    from lib.services.engine.model_watcher import ModelWatcher

    watcher = ModelWatcher()
    watcher.start()
    # ... engine loop ...
    watcher.stop()

The watcher monitors the ``models/`` directory for:
  - ``breakout_cnn_best.pt``       — champion PyTorch checkpoint
  - ``breakout_cnn_best_meta.json`` — promotion metadata sidecar
  - ``feature_contract.json``       — feature/contract mapping

When any of these files are created, modified, or moved into place, the
watcher calls ``invalidate_model_cache()`` from ``lib.analysis.breakout_cnn``
so the next inference request loads the fresh model.

Thread-safe.  The watcher runs its own daemon thread and can be safely
started/stopped from the engine's main loop.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("engine.model_watcher")

# Files we care about in the models/ directory
_WATCHED_FILES = frozenset(
    {
        "breakout_cnn_best.pt",
        "breakout_cnn_best_meta.json",
        "feature_contract.json",
    }
)

# Debounce window — if multiple events arrive within this many seconds
# (e.g. atomic rename = delete + create), we only invalidate once.
_DEBOUNCE_SECONDS = 2.0

# Polling fallback interval when watchdog is not available
_POLL_INTERVAL = 15  # seconds (faster than the old 30s engine-loop polling)

# Candidate directories (Docker vs bare-metal)
_MODEL_DIR_CANDIDATES = [
    Path("/app/models"),
    Path(__file__).resolve().parents[4] / "models",
]


def _find_model_dir() -> Path | None:
    """Return the first existing models/ directory, or None."""
    for d in _MODEL_DIR_CANDIDATES:
        if d.is_dir():
            return d
    return None


def _invalidate_cache() -> None:
    """Call the breakout_cnn module's cache invalidation."""
    try:
        from lib.analysis.ml.breakout_cnn import invalidate_model_cache

        evicted = invalidate_model_cache()
        if evicted:
            logger.info("✅ CNN model cache invalidated — next inference will use the new model")
        else:
            logger.debug("CNN model cache was already empty — no eviction needed")
    except ImportError:
        logger.debug("breakout_cnn module not available — skipping cache invalidation")
    except Exception as exc:
        logger.warning("CNN cache invalidation failed (non-fatal): %s", exc)


def _publish_model_reload_event() -> None:
    """Publish a Redis event so the dashboard knows the model changed."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return

        import json
        from datetime import datetime
        from zoneinfo import ZoneInfo

        _EST = ZoneInfo("America/New_York")
        payload = json.dumps(
            {
                "event": "model_reloaded",
                "timestamp": datetime.now(tz=_EST).isoformat(),
            }
        )
        _r.publish("futures:events", payload)
        logger.debug("Published model_reloaded event to Redis")
    except Exception as exc:
        logger.debug("Failed to publish model reload event (non-fatal): %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Watchdog-based watcher (preferred)
# ═══════════════════════════════════════════════════════════════════════════

_WATCHDOG_AVAILABLE = False
_WatchdogHandler: Any = None
_WatchdogObserver: Any = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class _ModelFileHandler(FileSystemEventHandler):
        """Handle filesystem events for model files with debouncing."""

        def __init__(self) -> None:
            super().__init__()
            self._last_invalidation = 0.0
            self._lock = threading.Lock()

        def _should_handle(self, event) -> bool:
            """Return True if this event is for a watched model file."""
            if event.is_directory:
                return False
            src = Path(event.src_path).name
            # Also handle dest_path for move events
            dst = ""
            if hasattr(event, "dest_path") and event.dest_path:
                dst = Path(event.dest_path).name
            return src in _WATCHED_FILES or dst in _WATCHED_FILES

        def _debounced_invalidate(self, trigger: str) -> None:
            """Invalidate the model cache with debouncing."""
            now = time.monotonic()
            with self._lock:
                if now - self._last_invalidation < _DEBOUNCE_SECONDS:
                    logger.debug(
                        "Model watcher: debounced %s (%.1fs since last invalidation)",
                        trigger,
                        now - self._last_invalidation,
                    )
                    return
                self._last_invalidation = now

            logger.info("🔄 Model watcher detected change: %s", trigger)
            _invalidate_cache()
            _publish_model_reload_event()

        def on_created(self, event) -> None:
            if self._should_handle(event):
                src = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
                self._debounced_invalidate(f"created {Path(src).name}")

        def on_modified(self, event) -> None:
            if self._should_handle(event):
                src = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
                self._debounced_invalidate(f"modified {Path(src).name}")

        def on_moved(self, event) -> None:
            if self._should_handle(event):
                dst_raw = event.dest_path if event.dest_path else None
                if dst_raw is not None:
                    dst_str = dst_raw.decode() if isinstance(dst_raw, bytes) else dst_raw
                    dst_name = Path(dst_str).name
                else:
                    dst_name = "?"
                self._debounced_invalidate(f"moved → {dst_name}")

        def on_deleted(self, event) -> None:
            if self._should_handle(event):
                src = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
                logger.warning(
                    "⚠️  Model file deleted: %s — model will be unavailable until re-synced",
                    Path(src).name,
                )

    _WatchdogHandler = _ModelFileHandler
    _WatchdogObserver = Observer
    _WATCHDOG_AVAILABLE = True
    logger.debug("watchdog library available — will use native filesystem events")

except ImportError:
    logger.debug("watchdog library not installed — will use polling fallback")


# ═══════════════════════════════════════════════════════════════════════════
# Polling-based watcher (fallback)
# ═══════════════════════════════════════════════════════════════════════════


class _PollingWatcher:
    """Fallback watcher that polls file mtimes on a timer thread.

    Used when watchdog is not installed or when the filesystem doesn't
    support inotify (e.g. some network mounts, Docker on macOS with
    osxfs before virtiofs).
    """

    def __init__(self, model_dir: Path, interval: float = _POLL_INTERVAL) -> None:
        self._model_dir = model_dir
        self._interval = interval
        self._mtimes: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _snapshot_mtimes(self) -> dict[str, float]:
        """Return a dict of filename → mtime for all watched files."""
        result: dict[str, float] = {}
        for name in _WATCHED_FILES:
            p = self._model_dir / name
            if p.is_file():
                with contextlib.suppress(OSError):
                    result[name] = p.stat().st_mtime
        return result

    def _loop(self) -> None:
        """Polling loop that runs in a daemon thread."""
        # Record baseline mtimes
        self._mtimes = self._snapshot_mtimes()
        logger.info(
            "📂 Polling model watcher started: %s (every %ds, tracking %d files)",
            self._model_dir,
            self._interval,
            len(self._mtimes),
        )

        while not self._stop_event.wait(self._interval):
            current = self._snapshot_mtimes()
            changed = False

            for name, mtime in current.items():
                prev = self._mtimes.get(name)
                if prev is None:
                    # New file appeared
                    logger.info("🔄 Model watcher: new file detected: %s", name)
                    changed = True
                elif mtime != prev:
                    logger.info("🔄 Model watcher: file changed: %s", name)
                    changed = True

            if changed:
                self._mtimes = current
                _invalidate_cache()
                _publish_model_reload_event()
            else:
                self._mtimes = current

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="model-poll-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ═══════════════════════════════════════════════════════════════════════════
# Unified ModelWatcher API
# ═══════════════════════════════════════════════════════════════════════════


class ModelWatcher:
    """Unified model file watcher with automatic backend selection.

    Prefers watchdog (native OS events) when available, falls back to
    a polling-based watcher.  Both backends:
      - Run in a daemon thread
      - Debounce rapid file changes
      - Call ``invalidate_model_cache()`` when champion model changes
      - Publish a Redis event for the dashboard

    Usage::

        watcher = ModelWatcher()
        watcher.start()
        # ... run engine ...
        watcher.stop()
    """

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self._model_dir: Path | None
        if model_dir is not None:
            self._model_dir = Path(model_dir)
        else:
            self._model_dir = _find_model_dir()

        self._backend: Any = None
        self._backend_name: str = "none"

    def start(self) -> bool:
        """Start watching the models directory.

        Returns True if a watcher was started, False if models/ doesn't
        exist or watcher is already running.
        """
        if self._model_dir is None or not self._model_dir.is_dir():
            logger.warning(
                "⚠️  Cannot start model watcher — models/ directory not found. "
                "Run `bash scripts/sync_models.sh` to create it."
            )
            return False

        if self._backend is not None:
            logger.debug("Model watcher already running (%s)", self._backend_name)
            return True

        # Try watchdog first
        if _WATCHDOG_AVAILABLE and _WatchdogObserver is not None and _WatchdogHandler is not None:
            try:
                observer = _WatchdogObserver()
                handler = _WatchdogHandler()
                observer.schedule(handler, str(self._model_dir), recursive=False)
                observer.daemon = True
                observer.start()
                self._backend = observer
                self._backend_name = "watchdog"
                logger.info(
                    "👁️  Model watcher started (watchdog/inotify): %s",
                    self._model_dir,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "watchdog observer failed to start (%s) — falling back to polling",
                    exc,
                )

        # Fallback to polling
        poller = _PollingWatcher(self._model_dir)
        poller.start()
        self._backend = poller
        self._backend_name = "polling"
        logger.info(
            "👁️  Model watcher started (polling, %ds interval): %s",
            _POLL_INTERVAL,
            self._model_dir,
        )
        return True

    def stop(self) -> None:
        """Stop the watcher and release resources."""
        if self._backend is None:
            return

        try:
            if self._backend_name == "watchdog":
                self._backend.stop()
                self._backend.join(timeout=5)
            elif self._backend_name == "polling":
                self._backend.stop()
        except Exception as exc:
            logger.debug("Model watcher stop error (non-fatal): %s", exc)

        logger.info("Model watcher stopped (%s)", self._backend_name)
        self._backend = None
        self._backend_name = "none"

    @property
    def is_alive(self) -> bool:
        """Return True if the watcher backend is running."""
        if self._backend is None:
            return False
        if self._backend_name == "watchdog":
            return self._backend.is_alive()
        elif self._backend_name == "polling":
            return self._backend.is_alive
        return False

    @property
    def backend(self) -> str:
        """Return the name of the active backend ('watchdog', 'polling', or 'none')."""
        return self._backend_name

    @property
    def model_dir(self) -> Path | None:
        """Return the path being watched."""
        return self._model_dir

    def status(self) -> dict[str, Any]:
        """Return a status dict suitable for health checks / dashboard."""
        return {
            "running": self.is_alive,
            "backend": self._backend_name,
            "model_dir": str(self._model_dir) if self._model_dir else None,
            "watchdog_available": _WATCHDOG_AVAILABLE,
            "watched_files": sorted(_WATCHED_FILES),
            "debounce_seconds": _DEBOUNCE_SECONDS,
            "poll_interval": _POLL_INTERVAL if self._backend_name == "polling" else None,
        }
