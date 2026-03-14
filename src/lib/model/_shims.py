"""
Compatibility shims for the model library.

Provides stubs for dependencies that exist in the reference codebase
(fks_python) but are not present in this project.  Allows model files
to import cleanly without requiring every external package.
"""

from __future__ import annotations

import functools
from typing import Any

from lib.core.logging_config import get_logger

# ---------------------------------------------------------------------------
# Logger — structured logging via structlog
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exception stubs
# ---------------------------------------------------------------------------


class ModelError(Exception):
    """Generic model-layer error (replaces core.exceptions.model.ModelError)."""


# ---------------------------------------------------------------------------
# Decorator stubs
# ---------------------------------------------------------------------------


def log_execution(func):
    """No-op decorator replacing utils.logging_utils.log_execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Missing constants / classes
# ---------------------------------------------------------------------------
DEFAULT_DEVICE = "cpu"
ModelEvaluator: Any = None
