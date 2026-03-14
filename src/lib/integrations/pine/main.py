"""
Pine Script Generator — package entry point.

This module provides the PineScriptGenerator for use by the API router.
The generator is served via the HTMX-based page in the data service.
"""

from __future__ import annotations

import logging
import os

from lib.integrations.pine.generate import PineScriptGenerator

logger = logging.getLogger("pine")

# Package-level paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("PINE_OUTPUT_DIR", os.path.join(BASE_DIR, "pine_output"))

# Singleton generator instance
_generator: PineScriptGenerator | None = None


def get_generator() -> PineScriptGenerator:
    """Return the singleton PineScriptGenerator, creating it on first call."""
    global _generator  # noqa: PLW0603
    if _generator is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        _generator = PineScriptGenerator(BASE_DIR)
        logger.info("PineScriptGenerator initialised (base=%s)", BASE_DIR)
    return _generator


def reset_generator() -> None:
    """Force re-creation of the generator (e.g. after params.yaml edit)."""
    global _generator  # noqa: PLW0603
    _generator = None
