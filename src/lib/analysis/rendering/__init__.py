"""
lib.analysis.rendering — Chart rendering sub-package.

Re-exports the public API from both chart renderers so callers can use either:

    from lib.analysis.rendering import render_ruby_snapshot
    from lib.analysis.rendering.chart_renderer import render_ruby_snapshot
    from lib.analysis.rendering.chart_renderer_parity import render_parity_snapshot
"""

# chart_renderer — mplfinance/Pillow-based renderer
try:
    from lib.analysis.rendering.chart_renderer import (
        RenderConfig,
        cleanup_inference_images,
        render_batch_snapshots,
        render_ruby_snapshot,
        render_snapshot_for_inference,
    )
except ImportError:
    RenderConfig = None  # type: ignore[assignment,misc]
    render_ruby_snapshot = None  # type: ignore[assignment,misc]
    render_batch_snapshots = None  # type: ignore[assignment,misc]
    render_snapshot_for_inference = None  # type: ignore[assignment,misc]
    cleanup_inference_images = None  # type: ignore[assignment,misc]

# chart_renderer_parity — pure-Pillow renderer (parity with mplfinance output)
try:
    from lib.analysis.rendering.chart_renderer_parity import (
        ParityBar,
        compare_with_reference,
        compute_vwap_from_bars,
        dataframe_to_parity_bars,
        render_parity_batch,
        render_parity_snapshot,
        render_parity_to_file,
        render_parity_to_temp,
    )
except ImportError:
    ParityBar = None  # type: ignore[assignment,misc]
    render_parity_snapshot = None  # type: ignore[assignment,misc]
    render_parity_to_file = None  # type: ignore[assignment,misc]
    render_parity_to_temp = None  # type: ignore[assignment,misc]
    render_parity_batch = None  # type: ignore[assignment,misc]
    dataframe_to_parity_bars = None  # type: ignore[assignment,misc]
    compute_vwap_from_bars = None  # type: ignore[assignment,misc]
    compare_with_reference = None  # type: ignore[assignment,misc]

__all__ = [
    # chart_renderer
    "RenderConfig",
    "render_ruby_snapshot",
    "render_batch_snapshots",
    "render_snapshot_for_inference",
    "cleanup_inference_images",
    # chart_renderer_parity
    "ParityBar",
    "render_parity_snapshot",
    "render_parity_to_file",
    "render_parity_to_temp",
    "render_parity_batch",
    "dataframe_to_parity_bars",
    "compute_vwap_from_bars",
    "compare_with_reference",
]
