"""
Chart Renderer — ORB Snapshot Generator
========================================
Generates chart images with a dark theme: colored candlesticks, ORB shaded
box, VWAP line, EMA9 overlay, volume panel, and quality badge.

These images serve two purposes:
  1. **CNN Training** — the breakout_cnn model learns from labeled chart
     snapshots of ORB setups.
  2. **Visual Validation** — the dashboard can display the snapshot the CNN
     scored, so you can eyeball whether the model is sane.

Dependencies:
  - mplfinance >= 0.12.10b0
  - matplotlib >= 3.7.0
  - pandas, numpy (already in project)

Public API:
    from chart_renderer import (
        render_ruby_snapshot,
        render_batch_snapshots,
        RenderConfig,
    )

    path = render_ruby_snapshot(
        bars_1m,
        symbol="MGC",
        orb_high=2345.6,
        orb_low=2332.0,
        direction="LONG",
        quality_pct=87,
    )
    # → "dataset/images/MGC_20260228_093500.png"

Design:
  - Pure function — no Redis, no side-effects beyond writing a PNG.
  - Configurable via RenderConfig dataclass (colors, DPI, figure size, etc.).
  - Graceful fallback if mplfinance is not installed (logs warning, returns None).
  - Thread-safe: each call creates its own figure and closes it.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from lib.indicators.helpers import ema as _ema_helper
from lib.indicators.helpers import vwap as _vwap_helper

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("analysis.chart_renderer")

# Guard import — mplfinance is an optional dependency for headless servers
try:
    import matplotlib
    import mplfinance as mpf

    matplotlib.use("Agg")  # headless backend — no GUI needed
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _MPF_AVAILABLE = True
except ImportError:
    _MPF_AVAILABLE = False
    logger.warning("mplfinance not installed — chart rendering disabled. Install with: pip install mplfinance")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    """Rendering configuration — dark theme color palette."""

    # Canvas
    figsize: tuple[float, float] = (14, 9)
    dpi: int = 180
    background_color: str = "#0F0F1A"
    panel_ratios: tuple[int, int] = (4, 1)

    # Candlestick colors
    candle_up: str = "#00FFAA"  # green-cyan for bullish
    candle_down: str = "#FF3366"  # hot pink for bearish
    candle_edge: str = "#FFFFFF"
    candle_wick: str = "#CCCCCC"

    # Overlay colors
    ema9_color: str = "#1E90FF"  # DodgerBlue
    vwap_color: str = "#FFD700"  # Gold
    orb_line_color: str = "#FFD700"  # Gold dashed
    orb_fill_alpha: float = 0.12
    orb_fill_color: str = "#FFD700"

    # Volume bar colors
    volume_up_color: str = "#00FFAA"
    volume_down_color: str = "#FF3366"

    # Quality badge
    badge_high_color: str = "#00FF00"  # lime for >= 70%
    badge_mid_color: str = "#FFD700"  # gold for 40-69%
    badge_low_color: str = "#FF3366"  # red for < 40%
    badge_font_size: int = 14
    badge_bg_color: str = "#000000"
    badge_bg_alpha: float = 0.8

    # Direction arrow
    arrow_long_color: str = "#00FFAA"
    arrow_short_color: str = "#FF3366"

    # Tight layout padding
    pad_inches: float = 0.1

    # mplfinance base style
    base_style: str = "nightclouds"

    # Output
    output_dir: str = "dataset/images"


# Singleton default config
DEFAULT_CONFIG = RenderConfig()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a proper DatetimeIndex for mplfinance."""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common column names
        for col in ("Date", "date", "Datetime", "datetime", "Timestamp", "timestamp", "time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        else:
            # Last resort: try to parse the existing index
            with contextlib.suppress(Exception):
                df.index = pd.to_datetime(df.index)

    # mplfinance requires the index name to be one of a few specific values
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "Date"

    return df


def _compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute EMA using pandas ewm (matches Ruby's EMA calculation)."""
    return _ema_helper(series, span)


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute session VWAP from OHLCV bars."""
    return _vwap_helper(df)


def _build_style(config: RenderConfig, direction: str | None = None) -> Any:
    """Build an mplfinance style matching the Ruby dark theme."""
    mc = mpf.make_marketcolors(  # type: ignore[possibly-unbound]
        up=config.candle_up,
        down=config.candle_down,
        edge=config.candle_edge,
        wick=config.candle_wick,
        volume={"up": config.volume_up_color, "down": config.volume_down_color},
        ohlc=config.candle_edge,
    )

    style = mpf.make_mpf_style(  # type: ignore[possibly-unbound]
        base_mpf_style=config.base_style,
        marketcolors=mc,
        facecolor=config.background_color,
        edgecolor=config.background_color,
        figcolor=config.background_color,
        gridcolor="#1A1A2E",
        gridstyle="--",
        gridaxis="both",
        y_on_right=True,
        rc={
            "font.size": 11,
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
    )

    return style


def _get_badge_color(quality_pct: int, config: RenderConfig) -> str:
    """Return badge color based on quality percentage."""
    if quality_pct >= 70:
        return config.badge_high_color
    elif quality_pct >= 40:
        return config.badge_mid_color
    return config.badge_low_color


def _generate_filename(
    symbol: str,
    timestamp: datetime | None = None,
    label: str | None = None,
    output_dir: str = "dataset/images",
) -> str:
    """Generate a unique filename for the chart image."""
    ts = timestamp or datetime.utcnow()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    parts = [symbol, ts_str]
    if label:
        parts.append(label)
    filename = "_".join(parts) + ".png"
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_ruby_snapshot(
    bars: pd.DataFrame,
    symbol: str = "",
    orb_high: float | None = None,
    orb_low: float | None = None,
    vwap: float | None = None,
    ema9: pd.Series | None = None,
    direction: str | None = None,
    quality_pct: int = 0,
    label: str | None = None,
    save_path: str | None = None,
    config: RenderConfig | None = None,
    extra_hlines: Sequence[float] | None = None,
    title_suffix: str = "",
) -> str | None:
    """Render a candlestick chart snapshot in Ruby indicator style.

    Produces a PNG image with:
      - OHLCV candlesticks (Ruby color scheme on dark background)
      - ORB range box (gold dashed lines, shaded fill)
      - VWAP line (gold, computed from bars if not provided)
      - EMA9 line (DodgerBlue, computed from bars if not provided)
      - Volume panel (colored by candle direction)
      - Quality badge (top-left, color-coded)
      - Direction label (LONG/SHORT)
      - Optional extra horizontal lines (e.g. support/resistance)

    Args:
        bars: OHLCV DataFrame (must have Open, High, Low, Close, Volume).
              DatetimeIndex preferred; will attempt conversion otherwise.
        symbol: Instrument symbol for the title (e.g. "MGC").
        orb_high: Opening range high — drawn as gold dashed line.
        orb_low: Opening range low — drawn as gold dashed line.
        vwap: Pre-computed VWAP value. If None, computed from bars.
              Pass a float for a flat line, or None for dynamic VWAP.
        ema9: Pre-computed EMA9 Series. If None, computed from Close.
        direction: "LONG" or "SHORT" (shown in title and affects accents).
        quality_pct: Signal quality percentage (0–100) for the badge.
        label: Optional label appended to filename (e.g. "good_long").
        save_path: Explicit output path. If None, auto-generated.
        config: RenderConfig override. If None, uses DEFAULT_CONFIG.
        extra_hlines: Additional horizontal price lines to draw.
        title_suffix: Extra text appended to the chart title.

    Returns:
        Path to the saved PNG file, or None if rendering failed.
    """
    if not _MPF_AVAILABLE:
        logger.warning("mplfinance not available — cannot render chart")
        return None

    cfg = config or DEFAULT_CONFIG

    # --- Validate and prepare data ---
    if bars is None or bars.empty:
        logger.warning("No bar data provided — cannot render chart")
        return None

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(bars.columns)
    if missing:
        logger.warning("Missing columns for chart render: %s", missing)
        return None

    if len(bars) < 5:
        logger.warning("Too few bars (%d) for meaningful chart", len(bars))
        return None

    df = _ensure_datetime_index(bars)

    # Drop rows with NaN OHLC (mplfinance will error)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        logger.warning("All bars have NaN OHLC — cannot render")
        return None

    # Ensure numeric types
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # Fill NaN volume with 0 (mplfinance tolerates it)
    df["Volume"] = df["Volume"].fillna(0)

    if df.empty:
        return None

    # --- Compute overlays ---
    addplots = []

    # EMA9
    _close_series: pd.Series = df["Close"]  # type: ignore[assignment]
    ema_series = ema9.reindex(df.index) if ema9 is not None else _compute_ema(_close_series, span=9)

    addplots.append(
        mpf.make_addplot(  # type: ignore[possibly-unbound]
            ema_series,
            color=cfg.ema9_color,
            width=2.0,
            label="EMA9",
        )
    )

    # VWAP (dynamic line or flat)
    if vwap is not None and isinstance(vwap, (int, float)):
        vwap_series = pd.Series(vwap, index=df.index, dtype=float)
    else:
        vwap_series = _compute_vwap(df)

    addplots.append(
        mpf.make_addplot(  # type: ignore[possibly-unbound]
            vwap_series,
            color=cfg.vwap_color,
            width=2.0,
            linestyle="-",
            label="VWAP",
        )
    )

    # ORB lines (horizontal)
    if orb_high is not None and orb_high > 0:
        orb_h_series = pd.Series(orb_high, index=df.index, dtype=float)
        addplots.append(
            mpf.make_addplot(  # type: ignore[possibly-unbound]
                orb_h_series,
                color=cfg.orb_line_color,
                width=1.5,
                linestyle="--",
                label="ORB High",
            )
        )

    if orb_low is not None and orb_low > 0:
        orb_l_series = pd.Series(orb_low, index=df.index, dtype=float)
        addplots.append(
            mpf.make_addplot(  # type: ignore[possibly-unbound]
                orb_l_series,
                color=cfg.orb_line_color,
                width=1.5,
                linestyle="--",
                label="ORB Low",
            )
        )

    # Extra horizontal lines (support/resistance, targets, etc.)
    if extra_hlines:
        for level in extra_hlines:
            hline_series = pd.Series(level, index=df.index, dtype=float)
            addplots.append(
                mpf.make_addplot(  # type: ignore[possibly-unbound]
                    hline_series,
                    color="#888888",
                    width=1.0,
                    linestyle=":",
                )
            )

    # --- Build title ---
    dir_str = f" {direction.upper()}" if direction else ""
    title = f"{symbol}{dir_str}  Q:{quality_pct}%"
    if title_suffix:
        title += f"  {title_suffix}"

    # --- Build style ---
    style = _build_style(cfg, direction=direction)

    # --- Output path ---
    if save_path is None:
        os.makedirs(cfg.output_dir, exist_ok=True)
        # Use the last bar's timestamp if available
        try:
            last_ts = pd.Timestamp(df.index[-1]).to_pydatetime()  # type: ignore[arg-type]
        except Exception:
            last_ts = datetime.utcnow()
        save_path = _generate_filename(symbol, last_ts, label=label, output_dir=cfg.output_dir)
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # --- Render ---
    try:
        fig, axes = mpf.plot(  # type: ignore[possibly-unbound]
            df,
            type="candle",
            volume=True,
            style=style,
            title=title,
            addplot=addplots if addplots else None,
            figsize=cfg.figsize,
            panel_ratios=cfg.panel_ratios,
            returnfig=True,
            tight_layout=True,
            scale_padding={"left": 0.05, "right": 0.1, "top": 0.25, "bottom": 0.1},
            warn_too_much_data=10_000,
        )

        ax_price = axes[0]

        # --- ORB shaded box ---
        if orb_high is not None and orb_low is not None and orb_high > orb_low:
            ax_price.axhspan(
                orb_low,
                orb_high,
                alpha=cfg.orb_fill_alpha,
                color=cfg.orb_fill_color,
                zorder=0,
            )

        # --- Quality badge (top-left corner) ---
        badge_color = _get_badge_color(quality_pct, cfg)
        ax_price.text(
            0.02,
            0.95,
            f"QUALITY {quality_pct}%",
            transform=ax_price.transAxes,
            fontsize=cfg.badge_font_size,
            fontweight="bold",
            color=badge_color,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=cfg.badge_bg_color,
                alpha=cfg.badge_bg_alpha,
                edgecolor=badge_color,
                linewidth=1.5,
            ),
        )

        # --- Direction label (top-right corner) ---
        if direction:
            dir_upper = direction.upper()
            dir_color = cfg.arrow_long_color if dir_upper == "LONG" else cfg.arrow_short_color
            arrow_char = "▲" if dir_upper == "LONG" else "▼"
            ax_price.text(
                0.98,
                0.95,
                f"{arrow_char} {dir_upper}",
                transform=ax_price.transAxes,
                fontsize=cfg.badge_font_size + 2,
                fontweight="bold",
                color=dir_color,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=cfg.badge_bg_color,
                    alpha=cfg.badge_bg_alpha,
                    edgecolor=dir_color,
                    linewidth=1.5,
                ),
            )

        # --- Legend for overlays (compact, top-center) ---
        legend_handles = [
            mpatches.Patch(color=cfg.ema9_color, label="EMA9"),  # type: ignore[possibly-unbound]
            mpatches.Patch(color=cfg.vwap_color, label="VWAP"),  # type: ignore[possibly-unbound]
        ]
        if orb_high is not None:
            legend_handles.append(mpatches.Patch(color=cfg.orb_line_color, label="ORB Range"))  # type: ignore[possibly-unbound]
        ax_price.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            fontsize=9,
            framealpha=0.6,
            facecolor=cfg.badge_bg_color,
            edgecolor="#333333",
            labelcolor="white",
        )

        # --- Save ---
        fig.savefig(
            save_path,
            dpi=cfg.dpi,
            bbox_inches="tight",
            pad_inches=cfg.pad_inches,
            facecolor=cfg.background_color,
        )
        plt.close(fig)  # type: ignore[possibly-unbound]

        logger.debug("Chart rendered: %s", save_path)
        return save_path

    except Exception as exc:
        logger.error("Chart rendering failed: %s", exc, exc_info=True)
        # Ensure we don't leak figures
        with contextlib.suppress(Exception):
            plt.close("all")  # type: ignore[possibly-unbound]
        return None


def render_batch_snapshots(
    bars_by_symbol: dict[str, pd.DataFrame],
    orb_results: dict[str, dict[str, Any]] | None = None,
    quality_by_symbol: dict[str, int] | None = None,
    config: RenderConfig | None = None,
    output_dir: str | None = None,
) -> dict[str, str | None]:
    """Render chart snapshots for multiple symbols.

    Convenience wrapper around ``render_ruby_snapshot`` for batch jobs
    (e.g. the nightly dataset generator or multi-asset live scan).

    Args:
        bars_by_symbol: Dict mapping symbol → OHLCV DataFrame.
        orb_results: Dict mapping symbol → {"or_high": float, "or_low": float,
                     "direction": str, ...} from the ORB detector.
        quality_by_symbol: Dict mapping symbol → quality_pct.
        config: RenderConfig override.
        output_dir: Override output directory.

    Returns:
        Dict mapping symbol → path to saved PNG (or None on failure).
    """
    cfg = config or DEFAULT_CONFIG
    if output_dir:
        cfg = RenderConfig(**{**cfg.__dict__, "output_dir": output_dir})

    results: dict[str, str | None] = {}

    for symbol, bars in bars_by_symbol.items():
        orb = (orb_results or {}).get(symbol, {})
        quality = (quality_by_symbol or {}).get(symbol, 0)

        path = render_ruby_snapshot(
            bars=bars,
            symbol=symbol,
            orb_high=orb.get("or_high"),
            orb_low=orb.get("or_low"),
            direction=orb.get("direction"),
            quality_pct=quality,
            config=cfg,
        )
        results[symbol] = path

    rendered = sum(1 for p in results.values() if p is not None)
    logger.info("Batch render complete: %d/%d symbols", rendered, len(bars_by_symbol))

    return results


def render_snapshot_for_inference(
    bars: pd.DataFrame,
    symbol: str,
    orb_high: float,
    orb_low: float,
    direction: str,
    quality_pct: int = 0,
    config: RenderConfig | None = None,
) -> str | None:
    """Render a snapshot specifically for CNN inference.

    This is a thin wrapper that stores images in a temporary inference
    subdirectory and uses a standardized naming convention so the
    inference pipeline can find and clean up images easily.

    Returns:
        Path to the saved PNG, or None on failure.
    """
    cfg = config or DEFAULT_CONFIG
    inference_dir = os.path.join(cfg.output_dir, "_inference")
    os.makedirs(inference_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(inference_dir, f"infer_{symbol}_{ts}.png")

    return render_ruby_snapshot(
        bars=bars,
        symbol=symbol,
        orb_high=orb_high,
        orb_low=orb_low,
        direction=direction,
        quality_pct=quality_pct,
        save_path=save_path,
        config=cfg,
    )


def cleanup_inference_images(max_age_seconds: int = 3600, config: RenderConfig | None = None) -> int:
    """Remove inference images older than ``max_age_seconds``.

    Called periodically by the scheduler to prevent disk bloat.

    Returns:
        Number of files deleted.
    """
    cfg = config or DEFAULT_CONFIG
    inference_dir = os.path.join(cfg.output_dir, "_inference")

    if not os.path.isdir(inference_dir):
        return 0

    deleted = 0
    now = datetime.utcnow().timestamp()

    for fname in os.listdir(inference_dir):
        fpath = os.path.join(inference_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            age = now - os.path.getmtime(fpath)
            if age > max_age_seconds:
                os.remove(fpath)
                deleted += 1
        except OSError:
            pass

    if deleted:
        logger.debug("Cleaned up %d old inference images", deleted)

    return deleted
