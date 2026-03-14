"""
Chart Renderer — Pillow-Based ORB Snapshot Generator
=====================================================

Renders 224×224 pixel chart images using Pillow with precise integer
coordinate math.  Produces dark-theme candlestick charts with an ORB
range box, VWAP line, and volume panel.

**Canvas layout:**
  - Canvas:    224 × 224 pixels
  - VolPanel:  40 pixels at the bottom
  - PriceH:    224 - 40 - 4 = 180 pixels (price chart area)
  - PriceTop:  4 pixels
  - LeftPad:   4 pixels
  - RightPad:  4 pixels
  - Background:    RGB(13, 13, 13)     = #0D0D0D
  - Bull candle:   RGB(38, 166, 154)   = #26A69A
  - Bear candle:   RGB(239, 83, 80)    = #EF5350
  - ORB fill:      RGBA(255, 215, 0, 40)
  - ORB border:    RGBA(255, 215, 0, 100)
  - VWAP line:     RGB(0, 229, 255)    = #00E5FF
  - Vol bull:      RGBA(38, 166, 154, 100)
  - Vol bear:      RGBA(239, 83, 80, 100)

**Render pipeline:**
  1. Fill background
  2. Compute price min/max from bar data + ORB levels
  3. Compute volume max
  4. Draw ORB fill rectangle + border lines
  5. Draw VWAP line across all bars
  6. Draw volume bars (bottom panel)
  7. Draw candlestick bodies + wicks

Public API:
    from chart_renderer_parity import (
        render_parity_snapshot,
        render_parity_to_temp,
        ParityBar,
    )

Dependencies:
    - Pillow >= 10.0.0 (already in project)
    - numpy (already in project)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("analysis.chart_renderer_parity")

if TYPE_CHECKING:
    from PIL import Image, ImageDraw

try:
    from PIL import Image, ImageDraw  # type: ignore[assignment]  # re-import at runtime

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow not installed — Pillow-based chart rendering disabled")


# ---------------------------------------------------------------------------
# Constants — canvas layout and color palette
# ---------------------------------------------------------------------------

W = 224
H = 224
VOL_PANEL_H = 40
PRICE_H = H - VOL_PANEL_H - 4  # 180
PRICE_TOP = 4
LEFT_PAD = 4
RIGHT_PAD = 4

# Colors — RGB/RGBA
BG_COLOR = (13, 13, 13)
BULL_CANDLE = (38, 166, 154)
BEAR_CANDLE = (239, 83, 80)
VWAP_LINE = (0, 229, 255)
VOL_BULL = (38, 166, 154, 100)
VOL_BEAR = (239, 83, 80, 100)

# ---------------------------------------------------------------------------
# Per-BreakoutType box colors
# ---------------------------------------------------------------------------
# Each entry is (fill_rgba, border_rgba, dashed).
# "dashed" True → drawn with on/off dash segments.
# "dashed" False → solid line.

# ORB — gold dashed
ORB_FILL = (255, 215, 0, 30)
ORB_BORDER = (255, 215, 0, 100)

# PrevDay — silver solid
PREV_DAY_FILL = (192, 192, 192, 20)
PREV_DAY_BORDER = (192, 192, 192, 120)

# InitialBalance — cyan dashed
IB_FILL = (0, 229, 255, 18)
IB_BORDER = (0, 229, 255, 110)

# Consolidation — purple solid
CONSOL_FILL = (147, 0, 211, 22)
CONSOL_BORDER = (147, 0, 211, 130)

# box_style token → (fill_rgba, border_rgba, dashed)
_BOX_STYLES: dict[str, tuple[tuple, tuple, bool]] = {
    "gold_dashed": (ORB_FILL, ORB_BORDER, True),
    "silver_solid": (PREV_DAY_FILL, PREV_DAY_BORDER, False),
    "blue_dashed": (IB_FILL, IB_BORDER, True),
    "purple_solid": (CONSOL_FILL, CONSOL_BORDER, False),
}
_DEFAULT_BOX_STYLE = "gold_dashed"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParityBar:
    """Single OHLCV bar (OHLCV values)."""

    open: float
    high: float
    low: float
    close: float
    volume: float

    def __init__(
        self,
        open: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        volume: float = 0.0,
    ):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _resolve_box_style(breakout_type: str | None) -> tuple[tuple, tuple, bool]:
    """Return ``(fill_rgba, border_rgba, dashed)`` for *breakout_type*.

    Accepts either a ``box_style`` token (e.g. ``"gold_dashed"``) or a
    breakout type name (e.g. ``"ORB"``, ``"PrevDay"``).  Falls back to
    ORB gold-dashed for unknown values.
    """
    if breakout_type is None:
        return _BOX_STYLES[_DEFAULT_BOX_STYLE]

    _bt = breakout_type.strip().lower()

    # Direct box_style token lookup
    for key, val in _BOX_STYLES.items():
        if _bt == key.lower():
            return val

    # BreakoutType name → box_style mapping
    _name_map = {
        "orb": "gold_dashed",
        "prevday": "silver_solid",
        "prev_day": "silver_solid",
        "initialbalance": "blue_dashed",
        "initial_balance": "blue_dashed",
        "consolidation": "purple_solid",
    }
    style_key = _name_map.get(_bt, _DEFAULT_BOX_STYLE)
    return _BOX_STYLES[style_key]


def _draw_dashed_hline(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    color: tuple,
    dash_on: int = 6,
    dash_off: int = 4,
) -> None:
    """Draw a horizontal dashed line on *draw* from x0 to x1 at row y.

    Args:
        draw:     Pillow ``ImageDraw`` instance.
        x0, x1:  Start and end x coordinates.
        y:        Row y coordinate.
        color:    RGBA or RGB color tuple.
        dash_on:  Number of pixels in the "on" (drawn) segment.
        dash_off: Number of pixels in the "off" (gap) segment.
    """
    x = x0
    cycle = dash_on + dash_off
    while x < x1:
        seg_end = min(x + dash_on, x1)
        draw.line([(x, y), (seg_end, y)], fill=color, width=1)
        x += cycle


def _price_to_y(price: float, price_min: float, price_range: float) -> int:
    """Map a price value to a Y pixel coordinate in the price panel.

    Uses integer truncation (matching ``int()`` for non-negative values):
        y = PRICE_TOP + int((price_max - price) / price_range * PRICE_H)
    """
    if price_range <= 0:
        return PRICE_TOP + PRICE_H // 2
    # priceMax = price_min + price_range
    price_max = price_min + price_range
    y = PRICE_TOP + int((price_max - price) / price_range * PRICE_H)
    return y


def _vol_to_h(vol: float, vol_max: float) -> int:
    """Map a volume value to a bar height in the volume panel.

    Returns ``int(vol / vol_max * VOL_PANEL_H)``.
    """
    if vol_max <= 0:
        return 0
    return int(vol / vol_max * VOL_PANEL_H)


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------


def render_parity_snapshot(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    breakout_type: str | None = None,
) -> Image.Image | None:
    """Render a 224×224 chart image using Pillow.

    Args:
        bars: Sequence of ParityBar objects (OHLCV).
        orb_high: Opening range high level.
        orb_low: Opening range low level.
        vwap_values: Per-bar VWAP values (same length as bars).
                     If None, VWAP line is not drawn.
        direction: "long" or "short" (reserved for future use).
        breakout_type: Box style selector.  Accepts a breakout type name
                       (``"ORB"``, ``"PrevDay"``, ``"InitialBalance"``,
                       ``"Consolidation"``) or a ``box_style`` token
                       (``"gold_dashed"``, ``"silver_solid"``, etc.).
                       Defaults to ``"gold_dashed"`` (ORB) when None.

    Returns:
        PIL Image (224×224 RGB), or None if rendering failed.
    """
    if not _PIL_AVAILABLE:
        logger.warning("Pillow not available — cannot render chart")
        return None

    if not bars or len(bars) == 0:
        logger.warning("No bars provided — cannot render")
        return None

    n = len(bars)

    # ── Step 1: Create canvas and fill background ──────────────────────
    img = Image.new("RGB", (W, H), BG_COLOR)

    # RGBA overlay for semi-transparent elements (ORB fill, volume bars).
    # Composited onto the main image at the end.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_main = ImageDraw.Draw(img)
    draw_overlay = ImageDraw.Draw(overlay)

    # ── Step 2: Compute price range from bars + ORB levels ─────────────
    price_min = min(b.low for b in bars)
    price_max = max(b.high for b in bars)
    price_min = min(price_min, orb_low)
    price_max = max(price_max, orb_high)
    price_range = price_max - price_min
    if price_range <= 0:
        price_range = 1.0

    # ── Step 3: Compute volume max ─────────────────────────────────────
    vol_max = max(b.volume for b in bars)
    if vol_max <= 0:
        vol_max = 1.0

    # ── Step 4: Compute bar geometry ───────────────────────────────────
    usable_w = W - LEFT_PAD - RIGHT_PAD
    bar_w = max(1, usable_w // n)
    body_w = max(1, bar_w - 2)

    # ── Step 5: Resolve box colors for this BreakoutType ───────────────
    box_fill, box_border, box_dashed = _resolve_box_style(breakout_type)

    # ── Step 6: Draw range fill rectangle ──────────────────────────────
    orb_y_top = _price_to_y(orb_high, price_min, price_range)
    orb_y_bot = _price_to_y(orb_low, price_min, price_range)

    if orb_y_bot > orb_y_top:
        draw_overlay.rectangle(
            [LEFT_PAD, orb_y_top, LEFT_PAD + usable_w - 1, orb_y_bot],
            fill=box_fill,
        )

    # ── Step 7: Draw range border lines ────────────────────────────────
    if box_dashed:
        # Dashed lines drawn directly on main image (not overlay) so the
        # gaps remain transparent rather than blending with the dark bg.
        _draw_dashed_hline(draw_main, LEFT_PAD, LEFT_PAD + usable_w, orb_y_top, box_border)
        _draw_dashed_hline(draw_main, LEFT_PAD, LEFT_PAD + usable_w, orb_y_bot, box_border)
    else:
        draw_overlay.line(
            [(LEFT_PAD, orb_y_top), (LEFT_PAD + usable_w, orb_y_top)],
            fill=box_border,
            width=1,
        )
        draw_overlay.line(
            [(LEFT_PAD, orb_y_bot), (LEFT_PAD + usable_w, orb_y_bot)],
            fill=box_border,
            width=1,
        )

    # ── Step 8: Draw VWAP line ─────────────────────────────────────────
    if vwap_values is not None and len(vwap_values) == n:
        for i in range(1, n):
            x0 = LEFT_PAD + (i - 1) * bar_w + bar_w // 2
            x1 = LEFT_PAD + i * bar_w + bar_w // 2
            y0 = _price_to_y(vwap_values[i - 1], price_min, price_range)
            y1 = _price_to_y(vwap_values[i], price_min, price_range)
            draw_main.line([(x0, y0), (x1, y1)], fill=VWAP_LINE, width=1)

    # ── Step 9: Draw volume bars ───────────────────────────────────────
    for i in range(n):
        x = LEFT_PAD + i * bar_w
        bar = bars[i]
        vh = _vol_to_h(bar.volume, vol_max)
        if vh <= 0:
            continue
        vy = H - vh
        vc = VOL_BULL if bar.is_bullish else VOL_BEAR
        draw_overlay.rectangle(
            [x, vy, x + body_w - 1, vy + vh - 1],
            fill=vc,
        )

    # ── Step 10: Draw candlesticks (bodies + wicks) ────────────────────
    for i in range(n):
        bar = bars[i]
        x = LEFT_PAD + i * bar_w
        x_mid = x + bar_w // 2
        is_bull = bar.is_bullish
        cc = BULL_CANDLE if is_bull else BEAR_CANDLE

        # Wick
        y_high = _price_to_y(bar.high, price_min, price_range)
        y_low = _price_to_y(bar.low, price_min, price_range)
        if y_low > y_high:
            draw_main.line([(x_mid, y_high), (x_mid, y_low)], fill=cc, width=1)

        # Body
        body_top_price = max(bar.open, bar.close)
        body_bot_price = min(bar.open, bar.close)
        y_body_top = _price_to_y(body_top_price, price_min, price_range)
        y_body_bot = _price_to_y(body_bot_price, price_min, price_range)
        body_h = max(1, y_body_bot - y_body_top)

        draw_main.rectangle(
            [x + 1, y_body_top, x + 1 + body_w - 1, y_body_top + body_h - 1],
            fill=cc,
        )

    # ── Step 11: Composite overlay onto main image ─────────────────────
    # Convert main image to RGBA for compositing, then back to RGB
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    img_final = img_rgba.convert("RGB")

    return img_final


# ---------------------------------------------------------------------------
# File-saving helpers
# ---------------------------------------------------------------------------


def render_parity_to_file(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    save_path: str = "",
    breakout_type: str | None = None,
) -> str | None:
    """Render and save a snapshot to a specific file path.

    Args:
        bars: Sequence of ParityBar (OHLCV).
        orb_high: Opening range high.
        orb_low: Opening range low.
        vwap_values: Per-bar VWAP values.
        direction: "long" or "short".
        save_path: Where to write the PNG.
        breakout_type: Box style selector — breakout type name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to the saved file, or None on failure.
    """
    img = render_parity_snapshot(bars, orb_high, orb_low, vwap_values, direction, breakout_type)
    if img is None:
        return None

    try:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path, format="PNG")
        return save_path
    except Exception as exc:
        logger.error("Failed to save snapshot to %s: %s", save_path, exc)
        return None


def render_parity_to_temp(
    bars: Sequence[ParityBar],
    orb_high: float,
    orb_low: float,
    vwap_values: Sequence[float] | None = None,
    direction: str | None = None,
    temp_dir: str | None = None,
    label: str = "parity",
    breakout_type: str | None = None,
) -> str | None:
    """Render a snapshot to a temporary file.

    Args:
        bars: Sequence of ParityBar (OHLCV).
        orb_high: Opening range high.
        orb_low: Opening range low.
        vwap_values: Per-bar VWAP values.
        direction: "long" or "short".
        temp_dir: Directory for temp files (uses system temp if None).
        label: Filename label prefix.
        breakout_type: Box style selector — breakout type name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to the saved temp PNG, or None on failure.
    """
    img = render_parity_snapshot(bars, orb_high, orb_low, vwap_values, direction, breakout_type)
    if img is None:
        return None

    try:
        dir_path = temp_dir or tempfile.gettempdir()
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{label}.png")
        img.save(path, format="PNG")
        return path
    except Exception as exc:
        logger.error("Failed to save temp snapshot: %s", exc)
        return None


# ---------------------------------------------------------------------------
# DataFrame adapter — converts pandas OHLCV to ParityBar list
# ---------------------------------------------------------------------------


def dataframe_to_parity_bars(
    df: Any,
    max_bars: int = 0,
) -> list[ParityBar]:
    """Convert a pandas DataFrame with OHLCV columns to a list of ParityBar.

    Accepts column names in any common capitalisation:
    Open/open, High/high, Low/low, Close/close, Volume/volume.

    Uses vectorised numpy extraction instead of ``iterrows()`` for a
    significant speedup (~50x) on typical 240-bar windows.

    Args:
        df: pandas DataFrame with OHLCV data.
        max_bars: If > 0, take only the last ``max_bars`` rows.

    Returns:
        List of ParityBar objects.
    """
    import numpy as np
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    # Normalise column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        low = str(col).lower()
        if low == "open":
            col_map["open"] = col
        elif low == "high":
            col_map["high"] = col
        elif low == "low":
            col_map["low"] = col
        elif low == "close":
            col_map["close"] = col
        elif low == "volume":
            col_map["volume"] = col

    required = {"open", "high", "low", "close"}
    if not required.issubset(col_map.keys()):
        logger.warning(
            "DataFrame missing required OHLC columns: %s (have: %s)",
            required - col_map.keys(),
            list(df.columns),
        )
        return []

    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:]

    has_volume = "volume" in col_map

    # ── Vectorised bulk extraction via numpy ──────────────────────────
    # to_numpy() is dramatically faster than iterrows() because it avoids
    # per-row Python object overhead.  astype(float) coerces in one pass
    # and raises on non-numeric data so we catch that below.
    try:
        opens = df[col_map["open"]].to_numpy(dtype=float, na_value=0.0)
        highs = df[col_map["high"]].to_numpy(dtype=float, na_value=0.0)
        lows = df[col_map["low"]].to_numpy(dtype=float, na_value=0.0)
        closes = df[col_map["close"]].to_numpy(dtype=float, na_value=0.0)
        volumes = (
            df[col_map["volume"]].to_numpy(dtype=float, na_value=0.0) if has_volume else np.zeros(len(df), dtype=float)
        )
    except (ValueError, TypeError) as exc:
        logger.warning("dataframe_to_parity_bars: vectorised extraction failed (%s) — falling back to iterrows", exc)
        # Fallback: row-by-row (original behaviour)
        bars: list[ParityBar] = []
        for _, row in df.iterrows():
            try:
                bars.append(
                    ParityBar(
                        open=float(row[col_map["open"]]),  # type: ignore[arg-type]
                        high=float(row[col_map["high"]]),  # type: ignore[arg-type]
                        low=float(row[col_map["low"]]),  # type: ignore[arg-type]
                        close=float(row[col_map["close"]]),  # type: ignore[arg-type]
                        volume=float(row[col_map["volume"]]) if has_volume else 0.0,  # type: ignore[arg-type]
                    )
                )
            except (ValueError, TypeError):
                continue
        return bars

    # Build ParityBar list from numpy arrays — one list-comprehension pass,
    # no per-row attribute lookups or pandas overhead.
    return [
        ParityBar(
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
        )
        for i in range(len(opens))
    ]


def compute_vwap_from_bars(bars: Sequence[ParityBar]) -> list[float]:
    """Compute a running VWAP series from ParityBar data.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3.

    Uses numpy cumsum for a vectorised implementation (~20x faster than
    the original Python loop on typical 240-bar windows).

    Returns a list of floats the same length as bars. If a bar has
    zero cumulative volume, the VWAP defaults to the bar's close.
    """
    if not bars:
        return []

    import numpy as np

    n = len(bars)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)

    for i, bar in enumerate(bars):
        highs[i] = bar.high
        lows[i] = bar.low
        closes[i] = bar.close
        volumes[i] = bar.volume

    typical = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical * volumes)
    cum_vol = np.cumsum(volumes)

    # Where cumulative volume is zero, fall back to close price
    vwap = np.where(cum_vol > 0, cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1.0), closes)

    return vwap.tolist()


# ---------------------------------------------------------------------------
# Batch rendering for dataset generation
# ---------------------------------------------------------------------------


def render_parity_batch(
    bars_df: Any,
    orb_high: float,
    orb_low: float,
    direction: str | None = None,
    save_path: str = "",
    max_bars: int = 60,
    compute_vwap: bool = True,
    breakout_type: str | None = None,
) -> str | None:
    """Convenience function: DataFrame → PNG file.

    Combines ``dataframe_to_parity_bars()``, optional VWAP computation,
    and ``render_parity_to_file()`` into a single call suitable for
    the dataset generator.

    Args:
        bars_df: pandas DataFrame with OHLCV columns.
        orb_high: Opening range high.
        orb_low: Opening range low.
        direction: "long" or "short".
        save_path: Output PNG path.
        max_bars: Maximum bars to include (tail).
        compute_vwap: If True, compute VWAP from bar data.
        breakout_type: Box style selector — breakout type name or
                       ``box_style`` token.  Defaults to ORB gold-dashed.

    Returns:
        Path to saved PNG, or None on failure.
    """
    bars = dataframe_to_parity_bars(bars_df, max_bars=max_bars)
    if not bars:
        return None

    vwap = compute_vwap_from_bars(bars) if compute_vwap else None

    return render_parity_to_file(
        bars=bars,
        orb_high=orb_high,
        orb_low=orb_low,
        vwap_values=vwap,
        direction=direction,
        save_path=save_path,
        breakout_type=breakout_type,
    )


# ---------------------------------------------------------------------------
# Validation: compare two rendered images
# ---------------------------------------------------------------------------


def compare_with_reference(
    img_a: Image.Image | str,
    img_b: Image.Image | str,
    tolerance: int = 3,
) -> dict[str, Any]:
    """Compare two rendered chart images and report pixel-level differences.

    Computes per-pixel absolute differences and reports statistics.
    Useful for regression testing the renderer.

    Args:
        img_a: PIL Image or path to first PNG.
        img_b: PIL Image or path to second PNG.
        tolerance: Maximum per-channel pixel difference considered a match.

    Returns:
        Dict with keys: match (bool), max_diff, mean_diff,
        mismatch_pixels, mismatch_pct, total_pixels.
    """
    if not _PIL_AVAILABLE:
        return {"error": "Pillow not available"}

    if isinstance(img_a, str):
        img_a = Image.open(img_a).convert("RGB")
    if isinstance(img_b, str):
        img_b = Image.open(img_b).convert("RGB")

    # Ensure same size
    if img_a.size != img_b.size:  # type: ignore[union-attr]
        return {
            "match": False,
            "error": f"Size mismatch: {img_a.size} vs {img_b.size}",  # type: ignore[union-attr]
        }

    arr_a = np.array(img_a, dtype=np.int16)
    arr_b = np.array(img_b, dtype=np.int16)

    diff = np.abs(arr_a - arr_b)
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    total_pixels = arr_a.shape[0] * arr_a.shape[1]
    # A pixel mismatches if ANY channel exceeds tolerance
    mismatch_mask = diff.max(axis=2) > tolerance
    mismatch_pixels = int(mismatch_mask.sum())
    mismatch_pct = mismatch_pixels / total_pixels * 100 if total_pixels > 0 else 0.0

    return {
        "match": max_diff <= tolerance,
        "max_diff": max_diff,
        "mean_diff": round(mean_diff, 3),
        "mismatch_pixels": mismatch_pixels,
        "mismatch_pct": round(mismatch_pct, 2),
        "total_pixels": total_pixels,
    }
