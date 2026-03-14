"""
Phase 3B — Dashboard Focus Mode Tests
======================================
Tests for the focus mode rendering functions, API endpoints, and the
focus-mode grid layout logic added in Phase 3B.

Covers:
  - _get_daily_plan_data() Redis read
  - _render_daily_plan_header() Grok brief + focus chip summary
  - _render_why_these_assets() composite score breakdown
  - _render_swing_card() amber-styled swing candidate cards
  - _render_focus_mode_grid() tiered layout (scalp / swing / background)
  - _render_asset_card() focus badge integration
  - get_focus_html() focus mode rendering
  - get_daily_plan_html() endpoint
  - publish_focus_to_redis() daily plan PubSub event
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the src directory is importable
# ---------------------------------------------------------------------------
os.environ.setdefault("DISABLE_REDIS", "1")

_src = os.path.join(os.path.dirname(__file__), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Fixtures — synthetic focus data payloads
# ---------------------------------------------------------------------------


def _make_asset(
    symbol: str = "MGC",
    bias: str = "LONG",
    quality_pct: float = 72.0,
    wave_ratio: float = 1.35,
    last_price: float = 2650.0,
    focus_category: str = "",
    focus_rank: int = 999,
    has_live_position: bool = False,
    skip: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal asset dict matching the shape produced by compute_asset_focus."""
    asset: dict[str, Any] = {
        "symbol": symbol,
        "bias": bias,
        "bias_emoji": "🟢" if bias == "LONG" else "🔴" if bias == "SHORT" else "⚪",
        "last_price": last_price,
        "quality_pct": quality_pct,
        "quality": quality_pct / 100.0,
        "wave_ratio": wave_ratio,
        "vol_cluster": "MEDIUM",
        "vol_percentile": 0.5,
        "entry_low": last_price * 0.998,
        "entry_high": last_price * 1.002,
        "stop": last_price * 0.99,
        "tp1": last_price * 1.01,
        "tp2": last_price * 1.02,
        "position_size": 2,
        "risk_dollars": 150.0,
        "target1_dollars": 200.0,
        "target2_dollars": 400.0,
        "price_decimals": 2,
        "trend_direction": "LONG ↑",
        "dominance_text": "Bull",
        "market_phase": "TRENDING",
        "notes": "",
        "skip": skip,
        "focus_category": focus_category,
        "focus_rank": focus_rank,
        "has_live_position": has_live_position,
        "risk_blocked": False,
        "max_positions_reached": False,
        "dual_sizing": {},
    }
    asset.update(overrides)
    return asset


def _make_scalp_focus_data(name: str = "MGC", composite: float = 78.0) -> dict[str, Any]:
    return {
        "asset_name": name,
        "composite_score": composite,
        "signal_quality_score": 82.0,
        "atr_opportunity_score": 70.0,
        "rb_setup_density_score": 65.0,
        "session_fit_score": 80.0,
        "catalyst_score": 50.0,
        "bias_direction": "LONG",
        "bias_confidence": 0.72,
        "last_price": 2650.0,
        "atr": 12.5,
    }


def _make_swing_candidate_data(
    name: str = "MNQ",
    direction: str = "LONG",
    score: float = 85.0,
) -> dict[str, Any]:
    return {
        "asset_name": name,
        "direction": direction,
        "confidence": 0.78,
        "swing_score": score,
        "entry_zone_low": 18500.0,
        "entry_zone_high": 18550.0,
        "stop_loss": 18350.0,
        "tp1": 18700.0,
        "tp2": 18900.0,
        "tp3": 19200.0,
        "atr": 120.0,
        "last_price": 18520.0,
        "risk_dollars": 300.0,
        "position_size": 1,
        "reasoning": "Strong trend continuation with pullback entry",
        "entry_styles": ["pullback", "breakout"],
        "key_levels": {"vwap": 18480.0, "pdh": 18600.0},
    }


def _make_daily_plan(
    scalp_names: list[str] | None = None,
    swing_names: list[str] | None = None,
    market_context: str = "",
    grok_available: bool = False,
) -> dict[str, Any]:
    if scalp_names is None:
        scalp_names = ["MGC", "MNQ", "MES"]
    if swing_names is None:
        swing_names = ["MCL"]

    return {
        "scalp_focus": [_make_scalp_focus_data(n, 80 - i * 5) for i, n in enumerate(scalp_names)],
        "swing_candidates": [_make_swing_candidate_data(n) for n in swing_names],
        "scalp_focus_names": scalp_names,
        "swing_candidate_names": swing_names,
        "market_context": market_context,
        "grok_available": grok_available,
        "no_trade": False,
        "no_trade_reason": "",
        "all_biases": {},
        "computed_at": "2025-01-15T07:30:00-05:00",
        "account_size": 50_000,
        "session": "active",
        "total_scalp_focus": len(scalp_names),
        "total_swing_candidates": len(swing_names),
    }


def _make_focus_data(
    focus_mode_active: bool = True,
    scalp_focus_names: list[str] | None = None,
    swing_candidate_names: list[str] | None = None,
    daily_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scalp_focus_names is None:
        scalp_focus_names = ["MGC", "MNQ", "MES"]
    if swing_candidate_names is None:
        swing_candidate_names = ["MCL"]

    assets = []
    for i, name in enumerate(scalp_focus_names):
        assets.append(
            _make_asset(
                symbol=name,
                focus_category="scalp_focus",
                focus_rank=i + 1,
                quality_pct=75 - i * 3,
            )
        )
    for name in swing_candidate_names:
        assets.append(
            _make_asset(
                symbol=name,
                focus_category="swing",
                focus_rank=1,
                quality_pct=68.0,
                bias="SHORT",
            )
        )
    # Add some background assets
    for name in ["SIL", "HG"]:
        assets.append(
            _make_asset(
                symbol=name,
                focus_category="background",
                focus_rank=999,
                quality_pct=45.0,
            )
        )

    return {
        "assets": assets,
        "no_trade": False,
        "no_trade_reason": "",
        "computed_at": "2025-01-15T08:00:00-05:00",
        "account_size": 50_000,
        "session_mode": "active",
        "total_assets": len(assets),
        "tradeable_assets": len(assets),
        "live_risk_active": False,
        "live_positions": 0,
        "risk_blocked": "",
        "daily_plan": daily_plan or _make_daily_plan(scalp_focus_names, swing_candidate_names),
        "scalp_focus_names": scalp_focus_names,
        "swing_candidate_names": swing_candidate_names,
        "focus_mode_active": focus_mode_active,
    }


# ---------------------------------------------------------------------------
# Import the dashboard module (deferred so env vars are set first)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_redis():
    """Prevent any actual Redis calls during tests."""
    with patch("lib.core.cache.cache_get", return_value=None), patch("lib.core.cache.REDIS_AVAILABLE", False):
        yield


def _import_dashboard():
    """Lazily import dashboard module to avoid import-time Redis connections."""
    from lib.services.data.api.dashboard import (
        _get_daily_plan_data,
        _render_asset_card,
        _render_daily_plan_header,
        _render_focus_mode_grid,
        _render_swing_card,
        _render_why_these_assets,
    )

    return {
        "_get_daily_plan_data": _get_daily_plan_data,
        "_render_asset_card": _render_asset_card,
        "_render_daily_plan_header": _render_daily_plan_header,
        "_render_focus_mode_grid": _render_focus_mode_grid,
        "_render_swing_card": _render_swing_card,
        "_render_why_these_assets": _render_why_these_assets,
    }


# ===========================================================================
# Tests: _get_daily_plan_data
# ===========================================================================


class TestGetDailyPlanData:
    """Tests for reading daily plan from Redis."""

    def test_returns_none_when_no_data(self):
        fns = _import_dashboard()
        with patch("lib.core.cache.cache_get", return_value=None):
            result = fns["_get_daily_plan_data"]()
        assert result is None

    def test_returns_parsed_plan_from_redis(self):
        plan = _make_daily_plan()
        fns = _import_dashboard()
        with patch("lib.core.cache.cache_get", return_value=json.dumps(plan)):
            result = fns["_get_daily_plan_data"]()
        assert result is not None
        assert result["scalp_focus_names"] == ["MGC", "MNQ", "MES"]
        assert result["swing_candidate_names"] == ["MCL"]

    def test_returns_none_on_invalid_json(self):
        fns = _import_dashboard()
        with patch("lib.core.cache.cache_get", return_value="not-valid-json{{{"):
            result = fns["_get_daily_plan_data"]()
        assert result is None

    def test_returns_none_on_non_dict_json(self):
        fns = _import_dashboard()
        with patch("lib.core.cache.cache_get", return_value=json.dumps([1, 2, 3])):
            result = fns["_get_daily_plan_data"]()
        assert result is None

    def test_handles_cache_import_error(self):
        fns = _import_dashboard()
        with patch("lib.core.cache.cache_get", side_effect=ImportError("no cache")):
            result = fns["_get_daily_plan_data"]()
        assert result is None


# ===========================================================================
# Tests: _render_daily_plan_header
# ===========================================================================


class TestRenderDailyPlanHeader:
    """Tests for the daily plan header (Grok brief + focus chips)."""

    def test_empty_when_no_plan_and_no_focus_mode(self):
        fns = _import_dashboard()
        result = fns["_render_daily_plan_header"](None, None)
        assert result == ""

    def test_empty_when_focus_mode_inactive(self):
        fns = _import_dashboard()
        focus = _make_focus_data(focus_mode_active=False)
        result = fns["_render_daily_plan_header"](None, focus)
        assert result == ""

    def test_renders_focus_chips_without_grok(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(market_context="")
        focus = _make_focus_data()
        result = fns["_render_daily_plan_header"](plan, focus)

        assert "daily-plan-header" in result
        assert (
            "Today's Focus" in result
            or "Today&#x27;s Focus" in result
            or "Today\\'s Focus" in result
            or "Today" in result
        )
        # Check scalp focus chips are present
        assert "#1 MGC" in result
        assert "#2 MNQ" in result
        assert "#3 MES" in result
        # Swing chip
        assert "MCL" in result
        # No Grok brief
        assert "Grok Morning Brief" not in result

    def test_renders_grok_morning_brief(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(
            market_context="Risk-on today. ES leading. Gold pullback expected.",
            grok_available=True,
        )
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_daily_plan_header"](plan, focus)

        assert "Grok Morning Brief" in result
        assert "Risk-on today" in result
        assert "LIVE" in result  # LIVE badge for grok_available=True

    def test_grok_cached_badge_when_not_live(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(
            market_context="Some cached analysis",
            grok_available=False,
        )
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_daily_plan_header"](plan, focus)

        assert "CACHED" in result

    def test_long_grok_context_is_truncated(self):
        fns = _import_dashboard()
        long_text = "A" * 800
        plan = _make_daily_plan(market_context=long_text)
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_daily_plan_header"](plan, focus)

        # Should be truncated to 600 chars + "..."
        assert "..." in result
        assert "A" * 601 not in result

    def test_plan_time_is_formatted(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        plan["computed_at"] = "2025-01-15T07:30:00-05:00"
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_daily_plan_header"](plan, focus)

        # Should contain a formatted time like "07:30 AM ET"
        assert "Plan:" in result

    def test_renders_when_only_focus_data_has_mode_active(self):
        """Even without plan data, if focus_mode_active=True, should render."""
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_daily_plan_header"](None, focus)

        # Should still produce some output (the focus chip strip at minimum)
        assert "daily-plan-header" in result


# ===========================================================================
# Tests: _render_why_these_assets
# ===========================================================================


class TestRenderWhyTheseAssets:
    """Tests for the 'Why these assets?' scoring breakdown panel."""

    def test_empty_when_no_plan(self):
        fns = _import_dashboard()
        result = fns["_render_why_these_assets"](None)
        assert result == ""

    def test_empty_when_plan_has_no_selections(self):
        fns = _import_dashboard()
        plan = {"scalp_focus": [], "swing_candidates": []}
        result = fns["_render_why_these_assets"](plan)
        assert result == ""

    def test_renders_scalp_score_table(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        assert "why-these-assets" in result
        assert "Scalp Focus" in result
        assert "Composite Scores" in result
        # Asset names from the plan
        assert "MGC" in result
        assert "MNQ" in result
        assert "MES" in result
        # Column headers
        assert "Signal" in result
        assert "ATR" in result
        assert "RB Density" in result
        assert "Session" in result
        assert "Catalyst" in result
        assert "Bias" in result

    def test_renders_swing_candidates_section(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        assert "Swing Candidates" in result
        assert "MCL" in result

    def test_shows_bias_direction_and_confidence(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        # Should show LONG bias for scalp assets
        assert "LONG" in result

    def test_renders_swing_reasoning(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        assert "Strong trend continuation" in result

    def test_renders_swing_entry_styles(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        assert "pullback" in result
        assert "breakout" in result

    def test_is_collapsible_details_element(self):
        fns = _import_dashboard()
        plan = _make_daily_plan()
        result = fns["_render_why_these_assets"](plan)

        assert "<details" in result
        assert "<summary" in result
        assert "Why these assets?" in result

    def test_only_scalp_when_no_swing(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(swing_names=[])
        result = fns["_render_why_these_assets"](plan)

        assert "Scalp Focus" in result
        assert "Swing Candidates" not in result

    def test_only_swing_when_no_scalp(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(scalp_names=[])
        result = fns["_render_why_these_assets"](plan)

        assert "Swing Candidates" in result
        # No scalp table rendered
        assert "Composite Scores" not in result


# ===========================================================================
# Tests: _render_swing_card
# ===========================================================================


class TestRenderSwingCard:
    """Tests for swing candidate card rendering."""

    def test_renders_basic_swing_card(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL", bias="SHORT")
        swing_data = _make_swing_candidate_data("MCL", "SHORT", 85.0)
        result = fns["_render_swing_card"](asset, swing_data)

        assert "asset-card-mcl" in result
        assert "DAILY SWING" in result
        assert "MCL" in result

    def test_swing_card_shows_tp3(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        # TP3 should be present (swing-specific)
        assert "TP3" in result
        assert "19,200" in result

    def test_swing_card_shows_entry_styles(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        assert "pullback" in result
        assert "breakout" in result

    def test_swing_card_shows_reasoning(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        assert "Strong trend continuation" in result

    def test_swing_card_shows_score_and_confidence(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL", score=92.0)
        swing_data["confidence"] = 0.85
        result = fns["_render_swing_card"](asset, swing_data)

        assert "Score: 92" in result
        assert "85%" in result

    def test_swing_card_amber_border(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        assert "border-yellow-400" in result

    def test_swing_card_without_swing_data_falls_back(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL", bias="LONG")
        result = fns["_render_swing_card"](asset, None)

        assert "DAILY SWING" in result
        assert "MCL" in result
        # No TP3 when no swing_data
        assert "TP3" not in result

    def test_swing_card_live_position_overlay(self):
        fns = _import_dashboard()
        asset = _make_asset(
            symbol="MCL",
            has_live_position=True,
            live_position={
                "position_side": "LONG",
                "position_quantity": 2,
                "position_entry_price": 72.50,
                "position_current_price": 73.10,
                "position_stop_price": 71.80,
                "position_unrealized_pnl": 120.0,
                "position_r_multiple": 1.7,
                "position_bracket_phase": "TP1_HIT",
                "position_hold_duration": "45m",
                "position_source": "engine",
                "position_symbol": "MCL",
            },
        )
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        assert "LIVE" in result
        assert "+$120" in result

    def test_swing_card_has_focus_category_attribute(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        result = fns["_render_swing_card"](asset, swing_data)

        assert 'data-focus-category="swing"' in result

    def test_swing_card_uses_direction_from_swing_data(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL", bias="NEUTRAL")
        swing_data = _make_swing_candidate_data("MCL", direction="SHORT")
        result = fns["_render_swing_card"](asset, swing_data)

        assert "SHORT" in result


# ===========================================================================
# Tests: _render_asset_card focus badge integration
# ===========================================================================


class TestAssetCardFocusBadge:
    """Tests for the focus category badge on scalp focus asset cards."""

    def test_scalp_focus_badge_shown(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MGC", focus_category="scalp_focus", focus_rank=1)
        result = fns["_render_asset_card"](asset)

        assert "#1 FOCUS" in result

    def test_scalp_focus_badge_with_rank_2(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MNQ", focus_category="scalp_focus", focus_rank=2)
        result = fns["_render_asset_card"](asset)

        assert "#2 FOCUS" in result

    def test_swing_badge_shown(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL", focus_category="swing", focus_rank=1)
        result = fns["_render_asset_card"](asset)

        assert "SWING" in result

    def test_no_focus_badge_for_background(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="SIL", focus_category="background", focus_rank=999)
        result = fns["_render_asset_card"](asset)

        assert "FOCUS" not in result
        assert "SWING" not in result

    def test_no_focus_badge_when_category_empty(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="HG", focus_category="", focus_rank=999)
        result = fns["_render_asset_card"](asset)

        assert "FOCUS" not in result

    def test_focus_category_in_data_attribute(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MGC", focus_category="scalp_focus", focus_rank=1)
        result = fns["_render_asset_card"](asset)

        assert 'data-focus-category="scalp_focus"' in result

    def test_live_position_badge_takes_priority_over_focus(self):
        """When in a live position, the LIVE badge should still appear."""
        fns = _import_dashboard()
        asset = _make_asset(
            symbol="MGC",
            focus_category="scalp_focus",
            focus_rank=1,
            has_live_position=True,
            live_position={
                "position_side": "LONG",
                "position_quantity": 1,
                "position_entry_price": 2650.0,
                "position_current_price": 2660.0,
                "position_stop_price": 2640.0,
                "position_unrealized_pnl": 100.0,
                "position_r_multiple": 1.0,
                "position_bracket_phase": "INITIAL",
                "position_hold_duration": "10m",
                "position_source": "engine",
                "position_symbol": "MGC",
            },
        )
        result = fns["_render_asset_card"](asset)

        # Both badges should appear
        assert "LIVE" in result
        assert "#1 FOCUS" in result


# ===========================================================================
# Tests: _render_focus_mode_grid
# ===========================================================================


class TestRenderFocusModeGrid:
    """Tests for the complete focus mode grid layout."""

    def test_empty_data_shows_waiting_message(self):
        fns = _import_dashboard()
        result = fns["_render_focus_mode_grid"](None)
        assert "Waiting for engine" in result

    def test_empty_assets_shows_waiting_message(self):
        fns = _import_dashboard()
        result = fns["_render_focus_mode_grid"]({"assets": []})
        assert "Waiting for engine" in result

    def test_flat_grid_when_focus_mode_inactive(self):
        fns = _import_dashboard()
        focus = _make_focus_data(focus_mode_active=False)
        result = fns["_render_focus_mode_grid"](focus)

        # Should render flat cards without section headers
        assert "Scalp Focus" not in result
        assert "Daily Swing" not in result
        assert "Background Assets" not in result
        # But cards should still be present
        assert "asset-card-mgc" in result

    def test_focus_mode_shows_scalp_section_header(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "Scalp Focus" in result

    def test_focus_mode_shows_swing_section_header(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "Daily Swing Candidates" in result

    def test_focus_mode_shows_background_section(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "Background Assets" in result

    def test_background_section_is_collapsed(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        # Background assets are inside a <details> element
        assert "<details>" in result
        assert "Background Assets (2)" in result

    def test_scalp_assets_are_rendered(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "asset-card-mgc" in result
        assert "asset-card-mnq" in result
        assert "asset-card-mes" in result

    def test_swing_assets_use_swing_card_renderer(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        # Swing cards have the "DAILY SWING" badge
        assert "DAILY SWING" in result

    def test_background_assets_are_rendered_inside_details(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "asset-card-sil" in result
        assert "asset-card-hg" in result

    def test_daily_plan_header_rendered(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "daily-plan-header" in result

    def test_why_these_assets_rendered(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "why-these-assets" in result

    def test_asset_count_in_section_headers(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        assert "3 assets" in result  # 3 scalp
        assert "1 candidate" in result  # 1 swing

    def test_no_swing_section_when_no_swing_assets(self):
        fns = _import_dashboard()
        focus = _make_focus_data(swing_candidate_names=[])
        # Remove swing assets from the list
        focus["assets"] = [a for a in focus["assets"] if a["focus_category"] != "swing"]
        focus["daily_plan"]["swing_candidates"] = []
        result = fns["_render_focus_mode_grid"](focus)

        assert "Daily Swing Candidates" not in result

    def test_no_background_section_when_no_background_assets(self):
        fns = _import_dashboard()
        focus = _make_focus_data()
        # Remove background assets
        focus["assets"] = [a for a in focus["assets"] if a["focus_category"] != "background"]
        result = fns["_render_focus_mode_grid"](focus)

        assert "Background Assets" not in result

    def test_swing_data_lookup_used_for_levels(self):
        """Swing card levels should come from the plan's swing_candidate data."""
        fns = _import_dashboard()
        focus = _make_focus_data()
        result = fns["_render_focus_mode_grid"](focus)

        # The swing candidate data has TP3 = 19200
        # If swing_data is being used, TP3 should appear
        assert "TP3" in result

    def test_focus_mode_grid_with_grok_context(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(
            market_context="Bullish macro environment. Watch NQ for breakout.",
            grok_available=True,
        )
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_focus_mode_grid"](focus)

        assert "Grok Morning Brief" in result
        assert "Bullish macro" in result


# ===========================================================================
# Tests: publish_focus_to_redis daily plan PubSub event
# ===========================================================================


class TestPublishFocusDailyPlanEvent:
    """Tests for the Phase 3B daily plan PubSub event in publish_focus_to_redis."""

    def test_publishes_daily_plan_event_when_focus_mode_active(self):
        from lib.services.engine.focus import publish_focus_to_redis

        mock_redis = MagicMock()
        mock_redis.xadd = MagicMock()
        mock_redis.publish = MagicMock()

        focus_data = _make_focus_data()

        with (
            patch("lib.core.cache.REDIS_AVAILABLE", True),
            patch("lib.core.cache._r", mock_redis),
            patch("lib.core.cache.cache_set"),
        ):
            publish_focus_to_redis(focus_data)

        # Check that dashboard:daily_plan was published
        publish_calls = mock_redis.publish.call_args_list
        channels = [call[0][0] for call in publish_calls]
        assert "dashboard:daily_plan" in channels

        # Find the daily_plan publish call and verify payload
        for call in publish_calls:
            if call[0][0] == "dashboard:daily_plan":
                payload = json.loads(call[0][1])
                assert payload["focus_mode_active"] is True
                assert payload["scalp_focus_names"] == ["MGC", "MNQ", "MES"]
                assert payload["swing_candidate_names"] == ["MCL"]
                break

    def test_no_daily_plan_event_when_focus_mode_inactive(self):
        from lib.services.engine.focus import publish_focus_to_redis

        mock_redis = MagicMock()
        mock_redis.xadd = MagicMock()
        mock_redis.publish = MagicMock()

        focus_data = _make_focus_data(focus_mode_active=False)

        with (
            patch("lib.core.cache.REDIS_AVAILABLE", True),
            patch("lib.core.cache._r", mock_redis),
            patch("lib.core.cache.cache_set"),
        ):
            publish_focus_to_redis(focus_data)

        # Check that dashboard:daily_plan was NOT published
        publish_calls = mock_redis.publish.call_args_list
        channels = [call[0][0] for call in publish_calls]
        assert "dashboard:daily_plan" not in channels

    def test_no_daily_plan_event_when_no_plan_data(self):
        from lib.services.engine.focus import publish_focus_to_redis

        mock_redis = MagicMock()
        mock_redis.xadd = MagicMock()
        mock_redis.publish = MagicMock()

        focus_data = _make_focus_data()
        focus_data["daily_plan"] = None

        with (
            patch("lib.core.cache.REDIS_AVAILABLE", True),
            patch("lib.core.cache._r", mock_redis),
            patch("lib.core.cache.cache_set"),
        ):
            publish_focus_to_redis(focus_data)

        publish_calls = mock_redis.publish.call_args_list
        channels = [call[0][0] for call in publish_calls]
        assert "dashboard:daily_plan" not in channels


# ===========================================================================
# Tests: Focus mode summary bar badge
# ===========================================================================


class TestFocusModeSummaryBadge:
    """Tests for the Focus Mode badge in the summary bar."""

    def test_focus_mode_badge_in_full_dashboard(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data()
        session = {
            "mode": "active",
            "emoji": "🟢",
            "label": "US OPEN",
            "css_class": "text-green-400",
            "color_hex": "#4ade80",
            "time": "08:30:00",
            "date": "Wednesday, January 15, 2025",
            "time_et": "08:30:00 AM ET",
        }
        result = _render_full_dashboard(focus, session)

        assert "Focus Mode" in result

    def test_no_focus_mode_badge_when_inactive(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data(focus_mode_active=False)
        session = {
            "mode": "active",
            "emoji": "🟢",
            "label": "US OPEN",
            "css_class": "text-green-400",
            "color_hex": "#4ade80",
            "time": "08:30:00",
            "date": "Wednesday, January 15, 2025",
            "time_et": "08:30:00 AM ET",
        }
        result = _render_full_dashboard(focus, session)

        # The Focus Mode chip should not be present
        assert "🎯 Focus Mode" not in result


# ===========================================================================
# Tests: get_daily_plan_html endpoint
# ===========================================================================


class TestGetDailyPlanHtmlEndpoint:
    """Tests for the GET /api/daily-plan/html endpoint."""

    def test_returns_no_plan_message_when_empty(self):
        from lib.services.data.api.dashboard import get_daily_plan_html

        with (
            patch(
                "lib.services.data.api.dashboard._get_daily_plan_data",
                return_value=None,
            ),
            patch(
                "lib.services.data.api.dashboard._get_focus_data",
                return_value=None,
            ),
        ):
            response = get_daily_plan_html()

        assert response.status_code == 200
        body = response.body.decode()  # type: ignore[union-attr]
        assert "No daily plan active" in body

    def test_returns_plan_header_when_available(self):
        from lib.services.data.api.dashboard import get_daily_plan_html

        plan = _make_daily_plan(
            market_context="Risk-on. NQ leading.",
            grok_available=True,
        )
        focus = _make_focus_data(daily_plan=plan)

        with (
            patch(
                "lib.services.data.api.dashboard._get_daily_plan_data",
                return_value=plan,
            ),
            patch(
                "lib.services.data.api.dashboard._get_focus_data",
                return_value=focus,
            ),
        ):
            response = get_daily_plan_html()

        body = response.body.decode()  # type: ignore[union-attr]
        assert "daily-plan-panel" in body
        assert "Risk-on" in body
        assert "Grok Morning Brief" in body
        assert "why-these-assets" in body

    def test_returns_why_these_assets_in_plan(self):
        from lib.services.data.api.dashboard import get_daily_plan_html

        plan = _make_daily_plan()
        focus = _make_focus_data(daily_plan=plan)

        with (
            patch(
                "lib.services.data.api.dashboard._get_daily_plan_data",
                return_value=plan,
            ),
            patch(
                "lib.services.data.api.dashboard._get_focus_data",
                return_value=focus,
            ),
        ):
            response = get_daily_plan_html()

        body = response.body.decode()  # type: ignore[union-attr]
        assert "MGC" in body
        assert "Composite Scores" in body


# ===========================================================================
# Tests: get_focus_html with focus mode
# ===========================================================================


class TestGetFocusHtmlFocusMode:
    """Tests for get_focus_html rendering with focus mode active."""

    def test_focus_html_uses_focus_mode_grid(self):
        from lib.services.data.api.dashboard import get_focus_html

        focus = _make_focus_data()

        mock_request = MagicMock()
        mock_request.headers = {"HX-Request": "false"}

        with patch(
            "lib.services.data.api.dashboard._get_focus_data",
            return_value=focus,
        ):
            response = get_focus_html(mock_request)

        body = response.body.decode()  # type: ignore[union-attr]
        # Focus mode grid features
        assert "Scalp Focus" in body
        assert "Daily Swing Candidates" in body
        assert "Background Assets" in body

    def test_focus_html_flat_when_mode_inactive(self):
        from lib.services.data.api.dashboard import get_focus_html

        focus = _make_focus_data(focus_mode_active=False)

        mock_request = MagicMock()
        mock_request.headers = {"HX-Request": "false"}

        with patch(
            "lib.services.data.api.dashboard._get_focus_data",
            return_value=focus,
        ):
            response = get_focus_html(mock_request)

        body = response.body.decode()  # type: ignore[union-attr]
        # Flat mode — no section headers
        assert "Scalp Focus" not in body
        # But asset cards are present
        assert "asset-card-mgc" in body

    def test_focus_html_204_on_htmx_with_no_data(self):
        from lib.services.data.api.dashboard import get_focus_html

        mock_request = MagicMock()
        mock_request.headers = {"HX-Request": "true"}

        with patch(
            "lib.services.data.api.dashboard._get_focus_data",
            return_value=None,
        ):
            response = get_focus_html(mock_request)

        assert response.status_code == 204


# ===========================================================================
# Tests: Edge cases and robustness
# ===========================================================================


class TestFocusModeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_assets_are_scalp_focus(self):
        fns = _import_dashboard()
        focus = _make_focus_data(
            scalp_focus_names=["MGC", "MNQ", "MES", "MCL"],
            swing_candidate_names=[],
        )
        # Make all assets scalp_focus
        for i, asset in enumerate(focus["assets"]):
            asset["focus_category"] = "scalp_focus"
            asset["focus_rank"] = i + 1
        focus["daily_plan"]["swing_candidates"] = []
        result = fns["_render_focus_mode_grid"](focus)

        assert "Scalp Focus" in result
        assert "Daily Swing" not in result
        assert "Background Assets" not in result

    def test_all_assets_are_background(self):
        fns = _import_dashboard()
        focus = _make_focus_data(
            scalp_focus_names=[],
            swing_candidate_names=[],
        )
        for asset in focus["assets"]:
            asset["focus_category"] = "background"
            asset["focus_rank"] = 999
        focus["daily_plan"]["scalp_focus"] = []
        focus["daily_plan"]["swing_candidates"] = []

        # Still active focus mode but everything is background
        result = fns["_render_focus_mode_grid"](focus)

        assert "Background Assets" in result

    def test_single_scalp_asset_singular_label(self):
        fns = _import_dashboard()
        focus = _make_focus_data(
            scalp_focus_names=["MGC"],
            swing_candidate_names=[],
        )
        focus["assets"] = [a for a in focus["assets"] if a["focus_category"] != "swing"]
        result = fns["_render_focus_mode_grid"](focus)

        # "1 asset" not "1 assets"
        assert "1 asset" in result

    def test_swing_card_with_zero_tp3(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        swing_data["tp3"] = 0
        result = fns["_render_swing_card"](asset, swing_data)

        # No TP3 cell when tp3 is 0
        assert "TP3" not in result

    def test_swing_card_with_no_entry_styles(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        swing_data["entry_styles"] = []
        result = fns["_render_swing_card"](asset, swing_data)

        # Card should still render without entry style chips
        assert "MCL" in result
        assert "DAILY SWING" in result

    def test_swing_card_with_no_reasoning(self):
        fns = _import_dashboard()
        asset = _make_asset(symbol="MCL")
        swing_data = _make_swing_candidate_data("MCL")
        swing_data["reasoning"] = ""
        result = fns["_render_swing_card"](asset, swing_data)

        assert "MCL" in result

    def test_daily_plan_header_with_no_names(self):
        fns = _import_dashboard()
        plan = _make_daily_plan(scalp_names=[], swing_names=[])
        focus = _make_focus_data(
            scalp_focus_names=[],
            swing_candidate_names=[],
        )
        result = fns["_render_daily_plan_header"](plan, focus)

        # Should still render the header (just without chips)
        assert "daily-plan-header" in result

    def test_asset_card_focus_rank_999_no_badge(self):
        """Rank 999 should not show a focus badge even if category is scalp_focus."""
        fns = _import_dashboard()
        asset = _make_asset(symbol="MGC", focus_category="scalp_focus", focus_rank=999)
        result = fns["_render_asset_card"](asset)

        # Rank 999 means not really focused, no badge
        assert "FOCUS" not in result

    def test_focus_mode_grid_preserves_plan_data_in_header(self):
        """The plan header should use data from focus_data['daily_plan']."""
        fns = _import_dashboard()
        plan = _make_daily_plan(
            market_context="Custom grok analysis here",
            grok_available=True,
        )
        focus = _make_focus_data(daily_plan=plan)
        result = fns["_render_focus_mode_grid"](focus)

        assert "Custom grok analysis here" in result


# ===========================================================================
# Tests: Integration — full dashboard with focus mode
# ===========================================================================


class TestFullDashboardFocusMode:
    """Integration tests for the full dashboard with focus mode active."""

    def _make_session(self) -> dict[str, str]:
        return {
            "mode": "active",
            "emoji": "🟢",
            "label": "US OPEN",
            "css_class": "text-green-400",
            "color_hex": "#4ade80",
            "time": "08:30:00",
            "date": "Wednesday, January 15, 2025",
            "time_et": "08:30:00 AM ET",
        }

    def test_full_dashboard_contains_focus_mode_grid(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data()
        html = _render_full_dashboard(focus, self._make_session())

        assert "Scalp Focus" in html
        assert "Daily Swing Candidates" in html
        assert "Background Assets" in html
        assert "daily-plan-header" in html

    def test_full_dashboard_without_focus_mode_is_flat(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data(focus_mode_active=False)
        html = _render_full_dashboard(focus, self._make_session())

        # Cards rendered but no section headers
        assert "asset-card-mgc" in html
        assert "Scalp Focus" not in html

    def test_full_dashboard_no_data(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        html = _render_full_dashboard(None, self._make_session())

        assert "Waiting for engine" in html

    def test_sse_daily_plan_listener_in_js(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data()
        html = _render_full_dashboard(focus, self._make_session())

        # The SSE listener for daily-plan-update should be in the JS
        assert "daily-plan-update" in html
        assert "Daily plan updated" in html

    def test_focus_mode_badge_in_summary_bar(self):
        from lib.services.data.api.dashboard import _render_full_dashboard

        focus = _make_focus_data()
        html = _render_full_dashboard(focus, self._make_session())

        assert "Focus Mode" in html
