"""
Tests for the Grok Compact Live Update Output.

Covers:
  - format_live_compact() — local ≤8-line formatter
  - _enforce_compact_limit() — line truncation logic
  - _run_live_compact() — compact prompt builder (mocked API)
  - run_live_analysis() compact=True routing
  - GrokSession.run_update() with compact flag
  - Edge cases: empty assets, all skipped, single asset, many assets
  - Output format validation: emoji, bias status, DO NOW line
"""

from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
from lib.integrations.grok_helper import (
    DEFAULT_MAX_TOKENS_LIVE_COMPACT,
    GrokSession,
    _enforce_compact_limit,
    format_live_compact,
    run_live_analysis,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers: build focus asset dicts
# ---------------------------------------------------------------------------


def _asset(
    symbol: str = "MGC",
    quality_pct: float = 72.0,
    quality: float = 0.72,
    vol_percentile: float = 0.50,
    bias: str = "LONG",
    skip: bool = False,
    last_price: float = 2700.0,
    tp1: float = 2720.0,
    stop: float = 2680.0,
    entry_low: float = 2695.0,
    entry_high: float = 2705.0,
    **kwargs,
) -> dict[str, Any]:
    """Build a minimal focus asset dict for testing."""
    d = {
        "symbol": symbol,
        "quality": quality,
        "quality_pct": quality_pct,
        "vol_percentile": vol_percentile,
        "bias": bias,
        "skip": skip,
        "last_price": last_price,
        "tp1": tp1,
        "stop": stop,
        "entry_low": entry_low,
        "entry_high": entry_high,
    }
    d.update(kwargs)
    return d


# ===========================================================================
# Test: format_live_compact() — local formatter
# ===========================================================================


class TestFormatLiveCompact:
    def test_basic_output_structure(self):
        """Output should have status lines + blank line + DO NOW line."""
        assets = [_asset("MGC"), _asset("MNQ", bias="SHORT", last_price=21000.0)]
        result = format_live_compact(assets)
        lines = result.strip().split("\n")
        # Should have: 2 status lines, 1 blank, 1 DO NOW
        assert len(lines) >= 3
        assert any("DO NOW" in ln for ln in lines)

    def test_single_asset(self):
        assets = [_asset("MGC", quality_pct=72.0, skip=False)]
        result = format_live_compact(assets)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        # Should have 1 status line + 1 DO NOW line
        assert len(lines) >= 2
        assert "MGC" in lines[0]
        assert "DO NOW" in lines[-1]

    def test_max_8_lines(self):
        """Output should be ≤8 lines total."""
        assets = [
            _asset("MGC"),
            _asset("MNQ", bias="SHORT", last_price=21000.0),
            _asset("MES", last_price=5500.0),
            _asset("MCL", last_price=70.0),
            _asset("SIL", last_price=32.0),
        ]
        result = format_live_compact(assets)
        # Count non-empty lines + the blank separator
        all_lines = result.split("\n")
        assert len(all_lines) <= 8

    def test_long_bias_shows_green_emoji(self):
        assets = [_asset("MGC", bias="LONG")]
        result = format_live_compact(assets)
        assert "🟢" in result

    def test_short_bias_shows_red_emoji(self):
        assets = [_asset("MGC", bias="SHORT")]
        result = format_live_compact(assets)
        assert "🔴" in result

    def test_neutral_bias_shows_white_emoji(self):
        assets = [_asset("MGC", bias="NEUTRAL")]
        result = format_live_compact(assets)
        assert "⚪" in result

    def test_valid_bias_status_for_high_quality(self):
        """Quality >= 55% and not skip → Bias VALID."""
        assets = [_asset("MGC", quality_pct=72.0, skip=False)]
        result = format_live_compact(assets)
        assert "VALID" in result

    def test_invalid_bias_status_for_low_quality(self):
        """Quality < 55% → Bias INVALID."""
        assets = [_asset("MGC", quality_pct=40.0, skip=False)]
        result = format_live_compact(assets)
        assert "INVALID" in result

    def test_invalid_bias_status_for_skipped(self):
        """Skip=True → Bias INVALID."""
        assets = [_asset("MGC", quality_pct=72.0, skip=True)]
        result = format_live_compact(assets)
        assert "INVALID" in result

    def test_watch_level_tp1_for_valid(self):
        """For valid bias, Watch level should be TP1."""
        assets = [_asset("MGC", quality_pct=72.0, skip=False, tp1=2720.0)]
        result = format_live_compact(assets)
        assert "2,720.00" in result or "2720.00" in result

    def test_watch_level_stop_for_invalid(self):
        """For invalid bias, Watch level should be stop."""
        assets = [_asset("MGC", quality_pct=40.0, skip=False, stop=2680.0)]
        result = format_live_compact(assets)
        assert "2,680.00" in result or "2680.00" in result

    def test_contains_symbol_name(self):
        assets = [_asset("Gold (MGC)")]
        result = format_live_compact(assets)
        assert "Gold (MGC)" in result

    def test_contains_price(self):
        assets = [_asset("MGC", last_price=2712.50)]
        result = format_live_compact(assets)
        assert "2,712.50" in result or "2712.50" in result

    def test_do_now_auto_generated_no_tradeable(self):
        """When all assets are skipped, DO NOW says stand aside."""
        assets = [
            _asset("MGC", skip=True),
            _asset("MNQ", skip=True),
        ]
        result = format_live_compact(assets)
        assert "DO NOW" in result
        do_now_line = [ln for ln in result.split("\n") if "DO NOW" in ln]
        assert len(do_now_line) == 1
        assert "stand aside" in do_now_line[0].lower() or "wait" in do_now_line[0].lower()

    def test_do_now_auto_generated_single_tradeable(self):
        """Single tradeable asset → DO NOW references that specific asset."""
        assets = [
            _asset("MGC", skip=False, bias="LONG"),
            _asset("MNQ", skip=True),
        ]
        result = format_live_compact(assets)
        do_now_line = [ln for ln in result.split("\n") if "DO NOW" in ln][0]
        assert "MGC" in do_now_line
        assert "LONG" in do_now_line

    def test_do_now_auto_generated_multiple_tradeable(self):
        """Multiple tradeable → DO NOW says Watch ... for entries."""
        assets = [
            _asset("MGC", skip=False),
            _asset("MNQ", skip=False),
        ]
        result = format_live_compact(assets)
        do_now_line = [ln for ln in result.split("\n") if "DO NOW" in ln][0]
        assert "MGC" in do_now_line
        assert "MNQ" in do_now_line

    def test_custom_do_now(self):
        """Custom do_now string overrides auto-generation."""
        assets = [_asset("MGC")]
        result = format_live_compact(assets, do_now="DO NOW: Close all positions.")
        assert "DO NOW: Close all positions." in result

    def test_empty_assets(self):
        result = format_live_compact([])
        # Should still produce a DO NOW line
        assert "DO NOW" in result
        assert "stand aside" in result.lower() or "wait" in result.lower()

    def test_five_asset_limit(self):
        """At most 5 asset lines to stay within 8-line budget."""
        assets = [_asset(f"ASSET{i}", last_price=1000.0 + i) for i in range(8)]
        result = format_live_compact(assets)
        # Count status lines (non-blank, non-DO-NOW)
        lines = result.strip().split("\n")
        status_lines = [ln for ln in lines if ln.strip() and "DO NOW" not in ln.upper()]
        assert len(status_lines) <= 5

    def test_output_is_string(self):
        result = format_live_compact([_asset()])
        assert isinstance(result, str)


# ===========================================================================
# Test: _enforce_compact_limit()
# ===========================================================================


class TestEnforceCompactLimit:
    def test_short_text_unchanged(self):
        text = "MGC 🟢 2712 | Bias VALID | Watch 2725\n\nDO NOW: Hold."
        result = _enforce_compact_limit(text)
        assert result == text.strip()

    def test_long_text_truncated(self):
        lines = [f"ASSET{i} 🟢 {1000 + i} | Bias VALID | Watch {1020 + i}" for i in range(10)]
        lines.append("")
        lines.append("DO NOW: Do something.")
        text = "\n".join(lines)
        result = _enforce_compact_limit(text, max_lines=8)
        output_lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(output_lines) <= 8
        assert "DO NOW" in result

    def test_preserves_do_now_line(self):
        lines = [f"L{i}" for i in range(12)]
        lines.append("DO NOW: Exit immediately.")
        text = "\n".join(lines)
        result = _enforce_compact_limit(text, max_lines=8)
        assert "DO NOW: Exit immediately." in result

    def test_exactly_at_limit(self):
        lines = ["L1", "L2", "L3", "L4", "L5", "", "DO NOW: Hold."]
        text = "\n".join(lines)
        result = _enforce_compact_limit(text, max_lines=8)
        # 7 lines is within 8 limit, should be unchanged
        assert result == text.strip()

    def test_no_do_now_line_fallback_truncation(self):
        lines = [f"Line {i}" for i in range(15)]
        text = "\n".join(lines)
        result = _enforce_compact_limit(text, max_lines=5)
        output_lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(output_lines) <= 5

    def test_custom_max_lines(self):
        lines = ["A", "B", "C", "D", "", "DO NOW: X"]
        text = "\n".join(lines)
        result = _enforce_compact_limit(text, max_lines=3)
        output_lines = [ln for ln in result.split("\n") if ln.strip()]
        # Should have at most 3 content lines: 1 status + DO NOW
        assert len(output_lines) <= 3

    def test_whitespace_only_lines_handled(self):
        text = "MGC 🟢 2712\n  \n  \nDO NOW: Hold."
        result = _enforce_compact_limit(text, max_lines=8)
        assert "DO NOW" in result

    def test_empty_input(self):
        result = _enforce_compact_limit("", max_lines=8)
        assert result == ""

    def test_single_line(self):
        result = _enforce_compact_limit("DO NOW: Hold.", max_lines=8)
        assert "DO NOW" in result


# ===========================================================================
# Test: run_live_analysis() compact routing
# ===========================================================================


class TestRunLiveAnalysisCompactRouting:
    def test_compact_true_routes_to_compact(self):
        """compact=True should call _run_live_compact (mocked)."""
        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }
        with patch(
            "lib.integrations.grok_helper._run_live_compact",
            return_value="COMPACT OUTPUT",
        ) as mock_compact:
            result = run_live_analysis(context, "fake-key", compact=True)
            mock_compact.assert_called_once()
            assert result == "COMPACT OUTPUT"

    def test_compact_false_routes_to_verbose(self):
        """compact=False should call _run_live_verbose (mocked)."""
        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }
        with patch(
            "lib.integrations.grok_helper._run_live_verbose",
            return_value="VERBOSE OUTPUT",
        ) as mock_verbose:
            result = run_live_analysis(context, "fake-key", compact=False)
            mock_verbose.assert_called_once()
            assert result == "VERBOSE OUTPUT"

    def test_default_is_compact(self):
        """Default compact=True should route to compact handler."""
        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }
        with patch(
            "lib.integrations.grok_helper._run_live_compact",
            return_value="COMPACT",
        ) as mock:
            _result = run_live_analysis(context, "fake-key")
            mock.assert_called_once()


# ===========================================================================
# Test: GrokSession with compact flag
# ===========================================================================


class TestGrokSessionCompact:
    def test_run_update_passes_compact_flag(self):
        session = GrokSession()
        session.activate()
        session.last_update_time = 0  # Force update

        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }

        with patch(
            "lib.integrations.grok_helper.run_live_analysis",
            return_value="COMPACT RESULT",
        ) as mock_rla:
            _result = session.run_update(context, "fake-key", compact=True)
            mock_rla.assert_called_once()
            _, kwargs = mock_rla.call_args
            assert kwargs.get("compact") is True

    def test_run_update_verbose_flag(self):
        session = GrokSession()
        session.activate()
        session.last_update_time = 0

        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }

        with patch(
            "lib.integrations.grok_helper.run_live_analysis",
            return_value="VERBOSE RESULT",
        ) as mock_rla:
            _result = session.run_update(context, "fake-key", compact=False)
            mock_rla.assert_called_once()
            _, kwargs = mock_rla.call_args
            assert kwargs.get("compact") is False

    def test_session_records_compact_flag(self):
        session = GrokSession()
        session.activate()
        session.last_update_time = 0

        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }

        with patch(
            "lib.integrations.grok_helper.run_live_analysis",
            return_value="MGC 🟢 2712\n\nDO NOW: Hold.",
        ):
            session.run_update(context, "fake-key", compact=True)

        latest = session.get_latest_update()
        assert latest is not None
        assert latest["compact"] is True

    def test_session_cost_estimate_compact_cheaper(self):
        """Compact updates should be estimated cheaper ($0.005 vs $0.007)."""
        session = GrokSession()
        session.activate()
        session.last_update_time = 0

        context = {
            "time": "2025-01-15 08:00 EST",
            "account_size": 50_000,
            "session_status": "ACTIVE",
            "scanner_text": "No data",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "has_positions": False,
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "positions_text": "",
        }

        with patch(
            "lib.integrations.grok_helper.run_live_analysis",
            return_value="compact output",
        ):
            session.run_update(context, "fake-key", compact=True)

        # Compact cost should be $0.005
        assert session.estimated_cost == pytest.approx(0.005, abs=0.001)

    def test_session_inactive_no_update(self):
        session = GrokSession()
        # Not activated
        context = {"time": "now"}
        result = session.run_update(context, "fake-key", compact=True)
        assert result is None

    def test_session_summary_structure(self):
        session = GrokSession()
        summary = session.get_session_summary()
        assert "is_active" in summary
        assert "has_briefing" in summary
        assert "total_updates" in summary
        assert "total_calls" in summary
        assert "estimated_cost" in summary
        assert summary["is_active"] is False
        assert summary["total_updates"] == 0


# ===========================================================================
# Test: _run_live_compact prompt construction (mocked API)
# ===========================================================================


class TestRunLiveCompactPrompt:
    def test_compact_prompt_includes_scanner(self):
        """The compact prompt should include scanner text."""
        from lib.integrations.grok_helper import _run_live_compact

        captured_prompt = {}

        def fake_call_grok(prompt, api_key, max_tokens=None, system_prompt=None):
            captured_prompt["prompt"] = prompt
            captured_prompt["system_prompt"] = system_prompt
            captured_prompt["max_tokens"] = max_tokens
            return "MGC 🟢 2712 (+4) | Bias VALID | Watch 2725\n\nDO NOW: Hold."

        with patch(
            "lib.integrations.grok_helper._call_grok",
            side_effect=fake_call_grok,
        ):
            result = _run_live_compact(
                context={
                    "time": "2025-01-15 08:00 EST",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "MGC: 2712.50",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                    "positions_text": "",
                },
                api_key="test-key",
            )

        assert "MGC: 2712.50" in captured_prompt["prompt"]
        assert result is not None

    def test_compact_prompt_uses_lower_max_tokens(self):
        from lib.integrations.grok_helper import _run_live_compact

        captured = {}

        def fake_call_grok(prompt, api_key, max_tokens=None, system_prompt=None):
            captured["max_tokens"] = max_tokens
            return "DO NOW: Hold."

        with patch(
            "lib.integrations.grok_helper._call_grok",
            side_effect=fake_call_grok,
        ):
            _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                    "positions_text": "",
                },
                api_key="test-key",
            )

        assert captured["max_tokens"] == DEFAULT_MAX_TOKENS_LIVE_COMPACT

    def test_compact_prompt_includes_positions_when_present(self):
        from lib.integrations.grok_helper import _run_live_compact

        captured = {}

        def fake_call_grok(prompt, api_key, max_tokens=None, system_prompt=None):
            captured["prompt"] = prompt
            return "DO NOW: Hold."

        with patch(
            "lib.integrations.grok_helper._call_grok",
            side_effect=fake_call_grok,
        ):
            _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": True,
                    "positions_text": "MGC: LONG x2 @ 2700",
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                },
                api_key="test-key",
            )

        assert "MGC: LONG x2 @ 2700" in captured["prompt"]

    def test_compact_prompt_omits_positions_when_absent(self):
        from lib.integrations.grok_helper import _run_live_compact

        captured = {}

        def fake_call_grok(prompt, api_key, max_tokens=None, system_prompt=None):
            captured["prompt"] = prompt
            return "DO NOW: Hold."

        with patch(
            "lib.integrations.grok_helper._call_grok",
            side_effect=fake_call_grok,
        ):
            _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "positions_text": "",
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                },
                api_key="test-key",
            )

        assert "LIVE POSITIONS" not in captured["prompt"]

    def test_compact_result_enforced(self):
        """Even if API returns >8 lines, _enforce_compact_limit trims it."""
        from lib.integrations.grok_helper import _run_live_compact

        verbose_response = "\n".join(
            [f"ASSET{i} 🟢 {1000 + i} | Bias VALID" for i in range(12)] + ["", "DO NOW: Hold everything."]
        )

        with patch(
            "lib.integrations.grok_helper._call_grok",
            return_value=verbose_response,
        ):
            result = _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "positions_text": "",
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                },
                api_key="test-key",
            )

        assert result is not None
        content_lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(content_lines) <= 8
        assert "DO NOW" in result

    def test_compact_api_returns_none(self):
        """If API returns None, result should be None."""
        from lib.integrations.grok_helper import _run_live_compact

        with patch("lib.integrations.grok_helper._call_grok", return_value=None):
            result = _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "positions_text": "",
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                },
                api_key="test-key",
            )

        assert result is None

    def test_compact_includes_morning_plan_ref(self):
        from lib.integrations.grok_helper import _run_live_compact

        captured = {}

        def fake_call_grok(prompt, api_key, max_tokens=None, system_prompt=None):
            captured["prompt"] = prompt
            return "DO NOW: Hold."

        with patch(
            "lib.integrations.grok_helper._call_grok",
            side_effect=fake_call_grok,
        ):
            _run_live_compact(
                context={
                    "time": "now",
                    "account_size": 50_000,
                    "session_status": "ACTIVE",
                    "scanner_text": "N/A",
                    "ict_text": "N/A",
                    "conf_text": "N/A",
                    "has_positions": False,
                    "positions_text": "",
                    "fks_wave_text": "N/A",
                    "fks_sq_text": "N/A",
                },
                api_key="test-key",
                previous_briefing="Focus on Gold and NQ today. Bias bullish.",
            )

        assert "Focus on Gold" in captured["prompt"]


# ===========================================================================
# Test: Compact system prompt validates constraints
# ===========================================================================


class TestCompactSystemPrompt:
    def test_system_prompt_mentions_line_limit(self):
        from lib.integrations.grok_helper import _COMPACT_SYSTEM

        assert "8 lines" in _COMPACT_SYSTEM.lower() or "8" in _COMPACT_SYSTEM

    def test_system_prompt_mentions_do_now(self):
        from lib.integrations.grok_helper import _COMPACT_SYSTEM

        assert "DO NOW" in _COMPACT_SYSTEM

    def test_system_prompt_mentions_emoji_rules(self):
        from lib.integrations.grok_helper import _COMPACT_SYSTEM

        assert "🟢" in _COMPACT_SYSTEM
        assert "🔴" in _COMPACT_SYSTEM

    def test_system_prompt_mentions_no_dollar_signs(self):
        from lib.integrations.grok_helper import _COMPACT_SYSTEM

        assert "USD" in _COMPACT_SYSTEM


# ===========================================================================
# Test: DEFAULT_MAX_TOKENS_LIVE_COMPACT is smaller than verbose
# ===========================================================================


class TestCompactTokenLimit:
    def test_compact_tokens_less_than_verbose(self):
        from lib.integrations.grok_helper import (
            DEFAULT_MAX_TOKENS_LIVE,
            DEFAULT_MAX_TOKENS_LIVE_COMPACT,
        )

        assert DEFAULT_MAX_TOKENS_LIVE_COMPACT < DEFAULT_MAX_TOKENS_LIVE

    def test_compact_tokens_reasonable_range(self):
        """Compact tokens should be enough for 8 lines but much less than verbose."""
        assert 200 <= DEFAULT_MAX_TOKENS_LIVE_COMPACT <= 500


# ===========================================================================
# Test: Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_format_with_nan_price(self):
        """NaN price should not crash the formatter."""
        assets = [_asset("MGC", last_price=float("nan"))]
        # This might show 'nan' but should not raise
        result = format_live_compact(assets)
        assert isinstance(result, str)
        assert "MGC" in result

    def test_format_with_zero_price(self):
        assets = [_asset("MGC", last_price=0.0)]
        result = format_live_compact(assets)
        assert isinstance(result, str)
        assert "0.00" in result

    def test_format_with_very_long_symbol(self):
        assets = [_asset("A Very Long Asset Name Here")]
        result = format_live_compact(assets)
        assert "A Very Long Asset Name Here" in result

    def test_format_preserves_all_assets_up_to_five(self):
        """Exactly 5 assets should all appear."""
        assets = [_asset(f"A{i}") for i in range(5)]
        result = format_live_compact(assets)
        for i in range(5):
            assert f"A{i}" in result

    def test_format_six_assets_truncates_to_five(self):
        """6th asset should be omitted."""
        assets = [_asset(f"A{i}") for i in range(6)]
        result = format_live_compact(assets)
        assert "A0" in result
        assert "A4" in result
        assert "A5" not in result

    def test_enforce_compact_limit_preserves_do_now_casing(self):
        """DO NOW line should be preserved regardless of casing in input."""
        text = "L1\nL2\nL3\nL4\nL5\nL6\nL7\nL8\nL9\nL10\n\nDo Now: Keep holding."
        result = _enforce_compact_limit(text, max_lines=8)
        assert "Do Now: Keep holding." in result
