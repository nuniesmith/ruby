"""
Tests for the SSE (Server-Sent Events) endpoint
============================================================
Covers:
  - SSE message formatting (spec compliance)
  - Throttle logic (max 1 update per asset per N seconds)
  - Catch-up message retrieval from Redis Stream
  - Heartbeat event generation
  - Session event generation
  - SSE health endpoint response structure
  - Event generator connection/disconnection behavior
"""

import asyncio
import contextlib
import json
import sys
import time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lib.services.data.api.sse import (
    _CATCHUP_COUNT,
    _HEARTBEAT_INTERVAL,
    _THROTTLE_SECONDS,
    _format_sse,
    _get_catchup_messages,
    _get_engine_status,
    _get_focus_from_cache,
    _get_positions_from_cache,
    _get_redis,
    _last_sent,
    _make_heartbeat_event,
    _make_session_event,
    _should_throttle,
)

# We need to mock cache before importing sse module since it tries to connect to Redis
# Build a mock cache module that the lazy `from cache import ...` calls will resolve to.
# Since sse.py does `from cache import cache_get` inside function bodies, the mock must
# live in sys.modules["lib.core.cache"] BEFORE sse is imported.
_original_cache_module = sys.modules.get("lib.core.cache", None)

_mock_cache = MagicMock()
_mock_cache.REDIS_AVAILABLE = False
_mock_cache._r = None
_mock_cache.cache_get = MagicMock(return_value=None)
_mock_cache.cache_set = MagicMock()
_mock_cache._cache_key = MagicMock(side_effect=lambda *parts: "futures:mock:" + ":".join(str(p) for p in parts))
_mock_cache.get_data_source = MagicMock(return_value="mock")
_mock_cache.flush_all = MagicMock()

sys.modules["lib.core.cache"] = _mock_cache

# Immediately restore the real cache module after importing SSE symbols.
# The mock was only needed so that `api.sse` could be imported without a
# live Redis connection.  Leaving the mock in sys.modules["lib.core.cache"] would
# pollute every test file that pytest collects *after* this module
# (e.g. test_positions.py, test_data_service.py) because their fixtures
# do `from cache import _mem_cache` at runtime, which resolves via
# sys.modules.  The mock is re-installed by _reset_cache_mock() at the
# start of each SSE test that needs it.
if _original_cache_module is not None:
    sys.modules["lib.core.cache"] = _original_cache_module
else:
    sys.modules.pop("lib.core.cache", None)

_EST = ZoneInfo("America/New_York")


def _reset_cache_mock():
    """Reset the mock cache to default state between tests.

    Since the lazy imports inside sse.py functions resolve to
    _mock_cache attributes, we control behaviour by resetting
    the mock's return values here.

    Re-installs the mock into sys.modules["lib.core.cache"] so that lazy
    ``from cache import ...`` inside sse.py function bodies picks up
    _mock_cache regardless of test ordering.
    """
    sys.modules["lib.core.cache"] = _mock_cache

    _mock_cache.REDIS_AVAILABLE = False
    _mock_cache._r = None
    _mock_cache.cache_get.reset_mock()
    _mock_cache.cache_get.return_value = None
    _mock_cache.cache_get.side_effect = None
    _mock_cache.cache_set.reset_mock()
    _mock_cache._cache_key.reset_mock()
    _mock_cache._cache_key.side_effect = lambda *parts: "futures:mock:" + ":".join(str(p) for p in parts)


def _restore_real_cache_module():
    """Restore the original cache module into sys.modules.

    Called after each SSE test class / group finishes so that later test
    files (e.g. test_positions.py, test_data_service.py) that do
    ``from cache import _mem_cache`` get the real module, not the mock.
    """
    if _original_cache_module is not None:
        sys.modules["lib.core.cache"] = _original_cache_module
    else:
        sys.modules.pop("lib.core.cache", None)


@pytest.fixture(autouse=True, scope="module")
def _sse_cache_mock_lifecycle():
    """Module-scoped fixture: install mock before SSE tests, restore after."""
    sys.modules["lib.core.cache"] = _mock_cache
    yield
    _restore_real_cache_module()


# ===========================================================================
# SSE Format Tests
# ===========================================================================


class TestFormatSSE:
    """Test the _format_sse helper produces spec-compliant SSE messages."""

    def test_simple_data_only(self):
        result = _format_sse(data="hello")
        assert "data: hello\n" in result
        # Must end with two newlines (blank line terminates event)
        assert result.endswith("\n\n")

    def test_with_event_name(self):
        result = _format_sse(data="payload", event="focus-update")
        assert "event: focus-update\n" in result
        assert "data: payload\n" in result

    def test_with_id(self):
        result = _format_sse(data="x", id="12345-0")
        assert "id: 12345-0\n" in result

    def test_with_retry(self):
        result = _format_sse(data="x", retry=3000)
        assert "retry: 3000\n" in result

    def test_all_fields(self):
        result = _format_sse(data="test", event="heartbeat", id="99", retry=5000)
        assert "id: 99\n" in result
        assert "event: heartbeat\n" in result
        assert "retry: 5000\n" in result
        assert "data: test\n" in result

    def test_multiline_data(self):
        """Each line of data must get its own 'data:' prefix per SSE spec."""
        result = _format_sse(data="line1\nline2\nline3")
        assert "data: line1\n" in result
        assert "data: line2\n" in result
        assert "data: line3\n" in result

    def test_json_data(self):
        payload = json.dumps({"symbol": "MGC", "bias": "LONG"})
        result = _format_sse(data=payload, event="mgc-update")
        assert "event: mgc-update\n" in result
        assert f"data: {payload}\n" in result

    def test_empty_data(self):
        result = _format_sse(data="")
        assert "data: \n" in result

    def test_field_ordering(self):
        """id should come before event, event before retry, retry before data."""
        result = _format_sse(data="x", event="e", id="1", retry=100)
        lines = result.strip().split("\n")
        # Find positions of each field
        id_pos = next(i for i, line in enumerate(lines) if line.startswith("id:"))
        event_pos = next(i for i, line in enumerate(lines) if line.startswith("event:"))
        retry_pos = next(i for i, line in enumerate(lines) if line.startswith("retry:"))
        data_pos = next(i for i, line in enumerate(lines) if line.startswith("data:"))
        assert id_pos < event_pos < retry_pos < data_pos


# ===========================================================================
# Throttle Tests
# ===========================================================================


class TestThrottle:
    """Test the per-event throttle logic."""

    def setup_method(self):
        """Reset throttle state before each test."""
        _last_sent.clear()

    def test_first_event_not_throttled(self):
        assert _should_throttle("test-event") is False

    def test_immediate_repeat_throttled(self):
        _should_throttle("my-event")  # first call — not throttled, records time
        assert _should_throttle("my-event") is True  # second call — throttled

    def test_different_events_not_throttled(self):
        _should_throttle("event-a")
        assert _should_throttle("event-b") is False

    def test_after_cooldown_not_throttled(self):
        # Manually set last sent to well in the past
        _last_sent["old-event"] = time.monotonic() - _THROTTLE_SECONDS - 1
        assert _should_throttle("old-event") is False

    def test_throttle_records_time(self):
        before = time.monotonic()
        _should_throttle("record-test")
        after = time.monotonic()
        assert "record-test" in _last_sent
        assert before <= _last_sent["record-test"] <= after

    def test_multiple_events_independent(self):
        """Throttle state for one event doesn't affect another."""
        _should_throttle("mgc-update")
        _should_throttle("mnq-update")
        # Both are now on cooldown
        assert _should_throttle("mgc-update") is True
        assert _should_throttle("mnq-update") is True
        # A new event is fine
        assert _should_throttle("mes-update") is False


# ===========================================================================
# Heartbeat Event Tests
# ===========================================================================


class TestHeartbeatEvent:
    def test_heartbeat_is_valid_sse(self):
        event = _make_heartbeat_event()
        assert "event: heartbeat\n" in event
        assert "data:" in event
        assert event.endswith("\n\n")

    def test_heartbeat_contains_time(self):
        event = _make_heartbeat_event()
        # Extract the data line
        for line in event.split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data: ") :])
                assert payload["type"] == "heartbeat"
                assert "time_et" in payload
                assert "ET" in payload["time_et"]
                assert "timestamp" in payload
                break
        else:
            pytest.fail("No data line found in heartbeat event")


# ===========================================================================
# Session Event Tests
# ===========================================================================


class TestSessionEvent:
    def test_pre_market_event(self):
        event = _make_session_event("pre_market")
        assert "event: session-change\n" in event
        for line in event.split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data: ") :])
                assert payload["session"] == "pre_market"
                assert payload["type"] == "session-change"
                break

    def test_active_event(self):
        event = _make_session_event("active")
        for line in event.split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data: ") :])
                assert payload["session"] == "active"
                break

    def test_off_hours_event(self):
        event = _make_session_event("off_hours")
        for line in event.split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data: ") :])
                assert payload["session"] == "off_hours"
                break

    def test_unknown_session(self):
        """Unknown session mode should still produce valid event."""
        event = _make_session_event("whatever")
        assert "event: session-change\n" in event


# ===========================================================================
# Redis Helper Tests (with mocked cache)
# ===========================================================================


class TestGetRedis:
    def setup_method(self):
        _reset_cache_mock()

    def test_returns_none_when_redis_unavailable(self):
        # _mock_cache.REDIS_AVAILABLE is False by default, _mock_cache._r is None
        result = _get_redis()
        assert result is None

    def test_returns_client_when_redis_available(self):
        mock_client = MagicMock()
        _mock_cache.REDIS_AVAILABLE = True
        _mock_cache._r = mock_client
        try:
            result = _get_redis()
            assert result is mock_client
        finally:
            _mock_cache.REDIS_AVAILABLE = False
            _mock_cache._r = None

    def test_no_crash_on_import_error(self):
        """If cache module can't be imported, should return None gracefully."""
        saved = sys.modules["lib.core.cache"]
        sys.modules["lib.core.cache"] = None  # type: ignore[assignment]
        try:
            # This might raise or return None — either way, no unhandled exception
            with contextlib.suppress(ImportError, TypeError):
                _get_redis()
        finally:
            sys.modules["lib.core.cache"] = saved


class TestCatchupMessages:
    def setup_method(self):
        _reset_cache_mock()

    def test_returns_empty_when_no_redis(self):
        with patch("lib.services.data.api.sse._get_redis", return_value=None):
            result = _get_catchup_messages()
            assert result == []

    def test_returns_empty_when_stream_empty(self):
        mock_redis = MagicMock()
        mock_redis.xrevrange.return_value = []
        with patch("lib.services.data.api.sse._get_redis", return_value=mock_redis):
            result = _get_catchup_messages()
            assert result == []

    def test_parses_stream_messages(self):
        mock_redis = MagicMock()
        # Simulate two Redis stream messages (XREVRANGE returns newest first)
        mock_redis.xrevrange.return_value = [
            (b"1234-1", {b"data": b'{"assets":[]}', b"ts": b"2026-02-26T10:00:00"}),
            (
                b"1234-0",
                {b"data": b'{"assets":["MGC"]}', b"ts": b"2026-02-26T09:59:00"},
            ),
        ]
        with patch("lib.services.data.api.sse._get_redis", return_value=mock_redis):
            result = _get_catchup_messages(count=8)
            # Should be reversed (oldest first)
            assert len(result) == 2
            assert result[0]["id"] == "1234-0"
            assert result[1]["id"] == "1234-1"
            assert '"MGC"' in result[0]["data"]

    def test_handles_xrevrange_exception(self):
        mock_redis = MagicMock()
        mock_redis.xrevrange.side_effect = Exception("Redis down")
        with patch("lib.services.data.api.sse._get_redis", return_value=mock_redis):
            result = _get_catchup_messages()
            assert result == []

    def test_default_count(self):
        assert _CATCHUP_COUNT == 1


class TestGetFocusFromCache:
    """Test _get_focus_from_cache which does `from cache import cache_get` lazily."""

    def setup_method(self):
        _reset_cache_mock()

    def test_returns_none_when_no_data(self):
        # _mock_cache.cache_get already returns None by default
        result = _get_focus_from_cache()
        assert result is None

    def test_returns_decoded_json(self):
        focus = {"assets": [{"symbol": "MGC"}], "no_trade": False}
        _mock_cache.cache_get.return_value = json.dumps(focus).encode()
        result = _get_focus_from_cache()
        assert result is not None
        parsed = json.loads(result)
        assert parsed["assets"][0]["symbol"] == "MGC"

    def test_handles_exception(self):
        _mock_cache.cache_get.side_effect = Exception("boom")
        result = _get_focus_from_cache()
        assert result is None

    def test_returns_string_from_bytes(self):
        _mock_cache.cache_get.return_value = b'{"test": true}'
        result = _get_focus_from_cache()
        assert isinstance(result, str)
        assert json.loads(result)["test"] is True


class TestGetPositionsFromCache:
    def setup_method(self):
        _reset_cache_mock()

    def test_returns_none_when_no_data(self):
        result = _get_positions_from_cache()
        assert result is None

    def test_returns_decoded_positions(self):
        positions = {"account": "Sim101", "positions": []}
        _mock_cache.cache_get.return_value = json.dumps(positions).encode()
        result = _get_positions_from_cache()
        assert result is not None
        parsed = json.loads(result)
        assert parsed["account"] == "Sim101"


class TestGetEngineStatus:
    def setup_method(self):
        _reset_cache_mock()

    def test_returns_none_when_no_data(self):
        result = _get_engine_status()
        assert result is None

    def test_returns_parsed_dict(self):
        status = {"engine": "running", "session_mode": "active"}
        _mock_cache.cache_get.return_value = json.dumps(status).encode()
        result = _get_engine_status()
        assert result is not None
        assert result["session_mode"] == "active"

    def test_handles_invalid_json(self):
        _mock_cache.cache_get.return_value = b"not json at all"
        result = _get_engine_status()
        assert result is None


# ===========================================================================
# SSE Health Endpoint Tests (via TestClient)
# ===========================================================================


class TestSSEHealthEndpoint:
    """Test the /sse/health endpoint."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_cache_mock()

    @pytest.fixture
    def client(self):
        """Create a test client with just the SSE router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.sse import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_returns_200(self, client):
        with patch("lib.services.data.api.sse._get_redis", return_value=None):
            resp = client.get("/sse/health")
            assert resp.status_code == 200

    def test_health_structure(self, client):
        with patch("lib.services.data.api.sse._get_redis", return_value=None):
            data = client.get("/sse/health").json()
            assert "status" in data
            assert "redis_connected" in data
            assert "stream_length" in data
            assert "mode" in data
            assert "throttle_seconds" in data
            assert "heartbeat_interval" in data
            assert "catchup_count" in data

    def test_health_degraded_without_redis(self, client):
        with patch("lib.services.data.api.sse._get_redis", return_value=None):
            data = client.get("/sse/health").json()
            assert data["status"] == "degraded"
            assert data["redis_connected"] is False
            assert data["mode"] == "polling"
            assert data["stream_length"] == 0

    def test_health_ok_with_redis(self, client):
        mock_redis = MagicMock()
        mock_redis.xinfo_stream.return_value = {b"length": 42}
        with patch("lib.services.data.api.sse._get_redis", return_value=mock_redis):
            data = client.get("/sse/health").json()
            assert data["status"] == "ok"
            assert data["redis_connected"] is True
            assert data["stream_length"] == 42

    def test_health_ok_redis_stream_missing(self, client):
        """Redis connected but stream doesn't exist yet."""
        mock_redis = MagicMock()
        mock_redis.xinfo_stream.side_effect = Exception("no such key")
        with patch("lib.services.data.api.sse._get_redis", return_value=mock_redis):
            data = client.get("/sse/health").json()
            assert data["status"] == "ok"
            assert data["redis_connected"] is True
            assert data["stream_length"] == 0

    def test_health_reports_correct_constants(self, client):
        with patch("lib.services.data.api.sse._get_redis", return_value=None):
            data = client.get("/sse/health").json()
            assert data["throttle_seconds"] == _THROTTLE_SECONDS
            assert data["heartbeat_interval"] == _HEARTBEAT_INTERVAL
            assert data["catchup_count"] == _CATCHUP_COUNT


# ===========================================================================
# SSE Dashboard Endpoint Tests
# ===========================================================================


class TestSSEDashboardEndpoint:
    """Test the /sse/dashboard endpoint returns proper SSE response.

    These tests use ``client.stream("GET", ...)`` which opens a streaming
    connection to the infinite SSE generator.  To avoid timeouts we:
      1. Patch ``asyncio.sleep`` so the generator's polling loop yields
         immediately instead of blocking for 5 seconds.
      2. Collect only the *initial* batch of events (connected, catch-up,
         positions, session) which are all emitted *before* the infinite
         loop, so ``iter_bytes`` returns them in the first chunk.
      3. Break out of ``iter_bytes`` / ``iter_text`` after reading enough
         data rather than waiting for the stream to close.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        _reset_cache_mock()
        _last_sent.clear()

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.sse import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def _sse_patches(self):
        """Context manager stack that patches all Redis/cache helpers for SSE tests.

        Also patches ``asyncio.sleep`` inside the generator so the infinite
        loop yields control immediately and the first ``iter_bytes`` call
        returns all buffered events without blocking.
        """
        from contextlib import ExitStack

        async def _instant_sleep(_t):
            """Replace asyncio.sleep with a no-op that raises after first call
            so the generator exits quickly during tests."""
            raise asyncio.CancelledError("test: stop generator")

        stack = ExitStack()
        stack.enter_context(patch("lib.services.data.api.sse._get_redis", return_value=None))
        stack.enter_context(
            patch(
                "lib.services.data.api.sse._get_catchup_messages",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "lib.services.data.api.sse._get_focus_from_cache",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "lib.services.data.api.sse._get_positions_from_cache",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "lib.services.data.api.sse._get_engine_status",
                return_value=None,
            )
        )
        # Patch asyncio.sleep inside the sse module so the infinite loop
        # terminates after emitting the initial events.
        stack.enter_context(
            patch(
                "lib.services.data.api.sse.asyncio.sleep",
                side_effect=_instant_sleep,
            )
        )
        return stack

    def _read_initial_events(self, resp, max_bytes: int = 8192) -> str:
        """Read the initial burst of SSE events from a streaming response.

        Reads up to *max_bytes* of raw bytes from the stream and returns
        the decoded text.  This avoids blocking on the infinite generator.
        """
        collected = b""
        for chunk in resp.iter_bytes():
            collected += chunk
            if len(collected) >= max_bytes:
                break
            # The generator emits all initial events then hits the sleep
            # (which we patched to raise CancelledError), so the stream
            # will end naturally after the initial batch.
            break
        return collected.decode("utf-8", errors="replace")

    @pytest.mark.timeout(10)
    def test_sse_content_type(self, client):
        """SSE endpoint must return text/event-stream content type."""
        with self._sse_patches(), client.stream("GET", "/sse/dashboard") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            text = self._read_initial_events(resp)
            assert "event: connected" in text

    @pytest.mark.timeout(10)
    def test_sse_no_cache_headers(self, client):
        """SSE responses must not be cached."""
        with self._sse_patches(), client.stream("GET", "/sse/dashboard") as resp:
            cache_control = resp.headers.get("cache-control", "")
            assert "no-cache" in cache_control
            # Consume at least one chunk so the connection is used
            self._read_initial_events(resp)

    @pytest.mark.timeout(10)
    def test_sse_x_accel_buffering(self, client):
        """Should set X-Accel-Buffering: no for nginx compatibility."""
        with self._sse_patches(), client.stream("GET", "/sse/dashboard") as resp:
            assert resp.headers.get("x-accel-buffering") == "no"
            self._read_initial_events(resp)

    @pytest.mark.timeout(10)
    def test_sse_sends_connected_event_first(self, client):
        """First event must be `connected` with retry directive."""
        with self._sse_patches(), client.stream("GET", "/sse/dashboard") as resp:
            text = self._read_initial_events(resp)
            assert "event: connected" in text
            assert "retry: 3000" in text
            assert "data: connected" in text

    @pytest.mark.timeout(10)
    def test_sse_sends_catchup_when_available(self, client):
        """When catchup messages exist, they should be sent after connected."""
        catchup = [
            {
                "id": "100-0",
                "data": '{"assets":[{"symbol":"MGC"}],"no_trade":false}',
                "ts": "2026-02-26T09:00:00",
            }
        ]

        async def _instant_sleep(_t):
            raise asyncio.CancelledError("test: stop generator")

        with (
            patch("lib.services.data.api.sse._get_redis", return_value=None),
            patch(
                "lib.services.data.api.sse._get_catchup_messages",
                return_value=catchup,
            ),
            patch(
                "lib.services.data.api.sse._get_focus_from_cache",
                return_value=None,
            ),
            patch(
                "lib.services.data.api.sse._get_positions_from_cache",
                return_value=None,
            ),
            patch(
                "lib.services.data.api.sse._get_engine_status",
                return_value=None,
            ),
            patch(
                "lib.services.data.api.sse.asyncio.sleep",
                side_effect=_instant_sleep,
            ),
            client.stream("GET", "/sse/dashboard") as resp,
        ):
            text = self._read_initial_events(resp)
            assert "event: focus-update" in text
            assert "id: 100-0" in text
            assert "MGC" in text


# ===========================================================================
# Constants Sanity Checks
# ===========================================================================


class TestConstants:
    def test_throttle_seconds_reasonable(self):
        assert 1.0 <= _THROTTLE_SECONDS <= 30.0

    def test_heartbeat_interval_reasonable(self):
        assert 10.0 <= _HEARTBEAT_INTERVAL <= 120.0

    def test_catchup_count_reasonable(self):
        assert 1 <= _CATCHUP_COUNT <= 50
