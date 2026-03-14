"""
Grok AI API Router — Streaming SSE + Cached Endpoints
======================================================
Exposes the Grok AI analyst responses over both REST (cached) and
Server-Sent Events (streaming) transports so the dashboard can render
AI commentary progressively as tokens arrive from the xAI API.

Endpoints:
    GET  /api/grok/html               — Cached compact update panel (HTMX fragment)
    GET  /api/grok/latest             — Latest cached Grok update as JSON
    POST /api/grok/trigger/briefing   — Trigger a fresh morning briefing (async)
    POST /api/grok/trigger/update     — Trigger a fresh live update (async)
    GET  /sse/grok/briefing           — Stream morning briefing tokens via SSE
    GET  /sse/grok/update             — Stream live update tokens via SSE

Streaming design
----------------
The xAI API supports OpenAI-compatible streaming (``"stream": True``).
``_stream_grok()`` in ``grok_helper.py`` yields token strings from the
chunked HTTP response.  The SSE endpoints here wrap that generator in an
async FastAPI ``StreamingResponse`` using ``asyncio.to_thread`` so the
blocking ``requests`` library does not stall the event loop.

Each SSE event in the stream carries:
    event: grok-token
    data: <raw token text>

A final sentinel event signals completion:
    event: grok-done
    data: {"status": "complete", "chars": <total_chars>}

On error the stream emits:
    event: grok-error
    data: {"error": "<message>"}

The dashboard JS listener assembles the token stream into the panel
and swaps the completed text into the Grok panel via the ``grok-done``
event trigger.

Cache keys
----------
    engine:grok_update       — Latest compact live update (JSON)
    engine:grok_briefing     — Latest morning briefing text (raw string)

Both are written by the engine's ``_handle_grok_morning_brief`` /
``_handle_grok_live_update`` handlers AND by the streaming endpoints
(so a fresh stream also refreshes the cache for HTMX polling consumers).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("api.grok")

_ET = ZoneInfo("America/New_York")

router = APIRouter(tags=["grok"])

# ---------------------------------------------------------------------------
# Engine accessor — injected by main.py lifespan
# ---------------------------------------------------------------------------

_engine = None


def set_engine(engine) -> None:
    """Inject the engine singleton (called from data service lifespan)."""
    global _engine
    _engine = engine


def _get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not started yet")
    return _engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Return the xAI / Grok API key from environment."""
    key = os.environ.get("XAI_API_KEY", os.environ.get("XAI_API_KEY", ""))
    return key


def _grok_update_from_cache() -> dict[str, Any] | None:
    """Read the latest compact Grok update from Redis cache."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:grok_update")
        if raw:
            data = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            return json.loads(data)
    except Exception as exc:
        logger.debug("Grok update cache read error: %s", exc)
    return None


def _grok_briefing_from_cache() -> str | None:
    """Read the latest morning briefing text from Redis cache."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:grok_briefing")
        if raw:
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except Exception as exc:
        logger.debug("Grok briefing cache read error: %s", exc)
    return None


def _cache_briefing(text: str) -> None:
    """Persist the completed briefing to Redis so HTMX polling picks it up."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        cache_set("engine:grok_briefing", text.encode(), ttl=3600)

        # Also write in the engine:grok_update shape so the panel re-renders
        payload = {
            "text": text,
            "type": "briefing",
            "time_et": datetime.now(tz=_ET).strftime("%H:%M ET"),
            "generated_at": datetime.now(tz=_ET).isoformat(),
        }
        payload_str = json.dumps(payload)
        cache_set("engine:grok_update", payload_str.encode(), ttl=3600)

        # Notify SSE subscribers via Redis pub/sub
        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish("dashboard:grok", payload_str)
    except Exception as exc:
        logger.debug("Grok briefing cache write error: %s", exc)


def _cache_live_update(text: str) -> None:
    """Persist a completed live update to Redis."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        payload = {
            "text": text,
            "type": "live_update",
            "time_et": datetime.now(tz=_ET).strftime("%H:%M ET"),
            "generated_at": datetime.now(tz=_ET).isoformat(),
        }
        payload_str = json.dumps(payload)
        cache_set("engine:grok_update", payload_str.encode(), ttl=900)

        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish("dashboard:grok", payload_str)
    except Exception as exc:
        logger.debug("Grok live update cache write error: %s", exc)


def _format_sse(data: str, event: str | None = None) -> str:
    """Format a single SSE message."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def _build_context(engine) -> dict[str, Any] | None:
    """Build a minimal Grok context dict from Redis-cached engine state.

    Uses the focus data already published to Redis by the engine rather
    than calling the full ``format_market_context()`` (which requires
    all scanner / ICT / confluence data to be provided as arguments).
    Falls back to a minimal stub so streaming endpoints can still run.
    """
    try:
        from lib.core.cache import cache_get
        from lib.integrations.grok_helper import format_live_compact
        from lib.services.engine.scheduler import ScheduleManager

        account_size = int(os.environ.get("ACCOUNT_SIZE", "150000"))
        session = ScheduleManager.get_session_mode()

        # Read the focus assets list that the engine publishes each cycle
        assets: list[dict[str, Any]] = []
        raw_focus = cache_get("engine:daily_focus")
        if raw_focus:
            focus = json.loads(raw_focus.decode() if isinstance(raw_focus, bytes) else raw_focus)
            assets = focus.get("assets", [])

        # Read the latest Grok update text (used as previous_briefing context)
        prev_update_text = ""
        raw_update = cache_get("engine:grok_update")
        if raw_update:
            upd = json.loads(raw_update.decode() if isinstance(raw_update, bytes) else raw_update)
            prev_update_text = upd.get("text", "")

        # Build a compact scanner text the same way the engine does
        scanner_text = format_live_compact(assets) if assets else "No scanner data available"

        now_et = datetime.now(tz=_ET)

        context: dict[str, Any] = {
            "time": now_et.strftime("%Y-%m-%d %H:%M ET"),
            "account_size": account_size,
            "risk_dollars": max(int(account_size * 0.01), 100),
            "max_contracts": 2,
            "session_status": session.value,
            "scanner_text": scanner_text,
            "specs_text": "See contract specs at /api/info",
            "opt_text": "N/A (request from engine)",
            "bt_text": "N/A (request from engine)",
            "ict_text": "N/A",
            "conf_text": "N/A",
            "cvd_text": "N/A",
            "scorer_text": "N/A",
            "fks_wave_text": "N/A",
            "fks_vol_text": "N/A",
            "fks_sq_text": "N/A",
            "previous_update": prev_update_text,
        }

        # Enrich with richer data if the engine has published it
        for key, cache_key in [
            ("ict_text", "engine:ict_summary"),
            ("cvd_text", "engine:cvd_summary"),
            ("scorer_text", "engine:scorer_summary"),
        ]:
            raw = cache_get(cache_key)
            if raw:
                with contextlib.suppress(Exception):
                    context[key] = raw.decode() if isinstance(raw, bytes) else raw

        return context
    except Exception as exc:
        logger.warning("_build_context error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Streaming SSE generator (async wrapper over the blocking requests stream)
# ---------------------------------------------------------------------------


async def _stream_briefing_sse(context: dict[str, Any], api_key: str) -> AsyncGenerator[str]:
    """Async SSE generator for morning briefing streaming.

    Wraps the blocking ``stream_morning_briefing()`` generator in
    ``asyncio.to_thread`` calls, yielding SSE-formatted events.

    Accumulated text is written to the cache on completion so the HTMX
    polling consumers and the Redis pub/sub SSE dashboard both pick it up.
    """
    from lib.integrations.grok_helper import stream_morning_briefing

    accumulated: list[str] = []
    char_count = 0
    had_error = False

    # Run the blocking generator in a thread pool to avoid blocking the loop.
    # We collect chunks into a queue and yield from the async side.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _producer() -> None:
        """Blocking producer — runs in thread pool, pushes tokens to queue."""
        try:
            for token in stream_morning_briefing(context, api_key):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"ERROR: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _producer)

    # Yield a start event so the browser knows streaming has begun
    yield _format_sse(
        json.dumps({"status": "streaming", "type": "briefing"}),
        event="grok-start",
    )

    # Drain queue
    while True:
        try:
            token = await asyncio.wait_for(queue.get(), timeout=5.0)
        except TimeoutError:
            # Heartbeat to keep the connection alive during slow generation
            yield _format_sse(".", event="grok-heartbeat")
            continue

        if token is None:
            break  # sentinel — generation complete

        if isinstance(token, str) and token.startswith("ERROR:"):
            had_error = True
            yield _format_sse(
                json.dumps({"error": token[len("ERROR:") :].strip()}),
                event="grok-error",
            )
            break

        accumulated.append(token)
        char_count += len(token)
        # Emit each token as an SSE event
        yield _format_sse(token, event="grok-token")

    # Completion event
    full_text = "".join(accumulated)
    if full_text and not had_error:
        _cache_briefing(full_text)

    yield _format_sse(
        json.dumps({"status": "complete", "chars": char_count, "had_error": had_error}),
        event="grok-done",
    )


async def _stream_live_update_sse(
    context: dict[str, Any],
    api_key: str,
    update_number: int = 1,
) -> AsyncGenerator[str]:
    """Async SSE generator for live update streaming."""
    from lib.integrations.grok_helper import stream_live_analysis

    previous_briefing = _grok_briefing_from_cache()
    accumulated: list[str] = []
    char_count = 0
    had_error = False

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _producer() -> None:
        try:
            for token in stream_live_analysis(
                context,
                api_key,
                previous_briefing=previous_briefing,
                update_number=update_number,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"ERROR: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _producer)

    yield _format_sse(
        json.dumps({"status": "streaming", "type": "live_update", "update_number": update_number}),
        event="grok-start",
    )

    while True:
        try:
            token = await asyncio.wait_for(queue.get(), timeout=5.0)
        except TimeoutError:
            yield _format_sse(".", event="grok-heartbeat")
            continue

        if token is None:
            break

        if isinstance(token, str) and token.startswith("ERROR:"):
            had_error = True
            yield _format_sse(
                json.dumps({"error": token[len("ERROR:") :].strip()}),
                event="grok-error",
            )
            break

        accumulated.append(token)
        char_count += len(token)
        yield _format_sse(token, event="grok-token")

    full_text = "".join(accumulated)
    if full_text and not had_error:
        _cache_live_update(full_text)

    yield _format_sse(
        json.dumps({"status": "complete", "chars": char_count, "had_error": had_error}),
        event="grok-done",
    )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.get("/api/grok/latest")
def get_grok_latest():
    """Return the latest cached Grok update as JSON.

    Returns the cached ``engine:grok_update`` Redis key, or a placeholder
    if no update has been generated yet.
    """
    data = _grok_update_from_cache()
    if data:
        return {"status": "ok", "data": data}
    return {
        "status": "no_data",
        "message": "No Grok update available. Engine will generate one during the active session.",
        "data": None,
    }


@router.get("/api/grok/briefing")
def get_grok_briefing():
    """Return the latest cached morning briefing text as JSON."""
    text = _grok_briefing_from_cache()
    if text:
        return {
            "status": "ok",
            "briefing": text,
            "length": len(text),
            "generated_at": datetime.now(tz=_ET).isoformat(),
        }
    return {"status": "no_data", "briefing": None}


# ---------------------------------------------------------------------------
# Streaming SSE endpoints
# ---------------------------------------------------------------------------


@router.get("/sse/grok/briefing")
async def sse_grok_briefing(request: Request):
    """Stream a fresh Grok morning briefing via SSE.

    Calls the xAI API in streaming mode and emits SSE events:

        event: grok-start      — stream begins  (JSON meta)
        event: grok-token      — one text token  (raw string)
        event: grok-heartbeat  — keep-alive dot  (every 5s if slow)
        event: grok-error      — API error       (JSON {error: "..."})
        event: grok-done       — stream complete (JSON {chars, had_error})

    The completed text is written to the Redis cache (``engine:grok_briefing``
    and ``engine:grok_update``) so HTMX polling and the main SSE dashboard
    panel both pick up the new briefing automatically.

    **Dashboard integration (JS)**:

        const es = new EventSource('/sse/grok/briefing');
        let buffer = '';
        es.addEventListener('grok-token', e => {
            buffer += e.data;
            document.getElementById('grok-text').textContent = buffer;
        });
        es.addEventListener('grok-done', e => {
            es.close();
            const meta = JSON.parse(e.data);
            console.log('Briefing complete:', meta.chars, 'chars');
        });
        es.addEventListener('grok-error', e => {
            es.close();
            console.error('Grok error:', JSON.parse(e.data).error);
        });
    """
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="XAI_API_KEY / XAI_API_KEY environment variable not set",
        )

    engine = _get_engine()
    context = _build_context(engine)
    if context is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to build market context — engine may still be initialising",
        )

    return StreamingResponse(
        _stream_briefing_sse(context, api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/sse/grok/update")
async def sse_grok_update(request: Request, update_number: int = 1):
    """Stream a fresh Grok live-analysis update via SSE.

    Same event protocol as ``/sse/grok/briefing``.  The completed text
    is written to ``engine:grok_update`` in Redis and published to the
    ``dashboard:grok`` pub/sub channel so the main dashboard SSE
    stream also picks it up.

    Args:
        update_number: Sequence number for the update (used in the prompt
                       context so the model knows how many updates have
                       been generated this session).  Defaults to 1.
    """
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="XAI_API_KEY / XAI_API_KEY environment variable not set",
        )

    engine = _get_engine()
    context = _build_context(engine)
    if context is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to build market context — engine may still be initialising",
        )

    return StreamingResponse(
        _stream_live_update_sse(context, api_key, update_number=update_number),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Trigger endpoints — fire-and-forget background generation
# ---------------------------------------------------------------------------


@router.post("/api/grok/trigger/briefing")
async def trigger_briefing():
    """Trigger a background morning briefing generation (non-streaming).

    Returns immediately with a 202 Accepted.  The engine generates the
    briefing asynchronously and writes the result to Redis.  Poll
    ``GET /api/grok/briefing`` or listen on the main SSE stream for the
    ``grok-update`` event.

    Use ``GET /sse/grok/briefing`` if you want streaming output directly.
    """
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="XAI_API_KEY not set")

    engine = _get_engine()
    context = _build_context(engine)
    if context is None:
        raise HTTPException(status_code=503, detail="Unable to build market context")

    async def _background():
        try:
            from lib.integrations.grok_helper import run_morning_briefing

            text = await asyncio.to_thread(run_morning_briefing, context, api_key)
            if text:
                _cache_briefing(text)
                logger.info("Background briefing generated (%d chars)", len(text))
        except Exception as exc:
            logger.warning("Background briefing error: %s", exc)

    asyncio.create_task(_background())
    return {
        "status": "accepted",
        "message": "Morning briefing generation started in background. Poll /api/grok/briefing for result.",
        "timestamp": datetime.now(tz=_ET).isoformat(),
    }


@router.post("/api/grok/trigger/update")
async def trigger_live_update(update_number: int = 1):
    """Trigger a background live analysis update (non-streaming).

    Returns immediately with 202 Accepted.  The result is published to
    Redis ``engine:grok_update`` and the ``dashboard:grok`` pub/sub
    channel, which the main dashboard SSE stream will pick up.
    """
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="XAI_API_KEY not set")

    engine = _get_engine()
    context = _build_context(engine)
    if context is None:
        raise HTTPException(status_code=503, detail="Unable to build market context")

    async def _background():
        try:
            from lib.integrations.grok_helper import run_live_analysis

            text = await asyncio.to_thread(run_live_analysis, context, api_key)
            if text:
                _cache_live_update(text)
                logger.info("Background live update generated (%d chars)", len(text))
        except Exception as exc:
            logger.warning("Background live update error: %s", exc)

    asyncio.create_task(_background())
    return {
        "status": "accepted",
        "message": "Live update generation started in background. Listen on SSE stream for grok-update event.",
        "update_number": update_number,
        "timestamp": datetime.now(tz=_ET).isoformat(),
    }
