"""
RustAssistant Chat API Router
==============================
Provides a streaming chat interface backed by the RustAssistant LLM proxy
(Ollama + RAG + Redis context), with automatic fallback to direct xAI/Grok.

Tier order
----------
  1. RustAssistant proxy  (RA_BASE_URL / RA_API_KEY)
      - OpenAI-compatible /v1/chat/completions
      - Optional RAG context via x-repo-id header (RA_REPO_ID)
      - Powered by Ollama locally, routes to Grok externally as its own fallback
  2. Direct xAI / Grok   (XAI_API_KEY)
      - Used only when RA is unreachable, returns HTTP error, or stream fails

Endpoints
---------
  POST /api/chat          — Non-streaming single-turn chat (JSON in/out)
  GET  /sse/chat          — Streaming chat via Server-Sent Events
  GET  /api/chat/history  — Last N messages from Redis (per session_id)
  DELETE /api/chat/history — Clear history for a session
  GET  /api/chat/status   — Backend health / availability check

SSE event protocol (GET /sse/chat)
------------------------------------
  event: chat-start    data: {"session_id": "...", "backend": "ra|grok"}
  event: chat-token    data: <raw token string>
  event: chat-heartbeat data: .
  event: chat-error    data: {"error": "..."}
  event: chat-done     data: {"session_id": "...", "chars": N, "backend": "..."}

Chat history
------------
Each session maintains a rolling window of messages in Redis under the key
  chat:history:<session_id>
trimmed to MAX_HISTORY_MESSAGES pairs (user + assistant).
The history is injected as prior conversation turns so the model has
multi-turn context across page refreshes.

Market context injection
------------------------
When inject_context=true (default) the latest Redis-cached engine state
(scanner, ICT, CVD, session) is prepended as a system-level context block
so the assistant can answer questions about the current market.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError, AsyncOpenAI  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("api.chat")

_ET = ZoneInfo("America/New_York")

router = APIRouter(tags=["Chat"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = int(os.environ.get("CHAT_MAX_HISTORY", "20"))  # pairs
MAX_TOKENS_CHAT = int(os.environ.get("CHAT_MAX_TOKENS", "1024"))
CHAT_TEMPERATURE = float(os.environ.get("CHAT_TEMPERATURE", "0.4"))
HISTORY_TTL_SECONDS = int(os.environ.get("CHAT_HISTORY_TTL", str(6 * 3600)))  # 6 h

_REDIS_HISTORY_PREFIX = "chat:history:"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Ruby's AI trading assistant embedded in a live futures trading dashboard.
You have access to real-time market context injected below (when available).

Your role:
- Answer questions about the current market, positions, signals, and strategy.
- Explain Ruby's indicators: FKS wave analysis, volatility clusters, signal quality scores, ICT levels (FVGs, OBs, liquidity sweeps), CVD delta, ORB setups.
- Give concise, actionable answers grounded in the data shown.
- When asked for a trade idea, always include entry, stop-loss, and target levels.
- Flag when you are uncertain or when data is stale/unavailable.

Style rules:
- Be concise. Use bullet points for lists.
- Write USD instead of bare $ signs.
- Do not use LaTeX or math notation.
- Keep answers under 300 words unless the user explicitly asks for detail.
- You are talking to an experienced futures trader — skip beginner disclaimers.\
"""

# ---------------------------------------------------------------------------
# Engine accessor (injected by lifespan)
# ---------------------------------------------------------------------------

_engine = None


def set_engine(engine) -> None:
    """Inject the engine singleton (called from data service lifespan)."""
    global _engine
    _engine = engine


def _get_engine_safe():
    """Return engine or None — never raises, used for context enrichment."""
    return _engine


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _redis_get(key: str) -> bytes | None:
    """Read a key from Redis, returning None on any error."""
    try:
        from lib.core.cache import cache_get

        return cache_get(key)
    except Exception:
        return None


def _redis_set(key: str, value: bytes, ttl: int | None = None) -> None:
    with contextlib.suppress(Exception):
        from lib.core.cache import cache_set

        cache_set(key, value, ttl=ttl if ttl is not None else 0)


def _redis_delete(key: str) -> None:
    try:
        from lib.core.cache import _r

        if _r is not None:
            _r.delete(key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chat history (stored as JSON list in a single Redis key)
# ---------------------------------------------------------------------------


def _history_key(session_id: str) -> str:
    return f"{_REDIS_HISTORY_PREFIX}{session_id}"


def _load_history(session_id: str) -> list[dict[str, str]]:
    """Load conversation history for a session from Redis."""
    raw = _redis_get(_history_key(session_id))
    if not raw:
        return []
    try:
        data = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(data)
    except Exception:
        return []


def _save_history(session_id: str, messages: list[dict[str, str]]) -> None:
    """Persist conversation history, trimming to MAX_HISTORY_MESSAGES pairs."""
    # Keep most recent N*2 entries (user + assistant per pair)
    trimmed = messages[-(MAX_HISTORY_MESSAGES * 2) :]
    with contextlib.suppress(Exception):
        _redis_set(
            _history_key(session_id),
            json.dumps(trimmed).encode(),
            ttl=HISTORY_TTL_SECONDS,
        )


def _clear_history(session_id: str) -> None:
    _redis_delete(_history_key(session_id))


# ---------------------------------------------------------------------------
# Market context builder
# ---------------------------------------------------------------------------


def _build_market_context() -> str:
    """Build a compact market context block from Redis-cached engine state.

    Returns an empty string when no context is available so the system
    prompt degrades gracefully.
    """
    parts: list[str] = []

    now_et = datetime.now(tz=_ET).strftime("%Y-%m-%d %H:%M ET")
    parts.append(f"CURRENT TIME: {now_et}")

    # Session mode
    try:
        from lib.services.engine.scheduler import ScheduleManager

        session = ScheduleManager.get_session_mode()
        parts.append(f"SESSION: {session.value}")
    except Exception:
        pass

    # Account size
    account_size = int(os.environ.get("ACCOUNT_SIZE", "150000"))
    parts.append(f"ACCOUNT SIZE: USD {account_size:,}")

    # Scanner / focus assets
    raw_focus = _redis_get("engine:daily_focus")
    if raw_focus:
        with contextlib.suppress(Exception):
            focus = json.loads(raw_focus.decode() if isinstance(raw_focus, bytes) else raw_focus)
            assets = focus.get("assets", [])
            if assets:
                from lib.integrations.grok_helper import format_live_compact

                scanner_text = format_live_compact(assets)
                parts.append(f"\nSCANNER (focus assets):\n{scanner_text}")

    # Latest Grok analysis (gives the assistant prior AI commentary)
    raw_update = _redis_get("engine:grok_update")
    if raw_update:
        with contextlib.suppress(Exception):
            upd = json.loads(raw_update.decode() if isinstance(raw_update, bytes) else raw_update)
            text = upd.get("text", "")
            ts = upd.get("time_et", "")
            if text:
                parts.append(f"\nLATEST AI ANALYSIS ({ts}):\n{text[:600]}")

    # ICT levels
    raw_ict = _redis_get("engine:ict_summary")
    if raw_ict:
        with contextlib.suppress(Exception):
            ict = raw_ict.decode() if isinstance(raw_ict, bytes) else raw_ict
            parts.append(f"\nICT LEVELS:\n{ict[:400]}")

    # CVD
    raw_cvd = _redis_get("engine:cvd_summary")
    if raw_cvd:
        with contextlib.suppress(Exception):
            cvd = raw_cvd.decode() if isinstance(raw_cvd, bytes) else raw_cvd
            parts.append(f"\nCVD DELTA:\n{cvd[:200]}")

    # Positions
    raw_pos = _redis_get("engine:positions")
    if raw_pos:
        with contextlib.suppress(Exception):
            pos = json.loads(raw_pos.decode() if isinstance(raw_pos, bytes) else raw_pos)
            if pos:
                lines = []
                for p in pos[:6]:  # cap at 6
                    sym = p.get("symbol", "?")
                    side = p.get("side", "?")
                    qty = p.get("quantity", 0)
                    pnl = p.get("unrealized_pnl", 0)
                    lines.append(f"  {sym} {side} x{qty}  P&L: USD {pnl:+.2f}")
                parts.append("\nOPEN POSITIONS:\n" + "\n".join(lines))

    if len(parts) <= 2:
        return ""  # only time + session — not worth injecting

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM dispatch helpers  (AsyncOpenAI — no thread pools, no hand-rolled SSE)
# ---------------------------------------------------------------------------


def _get_xai_key() -> str:
    return os.environ.get("XAI_API_KEY", "")


def _make_ra_async_client() -> AsyncOpenAI | None:
    """Return an ``AsyncOpenAI`` pointed at RustAssistant, or None if unconfigured.

    The ``x-repo-id`` header is injected via ``default_headers`` so every
    request automatically carries the RAG context repo identifier.
    """
    base_url = os.environ.get("RA_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("RA_API_KEY", "")
    repo_id = os.environ.get("RA_REPO_ID", "")

    if not (base_url and api_key):
        return None

    extra: dict[str, str] = {}
    if repo_id:
        extra["x-repo-id"] = repo_id

    return AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        default_headers=extra,
        timeout=90.0,
        max_retries=1,
    )


def _make_grok_async_client() -> AsyncOpenAI | None:
    """Return an ``AsyncOpenAI`` pointed at xAI/Grok, or None if unconfigured."""
    api_key = _get_xai_key()
    if not api_key:
        return None
    return AsyncOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        timeout=90.0,
        max_retries=1,
    )


async def _call_chat_nonstreaming(
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """Send a non-streaming chat request via AsyncOpenAI.

    Returns ``(response_text, backend_name)``.
    Tries RustAssistant first, falls back to Grok.
    Raises ``RuntimeError`` when both backends fail.
    """
    from lib.integrations.grok_helper import GROK_MODEL

    kwargs: dict[str, Any] = {
        "model": GROK_MODEL,
        "messages": messages,  # type: ignore[arg-type]
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": MAX_TOKENS_CHAT,
    }

    # --- Primary: RustAssistant ---
    ra = _make_ra_async_client()
    if ra is not None:
        try:
            resp = await ra.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            logger.debug("chat non-stream: RA backend OK (%d chars)", len(content))
            return content, "ra"
        except APIConnectionError as exc:
            logger.warning("chat non-stream: RA connection error (%s) — falling back to Grok", exc)
        except APIStatusError as exc:
            logger.warning("chat non-stream: RA HTTP %s — falling back to Grok", exc.status_code)
        except Exception as exc:
            logger.warning("chat non-stream: RA unexpected error (%s) — falling back to Grok", exc)
        finally:
            await ra.close()

    # --- Fallback: direct xAI ---
    grok = _make_grok_async_client()
    if grok is None:
        raise RuntimeError("No LLM backend available: RA unreachable and XAI_API_KEY not set")

    try:
        resp = await grok.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        logger.debug("chat non-stream: Grok backend OK (%d chars)", len(content))
        return content, "grok"
    except Exception as exc:
        raise RuntimeError(f"Both RA and Grok failed: {exc}") from exc
    finally:
        await grok.close()


async def _stream_chat_async(
    messages: list[dict[str, str]],
) -> AsyncGenerator[tuple[str, str]]:
    """Stream a chat response via AsyncOpenAI, yielding ``(token, backend)`` tuples.

    The very first item is always ``("", backend_name)`` — a sentinel that
    tells the SSE generator which backend is active before real tokens flow.

    On RA failure before the first real token, transparently falls through
    to Grok and emits ``("RA_FALLBACK", "grok")`` as a switchover signal.
    """
    from lib.integrations.grok_helper import GROK_MODEL

    kwargs: dict[str, Any] = {
        "model": GROK_MODEL,
        "messages": messages,  # type: ignore[arg-type]
        "temperature": CHAT_TEMPERATURE,
        "max_tokens": MAX_TOKENS_CHAT,
    }

    # --- Primary: RustAssistant ---
    ra = _make_ra_async_client()
    if ra is not None:
        try:
            async with ra.chat.completions.stream(**kwargs) as stream:
                yield "", "ra"  # backend announcement
                had_token = False
                async for event in stream:
                    if event.type == "content.delta":
                        had_token = True
                        yield event.delta, "ra"  # type: ignore[attr-defined]
                if had_token:
                    return
                logger.warning("chat stream: RA returned empty stream — falling back to Grok")
        except APIConnectionError as exc:
            logger.warning("chat stream: RA connection error (%s) — falling back to Grok", exc)
        except APIStatusError as exc:
            logger.warning("chat stream: RA HTTP %s — falling back to Grok", exc.status_code)
        except Exception as exc:
            logger.warning("chat stream: RA error (%s) — falling back to Grok", exc)
        finally:
            await ra.close()

        yield "RA_FALLBACK", "grok"  # signal backend switch to SSE wrapper

    # --- Fallback: direct xAI ---
    grok = _make_grok_async_client()
    if grok is None:
        yield "ERROR: No LLM backend available — RA unreachable and XAI_API_KEY not set", "grok"
        return

    try:
        if ra is None:
            # RA was never attempted — emit the initial backend sentinel now
            yield "", "grok"
        async with grok.chat.completions.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content.delta":
                    yield event.delta, "grok"  # type: ignore[attr-defined]
    except APIConnectionError as exc:
        logger.error("chat stream: Grok connection error: %s", exc)
        yield f"ERROR: Grok connection error — {exc}", "grok"
    except APIStatusError as exc:
        logger.error("chat stream: Grok HTTP %s: %s", exc.status_code, exc.message)
        yield f"ERROR: Grok HTTP {exc.status_code}", "grok"
    except Exception as exc:
        logger.error("chat stream: Grok unexpected error: %s", exc)
        yield f"ERROR: {exc}", "grok"
    finally:
        await grok.close()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn history. Auto-generated if omitted.",
    )
    inject_context: bool = Field(
        default=True,
        description="Prepend live market context to the system prompt.",
    )
    clear_history: bool = Field(
        default=False,
        description="Clear existing history before this message.",
    )


class ChatResponse(BaseModel):
    session_id: str
    message: str
    backend: str
    elapsed_ms: int
    history_length: int


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    count: int


# ---------------------------------------------------------------------------
# SSE formatting helper
# ---------------------------------------------------------------------------


def _sse(data: str, event: str | None = None) -> str:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming single-turn chat with multi-turn history.

    Builds a full messages list from history + system context + new user
    message, calls the best available LLM backend, appends the exchange
    to history, and returns the full response in one JSON payload.
    """
    t0 = time.monotonic()

    session_id = req.session_id or str(uuid.uuid4())

    if req.clear_history:
        _clear_history(session_id)

    history = _load_history(session_id)

    # Build system prompt with optional market context
    system_content = _SYSTEM_PROMPT
    if req.inject_context:
        ctx = await asyncio.to_thread(_build_market_context)
        if ctx:
            system_content = f"{_SYSTEM_PROMPT}\n\n--- LIVE MARKET CONTEXT ---\n{ctx}\n---"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    try:
        response_text, backend = await _call_chat_nonstreaming(messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Persist to history (without system message)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": response_text})
    await asyncio.to_thread(_save_history, session_id, history)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return ChatResponse(
        session_id=session_id,
        message=response_text,
        backend=backend,
        elapsed_ms=elapsed_ms,
        history_length=len(history),
    )


@router.get("/sse/chat")
async def sse_chat(
    request: Request,
    message: str = Query(..., min_length=1, max_length=4000),
    session_id: str | None = Query(default=None),
    inject_context: bool = Query(default=True),
    clear_history: bool = Query(default=False),
):
    """Stream a chat response via Server-Sent Events.

    Query parameters mirror the POST /api/chat body so the browser's
    native EventSource API (GET-only) can be used directly.

    Event flow:
        chat-start      — {"session_id": "...", "backend": "ra|grok"}
        chat-token      — raw token string (may be multi-char)
        chat-heartbeat  — "." keep-alive every ~5s during slow generation
        chat-error      — {"error": "..."} on failure
        chat-done       — {"session_id":"...","chars":N,"backend":"...","elapsed_ms":N}
    """
    resolved_session_id = session_id or str(uuid.uuid4())

    return StreamingResponse(
        _sse_chat_generator(
            message=message,
            session_id=resolved_session_id,
            inject_context=inject_context,
            clear_history=clear_history,
            request=request,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _sse_chat_generator(
    message: str,
    session_id: str,
    inject_context: bool,
    clear_history: bool,
    request: Request,
) -> AsyncGenerator[str]:
    """Async generator that drives the SSE chat stream.

    Consumes ``_stream_chat_async`` directly — no thread pool, no queue,
    no blocking I/O.  The ``AsyncOpenAI`` client handles SSE framing and
    connection management internally.
    """
    t0 = time.monotonic()

    if clear_history:
        _clear_history(session_id)

    history = _load_history(session_id)

    # Build system prompt
    system_content = _SYSTEM_PROMPT
    if inject_context:
        ctx = await asyncio.to_thread(_build_market_context)
        if ctx:
            system_content = f"{_SYSTEM_PROMPT}\n\n--- LIVE MARKET CONTEXT ---\n{ctx}\n---"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    accumulated: list[str] = []
    backend = "ra"
    started = False
    had_error = False

    async for token, token_backend in _stream_chat_async(messages):
        # Check for client disconnect between tokens
        if await request.is_disconnected():
            logger.debug("SSE chat: client disconnected for session %s", session_id)
            return

        # First item is always the backend-announcement sentinel ("" or "RA_FALLBACK")
        if not started:
            started = True
            backend = token_backend
            yield _sse(
                json.dumps({"session_id": session_id, "backend": backend}),
                event="chat-start",
            )
            if token == "RA_FALLBACK":
                backend = "grok"
            # Either "" or "RA_FALLBACK" — no real content to emit yet
            if token in ("", "RA_FALLBACK"):
                continue

        if token.startswith("ERROR:"):
            had_error = True
            yield _sse(
                json.dumps({"error": token[len("ERROR:") :].strip()}),
                event="chat-error",
            )
            break

        accumulated.append(token)
        yield _sse(token, event="chat-token")

    # Persist completed exchange to history
    full_response = "".join(accumulated)
    if full_response and not had_error:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_response})
        _save_history(session_id, history)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    yield _sse(
        json.dumps(
            {
                "session_id": session_id,
                "chars": len(full_response),
                "backend": backend,
                "elapsed_ms": elapsed_ms,
                "had_error": had_error,
            }
        ),
        event="chat-done",
    )


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------


@router.get("/api/chat/history", response_model=HistoryResponse)
async def get_chat_history(
    session_id: str = Query(..., description="Session ID to fetch history for"),
):
    """Return stored conversation history for a session."""
    history = await asyncio.to_thread(_load_history, session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(role=m["role"], content=m["content"]) for m in history],
        count=len(history),
    )


@router.delete("/api/chat/history")
async def delete_chat_history(
    session_id: str = Query(..., description="Session ID to clear"),
):
    """Clear conversation history for a session."""
    await asyncio.to_thread(_clear_history, session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Status / health
# ---------------------------------------------------------------------------


@router.get("/api/chat/status")
async def chat_status():
    """Return the current LLM backend availability and configuration."""
    import httpx

    ra_base_url = os.environ.get("RA_BASE_URL", "").rstrip("/")
    ra_api_key = os.environ.get("RA_API_KEY", "")
    ra_repo_id = os.environ.get("RA_REPO_ID", "")
    ra_configured = bool(ra_base_url and ra_api_key)

    ra_reachable = False
    ra_latency_ms: int | None = None

    if ra_configured:
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                probe = await client.get(f"{ra_base_url}/health")
                ra_reachable = probe.status_code < 500
                ra_latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception:
            ra_latency_ms = None

    xai_configured = bool(os.environ.get("XAI_API_KEY", ""))

    return {
        "status": "ok",
        "backends": {
            "rust_assistant": {
                "configured": ra_configured,
                "base_url": ra_base_url or None,
                "repo_id": ra_repo_id or None,
                "reachable": ra_reachable,
                "latency_ms": ra_latency_ms,
            },
            "grok_fallback": {
                "configured": xai_configured,
            },
        },
        "active_backend": "ra" if (ra_reachable) else ("grok" if xai_configured else "none"),
        "model": "grok-4-1-fast-reasoning",
        "max_tokens": MAX_TOKENS_CHAT,
        "max_history_pairs": MAX_HISTORY_MESSAGES,
        "timestamp": datetime.now(tz=_ET).isoformat(),
    }
