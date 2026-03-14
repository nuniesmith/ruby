"""
Grok AI helper for live trading analysis.

Provides two main functions:
  1. run_morning_briefing()  — comprehensive pre-market game plan
  2. run_live_analysis()     — concise 15-minute update during active trading

Primary LLM backend: RustAssistant proxy (RA_BASE_URL / RA_API_KEY).
  - OpenAI-compatible ``/v1/chat/completions`` endpoint.
  - Optionally injects ``x-repo-id: <RA_REPO_ID>`` header for RAG context.
  - Falls back transparently to direct xAI / Grok on any error.

Fallback: xAI API (XAI_API_KEY) with grok-4-1-fast-reasoning model.
  - Used automatically when RA is unreachable, returns an HTTP error,
    or yields a stream error token.

Environment variables
---------------------
  RA_BASE_URL      Base URL of your RustAssistant instance, e.g.
                   ``http://oryx:3500`` (no trailing slash).
  RA_API_KEY       API key sent as ``Authorization: Bearer <key>``
                   (must match RA_PROXY_API_KEYS on the server).
  RA_REPO_ID       Optional repository/RAG context ID to inject.
                   E.g. ``futures-bot``.  Sent as ``x-repo-id`` header.
  XAI_API_KEY      Direct xAI API key — used only as fallback.

Cost when using RA proxy: depends on your RA server routing.
Cost when falling back to xAI: ~$0.007-0.01 per call, so a full trading
day (pre-market + ~16 live calls over 4 hours) costs well under $0.20.

Usage:
    from lib.grok_helper import run_live_analysis, run_morning_briefing, format_market_context

    # Build context from engine + scanner data
    context = format_market_context(engine, scanner_df, account_size, ...)

    # Pre-market
    briefing = run_morning_briefing(context, api_key)

    # Every 15 min during live trading
    update = run_live_analysis(context, api_key, previous_briefing=briefing)
"""

import logging
import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("grok_helper")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_MAX_TOKENS_BRIEFING = 3000
DEFAULT_MAX_TOKENS_LIVE = 800
DEFAULT_MAX_TOKENS_LIVE_COMPACT = 350
DEFAULT_MAX_TOKENS_DAILY_PLAN = 1500
DEFAULT_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# openai client factories  (RA primary → Grok fallback)
# ---------------------------------------------------------------------------
# Both RustAssistant and xAI expose an OpenAI-compatible
# /v1/chat/completions endpoint, so a single ``openai.OpenAI`` instance
# pointed at a different ``base_url`` handles both.  We build them lazily
# so import time is unaffected when the env vars are absent.
#
# RA extra headers (x-repo-id for RAG context) are injected via the
# ``default_headers`` kwarg so every request carries them automatically.
# ---------------------------------------------------------------------------


def _make_ra_client():
    """Return a configured ``openai.OpenAI`` for RustAssistant, or None."""
    from openai import OpenAI  # type: ignore[import-untyped]

    base_url = os.environ.get("RA_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("RA_API_KEY", "")
    repo_id = os.environ.get("RA_REPO_ID", "")

    if not (base_url and api_key):
        return None

    extra_headers: dict[str, str] = {}
    if repo_id:
        extra_headers["x-repo-id"] = repo_id

    return OpenAI(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        default_headers=extra_headers,
        timeout=90.0,
        max_retries=1,
    )


def _make_grok_client(api_key: str):
    """Return a configured ``openai.OpenAI`` for xAI/Grok direct access."""
    from openai import OpenAI  # type: ignore[import-untyped]

    return OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        timeout=90.0,
        max_retries=1,
    )


def _ra_available() -> bool:
    """True when RA_BASE_URL and RA_API_KEY are both set in the environment."""
    return bool(os.environ.get("RA_BASE_URL", "").strip() and os.environ.get("RA_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Unified LLM helpers  (RA primary → Grok fallback)
# ---------------------------------------------------------------------------


def _call_llm(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
) -> str | None:
    """Call the best available LLM backend and return the full response text.

    Attempt order:
        1. RustAssistant proxy  (RA_BASE_URL / RA_API_KEY)
        2. Direct xAI / Grok   (``api_key`` argument, i.e. XAI_API_KEY)

    Uses ``openai.OpenAI`` for both backends — retries, timeouts, and
    typed errors are all handled by the library.
    Returns ``None`` when both backends fail (errors are logged, not raised).
    """
    from openai import APIConnectionError, APIStatusError  # type: ignore[import-untyped]

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # --- Primary: RustAssistant ---
    if _ra_available():
        client = _make_ra_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=GROK_MODEL,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                logger.debug("_call_llm: RA backend OK (%d chars)", len(content or ""))
                return content
            except APIConnectionError as exc:
                logger.warning("_call_llm: RA connection error (%s) — falling back to Grok", exc)
            except APIStatusError as exc:
                logger.warning("_call_llm: RA HTTP %s — falling back to Grok", exc.status_code)
            except Exception as exc:
                logger.warning("_call_llm: RA unexpected error (%s) — falling back to Grok", exc)

    # --- Fallback: direct Grok ---
    return _call_grok(prompt, api_key, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt)


def _stream_llm(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
):
    """Stream from the best available LLM backend, yielding token strings.

    Attempt order:
        1. RustAssistant proxy — on any connection / HTTP error *before* the
           first token, falls through transparently to direct Grok.
        2. Direct xAI / Grok.

    Uses ``openai.OpenAI`` streaming (``stream=True``) for both backends,
    which handles SSE framing and ``[DONE]`` termination internally.
    Callers see a single seamless token stream regardless of which backend
    served the request.
    """
    from openai import APIConnectionError, APIStatusError  # type: ignore[import-untyped]

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # --- Primary: RustAssistant ---
    if _ra_available():
        client = _make_ra_client()
        if client is not None:
            try:
                with client.chat.completions.stream(
                    model=GROK_MODEL,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                ) as stream:
                    had_token = False
                    for event in stream:
                        if event.type == "content.delta":
                            had_token = True
                            yield event.delta  # type: ignore[attr-defined]
                    if had_token:
                        logger.debug("_stream_llm: RA backend served the stream")
                        return
                    # Empty stream — fall through to Grok
                    logger.warning("_stream_llm: RA returned empty stream — falling back to Grok")
            except APIConnectionError as exc:
                logger.warning("_stream_llm: RA connection error (%s) — falling back to Grok", exc)
            except APIStatusError as exc:
                logger.warning("_stream_llm: RA HTTP %s — falling back to Grok", exc.status_code)
            except Exception as exc:
                logger.warning("_stream_llm: RA error (%s) — falling back to Grok", exc)

    # --- Fallback: direct Grok ---
    logger.debug("_stream_llm: using direct Grok backend")
    yield from _stream_grok(
        prompt, api_key, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt
    )


# ---------------------------------------------------------------------------
# Backwards-compatible singleton shim
# ---------------------------------------------------------------------------
# Some call sites (api/chat.py, api/grok.py) import ``_ra_client`` directly
# to check ``.available`` or access ``._endpoint`` / ``._headers()``.
# We provide a lightweight shim so those imports continue to work without
# change while we migrate them to the openai-client helpers above.


class _RaClientShim:
    """Minimal shim that exposes the subset of RustAssistantClient used externally."""

    @property
    def available(self) -> bool:
        return _ra_available()

    @property
    def base_url(self) -> str:
        return os.environ.get("RA_BASE_URL", "").rstrip("/")

    @property
    def api_key(self) -> str:
        return os.environ.get("RA_API_KEY", "")

    @property
    def repo_id(self) -> str:
        return os.environ.get("RA_REPO_ID", "")

    # chat.py accesses _ra_client._endpoint for the raw URL
    @property
    def _endpoint(self) -> str:
        base = self.base_url
        return f"{base}/v1/chat/completions" if base else ""

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.repo_id:
            headers["x-repo-id"] = self.repo_id
        return headers


_ra_client = _RaClientShim()


def _call_grok(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
) -> str | None:
    """Call the Grok/xAI API directly and return the response text.

    Uses ``openai.OpenAI`` pointed at the xAI base URL.
    Returns None on error (logged, not raised).
    """
    from openai import APIConnectionError, APIStatusError  # type: ignore[import-untyped]

    if not api_key:
        logger.warning("No Grok API key provided")
        return None

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _make_grok_client(api_key)
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except APIConnectionError as exc:
        logger.error("Grok API connection error: %s", exc)
        return None
    except APIStatusError as exc:
        logger.error("Grok API HTTP %s: %s", exc.status_code, exc.message)
        return None
    except Exception as exc:
        logger.error("Grok API unexpected error: %s", exc)
        return None


def _stream_grok(
    prompt: str,
    api_key: str,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
):
    """Stream the Grok/xAI API response, yielding incremental text tokens.

    Uses ``openai.OpenAI`` streaming so SSE framing and ``[DONE]``
    termination are handled by the library.  On error, yields a single
    chunk starting with ``"ERROR: "`` so SSE consumers can surface it.

    Args:
        prompt: The user message to send.
        api_key: xAI API key (Bearer token).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (default 0.3).
        system_prompt: Optional system message prepended to the chat.

    Yields:
        str — incremental text fragments from the model.
    """
    from openai import APIConnectionError, APIStatusError  # type: ignore[import-untyped]

    if not api_key:
        yield "ERROR: No Grok API key configured"
        return

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _make_grok_client(api_key)
        with client.chat.completions.stream(
            model=GROK_MODEL,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    yield event.delta  # type: ignore[attr-defined]
    except APIConnectionError as exc:
        logger.error("Grok streaming API connection error: %s", exc)
        yield "ERROR: Grok API connection error"
    except APIStatusError as exc:
        logger.error("Grok streaming API HTTP %s: %s", exc.status_code, exc.message)
        yield f"ERROR: Grok API HTTP {exc.status_code}"
    except Exception as exc:
        logger.error("Grok streaming API unexpected error: %s", exc)
        yield f"ERROR: {exc}"


def stream_morning_briefing(context: dict, api_key: str):
    """Stream the pre-market morning briefing from the Grok API.

    Identical prompt to ``run_morning_briefing`` but uses the streaming
    endpoint so the dashboard can display tokens as they arrive.

    Yields:
        str — incremental text fragments.  The caller should accumulate
        them to reconstruct the full briefing for caching.
    """
    prompt = f"""Pre-market briefing for {context["time"]}.

Account: USD {context["account_size"]:,} | Risk/trade: USD {context["risk_dollars"]:,} | Max contracts: {context["max_contracts"]}
Session: {context["session_status"]}

CONTRACT SPECS:
{context["specs_text"]}

MARKET SCANNER (Last = current price):
{context["scanner_text"]}

OPTIMIZED STRATEGIES (auto-selected by engine):
{context["opt_text"]}

BACKTESTS (session-hours only):
{context["bt_text"]}

ICT LEVELS (FVGs, Order Blocks, Sweeps):
{context["ict_text"]}

CONFLUENCE (Multi-Timeframe):
{context["conf_text"]}

CVD (Volume Delta):
{context["cvd_text"]}

Ruby WAVE ANALYSIS (Bull/Bear wave dominance, trend speed, market phase):
{context.get("fks_wave_text", "Not available")}

Ruby VOLATILITY CLUSTERS (K-Means adaptive ATR, position sizing):
{context.get("fks_vol_text", "Not available")}

Ruby SIGNAL QUALITY (multi-factor score: vol sweet-spot, velocity, trend speed, candle patterns, HTF bias):
{context.get("fks_sq_text", "Not available")}

PRE-MARKET SCORES:
{context["scorer_text"]}

Give me today's game plan:
1. **Market Bias** — overall read on the session (1-2 sentences)
2. **Top 3 Focus Assets** — rank by setup quality, explain why each
3. **Key Levels to Watch** — entry zones, SL, TP for each focus asset (use scanner prices + ICT levels)
4. **Correlations** — what pairs to monitor together
5. **Risk Warnings** — anything that could trip us up today
6. **Session Plan** — when to be aggressive vs. patient
7. **Wave & Volatility Context** — note any assets with strong wave dominance (>1.5x ratio), high-vol clusters (widen stops / reduce size), or low-vol breakout setups
8. **Signal Quality** — highlight assets with high quality scores (>60%), note which have premium setup conditions, and flag any with poor quality (<40%) to avoid

Keep it actionable. No fluff. This is my reference card for the trading session."""

    yield from _stream_llm(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_BRIEFING,
        system_prompt=_MORNING_SYSTEM,
    )


def stream_live_analysis(
    context: dict,
    api_key: str,
    previous_briefing: str | None = None,
    update_number: int = 1,
):
    """Stream a live 15-minute market update from the Grok API.

    Uses the same compact prompt as ``_run_live_compact`` but streams the
    response token-by-token so the dashboard can render it progressively.

    Yields:
        str — incremental text fragments.
    """
    brief_section = ""
    if previous_briefing:
        brief_section = f"\nMORNING BRIEFING CONTEXT:\n{previous_briefing[:600]}\n"

    prompt = f"""Live update #{update_number} at {context["time"]}.
{brief_section}
CURRENT MARKET DATA:
{context["scanner_text"]}

ICT LEVELS: {context.get("ict_text", "N/A")}
CVD DELTA: {context.get("cvd_text", "N/A")}
SIGNAL QUALITY: {context.get("fks_sq_text", "N/A")}

What changed? What should I do right now? Keep it to ≤8 lines."""

    yield from _stream_llm(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_LIVE_COMPACT,
        system_prompt=_COMPACT_SYSTEM,
    )


# ---------------------------------------------------------------------------
# Market context builder
# ---------------------------------------------------------------------------


def format_market_context(
    engine,
    scanner_df,
    account_size: int,
    risk_dollars: int,
    max_contracts: int,
    contract_specs: dict,
    selected_assets: list[str],
    ict_summaries: dict | None = None,
    confluence_results: dict | None = None,
    cvd_summaries: dict | None = None,
    scorer_results: list[dict] | None = None,
    live_positions: dict | None = None,
    fks_wave_results: dict | None = None,
    fks_vol_results: dict | None = None,
    fks_signal_quality: dict | None = None,
) -> dict:
    """Build a structured context dict from all available market data.

    This context is passed to both morning briefing and live analysis
    so Grok has full visibility into the current market state.
    """
    now_est = datetime.now(tz=_EST)

    # Scanner text
    scanner_text = "No scanner data available"
    if scanner_df is not None and not scanner_df.empty:
        scanner_text = scanner_df.to_string(index=False)

    # Contract specs text
    specs_parts = []
    for asset_name, spec in contract_specs.items():
        data_ticker = spec.get("data_ticker", spec["ticker"])
        scan_price = "N/A"
        if scanner_df is not None and not scanner_df.empty:
            match = scanner_df.loc[scanner_df["Asset"] == asset_name, "Last"]
            if not match.empty:
                scan_price = str(match.iloc[0])
        specs_parts.append(
            f"  {asset_name} ({data_ticker}): "
            f"current_price={scan_price}, "
            f"tick_size={spec['tick']}, "
            f"point_value=USD {spec['point']}/point, "
            f"margin=USD {spec['margin']:,}"
        )
    specs_text = "\n".join(specs_parts)

    # Optimization results from engine cache
    from lib.core.cache import get_cached_optimization

    opt_parts = []
    from lib.core.models import ASSETS

    for name in selected_assets:
        ticker = ASSETS.get(name)
        if ticker:
            opt = get_cached_optimization(ticker, "5m", "5d")
            if opt:
                strat_label = opt.get("strategy_label", opt.get("strategy", "?"))
                confidence = opt.get("confidence", "?")
                regime = opt.get("regime", "?")
                opt_parts.append(
                    f"  {name}: strategy={strat_label}, "
                    f"return={opt.get('return_pct', '?')}%, "
                    f"sharpe={opt.get('sharpe', '?')}, "
                    f"win_rate={opt.get('win_rate', '?')}%, "
                    f"confidence={confidence}, regime={regime}"
                )
    opt_text = "\n".join(opt_parts) if opt_parts else "Not yet run"

    # Backtest results
    bt_results = engine.get_backtest_results() if engine else []
    bt_parts = []
    for r in bt_results:
        bt_parts.append(
            f"  {r['Asset']}: return={r['Return %']}%, "
            f"win_rate={r['Win Rate %']}%, "
            f"sharpe={r['Sharpe']}, trades={r['# Trades']}"
        )
    bt_text = "\n".join(bt_parts) if bt_parts else "Not yet run"

    # ICT summary text
    ict_text = "Not available"
    if ict_summaries:
        ict_parts = []
        for asset_name, summary in ict_summaries.items():
            stats = summary.get("stats", {})
            nearest = summary.get("nearest_levels", {})
            above = nearest.get("above", {})
            below = nearest.get("below", {})
            ict_parts.append(
                f"  {asset_name}: "
                f"unfilled_FVGs={stats.get('unfilled_fvgs', 0)}, "
                f"active_OBs={stats.get('active_obs', 0)}, "
                f"sweeps={stats.get('recent_sweeps', 0)}, "
                f"nearest_above={above.get('label', '—')} @ {above.get('price', '—')}, "
                f"nearest_below={below.get('label', '—')} @ {below.get('price', '—')}"
            )
        ict_text = "\n".join(ict_parts)

    # Confluence summary text
    conf_text = "Not available"
    if confluence_results:
        conf_parts = []
        for asset_name, conf in confluence_results.items():
            score = conf.get("score", 0)
            direction = conf.get("direction", "neutral")
            emoji = "🟢" if score >= 3 else "🟡" if score >= 2 else "🔴"
            conf_parts.append(f"  {asset_name}: {emoji} {score}/3 — bias={direction.upper()}")
        conf_text = "\n".join(conf_parts)

    # CVD summary text
    cvd_text = "Not available"
    if cvd_summaries:
        cvd_parts = []
        for asset_name, summary in cvd_summaries.items():
            bias = summary.get("bias", "neutral")
            slope = summary.get("cvd_slope", 0)
            cvd_parts.append(
                f"  {asset_name}: bias={bias}, slope={slope:+.3f}, delta={summary.get('delta_current', 0):,.0f}"
            )
        cvd_text = "\n".join(cvd_parts)

    # Scorer summary text
    scorer_text = "Not available"
    if scorer_results:
        scorer_parts = []
        for r in scorer_results:
            scorer_parts.append(f"  {r['asset']}: score={r['composite_score']:.0f}/100, signal={r['signal']}")
        scorer_text = "\n".join(scorer_parts)

    # Session status
    current_hour = now_est.hour
    if 3 <= current_hour < 10:
        session_status = "ACTIVE — primary entry window"
    elif 10 <= current_hour < 12:
        session_status = "WIND DOWN — manage only, no new entries"
    else:
        session_status = "CLOSED — no trading"

    # Live positions from broker
    positions_text = "No live positions"
    has_positions = False
    if live_positions and live_positions.get("has_positions"):
        has_positions = True
        pos_parts = []
        for p in live_positions.get("positions", []):
            symbol = p.get("symbol", "?")
            side = p.get("side", "?")
            qty = p.get("quantity", 0)
            avg = p.get("avgPrice", 0)
            upnl = p.get("unrealizedPnL", 0)
            pnl_emoji = "🟢" if upnl >= 0 else "🔴"
            pos_parts.append(f"  {symbol}: {side} x{qty} @ {avg:.2f} — {pnl_emoji} unrealized USD {upnl:+,.2f}")
        total_pnl = live_positions.get("total_unrealized_pnl", 0)
        acct_name = live_positions.get("account", "")
        positions_text = f"Account: {acct_name} | Total unrealized: USD {total_pnl:+,.2f}\n" + "\n".join(pos_parts)

    # Ruby Wave Analysis text
    fks_wave_text = "Not available"
    if fks_wave_results:
        wave_parts = []
        for asset_name, wave in fks_wave_results.items():
            bias = wave.get("bias", "NEUTRAL")
            bias_emoji = "🟢" if bias == "BULLISH" else "🔴" if bias == "BEARISH" else "⚪"
            wave_parts.append(
                f"  {asset_name}: {bias_emoji} {bias} — "
                f"wave_ratio={wave.get('wave_ratio_text', '?')}, "
                f"current={wave.get('current_ratio_text', '?')}, "
                f"dominance={wave.get('dominance_text', '?')}, "
                f"phase={wave.get('market_phase', '?')}, "
                f"momentum={wave.get('momentum_state', '?')}, "
                f"strength={wave.get('trend_strength', '?')}"
            )
        fks_wave_text = "\n".join(wave_parts)

    # Ruby Volatility Clustering text
    fks_vol_text = "Not available"
    if fks_vol_results:
        vol_parts = []
        for asset_name, vol in fks_vol_results.items():
            cluster = vol.get("cluster", "MEDIUM")
            cluster_emoji = "⚡" if cluster == "HIGH" else "🧘" if cluster == "LOW" else "〰️"
            vol_parts.append(
                f"  {asset_name}: {cluster_emoji} {cluster} cluster — "
                f"percentile={vol.get('percentile', 0):.0%}, "
                f"ATR={vol.get('raw_atr', 0):.4f}, "
                f"adaptive_ATR={vol.get('adaptive_atr', 0):.4f}, "
                f"position_size={vol.get('position_multiplier', 1.0)}x, "
                f"hint={vol.get('strategy_hint', '?')}"
            )
        fks_vol_text = "\n".join(vol_parts)

    # Ruby Signal Quality text
    fks_sq_text = "Not available"
    if fks_signal_quality:
        sq_parts = []
        for asset_name, sq in fks_signal_quality.items():
            score = sq.get("quality_pct", 0)
            hq = sq.get("high_quality", False)
            hq_emoji = "✅" if hq else "⚠️"
            context = sq.get("market_context", "?")
            direction = sq.get("trend_direction", "?")
            sq_parts.append(
                f"  {asset_name}: {hq_emoji} {score}% — "
                f"context={context}, direction={direction}, "
                f"RSI={sq.get('rsi', '?')}, AO={sq.get('ao', '?')}, "
                f"velocity={sq.get('normalized_velocity', '?')}"
            )
        fks_sq_text = "\n".join(sq_parts)

    return {
        "time": now_est.strftime("%Y-%m-%d %H:%M EST"),
        "account_size": account_size,
        "risk_dollars": risk_dollars,
        "max_contracts": max_contracts,
        "session_status": session_status,
        "scanner_text": scanner_text,
        "specs_text": specs_text,
        "opt_text": opt_text,
        "bt_text": bt_text,
        "ict_text": ict_text,
        "conf_text": conf_text,
        "cvd_text": cvd_text,
        "scorer_text": scorer_text,
        "positions_text": positions_text,
        "has_positions": has_positions,
        "fks_wave_text": fks_wave_text,
        "fks_vol_text": fks_vol_text,
        "fks_sq_text": fks_sq_text,
    }


# ---------------------------------------------------------------------------
# Morning briefing (pre-market)
# ---------------------------------------------------------------------------

_MORNING_SYSTEM = (
    "You are a disciplined futures trading analyst focused on micro contracts "
    "(MES, MNQ, MCL, MGC, MHG, SIL). You prioritize capital preservation and "
    "quality setups over quantity. Be concise, use bullet points, and always "
    "anchor prices to the scanner's 'Last' column. NEVER use bare $ signs — "
    "write 'USD' instead. Do not use LaTeX or math notation."
)

# ---------------------------------------------------------------------------
# Phase 3C: Structured Grok Daily Plan Analysis
# ---------------------------------------------------------------------------

_DAILY_PLAN_SYSTEM = (
    "You are a disciplined futures trading analyst. You will receive bias "
    "analysis and key levels for several assets. Your job is to provide "
    "structured macro context and asset-specific insights. Respond ONLY with "
    "valid JSON — no markdown fences, no commentary outside the JSON object. "
    "NEVER use bare $ signs — write 'USD' instead."
)


def run_morning_briefing(context: dict, api_key: str) -> str | None:
    """Generate a comprehensive pre-market game plan.

    Returns the formatted analysis text, or None on error.
    """
    prompt = f"""Pre-market briefing for {context["time"]}.

Account: USD {context["account_size"]:,} | Risk/trade: USD {context["risk_dollars"]:,} | Max contracts: {context["max_contracts"]}
Session: {context["session_status"]}

CONTRACT SPECS:
{context["specs_text"]}

MARKET SCANNER (Last = current price):
{context["scanner_text"]}

OPTIMIZED STRATEGIES (auto-selected by engine):
{context["opt_text"]}

BACKTESTS (session-hours only):
{context["bt_text"]}

ICT LEVELS (FVGs, Order Blocks, Sweeps):
{context["ict_text"]}

CONFLUENCE (Multi-Timeframe):
{context["conf_text"]}

CVD (Volume Delta):
{context["cvd_text"]}

Ruby WAVE ANALYSIS (Bull/Bear wave dominance, trend speed, market phase):
{context.get("fks_wave_text", "Not available")}

Ruby VOLATILITY CLUSTERS (K-Means adaptive ATR, position sizing):
{context.get("fks_vol_text", "Not available")}

Ruby SIGNAL QUALITY (multi-factor score: vol sweet-spot, velocity, trend speed, candle patterns, HTF bias):
{context.get("fks_sq_text", "Not available")}

PRE-MARKET SCORES:
{context["scorer_text"]}

Give me today's game plan:
1. **Market Bias** — overall read on the session (1-2 sentences)
2. **Top 3 Focus Assets** — rank by setup quality, explain why each
3. **Key Levels to Watch** — entry zones, SL, TP for each focus asset (use scanner prices + ICT levels)
4. **Correlations** — what pairs to monitor together
5. **Risk Warnings** — anything that could trip us up today
6. **Session Plan** — when to be aggressive vs. patient
7. **Wave & Volatility Context** — note any assets with strong wave dominance (>1.5x ratio), high-vol clusters (widen stops / reduce size), or low-vol breakout setups
8. **Signal Quality** — highlight assets with high quality scores (>60%), note which have premium setup conditions, and flag any with poor quality (<40%) to avoid

Keep it actionable. No fluff. This is my reference card for the trading session."""

    result = _call_llm(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_BRIEFING,
        system_prompt=_MORNING_SYSTEM,
    )
    return result


# ---------------------------------------------------------------------------
# Live trading analysis (every 15 minutes)
# ---------------------------------------------------------------------------

_LIVE_SYSTEM = (
    "You are a real-time futures trading analyst. Give concise, actionable "
    "15-minute updates. Use bullet points. Focus on what changed since last "
    "update and what to do about it. NEVER use bare $ signs — write 'USD' instead."
)


def run_live_analysis(
    context: dict,
    api_key: str,
    previous_briefing: str | None = None,
    previous_update: str | None = None,
    update_number: int = 1,
    compact: bool = True,
) -> str | None:
    """Generate a 15-minute market update during active trading.

    When compact=True (default), uses the simplified ≤8-line format.
    When compact=False, uses the original verbose format.

    This is designed to be cheap (~$0.007 per call) and fast.
    Returns formatted update text, or None on error.
    """
    if compact:
        return _run_live_compact(
            context=context,
            api_key=api_key,
            previous_briefing=previous_briefing,
            update_number=update_number,
        )

    return _run_live_verbose(
        context=context,
        api_key=api_key,
        previous_briefing=previous_briefing,
        previous_update=previous_update,
        update_number=update_number,
    )


# ---------------------------------------------------------------------------
# Compact live update — ≤8 lines total
# ---------------------------------------------------------------------------

_COMPACT_SYSTEM = (
    "You are a real-time futures trading analyst. Respond in EXACTLY this format with no extra lines:\n"
    "Line 1-N: One status line per focus asset: SYMBOL EMOJI PRICE (CHANGE) | Bias STATUS | Watch LEVEL\n"
    "Line N+1: blank line\n"
    "Line N+2: DO NOW: one clear actionable sentence\n\n"
    "Rules:\n"
    "- Use 🟢 for bullish/valid bias, 🔴 for bearish, ⚪ for neutral/invalid\n"
    "- CHANGE is the price move since prior check (e.g. +4, -12)\n"
    "- Bias STATUS is VALID or INVALID (did price action confirm or break the plan?)\n"
    "- Watch LEVEL is the single most important price level right now\n"
    "- DO NOW must be 1 sentence: hold, enter, exit, tighten stop, or wait\n"
    "- NEVER use bare $ signs — write USD instead\n"
    "- Total output MUST be 8 lines or fewer"
)


def _run_live_compact(
    context: dict,
    api_key: str,
    previous_briefing: str | None = None,
    update_number: int = 1,
) -> str | None:
    """Generate a compact ≤8-line live update.

    Format per asset:
        GOLD 🟢 2712 (+4) | Bias VALID | Watch 2725
        MNQ  🔴 21450 (-38) | Bias INVALID | Watch 21400

    Final line:
        DO NOW: Hold GOLD long, tighten stop to 2700; avoid MNQ — bias broken.
    """
    plan_ref = ""
    if previous_briefing:
        plan_ref = f"MORNING PLAN (brief):\n{previous_briefing[:400]}\n"

    positions_block = ""
    if context.get("has_positions"):
        positions_block = f"LIVE POSITIONS:\n{context['positions_text']}\n"

    prompt = f"""Update #{update_number} — {context["time"]}
Account: USD {context["account_size"]:,} | Session: {context["session_status"]}
{plan_ref}{positions_block}
SCANNER: {context["scanner_text"]}
ICT: {context["ict_text"]}
WAVE: {context.get("fks_wave_text", "N/A")}
SIGNAL QUALITY: {context.get("fks_sq_text", "N/A")}
CONFLUENCE: {context["conf_text"]}

Respond with exactly one status line per focus asset, then a blank line, then one DO NOW line.
Example format:
GOLD 🟢 2712 (+4) | Bias VALID | Watch 2725
MNQ 🔴 21450 (-38) | Bias INVALID | Watch 21400

DO NOW: Hold GOLD long, stop to 2700; skip MNQ — bias broken."""

    result = _call_grok(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_LIVE_COMPACT,
        system_prompt=_COMPACT_SYSTEM,
    )
    if result:
        result = _enforce_compact_limit(result)
    return result


def _enforce_compact_limit(text: str, max_lines: int = 8) -> str:
    """Ensure the output is at most max_lines lines, trimming if needed."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return text.strip()
    # Keep first (max_lines - 2) status lines + blank + last "DO NOW" line
    # Find the DO NOW line
    do_now_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("DO NOW"):
            do_now_idx = i
            break
    if do_now_idx is not None:
        status_lines = [ln for ln in lines[:do_now_idx] if ln.strip()]
        do_now_line = lines[do_now_idx]
        # Keep at most (max_lines - 2) status lines + blank + DO NOW
        max_status = max_lines - 2
        kept = status_lines[:max_status]
        return "\n".join(kept) + "\n\n" + do_now_line
    # Fallback: just truncate
    return "\n".join(lines[:max_lines])


def format_live_compact(
    focus_assets: list[dict],
    do_now: str = "",
) -> str:
    """Build a compact ≤8-line status string from focus asset data.

    This is the *local* formatter — used when Grok is unavailable or
    as a fast fallback.  Each focus asset gets one status line; the
    final line is the DO NOW action.

    Args:
        focus_assets: List of asset focus dicts (from compute_daily_focus).
        do_now: Optional action line. Auto-generated if empty.

    Returns:
        Formatted string, ≤8 lines.
    """
    lines: list[str] = []

    for asset in focus_assets[:5]:  # max 5 assets to stay ≤8 lines
        symbol = asset.get("symbol", "?")
        bias = asset.get("bias", "NEUTRAL")
        emoji = {"LONG": "🟢", "SHORT": "🔴"}.get(bias, "⚪")
        price = asset.get("last_price", 0)
        quality = asset.get("quality_pct", 0)
        skip = asset.get("skip", False)

        # Determine bias validity from quality
        bias_status = "VALID" if quality >= 55 and not skip else "INVALID"

        # Key watch level: TP1 for valid bias, stop for invalid
        watch = asset.get("tp1", 0) if bias_status == "VALID" else asset.get("stop", 0)

        # Pad symbol for alignment
        sym_padded = f"{symbol:<5s}"
        lines.append(f"{sym_padded} {emoji} {price:,.2f} | Bias {bias_status} | Watch {watch:,.2f}")

    # Blank separator
    lines.append("")

    # DO NOW line
    if not do_now:
        tradeable = [a for a in focus_assets if not a.get("skip")]
        if not tradeable:
            do_now = "DO NOW: No quality setups — stand aside and wait."
        elif len(tradeable) == 1:
            t = tradeable[0]
            do_now = (
                f"DO NOW: Focus on {t['symbol']} {t.get('bias', 'NEUTRAL')} "
                f"near {t.get('entry_low', 0):,.2f}–{t.get('entry_high', 0):,.2f}."
            )
        else:
            symbols = ", ".join(t["symbol"] for t in tradeable[:2])
            do_now = f"DO NOW: Watch {symbols} for entries at planned levels."

    lines.append(do_now)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verbose live update (original format, available via compact=False)
# ---------------------------------------------------------------------------


def _run_live_verbose(
    context: dict,
    api_key: str,
    previous_briefing: str | None = None,
    previous_update: str | None = None,
    update_number: int = 1,
) -> str | None:
    """Generate the original verbose 15-minute market update.

    This is the full-detail format for pre-market or when toggled.
    """
    # Include a summary of the morning plan for continuity
    plan_ref = ""
    if previous_briefing:
        # Take first 500 chars of the morning briefing as context
        plan_ref = f"\nMORNING PLAN SUMMARY (reference):\n{previous_briefing[:500]}...\n"

    prev_ref = ""
    if previous_update:
        prev_ref = f"\nLAST UPDATE:\n{previous_update[:300]}...\n"

    # Include live positions if available
    positions_block = ""
    if context.get("has_positions"):
        positions_block = f"""
LIVE POSITIONS (from broker):
{context["positions_text"]}
"""

    prompt = f"""15-minute market update #{update_number} — {context["time"]}

Account: USD {context["account_size"]:,} | Session: {context["session_status"]}
{plan_ref}{prev_ref}{positions_block}
CURRENT SCANNER:
{context["scanner_text"]}

ICT LEVELS:
{context["ict_text"]}

CVD:
{context["cvd_text"]}

CONFLUENCE:
{context["conf_text"]}

WAVE ANALYSIS:
{context.get("fks_wave_text", "Not available")}

VOLATILITY CLUSTERS:
{context.get("fks_vol_text", "Not available")}

SIGNAL QUALITY (multi-factor: vol sweet-spot, velocity, trend speed, candle patterns, HTF bias):
{context.get("fks_sq_text", "Not available")}

Give me a quick update (5-8 bullet points max):
- What moved since last check? Any significant price action?
- Are our focus assets still in play? Any setups triggered or invalidated?
- CVD/volume delta shifts — is buying or selling pressure changing?
- Any new ICT levels hit (FVGs filled, OBs tested, liquidity swept)?"""

    # Add position-specific prompts when positions are open
    if context.get("has_positions"):
        prompt += """
- POSITION CHECK — how are our open positions doing relative to key levels?
- Should we hold, scale, trail stops, or exit any positions?
- Risk check — unrealized P&L vs daily drawdown limit, approaching session wind-down?"""
    else:
        prompt += """
- Risk check — are we approaching session wind-down or any danger zones?"""

    prompt += """
- One-line summary: what to do RIGHT NOW

Be extremely concise. This is a quick check-in, not a full analysis."""

    result = _call_llm(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_LIVE,
        system_prompt=_LIVE_SYSTEM,
    )
    return result


# ---------------------------------------------------------------------------
# Grok session manager (tracks state across 15-min intervals)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 3C: Structured Grok analysis for DailyPlan integration
# ---------------------------------------------------------------------------


def run_daily_plan_grok_analysis(
    biases: dict,
    asset_names: list[str],
    swing_candidate_names: list[str] | None = None,
    scalp_focus_names: list[str] | None = None,
    account_size: int = 50_000,
    api_key: str | None = None,
) -> dict | None:
    """Generate structured Grok analysis for the daily plan.

    Phase 3C: Called during daily plan generation to get macro context,
    asset-specific insights, and risk warnings in a structured format
    that can be parsed and stored on the DailyPlan dataclass.

    Unlike the free-text morning briefing, this returns a parsed dict
    with well-defined fields the dashboard and engine can consume
    programmatically.

    Args:
        biases: {asset_name: DailyBias} or {asset_name: dict} from bias analyzer.
        asset_names: All tracked asset names.
        swing_candidate_names: Assets selected as swing candidates.
        scalp_focus_names: Assets selected for scalp focus.
        account_size: Account size for context.
        api_key: xAI API key. If None, reads from XAI_API_KEY env var.

    Returns:
        Parsed dict with structured fields, or None on failure.
        Keys:
          - macro_bias: "risk_on" | "risk_off" | "mixed"
          - macro_summary: str (1-2 sentences)
          - top_assets: list[dict] with {name, reason, key_level, bias_agreement}
          - risk_warnings: list[str]
          - economic_events: list[str]
          - session_plan: str (1-2 sentences on when to be aggressive vs patient)
          - correlation_notes: list[str]
          - swing_insights: dict[str, str] — per-asset swing trade notes
          - raw_text: str — the full Grok response for logging/display
    """
    import os

    if api_key is None:
        api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return None

    now = datetime.now(tz=_EST)

    # Build bias summary for the prompt
    bias_lines: list[str] = []
    for name in asset_names[:10]:
        b = biases.get(name)
        if b is None:
            continue
        # Support both DailyBias objects and dicts
        if hasattr(b, "direction"):
            direction = b.direction.value if hasattr(b.direction, "value") else str(b.direction)
            confidence = b.confidence
            reasoning = b.reasoning[:100] if b.reasoning else ""
            key_lvl = ""
            if hasattr(b, "key_levels") and b.key_levels:
                kl = b.key_levels
                if hasattr(kl, "prior_day_high"):
                    key_lvl = f"PDH={kl.prior_day_high:.2f} PDL={kl.prior_day_low:.2f} WH={kl.weekly_high:.2f}"
        else:
            direction = b.get("direction", "NEUTRAL")
            confidence = b.get("confidence", 0.0)
            reasoning = b.get("reasoning", "")[:100]
            kl_d = b.get("key_levels", {})
            key_lvl = (
                f"PDH={kl_d.get('prior_day_high', 0):.2f} "
                f"PDL={kl_d.get('prior_day_low', 0):.2f} "
                f"WH={kl_d.get('weekly_high', 0):.2f}"
            )
        bias_lines.append(
            f"  {name}: {direction} ({confidence:.0%}) — {reasoning}" + (f" | {key_lvl}" if key_lvl else "")
        )

    bias_block = "\n".join(bias_lines) if bias_lines else "No bias data available"

    # Swing / scalp context
    swing_names_str = ", ".join(swing_candidate_names) if swing_candidate_names else "None selected yet"
    scalp_names_str = ", ".join(scalp_focus_names) if scalp_focus_names else "None selected yet"

    prompt = (
        f"Daily plan analysis for {now.strftime('%A %B %d, %Y')} "
        f"({now.strftime('%H:%M')} ET).\n\n"
        f"Account: USD {account_size:,}\n"
        f"Scalp focus assets: {scalp_names_str}\n"
        f"Swing candidates: {swing_names_str}\n\n"
        "My system's directional bias for today:\n"
        f"{bias_block}\n\n"
        "Respond with a JSON object (no markdown fences) with these exact keys:\n"
        "{\n"
        '  "macro_bias": "risk_on" | "risk_off" | "mixed",\n'
        '  "macro_summary": "1-2 sentence macro read for the day",\n'
        '  "top_assets": [\n'
        '    {"name": "Gold", "reason": "why this asset today", '
        '"key_level": "2725.0 PDH", "bias_agreement": true}\n'
        "  ],\n"
        '  "risk_warnings": ["warning 1", "warning 2"],\n'
        '  "economic_events": ["event 1 at HH:MM ET", "event 2"],\n'
        '  "session_plan": "When to be aggressive vs patient today",\n'
        '  "correlation_notes": ["Gold/Silver moving together", "ES/NQ diverging"],\n'
        '  "swing_insights": {"asset_name": "specific swing setup note"}\n'
        "}\n\n"
        "Rules:\n"
        "- top_assets: rank by setup quality, max 4 assets\n"
        "- bias_agreement: true if you agree with my system's bias, false if not\n"
        "- risk_warnings: real risks for today (FOMC, data releases, geopolitics)\n"
        "- economic_events: only events that could move futures today\n"
        "- swing_insights: for each swing candidate, note the best entry approach\n"
        "- Be concise and actionable. No fluff."
    )

    raw_text = _call_grok(
        prompt,
        api_key,
        max_tokens=DEFAULT_MAX_TOKENS_DAILY_PLAN,
        system_prompt=_DAILY_PLAN_SYSTEM,
    )

    if not raw_text:
        return None

    # Parse the JSON response
    parsed = parse_grok_daily_plan_response(raw_text)
    if parsed is not None:
        parsed["raw_text"] = raw_text
    return parsed


def parse_grok_daily_plan_response(text: str) -> dict | None:
    """Parse a Grok daily plan response into a structured dict.

    Handles common response quirks:
      - Markdown code fences (```json ... ```)
      - Leading/trailing whitespace
      - Partial JSON (best-effort extraction)

    Returns:
        Parsed dict with validated fields, or None on failure.
    """
    import json as _json
    import re

    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strip markdown code fences if present
    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct JSON parse first
    try:
        data = _json.loads(cleaned)
        return _validate_grok_plan_response(data)
    except _json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text (Grok sometimes adds preamble)
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        json_candidate = cleaned[brace_start : brace_end + 1]
        try:
            data = _json.loads(json_candidate)
            return _validate_grok_plan_response(data)
        except _json.JSONDecodeError:
            pass

    # Last resort: try to fix common JSON issues
    # (trailing commas, single quotes, unquoted keys)
    try:
        # Replace single quotes with double quotes (very rough)
        fixed = cleaned.replace("'", '"')
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        brace_start = fixed.find("{")
        brace_end = fixed.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            data = _json.loads(fixed[brace_start : brace_end + 1])
            return _validate_grok_plan_response(data)
    except (ValueError, _json.JSONDecodeError):
        pass

    logger.warning(
        "Failed to parse Grok daily plan JSON response (%d chars). First 200 chars: %s",
        len(text),
        text[:200],
    )

    # Return a minimal dict with just the raw text so the caller
    # can still display it even if structured parsing failed
    return {
        "macro_bias": "mixed",
        "macro_summary": text[:300] if len(text) > 300 else text,
        "top_assets": [],
        "risk_warnings": [],
        "economic_events": [],
        "session_plan": "",
        "correlation_notes": [],
        "swing_insights": {},
        "_parse_failed": True,
    }


def _validate_grok_plan_response(data: dict) -> dict:
    """Validate and normalize the parsed Grok daily plan response.

    Ensures all expected keys exist with correct types, fills defaults
    for missing fields.
    """
    result: dict[str, Any] = {
        "macro_bias": "mixed",
        "macro_summary": "",
        "top_assets": [],
        "risk_warnings": [],
        "economic_events": [],
        "session_plan": "",
        "correlation_notes": [],
        "swing_insights": {},
    }

    # macro_bias
    mb = data.get("macro_bias", "mixed")
    if isinstance(mb, str) and mb.lower() in ("risk_on", "risk_off", "mixed"):
        result["macro_bias"] = mb.lower()

    # macro_summary
    ms = data.get("macro_summary", "")
    if isinstance(ms, str):
        result["macro_summary"] = ms[:500]

    # top_assets
    ta = data.get("top_assets", [])
    if isinstance(ta, list):
        validated_assets = []
        for item in ta[:6]:
            if isinstance(item, dict):
                validated_assets.append(
                    {
                        "name": str(item.get("name", "")),
                        "reason": str(item.get("reason", "")),
                        "key_level": str(item.get("key_level", "")),
                        "bias_agreement": bool(item.get("bias_agreement", True)),
                    }
                )
        result["top_assets"] = validated_assets

    # risk_warnings
    rw = data.get("risk_warnings", [])
    if isinstance(rw, list):
        result["risk_warnings"] = [str(w)[:200] for w in rw[:10] if w]

    # economic_events
    ee = data.get("economic_events", [])
    if isinstance(ee, list):
        result["economic_events"] = [str(e)[:200] for e in ee[:10] if e]

    # session_plan
    sp = data.get("session_plan", "")
    if isinstance(sp, str):
        result["session_plan"] = sp[:300]

    # correlation_notes
    cn = data.get("correlation_notes", [])
    if isinstance(cn, list):
        result["correlation_notes"] = [str(n)[:200] for n in cn[:10] if n]

    # swing_insights
    si = data.get("swing_insights", {})
    if isinstance(si, dict):
        result["swing_insights"] = {str(k): str(v)[:300] for k, v in si.items()}

    return result


def format_grok_daily_plan_for_display(grok_data: dict) -> str:
    """Format a structured Grok daily plan response for dashboard display.

    Converts the parsed JSON dict back into a human-readable summary
    suitable for the Grok panel or daily plan header card.

    Args:
        grok_data: Parsed dict from run_daily_plan_grok_analysis().

    Returns:
        Formatted multi-line string for display.
    """
    if not grok_data:
        return ""

    lines: list[str] = []

    # Macro bias
    macro_emoji = {
        "risk_on": "🟢 RISK-ON",
        "risk_off": "🔴 RISK-OFF",
        "mixed": "🟡 MIXED",
    }
    bias = grok_data.get("macro_bias", "mixed")
    lines.append(f"📊 Macro: {macro_emoji.get(bias, bias)}")

    summary = grok_data.get("macro_summary", "")
    if summary:
        lines.append(f"   {summary}")

    # Top assets
    top_assets = grok_data.get("top_assets", [])
    if top_assets:
        lines.append("")
        lines.append("🎯 Top Assets:")
        for i, a in enumerate(top_assets, 1):
            agree = "✅" if a.get("bias_agreement") else "⚠️"
            lines.append(f"   {i}. {a.get('name', '?')} {agree} — {a.get('reason', '')} [{a.get('key_level', '')}]")

    # Economic events
    events = grok_data.get("economic_events", [])
    if events:
        lines.append("")
        lines.append("📅 Events:")
        for e in events:
            lines.append(f"   • {e}")

    # Risk warnings
    warnings = grok_data.get("risk_warnings", [])
    if warnings:
        lines.append("")
        lines.append("⚠️ Risks:")
        for w in warnings:
            lines.append(f"   • {w}")

    # Session plan
    plan = grok_data.get("session_plan", "")
    if plan:
        lines.append("")
        lines.append(f"📋 Session: {plan}")

    # Correlation notes
    corr = grok_data.get("correlation_notes", [])
    if corr:
        lines.append("")
        lines.append("🔗 Correlations:")
        for c in corr:
            lines.append(f"   • {c}")

    # Swing insights
    swing = grok_data.get("swing_insights", {})
    if swing:
        lines.append("")
        lines.append("📈 Swing Notes:")
        for asset, note in swing.items():
            lines.append(f"   {asset}: {note}")

    return "\n".join(lines)


class GrokSession:
    """Manages Grok analysis state across a trading session.

    Tracks timing, stores briefings/updates, and handles the 15-minute
    interval logic so the dashboard stays simple.
    """

    LIVE_INTERVAL_SEC = 900  # 15 minutes

    def __init__(self):
        self.morning_briefing: str | None = None
        self.updates: list[dict] = []  # [{time, text, number}]
        self.last_update_time: float = 0
        self.is_active: bool = False
        self.total_calls: int = 0
        self.estimated_cost: float = 0.0

    def activate(self) -> None:
        """Start the live analysis session."""
        self.is_active = True
        # Force first update on next check
        self.last_update_time = 0
        logger.info("Grok live session activated")

    def deactivate(self) -> None:
        """Stop the live analysis session."""
        self.is_active = False
        logger.info(
            "Grok live session deactivated after %d updates (est. cost: $%.4f)",
            len(self.updates),
            self.estimated_cost,
        )

    def set_morning_briefing(self, text: str) -> None:
        """Store the morning briefing for reference during live updates."""
        self.morning_briefing = text
        self.total_calls += 1
        self.estimated_cost += 0.008  # ~$0.008 per briefing call

    def needs_update(self) -> bool:
        """Check if enough time has passed for the next 15-min update."""
        if not self.is_active:
            return False
        return (time.time() - self.last_update_time) >= self.LIVE_INTERVAL_SEC

    def run_update(self, context: dict, api_key: str, compact: bool = True) -> str | None:
        """Run a live analysis update if the interval has elapsed.

        Args:
            context: Market context dict from format_market_context().
            api_key: Grok API key.
            compact: If True (default), use ≤8-line compact format.
                     If False, use original verbose format.

        Returns the update text, or None if not yet time or on error.
        """
        if not self.needs_update():
            return None

        update_number = len(self.updates) + 1
        previous_update = self.updates[-1]["text"] if self.updates else None

        result = run_live_analysis(
            context=context,
            api_key=api_key,
            previous_briefing=self.morning_briefing,
            previous_update=previous_update,
            update_number=update_number,
            compact=compact,
        )

        if result:
            self.updates.append(
                {
                    "time": datetime.now(tz=_EST).strftime("%H:%M EST"),
                    "text": result,
                    "number": update_number,
                    "compact": compact,
                }
            )
            self.last_update_time = time.time()
            self.total_calls += 1
            self.estimated_cost += 0.005 if compact else 0.007
            logger.info(
                "Grok live update #%d completed (%s)",
                update_number,
                "compact" if compact else "verbose",
            )

        return result

    def get_latest_update(self) -> dict | None:
        """Return the most recent update dict, or None."""
        return self.updates[-1] if self.updates else None

    def get_session_summary(self) -> dict:
        """Return a summary of this Grok session."""
        return {
            "is_active": self.is_active,
            "has_briefing": self.morning_briefing is not None,
            "total_updates": len(self.updates),
            "total_calls": self.total_calls,
            "estimated_cost": round(self.estimated_cost, 4),
            "last_update": self.updates[-1]["time"] if self.updates else None,
        }
