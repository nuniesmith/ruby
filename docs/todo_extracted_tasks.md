# Ruby Futures — Extracted Tasks from `todo/` Directory

> **Generated from**: `todo/` directory review (13 files)
> **Purpose**: Consolidate all ideas, prototypes, and rough notes into actionable tasks
> **Disposition**: After review, merge relevant items into `todo.md` and `docs/backlog.md`, then delete `todo/`

---

## Source File Inventory

| File | Type | Status | Disposition |
|------|------|--------|-------------|
| `todo/README.md` | Integration guide for trading SPA prototype | ✅ **Already done** — `pipeline.py` wires real modules | Delete |
| `todo/notes.md` | Original feature request / vision statement | ✅ **Already done** — trading dashboard is integrated | Delete |
| `todo/trading_webui_review.md` | 6-page UI blueprint + backend audit | ⚠️ **Partially done** — extract remaining ideas | Delete after extraction |
| `todo/app.py` | Standalone FastAPI prototype (v1) | ✅ **Already done** — integrated into `pipeline.py` | Delete |
| `todo/app1.py` | Standalone FastAPI prototype (v2 with position routes) | ⚠️ **Partially done** — position intelligence routes not yet integrated | Delete after extraction |
| `todo/index.html` | Full SPA prototype (ApexCharts) | ✅ **Already done** — copied to `static/trading.html` | Delete |
| `todo/live_page.html` | Enhanced live page with L2 DOM + position intel | 🔴 **Not done** — new feature, extract tasks | Delete after extraction |
| `todo/live_page1.html` | Same as `live_page.html` (duplicate) | 🔴 Duplicate of above | Delete |
| `todo/trading-dashboard.jsx` | React prototype with mock data | ❌ **Not using React** — but has good mock data patterns | Delete |
| `todo/position_engine.py` | Position intelligence engine (Rithmic L1/L2/DOM) | 🔴 **Not done** — significant new feature | Delete after extraction |
| `todo/position_engine1.py` | Duplicate of `position_engine.py` | 🔴 Duplicate | Delete |
| `todo/data_news.md` | News sentiment pipeline (Finnhub + Alpha Vantage + Grok) | 🔴 **Not done** — new feature, complements Reddit phase | Delete after extraction |
| `todo/requirements.txt` | 3 deps for standalone prototype | ✅ Already in project deps | Delete |

---

## Extracted Tasks — Organised by Phase

### Phase POSINT — Position Intelligence Engine (from `position_engine.py` + `live_page.html`)

> **Priority**: 🔴 High — this is the core "live trading co-pilot" feature.
> **Depends on**: Rithmic credentials (can build with mock data first, swap later).
> **Source files**: `todo/position_engine.py`, `todo/live_page.html`, `todo/app1.py`

The position engine prototype computes real-time intelligence per open position:
- L1 best bid/ask, spread, VWAP distance
- L2 depth-of-market with visual DOM ladder
- Book pressure analysis (absorbing / distributed / opposed)
- Sweep zone detection (thin DOM spots where stops could be run)
- Multi-TP zone calculation (plan-aware + Fibonacci + liquidity targets)
- Sweep-aware breakeven (places BE stop past DOM clusters, not at entry)
- Risk action recommendations (hold / scale / move stop / exit)
- Live analysis signals (CVD, VWAP, DOM pressure, 15m momentum)
- Time & sales tape

#### Phase POSINT-A: Position Intelligence Module

- [ ] **`src/lib/services/engine/position_intelligence.py`** — new module
  - Port `compute_position_intelligence()` from `todo/position_engine.py`
  - Port helper functions: `_compute_sweep_zones()`, `_compute_buffer_be()`, `_compute_tp_zones()`, `_compute_risk_actions()`, `_gen_live_analysis_signals()`
  - Add `SPECS` instrument dictionary (tick size, tick value, multiplier) — or pull from existing `asset_registry.py`
  - Wire real analysis modules where `TODO` markers exist:
    - `ict.py` → liquidity levels for sweep zone cross-reference
    - `confluence.py` → scored confluence at each TP zone
    - `volume_profile.py` → VAH/VAL/POC as TP targets
    - `cvd.py` → real CVD divergence detection
    - `regime.py` → current regime classification
    - `signal_quality.py` → entry/exit quality score
    - `mtf_analyzer.py` → real-time bias signal
  - Input: position dict, L1 quote, L2 book, optional plan data
  - Output: `PositionIntelligence` dataclass with all computed fields
  - **Mock mode**: when Rithmic is not connected, generate realistic mock data (functions already written in prototype: `_mock_positions()`, `_mock_l1()`, `_mock_l2()`, `_mock_trades_tape()`)

#### Phase POSINT-B: Rithmic Position Engine Wrapper

- [ ] **`src/lib/services/engine/rithmic_position_engine.py`** — new module
  - Port `RithmicPositionEngine` class from `todo/position_engine.py`
  - Methods: `connect()`, `get_positions()`, `get_l1(symbol)`, `get_l2(symbol, depth)`, `get_recent_trades(symbol, n)`
  - Each method has a `TODO` comment showing the exact `rithmic_client.py` call to make when creds arrive
  - Mock fallbacks for each method (already implemented in prototype)
  - Connection state tracking: `connected`, `last_sync`, `error_count`
  - Auto-reconnect logic with exponential backoff

#### Phase POSINT-C: Position Intelligence API Routes

- [ ] **Add routes to `src/lib/services/data/api/pipeline.py`** (or new `position_api.py`)
  - `GET /api/live/positions` — SSE stream of position intelligence updates (1.5s interval)
    - Port `position_intelligence_stream()` generator from prototype
    - Streams: `position_update`, `no_positions`, `error` event types
    - Each position update includes full intelligence payload (book, TPs, risk actions, signals)
  - `GET /api/live/book?symbol=MES` — snapshot of L1 + L2 book for a symbol
  - `GET /api/live/tape?symbol=MES&n=20` — recent time & sales
  - `GET /api/live/positions/snapshot` — non-SSE snapshot of all positions
  - Wire corresponding web service proxy routes in `src/lib/services/web/main.py`

#### Phase POSINT-D: Live Page UI Enhancement

- [ ] **Update `static/trading.html`** — enhance the Live Trading page
  - Replace the current simulated price chart with position intelligence cards
  - Port CSS from `todo/live_page.html` (`.pos-intel-card`, `.dom-grid`, `.tp-zone`, `.risk-action`, etc.)
  - Per-position card layout (from prototype):
    - **Header**: symbol, direction badge, entry price, live price, unrealized P&L, quantity
    - **Column 1 — Book**: L1 bid/ask with size, spread indicator, time & sales tape
    - **Column 2 — DOM**: visual depth-of-market ladder with bid/ask bars, sweep zone warnings
    - **Column 3 — TP Zones**: 4-tier TP targets with price, points, dollar value, R:R, % to exit
    - **Column 4 — Actions**: breakeven panel (hard/soft/buffer BE), risk action recommendations, live analysis signal pills
  - Session stats header bar: session P&L, win rate, max drawdown, trades today, risk used
  - Rithmic connection banner (demo mode when not connected, with symbol selector)
  - No-position state: "Waiting for positions..." with instrument quick-select grid
  - Wire to `GET /api/live/positions` SSE stream for real-time updates

---

### Phase NEWS — News Sentiment Pipeline (from `data_news.md`)

> **Priority**: 🟡 Medium — enhances the Research page morning workflow.
> **Depends on**: API keys (Finnhub free, Alpha Vantage free, XAI_API_KEY already configured).
> **Source file**: `todo/data_news.md`

The prototype describes a 3-tier news sentiment system:
1. **Finnhub** — high-volume free market news (60 calls/min)
2. **Alpha Vantage** — AI-scored news sentiment + commodity price proxies
3. **Grok hybrid scoring** — VADER (fast, free) + Grok 4.1 (context-aware) = weighted hybrid score

#### Phase NEWS-A: News Data Collector

- [ ] **`src/lib/integrations/news_client.py`** — new module
  - `FinnhubClient` class:
    - `fetch_general_news(category='general')` → market-moving headlines
    - `fetch_company_news(ticker, days_back=7)` → per-ticker news (USO, GLD, SPY as futures proxies)
    - Rate limit: 60 calls/min (generous)
    - Auth: `FINNHUB_API_KEY` env var
  - `AlphaVantageClient` class:
    - `fetch_news_sentiment(tickers='CL,GC,ES', topics='energy,commodities')` → articles with AI sentiment scores
    - `fetch_commodity_price(commodity='WTI', interval='daily')` → oil/gold/natgas price data
    - Rate limit: 25 calls/day (tight — use sparingly, cache aggressively)
    - Auth: `ALPHA_VANTAGE_API_KEY` env var
  - Add `finnhub-python` to `pyproject.toml` dependencies
  - Proxy ticker mapping: `MES→SPY`, `MCL→USO`, `MGC→GLD`, `MNQ→QQQ` for Finnhub company news

#### Phase NEWS-B: Hybrid Sentiment Scorer

- [ ] **`src/lib/analysis/news_sentiment.py`** — new module
  - **VADER with futures lexicon** (from prototype):
    - Custom lexicon additions: `surge: 3.0`, `rally: 2.8`, `plunge: -3.0`, `crash: -3.5`, `opec: 0.0`, `rate hike: -2.0`, `inventory draw: 2.0`, etc. (full list in `data_news.md`)
    - `vader_score(text)` → float [-1, +1]
    - Add `vaderSentiment` to `pyproject.toml` dependencies
  - **Grok sentiment scoring** (uses existing `grok_helper.py` infrastructure):
    - `grok_futures_sentiment(headline, summary, ticker)` → (score, label, reason)
    - Prompt from prototype: "You are a professional futures trader specializing in {ticker}... Respond ONLY with valid JSON"
    - Model: `grok-4-1-fast-reasoning` (cheap: ~$0.01 per 100 articles)
    - Only call Grok on ambiguous articles where `abs(vader_score) < 0.3` (cost optimization)
  - **Hybrid score** (weighted):
    - `hybrid_score = 0.4 * vader + 0.4 * alpha_vantage_score + 0.2 * grok_score`
    - `compute_news_sentiment(symbol)` → `NewsSentiment` dataclass
    - `compute_all_news_sentiments()` → dict of symbol → `NewsSentiment`

#### Phase NEWS-C: Scheduler Integration + Caching

- [ ] **Engine scheduler integration**:
  - `CHECK_NEWS_SENTIMENT` action — run at 07:00 ET (pre-market) and 12:00 ET (midday refresh)
  - Cache in Redis: `engine:news_sentiment:<SYMBOL>` (2-hour TTL)
  - Store daily aggregates in Postgres `news_sentiment_history` table
  - Finnhub news fetched daily (high volume OK); Alpha Vantage fetched once/day (25 call limit); Grok called on top 50 articles only
- [ ] **Spike detection**: if `mention_count_1h` > 3× rolling average → publish SSE event
  - Dashboard shows: "📰 News Spike: MCL — 12 articles in last hour, sentiment -0.6 (bearish)"

#### Phase NEWS-D: Dashboard Integration

- [ ] **News panel on Research/Morning page**:
  - Top headlines with sentiment badges (🟢 bullish / 🔴 bearish / ⚪ neutral)
  - Hybrid sentiment score bar per focus asset
  - Grok AI reason tooltip on hover
  - "News Pulse" strip alongside Reddit sentiment and risk strip
- [ ] **API route**: `GET /api/news/sentiment?symbols=MES,MGC,MCL` → aggregated sentiment per symbol
- [ ] **API route**: `GET /api/news/headlines?symbol=MES&limit=10` → recent headlines with scores

---

### Phase UI-ENHANCE — Trading Dashboard Improvements (from `trading_webui_review.md`)

> **Priority**: 🟡 Medium — polish items from the UI blueprint not yet implemented.
> **Source file**: `todo/trading_webui_review.md`

The blueprint describes a 6-page UI (Research → Analysis → Plan → Live → Journal → Settings).
The current `static/trading.html` implements a 5-step flow (Morning Run → Confirm Plan → Live → Journal → Settings).
Several features from the blueprint are not yet wired:

#### Phase UI-A: Research Page Enhancements (Page 1 in blueprint)

- [ ] **Cross-asset context panel**:
  - ES/NQ/RTY correlation mini-heatmap
  - DXY, VIX, yields as leading indicator badges
  - Wire to existing `cross_asset.py` module
  - API: `GET /api/analysis/cross_asset` (may already exist)
- [ ] **Economic calendar integration**:
  - Free source: Forex Factory RSS or TradingEconomics free API
  - Show today's high-impact events with time, expected, previous values
  - Warn on Plan page if trading during CPI/FOMC/NFP release
- [ ] **Sentiment gauges**:
  - Reddit sentiment bar (when Phase REDDIT is built)
  - News sentiment bar (when Phase NEWS is built)
  - Combined "Market Mood" gauge: bullish ◄──────► bearish

#### Phase UI-B: Analysis Page Enhancements (Page 2 in blueprint)

- [ ] **Asset fingerprint display**:
  - Wire `asset_fingerprint.py` output to Analysis step in pipeline
  - Show: "This instrument tends to: run stops at open, respect VWAP, mean-revert from extremes..."
  - Optional: "Asset DNA" radar chart (6 fingerprint features as radar overlay)
- [ ] **Wave structure panel**:
  - Wire `wave_analysis.py` + `swing_detector.py` output to pipeline
  - Show labeled swing highs/lows + current wave count
- [ ] **Asset selection output**:
  - After Analysis step, user picks 1-2 focus assets
  - Selection persists as session state, filters all downstream pages

#### Phase UI-C: Plan Page Enhancements (Page 3 in blueprint)

- [ ] **Range builders status panel**:
  - Wire `rb/detector.py` output to plan data
  - Show current range boundaries, breakout direction to watch, targets
- [ ] **Backtest validation button**:
  - "Backtest this level" button on each entry zone
  - Wire to `backtesting.py` to show historical hit rate for similar setups
  - Display: "This type of setup has hit T1 72% of the time over last 90 days"
- [ ] **CNN confidence badge on levels**:
  - Show CNN breakout probability next to each level/zone in the plan
  - Wire to `breakout_cnn.py` inference output from engine

#### Phase UI-D: Journal Page Enhancements (Page 5 in blueprint)

- [ ] **Auto-populate from Rithmic fills** (when creds arrive):
  - Pre-fill trade log entries from `position_manager.py` fill data
  - User only needs to add: was it planned? entry/exit reason, grade, notes
- [ ] **Plan adherence scoring**:
  - Compare each trade's entry price to the locked plan zones
  - Auto-calculate: "Trade 1: MES Long 5,784 ✅ Matched Zone A"
  - Session adherence score: percentage of trades that matched the plan
- [ ] **Session stats panel**:
  - Net P&L, win rate, avg R:R, best trade, worst trade
  - Equity curve chart (small area chart of session P&L over time)

#### Phase UI-E: UX Polish (from Design Recommendations section)

- [ ] **Keyboard shortcuts**: `1-5` to jump between pages, `Space` to lock plan
- [ ] **One-click copy**: every level/price copies to clipboard on click (for pasting into MotiveWave)
- [ ] **Progress indicator in nav**: `Research ✅ → Analysis ✅ → Plan ✅ → Live ● → Journal`
  - Show current step highlighted, dim completed steps, indicate locked plan
- [ ] **Mobile-friendly live page**: responsive layout for checking positions on phone during a trade
- [ ] **Dark terminal theme refinements**:
  - Deep charcoal (`#0D0F14`) background (current `#07090F` is close)
  - Level line color coding: ICT = purple, volume profile = blue, range = amber (already partially done)
  - `DM Sans` for labels alongside `JetBrains Mono` for prices (add font import)

---

### Miscellaneous Items Extracted

#### From `trading_webui_review.md` — Potential Issues Found

These were flagged in the blueprint review. Some are already resolved, some remain:

- [x] **Duplicate filenames** — `risk.py`, `live_risk.py`, etc. appear in multiple directories
  - Status: Known; absolute imports are used throughout. Not a blocking issue.
- [x] **`main.py` in 4 places** — data, web, engine, trainer_server
  - Status: Known; each is a separate service entry point. Docker/compose calls the right one.
- [x] **No `tests/` directory visible** — flagged in the review
  - Status: ✅ Resolved — `src/tests/` has 33 test files, 2500+ passing tests
- [ ] **`core/models.py` vs `services/data/api/` schema duplication** — verify Pydantic schemas have single source of truth
  - Low priority, non-blocking

#### From `trading_webui_review.md` — Missing Integrations

- [ ] **ORB data on Plan page** — `orb_filters.py` exists but ORB levels not shown on plan page
  - Wire ORB high/low as a level category in the plan
  - Pipeline step "orb" already exists but output not surfaced in plan zones
- [ ] **Economic calendar** — no integration yet (see Phase UI-A above)
- [ ] **"Backtest this level" button** — (see Phase UI-C above)

#### From `app1.py` — Additional Routes Not Yet in Production

These routes exist in the `app1.py` prototype but are not in the current `pipeline.py`:

- [ ] `GET /api/live/positions/stream` — SSE stream for position intelligence (see Phase POSINT-C)
- [ ] `GET /api/live/book?symbol=MES` — L1 + L2 depth of market snapshot
- [ ] `GET /api/live/positions/snapshot` — non-SSE current positions

#### From `trading-dashboard.jsx` — Mock Data Patterns Worth Preserving

The React prototype has well-structured mock data that can be reused for demo mode:

- `CROSS_ASSET` — 8 cross-asset tickers with price, change%, direction
- `EVENTS` — economic calendar entries with time, label, impact level, previous, expected
- `MTF_DATA` — 5-timeframe structure read with bias arrows
- `LEVELS` — 10 key levels with type tags (range/volume/ict/liq/orb) and color coding
- `ZONES` — 2 scored entry zones with full stop/target/R:R data
- `MOCK_TRADES` — 4 journal entries with planned flag, grade, reason

These can be extracted into `static/mock_data.js` or kept inline in `trading.html` for demo mode.

---

## Integration Priority Matrix

| Phase | Value | Effort | Depends On | Build When |
|-------|-------|--------|------------|------------|
| **POSINT** (Position Intelligence) | 🔴 Very High | Large (3-5 days) | Rithmic creds (mock first) | After v8 training starts |
| **NEWS** (News Sentiment) | 🟡 Medium | Medium (2-3 days) | Finnhub + Alpha Vantage API keys | Parallel with v8 training |
| **UI-A** (Research enhancements) | 🟡 Medium | Small (1 day) | `cross_asset.py` already exists | After NEWS |
| **UI-B** (Analysis enhancements) | 🟡 Medium | Small (1 day) | `asset_fingerprint.py` exists | After UI-A |
| **UI-C** (Plan enhancements) | 🟡 Medium | Small (1 day) | Engine running | After UI-B |
| **UI-D** (Journal enhancements) | 🟢 Low | Small (1 day) | Rithmic fills | After first live trades |
| **UI-E** (UX polish) | 🟢 Low | Small (1 day) | None | Anytime |

---

## Recommended Sequencing

```
Week 1 (now):     v8 training running on GPU rig
                  ├── Phase NEWS-A + NEWS-B (build news client + sentiment scorer)
                  └── Phase POSINT-A + POSINT-B (build position intelligence module)

Week 2:           v8 training completes, model evaluation
                  ├── Phase POSINT-C + POSINT-D (API routes + Live page UI)
                  ├── Phase NEWS-C + NEWS-D (scheduler + dashboard)
                  └── Phase UI-A (cross-asset + calendar on Research page)

Week 3:           v8 deployed, first live signals
                  ├── Phase UI-B + UI-C (analysis + plan page polish)
                  └── Phase UI-E (keyboard shortcuts, mobile, copy-to-clipboard)

Post first trades: Phase UI-D (journal auto-populate from Rithmic fills)
```

---

## What to Update in Project Docs

### `todo.md` — Add These Sections

1. Add **Phase POSINT** as a new 🟡 section under "Next Up" (after "Wire Real Modules into Trading Pipeline")
2. Add **Phase NEWS** as a new 🟡 section (can reference `docs/backlog.md` for Reddit which is related)
3. Add **Phase UI-ENHANCE** tasks under the existing "Post-Training Cleanup" section

### `docs/backlog.md` — Add These Phases

1. **Phase POSINT** (Position Intelligence) — full spec above
2. **Phase NEWS** (News Sentiment Pipeline) — full spec above
3. Merge **Phase UI-ENHANCE** sub-tasks into relevant existing phases (CHARTS, REDDIT, etc.)

### Files to Delete After Extraction

```
todo/README.md              — fully superseded by pipeline.py integration
todo/notes.md               — original vision, fully implemented
todo/trading_webui_review.md — extracted all remaining tasks above
todo/app.py                 — integrated into pipeline.py
todo/app1.py                — position routes extracted to Phase POSINT
todo/index.html             — copied to static/trading.html
todo/live_page.html         — extracted to Phase POSINT-D
todo/live_page1.html        — duplicate of live_page.html
todo/trading-dashboard.jsx  — React prototype, not used (mock data patterns noted)
todo/position_engine.py     — extracted to Phase POSINT-A/B
todo/position_engine1.py    — duplicate of position_engine.py
todo/data_news.md           — extracted to Phase NEWS
todo/requirements.txt       — deps already in project
```

**All 13 files can be deleted.** Every actionable item has been captured in this document.