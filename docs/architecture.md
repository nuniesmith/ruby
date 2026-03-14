# Ruby Futures — System Architecture

> Extracted from `todo.md`. This is the canonical reference for how all services fit
> together end-to-end. Update this file whenever the topology changes.

---

## Repo Layout

```
futures/
├── src/
│   ├── lib/
│   │   ├── core/          # BreakoutType, RangeConfig, session, models, alerts, cache, asset_registry
│   │   ├── analysis/      # breakout_cnn, breakout_filters, chart_renderer, mtf_analyzer, regime, scorer, …
│   │   ├── strategies/
│   │   │   ├── rb/        # range breakout scalping (detector, range_builders, publisher)
│   │   │   ├── daily/     # bias_analyzer, daily_plan, swing_detector
│   │   │   └── costs.py   # slippage, commission modelling
│   │   ├── services/
│   │   │   ├── engine/    # main, handlers, scheduler, position_manager, backfill, risk, focus, live_risk, …
│   │   │   ├── web/       # HTMX dashboard, FastAPI reverse-proxy (port 8080)
│   │   │   ├── data/      # FastAPI REST + SSE API (port 8000) — bars, journal, positions, kraken, sse, …
│   │   │   └── training/  # GPU CNN training server (port 8501)
│   │   └── integrations/  # kraken_client, massive_client, grok_helper, rithmic_client
│   ├── entrypoints/
│   │   ├── data/main.py   # python -m entrypoints.data.main  → lib.services.data.main:app
│   │   ├── engine/main.py # python -m entrypoints.engine.main
│   │   ├── web/main.py    # python -m entrypoints.web.main
│   │   └── training/main.py
│   └── pine/
│       └── ruby_futures.pine   # TradingView Pine Script indicator
├── models/                # champion .pt, feature_contract.json (Git LFS)
├── scripts/
│   ├── sync_models.sh     # pull .pt from repo → restart engine
│   └── …
├── config/                # Prometheus, Grafana, Alertmanager
├── docker/
│   ├── data/              # Dockerfile + entrypoint.sh  (:data image)
│   ├── engine/            # Dockerfile + entrypoint.sh  (:engine image)
│   ├── web/               # Dockerfile + entrypoint.sh  (:web image)
│   ├── trainer/           # Dockerfile                  (:trainer image)
│   ├── charting/          # Dockerfile                  (:charting image)
│   └── monitoring/        # prometheus/ + grafana/
└── docker-compose.yml
```

---

## Infrastructure Topology

```
Ubuntu Server (100.122.184.58)                                        Home Laptop (100.113.72.63)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐
│    :data     │  │   :engine    │  │    :web      │  │  :charting │  │   :trainer   │
│  FastAPI     │  │  main.py     │  │  FastAPI     │  │  ApexCharts│  │  FastAPI     │
│  REST + SSE  │  │  scheduler   │  │  reverse-    │  │  + nginx   │  │  dataset gen │
│  bar cache   │  │  risk mgr    │  │  proxy only  │  │  port 8003 │  │  CNN train   │
│  Kraken feed │  │  position mgr│  │  port 8080   │  │            │  │  promote .pt │
│  Reddit poll │  │  all handlers│  │              │  │            │  │  CUDA GPU    │
│  Rithmic mgr │  │  (no HTTP)   │  │              │  │            │  │  port 8501   │
│  port 8000   │  │              │  │              │  │            │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
       │    publishes     │   reads          │ proxies        │                │
       │    Redis state ←─┘   Redis          │ → :data        │                │
       ↓                                     ↓                ↓                │
┌──────────┐  ┌──────────┐           Browser (8080)                           │
│  Redis   │  │ Postgres │                                                    │
└──────────┘  └──────────┘                                                    │
┌─────────────┐  ┌──────────┐                                                 │
│ Prometheus  │  │ Grafana  │                                                 │
└──────┬──────┘  └──────────┘                                                 │
       └──────────────────────── Tailscale mesh ──────────────────────────────┘

Rithmic (async_rithmic)
  └─ CME Gateway connection from :data service
     ├─ Live ticks + time bars (when creds active)
     ├─ Order book data
     └─ Order execution (CopyTrader across prop-firm accounts)
```

**Service responsibilities:**

| Service    | Responsibility |
|------------|---------------|
| `:data`    | All REST/SSE endpoints, bar cache (Postgres + Redis), Kraken WS feed, Reddit sentiment polling, `/bars/{symbol}` auto-fill, `/api/charts/*`, Rithmic account manager (EOD close endpoint, CopyTrader) |
| `:engine`  | `DashboardEngine`, `ScheduleManager`, `RiskManager`, `PositionManager`, breakout detection, CNN inference, Grok briefs, Redis publish, EOD safety scheduler (15:45 warning + 16:00 hard-close). Writes `/tmp/engine_health.json` as heartbeat — no HTTP port |
| `:web`     | Stateless reverse-proxy; proxies all `/api/*` and `/sse/*` to `:data` |
| `:charting`| ApexCharts-based charting UI served via nginx |
| `:trainer` | Dataset generation, CNN training, gate check, model promotion |

**Port map:**

| Port | Service | Internal |
|------|---------|----------|
| 8000 | `:data` | 8000 |
| 8080 | `:web` | 8080 |
| 8003 | `:charting` | 8003 |
| 8501 | `:trainer` | 8501 |
| 9095 | Prometheus | — |
| 3010 | Grafana | — |

**Tailscale mesh:**
- Ubuntu Server → data + engine + web + charting + postgres + redis + monitoring (always on, 24/7)
- Home Laptop → trainer (on-demand, CUDA GPU, port 8501)

**CI/CD (7-image matrix — `nuniesmith/futures`):**

| Image | Platforms | Notes |
|-------|-----------|-------|
| `:data` | amd64 + arm64 | `is_default: true` → `:latest` alias |
| `:engine` | amd64 + arm64 | |
| `:web` | amd64 + arm64 | |
| `:charting` | amd64 + arm64 | ApexCharts + nginx |
| `:trainer` | amd64 only | GPU build |
| `:prometheus` | amd64 + arm64 | |
| `:grafana` | amd64 + arm64 | |

Pipeline: Lint → Test → Build & push → Deploy (Ubuntu Server via Tailscale SSH) → Deploy trainer (home laptop via Tailscale SSH) → Health checks → Discord notifications.

---

## End-to-End Data & Signal Flow

### 1. Data Ingestion

```
External Sources
  ├─ Rithmic (async_rithmic)     ← primary for CME futures (when creds active): live ticks, time bars, order book
  ├─ MassiveAPI (massive_client) ← current primary for CME futures (REST + WebSocket, futures beta)
  ├─ Kraken REST / WebSocket     ← crypto spot only via kraken_client.py (personal account)
  └─ Yahoo Finance (yfinance)    ← last-resort fallback (delayed data)

         ↓
  lib/core/cache.py → get_data(ticker, interval, period)
         │  Fetches bars, caches in Redis
         │  Keys: engine:bars_1m:<TICKER>, engine:bars_15m:<TICKER>, engine:bars_daily:<TICKER>
         ↓
  lib/trading/engine.py → DashboardEngine
         │  _fetch_tf_safe() / _refresh_data() / _loop()
         ↓
  Redis (pub/sub + key-value) — central message bus
```

### 2. Engine Startup & Scheduler

```
src/lib/services/engine/main.py → main()
  ├─ Env: ACCOUNT_SIZE, ENGINE_INTERVAL, ENGINE_PERIOD
  ├─ Creates DashboardEngine, ScheduleManager, RiskManager, PositionManager, ModelWatcher
  └─ Main loop:
       ├─ scheduler.get_pending_actions()   ← time-of-day aware
       ├─ _check_redis_commands()           ← dashboard-triggered overrides
       ├─ Execute pending actions via dispatch table
       ├─ _handle_update_positions()        ← bracket/trailing stop updates
       ├─ _tick_live_risk_publisher()       ← publish LiveRiskState every loop
       └─ _publish_engine_status()          ← push state to Redis for web UI

Session Modes (Eastern Time):
  EVENING     18:00–00:00  →  CME, Sydney, Tokyo, Shanghai ORB sessions
  PRE_MARKET  00:00–03:00  →  Daily focus, Grok brief, generate_daily_plan()
  ACTIVE      03:00–12:00  →  Frankfurt, London, London-NY, US ORB + all 13 breakout types
  OFF_HOURS   12:00–18:00  →  Backfill, training, optimization, daily report
```

### 3. EOD Safety System

```
DashboardEngine._loop() — runs every 10s, checks ET wall-clock time

  15:45–15:59 ET  (once per calendar day)
    → _eod_warning()
         ├─ logs WARNING: "automated close fires in 15 minutes"
         └─ AlertDispatcher.send_risk_alert() → Discord 

  16:00–16:14 ET  (once per calendar day, catch-up guard for restarts)
    → _eod_close_positions()
         ├─ RithmicAccountManager.eod_close_all_positions()
         │    For each enabled account:
         │      1. cancel_all_orders(account_id)   ← kills all working entries/stops/targets
         │      2. asyncio.sleep(0.5)              ← exchange ack pause
         │      3. exit_position(account_id, MANUAL) ← market-flatten net position
         └─ AlertDispatcher.send_risk_alert() → per-account summary (✅ / ❌ / ⏭)

  Manual trigger:
    POST /api/rithmic/eod-close          ← dashboard button or curl
    Body: { "dry_run": true }            ← connect + discover, skip cancel/exit
```

> **Note:** `OrderPlacement.MANUAL` is an audit tag only — it tells Rithmic's backend
> "human-initiated" vs `AUTO`. Both execute real orders. The 15:45 warning exists so
> you are flat before the 16:00 auto-close ever fires. The auto-close is a last-resort
> safety net, not the normal workflow.

### 4. Daily Focus Computation

```
PRE_MARKET (00:00–03:00 ET) → generate_daily_plan()
  │
  ├─ compute_all_daily_biases()       ← 6-component scoring per asset
  ├─ Grok macro brief (optional)      ← if XAI_API_KEY set
  ├─ select_daily_focus_assets()      ← 5-factor composite ranking (0-100)
  │    signal quality 30%, ATR opportunity 25%, RB density 20%, session fit 15%, catalyst 10%
  ├─ _build_swing_candidate()         ← wider SL/TP (1.75×/2.5×/4×/5.5× ATR)
  └─ DailyPlan.publish_to_redis()     ← engine:daily_plan, engine:focus_assets (18h TTL)
```

### 5. Breakout Detection (13 Types, 10 Sessions)

```
CHECK_ORB_* / CHECK_PDR / CHECK_IB / CHECK_CONSOLIDATION / CHECK_BREAKOUT_MULTI
  │
  ├─ Fetch 1m bars from Redis cache
  ├─ detect_range_breakout(bars, symbol, config)
  │    └─ _build_*_range() → _scan_for_breakout() → BreakoutResult
  │
  ├─ apply_all_filters() ← NR7, premarket, session window, lunch, MTF bias, VWAP
  │
  │  IF passed:
  │    ├─ predict_breakout(image, tabular, session_key)  ← CNN inference
  │    │    threshold per session (us:0.82 → sydney:0.72)
  │    │
  │    │  IF cnn_signal:
  │    │    ├─ RiskManager.can_enter_trade()
  │    │    ├─ PositionManager.process_signal()  ← bracket, P&L tracking (informational)
  │    │    ├─ signals_publisher.write_signal()  ← append to signals.csv → GitHub push
  │    │    ├─ publish_breakout_result()          ← Redis pub/sub → dashboard SSE
  │    │    └─ alerts.send_signal()               ← push notification
```

### 6. CNN Inference (Python)

```
predict_breakout(image_path, tabular_features, session_key)
  │
  ├─ Image branch: chart_renderer_parity.py → 224×224 Ruby-style chart snapshot
  │    → ImageNet normalisation → (1, 3, 224, 224) tensor
  │
  ├─ Tabular branch — v8 contract (37 features + embedding IDs):
  │    _normalise_tabular_for_inference(features) → (1, 37) float tensor
  │    [0-17]   v6 features (quality, volume, ATR, CVD, direction, session, etc.)
  │    [18-23]  v7 daily features (bias direction/confidence, prior day pattern,
  │              weekly range position, monthly trend, crypto momentum)
  │    [24-27]  v7.1 sub-features (breakout type category, session overlap,
  │              ATR trend, volume trend)
  │    [28-30]  v8-B cross-asset correlation (peer_corr, class_corr, corr_regime)
  │    [31-36]  v8-C asset fingerprint (daily range norm, session concentration,
  │              breakout follow-through, hurst exponent, overnight gap, vol profile shape)
  │    + asset_class_idx (int) → Embedding(5, 4)
  │    + asset_idx       (int) → Embedding(25, 8)
  │
  ├─ Forward pass:
  │    EfficientNetV2-S(image)          → (1, 1280)
  │    tabular_head(tabular)            → (1, 64)    [wider head: 37→256→128→64]
  │    asset_class_emb(class_idx)       → (1, 4)
  │    asset_emb(asset_idx)             → (1, 8)
  │    classifier(cat([img, tab, embs])) → (1, 2) → softmax → P(clean breakout)
  │
  └─ Returns: { prob, signal, confidence, threshold }
       signal = True if prob ≥ session threshold

Backward-compat padding (live inference with older checkpoints):
  v5(8) → v4(14) → v6(18) → v7(24) → v7.1(28) → v8(37)
  Handled in _normalise_tabular_for_inference() — neutral defaults for missing slots
```

### 7. Live Risk State

```
LiveRiskPublisher (ticked every engine loop, force-publish on position change)
  │
  ├─ compute_live_risk(risk_manager, position_manager)
  │    → LiveRiskState: daily_pnl, open_positions, remaining_risk_budget,
  │                     total_unrealized_pnl, margin_used, can_trade, block_reason
  │
  ├─ Publish to Redis: engine:live_risk
  ├─ SSE channel: dashboard:live_risk → risk strip updates every 5s
  └─ Focus cards: dual micro/regular sizing reflects remaining_risk_budget
```

### 8. Dashboard → Manual Trading Signal Flow

```
Engine fires CNN-gated signal
  │
  ├─ signals_publisher.append_and_push(signal)
  │    → signals.csv committed to nuniesmith/futures-signals (GitHub API)
  │    → SSE push to dashboard
  │
  ├─ Dashboard (primary decision surface)
  │    → Focus cards with CNN probability, entry/stop/TP, dual sizing, risk strip
  │    → Reddit sentiment badge + spike alerts
  │    → Trader decides whether to execute
  │
  ├─ Rithmic CopyTrader (automated execution)
  │    → RithmicAccountManager places order on leader account
  │    → CopyTrader replicates to all enabled follower accounts
  │
  ├─ TradingView (reference overlay — no position sendback)
  │    → Ruby Futures indicator shows levels on chart for visual confirmation
  │    → NOT used for order execution or position management
  │
  └─ EOD Safety Net
       → 15:45 ET warning → 16:00 ET auto-close via Rithmic
```

### 9. Training Pipeline

```
trainer_server.py → _run_training_pipeline(TrainRequest)
  │
  ├─ dataset_generator.py → generate_dataset(symbols, days_back, config)
  │    For each of 25 symbols × 13 types × 9 sessions:
  │      ├─ load_bars() ← DataResolver (Redis → Postgres → Massive/Kraken)
  │      ├─ _resolve_peer_tickers() → bars_by_ticker dict (for v8-B cross-asset features)
  │      ├─ rb_simulator.py → bracket replay → good/bad labels
  │      ├─ chart_renderer_parity.py → 224×224 PNG per sample
  │      └─ _build_row() → 37 tabular features + embedding IDs
  │
  ├─ split_dataset(85/15 stratified by label × breakout_type × session)
  ├─ train_model(epochs=80, batch_size=64, grad_accum=2)
  │    Phase 1 (5 epochs): freeze EfficientNetV2-S backbone, train tabular head + embeddings
  │    Phase 2 (75 epochs): unfreeze all, cosine decay, separate LR groups
  │      backbone lr=2e-4,  head+embeddings lr=1e-3
  │    Regularisation: mixup α=0.2, label smoothing 0.10, weight_decay=1e-4
  ├─ evaluate_model() → acc / prec / rec
  ├─ Gate check → ≥89% acc, ≥87% prec, ≥84% rec → promote to breakout_cnn_best.pt
  └─ ModelWatcher detects new .pt → engine hot-reloads
```

### 10. Reddit Sentiment Pipeline (Phase REDDIT — not yet built)

```
Engine Scheduler (every 15 min during ACTIVE + EVENING)
  │
  ├─ reddit_client.py → PRAW OAuth → fetch hot/new posts from 4 subreddits
  │    r/FuturesTrading, r/Daytrading, r/wallstreetbets, r/InnerCircleTraders
  │
  ├─ reddit_sentiment.py → per-asset scoring
  │    mention_count, velocity, avg_sentiment, engagement, wsb_euphoria
  │
  ├─ Cache: Redis engine:reddit_sentiment:<SYMBOL> (30-min TTL)
  ├─ History: Postgres reddit_sentiment_history (daily aggregates)
  │
  ├─ Spike detection: mention_velocity > 3× rolling avg
  │    → SSE engine:reddit_spike → dashboard alert
  │
  └─ Dashboard: sentiment badges on focus cards + Reddit Pulse strip
```

---

## Key Redis Key Schema

| Key | Producer | Consumer | TTL |
|-----|----------|----------|-----|
| `engine:bars_1m:<TICKER>` | data / cache | engine, dashboard | 5 min |
| `engine:bars_daily:<TICKER>` | data / cache | engine, training | 1 h |
| `engine:daily_plan` | engine | dashboard, focus | 18 h |
| `engine:focus_assets` | engine | dashboard | 18 h |
| `engine:live_risk` | engine | dashboard SSE | live |
| `engine:swing_signals` | engine | dashboard SSE | session |
| `engine:swing_states` | engine | swing actions API | session |
| `engine:orb_results` | engine | dashboard, charts | session |
| `engine:reddit_sentiment:<SYMBOL>` | data (future) | dashboard | 30 min |
| `rithmic:account_configs` | rithmic_client | rithmic_client | permanent |
| `rithmic:account_status:<KEY>` | rithmic_client | dashboard | 5 min |
| `settings:overrides` | settings API | engine, data | permanent |

---

## Scaling Plan

```
Stage 1 — TPT:   5 × $150K accounts  =  $750K total buying power
Stage 2 — Apex: 20 × $300K accounts  =  ~$6M total buying power

Copy layer:
  Rithmic CopyTrader (leader account)
    → replicates orders to all enabled follower accounts
    → same Rithmic gateway, per-account credentials

Own-accounts-only copy trading explicitly permitted by both TPT and Apex.
```
