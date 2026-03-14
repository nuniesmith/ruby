# futures — TODO

> **Last updated**: 2026-03-14 — Oryx session: dataset paths fixed (Docker `/app/` prefix stripped from all CSVs), all 28,548 images verified on disk, 2-epoch CUDA training test passed on RTX 3080. New files created: position_intelligence.py, rithmic_position_engine.py, dom.py API, static_pages.py, chat.html, dom.html, journal.html, 6 new scripts, 93 new tests (all passing). DOM routes + static page routes registered in data service.

> **Repo**: `github.com/nuniesmith/futures`
> **Docker Hub**: `nuniesmith/futures` — `:data` · `:engine` · `:web` · `:trainer` · `:charting`
> **Infrastructure**: Ubuntu Server `100.122.184.58` (data + engine + web + charting + monitoring), Oryx GPU rig `oryx` (RTX 3080 16GB, CUDA 13.1, torch 2.10+cu128 — trainer deploys here)
>
> 📐 **Architecture reference**: [`docs/architecture.md`](docs/architecture.md)
> 📦 **Completed work**: [`docs/completed.md`](docs/completed.md)
> 🗂️ **Deferred backlog**: [`docs/backlog.md`](docs/backlog.md)

---

## 🎯 Goal

**Manual trading co-pilot with prop-firm compliant execution via Rithmic.** The system informs entries via CNN + Ruby signals — the trader pushes "SEND ALL" in the WebUI or uses the DOM. All execution flows through Rithmic with `MANUAL` flag + humanized delays. No autonomous/bot orders — ever.

```
Python Engine  →  CNN signal + Ruby signal + daily bias + risk strip + Grok brief
Python Dashboard  →  Focus cards, swing signals, Reddit sentiment, one-click execution
Rithmic (async_rithmic)  →  Main account order + 1:1 copy to all slave accounts
```

**Data hierarchy (corrected):**
- **Rithmic** (async_rithmic) — primary for CME futures when creds active: live tick data, time bars, order book, market depth, historical data, PnL
- **MassiveAPI** (massive_client) — current primary for CME futures: REST + WebSocket live data (futures beta)
- **Yahoo Finance** (yfinance) — last-resort fallback only (delayed data)
- **Kraken** (kraken_client) — crypto spot + futures (personal accounts, fully app-managed) + live tick data for simulation

**Two-stage scaling plan:**
- Stage 1 — TPT: 5 × $150K accounts = $750K buying power
- Stage 2 — Apex: 20 × $300K accounts = ~$6M buying power
- Copy layer: Rithmic `CopyTrader` (main → slaves) with `OrderPlacementMode.MANUAL` + 200–800 ms delay

**Prop-firm compliance:** Every order tagged `MANUAL` + humanized delay. Main account = human button push only. No autonomous entries. Server-side hard stops via `stop_ticks`. See Phase RITHMIC below + [`docs/rithmic_notes.md`](docs/rithmic_notes.md).

**EOD Safety (live now):** Rithmic EOD cancel-all + exit-position fires at 16:00 ET daily via the engine scheduler. 15:45 warning alert fires first. Manual trigger: `POST /api/rithmic/eod-close`. See [`docs/architecture.md`](docs/architecture.md) for full sequence.

**Training assets (focused):** `MGC SIL MES MNQ M2K MYM ZN ZB ZW` — generate signals only for these. Other forex/low-liquidity assets tracked for broad view only (no signal generation until further notice).

---

## Current State

| Item | Status |
|------|--------|
| Champion model | v6 — 87.1% acc / 87.15% prec / 87.27% rec — 18 features, 25 epochs (still champion) |
| v8 model (latest) | ⚠️ 83.3% acc / 83.4% prec / 83.3% rec — 37 features, epoch 56/60 — **overfitting** (99.7% train vs 83.3% val = 16.4% gap) — see RETRAIN phase |
| Feature contract | v8 code complete — 37 tabular features + embeddings |
| v8 smoke test | ✅ 31/31 tests passing (`test_v8_smoke.py`) |
| Full test suite | ✅ 2834+ passed, 1 skipped, 1 pre-existing risk test failure (account_size change) |
| Dataset paths | ✅ **FIXED 2026-03-14** — Docker `/app/` prefix stripped, all 28,548 images verified on disk |
| CUDA training | ✅ **VERIFIED 2026-03-14** — 2-epoch test passed on oryx RTX 3080 16GB (torch 2.10+cu128) |
| Dataset validation | ✅ All 28,548 rows valid, 0 missing images, labels balanced (52.9% good / 47.1% bad), all 9 sessions represented |
| Per-asset training | ✅ Infrastructure built — `train_mode=per_asset\|per_group\|combined` in TrainRequest, per-asset model loading with fallback chain |
| CNN regularization | ✅ Upgraded — dropout 0.4→0.5, label smoothing 0.10→0.15, weight decay 1e-4→2e-4, stronger augmentation, patience 15→12 |
| Rithmic EOD close | ✅ wired into `DashboardEngine._loop()` — uses `OrderPlacement.MANUAL` |
| Rithmic copy trading | ✅ `CopyTrader` class built — 114 tests passing — see Phase RITHMIC |
| Rithmic account manager | ✅ `RithmicAccountManager` — multi-account config, encrypted creds, prop firm presets |
| Prop-firm compliance | ✅ `MANUAL` flag + 200–800 ms delay enforced on all orders — see RITHMIC-B |
| PositionManager → Rithmic | ✅ `execute_order_commands()` fully wired — MODIFY_STOP/CANCEL/BUY/SELL all routed — see RITHMIC-C |
| Server-side brackets | ✅ `stop_price_to_stop_ticks()` + `TICK_SIZE` table for all 14 micro products — see RITHMIC-C |
| Copy trading engine gate | ✅ `RITHMIC_COPY_TRADING=1` env var gates Rithmic path |
| Ruby signal engine | ✅ Full Pine → Python port — see RITHMIC-G |
| CI/CD secrets | ✅ verification script created (`scripts/verify_cicd.sh`) — run on each machine to confirm |
| TRAINER_SERVICE_URL | ✅ moved from hardcode to env var in `docker-compose.yml` |
| ENGINE_DATA_URL port | ✅ fixed — was `:8100` (wrong), now `:8050` (matches data service `8050:8000` mapping) |
| sync_models.sh | ✅ audited — platform-agnostic, works on Ubuntu Server |
| Trading dashboard | ✅ integrated — pipeline API + trading.html wired into data + web services |
| Dataset smoke test | ✅ `scripts/smoke_test_dataset.py` — validates engine connectivity, bar loading, rendering |
| Charts service | ✅ VWAP ±σ bands, CVD sub-pane, Volume Profile (POC/VAH/VAL), Anchored VWAP, localStorage persistence |
| News sentiment | ✅ `news_client.py` + `news_sentiment.py` + API router + scheduler wired (07:00 + 12:00 ET) |
| RustAssistant LLM integration | ✅ `openai` SDK — RA primary + Grok fallback — `grok_helper.py`, `chat.py`, `tasks.py` |
| Chat window API | ✅ `POST /api/chat`, `GET /sse/chat`, history, status — multi-turn, market context injected |
| Task/issue capture API | ✅ `POST /api/tasks` — bug/task/note with GitHub push via RA, HTMX feed, Redis pub/sub |
| Logging standard | ✅ structlog + stdlib adopted — `docs/logging.md` written, **all loguru consumers migrated** (LOGGING-A+B done). ~60 stdlib files remain for key-value conversion (LOGGING-C, low priority) |
| Settings Rithmic bug | ✅ **FIXED** — added defensive JS stubs + script tag execution on innerHTML inject |
| Charting WebUI connection | ✅ **FIXED** — port 8001→8000 in Dockerfile/entrypoint/nginx, 8050 fallback in chart.js, web proxy wired |
| Nav: Trading button | ✅ **FIXED** — added missing `🚀 Trading` link to trainer.py and settings.py nav bars |
| PURGE: Tradovate/NT8 bridge | ✅ **DONE** (PURGE-A + PURGE-B) — bridge code removed from settings, positions, sse, health, live_risk, dashboard, trainer, engine/main, position_manager, copy_trader, risk, engine/live_risk |
| NinjaTrader/Tradovate cleanup | ✅ **DONE** (PURGE-A through PURGE-H) — all bridge code, comments, configs, tests, docs cleaned |
| Trade journal | ✅ Full CRUD exists — `journal.py` API + standalone page + HTMX panel + stats/tags |
| Engine settings UI | ✅ Account size ($50K/$100K/$150K), interval, lookback — all in settings page |
| Rithmic account sizing | ✅ **ACCOUNT-SIZE-A/B DONE** — `account_size` field added to config + Redis + UI, wired into RiskManager + CopyTrader per-account sizing. C remains (engine settings simplification) |
| Position Intelligence | ✅ **Scaffolded 2026-03-14** — `position_intelligence.py` + `rithmic_position_engine.py` created with TODO stubs |
| DOM API | ✅ **Scaffolded 2026-03-14** — `dom.py` API routes (snapshot, SSE, config) + registered in data service |
| Static pages | ✅ **Created 2026-03-14** — `chat.html`, `dom.html`, `journal.html` + `static_pages.py` route handler registered |
| Rithmic trade history | ❌ `show_order_history_summary()` / `list_orders()` available but not called — see JOURNAL-SYNC phase |
| Signal naming | ✅ **SIGNAL-NAMING-A/B DONE** — canonical URL `/signals`, nav updated everywhere, `/rb-history` + `/orb-history` → 301 redirects. C/D remain (strategy display improvements) |
| README.md | ✅ **DONE** — NinjaTrader removed, Rithmic added, data hierarchy fixed, project structure updated, ports corrected |
| architecture.md | ✅ **DONE** — data hierarchy fixed, Tradovate sections removed, charting service + Rithmic added, port map updated |
| Pine Script WebUI download | ❌ pine.html exists but no download-from-browser flow — see PINE-WEBUI |
| DOM (Depth of Market) | 🟡 API scaffolded + HTML created — needs Rithmic live data for real feed |
| Chat page | 🟡 HTML created at `/chat` — needs backend wiring verification |
| Journal page | 🟡 Standalone HTML at `/journal` — needs Rithmic fill sync for auto-population |
| Kraken integration | ✅ REST + WebSocket client, crypto ORB sessions, portfolio queries + tick-level trade streaming |
| Simulation environment | ✅ **BUILT 2026-03-15** — SimulationEngine + API routes + DOM live data (gated by `SIM_ENABLED=1`) |
| 1-year rolling data | ✅ **BUILT 2026-03-15** — DataSyncService background task, 365-day backfill, 5-min incremental, retention cleanup |
| WebUI API key management | ❌ Not started — API keys still in .env, should move to settings page |

---

## ✅ Phase ACTIVE-TRAINING — CNN v8 (Complete — Needs Retrain)

> v8 training completed 2026-03-13. **Did NOT beat v6 champion** (83.3% vs 87.1%).
> Severe overfitting: 99.7% train acc vs 83.3% val acc = 16.4% gap.
> Promoted to `breakout_cnn_best.pt` (passed 80% gate) but v6 remains the real champion.
> See Phase RETRAIN below for the fix plan.

### ✅ Completed
- [x] Training completed: 60 epochs, best val_acc=83.3% at epoch 56
- [x] Evaluation: 83.3% acc / 83.4% prec / 83.3% rec (5,355 val samples)
- [x] Model promoted (passed 80% gate — but below v6's 87.1%)
- [x] Feature contract v8 written (37 tabular + embeddings)

### Issues Found (addressed in RETRAIN phase)
- **Overfitting**: Train acc hit 90% by epoch 15 while val plateaued at ~80%. Last 40 epochs wasted.
- **44% missing images**: 24,259 of 54,634 train CSV rows dropped (images don't exist on disk)
- **ORB dominance**: 96.4% of training data is ORB type — model barely sees other strategies
- **Session imbalance**: 87% US session, 13% London — only 2 of 9 sessions represented
- **Dead strategies**: Weekly, Monthly, InsideDay produce 0 trades across all assets (need more data)

---

## 🔴 Phase RETRAIN — CNN v9 Retrain with Fixes

> Fix the v8 issues and retrain. Goal: beat v6 champion (87.1% acc) with the v8 architecture.
>
> **Dataset validation (2026-03-14 oryx session):**
> - 28,548 rows, ALL images verified ✅ (Docker `/app/` prefix fixed)
> - Labels: good_long 26.7%, good_short 26.3%, bad_long 24.5%, bad_short 22.6% — well balanced
> - Breakout types: ORB 95.8%, BollingerSqueeze 2.1%, Fibonacci 1.3%, Consolidation 0.5% — **still heavily ORB-dominant**
> - Sessions: all 9 represented (US 17.2%, london_ny 12.9%, london 12.3%, ...) — fixed from 87% US
> - Symbols: 18 symbols, focus 9 well-represented (8.7–11.0% each), forex tiny (0.1–0.8%)
> - CUDA training verified: 2-epoch test passed on RTX 3080 16GB
>
> **Scripts created for retrain workflow:**
> - `scripts/fix_dataset_paths.py` — strips Docker `/app/` prefix (done ✅)
> - `scripts/validate_dataset.py` — full dataset health report
> - `scripts/test_training_local.py` — local CUDA training test (3 epochs on real data)
> - `scripts/run_full_retrain.py` — orchestrates full v9 pipeline via trainer HTTP API
> - `scripts/run_per_group_training.py` — per-group training orchestrator

### RETRAIN-A: Fix missing images (biggest free win)
- [x] Investigate why 44% of CSV rows have no corresponding image files — **root cause: Docker `/app/` path prefix in CSV, images existed on disk all along**
- [x] Re-run dataset generation with `skip_existing=False` (or just for the missing images) — added `POST /train/repair` endpoint + `step="repair"` pipeline
- [x] Verify all 54,634 rows now have valid images before retraining — added `GET /train/validate` endpoint
- [x] **Fix dataset paths** — stripped `/app/` prefix from labels.csv, train.csv, val.csv (2026-03-14 oryx session). All 28,548 images verified on disk ✅
- [x] Expected impact: nearly doubles effective training data (30K → 55K samples) — **CONFIRMED: was a path issue not missing data**

### RETRAIN-B: Get more historical data
- [x] Current: ~50,000 bars per asset (~88-111 trading days)
- [x] Target: increase `CNN_RETRAIN_DAYS_BACK` to 365 (1 year) — changed `DEFAULT_DAYS_BACK` default from 180 → 365 in `trainer_server.py`
- [x] PrevDay/Weekly/Monthly/InsideDay all need longer timeframes to generate trades
- [x] This is the most impactful change for minority strategy data
- [ ] TODO: Re-generate dataset with 365 days to get more minority strategy samples (run `scripts/run_full_retrain.py --start-step 3`)

### RETRAIN-C: Try per-asset and per-group training
- [x] Infrastructure is now built: `train_mode=per_asset|per_group|combined` in TrainRequest
- [ ] TODO: Run per-group training: `metals` (MGC, SIL), `equity_micros` (MES, MNQ, M2K, MYM), `treasuries` (ZN, ZB), `agriculture` (ZW) — use `scripts/run_per_group_training.py`
- [ ] TODO: Compare per-group val accuracy vs combined model per asset — use `GET /models/compare?model_a=...&model_b=...`
- [ ] If per-group wins, update inference to use `_resolve_model_name()` (already wired)
- [ ] Rationale: ZN/ZB (treasuries) behave very differently from MES/MNQ (equity micros) and ZW (agriculture) — blending hurts

### RETRAIN-D: Address strategy imbalance
- [x] Use weighted sampler to oversample minority strategies (BollingerSqueeze, Fibonacci, Consolidation) — added `WeightedRandomSampler` in `train_model()` with 3× boost
- [ ] TODO: Increase `max_samples_per_type_label` cap (currently 800) after getting 365-day data
- [ ] TODO: Consider adding "no trade" / "no setup" samples for when conditions don't warrant entry
- [ ] **NOTE**: Current dataset is 95.8% ORB — need more data (365 days) to get meaningful minority samples

### RETRAIN-E: Verify regularization improvements take effect
- [x] Regularization already upgraded: dropout 0.5, label smoothing 0.15, weight decay 2e-4, stronger augmentation
- [x] Early stopping patience reduced to 12 — should stop ~epoch 25-30 instead of wasting 40 epochs
- [ ] TODO: Monitor train/val gap — target <8% gap (was 16.4%)
- [x] If still overfitting after RETRAIN-A+B: added Mixup on images (v9: both `imgs` and `tabs` are mixed in the training loop); Stochastic Depth not yet added

### RETRAIN-F: Validate and compare
- [x] Compare v9 vs v6 champion metrics side-by-side — added `compare_models()` in `breakout_cnn.py` + `GET /models/compare` endpoint
- [ ] TODO: Run inference on 10 known signals — sanity check predictions
- [ ] Paper-trade for 1 week with v9 before going live
- [ ] If per-asset models win, deploy the ensemble

**Retrain execution plan (run from oryx):**
1. `python scripts/fix_dataset_paths.py` — ✅ DONE
2. `python scripts/validate_dataset.py` — ✅ DONE (28,548 valid)
3. `python scripts/run_full_retrain.py --trainer-url http://localhost:8200` — TODO: runs steps 3–8
4. `python scripts/run_per_group_training.py --trainer-url http://localhost:8200` — TODO: compare groups

**Files**: `src/lib/analysis/ml/breakout_cnn.py`, `src/lib/services/training/trainer_server.py`, `src/lib/services/training/dataset_generator.py`
**New scripts**: `scripts/fix_dataset_paths.py`, `scripts/validate_dataset.py`, `scripts/test_training_local.py`, `scripts/run_full_retrain.py`, `scripts/run_per_group_training.py`
**Estimated effort**: 2–3 sessions (mostly waiting for training runs)

---

## ✅ Phase SETTINGS-FIX — Rithmic Account Add Button Bug

> **FIXED 2026-03-13.** Root cause: `loadRithmicPanel()` used `panel.innerHTML = html` which
> doesn't execute `<script>` tags. The `addRithmicAccount()` function was defined in the
> dynamically-loaded panel's `<script>` block, so it never ran.

### ✅ SETTINGS-FIX-A: Fix addRithmicAccount JS error
- [x] Added defensive JS stubs (`window.addRithmicAccount`, `saveRithmicAccount`, etc.) in the static settings HTML — if clicked before panel loads, they trigger `loadRithmicPanel()`
- [x] Updated `loadRithmicPanel()` to extract and re-create `<script>` elements after innerHTML inject — forces browser to execute the panel's JS
- [x] Also fixed: "🚀 Trading" nav button was missing from both trainer.py and settings.py nav bars (inconsistent with `_SHARED_NAV_LINKS`)

**Files changed**: `src/lib/services/data/api/settings.py`, `src/lib/services/data/api/trainer.py`

---

## ✅ Phase CHARTING-FIX — Charting Container WebUI Connectivity

> **FIXED 2026-03-13.** All port mismatches corrected, web proxy wired.

### ✅ CHARTING-FIX-A: Port mismatch in Dockerfile defaults
- [x] `docker/charting/Dockerfile`: `ENV DATA_SERVICE_URL=http://data:8001` → `http://data:8000`
- [x] `docker/charting/entrypoint.sh`: fallback `8001` → `8000`
- [x] `docker/charting/nginx.conf.template`: comment `8001` → `8000`

### ✅ CHARTING-FIX-B: chart.js fallback port
- [x] `docker/charting/static/chart.js`: last-resort fallback `8001` → `8050` (external-facing port)

### ✅ CHARTING-FIX-C: Web service proxy wired
- [x] `docker-compose.yml`: `CHARTING_SERVICE_URL=http://localhost:8003` → `http://charting:8003`
- [x] `src/lib/services/web/main.py`: added `/charting-proxy/` and `/charting/` proxy routes to charting service, with dedicated httpx client

### CHARTING-FIX-D: Verify end-to-end (needs manual test)
- [ ] Test: open `http://100.122.184.58:8080` dashboard → Charts tab → verify candlestick chart loads
- [ ] Test SSE live updates flow through to the chart
- [ ] Test all indicators (EMA9/21, BB, VWAP, RSI, CVD, VP, AVWAP) render correctly

---

## ✅ Phase PURGE — Remove NinjaTrader, Tradovate, NiceGUI, C# References (ALL DONE 2026-03-13)

> **Decision**: Rithmic is the bridge now. All NinjaTrader, Tradovate, NT8 bridge, and NiceGUI
> references must be removed or replaced with Rithmic equivalents. C# parity comments should
> be updated to generic language (the Python code is now the source of truth).
>
> PURGE-A and PURGE-B completed 2026-03-13. ~30 files remain (comments, docs, configs).

### ✅ PURGE-A: Remove Tradovate bridge code + settings UI (DONE 2026-03-13)
- [x] `settings.py` — removed `_get_bridge_heartbeat()`, "Broker Bridge Host/Port" fields, "Broker Bridge (Tradovate)" card, `bridge_host`/`bridge_port` JS, `loadBridgeStatus()`, `get_bridge_status()` endpoint
- [x] `positions.py` — removed `NT_BRIDGE_HOST`/`NT_BRIDGE_PORT` env vars, legacy aliases (`_is_bridge_alive` etc.), `/bridge_status` and `/bridge_orders` routes, `bridge_connected`/`bridge_version` refs
- [x] `sse.py` — removed `_BRIDGE_STALE_SECONDS`, `_get_bridge_status()`, `bridge-status` SSE event
- [x] `health.py` — removed `bridge_connected`/`bridge_state`/`bridge_version`/`bridge_account`/`bridge_age_seconds` from health dict
- [x] `dashboard.py` — renamed `_get_bridge_info()` → `_get_broker_info()`, removed Tradovate refs, renamed `bridge_*` params → `broker_*`
- [x] `live_risk.py` — removed `tradovate_positions`/`tradovate_position_count` refs + Tradovate indicator block
- [x] `trainer.py` — removed bridge status JS block + `dot-bridge`
- [x] `web/main.py` — removed `/settings/services/bridge_status` proxy route, updated SAR docstring

### ✅ PURGE-B: Remove NT8 bridge logic from engine (DONE 2026-03-13)
- [x] `engine/main.py` — removed NT8-bridge-only references, updated `_publish_pm_orders()` and `_handle_*` docstrings
- [x] `position_manager.py` — "NinjaTrader Bridge" → "Rithmic gateway" in all comments/docstrings
- [x] `copy_trader.py` — removed "NinjaTrader bridge" reference
- [x] `risk.py` — "NT8 bridge" → "Rithmic" in all docstrings
- [x] `engine/live_risk.py` — removed `_load_tradovate_positions()`, all `tradovate_positions` param threading, `LiveRiskState` Tradovate fields
- [x] `scripts/generate_mermaid.py` + `generate_mermaid_html.py` — removed `_load_tradovate_positions()` from diagrams

### ✅ PURGE-C: Remove SAR/NinjaTrader sync endpoint (DONE 2026-03-13)
- [x] `src/lib/services/data/api/sar.py` — **deleted** (entire file was NinjaTrader SAR sync endpoint)
- [x] `src/lib/services/web/main.py` — removed `/sar/{path}` proxy route
- [x] `src/lib/services/data/main.py` — removed SAR router import and registration

### ✅ PURGE-D: Update C# parity comments to generic language (DONE 2026-03-13)
- [x] `src/lib/analysis/ml/breakout_cnn.py` — 10+ comment edits: "NinjaTrader BreakoutStrategy" → "Ruby breakout engine", "C# OrbCnnPredictor" → "inference pipeline", etc.
- [x] `src/lib/core/breakout_types.py` — "C# NinjaTrader consumer" → "external consumers" (5 edits)
- [x] `src/lib/core/multi_session.py` — "C# NinjaTrader strategy" → "trading strategy", "NT8 SessionBracket struct" → "session bracket structure" (5 edits)
- [x] `src/lib/core/asset_registry.py` — renamed `_TRADOVATE_ALIASES` → `_LEGACY_ALIASES`, updated comment
- [x] `src/lib/services/training/dataset_generator.py` — "C# PrepareCnnTabular()" → "original tabular preparation", "GetAssetClassNorm()" → "asset class normalisation"
- [x] `src/lib/services/training/rb_simulator.py` — "Bridge.cs" → "original Ruby bridge", "Ruby.cs" → "Ruby indicator logic"
- [x] `src/lib/services/data/api/trades.py` — "NinjaTrader live bridge" → "live trading connection", "NinjaTrader scripts" → "trading scripts"
- [x] `src/lib/services/data/api/risk.py` — "NinjaTrader" → "the broker"

### ✅ PURGE-E: Update tests (DONE 2026-03-13)
- [x] `src/tests/test_positions.py` — "NinjaTrader Live Position Bridge API" → "Live Position API" (6 edits)
- [x] `src/tests/test_data_service.py` — "NinjaTrader bridge" → "Rithmic integration" (2 edits)
- [x] `src/tests/test_phase3_ema9_parity.py` — "live NT8 instance" → "live trading instance", C# parity → "Python-canonical"
- [x] `src/tests/test_risk.py` — "NT8 bridge" → "Rithmic trading connection" / "Rithmic integration"
- [x] `src/tests/test_integration.py` — "NT8 bridge" → "Rithmic integration"
- [x] `src/tests/test_volume_profile.py` — "NT8 Ruby.cs indicator" → "Ruby indicator"
- [x] `src/tests/test_gold_price.py` — "NinjaTrader MGC chart" → "MGC chart"
- [x] `src/tests/test_breakout_types.py` — "old bridge functions" → "old helper functions"

### ✅ PURGE-F: Clean up Grafana/Prometheus configs (DONE 2026-03-13)
- [x] `config/grafana/grafana-dashboard.json` — removed entire "🔗 NinjaTrader Bridge" row section + all 14 `bridge_*` panels (~790 lines)
- [x] `config/grafana/orb-trading-dashboard.json` — removed 6 bridge-only panels, cleaned `bridge_positions_count` fallback
- [x] `config/prometheus/prometheus.yml` — already clean (no `ninjatrader-bridge` job)

### ✅ PURGE-G: Clean up docs + scripts (DONE 2026-03-13)
- [x] `docs/architecture.md` — already clean (Tradovate removed in ARCH-UPDATE)
- [x] `docs/backlog.md` — replaced Phase TBRIDGE (46 lines) with removal note, updated 3 stale Tradovate/NinjaTrader refs
- [x] `docs/STRATEGY_PLAN.md` — "NinjaTrader Bridge" → "Rithmic integration", `.cs` file table → Python equivalents
- [x] `docs/rithmic_notes.md` — already had historical comparison note
- [x] `docs/futures_system_printable.html` — already uses `_load_rithmic_positions()`
- [x] Remaining files (scripts, .gitattributes, .gitignore) — already clean from prior passes

### ✅ PURGE-H: Remove NiceGUI references (DONE 2026-03-13)
- [x] `src/lib/integrations/pine/main.py` — already says "HTMX-based page" (no NiceGUI reference)

**All PURGE phases (A–H) complete.**

---

## 🟡 Phase ACCOUNT-SIZE — Per-Account Sizing from Rithmic

> Account size should come from the Rithmic account config, not a global setting.
> Default $150K but each prop account may have a different size ($50K, $100K, $150K, $300K).
> The engine settings UI already has account size — but it's global, not per-account.

### ✅ ACCOUNT-SIZE-A: Add account_size to RithmicAccountConfig (DONE 2026-03-13)
- [x] Add `account_size: int = 150_000` field to `RithmicAccountConfig.__init__()` in `rithmic_client.py`
- [x] Add to `to_storage_dict()` and `from_storage_dict()` for Redis persistence (with `d.get("account_size", 150_000)` default for backward compat)
- [x] Add to `to_ui_dict()` so it shows in the settings panel
- [x] Update `_render_settings_panel()` — account size dropdown per account (25K / 50K / 100K / 150K / 200K / 300K)
- [x] Update `saveRithmicAccount` JS to include `account_size` in POST body
- [x] Update `save_account_ui` and `save_account_config` endpoints to read and persist `account_size`

### ✅ ACCOUNT-SIZE-B: Wire into risk/position sizing (DONE 2026-03-13)
- [x] `CopyTrader._ConnectedAccount` — added `account_size: int = 150_000` field
- [x] `CopyTrader.add_account()` — stores `config.account_size` on the connected account
- [x] `CopyTrader.get_account_sizes()` — new method returning `dict[str, int]` of all account sizes
- [x] `CopyTrader.send_order_and_copy()` — added `scale_qty_by_account` param; when enabled, scales slave qty by `slave_size / main_size` ratio with compliance logging
- [x] `CopyTrader.status_summary()` — now includes `account_size` per account and top-level `account_sizes` dict
- [x] `RiskManager.update_account_size()` — new method to update size and recalculate `max_risk_per_trade` and `max_daily_loss`
- [x] `engine/main.py` `_get_risk_manager()` — now reads main Rithmic account's `account_size` from Redis before falling back to env var default

### ✅ ACCOUNT-SIZE-C: Engine settings simplification
- [x] Primary interval: default changed to `1m` in settings HTML — `<option value="1m" selected>` is now the default; `ⓘ` tooltip added: "1m is recommended for all intraday strategies"
- [x] Lookback: added `auto` as the first/default option — tooltip explains "System selects the appropriate lookback based on the active strategy type"; `renderStatus()` JS falls back to `auto` for unknown values
- [x] These defaults should be sensible enough that the user rarely needs to touch them

**Files**: `src/lib/integrations/rithmic_client.py`, `src/lib/services/engine/risk.py`, `src/lib/services/engine/position_manager.py`, `src/lib/services/engine/copy_trader.py`
**Estimated effort**: 1–2 sessions

---

## 🔴 Phase JOURNAL-SYNC — Auto-Sync Trade Journal from Rithmic

> The trade journal should automatically update based on Rithmic account fills/orders.
> Currently the journal exists (`journal.py` — full CRUD, standalone page, HTMX panel, stats)
> but requires manual entry. Rithmic has `show_order_history_summary()` and `list_orders()`
> available but not called.

### JOURNAL-SYNC-A: Wire Rithmic order/fill history retrieval
- [ ] In `rithmic_client.py` `refresh_account()`, add calls to `client.show_order_history_summary()` and/or `client.list_orders()` to pull today's fills
- [ ] Store raw fill data in Redis (`rithmic:fills:{account_key}:{date}`) with TTL
- [ ] Create a `get_today_fills(account_key)` method on `RithmicAccountManager`

### JOURNAL-SYNC-B: Auto-populate journal from fills
- [ ] On each `refresh_account()` cycle (or dedicated scheduler), aggregate fills into trade records
- [ ] Match entry fill + exit fill into a complete trade (by symbol, direction, time ordering)
- [ ] Calculate P&L, R:R, hold time, contracts from the fill data
- [ ] Write to `trades_v2` table and `daily_journal` automatically
- [ ] Set `source='rithmic_sync'` to distinguish from manual entries

### JOURNAL-SYNC-C: Multi-account journal view
- [ ] Journal page should show trades from all active Rithmic accounts
- [ ] Add account filter dropdown to journal page
- [ ] Aggregate P&L across accounts for the daily summary
- [ ] Copy-trade orders should appear as linked entries (main → slave)

### JOURNAL-SYNC-D: Trade grading integration
- [ ] The grading endpoint exists (`/api/journal/trades/{id}/grade`) but is a stub — wire to DB
- [ ] After auto-sync populates trades, user can grade them (A/B/C/D/F) from the journal page
- [ ] Store grade + notes in `trades_v2.notes` or a new `grade` column

**Files**: `src/lib/integrations/rithmic_client.py`, `src/lib/services/data/api/journal.py`, `src/lib/services/data/models.py`, `src/lib/services/data/api/trades.py`
**Estimated effort**: 3–4 sessions

---

## 🟡 Phase SIGNAL-NAMING — Unified Signal History & Strategy Naming

> Current naming is inconsistent: nav says "RB History", URL is `/orb-history`,
> page title says "ORB Signal History", HTMX fragment says "Signal History".
> The system now supports 13 breakout types — "ORB History" is misleading.
> Need a unified "Signal History" page that covers all strategy types.

### ✅ SIGNAL-NAMING-A: Rename nav + routes to "Signal History" (DONE 2026-03-13)
- [x] `dashboard.py` `_SHARED_NAV_LINKS`: `("/rb-history", "📅 RB History")` → `("/signals", "📡 Signals")`
- [x] `dashboard.py` hardcoded nav: `📅 RB History` → `📡 Signals`
- [x] `trainer.py` nav: `📅 RB History` → `📡 Signals`
- [x] `settings.py` nav: `📅 RB History` → `📡 Signals`
- [x] New canonical route `/signals` → serves Signal History page with `active_path="/signals"`
- [x] `/rb-history` → 301 redirect to `/signals`
- [x] `/orb-history` → 301 redirect to `/signals`
- [x] Note: `static/trading.html` uses step-based wizard nav — no "RB History" link, no change needed

### ✅ SIGNAL-NAMING-B: Update page title + content (DONE 2026-03-13)
- [x] `_ORB_HISTORY_BODY` heading: `📅 ORB Signal History` → `📡 Signal History`
- [x] Section comment: `# RB History` → `# Signal History`
- [x] Filter pills already support all 13 types — no changes needed, just the title was wrong

### ✅ SIGNAL-NAMING-C: Strategy type display improvements
- [x] Show the strategy type prominently in each signal row (not just as a filter) — type badge already in every row via `_BTYPE_COLORS`
- [x] Color-code strategy types for quick visual scanning — full 13-type color map already applied to row badges and filter pills
- [x] Add strategy type breakdown stats at the top — new `t-panel-inner` stats bar added: total signals, BO rate %, and clickable per-type count pills
- [x] Group by strategy type as an optional view mode — added `?group_by=type` query param; renders collapsible `<details>` per type with signal/breakout counts and a "filter only ↗" HTMX link; "⊞ Group by Type" / "⊞ Flat List" toggle button added to filter bar

### SIGNAL-NAMING-D: Future strategy expansion readiness
- [ ] The system has 13 breakout types, 25+ indicators, and multiple model types — make the signal page ready to show signals from any source
- [ ] Add a "source" column: CNN prediction, Ruby engine, manual, etc.
- [ ] Add confidence score display from the CNN model prediction

**Files**: `src/lib/services/data/api/dashboard.py`, `static/trading.html`, `src/lib/services/data/api/trainer.py`, `src/lib/services/data/api/settings.py`
**Estimated effort**: 1–2 sessions

---

## ✅ Phase README-UPDATE — Full README.md Rewrite (DONE 2026-03-13)

> README was outdated: referenced NinjaTrader for execution, wrong data hierarchy, missing
> charting service, missing Rithmic integration, project structure was stale.

### ✅ README-UPDATE-A: Full rewrite
- [x] **Architecture diagram** — charting service added, NinjaTrader removed, Rithmic added
- [x] **Docker services table** — all 5 services with correct ports and image tags
- [x] **Data hierarchy** — Rithmic → MassiveAPI → Yahoo Finance → Kraken
- [x] **Technologies table** — NinjaTrader replaced with Rithmic, Charting row added
- [x] **Project structure** — full rewrite reflecting actual layout (core/, indicators/, integrations/, model/, trading/, services/, static/, docs/, tests/)
- [x] **Quick Start** — `RITHMIC_*` env vars added
- [x] **Configuration** — Rithmic section added, MassiveAPI description updated
- [x] **Scripts & Tools** — removed NT8 C# patcher references, removed ONNX references
- [x] **Related Repos** — empty section removed
- [x] **Testing** — updated with 2,700+ tests, added `test_copy_trader.py` and `test_ruby_signal_engine.py` examples

**File**: `README.md`

---

## ✅ Phase ARCH-UPDATE — architecture.md Corrections (DONE 2026-03-13)

> Several sections were wrong or outdated — now fixed.

### ✅ ARCH-UPDATE-A: Fix data hierarchy
- [x] Data Ingestion section reordered: Rithmic (primary when creds active) → MassiveAPI (current primary) → Kraken (crypto only) → Yahoo Finance (last-resort fallback)

### ✅ ARCH-UPDATE-B: Remove Tradovate sections
- [x] Removed "Future Sidecar: Tradovate JS Bridge" section entirely
- [x] Updated "Dashboard → Manual Trading Signal Flow" to reference Rithmic CopyTrader
- [x] Updated "Scaling Plan" to reference Rithmic CopyTrader instead of Tradovate/PickMyTrade
- [x] Removed `broker_heartbeat` from Redis Key Schema

### ✅ ARCH-UPDATE-C: Add charting service + Rithmic to topology
- [x] Charting service (ApexCharts + nginx, port 8003) added to topology diagram
- [x] Charting added to port map
- [x] Rithmic connection details added to service responsibilities
- [x] CI/CD image matrix updated to include `:charting`

**File**: `docs/architecture.md`

---

## 🔴 Phase RITHMIC-STREAM — async_rithmic Full Integration

> When Rithmic creds arrive and the account is enabled in the WebUI settings, switch from
> MassiveAPI to Rithmic for live market data. `async_rithmic` supports 4 plants:
> TICKER_PLANT, ORDER_PLANT, HISTORY_PLANT, PNL_PLANT.
>
> Current state: `rithmic_client.py` has `RithmicAccountManager` with short-lived connections
> for account status polling. This phase adds persistent streaming connections.
>
> **Docs**: https://async-rithmic.readthedocs.io/en/latest

### RITHMIC-STREAM-A: Persistent Connection Manager
- [ ] Upgrade `rithmic_client.py` `RithmicAccountManager` — add persistent connection mode alongside current short-lived polling
- [ ] Connection lifecycle: connect on startup if creds present → reconnect on disconnect with exponential backoff
- [ ] Custom reconnection settings per async_rithmic docs
- [ ] Custom retry settings for transient failures
- [ ] Event handlers for connect/disconnect/error
- [ ] Debug logging gated by `RITHMIC_DEBUG_LOGGING=1` env var (already exists)
- [ ] Conformance testing with Rithmic test environment

### RITHMIC-STREAM-B: TICKER_PLANT — Live Market Data
- [ ] **Stream live tick data** — subscribe to tick stream for focus assets (`MGC SIL MES MNQ M2K MYM ZN ZB ZW`)
- [ ] Publish ticks to Redis: `rithmic:ticks:{symbol}` (rolling window, ~5min of ticks)
- [ ] **Stream live time bars** — subscribe to 1m/5m/15m time bar streams
- [ ] Replace MassiveAPI bar polling with Rithmic time bar stream when enabled
- [ ] **Order book** — subscribe to L1 best bid/ask per focus asset
- [ ] **Market depth** — subscribe to full L2 depth (10 levels) for DOM display
- [ ] Publish depth to Redis: `rithmic:depth:{symbol}` with 2s TTL
- [ ] **List exchanges** — populate available exchange list on startup
- [ ] **Search symbols** — symbol search endpoint for the WebUI
- [ ] **Front month contract** — auto-resolve front month for each product code
- [ ] Gate: only subscribe when `RITHMIC_LIVE_DATA=1` env var is set and creds are valid
- [ ] Fallback: if Rithmic connection drops, fall back to MassiveAPI seamlessly

### RITHMIC-STREAM-C: HISTORY_PLANT — Historical Data
- [ ] **Fetch historical tick data** — backfill tick data for training/analysis
- [ ] **Fetch historical time bars** — replace Massive/yfinance for bar backfill when Rithmic creds active
- [ ] Wire into `DataResolver` as highest-priority bar source (above Massive)
- [ ] Cache in Postgres + Redis like existing bar sources

### RITHMIC-STREAM-D: PNL_PLANT — Account P&L Tracking
- [ ] **Account PNL snapshot** — fetch current P&L on demand for all connected accounts
- [ ] **Stream PNL updates** — subscribe to real-time P&L stream per account
- [ ] Publish to Redis: `rithmic:pnl:{account_key}` with live updates
- [ ] Wire into LiveRiskPublisher — replace/augment current risk calculations with real Rithmic P&L
- [ ] Dashboard display: per-account P&L cards, master P&L summary (all accounts combined)
- [ ] Track: daily P&L, weekly P&L, monthly P&L, all-time P&L per account

### RITHMIC-STREAM-E: ORDER_PLANT — Enhanced Order Management
- [ ] **List accounts** — discover all accounts on connection
- [ ] **List orders** — fetch all working/filled orders
- [ ] **Show order history summary** — pull historical order data for journal
- [ ] Wire order history into Journal page auto-population
- [ ] **Cancel an order** / **Cancel all orders** — already implemented in CopyTrader, verify with live connection
- [ ] **Modify an order** — already implemented, verify with live connection
- [ ] **Exit a position** — already implemented in EOD close, verify with live connection

### RITHMIC-STREAM-F: Data Provider Routing Update
- [ ] Update `src/lib/services/data/resolver.py` — add Rithmic as data source with highest priority
- [ ] Priority chain: Rithmic (if creds + enabled) → MassiveAPI (if API key) → yfinance (fallback)
- [ ] Kraken stays separate for crypto only
- [ ] Settings page toggle: "Use Rithmic for market data" checkbox
- [ ] Engine startup: log which data provider is active

**Files**: `src/lib/integrations/rithmic_client.py`, `src/lib/services/data/resolver.py`, `src/lib/services/engine/main.py`, `src/lib/services/engine/live_risk.py`, `docker-compose.yml`

**Estimated effort**: ~6–8 agent sessions

---

## 🟡 Phase KRAKEN-SIM — Kraken Live Tick Simulation Environment

> Use Kraken WebSocket live tick data to simulate the full trading pipeline without Rithmic creds.
> All the same tools work (DOM, charts, account tracking) but signals go to Redis as mock trades.
> Tests with BTC/USD, ETH/USD, SOL/USD — same pipeline as futures, different data source.
>
> **Key insight**: Rithmic gives tick-level data for futures, Kraken gives tick-level data for crypto
> (free, no creds needed for public data). We can test everything with crypto before going live on futures.
>
> **Built 2026-03-15:** SimulationEngine + API routes + DOM live data integration.
> Existing KrakenFeedManager already streams trades via `_handle_trade()` callback.

### KRAKEN-SIM-A: Tick-Level WebSocket Streaming
- [x] Upgrade `kraken_client.py` WebSocket to stream raw tick/trade data (not just OHLC bars) — **already done**: `_handle_trade()` processes `{price, qty, side, ord_type, timestamp}` per trade
- [x] Subscribe to `trade` channel for BTC/USD, ETH/USD, SOL/USD — **already done**: `KrakenFeedManager` subscribes to `trade` channel for all configured pairs
- [ ] Publish ticks to Redis: `kraken:ticks:{pair}` (rolling window, ~5min of ticks) — SimulationEngine receives ticks via `on_tick()` callback, raw tick publishing TBD
- [ ] Build 1m bars from tick aggregation (in addition to Kraken's native OHLC stream)
- [ ] Publish L1 best bid/ask from `spread` channel to Redis: `kraken:l1:{pair}`

### ✅ KRAKEN-SIM-B: Simulation Environment (Mock Trading) (DONE 2026-03-15)
- [x] Create `SimulationEngine` class — receives signals, executes mock fills against live tick data — `src/lib/services/engine/simulation.py` (1,273 lines)
- [x] Mock order fills: limit orders fill when price crosses, market orders fill at current tick — `submit_market_order()`, `submit_limit_order()`, `_check_pending_orders()`
- [x] Track simulated positions, P&L, entry/exit times in Redis (`sim:positions`, `sim:orders`, `sim:pnl`, `sim:trades`) — `_publish_state()` after every state change
- [x] Record all sim trades to Postgres `sim_trades` table for analysis — `_record_trade()` with dual SQLite/Postgres DDL
- [x] Support both Kraken (crypto) and Rithmic (futures) data sources — switch via `SIM_DATA_SOURCE` env var
- [x] Send mock signals to Redis instead of real order flow — same keys, prefixed with `sim:` — all Redis keys use `sim:` prefix
- [x] API routes: `src/lib/services/data/api/simulation_api.py` (370 lines) — `/api/sim/status`, `/api/sim/order`, `/api/sim/close/{symbol}`, `/api/sim/close-all`, `/api/sim/reset`, `/api/sim/trades`, `/api/sim/pnl`, `/sse/sim`
- [x] DOM live data: `dom.py` updated — `_build_live_snapshot()` reads `kraken:live:{ticker}` from Redis, falls back to mock; sim position markers shown on DOM ladder
- [x] Wired into data service lifespan: `SIM_ENABLED=1` env var gates startup, engine stored in `app.state.sim_engine`

### KRAKEN-SIM-C: Pre-Trade Analysis Workflow
- [ ] Pre-trade analysis page: select assets based on daily opportunities (crypto and/or futures)
- [ ] Run CNN + Ruby signals + indicators + news on selected assets
- [ ] Pick assets → send to account monitor/manager for breakout watching
- [ ] Account monitor watches for setups on selected assets, sends mock signals to Redis
- [ ] Track time, prices, P&L, all trade info — same format as live trading

### KRAKEN-SIM-D: Data Source Switching
- [ ] Settings page toggle: "Data Source" dropdown — Rithmic (futures) / Kraken (crypto) / Both
- [x] When Kraken selected: DOM shows crypto order book — `_build_live_snapshot()` reads Kraken live data from Redis, `_CRYPTO_DOM_SYMBOLS` mapping added
- [ ] When Rithmic selected: DOM shows futures depth, charts show futures, signals for futures
- [ ] When Both: parallel tracking of futures + crypto assets, unified dashboard view
- [ ] Trading tools work identically regardless of data source — only the connection layer changes

**Files**: `src/lib/integrations/kraken_client.py`, `src/lib/services/engine/simulation.py` (**new**, 1,273 lines), `src/lib/services/data/api/simulation_api.py` (**new**, 370 lines), `src/lib/services/data/api/dom.py` (updated)
**Estimated effort**: ~~4–5 sessions~~ B done, A partially done, C+D remain (~2–3 sessions)

---

## 🟡 Phase DATA-ROLLING — 1-Year Rolling Data Window in Postgres

> Build and maintain a rolling window of ~1 year of 1-minute data for all enabled assets.
> Data service keeps this in sync. Engine and trainer pull from Postgres (or Redis cache).
>
> **Assets**: 9 futures (MGC SIL MES MNQ M2K MYM ZN ZB ZW) + 3 crypto (BTC/USD ETH/USD SOL/USD)
>
> **Built 2026-03-15:** `DataSyncService` created with background sync, retention, Redis cache.
> Uses existing `historical_bars` table + `backfill_symbol()` from backfill.py — no new table needed.

### ✅ DATA-ROLLING-A: Postgres 1m Bar Storage (DONE 2026-03-15 — uses existing `historical_bars`)
- [x] Create `bars_1m` table — **uses existing `historical_bars`** table from `backfill.py` (already has symbol, timestamp, OHLCV, interval, unique constraint)
- [x] Unique constraint on (symbol, timestamp) — already exists: `UNIQUE (symbol, timestamp, interval)` + `ON CONFLICT DO NOTHING`
- [ ] Partition by month for query performance — deferred (not needed until > 50M rows)
- [x] Retention policy: auto-delete bars older than 13 months — `_enforce_retention(days=395)` in `sync.py`, runs after each sync cycle

### ✅ DATA-ROLLING-B: Data Sync Service (DONE 2026-03-15)
- [x] Background task in data service: sync 1m bars for all enabled assets — `DataSyncService.run()` as `asyncio.Task` in data service lifespan
- [x] Futures: pull from Massive API (current) → Rithmic historical (when creds arrive) — delegates to existing `backfill_symbol()` which routes Massive → yfinance
- [x] Crypto: pull from Kraken REST OHLC API for BTC/USD, ETH/USD, SOL/USD — delegates to existing `_fetch_chunk_kraken()` in backfill.py
- [x] Backfill: on first run, fetch 365 days of history per asset — `_sync_symbol()` checks bar count, does full 365-day backfill if < 200K bars
- [x] Incremental: every 5 minutes, fetch latest bars and upsert — `SYNC_INTERVAL_SECONDS=300` env var, configurable
- [x] Track sync status per asset in Redis: `data:sync:{symbol}` — JSON dict with `last_synced`, `bar_count`, `status`, `error`, `duration_seconds`
- [x] Manual trigger: `POST /api/data/sync/trigger` wakes the sync service from its sleep interval
- [x] Wired into data service lifespan: starts as step 7 after cache warm, stops on shutdown

### ✅ DATA-ROLLING-C: Redis Cache Layer (DONE 2026-03-15)
- [x] Cache recent bars (last 24h) in Redis for fast access: `bars:1m:{symbol}` sorted set — `_warm_redis_cache()` in sync.py
- [ ] Engine/trainer request flow: Redis cache → Postgres → API fallback — partial (bars.py already has this for `get_bars`, trainer needs explicit wiring)
- [x] Data service populates Redis cache from Postgres on startup — existing `startup_warm_caches()` + sync service warms after each symbol sync
- [x] TTL management: Redis bars expire after 25h, refreshed on each sync cycle

### 🟡 DATA-ROLLING-D: Trainer Data Pipeline
- [ ] Trainer can request data from data service instead of fetching directly
- [x] `GET /api/data/bars?symbol=MES&interval=1m&days=365` — serves from Postgres via existing `get_stored_bars()` — route added in `sync_router`
- [ ] Dataset generation uses Postgres bars — no more direct API calls from trainer
- [ ] Enables offline training: once data is in Postgres, no external API needed

**Files**: `src/lib/services/data/sync.py` (**new**, 783 lines), `src/lib/services/data/main.py` (updated — lifespan + router), `src/lib/services/data/api/bars.py` (existing — unchanged, sync service delegates to `backfill_symbol()`)
**New API routes**: `GET /api/data/sync/status`, `POST /api/data/sync/trigger`, `GET /api/data/bars`
**Estimated effort**: ~~3–4 sessions~~ A/B/C done, D partially done (~1 session remaining)

---

## 🟡 Phase WEBUI-KEYS — API Key Management in WebUI Settings

> Move API keys from `.env` file to the WebUI settings page. Keys stored encrypted in Redis
> (same pattern as Rithmic credential storage). `.env` values used as fallback/initial seed.

### WEBUI-KEYS-A: Settings UI for API Keys
- [ ] Add "API Keys" section to settings page with masked input fields
- [ ] Keys to manage: Massive API, Finnhub, Alpha Vantage, Kraken (key + secret), xAI/Grok, Reddit (client ID + secret)
- [ ] Show connection status indicator per key (green dot = valid, red = invalid/missing)
- [ ] "Test Connection" button per service — verify key works
- [ ] Save encrypted to Redis using same Fernet encryption as Rithmic creds

### WEBUI-KEYS-B: Key Resolution Chain
- [ ] Priority: Redis (WebUI-set) → `.env` file → empty (disabled)
- [ ] On startup, if Redis has no keys, seed from `.env` values
- [ ] All services (`massive_client`, `news_client`, `kraken_client`, `grok_helper`, `reddit_watcher`) read from resolver
- [ ] Create `src/lib/core/api_keys.py` — centralized key resolver with caching

### WEBUI-KEYS-C: Security
- [ ] Keys never sent in plaintext over API responses — always masked (show last 4 chars only)
- [ ] Keys encrypted at rest in Redis with app SECRET_KEY derived Fernet key
- [ ] Audit log: log when keys are added/changed/removed (no plaintext in logs)

**Files**: `src/lib/services/data/api/settings.py`, new `src/lib/core/api_keys.py`
**Estimated effort**: 2–3 sessions

---

## 🟡 Phase KRAKEN-ACCOUNTS — Full Kraken Account Management

> Kraken accounts are 100% managed by this app. No compliance restrictions like prop firms.
> Build up spot accounts then futures. Use USDT/USDC as backbone currency.
>
> **Target**: Grow Kraken spot to 5K CAD, Kraken futures to 5K CAD.
> **Pairs**: BTC/USD, ETH/USD, SOL/USD (expandable later)

### KRAKEN-ACCOUNTS-A: Spot Account Management
- [ ] Dashboard card: Kraken spot balances (BTC, ETH, SOL, USDT, USDC, CAD)
- [ ] Auto-rebalancing: maintain target ratios across spot holdings
- [ ] DCA (Dollar Cost Average) scheduler: periodic buys on configurable schedule
- [ ] Trade execution: market and limit orders via Kraken REST API
- [ ] P&L tracking: cost basis, unrealized gains, realized gains per asset

### KRAKEN-ACCOUNTS-B: Kraken Futures Account
- [ ] Dashboard card: Kraken futures positions, margin, P&L
- [ ] Same signal pipeline as Rithmic futures but routed to Kraken futures API
- [ ] Position sizing based on account balance (same risk rules as prop accounts)
- [ ] No compliance restrictions — fully automated execution allowed

### KRAKEN-ACCOUNTS-C: USDT/USDC Backbone
- [ ] Track stablecoin balances as "cash" equivalent
- [ ] Auto-convert profits to USDT/USDC for stability
- [ ] Fund futures margin from stablecoin balance
- [ ] Cross-exchange stablecoin tracking (Kraken + crypto.com + Netcoins)

**Files**: `src/lib/integrations/kraken_client.py`, `src/lib/services/data/api/kraken.py`
**Estimated effort**: 4–5 sessions

---

## 🟢 Phase MULTI-EXCHANGE — Multi-Exchange & Wallet Portfolio Management

> After Kraken is stable, add support for additional exchanges and hardware wallets.
> Track total net worth across all accounts and platforms.
>
> **Exchanges**: Kraken (primary), crypto.com (Visa card + CRO staking), Netcoins (Mastercard)
> **Wallets**: BTC hardware wallet (Coldcard) for long-term holdings
> **Target per platform**: ~5K CAD spot + 5K CAD futures where available

### MULTI-EXCHANGE-A: crypto.com Integration
- [ ] REST API client for crypto.com exchange
- [ ] Track spot balances, CRO staking balance, Visa card cashback
- [ ] 5K CAD spot target with CRO for Visa card tier benefits
- [ ] 5K CAD futures account management
- [ ] Dashboard card: crypto.com balances + Visa card status

### MULTI-EXCHANGE-B: Netcoins Integration
- [ ] REST API client for Netcoins exchange
- [ ] Track spot balances, Mastercard integration
- [ ] 5K CAD account target
- [ ] Dashboard card: Netcoins balances

### MULTI-EXCHANGE-C: BTC Hardware Wallet Tracking
- [ ] Public key / xpub tracking for Coldcard wallet
- [ ] Blockchain API balance lookup (no private keys in app — read-only)
- [ ] Dashboard card: long-term BTC holdings with current CAD value
- [ ] Historical balance chart (BTC amount is static, value fluctuates)

### MULTI-EXCHANGE-D: Unified Net Worth Dashboard
- [ ] Aggregate all accounts: Rithmic (prop firms), Kraken, crypto.com, Netcoins, hardware wallets
- [ ] Total net worth in CAD with breakdown by platform and asset
- [ ] Daily/weekly/monthly P&L across all accounts
- [ ] Deposits and withdrawals tracking across all platforms
- [ ] Expense tracking: Visa/Mastercard crypto payments, gift card purchases

### MULTI-EXCHANGE-E: Tax Reporting (Canada)
- [ ] Export capital gains report for Canadian tax filing
- [ ] Track adjusted cost base (ACB) per asset across all exchanges
- [ ] Capital gains taxed at 50% inclusion rate (up to $250K, then 66.7%)
- [ ] Generate CSV/PDF export for accountant
- [ ] Track deposits vs withdrawals vs trading gains separately
- [ ] Note: at scale (>$250K gains), consider incorporating as a trading company

**Files**: new `src/lib/integrations/crypto_com_client.py`, new `src/lib/integrations/netcoins_client.py`, new `src/lib/services/data/api/portfolio.py`
**Estimated effort**: 8–10 sessions (future phase — after Kraken is stable)

---

## 🟡 Phase DOM — Depth of Market (Simple DOM)

> Build a simple DOM (Depth of Market) ladder that can be used with Rithmic tick data streams.
> Initially read-only (visualize order book). Later, add click-to-trade for manual order entry.
> This becomes an alternative to the "SEND ALL" button for more precise entries.
>
> **Created 2026-03-14:** API routes (`dom.py`), SSE stream, static page (`dom.html`), route handler (`static_pages.py`)

### DOM-A: DOM Data Pipeline
- [x] SSE endpoint: `GET /sse/dom?symbol=MES` — stream DOM updates to browser (mock data, 1s refresh) — `src/lib/services/data/api/dom.py`
- [x] API endpoint: `GET /api/dom/snapshot?symbol=MES` — current DOM state (mock data) — `src/lib/services/data/api/dom.py`
- [x] API endpoint: `GET /api/dom/config` — DOM display configuration
- [x] Routes registered in `data/main.py` (both `dom_router` and `dom_sse_router`)
- [ ] TODO: Subscribe to Rithmic market depth (L2) for active symbol via RITHMIC-STREAM-B — replace mock data
- [ ] TODO: Build DOM state object: price ladder with bid/ask quantities at each level
- [ ] TODO: Publish DOM state to Redis: `rithmic:dom:{symbol}` (rolling, 1s updates)

### DOM-B: DOM UI Component
- [x] New `static/dom.html` created — dark theme, price ladder with bid/ask bars, symbol selector
- [x] Route handler at `GET /dom` via `static_pages.py`
- [ ] TODO: Highlight POC from volume profile (requires real volume data)
- [ ] TODO: Last trade indicator: arrow showing last trade at price (requires tick stream)
- [ ] TODO: Cumulative delta at each level (requires tick stream)
- [ ] TODO: Auto-center on last trade price, with manual scroll

### DOM-C: DOM Click-to-Trade (Phase 2 — after funded)
- [ ] Click bid side → place limit buy at that price
- [ ] Click ask side → place limit sell at that price
- [ ] Market buy/sell buttons at top
- [ ] All orders tagged `MANUAL` per compliance
- [ ] Bracket order support: click entry → auto-attach stop + target
- [ ] Order display: show working orders on the DOM ladder
- [ ] Cancel order: click working order on ladder to cancel
- [ ] Confirmation modal before sending (same compliance flow as "SEND ALL")

**Files**: new `static/dom.html` or additions to `static/trading.html`, new API routes, `src/lib/integrations/rithmic_client.py`

**Estimated effort**: ~4–5 agent sessions (A+B without Rithmic data can use mock; C needs funded accounts)

---

## 🟡 Phase PINE-WEBUI — Pine Script Generator Download from WebUI

> `static/pine.html` already exists with module viewer, params editor, and generate button.
> But the generated Pine Script file needs a proper download-from-browser flow.

### PINE-WEBUI-A: Verify generate + download flow
- [ ] Test `POST /api/pine/generate` → verify it generates `ruby.pine` from modules + params
- [ ] Test `GET /api/pine/output` → verify it lists generated files
- [ ] Test `GET /api/pine/download/{filename}` → verify it returns the file with `Content-Disposition: attachment`
- [ ] Verify the "⬇ Download" button in `pine.html` (line ~860 area) actually triggers a browser download
- [ ] If the download endpoint doesn't exist, create it in `src/lib/services/data/api/pine.py`

### PINE-WEBUI-B: Improve the generate experience
- [ ] After clicking "Generate", auto-preview the output in the Generated Preview card
- [ ] Add a "Copy to Clipboard" button for the generated Pine Script
- [ ] Show generation stats: module count, total lines, file size
- [ ] Add a "Download Latest" button that always downloads the most recent `ruby.pine`
- [ ] Verify all 16 Pine modules are listed and viewable in the module browser

### PINE-WEBUI-C: Route verification
- [ ] Verify `/api/pine/*` routes are registered in `src/lib/services/data/main.py`
- [ ] Verify `/api/pine/*` routes are proxied through `src/lib/services/web/main.py`
- [ ] Test from browser at `http://100.122.184.58:8080/pine`

**Files**: `static/pine.html`, `src/lib/integrations/pine/generate.py`, `src/lib/services/data/api/pine.py`

---

## 🟡 Phase UI-SPLIT — Split Inline HTML into Static Files

> The trading dashboard has a different look/feel from the rest of the app. We should
> split the large inline HTML strings from Python files into separate static HTML files
> under `static/` for consistency and maintainability.

### UI-SPLIT-A: Inventory inline HTML
- [ ] `src/lib/services/data/api/settings.py` — `_SETTINGS_PAGE_HTML` is ~1,600 lines of inline HTML (line 142–1740)
- [ ] `src/lib/services/data/api/dashboard.py` — main dashboard HTML is inline (~6,500 lines total file)
- [ ] `src/lib/services/data/api/trainer.py` — trainer page HTML inline
- [ ] Already split: `static/trading.html` (~3,990 lines), `static/pine.html` (~1,619 lines)

### UI-SPLIT-B: Extract to static files
- [ ] Extract `_SETTINGS_PAGE_HTML` → `static/settings.html` and serve with `FileResponse`
- [ ] Extract dashboard HTML → `static/dashboard.html`
- [ ] Extract trainer HTML → `static/trainer.html`
- [ ] Update Python handlers to load from file instead of string constants
- [ ] Ensure HTMX dynamic fragments still work (they load via separate API endpoints, not inline)

### UI-SPLIT-C: Visual consistency pass
- [ ] Audit CSS variables across all pages — ensure same dark theme, same fonts, same spacing
- [ ] `trading.html` uses JetBrains Mono + Syne — verify other pages match
- [ ] Standardize card styling, button styles, status indicators across all pages
- [ ] Ensure all pages work with the same nav structure

**Files**: `static/`, `src/lib/services/data/api/settings.py`, `src/lib/services/data/api/dashboard.py`, `src/lib/services/data/api/trainer.py`

---

## 🟡 Phase TESTS — More Edge Case Coverage

> Need more tests for edge cases across the system.
>
> **Added 2026-03-14:** 93 new tests across 3 files, all passing.

### TESTS-A: Rithmic integration tests
- [x] Test encrypted credential storage/retrieval round-trip — `test_rithmic_account.py` (49 tests) ✅
- [x] Test `RithmicAccountConfig` defaults, creation, serialization round-trip
- [x] Test `_derive_fernet_key`, `_encrypt`, `_decrypt` round-trips
- [x] Test `to_ui_dict` masks credentials (no plaintext leaks)
- [x] Test `RithmicAccountManager` initialization with mocked Redis
- [ ] TODO: Test `addRithmicAccount` → save → test connection → remove lifecycle (needs live Rithmic)
- [ ] TODO: Test prop firm preset application (TPT, Apex, etc.)
- [ ] TODO: Test EOD close with multiple accounts (mock Rithmic client)
- [ ] TODO: Test rate limiter edge cases: exactly at warn threshold, exactly at hard limit, rollover at midnight

### TESTS-B: Pipeline edge cases
- [x] Test dataset validation: valid CSV, missing images, empty CSV, missing columns — `test_dataset_validation.py` (13 tests) ✅
- [x] Test trainer endpoints: health, status, train, cancel, validate, models, logs — `test_trainer_endpoints.py` (31 tests) ✅
- [ ] TODO: Test pipeline with missing/partial data (some symbols have no bars)
- [ ] TODO: Test pipeline with stale Redis cache
- [ ] TODO: Test CNN inference with v6 model on v8 features (backward compat padding)
- [ ] TODO: Test dataset generation with < 5 days of data per symbol

### TESTS-C: WebUI endpoint tests
- [ ] TODO: Test all `/api/pine/*` endpoints (generate, download, modules, params)
- [ ] TODO: Test `/api/copy-trade/*` endpoints with mock CopyTrader
- [ ] TODO: Test `/api/rithmic/*` endpoints with mock RithmicAccountManager
- [ ] TODO: Test `/api/dom/*` endpoints (snapshot, config, SSE) — routes already registered
- [ ] TODO: Test SSE streams don't leak connections on client disconnect

### TESTS-D: Fix known test gaps
- [ ] TODO: Mock network calls in `test_swing_engine_grok.py` (3 tests timeout on real yfinance + Grok)
- [ ] TODO: Fix `test_risk.py::TestRiskManagerInit::test_default_params` — `max_daily_loss` assertion stale after ACCOUNT-SIZE changes
- [x] Add `jsonschema` to `pyproject.toml` — added `jsonschema>=4.23.0` after `pydantic` in the Web/API group
- [x] Add `psutil` to `pyproject.toml` — added `psutil>=6.0.0` in the Observability group

---

## 🟡 Phase SIGNALS — Focus Asset Signal Generation

> Generate signals only for our training assets until further notice.

### SIGNALS-A: Restrict signal generation
- [ ] Verify `CNN_RETRAIN_SYMBOLS` env var matches: `MGC,SIL,MES,MNQ,M2K,MYM,ZN,ZB,ZW`
- [ ] Engine signal generation should only fire CNN inference + Ruby signals for these 9 symbols
- [ ] Other assets (forex, low-liquidity) still tracked in broad analysis (daily focus, regime) but no trade signals
- [ ] Settings page: editable focus symbol list that gates signal generation
- [ ] Dashboard focus cards: only show focus assets, gray out or hide non-focus

### SIGNALS-B: Broad market tracking (no signals)
- [ ] Continue forex/commodity data ingestion for cross-asset correlation features (v8-B)
- [ ] Continue regime detection across all tracked assets for global market context
- [ ] Display "tracking only" badge on non-focus assets in dashboard

---

## 🟢 Phase PROFIT — Post-Funding Profit Allocation Plan

> After first funded account profits. Not code tasks — financial planning tracked here.

### Profit allocation (% of net profits after fees):
1. **Buy more prop accounts** — reinvest to scale from 5 → 20 accounts
2. **Fund Kraken spot account** — build up crypto spot trading ratios for our pairs (fully managed by the app, no restrictions)
3. **Personal draw** — living expenses + personal needs
4. **Long-term BTC accumulation**:
   - Buy Bitcoin → Coldcard hardware wallet
   - Track long-term BTC wallet balance via public key (read-only, no signing)
   - Can add wallet balance tracking to dashboard

### Tracking features to build:
- [ ] Per-account profit tracking via Rithmic PNL_PLANT (RITHMIC-STREAM-D)
- [ ] Master P&L dashboard: all accounts combined, daily/weekly/monthly views
- [ ] Allocation calculator: input net profit → output split per category
- [ ] Kraken account balance display (already have `kraken_client.py`)
- [ ] BTC cold storage balance via public key lookup (blockchain API)

---

## ✅ Recently Completed — Phase 1 & 2 (Dashboard Integration)

### Phase 1 — Blocking items resolved
- [x] `TRAINER_SERVICE_URL` moved from hardcode to env var
- [x] `ENGINE_DATA_URL` port fixed (8100 → 8050)
- [x] `sync_models.sh` audited — platform-agnostic

### Phase 1.5 — PORT fix + dataset smoke test
- [x] `scripts/smoke_test_dataset.py` — validates engine connectivity + bar loading
- [x] Port mismatch verified in `docker-compose.yml`, `docker-compose.trainer.yml`, CI/CD workflow

### Phase 2 — Trading Dashboard integrated into production services
- [x] `src/lib/services/data/api/pipeline.py` — 17 routes, SSE streaming, unified analysis runner
- [x] `static/trading.html` — 4-page SPA (Pipeline → Plan → Live → Journal)
- [x] Docker build: `static/` COPY'd into image, served by FastAPI `StaticFiles`
- [x] 12 direct-proxy routes in `web/main.py` for pipeline/plan/actions/settings
- [x] Tests: 2543 passed, 0 failed at time of integration

---

## ✅ Phase RA-CHAT — RustAssistant Chat Window & Task Capture

### What was built
#### RA-CHAT-A: openai SDK standardisation (`grok_helper.py`)
- [x] Migrated from raw `httpx` to official `openai` SDK with `base_url` override
- [x] RustAssistant (RA) primary; Grok fallback; offline mode if neither available
- [x] `build_system_prompt()` — merges `SYSTEM_PROMPT.md` template with live market context
- [x] `generate_response()` — single entry point for all chat, task, and analysis calls
- [x] Removed `extra_headers` / `Content-Type` manual setting (SDK handles it)

#### RA-CHAT-B: Chat API router (`src/lib/services/data/api/chat.py`)
- [x] `POST /api/chat` — multi-turn chat with market context injection
- [x] `GET /sse/chat` — SSE streaming (token-by-token from RA/Grok)
- [x] `GET /api/chat/history` — session history (Redis, 2h TTL)
- [x] `GET /api/chat/status` — LLM provider status, model name, uptime

#### RA-CHAT-C: Task / issue capture (`src/lib/services/data/api/tasks.py`)
- [x] `POST /api/tasks` — create bug / task / note with severity + category
- [x] GitHub push: RA formats and pushes to `futures-signals` or `futures` repo
- [x] HTMX feed: `GET /htmx/tasks/feed` — recent tasks as HTML fragment
- [x] Redis pub/sub: `dashboard:tasks` channel for real-time notification

#### RA-CHAT-D: Service wiring
- [x] Routers registered in `data/main.py`; proxied in `web/main.py`
- [x] Tests for all endpoints (mocked LLM)

### Environment variables added
| Variable | Default | Description |
|---|---|---|
| `RA_API_KEY` | *(unset)* | RustAssistant API key (OpenAI-compatible) |
| `RA_BASE_URL` | `https://ra.example.com/v1` | RA server base URL |
| `RA_MODEL` | `ra-4` | RA model name |
| `XAI_API_KEY` | *(unset)* | xAI Grok fallback key |
| `XAI_MODEL` | `grok-3-mini` | Grok model name |

### 🟡 RA-CHAT — Next Up
#### RA-CHAT-E: Chat page HTML (`/chat`)
- [x] Standalone chat page created — `static/chat.html` (2026-03-14)
- [x] Message input with send button + Ctrl+Enter shortcut
- [x] Chat history display with user/assistant message bubbles (dark theme)
- [x] Market context indicator (shows what data is being injected)
- [x] Task creation shortcut from chat ("create task: ..." prefix detection)
- [x] Route handler at `GET /chat` via `static_pages.py`, registered in data service
- [x] Wired to `POST /api/chat`, `GET /sse/chat`, `GET /api/chat/history`, `GET /api/chat/status`
- [ ] TODO: Verify end-to-end with RA/Grok backend running

#### RA-CHAT-F: Dashboard integration
- [ ] TODO: Chat widget/drawer accessible from all pages
- [ ] TODO: Task feed in sidebar or notification area

#### RA-CHAT-G: Intent detection in chat
- [ ] TODO: Detect when user asks about a symbol → inject live data for that symbol
- [ ] TODO: Detect task/bug creation intent → route to task API

#### RA-CHAT-H: RustAssistant GitHub actions (requires RA server config)
- [ ] TODO: RA opens PRs for identified issues
- [ ] TODO: RA runs test suite and reports results

---

## ✅ Phase NEWS — News Sentiment Pipeline

> Completed. Multi-source hybrid sentiment: Finnhub + Alpha Vantage + VADER + Grok 4.1.

- [x] NEWS-A: `news_client.py` — Finnhub + AlphaVantage data collectors
- [x] NEWS-B: `news_sentiment.py` — VADER + 60 futures-specific lexicon terms + Grok batch scoring
- [x] NEWS-C: Scheduler at 07:00 ET + 12:00 ET, Redis cache with 2h TTL, spike detection
- [x] NEWS-D: API routes (`/api/news/*`), HTMX panel, web proxy

---

## ✅ Phase CHARTS — Charting Service Volume Indicators

> Completed. Full charting service with ApexCharts.

- [x] VWAP with ±1σ / ±2σ bands (daily reset)
- [x] CVD (Cumulative Volume Delta) sub-pane
- [x] Volume Profile — POC / VAH / VAL (100-bar rolling)
- [x] Anchored VWAP — session open + previous day low
- [x] EMA9 / EMA21 overlays
- [x] Bollinger Bands overlay
- [x] RSI sub-pane
- [x] localStorage indicator preference persistence
- [x] SSE live bar updates

---

## ✅ Phase RITHMIC — Copy Trading & Prop-Firm Compliance

> All sub-phases complete. `CopyTrader` class, compliance logging, position manager wiring,
> rate limiting, pyramiding, Ruby signal engine, WebUI integration — 114 tests passing.

### ✅ RITHMIC-A: CopyTrader Class (Core Multi-Account Engine)
- [x] `CopyTrader` class with main + slave account management
- [x] `send_order_and_copy()`, `send_order_from_ticker()`, `execute_order_commands()`
- [x] `TICKER_TO_RITHMIC` mapping, front-month contract cache
- [x] `RollingRateCounter`, compliance logging, Redis pub/sub
- [x] 79 → 114 tests passing

### ✅ RITHMIC-B: Compliance — MANUAL Flag + Humanized Delays
- [x] Every `submit_order` uses `OrderPlacement.MANUAL`
- [x] 200–800ms delay between slave copies (1–2s in high-impact mode)
- [x] Compliance log persisted to Redis (7-day TTL)

### ✅ RITHMIC-C: PositionManager → CopyTrader Wiring + Server-Side Brackets
- [x] `stop_price_to_stop_ticks()` + `TICK_SIZE` table for 14 micro products
- [x] `modify_stop_on_all()`, `cancel_on_all()`, `execute_order_commands()`
- [x] Gated by `RITHMIC_COPY_TRADING=1` env var

### ✅ RITHMIC-D: Rate-Limit Monitoring & Safety
- [x] Rolling 60-min warn at 3,000 / hard stop at 4,500
- [x] Daily action counter, rate limit error detection

### ✅ RITHMIC-E: PositionManager Upgrades (One-Asset Focus + Pyramiding)
- [x] Focus lock (`PM_FOCUS_LOCK=1`), quality-gated pyramiding (L1/L2/L3)
- [x] Max 3 contracts, 15-min cooldown, regime + wave gates for L3

### ✅ RITHMIC-F: WebUI Integration
- [x] "SEND ALL" button, pyramid button, compliance modal
- [x] Account status cards, copy-trade log, rate-limit strip, focus lock display

### ✅ RITHMIC-G: Ruby Signal Engine (Pine → Python Port)
- [x] Full Pine Script v6 port — 10 sections, all indicators
- [x] Wired into engine via `handle_ruby_recompute()` (every 5 min)
- [x] API routes (`/api/ruby/*`), HTMX status fragments
- [x] 114 tests passing total across copy trader + Ruby engine

---

## ✅ Phase INDICATORS — Codebase Reorganization & Indicator Library Integration

> Completed. Indicators library copied from reference, deduplicated, wired into analysis pipeline.

- [x] INDICATORS-A: Fixed import paths, established base classes
- [x] INDICATORS-B: Resolved duplicates & column inconsistencies
- [x] INDICATORS-C: Extracted pure math from analysis into indicators
- [x] INDICATORS-D: Reorganized `src/lib/analysis/` — `rendering/`, `sentiment/`, `ml/` sub-packages
- [x] INDICATORS-E: Wired indicators into analysis pipeline with presets
- [x] INDICATORS-G: Cleanup & documentation (partial)
- [ ] INDICATORS-F: Reference code evaluation — some items deferred (see `docs/backlog.md`)

---

## ✅ Phase TRAINER-UX — Trainer Page Redesign & Defaults Update

- [x] Trainer defaults updated (60 epochs, patience 12, LR 0.0001, 180 days back)
- [x] Trainer page redesign with live log streaming
- [x] Charting container connectivity: `CHARTING_SERVICE_URL=http://charting:8003` added
- [x] Dead code removed
- [x] **📋 Copy All** button fix (2026-03-13): `navigator.clipboard` fails on non-HTTPS (Tailscale IP) — added `document.execCommand('copy')` fallback + error handling

---

## ✅ Phase CLEANUP — Full-Project Lint & Test Sweep (2026-03-11)

| Metric | Before | After |
|---|---|---|
| **Ruff errors** | **2,615** | **0** ✅ |
| **Broken imports** | **~15 files** | **0** ✅ |
| **Failing tests** | **20 failures** | **0** ✅ (2,851 pass, 1 skip) |

- [x] CLEANUP-1: Ruff auto-fix (2,075 errors)
- [x] CLEANUP-2: Fixed `from lib.core.*` broken imports
- [x] CLEANUP-3: Fixed `from src.lib.*` broken imports
- [x] CLEANUP-4: Manual ruff fixes (60 remaining)
- [x] CLEANUP-5: Newly-created core files cleaned
- [x] CLEANUP-6: Test failures fixed (alerts, position_manager, crypto_momentum, backfill, data_provider_routing)

### Known Remaining Issues (not blocking)
- **3 tests in `test_swing_engine_grok.py`** timeout due to real network calls — need mocked network access
- **`jsonschema` not in dependencies** — imported by `core/exceptions/validation.py` and `loader.py`
- **`psutil` not in dependencies** — imported by `core/health.py`
- **`src/lib/model/__init__.py` is empty** — needs guarded re-exports (see MODEL-INT-E)

---

## 🟡 Phase POSINT — Position Intelligence Engine

> The core "live trading co-pilot." Real-time per-position analysis.
> Builds with mock data first; swaps to real Rithmic when creds arrive.
>
> **Scaffolded 2026-03-14:** Both core modules created with full structure + TODO stubs.

### POSINT-A: Position Intelligence Module
- [x] `src/lib/services/engine/position_intelligence.py` created (495 lines) — `compute_position_intelligence()`, `compute_multi_tp()`, `assess_book_pressure()`, `suggest_risk_actions()`
  - Lazy imports for: `ict_summary`, `check_confluence`, `compute_volume_profile`, `compute_cvd`, `RegimeDetector`
  - `PositionIntel` dataclass with `to_dict()` serializer
  - Each sub-call in try/except with sensible defaults
- [ ] TODO: Wire real analysis modules (currently returns mock/default data)
- [ ] TODO: Add tests for position_intelligence.py

### POSINT-B: Rithmic Position Engine Wrapper
- [x] `src/lib/services/engine/rithmic_position_engine.py` created (384 lines) — `RithmicPositionEngine` class
  - Methods: `connect()`, `get_positions()`, `get_l1()`, `get_l2()`, `get_recent_trades()`, `get_pnl()`, `is_connected()`
  - Lazy imports of `RithmicStreamManager` / `get_stream_manager`
  - All methods return realistic mock data with TODO comments for real wiring
- [ ] TODO: Wire to live Rithmic stream when creds available

### POSINT-C: Position Intelligence API Routes
- [ ] TODO: `GET /api/live/positions` — SSE stream with full intel payload
- [ ] TODO: `GET /api/live/book?symbol=MES` — L1 + L2 depth snapshot
- [ ] TODO: `GET /api/live/tape?symbol=MES&n=20` — time & sales

### POSINT-D: Live Page UI Enhancement
- [ ] TODO: Per-position cards: book, DOM pressure, TP zones, risk actions
- [ ] TODO: Session stats bar, Rithmic connection banner

---

## 🟡 Phase UI-ENHANCE — Trading Dashboard Improvements

### UI-A: Research Page
- [ ] Cross-asset context panel (ES/NQ/RTY heatmap, DXY/VIX badges)
- [ ] Economic calendar integration
- [ ] Combined sentiment gauges (Reddit + News → "Market Mood")

### UI-B: Analysis Page
- [ ] Asset fingerprint display — wire `asset_fingerprint.py`
- [ ] Wave structure panel — wire `wave_analysis.py` + `swing_detector.py`

### UI-C: Plan Page
- [ ] Range builders status, "Backtest this level" button
- [ ] CNN confidence badge on entry zones

### UI-D: Journal Page
- [x] Standalone journal page created — `static/journal.html` (2026-03-14)
- [x] Route handler at `GET /journal` via `static_pages.py`
- [ ] TODO: Auto-populate from Rithmic fills (when creds arrive — see JOURNAL-SYNC)
- [ ] TODO: Plan adherence scoring, session stats panel

### UI-E: UX Polish
- [ ] Keyboard shortcuts, one-click copy prices, nav progress indicator
- [ ] Mobile-friendly Live page layout

---

## 🟡 Next Up — Wire Real Modules into Trading Pipeline

> Non-blocking improvements to replace simulated data with live module calls.

- [ ] Wire overnight step to `massive_client` real bars
- [ ] Wire regime step to `RegimeDetector` with cached 15m bars
- [ ] Wire ICT step to `ict_summary()` with cached 5m bars
- [ ] Wire volume profile step to `compute_volume_profile()` with cached bars
- [ ] Wire ORB step to `engine:orb:{symbol}` Redis cache
- [ ] Wire CNN step to live model inference
- [ ] Wire Grok step to live `run_morning_briefing()`
- [ ] Wire Kraken step to live `KrakenClient.get_ticker()`
- [ ] Replace simulated live stream with Rithmic tick data (when creds arrive)
- [ ] Persist journal trades to Postgres via existing journal API

---

## 🟡 Phase MODEL-INT — Model Library Integration & Lint Fixes

> `src/lib/model/` has a rich ML library (CNN, LSTM, TFT, Transformer, XGBoost, LightGBM,
> CatBoost, ARIMA, GARCH, HMM, Prophet, Bayesian) — mostly scaffolded, needs wiring.

### MODEL-INT-A: Fix Syntax Errors & Broken Imports *(blocking)*
- [ ] Fix import paths across 42 files in `src/lib/model/`
- [ ] Create shims for missing external dependencies (`_shims.py`)
- [ ] Fix runtime bugs (broken `np.mean`, `pl.training` → `pl.Trainer`, missing `super().__init__()`)

### MODEL-INT-B: Auto-Fix & Wiring
- [ ] Auto-fix whitespace, deprecated typing, import sorting via ruff
- [ ] Wire `__init__.py` exports & populate empty files
- [ ] Add basic tests for model imports, registry, base classes

---

## 🟡 Phase PINE-INT — Pine Script Generator Integration & Lint Fixes

### PINE-INT-A: Fix Import Paths *(blocking)*
- [ ] Fix `from pine.generate` → relative imports in `src/lib/integrations/pine/`

### PINE-INT-B: Fix `fks`/`ruby` Key Mismatch
- [ ] Resolve key mismatch in `generate.py`, `main.py`, and `params.yaml`

### PINE-INT-C: Auto-Fix & Tests
- [ ] Auto-fix ~333 ruff errors
- [ ] Fix Pine Script module issues (duplicates, ordering)
- [ ] Add basic tests

---

## 🟢 Phase CLEANUP-REMAINING — Codebase Audit Items

### CLEANUP-A: Consolidate Shared Utility Functions
- [ ] `_safe_float()` — 8 identical copies → `lib/core/utils.py`
- [ ] `_ema()` — 4 copies → `lib/core/math.py`
- [ ] `_atr()` — 4 copies → `lib/core/math.py`
- [ ] `_rsi()` — 2 copies
- [ ] `compute_atr()` — 3 copies (Wilder-smoothed)
- [ ] `_run_mtf_on_result()` — 2 copies

### CLEANUP-C: Split Oversized Files *(do incrementally)*
- [ ] `dashboard.py` (~6,500 lines) — split into sub-modules
- [ ] `breakout_cnn.py` — split inference vs training code
- [ ] `settings.py` — extract HTML to static file (see UI-SPLIT)

---

## 🟡 Phase LOGGING — Standardize on structlog + stdlib

> **Decision**: The project had 3 logging systems in conflict — stdlib `logging` (~60 files),
> loguru (~36 files), and structlog (2 entry points). Standardized on structlog + stdlib
> as the single pipeline. **All loguru consumers migrated (LOGGING-A+B done).** Remaining
> work is mechanical stdlib→key-value conversion (LOGGING-C) and dead file cleanup (LOGGING-D).
>
> **Guide**: [`docs/logging.md`](docs/logging.md) — the full standard, migration patterns, and priority list.

### ✅ LOGGING-A: Core infrastructure (DONE 2026-03-13)
- [x] `lib/core/logging_config.py` — `setup_logging()` + `get_logger()` already exist and work
- [x] `docs/logging.md` — full logging standard guide written
- [x] Entry points migrated: `web/main.py` (was `logging.basicConfig`), `trainer_server.py` (was `logging.basicConfig` + manual `structlog.configure`)
- [x] Framework migrated: `core/base.py`, `core/service.py`, `core/runner.py` — switched from legacy `lib.utils.setup_logging` to `lib.core.logging_config`
- [x] DB layer migrated: `core/db/__init__.py`, `core/db/base.py`, `core/db/postgres.py`, `core/db/repository.py` — switched from loguru to structlog, f-strings → key-value logging

### ✅ LOGGING-B: Migrate remaining loguru files (DONE 2026-03-13)
- [x] `src/lib/core/db/orm.py` — loguru `.bind()` → `get_logger(__name__)`, f-strings → key-value
- [x] `src/lib/core/db/redis_clients/` (6 files: `base.py`, `async_client.py`, `sync_client.py`, `service.py`, `queue.py`, `utils.py`) — removed `_log_prefix_*` patterns, f-strings → key-value
- [x] `src/lib/core/exceptions/` (6 files: `boundary.py`, `classes.py`, `general_error.py`, `loader.py`, `utils.py`, `validation.py`) — f-strings → key-value, `logger.log()` → `getattr()` dispatch
- [x] `src/lib/core/` top-level (7 files: `feature_detection.py`, `helpers.py`, `initialization.py`, `lifespan.py`, `registry.py`, `teardown.py`, `text.py`) — removed loguru fallback patterns, f-strings → key-value
- [x] `src/lib/indicators/` (17 files) — all loguru imports replaced, `.bind()` patterns removed, f-strings → key-value
- [x] `src/lib/model/_shims.py` — loguru fallback → `get_logger(__name__)` (re-exported to downstream consumers)
- [x] Verified: `src/lib/integrations/` already used stdlib `logging` (not loguru) — no migration needed
- [x] Only remaining loguru import: `src/lib/utils/logging_utils.py` itself (dead code — zero consumers)
- [x] **Result: 0 loguru consumers in production code** — all 39 files migrated to structlog pipeline

### LOGGING-C: Convert stdlib files to structured key-value logging (low priority — already works via bridge)
- [ ] `src/lib/services/data/api/*.py` (~20 files) — `logging.getLogger("api.X")` → `get_logger(__name__)` + convert f-string messages to key-value
- [ ] `src/lib/services/engine/*.py` (~12 files) — same pattern
- [ ] `src/lib/integrations/*.py` (~8 files) — stdlib users, already formatted via bridge
- [ ] `src/lib/analysis/**/*.py` — remaining stdlib users
- [ ] Note: these already get structlog formatting since `setup_logging()` wires the root handler — migration just enables structured key-value logging

### ✅ LOGGING-D: Deprecate legacy logging_utils.py
- [x] `lib/utils/logging_utils.py` — added `warnings.warn(DeprecationWarning)` at module level; fires on import pointing to `lib.core.logging_config.get_logger()`
- [x] `lib/utils/setup_logging.py` — same deprecation warning added
- [ ] Remove from `lib/core/base.py` `__all__` exports if present
- [ ] Eventually delete both files (safe — no code imports them; loguru dep can be removed at the same time)

**Files**: `src/lib/core/logging_config.py` (source of truth), `docs/logging.md` (guide), ~60 stdlib files remain for key-value conversion
**Estimated effort**: LOGGING-C: 2–3 sessions (mechanical, can be done incrementally), LOGGING-D: 10 min

---

## 🟢 After First Live Profits

1. **Phase REDDIT** — Reddit sentiment panel on dashboard
2. **Phase 9A** — correlation anomaly heatmap
3. **Phase 6** — Kraken spot portfolio management (personal account, full app control)
4. **Phase v9** — cross-attention fusion, Ruby/Reddit/News CNN features (only if >2% accuracy lift)
5. **Phase COMPLIANCE-AUDIT** — one-page compliance log PDF exporter for prop-firm audits
6. **BTC cold storage tracking** — public key balance display on dashboard
7. **Phase KRAKEN-ACCOUNTS** — Full Kraken spot + futures management (5K CAD each target)
8. **Phase MULTI-EXCHANGE** — crypto.com, Netcoins, BTC hardware wallet, unified net worth
9. **Phase TAX** — Canadian capital gains tracking + export (included in MULTI-EXCHANGE-E)

Full specs for all of the above: [`docs/backlog.md`](docs/backlog.md)

---

## Pre-Retrain Readiness — Summary

> v8 training ran but did NOT beat v6 champion (83.3% vs 87.1%).
> Regularization upgraded, per-asset training infra built. Ready for v9 retrain.
>
> **2026-03-14 oryx session findings:**
> - Dataset path issue FIXED — all 28,548 images were on disk, just had wrong prefix
> - CUDA training VERIFIED — 2-epoch test on RTX 3080, model loads/trains/saves correctly
> - torchvision was missing from venv — installed (0.25.0)
> - 6 retrain scripts created for automated pipeline execution

### ✅ Confirmed working
- `feature_contract.json` v8: 37 features, embeddings (4+8=12), gate checks
- `HybridBreakoutCNN` v8: wider tabular head (37→256→128→64, GELU+BN)
- `_normalise_tabular_for_inference()`: v5→v4→v6→v7→v7.1→v8 backward-compat padding
- `_build_row()`: all 37 features computed with real data
- `train_model()`: grad accumulation, mixup, label smoothing 0.15, cosine warmup, separate LR groups
- Per-asset/per-group training: `TrainRequest.train_mode`, `ASSET_GROUPS`, `_filter_csv_by_symbols()`
- Per-asset model loading: `_resolve_model_name()` → per-asset → per-group → combined fallback
- Multi-model cache: `_model_cache` dict keyed by path (supports concurrent per-asset + combined)
- Peer bar loading: `_resolve_peer_tickers()` → `bars_by_ticker` dict
- Test suite: 2834+ passed (93 new tests added 2026-03-14)
- Dataset paths: all 28,548 rows → images verified on disk ✅ (2026-03-14)
- CUDA training: 2-epoch test passed on RTX 3080 (2026-03-14)
- Model: 20,991,086 parameters, EfficientNetV2-S backbone + tabular head

### v8 Training Results (2026-03-13)
| Metric | v6 Champion | v8 Result | Delta |
|--------|-------------|-----------|-------|
| Val Accuracy | 87.1% | 83.3% | -3.8% |
| Val Precision | 87.15% | 83.4% | -3.75% |
| Val Recall | 87.27% | 83.3% | -3.97% |
| Train Accuracy | ~87% | 99.7% | +12.7% (overfitting!) |
| Train/Val Gap | ~0% | 16.4% | ← root problem |
| Best Epoch | 25 | 56/60 | — |
| Dataset Size | — | 30,375 used / 54,634 available | Path prefix issue — **all 28,548 are valid** |

### Dataset Analysis (2026-03-14 oryx validation)
| Category | Value | Note |
|----------|-------|------|
| Total rows | 28,548 | All images verified |
| Good/Bad split | 52.9% / 47.1% | Well balanced |
| ORB dominance | 95.8% | **Key issue — need 365-day regen** |
| BollingerSqueeze | 2.1% (606) | Needs more data |
| Fibonacci | 1.3% (365) | Needs more data |
| Consolidation | 0.5% (155) | Needs more data |
| Focus symbols (9) | 8.7–11.0% each | Good coverage |
| Forex symbols | 0.1–0.8% each | Consider excluding |
| Sessions | All 9 represented | US=17.2%, balanced |

---

## `todo/` Directory — Consolidated & Deleted

> All 13 files from the former `todo/` directory have been reviewed and extracted into phases above.
> Full audit trail: [`docs/todo_extracted_tasks.md`](docs/todo_extracted_tasks.md).

---

## 📊 Priority Matrix — Session Planning Guide

> Use this to pick tasks for each AI agent session. Top-to-bottom priority.
> Updated 2026-03-14 after oryx session: dataset fixed, CUDA verified, scaffolding done, 93 new tests.

| Priority | Phase | Est. Sessions | Depends On | Status |
|----------|-------|---------------|------------|--------|
| 🔴 1 | **RETRAIN v9** | 1–2 | — | **NEXT: Run `scripts/run_full_retrain.py` on oryx** — dataset fixed, CUDA verified, scripts ready |
| 🔴 2 | RETRAIN per-group | 1 | RETRAIN combined | Run `scripts/run_per_group_training.py` — compare groups vs combined |
| 🟡 3 | **KRAKEN-SIM** (C–D remain) | 2–3 | — | ✅ SimulationEngine + API + DOM live data built — pre-trade analysis + settings toggle remain |
| 🟡 4 | **DATA-ROLLING** (D remains) | 1 | — | ✅ Sync service + retention + Redis cache built — trainer pipeline wiring remains |
| 🔴 5 | JOURNAL-SYNC | 3–4 | Rithmic creds | Auto-sync trades from Rithmic fills |
| 🟡 6 | RITHMIC-STREAM (A–F) | 6–8 | Rithmic creds | Persistent streaming integration |
| 🟡 7 | DOM live data | 2–3 | RITHMIC-STREAM-B | Replace mock data in `dom.py` with real L2 |
| 🟡 8 | POSINT wiring | 2–3 | RITHMIC-STREAM | Wire real analysis modules into position_intelligence.py |
| 🟡 9 | RA-CHAT verify | 0.5 | — | Test chat.html end-to-end with RA/Grok backend |
| 🟡 10 | PINE-WEBUI | 1 | — | Quick verify + polish |
| 🟡 11 | UI-SPLIT | 2–3 | — | Non-blocking, improves DX |
| 🟡 12 | TESTS remaining | 1–2 | — | WebUI endpoints, pipeline edge cases |
| 🟡 13 | SIGNALS | 1 | — | Config + gating changes |
| 🟡 14 | Pipeline wiring | 2–3 | — | Non-blocking enhancements |
| 🟡 15 | **WEBUI-KEYS** | 2–3 | — | Move API keys from .env to WebUI settings page |
| 🟡 16 | **KRAKEN-ACCOUNTS** | 4–5 | KRAKEN-SIM | Full Kraken spot + futures account management |
| 🟢 17 | LOGGING-C+D | 2–3 | LOGGING-B ✅ | stdlib→key-value (low priority) |
| 🟢 18 | MODEL-INT / PINE-INT | 3–4 | — | Library polish, not urgent |
| 🟢 19 | CLEANUP-REMAINING | 2 | — | Dedup + file splits |
| 🟢 20 | PROFIT tracking | 1–2 | Funded accounts | After first profits |
| 🟢 21 | Kraken spot ratios | 2–3 | Funded + Kraken deposit | Manage spot portfolio with ratio strategy |
| 🟢 22 | **MULTI-EXCHANGE** | 8–10 | KRAKEN-ACCOUNTS | crypto.com, Netcoins, BTC wallet, tax reporting |

## 📋 New Files Created (2026-03-14 Oryx Session)

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib/services/engine/position_intelligence.py` | 495 | Position Intelligence Engine (POSINT-A) — TODO stubs |
| `src/lib/services/engine/rithmic_position_engine.py` | 384 | Rithmic Position Engine wrapper (POSINT-B) — TODO stubs |
| `src/lib/services/data/api/dom.py` | 283 | DOM API routes: snapshot, SSE, config (mock data) |
| `src/lib/services/data/api/static_pages.py` | 122 | Route handlers for /chat, /dom, /journal |
| `static/chat.html` | 1048 | RustAssistant chat interface |
| `static/dom.html` | 324 | Depth of Market ladder (mock data) |
| `static/journal.html` | 613 | Trade journal page |
| `scripts/validate_dataset.py` | 490 | Dataset health report + validation |
| `scripts/fix_dataset_paths.py` | 315 | Strip Docker `/app/` prefix from CSV paths |
| `scripts/test_training_local.py` | 628 | Local CUDA training test (real data, 3 epochs) |
| `scripts/run_full_retrain.py` | 1038 | Full v9 retrain pipeline orchestrator |
| `scripts/run_per_group_training.py` | 731 | Per-group training comparison orchestrator |
| `src/tests/test_dataset_validation.py` | 303 | 13 tests for validate_dataset() |
| `src/tests/test_trainer_endpoints.py` | 503 | 31 tests for trainer server HTTP endpoints |
| `src/tests/test_rithmic_account.py` | 537 | 49 tests for Rithmic account config + encryption |

## 📋 New Files Created (2026-03-15 Session)

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib/services/engine/simulation.py` | 1273 | SimulationEngine — paper trading with live tick data (KRAKEN-SIM-B) |
| `src/lib/services/data/api/simulation_api.py` | 370 | Simulation API routes: /api/sim/*, /sse/sim (KRAKEN-SIM-B) |
| `src/lib/services/data/sync.py` | 783 | DataSyncService — rolling 1-year data window, background sync, retention (DATA-ROLLING-A/B/C) |

---

## 📊 Lint Status Dashboard

> ✅ **ACHIEVED: 0 ruff errors across all of `src/`** (as of 2026-03-11)

| Directory | Status |
|---|---|
| `src/lib/model/` | **0** ✅ |
| `src/lib/integrations/pine/` | **0** ✅ |
| `src/lib/core/` | **0** ✅ |
| `src/lib/indicators/` | **0** ✅ |
| `src/lib/utils/` | **0** ✅ |
| `src/lib/services/` | **0** ✅ |
| `src/tests/` | **0** ✅ |
| **Total** | **0** ✅ |