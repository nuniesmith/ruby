# Ruby Futures — Completed Work

> Archive of all resolved items, historical architecture notes, and pre-refactor decisions.
> These are done — they live here so `todo.md` stays scannable.

---

## Architecture Issues Resolved (Pre-Refactor)

### Triple duplication of breakout types & config ✅
- `lib/core/breakout_types.py` — canonical `BreakoutType` (IntEnum) + `RangeConfig` (CNN/training source)
- `lib/services/engine/breakout.py` — **second** `BreakoutType` (StrEnum) + **second** `RangeConfig` (engine runtime)
- `lib/services/engine/orb.py` — **third** dataclass `ORBSession` with its own ATR params

Mapping dicts (`_ENGINE_TO_TRAINING`, `_TRAINING_TO_ENGINE`) existed purely to convert between the two enums.
**Resolution:** Eliminated the engine `StrEnum` and the runtime `RangeConfig`. All callers import from
`lib.core.breakout_types`. Mapping dicts removed.

### `orb.py` isolated silo (1800+ lines) ✅
Had its own `ORBResult`, `detect_opening_range_breakout()`, `compute_atr()`. `breakout.py` was built to
generalise ORB but lived alongside it with parallel code paths. `main.py` had **10 separate
`_handle_check_orb_*` functions** that all delegated to the same `_handle_check_orb`.
**Resolution:** Unified detector in `strategies/rb/detector.py`. All handlers are now one-liners.

### `main.py` god module (3285 lines) ✅
`_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` were 90% copy-paste.
**Resolution:** Single `handle_breakout_check()` in `handlers.py`. Shared helpers extracted.

### `analysis/orb_filters.py` misnamed ✅
The filters (NR7, premarket range, session window, lunch filter, MTF bias, VWAP confluence) are
NOT ORB-specific. **Resolution:** Renamed to `breakout_filters.py`. `ORBFilterResult` → `BreakoutFilterResult`.

### Risk system not real-time position-aware ✅
**Resolution:** `LiveRiskState` wired, dual micro/regular sizing on cards, live position overlays,
risk strip all shipped.

---

## Phase 1 — Pre-Retrain RB System Refactor ✅

### Phase 1A: Merge BreakoutType Enums → Single Source of Truth ✅
- [x] Eliminated engine `StrEnum` in `services/engine/breakout.py` — use `core/breakout_types.BreakoutType` (IntEnum) everywhere
- [x] Removed `_ENGINE_TO_TRAINING` / `_TRAINING_TO_ENGINE` mapping dicts
- [x] Removed `to_training_type()` / `from_training_type()` / `breakout_type_ordinal()` bridge functions
- [x] All engine callers import from `lib.core.breakout_types`
- [x] `BreakoutResult.to_dict()` uses `.name` for JSON serialisation and `.value` for ordinals
- [x] Short-name aliases (`"PDR"` → `PrevDay`, etc.) retained in `breakout.py` for backward compat

### Phase 1B: Merge `RangeConfig` → Single Dataclass ✅
- [x] Unified the two `RangeConfig` dataclasses into `core/breakout_types.py`
- [x] Detection-threshold fields (ATR mult, body ratio, range caps, squeeze params) merged INTO core `RangeConfig`
- [x] All 13 `_*_CONFIG` registry entries have detection fields
- [x] Engine-side `RangeConfig` eliminated — `get_range_config(BreakoutType.ORB)` returns everything
- [x] `DEFAULT_CONFIGS` in `breakout.py` delegates to `get_range_config()`

### Phase 1C: Merge ORB Detection into Unified RB Detector ✅
- [x] `detect_range_breakout(config=ORB_CONFIG)` handles all 13 types including ORB
- [x] All `_build_*_range()` functions extracted into `strategies/rb/range_builders.py`
- [x] Single `detect_range_breakout(bars, symbol, config)` in `strategies/rb/detector.py`
- [x] `BreakoutResult` covers all types (ORB fields mapped: `range_high`↔`or_high`, etc.)
- [x] Single `compute_atr()` in `strategies/rb/range_builders.py` (canonical implementation)
- [x] `_handle_check_orb()` (~800 lines) replaced by `handle_orb_check()` delegation
- [x] Quality filters pipeline extracted to `handlers.run_quality_filters()`
- [x] CNN inference pipeline extracted to `handlers.run_cnn_inference()`
- [x] CNN tabular feature construction extracted to `handlers.build_cnn_tabular_features()`
- [x] Session-aware filter windows extracted to `handlers.get_filter_windows_for_session()`
- [x] All 11 ORB session handlers (`_handle_check_orb_london`, etc.) are now one-liners
- [x] ORB-specific Redis publishing handled via `_publish_orb_result()` shim for backward compat

### Phase 1D: Extract Generic Handler Pipeline from `main.py` ✅
- [x] One handler function for all 13 breakout types — `handle_breakout_check()` in `handlers.py`
- [x] `_handle_check_pdr`, `_handle_check_ib`, `_handle_check_consolidation` are one-liners
- [x] Shared helpers extracted: `fetch_bars_1m`, `get_htf_bars`, `run_mtf_on_result`, `persist_breakout_result`, `publish_breakout_result`, `send_breakout_alert`
- [x] `handle_breakout_multi()` runs multiple types in parallel via ThreadPoolExecutor
- [x] `enable_filters=True` / `enable_cnn=True` flags bring filter+CNN support to any type

### Phase 1E: Rename `orb_filters.py` → `breakout_filters.py` ✅
- [x] `ORBFilterResult` → `BreakoutFilterResult`. Backward-compat shim in place.

### Phase 1F: Rename `rb_simulator.py` → `rb_simulator.py` ✅
- [x] `simulate_orb_outcome` → `simulate_rb_outcome`. Shim in place.

### Phase 1G: Create `lib/strategies/` Package ✅
- [x] Clean separation of strategy code from infrastructure
  - `lib/strategies/rb/` — Range Breakout scalping system (detector, range_builders, publisher)
  - `lib/trading/costs.py` → `lib/strategies/costs.py` (shim in place)
  - `lib/trading/strategies.py` → `lib/strategies/strategy_defs.py` (shim in place)
  - `lib/trading/engine.py` → `lib/strategies/backtesting.py` (shim in place)
  - `RBSession` alias added in `multi_session.py`, exported from `lib.core`

### Phase 1H: Pre-Retrain Test Cleanup ✅
- [x] All unit tests passing (2552 passed, 0 failed, 1 skipped)
- [x] Fixed all `BreakoutType.UPPER_CASE` → `BreakoutType.PascalCase` enum name mismatches in `test_breakout_types.py` (76 occurrences)
- [x] Updated feature contract test expectations from v6 (18 features) to v7.1 (28 features)
- [x] Fixed `lib/trading/strategies.py` backward-compat shim — added missing re-exports: `_atr`, `_ema`, `_ict_confluence_array`, `_compute_ict_confluence`
- [x] Fixed `positions.py` `clear_positions()` cache isolation bug — `REDIS_AVAILABLE` was a stale name binding from import time; now reads from `lib.core.cache` module at call time

---

## Phase 2 — Daily Strategy Layer ✅

### Phase 2A: Daily Bias Analyzer ✅
- [x] `BiasDirection`, `CandlePattern`, `KeyLevels`, `DailyBias` dataclasses
- [x] `compute_daily_bias()` — 6-component weighted scoring (candle 25%, weekly 20%, monthly 25%, volume 10%, gap 10%, ATR 10%)
- [x] `compute_all_daily_biases()`, `rank_assets_by_conviction()`, CNN feature helpers

### Phase 2B: Daily Plan Generator + Focus Asset Selection ✅
- [x] `DailyPlan`, `SwingCandidate`, `ScalpFocusAsset` dataclasses with full serialisation
- [x] `generate_daily_plan()` orchestrator, `select_daily_focus_assets()` 5-factor composite ranking
- [x] `DailyPlan.publish_to_redis()` / `load_from_redis()` — 18h TTL
- [x] `get_daily_plan_focus_assets()` in `focus.py`, `compute_daily_focus(use_daily_plan=True)`
- [x] 66 tests passing

### Phase 2C: Swing Detector ✅
- [x] Three entry detectors: `detect_pullback_entry()`, `detect_breakout_entry()`, `detect_gap_continuation()`
- [x] Exit engine `evaluate_swing_exits()`: stop loss → TP1 scale 50% → TP2 close → EMA-21 trail → time stop
- [x] State machine: WATCHING → ENTRY_READY → ACTIVE → TP1_HIT → TRAILING → CLOSED
- [x] Redis publish/load, `engine:swing_signals`, `engine:swing_states`
- [x] 150 tests passing

---

## Phase 3 — Dashboard Focus Mode ✅

### Phase 3A: Top-4 Asset Selection ✅
### Phase 3B: Dashboard Focus Mode ✅
- [x] `_render_focus_mode_grid()` — tiered layout: scalp focus (prominent), swing cards (amber), background collapse
- [x] `_render_daily_plan_header()` — Grok morning brief card + focus chip strip
- [x] `_render_why_these_assets()` — collapsible score breakdown table with mini score bars
- [x] `_render_swing_card()` — amber-bordered with TP1/TP2/TP3, entry style chips, confidence badge
- [x] SSE `daily-plan-update` listener, `GET /api/daily-plan/html` endpoint
- [x] 82 tests passing

### Phase 3C: Grok Integration for Daily Selection ✅
- [x] `grok_helper.py` — parsed JSON → `DailyPlan.market_context`, dashboard rendering
- [x] 77 tests passing

### Phase 3D: Swing Action Buttons ✅
- [x] `swing_actions.py` router — 10 endpoints: accept, ignore, close, stop-to-BE, update-stop, status
- [x] HTMX fragments (success/error toasts + updated buttons) — no full page reload
- [x] Signal lifecycle: detect → pending → accept/ignore → active → manage → close → archive
- [x] SSE `swing-update` listener, structured action publishing via `_publish_swing_action()`
- [x] 88 tests passing

---

## Phase 4 — CNN v8 Feature Code ✅

### Phase 4A: New Features from Daily Strategy Layer ✅
- [x] 6 new v7 features (features [18]–[23]): `daily_bias_direction`, `daily_bias_confidence`, `prior_day_pattern`, `weekly_range_position`, `monthly_trend_score`, `crypto_momentum_score`
- [x] `feature_contract.json` updated to v7.1 (28 features)
- [x] `dataset_generator.py` `_build_row()` computes all 6 features with neutral fallbacks
- [x] `_normalise_tabular_for_inference()` handles v6→v7→v7.1 backward-compat padding

### Phase 4B: Sub-Features and Richer Encoding ✅
- [x] 4 sub-features (features [24]–[27]): `breakout_type_category`, `session_overlap_flag`, `atr_trend`, `volume_trend`
- [x] All 4 computed in `dataset_generator.py` `_build_row()` with neutral fallbacks

### Phase v8-A: Hierarchical Asset Embedding ✅
- [x] `nn.Embedding(num_classes=5, embedding_dim=4)` + `nn.Embedding(num_assets=25, embedding_dim=8)` added to `HybridBreakoutCNN`
- [x] `feature_contract.json` — `asset_class_lookup` and `asset_id_lookup` tables added
- [x] `_build_row()` and `BreakoutDataset.__getitem__()` pass `asset_class_idx` and `asset_idx` as integer IDs
- [x] `_normalise_tabular_for_inference()` routes embedding IDs separately from float tabular vector
- [x] Backward compat: if checkpoint lacks embedding weights, falls back to flat v7.1 mode

### Phase v8-B: Cross-Asset Correlation Features (+3 tabular features) ✅
- [x] `lib/analysis/cross_asset.py` — `rolling_peer_correlation()`, `cross_class_correlation()`, `correlation_regime()`
- [x] `Asset.peers` field in `asset_registry.py` — Gold→[Silver, Copper], MNQ→[MES, M2K], etc.
- [x] 3 new features in `feature_contract.json` v8: `primary_peer_corr` [28], `cross_class_corr` [29], `correlation_regime` [30]
- [x] `generate_dataset()` pre-loads peer bars via `_resolve_peer_tickers()`, builds `bars_by_ticker` dict — real correlations computed, not neutral fallbacks

### Phase v8-C: Asset Fingerprint Features (+6 tabular features) ✅
- [x] `lib/analysis/asset_fingerprint.py` — `typical_daily_range_norm`, `session_concentration`, `breakout_follow_through`, `hurst_exponent`, `overnight_gap_tendency`, `volume_profile_shape`
- [x] 6 new features in `feature_contract.json` v8: features [31]–[36]
- [x] `_build_row()` computes from daily bars + 1m bars with neutral fallbacks

### Phase v8-D: Architecture Upgrades to `HybridBreakoutCNN` ✅
- [x] Wider tabular head: Linear(N→256) → BN → GELU → Dropout(0.3) → Linear(256→128) → BN → GELU → Linear(128→64)
- [x] Mixup augmentation: α=0.2 on tabular features during training
- [x] Label smoothing: 0.05 → 0.10
- [x] Cosine warmup: 5-epoch linear warmup before cosine decay
- [x] Gradient accumulation: effective batch size 128 (2× accumulation with batch_size=64)

### Phase v8-E: Training Recipe & Hyperparameters ✅ (code — not yet run)
- [x] `epochs=80`, `patience=15`
- [x] `freeze_epochs=5`
- [x] `batch_size=64`, grad accumulation → effective 128
- [x] `lr=2e-4` (backbone), `lr=1e-3` (tabular head + embeddings) — separate param groups
- [x] `weight_decay=1e-4`
- [x] `split_dataset()` stratifies by `(label, breakout_type, session)` triple
- [x] Gate check documented in `feature_contract.json` `v8_training_recipe.gate_check`: ≥89% acc, ≥87% prec, ≥84% rec
- [x] `DatasetConfig` defaults: `breakout_type="all"`, `orb_session="all"`, `max_samples_per_type_label=800`, `max_samples_per_session_label=400`

---

## Phase 5 — Live Risk-Aware Position Sizing ✅

### Phase 5A: Generalized Asset Model ✅
- [x] `Asset`, `ContractVariant`, `AssetClass`, `ASSET_REGISTRY` in `src/lib/core/asset_registry.py`
- [x] `dual_sizing()`, `compute_position_size()`, `get_asset_by_ticker()`, `get_asset_group()`
- [x] Replaces split `MICRO_CONTRACT_SPECS` / `FULL_CONTRACT_SPECS`

### Phase 5B: Real-Time Risk Budget Integration ✅
- [x] `LiveRiskState`, `LiveRiskPublisher`, `compute_live_risk()` — `src/lib/services/engine/live_risk.py`
- [x] API endpoints: `/api/live-risk`, `/api/live-risk/html`, `/api/live-risk/summary`
- [x] Force-publish on position changes (1–2s latency)

### Phase 5C: Dynamic Position Sizing on Focus Cards ✅
- [x] `_compute_dual_sizing()` — micro + regular side-by-side on focus cards

### Phase 5D: Live Position Overlay on Focus Cards ✅
- [x] `_render_live_position_overlay()` — LIVE badge, P&L, R-multiple, bracket progress bar

### Phase 5E: Risk Dashboard Strip ✅
- [x] Risk strip (`get_live_risk_html()`) — health-coloured, HTMX polling + SSE

---

## Immediate Fixes — Post-Cleanup ✅

### Python Model & Training Fixes ✅
- [x] **Run full test suite** — `pytest src/tests/` → 2552 passed, 1 skipped. Fixed `test_bridge_trading.py` heartbeat cache key (`bridge_heartbeat` → `broker_heartbeat`) and `test_kraken_training_pipeline.py` feature count (28 → 37 for v8). Also fixed `positions.py` `_get_broker_url()` to derive localhost from heartbeat `listenerPort` when no explicit broker host is configured.
- [x] **`test_bridge_trading.py`** — updated 4 cache key references from `bridge_heartbeat` to `broker_heartbeat`. All 37 bridge trading tests pass.
- [x] **`_build_row()` peer bars** — `generate_dataset()` pre-loads peer bars via `_resolve_peer_tickers()`, builds `bars_by_ticker` dict per symbol, attaches `_daily_bars`, `_bars_1m`, `_bars_by_ticker` to each `ORBSimResult`.
- [x] **Smoke-test training loop** — `tests/test_v8_smoke.py` added (31 tests, all passing): architecture, dataset loading, full 2-epoch train, `evaluate_model`, `predict_breakout`, `predict_breakout_batch`, grad accumulation, mixup, separate LR groups, cosine warmup, label smoothing. Fixed 119 stale `@patch("lib.strategies.…")` paths and 44 `lib.trading.strategies.rb.orb` import paths. Full suite: **2543 passed, 0 failed**.

---

## Infrastructure Milestones ✅

### Data Service Split (`:data` + `:engine` separation) ✅
- [x] `docker/data/Dockerfile` — standalone image: `python:3.13-slim`, copies `src/`, runs via `entrypoints/data/main.py`
- [x] `docker/data/entrypoint.sh` — `exec python -m entrypoints.data.main`
- [x] `docker/engine/entrypoint.sh` — stripped to `exec python -m lib.services.engine.main` (uvicorn removed)
- [x] `docker/engine/Dockerfile` — removed `EXPOSE 8000`, removed HTTP healthcheck, uses `test -f /tmp/engine_health.json`
- [x] `lib.services.data.main` — added `main()` function; `LOG_LEVEL` env var wired
- [x] `docker-compose.yml` — `data` service (port 8050), `engine` (no ports, depends on data), `web`/`trainer`/`prometheus` all point at `http://data:8000`; `ENABLE_KRAKEN_CRYPTO=1` on data, `=0` on engine
- [x] CI/CD — `data` added to docker matrix (amd64+arm64, `is_default: true` → `:latest`); engine `is_default: false`; deploy pulls + starts `data engine web prometheus grafana`

### NinjaTrader Bridge Removed ✅
- [x] All NT8 Bridge code, deploy scripts, and C# patchers removed from Python codebase
- [x] Position management is now broker-agnostic (`positions.py`)
- [x] `src/ninja/` and `src/pine/` directories contain C#/Pine source for reference only — not part of Python runtime
- [x] `broker_heartbeat` Redis key replaces `bridge_heartbeat` throughout

### CNN v7.1 Feature Contract ✅
- [x] Features [18]–[23]: daily strategy features in `breakout_cnn.py` + `dataset_generator.py`
- [x] Features [24]–[27]: sub-features (breakout type category, session overlap, ATR trend, volume trend)
- [x] `feature_contract.json` updated to v7.1 (28 features)
- [x] `_normalise_tabular_for_inference()` v6→v7→v7.1 backward-compat padding

### CNN Model — v6 Champion ✅
- [x] 22-symbol training, 13 types, 9 sessions, 25 epochs
- [x] **87.1% accuracy**, 87.15% precision, 87.27% recall — all gates passed
- [x] `breakout_cnn_best.pt` promoted, `feature_contract.json` v6 generated

### Unified Data Resolver ✅
- [x] `DataResolver` — Redis → Postgres → Massive/Kraken API three-tier resolution
- [x] `resolve()`, `resolve_batch()`, `resolve_with_meta()`

### Kraken Training Pipeline Integration ✅
- [x] `dataset_generator.py` — Kraken routing, `_is_kraken_symbol()`, `_load_bars_from_kraken()`
- [x] 25 total training symbols: 22 CME micros + BTC, ETH, SOL

### Web UI — Trading / Review Mode ✅
- [x] `⚡ Trading` / `🔍 Review` pill toggle — auto-detects from ET hour
- [x] CSS visibility gates: review panels hidden in trading mode
- [x] Decimal precision fix for forex tickers (5–7dp)

### Trainer UI Separation ✅
- [x] `trainer_server.py` HTML endpoint removed — pure API server
- [x] `src/lib/services/data/api/trainer.py` — full dashboard page at `GET /trainer`

### Web UI — Settings Page ✅
- [x] `settings.py` — 5 tabbed sections: Engine, Services, Features, Risk & Trading, API Keys
- [x] All settings persisted to Redis via `settings:overrides`

### SSE Swing + TV Alert Wiring ✅
- [x] `swing-update` SSE listener — auto-refreshes focus grid, parses action metadata for market events feed
- [x] `tv-alert` SSE listener — shows TradingView webhook alerts in market events feed
- [x] `dashboard:tv_alert` Redis PubSub channel handler in `sse.py`

### Global Session Hours — DST-Aware ✅
- [x] `multi_session.py` — `exchange_hours_in_et()` converts all exchange sessions from UTC using `ZoneInfo("America/New_York")` — handles EDT/EST transitions and EU/Australian DST differences automatically
- [x] Dashboard replaced hardcoded ET hour floats with `exchange_hours_in_et()` output — no more brittle seasonal offsets
- [x] Session data injected from Python into client JS so badge and cursor logic uses the same authoritative ET values

### EOD Safety System ✅
- [x] `RithmicAccountManager.eod_close_all_positions(dry_run=False)` — per-account: cancel all orders → 0.5s pause → exit_position at market (MANUAL tag)
- [x] `POST /api/rithmic/eod-close` — manual trigger endpoint with optional `{ "dry_run": true }` body
- [x] `DashboardEngine._eod_warning()` — fires at 15:45 ET once per calendar day; logs WARNING + dispatches `send_risk_alert()` to all configured alert channels
- [x] `DashboardEngine._eod_close_positions()` — fires at 16:00 ET once per calendar day (catch-up window to 16:14 for restarts); runs async EOD close in fresh event loop on engine thread; dispatches per-account summary alert
- [x] Both actions are safe no-ops when `async-rithmic` is not installed or no accounts are configured