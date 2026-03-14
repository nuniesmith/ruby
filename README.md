# Ruby Futures

> Live dashboard, market stats & web UI for futures trading — real-time range breakout detection,
> session-aware scheduling, and a full HTMX dashboard powered by FastAPI.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Local Development](#local-development)
- [CNN Model Training](#cnn-model-training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Scripts & Tools](#scripts--tools)
- [Technologies](#technologies)
- [License](#license)

---

## Architecture

Everything lives in this single repo. There is no external training repo — models are trained
by the built-in trainer service and stored in `models/`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Ruby Futures                                 │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Postgres   │  │    Redis     │  │ Data Service │  │   Engine   │  │
│  │  (journal,   │  │  (hot cache, │  │  (FastAPI +  │  │ (scheduler,│  │
│  │   history,   │  │   live bars, │  │   REST API,  │  │  analysis, │  │
│  │   risk)      │  │   positions) │  │   SSE)       │  │  ORB, risk │  │
│  │              │  │              │  │              │  │  scoring)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         └─────────────────┴─────────────────┴────────────────┘         │
│                                    │                                    │
│  ┌─────────────────────────────────┴─────────────────────────────────┐  │
│  │                    Analysis Pipeline                               │  │
│  │                                                                   │  │
│  │  Wave Analysis · Volatility Clustering · Regime Detection (HMM)   │  │
│  │  ICT/SMC (FVGs, OBs, Sweeps) · Volume Profile · CVD              │  │
│  │  Multi-TF Confluence · Signal Quality · Pre-Market Scoring        │  │
│  │  Range Breakout Detection · 6 Deterministic Filters · CNN Inference│  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │  Trainer Service     │  │  Monitoring (optional profile)          │  │
│  │  GPU CNN training    │  │  Prometheus · Grafana                   │  │
│  │  port 8501           │  │                                         │  │
│  └──────────────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Docker services** (`nuniesmith/futures`):

| Service | Image Tag | Role | Port |
|---|---|---|---|
| **data** | `:data` | FastAPI REST + SSE API, bar cache, Kraken feed, Rithmic account manager | 8050 → 8000 |
| **engine** | `:engine` | Scheduler, risk manager, breakout detection, CNN inference, Grok briefs | *(no public port)* |
| **web** | `:web` | HTMX dashboard frontend (reverse-proxies to data) | 8080 |
| **trainer** | `:trainer` | GPU CNN training server — triggered via the 🧠 Trainer UI *(training profile)* | 8501 |
| **charting** | `:charting` | ApexCharts + nginx charting service | 8003 |
| **Postgres** | — | Durable storage — trade journal, historical bars, risk events | 5433 |
| **Redis** | — | Hot cache — live bars, analysis metrics, positions, focus, SSE pub/sub | 6380 |
| **Prometheus** | — | Metrics collection *(monitoring profile)* | 9095 |
| **Grafana** | — | Dashboards & visualization *(monitoring profile)* | 3010 |

**Data hierarchy** (highest → lowest priority):

```
Rithmic (async_rithmic)      ← primary for CME futures when creds are active: live ticks, time bars, order book
MassiveAPI (massive_client)  ← current primary for CME futures (REST + WebSocket, futures beta)
Yahoo Finance (yfinance)     ← last-resort fallback (delayed data)
Kraken REST / WebSocket      ← crypto spot only via kraken_client.py (personal account)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 1. Clone

```bash
git clone https://github.com/nuniesmith/futures.git
cd futures
```

### 2. One-Command Start

```bash
./run.sh
```

This will:
1. Create a Python virtualenv and install all dependencies
2. Generate a `.env` file with secure random secrets
3. Run the test suite and linter
4. Build and start all Docker services

### 3. Add Your API Keys

Edit `.env` and set:

```
MASSIVE_API_KEY=your_key_here    # https://massive.com/dashboard  (real-time CME futures data)
XAI_API_KEY=your_key_here        # https://console.x.ai           (Grok AI analyst)

# Rithmic (prop-firm order execution + market data)
RITHMIC_USERNAME=your_username
RITHMIC_PASSWORD=your_password
RITHMIC_SYSTEM_NAME=your_system
RITHMIC_GATEWAY=your_gateway
```

Without `MASSIVE_API_KEY` the system falls back to yfinance (delayed data).
Without `XAI_API_KEY` the Grok AI analyst tab is disabled — everything else works normally.
Without `RITHMIC_*` credentials the engine runs in signal-only mode (no order execution).

### 4. Verify

```bash
docker compose ps                 # all services should be "healthy"
docker compose logs -f engine     # watch the engine schedule actions
```

Open the dashboard at **http://localhost:8080**.

---

## Docker Deployment

### Standard (engine + web + postgres + redis)

```bash
docker compose up -d --build
```

### With CNN Trainer (+ GPU training server)

```bash
docker compose --profile training up -d --build
```

Requires an NVIDIA GPU with `nvidia-container-toolkit` installed on the host.
Once running, open the **🧠 Trainer** tab in the dashboard to configure and launch training runs.

### With Monitoring (+ Prometheus & Grafana)

```bash
docker compose --profile monitoring up -d --build
```

### Combine Profiles

```bash
docker compose --profile training --profile monitoring up -d --build
```

### Useful Commands

```bash
docker compose logs -f engine           # follow engine logs
docker compose logs -f data             # follow data API logs
docker compose logs -f web              # follow web frontend logs
docker compose logs -f trainer          # follow trainer logs
docker compose exec engine bash         # shell into engine container
docker compose restart engine           # restart engine (picks up new model)
docker compose down                     # stop all services
docker compose down -v                  # stop + remove volumes (⚠️ deletes all data)
```

---

## Local Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the Web Frontend Locally

```bash
./run.sh --local
```

This starts only the web service (port 8080) and expects the engine + Redis + Postgres
to already be running (e.g. via Docker Compose).

### Run Tests

```bash
pytest tests/ -x -q --tb=short         # full suite (2,700+ tests)
ruff check src/                         # linting
./run.sh --test                         # tests + lint together
```

---

## CNN Model Training

Training runs entirely within this repo — no external training service is needed.

### Via the Dashboard UI (recommended)

1. Start the trainer service: `docker compose --profile training up -d trainer`
2. Open the dashboard → **🧠 Trainer** tab
3. Configure epochs, batch size, learning rate, symbols, validation gates
4. Click **▶ Start Training** — live logs stream into the UI
5. On successful promotion the champion model is written to `models/` automatically
   and the engine hot-reloads it within one refresh cycle

### Via CLI (one-shot)

```bash
docker compose --profile training run --rm trainer \
    python -m lib.services.training.trainer_server
```

### Model Files

| File | Description |
|---|---|
| `models/breakout_cnn_best.pt` | PyTorch checkpoint — used by the Python engine for live inference |
| `models/breakout_cnn_best_meta.json` | Metadata: accuracy, precision, recall, training date |
| `models/feature_contract.json` | Feature names + normalization constants for inference |

### Syncing Models Locally

`run.sh` automatically calls `sync_models.sh` if `models/breakout_cnn_best.pt` is missing.

---

## Project Structure

```
futures/
├── src/
│   └── lib/
│       ├── core/                           # Infrastructure
│       │   ├── config.py                   #   Global configuration
│       │   ├── cache.py                    #   Redis cache + data source abstraction
│       │   ├── models.py                   #   Database models + Postgres ORM
│       │   ├── alerts.py                   #   Alert dispatch (email, webhook)
│       │   ├── logging_config.py           #   Structured logging (structlog)
│       │   ├── db/                         #   Database helpers + migrations
│       │   └── exceptions/                 #   Custom exception types
│       │
│       ├── indicators/                     # Technical indicators
│       │   ├── momentum/                   #   RSI, MACD, stochastic, etc.
│       │   ├── trend/                      #   Moving averages, ADX, etc.
│       │   ├── volatility/                 #   ATR, Bollinger, Keltner, etc.
│       │   ├── volume/                     #   OBV, CVD, VWAP, etc.
│       │   └── other/                      #   Miscellaneous indicators
│       │
│       ├── integrations/                   # External services
│       │   ├── rithmic_client.py           #   Rithmic (async_rithmic) order execution + market data
│       │   ├── massive_client.py           #   Massive.com REST + WebSocket client
│       │   ├── kraken_client.py            #   Kraken REST + WebSocket crypto client
│       │   ├── news_client.py              #   News feed integration
│       │   ├── grok_helper.py              #   xAI Grok AI analyst
│       │   ├── reddit_watcher.py           #   Reddit sentiment polling
│       │   └── pine/                       #   TradingView Pine Script tools
│       │
│       ├── model/                          # ML model library
│       │   ├── base/                       #   Base model classes
│       │   ├── deep/                       #   CNN, transformer architectures
│       │   ├── ensemble/                   #   Ensemble methods
│       │   ├── evaluation/                 #   Model evaluation + metrics
│       │   ├── ml/                         #   Classical ML models
│       │   ├── prediction/                 #   Prediction pipelines
│       │   └── statistical/                #   Statistical models
│       │
│       ├── trading/                        # Trading logic
│       │   └── strategies/                 #   Strategy implementations
│       │       ├── rb/                     #     Range breakout (breakout.py)
│       │       └── ruby_signal_engine.py   #     Core signal engine
│       │
│       └── services/                       # Deployable services
│           ├── data/                       #   FastAPI data service (port 8000)
│           ├── engine/                     #   Background engine worker (no HTTP)
│           ├── web/                        #   Reverse proxy (port 8080)
│           └── training/                   #   GPU trainer (port 8501)
│
├── static/                                 # Frontend static assets
│   ├── trading.html                        #   Main trading dashboard
│   └── pine.html                           #   Pine Script viewer
│
├── models/                                 # CNN model files (git-tracked via LFS)
│   ├── breakout_cnn_best.pt                #   Champion PyTorch checkpoint
│   ├── breakout_cnn_best_meta.json         #   Champion metadata (acc, prec, recall, date)
│   └── feature_contract.json               #   Feature names + normalization constants
│
├── scripts/
│   ├── daily_report.py                     # End-of-day breakout session summary
│   ├── monitor_signals.py                  # Live breakout signal terminal monitor
│   ├── session_signal_audit.py             # Per-session signal quality audit
│   └── smoke_test_trainer.py               # Quick end-to-end trainer smoke test
│
├── docker/                                 # Docker build contexts
│   ├── data/Dockerfile                     #   Data API container
│   ├── engine/Dockerfile                   #   Background engine container
│   ├── web/Dockerfile                      #   Web frontend container
│   ├── trainer/Dockerfile                  #   GPU trainer container
│   ├── charting/Dockerfile                 #   Charting service container
│   └── docker-compose.yml                  #   Full service stack
│
├── docs/                                   # Design docs
│   ├── architecture.md                     #   System architecture reference
│   ├── logging.md                          #   Logging conventions
│   ├── completed.md                        #   Completed work log
│   └── backlog.md                          #   Backlog / roadmap
│
├── tests/                                  # Pytest test suite (2,700+ tests)
│
├── config/
│   ├── grafana/                            # Grafana provisioning + dashboards
│   └── prometheus/                         # Prometheus scrape config
│
├── dataset/                                # Generated training datasets (git-ignored)
├── data/                                   # Persistent app data (git-ignored)
├── docker-compose.yml                      # Full service stack
├── pyproject.toml                          # Python project config (hatch + deps)
├── run.sh                                  # One-command build + deploy script
└── todo.md                                 # Project status & phase tracking
```

---

## Configuration

### Environment Variables

#### Required (auto-generated by `run.sh`)

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `REDIS_PASSWORD` | Redis password |

#### API Keys

| Variable | Description | Fallback |
|---|---|---|
| `MASSIVE_API_KEY` | [Massive.com](https://massive.com) — current primary for CME futures (REST + WebSocket, futures beta) | yfinance (delayed) |
| `XAI_API_KEY` | [xAI](https://console.x.ai) Grok AI analyst | AI features disabled |
| `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` | [Kraken](https://www.kraken.com) crypto spot data | Crypto panels hidden |

#### Rithmic (Execution + Market Data)

| Variable | Description |
|---|---|
| `RITHMIC_USERNAME` | Rithmic login username |
| `RITHMIC_PASSWORD` | Rithmic login password |
| `RITHMIC_SYSTEM_NAME` | Rithmic system name (e.g. `Rithmic Paper Trading`) |
| `RITHMIC_GATEWAY` | Rithmic gateway (e.g. `Chicago`) |

Without Rithmic credentials the engine runs in signal-only mode — all detection and CNN
inference still works, but no orders are placed.

#### Trading

| Variable | Default | Description |
|---|---|---|
| `ACCOUNT_SIZE` | `150000` | Account size for risk calculations ($50K, $100K, or $150K) |
| `ORB_FILTER_GATE` | `majority` | Filter strictness: `all`, `majority`, or `none` |
| `ORB_CNN_GATE` | `0` | `0` = CNN advisory only, `1` = CNN hard gate (blocks trade signal) |

#### Trainer

| Variable | Default | Description |
|---|---|---|
| `CNN_RETRAIN_EPOCHS` | `25` | Default training epochs |
| `CNN_RETRAIN_BATCH_SIZE` | `64` | Default batch size |
| `CNN_RETRAIN_LR` | `0.0002` | Default learning rate |
| `CNN_RETRAIN_PATIENCE` | `8` | Early stopping patience |
| `CNN_RETRAIN_MIN_ACC` | `80.0` | Minimum validation accuracy to promote a model (%) |
| `CNN_RETRAIN_MIN_PRECISION` | `75.0` | Minimum precision gate (%) |
| `CNN_RETRAIN_MIN_RECALL` | `70.0` | Minimum recall gate (%) |
| `CNN_RETRAIN_DAYS_BACK` | `90` | Days of history for dataset generation |
| `CNN_RETRAIN_SYMBOLS` | *(22 CME futures + BTC/ETH/SOL)* | Comma-separated symbol list |
| `TRAINER_API_KEY` | *(unset)* | Optional bearer token to protect the `/train` endpoint |

### Key Runtime Defaults

| Setting | Value | Location |
|---|---|---|
| CNN inference threshold | 0.82 | `breakout_cnn.py` |
| Engine refresh interval | 5 minutes | `scheduler.py` |
| Grok update interval | 15 minutes | `scheduler.py` |
| Risk check interval | 1 minute | `scheduler.py` |

---

## Testing

```bash
# Full test suite (2,700+ tests)
pytest tests/ -x -q --tb=short

# Specific modules
pytest tests/test_copy_trader.py -v            # Rithmic copy trader
pytest tests/test_ruby_signal_engine.py -v     # core signal engine
pytest tests/test_orb_filters.py -v            # ORB filter logic
pytest tests/test_scheduler.py -v              # session scheduling
pytest tests/test_risk.py -v                   # risk manager
pytest tests/test_ict.py -v                    # ICT/SMC concepts
pytest tests/test_cvd.py -v                    # cumulative volume delta
pytest tests/test_volume_profile.py -v         # volume profile analysis
pytest tests/test_data_service.py -v           # FastAPI endpoints

# With coverage
pytest tests/ --cov=lib --cov-report=html

# Linting
ruff check src/

# Quick trainer smoke test (requires trainer service running)
PYTHONPATH=src python scripts/smoke_test_trainer.py
```

---

## Scripts & Tools

### Model Sync

`run.sh` automatically calls `sync_models.sh` if `models/breakout_cnn_best.pt` is missing.

### Daily Report

```bash
PYTHONPATH=src python scripts/daily_report.py              # today's report
PYTHONPATH=src python scripts/daily_report.py --days 5     # last 5 days
PYTHONPATH=src python scripts/daily_report.py --json       # JSON output
```

### Live Signal Monitor

```bash
PYTHONPATH=src python scripts/monitor_signals.py              # watch live signals
PYTHONPATH=src python scripts/monitor_signals.py --interval 2 # 2s polling
PYTHONPATH=src python scripts/monitor_signals.py --json       # JSON output
```

### Session Signal Audit

```bash
PYTHONPATH=src python scripts/session_signal_audit.py                      # all sessions, 30 days
PYTHONPATH=src python scripts/session_signal_audit.py --days 14            # last 14 days
PYTHONPATH=src python scripts/session_signal_audit.py --export-json out.json
```

---

## Technologies

| Layer | Stack |
|---|---|
| **Language** | Python 3.11+ |
| **Web** | FastAPI, HTMX, SSE |
| **Data** | Rithmic (async_rithmic), Massive.com (REST + WebSocket), yfinance (fallback), Kraken (crypto), pandas |
| **Storage** | PostgreSQL 16, Redis 7 |
| **AI / ML** | PyTorch (CNN training + inference), xAI Grok (AI analyst) |
| **Analysis** | scikit-learn, hmmlearn (HMM regime), backtesting.py, Optuna |
| **Execution** | Rithmic (async_rithmic) — prop-firm order execution + market data |
| **Charting** | ApexCharts + nginx |
| **Observability** | structlog, Prometheus, Grafana |
| **Deployment** | Docker Compose, Tailscale mesh |

---

## License

[MIT](LICENSE) — Copyright (c) 2026 nuniesmith