#!/usr/bin/env python3
"""
Futures Trading System — Printable Section Pages (v7)

Generates a single print-ready HTML file where each page corresponds
to one major system section. Open in any browser and Ctrl+P to print.

CSS enforces 8.5"×11" landscape with correct margins.
Mermaid diagrams render via CDN (requires internet when opening in browser).

Run:   python scripts/generate_printable_pages.py
Output: docs/futures_system_printable.html
"""

from pathlib import Path
from typing import Any

# =============================================================================
# SECTIONS — each becomes one printed page
# title:        page header
# subtitle:     short description printed under the title
# receives:     list of (label, page_num) tuples for incoming connections
# sends:        list of (label, page_num) tuples for outgoing connections
# mermaid:      standalone flowchart for this section
#               Use STUB nodes (class "stub") for cross-section references.
# =============================================================================

SECTIONS: list[dict[str, Any]] = [
    # ──────────────────────────────────────────────────────────────
    #  PAGE 0 — OVERVIEW
    # ──────────────────────────────────────────────────────────────
    {
        "title": "System Overview",
        "subtitle": "Full architecture at a glance — connect pages along the arrows",
        "receives": [],
        "sends": [("all sections", "2-17")],
        "mermaid": """
flowchart LR
    EXT["🌐 External Data\n(p.2)"]
    DATA["📥 Data Layer\n(p.3)"]
    REG["📋 Asset Registry\n(p.4)"]
    SCHED["⏰ Scheduler\n(p.4)"]
    PRE["🌅 Pre-Market\n(p.5)"]
    RUBY["💎 Ruby Signals\n(p.6)"]
    BCORE["🔍 Breakout Core\n(p.7)"]
    ORB["📡 ORB Sessions\n(p.8)"]
    QUAL["✅ Signal Quality\n(p.9)"]
    NOTRADE["🚫 No-Trade\n(p.10)"]
    RISK["🛡️ Risk Mgmt\n(p.10)"]
    POS["📈 Position Mgmt\n(p.11)"]
    SWING["📊 Swing\n(p.12)"]
    IND["📐 Indicators\n(p.13)"]
    MOD["🧠 Models\n(p.13)"]
    TRAIN["🏋️ Training\n(p.14)"]
    OFF["⚙️ Off-Hours\n(p.14)"]
    KRAK["💰 Kraken\n(p.15)"]
    CHART["📉 Charting\n(p.15)"]
    PINE["🌲 Pine Script\n(p.15)"]
    WEB["🖥️ Dashboard\n(p.16)"]
    ALERTS["🔔 Alerts\n(p.17)"]
    MON["📈 Monitoring\n(p.17)"]

    EXT -->|OHLCV / fills / news| DATA
    DATA <--> REG
    DATA -->|Redis cmds| SCHED
    SCHED --> PRE & ORB & RUBY & SWING & OFF
    PRE -->|focus assets| BCORE & SWING
    ORB -->|bar fetch + symbols| BCORE
    KRAK --> BCORE
    BCORE -->|CNN verdict| QUAL
    IND --> QUAL & RUBY
    QUAL -->|enriched signal| NOTRADE
    NOTRADE --> RISK
    RUBY --> RISK
    SWING --> RISK
    RISK -->|can_enter=True| POS
    MOD --> TRAIN
    OFF --> TRAIN
    TRAIN -->|hot-reload weights| BCORE
    POS & BCORE & RISK & NOTRADE --> ALERTS
    ALERTS -->|Discord webhooks| EXT
    DATA & BCORE & RISK & POS -->|Redis SSE| WEB
    WEB --> CHART & PINE
    DATA & TRAIN --> MON
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 2 — EXTERNAL DATA SOURCES
    # ──────────────────────────────────────────────────────────────
    {
        "title": "External Data Sources",
        "subtitle": "All third-party APIs and data providers",
        "receives": [("Discord webhooks ← Alerts", 17)],
        "sends": [
            ("OHLCV bars → DataResolver", 3),
            ("WebSocket ticks → Kraken feed", 3),
            ("sentiment snapshots → Reddit job", 3),
            ("news scores → News Sentiment", 5),
            ("positions/fills → LiveRiskPublisher", 10),
            ("morning brief → Grok card", 5),
            ("live updates → Dashboard", 16),
        ],
        "mermaid": """
flowchart TD
    subgraph External["🌐 External Data Sources"]
        A1["MassiveAPI
CME futures OHLCV
integrations/massive_client.py"]
        A2["Kraken REST + WebSocket
Crypto spot pairs 24/7
integrations/kraken_client.py"]
        A3["Reddit via PRAW + VADER
integrations/reddit_watcher.py"]
        A4["Finnhub + Alpha Vantage
integrations/news_client.py"]
        A5["Rithmic
Live account + order management
integrations/rithmic_client.py"]
        A6["Grok / xAI
Macro briefs + live updates
integrations/grok_helper.py"]
    end

    DATA_IN["→ Data Layer  (p.3)"]:::stub
    NEWS_OUT["→ News Sentiment  (p.5)"]:::stub
    RISK_OUT["→ Live Risk  (p.10)"]:::stub
    WEB_OUT["→ Dashboard  (p.16)"]:::stub
    ALERTS_IN["← Discord Alerts  (p.17)"]:::stub

    A1 -->|OHLCV bars| DATA_IN
    A2 -->|REST OHLCV + ticker| DATA_IN
    A2 -->|WebSocket ticks| DATA_IN
    A3 -->|sentiment snapshots| DATA_IN
    A4 -->|news scores| NEWS_OUT
    A5 -->|positions / fills| RISK_OUT
    A6 -->|morning brief| DATA_IN
    A6 -->|live updates| WEB_OUT
    ALERTS_IN -->|webhooks inbound| A6

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 3 — DATA LAYER
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Data Layer",
        "subtitle": "Docker: data — DataResolver, Redis, Postgres, FastAPI, WebSocket feeds",
        "receives": [
            ("OHLCV / ticks / sentiment ← External", 2),
        ],
        "sends": [
            ("Redis cmds → Scheduler", 4),
            ("asset specs ↔ Asset Registry", 4),
            ("SSE stream → Dashboard", 16),
            ("/metrics → Monitoring", 17),
        ],
        "mermaid": """
flowchart TD
    EXT_IN["← External Data Sources  (p.2)"]:::stub
    SCHED_OUT["→ Scheduler via Redis cmds  (p.4)"]:::stub
    REG_IO["↔ Asset Registry  (p.4)"]:::stub
    WEB_OUT["→ Dashboard SSE  (p.16)"]:::stub
    MON_OUT["→ Monitoring /metrics  (p.17)"]:::stub

    subgraph Data["📥 Data Service  [Docker: data]"]
        B["DataResolver
Redis hot → Postgres durable → External fallback
services/data/resolver.py"]
        C["Redis :6379
Bar caches · focus assets · daily plan
risk state · Ruby signals · model events
live risk · swing states"]
        D["Postgres :5432
trades · journal · orb_events
risk_events · bars · audit
core/models.py + core/db/"]
        E["FastAPI data service  :8000
services/data/main.py
REST + SSE + API routers
services/data/api/*"]
        F["Kraken WebSocket feed
start_kraken_feed()
Bars pushed → Redis on bar close
services/data/main.py lifespan"]
        G["Reddit aggregation job
5-min polling loop
get_full_snapshot() → Redis
services/data/main.py"]
    end

    EXT_IN -->|OHLCV bars| B
    EXT_IN -->|WebSocket ticks| F
    EXT_IN -->|sentiment snapshots| G
    B -->|bars_1m / bars_15m / daily| C
    B -->|trades / journal / events| D
    E --> B
    F --> C
    G --> C
    B --> REG_IO
    C -->|Redis commands| SCHED_OUT
    C --> WEB_OUT
    E -->|/metrics :8000| MON_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 4 — ASSET REGISTRY + SCHEDULER
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Asset Registry & Scheduler",
        "subtitle": "core/asset_registry.py + services/engine/scheduler.py",
        "receives": [
            ("Redis commands ← Data Layer", 3),
            ("asset specs ↔ Data Layer", 3),
        ],
        "sends": [
            ("triggers → Pre-Market Pipeline", 5),
            ("triggers → ORB Sessions", 8),
            ("triggers → Ruby Signal Engine", 6),
            ("triggers → Swing Detector", 12),
            ("triggers → Off-Hours Tasks", 14),
        ],
        "mermaid": """
flowchart TD
    DATA_IN["← Data Layer / Redis cmds  (p.3)"]:::stub
    PRE_OUT["→ Pre-Market Pipeline  (p.5)"]:::stub
    ORB_OUT["→ ORB Sessions  (p.8)"]:::stub
    RUBY_OUT["→ Ruby Signal Engine  (p.6)"]:::stub
    SWING_OUT["→ Swing Detector  (p.12)"]:::stub
    OFF_OUT["→ Off-Hours Tasks  (p.14)"]:::stub

    subgraph Registry["📋 Asset Registry  (core/asset_registry.py)"]
        H["AssetRegistry
Futures: Metals · Energy · Equity Index
FX · Treasuries · Agriculture
Crypto: BTC · ETH · SOL + more
Micro + Full + Spot variants
Asset.compute_position_size()
Asset.dual_sizing()"]
    end

    subgraph Scheduler["⏰ ScheduleManager  (services/engine/scheduler.py)"]
        I1["EVENING  18:00–00:00 ET
CME · Sydney · Tokyo · Shanghai ORB sessions"]
        I2["PRE_MARKET  00:00–03:00 ET
COMPUTE_DAILY_FOCUS · GROK_MORNING_BRIEF
CHECK_NEWS_SENTIMENT 07:00 ET · PREP_ALERTS"]
        I3["ACTIVE  03:00–12:00 ET
Frankfurt · London · LN-NY · US Open ORB
RUBY_RECOMPUTE every 5 min
GROK_LIVE_UPDATE every 15 min
CHECK_RISK_RULES · CHECK_NO_TRADE
CHECK_SWING every 2 min
CHECK_ORB_CME_SETTLE 14:00–15:30 ET
POSITION_CLOSE_WARNING 15:45 ET
EOD_POSITION_CLOSE 16:00 ET"]
        I4["OFF_HOURS  12:00–18:00 ET
HISTORICAL_BACKFILL · RUN_OPTIMIZATION
RUN_BACKTEST · GENERATE_CHART_DATASET
TRAIN_BREAKOUT_CNN · DAILY_REPORT
CHECK_NEWS_SENTIMENT_MIDDAY · NEXT_DAY_PREP"]
    end

    DATA_IN -->|Redis commands| Scheduler
    H -->|asset specs / tickers / sizing| DATA_IN
    Scheduler --> PRE_OUT
    Scheduler --> ORB_OUT
    Scheduler --> RUBY_OUT
    Scheduler --> SWING_OUT
    Scheduler --> OFF_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 5 — PRE-MARKET PIPELINE
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Pre-Market Pipeline",
        "subtitle": "00:00–03:00 ET — scoring, bias analysis, daily plan, Grok brief",
        "receives": [
            ("trigger ← Scheduler", 4),
            ("news scores ← External (Finnhub/AV)", 2),
        ],
        "sends": [
            ("focus assets + daily plan → Breakout Core", 7),
            ("swing candidates → Swing Detector", 12),
            ("daily plan → Redis / Dashboard", 16),
        ],
        "mermaid": """
flowchart TD
    SCHED_IN["← Scheduler trigger  (p.4)"]:::stub
    EXT_IN["← Finnhub + Alpha Vantage  (p.2)"]:::stub
    BCORE_OUT["→ Breakout Core  (p.7)"]:::stub
    SWING_OUT["→ Swing Detector  (p.12)"]:::stub
    WEB_OUT["→ Dashboard / Redis  (p.16)"]:::stub

    subgraph PreMarket["🌅 Pre-Market Pipeline  (00:00–03:00 ET)"]
        J["COMPUTE_DAILY_FOCUS
PreMarketScorer: NATR · RVOL · gap
momentum · catalyst scores
analysis/scorer.py"]
        K["DailyBiasAnalyzer + DailyPlanGenerator
trading/strategies/daily/bias_analyzer.py
trading/strategies/daily/daily_plan.py
SwingDetector candidates
trading/strategies/daily/swing_detector.py"]
        L["compute_daily_focus()
ConvictionStack multipliers:
news sentiment · crypto momentum
Grok brief · regime
services/engine/focus.py"]
        M["publish_focus_to_redis()
engine:focus_assets
engine:daily_plan → Redis"]
        N["GROK_MORNING_BRIEF
grok_helper.py
Macro context + daily plan
→ dashboard chat cards"]
        O["CHECK_NEWS_SENTIMENT  07:00 ET
Finnhub + Alpha Vantage + VADER
engine:news_sentiment:[SYM]  2h TTL
analysis/sentiment/news_sentiment.py"]
    end

    SCHED_IN --> J
    EXT_IN -->|news scores| O
    J --> K --> L --> M
    M --> N --> O
    M -->|focus assets + daily plan| BCORE_OUT
    M -->|swing candidates| SWING_OUT
    M --> WEB_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 6 — RUBY SIGNAL ENGINE
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Ruby Signal Engine",
        "subtitle": "services/engine/ruby_signal_engine.py — parallel HMA/ORB signal path",
        "receives": [
            ("trigger every 5 min ← Scheduler", 4),
            ("computed series ← Indicators", 13),
        ],
        "sends": [
            ("RubySignal → Risk Management", 10),
            ("RubySignal → Redis engine:ruby:[SYM]", 3),
        ],
        "mermaid": """
flowchart TD
    SCHED_IN["← Scheduler RUBY_RECOMPUTE every 5 min  (p.4)"]:::stub
    IND_IN["← Indicator Library ATR / VWAP / EMA / squeeze  (p.13)"]:::stub
    RISK_OUT["→ Risk Management  (p.10)"]:::stub
    REDIS_OUT["→ Redis engine:ruby:[SYM]  (p.3)"]:::stub

    subgraph Ruby["💎 Ruby Signal Engine"]
        RB1["RubySignalEngine
HMA channel + EMA bias
ORB / IB detection
VWAP alignment
ATR14 stop sizing
TP1 / TP2 / TP3 R-multiples
RSI + squeeze bands
services/engine/ruby_signal_engine.py"]
        RB2["RubySignal
Compatible with PositionManager
process_signal() interface
Published → Redis engine:ruby:[SYM]"]
        RB3["handle_ruby_recompute()
Scans focus assets with latest bars
services/engine/handlers.py"]
    end

    SCHED_IN --> RB3
    IND_IN --> RB1
    RB1 --> RB2 --> RB3
    RB3 -->|RubySignal| RISK_OUT
    RB3 -->|RubySignal → Redis| REDIS_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 7 — BREAKOUT DETECTION CORE
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Breakout Detection Core",
        "subtitle": "trading/strategies/rb/ — 13 breakout types, quality filters, CNN inference",
        "receives": [
            ("focus assets + daily plan ← Pre-Market", 5),
            ("bar fetch + symbols ← ORB Sessions", 8),
            ("Kraken crypto data ← Kraken Portfolio", 15),
            ("hot-reload weights ← Training Pipeline", 14),
        ],
        "sends": [
            ("CNN verdict + features → Signal Quality", 9),
            ("breakout results → Postgres", 3),
            ("signal alerts → Alerts", 17),
        ],
        "mermaid": """
flowchart TD
    PRE_IN["← Pre-Market: focus assets + daily plan  (p.5)"]:::stub
    ORB_IN["← ORB Sessions: bar fetch + symbols  (p.8)"]:::stub
    KRAK_IN["← Kraken Portfolio  (p.15)"]:::stub
    TRAIN_IN["← Training: hot-reload model weights  (p.14)"]:::stub
    QUAL_OUT["→ Signal Quality & Confluence  (p.9)"]:::stub
    DB_OUT["→ Postgres: breakout results  (p.3)"]:::stub
    ALERTS_OUT["→ Alerts  (p.17)"]:::stub

    subgraph BreakoutCore["🔍 Breakout Detection  (trading/strategies/rb/)"]
        P["13 BreakoutTypes  (core/breakout_types.py)
ORB · PrevDay · InitialBalance
Consolidation · Weekly · Monthly
Asian · BollingerSqueeze · ValueArea
InsideDay · GapRejection · PivotPoints · Fibonacci"]
        Q["detect_range_breakout()
range_builders.py + breakout.py
detector.py facade · RangeConfig per type"]
        R["run_quality_filters()
NR7 · pre-market range · session window
lunch filter · VWAP confluence · MTF bias
services/engine/handlers.py
analysis/breakout_filters.py"]
        S["build_cnn_tabular_features()
25+ features: ATR trend · RVOL · daily bias
session overlap · crypto momentum · news score
regime · Ruby signal
services/engine/handlers.py"]
        T["HybridBreakoutCNN inference
PyTorch tabular + type/asset embeddings
→ LONG / SHORT / SKIP
analysis/ml/breakout_cnn.py
run_cnn_inference()"]
    end

    PRE_IN --> P
    ORB_IN --> P
    KRAK_IN --> P
    TRAIN_IN -->|model weights| T
    P --> Q --> R --> S --> T
    T -->|CNN verdict + tabular features| QUAL_OUT
    T --> DB_OUT
    T --> ALERTS_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 8 — ORB SESSION HANDLERS
    # ──────────────────────────────────────────────────────────────
    {
        "title": "ORB Session Handlers",
        "subtitle": "services/engine/main.py — all global trading session windows",
        "receives": [("session triggers ← Scheduler", 4)],
        "sends": [
            ("bar fetch + symbol list → Breakout Core", 7),
        ],
        "mermaid": """
flowchart TD
    SCHED_IN["← Scheduler session triggers  (p.4)"]:::stub
    BCORE_OUT["→ Breakout Core: handle_orb_check / handle_breakout_check  (p.7)"]:::stub

    SCHED_IN --> Evening & Morning & USDay & Settlement & Multi

    subgraph Evening["🌙 Evening Sessions  (ET)"]
        direction LR
        U1["CHECK_ORB_CME
18:00–20:00
CME Globex re-open"]
        U2["CHECK_ORB_SYDNEY
18:30–20:30"]
        U3["CHECK_ORB_TOKYO
19:00–21:00"]
        U4["CHECK_ORB_SHANGHAI
21:00–23:00"]
    end

    subgraph Morning["🌅 Morning Sessions  (ET)"]
        direction LR
        U5["CHECK_ORB_FRANKFURT
03:00–04:30"]
        U6["CHECK_ORB_LONDON
03:00–05:00"]
        U7["CHECK_ORB_LONDON_NY
08:00–10:00"]
        U8["CHECK_ORB
09:30–11:00
US Open"]
    end

    subgraph USDay["☀️ US Day / Crypto  (ET)"]
        direction LR
        U9["CHECK_ORB_CME_SETTLE
14:00–15:30
metals / energy"]
        U10["CHECK_ORB_CRYPTO
UTC0 + UTC12
BTC · ETH · SOL"]
    end

    subgraph Multi["🔄 Multi-Type Scan  (every 2 min)"]
        U11["CHECK_BREAKOUT_MULTI
PDR · IB · Consolidation + 9 additional types in parallel
handle_breakout_multi
trading/strategies/rb/open/sessions.py"]
    end

    Evening & Morning & USDay & Multi --> BCORE_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 9 — SIGNAL QUALITY & CONFLUENCE
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Signal Quality & Confluence",
        "subtitle": "analysis/ — MTF, regime, volume, ICT, wave, quality scoring",
        "receives": [
            ("CNN verdict + tabular features ← Breakout Core", 7),
            ("computed series ← Indicator Library", 13),
        ],
        "sends": [("enriched signal → No-Trade Guard", 10)],
        "mermaid": """
flowchart TD
    BCORE_IN["← Breakout Core: CNN verdict + features  (p.7)"]:::stub
    IND_IN["← Indicator Library: computed series  (p.13)"]:::stub
    NT_OUT["→ No-Trade Guard: enriched signal  (p.10)"]:::stub

    subgraph Quality["✅ Signal Quality & Confluence  (analysis/)"]
        subgraph TimeSeries["📈 Price & Regime Analysis"]
            direction LR
            V1["MTFAnalyzer / analyze_mtf()
HTF bias + EMA alignment
divergence detection
analysis/mtf_analyzer.py"]
            V2["RegimeFilter
Volatility regime
trend strength
analysis/regime.py"]
            V4["CVD + VolumeProfile
analysis/cvd.py
analysis/volume_profile.py"]
            V7["SignalQuality
velocity · acceleration
trend context · candle patterns
analysis/signal_quality.py"]
        end
        subgraph CrossMarket["🌐 Cross-Market & Structure"]
            direction LR
            V3["CryptoMomentumScore
analysis/cross_asset.py
analysis/crypto_momentum.py"]
            V5["ICT concepts
FVG · Order Blocks · liquidity
analysis/ict.py"]
            V6["WaveAnalysis
EW structure
analysis/wave_analysis.py"]
            V8["AssetFingerprint
analysis/asset_fingerprint.py"]
        end
    end

    BCORE_IN --> TimeSeries & CrossMarket
    IND_IN --> V1 & V2 & V4 & V7
    TimeSeries & CrossMarket -->|enriched signal| NT_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 10 — NO-TRADE GUARD + RISK MANAGEMENT
    # ──────────────────────────────────────────────────────────────
    {
        "title": "No-Trade Guard & Risk Management",
        "subtitle": "services/engine/patterns.py + risk.py + live_risk.py",
        "receives": [
            ("enriched signal ← Signal Quality", 9),
            ("RubySignal ← Ruby Engine", 6),
            ("swing signal ← Swing Detector", 12),
            ("positions/fills ← Rithmic / External", 2),
        ],
        "sends": [
            ("can_enter=True → Position Manager", 11),
            ("live_risk state → Redis", 3),
            ("risk events → Alerts", 17),
            ("no-trade alerts → Alerts", 17),
        ],
        "mermaid": """
flowchart TD
    QUAL_IN["← Signal Quality: enriched signal  (p.9)"]:::stub
    RUBY_IN["← Ruby Signal Engine: RubySignal  (p.6)"]:::stub
    SWING_IN["← Swing Detector: swing signal  (p.12)"]:::stub
    EXT_IN["← Rithmic: positions / fills  (p.2)"]:::stub
    POS_OUT["→ Position Manager: can_enter=True  (p.11)"]:::stub
    REDIS_OUT["→ Redis: live_risk state  (p.3)"]:::stub
    ALERTS_OUT["→ Alerts  (p.17)"]:::stub

    subgraph NoTrade["🚫 No-Trade Guard  (services/engine/patterns.py)"]
        NT["evaluate_no_trade()
NoTradeConditions:
• low market data quality
• extreme volatility
• daily loss limit hit
• consecutive loss limit
• late session / no setups
• session ended
publish_no_trade_alert() → Redis"]
    end

    subgraph Risk["🛡️ Risk Management  (services/engine/risk.py + live_risk.py)"]
        W["RiskManager.can_enter_trade()
Daily P&L gate
Consecutive loss limit
Open position cap
Overnight risk check
services/engine/risk.py"]
        X["LiveRiskState
Per-asset position snapshots
Dynamic sizing · Health score
compute_live_risk()
services/engine/live_risk.py"]
        Y["LiveRiskPublisher
Ticks every 5s
Publishes → Redis engine:live_risk
_load_rithmic_positions()"]
    end

    QUAL_IN -->|enriched signal| NT
    NT -->|no active block| W
    RUBY_IN -->|RubySignal| W
    SWING_IN -->|swing signal| W
    EXT_IN -->|positions / fills| Y
    W -->|can_enter=True| POS_OUT
    W --> X --> Y
    Y -->|live_risk state| REDIS_OUT
    Y -->|risk events| ALERTS_OUT
    NT -->|no-trade alerts| ALERTS_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 11 — POSITION & ORDER MANAGEMENT
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Position & Order Management",
        "subtitle": "services/engine/position_manager.py + copy_trader.py + rithmic_client.py",
        "receives": [("can_enter=True ← Risk Management", 10)],
        "sends": [
            ("positions + P&L + OrderCommands → Redis", 3),
        ],
        "mermaid": """
flowchart TD
    RISK_IN["← Risk Management: can_enter=True  (p.10)"]:::stub
    REDIS_OUT["→ Redis: positions + PnL + OrderCommands  (p.3)"]:::stub

    subgraph Positions["📈 Position & Order Management"]
        Z["PositionManager.process_signal()
MicroPosition state machine
BracketPhase:
  ENTRY → TP1 → BE → TRAIL → TP3
SAR always-in reversal logic
Pyramid scaling via get_next_pyramid_level()
services/engine/position_manager.py"]
        AA["3-Phase Bracket
TP1 → move stop to BE
EMA9 trailing stop → TP3
_update_bracket_phase()
_check_stop_hit() / _check_tp3_hit()"]
        AB["CopyTrader.execute_order_commands()
Rithmic multi-account copy
Compliance checklist + RollingRateCounter
services/engine/copy_trader.py"]
        AC["RithmicAccountManager
Order placement on prop accounts
resolve_front_month()
EOD hard close 16:00 ET
integrations/rithmic_client.py"]
    end

    RISK_IN -->|can_enter=True| Z
    Z --> AA --> AB --> AC
    Z -->|positions + PnL + OrderCommands| REDIS_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 12 — SWING DETECTOR
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Swing Detector",
        "subtitle": "services/engine/swing.py — daily-plan swing trades, manual dashboard control",
        "receives": [
            ("trigger every 2 min ← Scheduler", 4),
            ("swing candidates ← Pre-Market", 5),
        ],
        "sends": [("swing signal → Risk Management", 10)],
        "mermaid": """
flowchart TD
    SCHED_IN["← Scheduler CHECK_SWING every 2 min  (p.4)"]:::stub
    PRE_IN["← Pre-Market: swing candidates  (p.5)"]:::stub
    RISK_OUT["→ Risk Management: swing signal  (p.10)"]:::stub

    subgraph Swing["📊 Swing Detector  (services/engine/swing.py)"]
        AD["CHECK_SWING  every 2 min  03:00–15:30 ET
tick_swing_detector()
Scans daily-plan swing candidates
SwingState: PENDING → ACTIVE → CLOSED"]
        AE["Entry Types
Pullback · Breakout · Gap
15m + 5m bar fetch

Manual dashboard actions:
accept_swing_signal()
ignore_swing_signal()
close_swing_position()
move_stop_to_breakeven()"]
    end

    SCHED_IN --> AD
    PRE_IN -->|swing candidates| AD
    AD --> AE
    AE -->|swing signal| RISK_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 13 — INDICATOR LIBRARY + MODEL LAYER
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Indicator Library & Model Layer",
        "subtitle": "indicators/ + model/ — computed series feeding quality, Ruby, and training",
        "receives": [],
        "sends": [
            ("computed series → Signal Quality", 9),
            ("ATR/VWAP/EMA/squeeze → Ruby Engine", 6),
            ("deep/ml/statistical models → Training", 14),
        ],
        "mermaid": """
flowchart TD
    QUAL_OUT["→ Signal Quality  (p.9)"]:::stub
    RUBY_OUT["→ Ruby Signal Engine  (p.6)"]:::stub
    TRAIN_OUT["→ Training Pipeline  (p.14)"]:::stub

    subgraph Indicators["📐 Indicator Library  (indicators/)"]
        IND1["IndicatorManager + Registry
indicators/manager.py · registry.py · factory.py"]
        subgraph IndTypes["Indicator Types"]
            direction LR
            IND2["Trend: EMA · HMA · VWAP
indicators/trend/
Momentum: RSI · MACD
indicators/momentum/"]
            IND3["Volume: OBV · CVD
indicators/volume/
Other: ATR · BB · KC
indicators/other/"]
            IND4["CandlePatterns
indicators/candle_patterns.py
AreasOfInterest
indicators/areas_of_interest.py
MarketTiming
indicators/market_timing.py"]
        end
    end

    subgraph Models["🧠 Model Layer  (model/)"]
        MOD1["ModelService + ModelRegistry
model/service.py · model/registry.py · model/factory.py"]
        subgraph ModTypes["Model Types"]
            direction LR
            MOD2["Deep
CNN · LSTM · Transformer
TFT · NN
model/deep/"]
            MOD3["ML
XGBoost · LightGBM · CatBoost
Logistic · Gaussian · Polynomial
model/ml/"]
            MOD4["Statistical
ARIMA · GARCH · HMM
Prophet · Bayesian
model/statistical/
Ensemble  model/ensemble/"]
            MOD5["Prediction
PredictionManager
single/multi generators
model/prediction/
Evaluation: cross_val · metrics
model/evaluation/"]
        end
    end

    IND1 --> IndTypes
    MOD1 --> ModTypes
    IND1 -->|computed series| QUAL_OUT
    IND1 -->|ATR · VWAP · EMA · squeeze| RUBY_OUT
    MOD1 -->|deep / ml / statistical models| TRAIN_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 14 — TRAINING PIPELINE + OFF-HOURS TASKS
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Training Pipeline & Off-Hours Tasks",
        "subtitle": "Docker: trainer — CNN train, backfill, optimization, backtest, daily report",
        "receives": [
            ("triggers ← Scheduler", 4),
            ("deep/ml models ← Model Layer", 13),
        ],
        "sends": [
            ("hot-reload model weights → Breakout Core", 7),
            ("daily report → Alerts", 17),
            ("/metrics → Monitoring", 17),
        ],
        "mermaid": """
flowchart TD
    SCHED_IN["← Scheduler OFF_HOURS triggers  (p.4)"]:::stub
    MOD_IN["← Model Layer: deep/ml/statistical models  (p.13)"]:::stub
    BCORE_OUT["→ Breakout Core: hot-reload model weights  (p.7)"]:::stub
    ALERTS_OUT["→ Alerts: daily report  (p.17)"]:::stub
    MON_OUT["→ Monitoring /metrics :8001  (p.17)"]:::stub

    subgraph Training["🏋️ Training Pipeline  [Docker: trainer]"]
        AF["OFF_HOURS triggers
GENERATE_CHART_DATASET → DatasetGenerator
RBSimulator
services/training/rb_simulator.py"]
        AG["dataset_generator.py
180-day lookback
13 breakout types × all assets
_build_row() 25+ features
split_dataset() + validate_dataset()"]
        AH["TRAIN_BREAKOUT_CNN
HybridBreakoutCNN
PyTorch + Optuna walk-forward
train_model() + evaluate_model()
analysis/ml/breakout_cnn.py
trainer_server.py  FastAPI :8001"]
        AI["Champion promotion
breakout_cnn_best.pt
breakout_cnn_best_meta.json
feature_contract.json → models/
_archive_champion()"]
        AJ["ModelWatcher hot-reload
watchdog inotify / polling fallback
invalidate_model_cache()
Publishes model_reloaded → Redis
services/engine/model_watcher.py"]
    end

    subgraph OffHours["⚙️ Off-Hours  (12:00–18:00 ET)"]
        subgraph DataPrep["Data & Optimization"]
            direction LR
            AK["HISTORICAL_BACKFILL
run_backfill()
Warms Redis + Postgres
MassiveAPI / Kraken / yfinance"]
            AL["RUN_OPTIMIZATION
Optuna nightly study
walk-forward 30–90 days"]
            AM["RUN_BACKTEST
P&L + win-rate stats"]
        end
        subgraph EOD["End-of-Day"]
            direction LR
            AN["DAILY_REPORT
P&L · trades · signals
Grok review → email + Discord"]
            AO["POSITION_CLOSE_WARNING 15:45 ET
EOD_POSITION_CLOSE 16:00 ET
Rithmic cancel_all + exit_all"]
        end
    end

    SCHED_IN --> AF & DataPrep & EOD
    MOD_IN --> AF
    AF --> AG --> AH --> AI --> AJ
    AJ -->|hot-reload weights| BCORE_OUT
    AH -->|/metrics :8001| MON_OUT
    AN -->|daily report| ALERTS_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 15 — KRAKEN + CHARTING + PINE SCRIPT
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Kraken Portfolio, Charting & Pine Script",
        "subtitle": "24/7 crypto feed · Docker: charting (nginx) · TradingView Pine generator",
        "receives": [
            ("proxy requests ← Dashboard", 16),
        ],
        "sends": [
            ("crypto data → Breakout Core", 7),
        ],
        "mermaid": """
flowchart TD
    WEB_IN["← Dashboard proxy requests  (p.16)"]:::stub
    BCORE_OUT["→ Breakout Core: crypto ORB detection  (p.7)"]:::stub

    subgraph Kraken["💰 Kraken Crypto Portfolio  24/7"]
        AP["KrakenDataProvider
REST OHLCV + ticker
Portfolio balance queries
integrations/kraken_client.py"]
        AQ["Kraken WebSocket feed
Real-time OHLC + trades
Bars pushed → Redis on close
services/data/main.py lifespan"]
        AR["Crypto ORB sessions
CHECK_ORB_CRYPTO_UTC0
CHECK_ORB_CRYPTO_UTC12
Same 13-type detection pipeline"]
    end

    subgraph Charting["📉 Charting Service  [Docker: charting]"]
        CH["nginx :3001
Lightweight Charts + custom JS
Serves OHLCV candle charts
proxied via web service
docker/charting/"]
    end

    subgraph Pine["🌲 Pine Script Generator  (integrations/pine/)"]
        PI["PineScriptGenerator
Generates TradingView Pine modules
params.yaml driven
Output served via /pine/* API
integrations/pine/generate.py"]
    end

    AP --> AQ --> AR
    AR -->|crypto data| BCORE_OUT
    WEB_IN -->|proxied /charts/*| CH
    WEB_IN -->|proxied /pine/* API| PI

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 16 — DASHBOARD & WEB
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Dashboard & Web",
        "subtitle": "Docker: web + data — FastAPI :8080, HTMX + SSE, all dashboard panels",
        "receives": [
            ("Redis SSE stream ← Data Layer", 3),
            ("Grok live updates ← External", 2),
        ],
        "sends": [
            ("proxied /charts/* → Charting", 15),
            ("proxied /pine/* → Pine Script", 15),
        ],
        "mermaid": """
flowchart TD
    REDIS_IN["← Redis SSE stream / Data Layer  (p.3)"]:::stub
    CHART_OUT["→ Charting Service  (p.15)"]:::stub
    PINE_OUT["→ Pine Script Generator  (p.15)"]:::stub

    subgraph Web["🖥️ Web / Dashboard  [Docker: web + data]  —  services/web/main.py  :8080"]
        AS["FastAPI web proxy  :8080
HTMX + SSE dashboard"]

        subgraph LiveOps["📊 Live Operations"]
            direction LR
            AT["Live Risk strip
live_risk → SSE
P&L + health
/live_risk/*"]
            AU["Focus cards
Daily plan + conviction
Swing accept/ignore
/plan/*"]
            AW["Ruby signal panel
Per-asset state
/ruby/signals"]
        end

        subgraph AIFeatures["🤖 AI & Models"]
            direction LR
            AV["Grok chat + review
Live update every N min
/chat/* + /grok/*"]
            AX["CNN panel + model info
Trainer log stream
/cnn/* + /trainer/*"]
        end

        subgraph Records["📋 Records & Config"]
            direction LR
            AY["Journal + audit
Trade grading · ORB / RB history
/journal/* + /audit/*"]
            AZ2["Settings panel
Services · risk config · API keys
/settings/*"]
            BA2["Tasks panel
CRUD + GitHub push
/tasks/*"]
        end

        subgraph TradingCtrl["⚙️ Trading Controls"]
            direction LR
            BB2["Copy trade panel
Rate alerts · pyramid · focus
/copy_trade/*"]
            BC2["Trading settings
Test Rithmic / MassiveAPI / Kraken
/trading/*"]
        end
    end

    REDIS_IN --> AS
    AS --> LiveOps & AIFeatures & Records & TradingCtrl
    AS -->|proxied /charts/*| CHART_OUT
    AS -->|proxied /pine/* API| PINE_OUT

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
    # ──────────────────────────────────────────────────────────────
    #  PAGE 17 — ALERTS + MONITORING
    # ──────────────────────────────────────────────────────────────
    {
        "title": "Alerts & Monitoring",
        "subtitle": "core/alerts.py · Docker: prometheus + grafana",
        "receives": [
            ("signal alerts ← Breakout Core", 7),
            ("daily report ← Off-Hours", 14),
            ("risk events ← Risk Management", 10),
            ("no-trade alerts ← No-Trade Guard", 10),
            ("/metrics ← Data Service :8000", 3),
            ("/metrics ← Trainer Service :8001", 14),
        ],
        "sends": [("Discord webhooks → External", 2)],
        "mermaid": """
flowchart TD
    BCORE_IN["← Breakout Core: signal alerts  (p.7)"]:::stub
    OFF_IN["← Off-Hours: daily report  (p.14)"]:::stub
    RISK_IN["← Risk Management: risk events  (p.10)"]:::stub
    NT_IN["← No-Trade Guard: no-trade alerts  (p.10)"]:::stub
    DATA_IN["← Data Service /metrics :8000  (p.3)"]:::stub
    TRAIN_IN["← Trainer /metrics :8001  (p.14)"]:::stub
    EXT_OUT["→ External: Discord webhooks  (p.2)"]:::stub

    subgraph Alerts["🔔 Alerts  (core/alerts.py)"]
        AZ["Discord smart gate
Master toggle
Focus-only filter
Live breakout events
No-trade alerts
Gap alerts + daily report
Risk events"]
    end

    subgraph Monitoring["📈 Monitoring  [Docker: prometheus + grafana]"]
        BA["Prometheus :9090
Scrapes /metrics from
data :8000 · engine · trainer :8001"]
        BB["Grafana :3000
Dashboards:
P&L · signal quality
CNN accuracy · risk utilisation"]
    end

    BCORE_IN -->|signal alerts| AZ
    OFF_IN -->|daily report| AZ
    RISK_IN -->|risk events| AZ
    NT_IN -->|no-trade alerts| AZ
    AZ -->|Discord webhooks| EXT_OUT

    DATA_IN --> BA
    TRAIN_IN --> BA
    BA --> BB

    classDef stub fill:#e8e8e8,stroke:#aaa,stroke-dasharray:5 5,color:#555,font-style:italic
""",
    },
]

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Futures Trading System — Printable Architecture Pages</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  /* ── Base ─────────────────────────────────────────────────── */
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; }}

  /* ── Print page setup ─────────────────────────────────────── */
  @page {{ size: letter landscape; margin: 0.35in 0.35in; }}

  /* ── Screen preview ──────────────────────────────────────── */
  .page {{
    width: 10.3in;
    height: 7.6in;          /* fixed — matches printable landscape height */
    margin: 0.25in auto;
    padding: 0.15in 0.25in 0.1in;
    border: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0,0,0,.15);
    page-break-after: always;
    break-after: page;
    background: #fff;
    display: flex;
    flex-direction: column;
  }}
  .page:last-child {{ page-break-after: avoid; break-after: avoid; }}

  /* ── Page header ─────────────────────────────────────────── */
  .page-header {{
    flex-shrink: 0;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #1a3a5c;
    padding-bottom: 0.06in;
    margin-bottom: 0.07in;
  }}
  .page-header h1 {{
    font-size: 13pt;
    color: #1a3a5c;
    font-weight: 700;
    line-height: 1.2;
  }}
  .page-header .subtitle {{
    font-size: 7.5pt;
    color: #555;
    margin-top: 1px;
    font-style: italic;
  }}
  .page-num {{
    font-size: 8.5pt;
    color: #888;
    text-align: right;
    white-space: nowrap;
    padding-top: 2px;
    min-width: 55px;
  }}

  /* ── Diagram area ─────────────────────────────────────────────
     flex:1 makes this fill all space between header and footer.
     We let the SVG grow to fill height — so tall-narrow diagrams
     use vertical space instead of being squished to fit width.
  ──────────────────────────────────────────────────────────── */
  .diagram {{
    flex: 1;
    min-height: 0;          /* critical: allows flex child to shrink */
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  .diagram .mermaid {{
    /* fill the full diagram area height */
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .diagram .mermaid svg {{
    /* scale to fill height; only shrink width if needed */
    max-height: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    width: auto !important;
    display: block;
  }}

  /* ── Page footer ─────────────────────────────────────────── */
  .page-footer {{
    flex-shrink: 0;
    border-top: 1px solid #ddd;
    padding-top: 0.05in;
    margin-top: 0.06in;
    display: flex;
    gap: 0.15in;
    font-size: 7pt;
    color: #444;
    flex-wrap: wrap;
  }}
  .page-footer .col {{ flex: 1; min-width: 2in; }}
  .page-footer strong {{ color: #1a3a5c; }}
  .page-footer ul {{ list-style: none; padding: 0; margin: 1px 0 0 0; }}
  .page-footer ul li::before {{ content: "→ "; color: #888; }}
  .page-footer ul.receives li::before {{ content: "← "; color: #888; }}
  .page-footer .none {{ color: #aaa; font-style: italic; }}
</style>
</head>
<body>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'neutral',
    flowchart: {{ htmlLabels: true, curve: 'basis', rankSpacing: 40, nodeSpacing: 30 }},
    fontSize: 11,
    securityLevel: 'loose',
  }});
</script>

{pages}

</body>
</html>
"""

PAGE_TEMPLATE = """
<div class="page">
  <div class="page-header">
    <div>
      <h1>{icon_title}</h1>
      <div class="subtitle">{subtitle}</div>
    </div>
    <div class="page-num">Page {page_num} / {total_pages}</div>
  </div>

  <div class="diagram">
    <div class="mermaid">
{mermaid}
    </div>
  </div>

  <div class="page-footer">
    <div class="col">
      <strong>Receives from:</strong>
      {receives_html}
    </div>
    <div class="col">
      <strong>Sends to:</strong>
      {sends_html}
    </div>
  </div>
</div>
"""


def build_connection_list(items: list, css_class: str) -> str:
    if not items:
        return '<span class="none">none</span>'
    lis = "\n".join(f'        <li>{label}  <em style="color:#1a3a5c">(p.{page})</em></li>' for label, page in items)
    return f'<ul class="{css_class}">\n{lis}\n      </ul>'


def generate_html() -> str:
    total = len(SECTIONS)
    pages_html = []

    for idx, section in enumerate(SECTIONS):
        page_num = idx + 1
        receives_html = build_connection_list(section["receives"], "receives")
        sends_html = build_connection_list(section["sends"], "sends")

        page_html = PAGE_TEMPLATE.format(
            icon_title=section["title"],
            subtitle=section["subtitle"],
            page_num=page_num,
            total_pages=total,
            mermaid=section["mermaid"].strip(),
            receives_html=receives_html,
            sends_html=sends_html,
        )
        pages_html.append(page_html)

    return HTML_TEMPLATE.format(pages="\n".join(pages_html))


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "docs"
    output_dir.mkdir(exist_ok=True)

    out_path = output_dir / "futures_system_printable.html"
    html = generate_html()
    out_path.write_text(html, encoding="utf-8")

    print(f"✅  Generated {len(SECTIONS)} pages → {out_path}")
    print()
    print("How to print:")
    print("  1. Open the HTML file in Chrome or Firefox")
    print("  2. Wait for all Mermaid diagrams to render  (~5 sec)")
    print("  3. Ctrl+P  →  Layout: Landscape  →  Paper: Letter  →  Margins: None/Minimum")
    print("  4. Print!  Each section is one page.")
    print()
    print("Tip: Pages are ordered so edges align when laid side by side.")
    print("     Use the Overview page (p.1) as your connection map.")


if __name__ == "__main__":
    main()
