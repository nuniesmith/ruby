from pathlib import Path

import requests  # type: ignore[import-untyped]

# =============================================================================
# FUTURES TRADING SYSTEM — FULL LOGIC FLOW (v7)
# Accurately reflects the real project structure under src/lib/
#
# Run: python scripts/generate_mermaid.py
# Outputs:
#   - docs/futures_logic_flow.mmd     (open in mermaid.live or VSCode)
#   - docs/futures_logic_flow.svg     (rendered SVG — primary output)
#   - docs/futures_logic_flow.png     (rendered PNG — best-effort)
# =============================================================================

MERMAID = """flowchart TD
    %% ==================== EXTERNAL DATA SOURCES ====================
    subgraph External["🌐 External Data Sources"]
        A1["MassiveAPI\nCME futures OHLCV\n(integrations/massive_client.py)"]
        A2["Kraken REST + WebSocket\nCrypto spot pairs 24/7\n(integrations/kraken_client.py)"]
        A3["Reddit via PRAW + VADER\n(integrations/reddit_watcher.py)"]
        A4["Finnhub + Alpha Vantage\n(integrations/news_client.py)"]
        A5["Rithmic\nLive account + order management\n(integrations/rithmic_client.py)"]
        A6["Grok / xAI\nMacro briefs + live updates\n(integrations/grok_helper.py)"]
    end

    %% ==================== DATA LAYER ====================
    subgraph Data["📥 Data Service  [Docker: data]"]
        B["DataResolver\nRedis hot → Postgres durable → External fallback\n(services/data/resolver.py)"]
        B -->|"bars_1m / bars_15m / daily"| C["Redis :6379\nBar caches, focus assets,\ndaily plan, risk state,\nRuby signals, model events,\nlive risk, swing states"]
        B -->|"trades, journal, orb_events,\nrisk_events, bars, audit"| D["Postgres :5432\nDurable store\n(core/models.py + core/db/)"]
        E["FastAPI data service\n(services/data/main.py)\nREST + SSE + API routers\n(services/data/api/*)"]
        E --> B
        F["Kraken WebSocket feed\nstart_kraken_feed()\nBars pushed → Redis on bar close\n(services/data/main.py lifespan)"]
        F --> C
        G["Reddit aggregation job\n5-min polling loop\nget_full_snapshot() → Redis\n(services/data/main.py _reddit_aggregation_job)"]
        G --> C
    end

    %% ==================== ASSET REGISTRY ====================
    subgraph Registry["📋 Asset Registry  (core/asset_registry.py)"]
        H["AssetRegistry\nFutures: Metals, Energy, Equity Index,\nFX, Treasuries, Agriculture\nCrypto: BTC, ETH, SOL + more\nMicro + Full + Spot variants\nAsset.compute_position_size()\nAsset.dual_sizing()"]
    end

    %% ==================== SCHEDULER ====================
    subgraph Scheduler["⏰ ScheduleManager  (services/engine/scheduler.py)"]
        I1["EVENING  18:00–00:00 ET\nCME, Sydney, Tokyo,\nShanghai ORB sessions"]
        I2["PRE_MARKET  00:00–03:00 ET\nCOMPUTE_DAILY_FOCUS\nGROK_MORNING_BRIEF\nCHECK_NEWS_SENTIMENT 07:00 ET\nPREP_ALERTS"]
        I3["ACTIVE  03:00–12:00 ET\nFrankfurt, London, LN-NY,\nUS Open ORB + multi-type scans\nRUBY_RECOMPUTE every 5 min\nGROK_LIVE_UPDATE every 15 min\nCHECK_RISK_RULES + CHECK_NO_TRADE\nCHECK_SWING every 2 min\nCHECK_ORB_CME_SETTLE 14:00–15:30 ET\nPOSITION_CLOSE_WARNING 15:45 ET\nEOD_POSITION_CLOSE 16:00 ET"]
        I4["OFF_HOURS  12:00–18:00 ET\nHISTORICAL_BACKFILL\nRUN_OPTIMIZATION\nRUN_BACKTEST\nGENERATE_CHART_DATASET\nTRAIN_BREAKOUT_CNN\nDAILY_REPORT\nCHECK_NEWS_SENTIMENT_MIDDAY\nNEXT_DAY_PREP"]
    end

    %% ==================== PRE-MARKET ROUTINE ====================
    subgraph PreMarket["🌅 Pre-Market Pipeline  (00:00–03:00 ET)"]
        J["COMPUTE_DAILY_FOCUS\nPreMarketScorer: NATR, RVOL, gap,\nmomentum, catalyst scores\n(analysis/scorer.py)"]
        J --> K["DailyBiasAnalyzer + DailyPlanGenerator\n(trading/strategies/daily/bias_analyzer.py)\n(trading/strategies/daily/daily_plan.py)\
nSwingDetector candidates\n(trading/strategies/daily/swing_detector.py)"]
        K --> L["compute_daily_focus()\nConvictionStack multipliers:\nnews sentiment, crypto momentum,\nGrok brief, regime\n(services/engine/focus.py)"]
        L --> M["publish_focus_to_redis()\nengine:focus_assets\nengine:daily_plan → Redis"]
        M --> N["GROK_MORNING_BRIEF\ngrok_helper.py\nMacro context + daily plan\n→ dashboard chat cards"]
        N --> O["CHECK_NEWS_SENTIMENT  07:00 ET\nFinnhub + Alpha Vantage + VADER\nengine:news_sentiment:<SYM>  2h TTL\n(analysis/sentiment/news_sentiment.py)"]
    end

    %% ==================== RUBY SIGNAL ENGINE ====================
    subgraph Ruby["💎 Ruby Signal Engine  (services/engine/ruby_signal_engine.py)"]
        RB1["RubySignalEngine\nHMA channel + EMA bias\nORB / IB detection\nVWAP alignment\nATR14 stop sizing\nTP1/TP2/TP3 R-multiples\nRSI + squeeze bands"]
        RB1 --> RB2["RubySignal\nCompatible with PositionManager\nprocess_signal() interface\nPublished → Redis engine:ruby:<SYM>"]
        RB2 --> RB3["RUBY_RECOMPUTE  every 5 min\nhandle_ruby_recompute()\n(services/engine/handlers.py)\nScans focus assets with latest bars"]
    end

    %% ==================== BREAKOUT DETECTION CORE ====================
    subgraph BreakoutCore["🔍 Breakout Detection  (trading/strategies/rb/)"]
        P["13 BreakoutTypes\n(core/breakout_types.py)\nORB · PrevDay · InitialBalance\nConsolidation · Weekly · Monthly\nAsian · BollingerSqueeze · ValueArea\nInsideDay · GapRejection · PivotPoints · Fibonacci"]
        P --> Q["detect_range_breakout()\nrange_builders.py + breakout.py\ndetector.py facade\nRangeConfig per type"]
        Q --> R["run_quality_filters()\nNR7, pre-market range,\nsession window, lunch filter,\nVWAP confluence, MTF bias\n(services/engine/handlers.py)\n(analysis/breakout_filters.py)"]
        R --> S["build_cnn_tabular_features()\n25+ features: ATR trend, RVOL,\ndaily bias, session overlap,\ncrypto momentum, news score,\nregime, Ruby signal\n(services/engine/handlers.py)"]
        S --> T["HybridBreakoutCNN inference\nPyTorch tabular + type/asset embeddings\n→ LONG / SHORT / SKIP\n(analysis/ml/breakout_cnn.py)\nrun_cnn_inference()"]
    end

    %% ==================== ORB SESSION HANDLERS ====================
    subgraph ORBSessions["📡 ORB Session Checks  (services/engine/main.py)"]
        U1["CHECK_ORB_CME  18:00–20:00 ET\n(CME Globex re-open)"]
        U2["CHECK_ORB_SYDNEY  18:30–20:30 ET"]
        U3["CHECK_ORB_TOKYO  19:00–21:00 ET"]
        U4["CHECK_ORB_SHANGHAI  21:00–23:00 ET"]
        U5["CHECK_ORB_FRANKFURT  03:00–04:30 ET"]
        U6["CHECK_ORB_LONDON  03:00–05:00 ET"]
        U7["CHECK_ORB_LONDON_NY  08:00–10:00 ET"]
        U8["CHECK_ORB  09:30–11:00 ET  (US open)"]
        U9["CHECK_ORB_CME_SETTLE  14:00–15:30 ET\n(metals/energy settlement)"]
        U10["CHECK_ORB_CRYPTO_UTC0 / UTC12\nBTC/ETH/SOL crypto windows"]
        U11["CHECK_BREAKOUT_MULTI  every 2 min\nPDR · IB · Consolidation +\n9 additional types in parallel\n(handle_breakout_multi)\nSession assets via ORBSession\n(trading/strategies/rb/open/sessions.py)"]
    end

    %% ==================== SIGNAL QUALITY & CONFLUENCE ====================
    subgraph Quality["✅ Signal Quality & Confluence  (analysis/)"]
        V1["MTFAnalyzer / analyze_mtf()\nHTF bias + EMA alignment\ndivergence detection\n(analysis/mtf_analyzer.py)"]
        V2["RegimeFilter\nVolatility regime, trend strength\n(analysis/regime.py)"]
        V3["CryptoMomentumScore\n(analysis/cross_asset.py)\n(analysis/crypto_momentum.py)"]
        V4["CVD + VolumeProfile\n(analysis/cvd.py)\n(analysis/volume_profile.py)"]
        V5["ICT concepts\n(analysis/ict.py)\nFVG, Order Blocks, liquidity"]
        V6["WaveAnalysis\n(analysis/wave_analysis.py)\nEW structure"]
        V7["SignalQuality\n(analysis/signal_quality.py)\nvelocity, acceleration,\ntrend context, candle patterns"]
        V8["AssetFingerprint\n(analysis/asset_fingerprint.py)"]
    end

    %% ==================== NO-TRADE GUARD ====================
    subgraph NoTrade["🚫 No-Trade Guard  (services/engine/patterns.py)"]
        NT["evaluate_no_trade()\nNoTradeConditions:\nlow market data quality\nextreme volatility\ndaily loss limit hit\nconsecutive loss limit\nlate session / no setups\nsession ended\npublish_no_trade_alert() → Redis"]
    end

    %% ==================== RISK MANAGEMENT ====================
    subgraph Risk["🛡️ Risk Management  (services/engine/risk.py + live_risk.py)"]
        W["RiskManager.can_enter_trade()\nDaily P&L gate\nConsecutive loss limit\nOpen position cap\nOvernight risk check\n(services/engine/risk.py)"]
        W --> X["LiveRiskState\nPer-asset position snapshots\nDynamic sizing\nHealth score\ncompute_live_risk()\n(services/engine/live_risk.py)"]
        X --> Y["LiveRiskPublisher\nTicks every 5s\nPublishes → Redis engine:live_risk"]
    end

    %% ==================== POSITION & ORDER MANAGEMENT ====================
    subgraph Positions["📈 Position & Order Management  (services/engine/position_manager.py)"]
        Z["PositionManager.process_signal()\nMicroPosition state machine\nBracketPhase: ENTRY→TP1→BE→TRAIL→TP3\nSAR always-in reversal logic\nPyramid scaling via get_next_pyramid_level()"]
        Z --> AA["3-Phase Bracket\nTP1 → move stop to BE\nEMA9 trailing stop → TP3\n_update_bracket_phase()\n_check_stop_hit() / _check_tp3_hit()"]
        AA --> AB["CopyTrader.execute_order_commands()\nRithmic multi-account copy\nCompliance checklist + RollingRateCounter\n(services/engine/copy_trader.py)"]
        AB --> AC["RithmicAccountManager\nOrder placement on prop accounts\nresolve_front_month()\nEOD hard close 16:00 ET\n(integrations/rithmic_client.py)"]
    end

    %% ==================== SWING DETECTOR ====================
    subgraph Swing["📊 Swing Detector  (services/engine/swing.py)"]
        AD["CHECK_SWING  every 2 min  03:00–15:30 ET\ntick_swing_detector()\nScans daily-plan swing candidates\nSwingState: PENDING→ACTIVE→CLOSED"]
        AD --> AE["Pullback / Breakout / Gap entries\n15m + 5m bar fetch\nManual: accept_swing_signal()\nignore_swing_signal()\nclose_swing_position()\nmove_stop_to_breakeven()\nvia dashboard actions"]
    end

    %% ==================== INDICATORS ====================
    subgraph Indicators["📐 Indicator Library  (indicators/)"]
        IND1["IndicatorManager + Registry\n(indicators/manager.py)\n(indicators/registry.py)\n(indicators/factory.py)"]
        IND1 --> IND2["Trend: EMA, HMA, VWAP, etc.\n(indicators/trend/)\nMomentum: RSI, MACD, etc.\n(indicators/momentum/)\nVolume: OBV, CVD, etc.\n(indicators/volume/)\nOther: ATR, BB, KC, etc.\n(indicators/other/)"]
        IND1 --> IND3["CandlePatterns\n(indicators/candle_patterns.py)\nAreasOfInterest\n(indicators/areas_of_interest.py)\nMarketTiming\n(indicators/market_timing.py)"]
    end

    %% ==================== MODEL LAYER ====================
    subgraph Models["🧠 Model Layer  (model/)"]
        MOD1["ModelService + ModelRegistry\n(model/service.py)\n(model/registry.py)\n(model/factory.py)"]
        MOD1 --> MOD2["Deep: CNN, LSTM, Transformer, TFT, NN\n(model/deep/)\nML: XGBoost, LightGBM, CatBoost,\nLogistic, Gaussian, Polynomial\n(model/ml/)"]
        MOD1 --> MOD3["Statistical: ARIMA, GARCH, HMM,\nProphet, Bayesian\n(model/statistical/)\nEnsemble\n(model/ensemble/)"]
        MOD1 --> MOD4["Prediction: PredictionManager,\nsingle/multi generators\n(model/prediction/)\nEvaluation: cross_val, metrics\n(model/evaluation/)"]
    end

    %% ==================== TRAINING PIPELINE ====================
    subgraph Training["🏋️ Training Pipeline  [Docker: trainer]"]
        AF["OFF_HOURS triggers\nGENERATE_CHART_DATASET\n→ DatasetGenerator\nRBSimulator\n(services/training/rb_simulator.py)"]
        AF --> AG["dataset_generator.py\n180-day lookback\n13 breakout types × all assets\n_build_row() 25+ features\nsplit_dataset() + validate_dataset()"]
        AG --> AH["TRAIN_BREAKOUT_CNN\nHybridBreakoutCNN\nPyTorch + Optuna walk-forward\ntrain_model() + evaluate_model()\n(analysis/ml/breakout_cnn.py)\ntrainer_server.py  FastAPI :8001"]
        AH --> AI["Champion promotion\nbreakout_cnn_best.pt\nbreakout_cnn_best_meta.json\nfeature_contract.json\n→ models/ directory\n_archive_champion()"]
        AI --> AJ["ModelWatcher hot-reload\nwatchdog inotify / polling fallback\ninvalidate_model_cache()\nPublishes model_reloaded → Redis\n(services/engine/model_watcher.py)"]
    end

    %% ==================== BACKFILL & OFF-HOURS ====================
    subgraph OffHours["⚙️ Off-Hours  (12:00–18:00 ET)"]
        AK["HISTORICAL_BACKFILL\nrun_backfill()\nWarms Redis + Postgres\nfrom MassiveAPI / Kraken / yfinance\n(services/engine/backfill.py)"]
        AL["RUN_OPTIMIZATION\nOptuna nightly study\nwalk-forward 30–90 days\n(trading/strategies/backtesting)"]
        AM["RUN_BACKTEST\nP&L + win-rate stats"]
        AN["DAILY_REPORT\n_handle_daily_report()\nP&L, trades, signals,\nGrok review → email + Discord\n(services/engine/main.py)"]
        AO["POSITION_CLOSE_WARNING 15:45 ET\nEOD_POSITION_CLOSE 16:00 ET\nRithmic cancel_all + exit_all\nhard safety net"]
    end

    %% ==================== CHARTING SERVICE ====================
    subgraph Charting["📉 Charting Service  [Docker: charting]"]
        CH["nginx :3001\nLightweight Charts + custom JS\nServes OHLCV candle charts\nproxied via web service\n(docker/charting/)"]
    end

    %% ==================== KRAKEN PORTFOLIO ====================
    subgraph Kraken["💰 Kraken Crypto Portfolio  24/7"]
        AP["KrakenDataProvider\nREST OHLCV + ticker\nPortfolio balance queries\n(integrations/kraken_client.py)"]
        AP --> AQ["Kraken WebSocket feed\nReal-time OHLC + trades\nBars pushed → Redis on close\n(services/data/main.py lifespan)"]
        AQ --> AR["Crypto ORB sessions\nCHECK_ORB_CRYPTO_UTC0\nCHECK_ORB_CRYPTO_UTC12\nSame 13-type detection pipeline"]
    end

    %% ==================== PINE SCRIPT GENERATOR ====================
    subgraph Pine["🌲 Pine Script Generator  (integrations/pine/)"]
        PI["PineScriptGenerator\nGenerates TradingView Pine modules\nparams.yaml driven\nOutput served via /pine/* API\n(integrations/pine/generate.py)"]
    end

    %% ==================== DASHBOARD & WEB ====================
    subgraph Web["🖥️ Web / Dashboard  [Docker: web + data]"]
        AS["FastAPI web proxy  :8080\n(services/web/main.py)\nHTMX + SSE dashboard"]
        AS --> AT["Live Risk strip\nengine:live_risk → Redis → SSE\nPer-asset P&L + health score\n(/live_risk/*)"]
        AS --> AU["Focus cards\nDaily plan + conviction scores\nManual swing accept/ignore\n(/plan/*)"]
        AS --> AV["Grok chat + review cards\nGrok live update every N min\n(/chat/* + /grok/*)"]
        AS --> AW["Ruby signal panel\n(/ruby/signals + /ruby/status)\nRubySignalEngine per-asset state"]
        AS --> AX["CNN panel + model info\nTrainer redirect + log stream\nChart dataset viewer\n(/cnn/* + /trainer/*)"]
        AS --> AY["Journal + audit history\nTrade grading, ORB history,\nRB history, journal pages\n(/journal/* + /audit/*)"]
        AS --> AZ2["Settings panel\nServices, features, risk config,\nAPI keys, Rithmic accounts\n(/settings/*)"]
        AS --> BA2["Tasks panel\nTask CRUD + GitHub push\n(/tasks/*)"]
        AS --> BB2["Copy trade panel\nRate alerts, pyramid, focus,\naccounts HTML\n(/copy_trade/*)"]
        AS --> BC2["Trading settings\nTest Rithmic/MassiveAPI/Kraken\n(/trading/*)"]
    end

    %% ==================== ALERTS ====================
    subgraph Alerts["🔔 Alerts  (core/alerts.py)"]
        AZ["Discord smart gate\nMaster toggle\nFocus-only filter\nLive breakout events\nNo-trade alerts\nGap alerts + daily report\nRisk events"]
    end

    %% ==================== MONITORING ====================
    subgraph Monitoring["📈 Monitoring  [Docker: prometheus + grafana]"]
        BA["Prometheus :9090\nScrapes /metrics from\ndata :8000, engine,\ntrainer :8001"]
        BA --> BB["Grafana :3000\nDashboards: P&L, signal\nquality, CNN accuracy,\nrisk utilisation"]
    end

    %% ==================== FLOW CONNECTIONS ====================

    %% External → Data layer
    A1 -->|"OHLCV bars"| B
    A2 -->|"REST OHLCV + ticker"| B
    A2 -->|"WebSocket ticks"| F
    A3 -->|"sentiment snapshots"| G
    A4 -->|"news scores"| O
    A5 -->|"positions / fills"| Y
    A6 -->|"morning brief"| N
    A6 -->|"live updates"| AV

    %% Data ↔ Registry
    B --> H
    H -->|"asset specs, tickers, position sizing"| B

    %% Data → Scheduler (Redis commands)
    C -->|"Redis commands\nforce_retrain, retrain_cnn, etc."| Scheduler

    %% Scheduler → Session flows
    Scheduler --> PreMarket
    Scheduler --> ORBSessions
    Scheduler --> Swing
    Scheduler --> OffHours
    Scheduler --> Ruby

    %% Pre-market feeds into live detection
    M -->|"focus assets + daily plan"| BreakoutCore
    M -->|"swing candidates"| Swing

    %% ORB session handlers → Breakout core
    ORBSessions -->|"fetch_bars_1m + symbol list\nhandle_orb_check() / handle_breakout_check()\n(services/engine/handlers.py)"| BreakoutCore

    %% Breakout core → Quality filters
    T -->|"CNN verdict + tabular features"| Quality

    %% Quality → No-Trade guard
    Quality -->|"enriched signal"| NT

    %% No-Trade guard → Risk gate
    NT -->|"no active block"| W

    %% Ruby → Risk gate (parallel signal path)
    RB3 -->|"RubySignal"| W

    %% Risk → Position
    W -->|"can_enter=True"| Z

    %% Swing → Risk
    AE -->|"swing signal"| W

    %% Indicators used by Quality + Ruby
    Indicators -->|"computed series"| Quality
    Indicators -->|"ATR, VWAP, EMA, squeeze"| Ruby

    %% Model layer used in training + inference
    Models -->|"deep/ml/statistical models"| Training

    %% Training pipeline
    OffHours --> AF
    AJ -->|"hot-reload new model weights"| T

    %% Kraken integrated into detection
    Kraken --> BreakoutCore

    %% Charting served via web proxy
    AS -->|"proxied candle charts\n/charts/*"| CH

    %% Pine script generation
    AS -->|"proxied /pine/* API"| Pine

    %% Publish results → Redis → Dashboard
    Y -->|"live_risk state"| C
    Z -->|"positions + PnL + OrderCommands"| C
    BreakoutCore -->|"breakout results\npersisted to Postgres"| D
    RB3 -->|"RubySignal → Redis\nengine:ruby:<SYM>"| C
    C -->|"SSE stream → HTMX dashboard"| Web

    %% Alerts wired to events
    BreakoutCore -->|"signal alerts"| AZ
    AN -->|"daily report"| AZ
    Y -->|"risk events"| AZ
    NT -->|"no-trade alerts"| AZ
    AZ -->|"Discord webhooks"| External

    %% Monitoring scrapes
    E -->|"/metrics  :8000"| BA
    AH -->|"/metrics  :8001"| BA
"""

# =============================================================================
# SCRIPT LOGIC
# =============================================================================


def generate_mermaid_files() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "docs"
    output_dir.mkdir(exist_ok=True)

    # 1. Save raw Mermaid markdown
    mmd_path = output_dir / "futures_logic_flow.mmd"
    mmd_path.write_text(MERMAID, encoding="utf-8")
    print(f"✅ Saved Mermaid source: {mmd_path}")

    # 2. Render SVG via kroki.io POST (handles large diagrams that exceed URL limits).
    #    SVG is the primary output — it scales perfectly and opens in any browser.
    #    A PNG render is also attempted; it may fail on the free kroki.io tier for
    #    very large diagrams due to puppeteer memory limits, which is non-fatal.
    svg_path = output_dir / "futures_logic_flow.svg"
    png_path = output_dir / "futures_logic_flow.png"

    # --- SVG (primary) ---
    try:
        response = requests.post(
            "https://kroki.io/mermaid/svg",
            json={"diagram_source": MERMAID},
            timeout=60,
        )
        response.raise_for_status()
        svg_path.write_bytes(response.content)
        print(f"✅ Rendered SVG image (kroki.io): {svg_path}")
        print("   Open in any browser, or paste the .mmd into https://mermaid.live")
    except Exception as svg_err:
        print(f"⚠️  Could not render SVG (no internet or API issue): {svg_err}")
        print("   Just open the .mmd file in https://mermaid.live instead")

    # --- PNG (best-effort, may time out for large diagrams on free tier) ---
    try:
        response = requests.post(
            "https://kroki.io/mermaid/png",
            json={"diagram_source": MERMAID},
            timeout=90,
        )
        response.raise_for_status()
        png_path.write_bytes(response.content)
        print(f"✅ Rendered PNG image (kroki.io): {png_path}")
    except Exception as png_err:
        print(f"ℹ️  PNG render skipped (diagram too large for free kroki.io tier): {png_err}")
        print(f"   Use the SVG instead: {svg_path}")


if __name__ == "__main__":
    print("🚀 Generating Futures Trading System Logic Flow (v7)...\n")
    generate_mermaid_files()
    print("\nDone! Check the 'docs/' folder.")
