"""
Account configurations, contract specifications, asset mappings,
and database helpers with OPEN/CLOSED status tracking.

Supports dual-database backends:
  - PostgreSQL via DATABASE_URL (production / Docker)
  - SQLite via DB_PATH (local dev / tests)

The active backend is chosen automatically at module load time:
  - If DATABASE_URL is set and starts with "postgresql", use Postgres.
  - Otherwise, fall back to SQLite at DB_PATH.

All CRUD functions use the same interface regardless of backend.
"""

import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

_EST = ZoneInfo("America/New_York")

logger = logging.getLogger("models")

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "futures_journal.db")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Detect which backend to use
_USE_POSTGRES = DATABASE_URL.startswith("postgresql")

# SQLAlchemy engine (lazy-initialised for Postgres)
_sa_engine = None

# ---------------------------------------------------------------------------
# Daily journal schema — simple end-of-day P&L entries
# ---------------------------------------------------------------------------
_SCHEMA_DAILY_JOURNAL_SQLITE = """
CREATE TABLE IF NOT EXISTS daily_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT    NOT NULL UNIQUE,
    account_size    INTEGER NOT NULL,
    gross_pnl       REAL    NOT NULL DEFAULT 0.0,
    net_pnl         REAL    NOT NULL DEFAULT 0.0,
    commissions     REAL    NOT NULL DEFAULT 0.0,
    num_contracts   INTEGER DEFAULT 0,
    instruments     TEXT    DEFAULT '',
    notes           TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
"""

_SCHEMA_DAILY_JOURNAL_PG = """
CREATE TABLE IF NOT EXISTS daily_journal (
    id              SERIAL PRIMARY KEY,
    trade_date      TEXT    NOT NULL UNIQUE,
    account_size    INTEGER NOT NULL,
    gross_pnl       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    net_pnl         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    commissions     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    num_contracts   INTEGER DEFAULT 0,
    instruments     TEXT    DEFAULT '',
    notes           TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Audit event tables — persistent storage for risk blocks & ORB detections
# ---------------------------------------------------------------------------

_SCHEMA_RISK_EVENTS_SQLITE = """
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    symbol          TEXT    NOT NULL DEFAULT '',
    side            TEXT    NOT NULL DEFAULT '',
    reason          TEXT    NOT NULL DEFAULT '',
    daily_pnl       REAL    NOT NULL DEFAULT 0.0,
    open_trades     INTEGER NOT NULL DEFAULT 0,
    account_size    INTEGER NOT NULL DEFAULT 0,
    risk_pct        REAL    NOT NULL DEFAULT 0.0,
    session         TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_re_timestamp ON risk_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_re_event_type ON risk_events (event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_re_symbol ON risk_events (symbol, timestamp);
"""

_SCHEMA_RISK_EVENTS_PG = """
CREATE TABLE IF NOT EXISTS risk_events (
    id              SERIAL PRIMARY KEY,
    timestamp       TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    symbol          TEXT    NOT NULL DEFAULT '',
    side            TEXT    NOT NULL DEFAULT '',
    reason          TEXT    NOT NULL DEFAULT '',
    daily_pnl       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    open_trades     INTEGER NOT NULL DEFAULT 0,
    account_size    INTEGER NOT NULL DEFAULT 0,
    risk_pct        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    session         TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_re_timestamp ON risk_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_re_event_type ON risk_events (event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_re_symbol ON risk_events (symbol, timestamp);
"""

_SCHEMA_ORB_EVENTS_SQLITE = """
CREATE TABLE IF NOT EXISTS orb_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    or_high         REAL    NOT NULL DEFAULT 0.0,
    or_low          REAL    NOT NULL DEFAULT 0.0,
    or_range        REAL    NOT NULL DEFAULT 0.0,
    atr_value       REAL    NOT NULL DEFAULT 0.0,
    breakout_detected INTEGER NOT NULL DEFAULT 0,
    direction       TEXT    NOT NULL DEFAULT '',
    trigger_price   REAL    NOT NULL DEFAULT 0.0,
    long_trigger    REAL    NOT NULL DEFAULT 0.0,
    short_trigger   REAL    NOT NULL DEFAULT 0.0,
    bar_count       INTEGER NOT NULL DEFAULT 0,
    session         TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    breakout_type   TEXT    NOT NULL DEFAULT 'ORB',
    mtf_score       REAL,
    macd_slope      REAL,
    divergence      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_orb_timestamp ON orb_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_orb_symbol ON orb_events (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_orb_breakout_type ON orb_events (breakout_type, timestamp);
"""

# ---------------------------------------------------------------------------
# Migration: add new columns to existing orb_events tables
# ---------------------------------------------------------------------------
_MIGRATE_ORB_EVENTS_ADD_COLS_SQLITE = """
ALTER TABLE orb_events ADD COLUMN breakout_type TEXT NOT NULL DEFAULT 'ORB';
ALTER TABLE orb_events ADD COLUMN mtf_score REAL;
ALTER TABLE orb_events ADD COLUMN macd_slope REAL;
ALTER TABLE orb_events ADD COLUMN divergence TEXT NOT NULL DEFAULT '';
"""

_MIGRATE_ORB_EVENTS_ADD_COLS_PG = [
    "ALTER TABLE orb_events ADD COLUMN IF NOT EXISTS breakout_type TEXT NOT NULL DEFAULT 'ORB'",
    "ALTER TABLE orb_events ADD COLUMN IF NOT EXISTS mtf_score DOUBLE PRECISION",
    "ALTER TABLE orb_events ADD COLUMN IF NOT EXISTS macd_slope DOUBLE PRECISION",
    "ALTER TABLE orb_events ADD COLUMN IF NOT EXISTS divergence TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_orb_breakout_type ON orb_events (breakout_type, timestamp)",
]

_SCHEMA_ORB_EVENTS_PG = """
CREATE TABLE IF NOT EXISTS orb_events (
    id              SERIAL PRIMARY KEY,
    timestamp       TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    or_high         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    or_low          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    or_range        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    atr_value       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    breakout_detected INTEGER NOT NULL DEFAULT 0,
    direction       TEXT    NOT NULL DEFAULT '',
    trigger_price   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    long_trigger    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    short_trigger   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    bar_count       INTEGER NOT NULL DEFAULT 0,
    session         TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    breakout_type   TEXT    NOT NULL DEFAULT 'ORB',
    mtf_score       DOUBLE PRECISION,
    macd_slope      DOUBLE PRECISION,
    divergence      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_orb_timestamp ON orb_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_orb_symbol ON orb_events (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_orb_breakout_type ON orb_events (breakout_type, timestamp);
"""


# ---------------------------------------------------------------------------
# Contract mode: "micro" (default) or "full"
# Set via environment variable CONTRACT_MODE or toggle at runtime.
# ---------------------------------------------------------------------------
CONTRACT_MODE = os.getenv("CONTRACT_MODE", "micro").lower()

# ---------------------------------------------------------------------------
# Account profiles – 50k is 1/3 of 150k, 100k is 2/3
# ---------------------------------------------------------------------------
ACCOUNT_PROFILES: dict[str, dict[str, Any]] = {
    "50k": {
        "size": 50_000,
        "risk_pct": 0.01,
        "risk_dollars": 500,
        "max_contracts": 2,
        "max_contracts_micro": 10,
        "soft_stop": -500,
        "hard_stop": -750,
        "eod_dd": 1_500,
        "label": "$50k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 1-2 full / 10 micro contracts on $50k (25% rule). Daily Loss Removed: $1,500."
        ),
    },
    "100k": {
        "size": 100_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_000,
        "max_contracts": 3,
        "max_contracts_micro": 20,
        "soft_stop": -1_000,
        "hard_stop": -1_500,
        "eod_dd": 3_000,
        "label": "$100k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 2-3 full / 20 micro contracts on $100k (25% rule). Daily Loss Removed: $3,000."
        ),
    },
    "150k": {
        "size": 150_000,
        "risk_pct": 0.01,
        "risk_dollars": 1_500,
        "max_contracts": 4,
        "max_contracts_micro": 30,
        "soft_stop": -1_500,
        "hard_stop": -2_250,
        "eod_dd": 4_500,
        "label": "$150k TakeProfit Trader",
        "playbook_note": (
            "TPT PLAYBOOK: Max 3-4 full / 30 micro contracts on $150k (25% rule). Daily Loss Removed: $4,500."
        ),
    },
}

# ---------------------------------------------------------------------------
# Contract specifications — Full-size CME contracts
# ---------------------------------------------------------------------------
FULL_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    "Gold": {"ticker": "GC=F", "point": 100, "tick": 0.10, "margin": 11_000},
    "Silver": {"ticker": "SI=F", "point": 5_000, "tick": 0.005, "margin": 9_000},
    "Copper": {"ticker": "HG=F", "point": 25_000, "tick": 0.0005, "margin": 6_000},
    "Crude Oil": {"ticker": "CL=F", "point": 1_000, "tick": 0.01, "margin": 7_000},
    "Natural Gas": {"ticker": "NG=F", "point": 10_000, "tick": 0.001, "margin": 3_500},
    "S&P": {"ticker": "ES=F", "point": 50, "tick": 0.25, "margin": 12_000},
    "Nasdaq": {"ticker": "NQ=F", "point": 20, "tick": 0.25, "margin": 17_000},
    "Russell 2000": {"ticker": "RTY=F", "point": 50, "tick": 0.10, "margin": 8_000},
    "Dow Jones": {"ticker": "YM=F", "point": 5, "tick": 1.0, "margin": 9_000},
    # FX futures (full-size CME)
    "Euro FX": {"ticker": "6E=F", "point": 125_000, "tick": 0.00005, "margin": 2_800},
    "British Pound": {"ticker": "6B=F", "point": 62_500, "tick": 0.0001, "margin": 2_600},
    "Japanese Yen": {"ticker": "6J=F", "point": 12_500_000, "tick": 0.0000005, "margin": 2_400},
    "Australian Dollar": {"ticker": "6A=F", "point": 100_000, "tick": 0.0001, "margin": 1_800},
    "Canadian Dollar": {"ticker": "6C=F", "point": 100_000, "tick": 0.0001, "margin": 1_600},
    "Swiss Franc": {"ticker": "6S=F", "point": 125_000, "tick": 0.0001, "margin": 3_000},
    # Interest rate futures (CBOT)
    "10Y T-Note": {"ticker": "ZN=F", "point": 1_000, "tick": 0.015625, "margin": 1_800},
    "30Y T-Bond": {"ticker": "ZB=F", "point": 1_000, "tick": 0.03125, "margin": 3_200},
    # Agricultural futures (CBOT)
    "Corn": {"ticker": "ZC=F", "point": 50, "tick": 0.25, "margin": 1_200},
    "Soybeans": {"ticker": "ZS=F", "point": 50, "tick": 0.25, "margin": 2_200},
    "Wheat": {"ticker": "ZW=F", "point": 50, "tick": 0.25, "margin": 1_700},
    # Crypto (full-size CME)
    "Bitcoin": {"ticker": "BTC=F", "point": 5, "tick": 5.0, "margin": 80_000},
    "Ether": {"ticker": "ETH=F", "point": 50, "tick": 0.25, "margin": 6_500},
}

# ---------------------------------------------------------------------------
# Contract specifications — Micro CME contracts
# Micro contracts are 1/10 of full size (except Silver 1/5, FX 1/10).
# These give more granularity for position sizing and scaling.
# ---------------------------------------------------------------------------
MICRO_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    # ── Metals ──────────────────────────────────────────────────────────────
    "Gold": {
        "ticker": "MGC=F",
        "data_ticker": "MGC=F",
        "point": 10,
        "tick": 0.10,
        "margin": 1_100,
    },
    "Silver": {
        "ticker": "SIL=F",
        "data_ticker": "SI=F",
        "point": 1_000,
        "tick": 0.005,
        "margin": 1_800,
    },
    "Copper": {
        "ticker": "MHG=F",
        "data_ticker": "HG=F",
        "point": 2_500,
        "tick": 0.0005,
        "margin": 600,
    },
    # ── Energy ──────────────────────────────────────────────────────────────
    "Crude Oil": {
        "ticker": "MCL=F",
        "data_ticker": "CL=F",
        "point": 100,
        "tick": 0.01,
        "margin": 700,
    },
    "Natural Gas": {
        # CME Micro Natural Gas (MNG) — 1/10 of the standard NG contract
        "ticker": "MNG=F",
        "data_ticker": "NG=F",
        "point": 1_000,
        "tick": 0.001,
        "margin": 350,
    },
    # ── Equity index ────────────────────────────────────────────────────────
    "S&P": {
        "ticker": "MES=F",
        "data_ticker": "ES=F",
        "point": 5,
        "tick": 0.25,
        "margin": 1_500,
    },
    "Nasdaq": {
        "ticker": "MNQ=F",
        "data_ticker": "NQ=F",
        "point": 2,
        "tick": 0.25,
        "margin": 2_100,
    },
    "Russell 2000": {
        "ticker": "M2K=F",
        "data_ticker": "RTY=F",
        "point": 5,
        "tick": 0.10,
        "margin": 1_200,
    },
    "Dow Jones": {
        "ticker": "MYM=F",
        "data_ticker": "YM=F",
        "point": 0.5,
        "tick": 1.0,
        "margin": 1_100,
    },
    # ── FX futures ──────────────────────────────────────────────────────────
    # M6E / M6B are genuine CME Micro FX contracts (1/10 of standard size).
    # The remaining pairs use the standard contract size alongside micros for
    # session-based ORB scanning.
    "Euro FX": {
        "ticker": "M6E=F",
        "data_ticker": "6E=F",
        "point": 12_500,
        "tick": 0.0001,
        "margin": 280,
    },
    "British Pound": {
        "ticker": "M6B=F",
        "data_ticker": "6B=F",
        "point": 6_250,
        "tick": 0.0001,
        "margin": 260,
    },
    "Japanese Yen": {
        "ticker": "6J=F",
        "data_ticker": "6J=F",
        "point": 12_500_000,
        "tick": 0.0000005,
        "margin": 2_400,
    },
    "Australian Dollar": {
        "ticker": "6A=F",
        "data_ticker": "6A=F",
        "point": 100_000,
        "tick": 0.0001,
        "margin": 1_800,
    },
    "Canadian Dollar": {
        "ticker": "6C=F",
        "data_ticker": "6C=F",
        "point": 100_000,
        "tick": 0.0001,
        "margin": 1_600,
    },
    "Swiss Franc": {
        "ticker": "6S=F",
        "data_ticker": "6S=F",
        "point": 125_000,
        "tick": 0.0001,
        "margin": 3_000,
    },
    # ── Interest rate futures (CBOT) ─────────────────────────────────────────
    # No standard CME micro exists for ZN/ZB; we use full-size contracts
    # here since margins are already relatively small and they're important
    # for macro regime context.
    "10Y T-Note": {
        "ticker": "ZN=F",
        "data_ticker": "ZN=F",
        "point": 1_000,
        "tick": 0.015625,
        "margin": 1_800,
    },
    "30Y T-Bond": {
        "ticker": "ZB=F",
        "data_ticker": "ZB=F",
        "point": 1_000,
        "tick": 0.03125,
        "margin": 3_200,
    },
    # ── Agricultural futures (CBOT) ──────────────────────────────────────────
    "Corn": {
        "ticker": "ZC=F",
        "data_ticker": "ZC=F",
        "point": 50,
        "tick": 0.25,
        "margin": 1_200,
    },
    "Soybeans": {
        "ticker": "ZS=F",
        "data_ticker": "ZS=F",
        "point": 50,
        "tick": 0.25,
        "margin": 2_200,
    },
    "Wheat": {
        "ticker": "ZW=F",
        "data_ticker": "ZW=F",
        "point": 50,
        "tick": 0.25,
        "margin": 1_700,
    },
    # ── Crypto ──────────────────────────────────────────────────────────────
    "Micro Bitcoin": {
        "ticker": "MBT=F",
        "data_ticker": "MBT=F",
        "point": 0.1,
        "tick": 5.0,
        "margin": 8_000,
    },
    "Micro Ether": {
        "ticker": "MET=F",
        "data_ticker": "MET=F",
        "point": 0.1,
        "tick": 0.25,
        "margin": 700,
    },
}

# ---------------------------------------------------------------------------
# Kraken Crypto Contract Specifications
# ---------------------------------------------------------------------------
# Spot crypto pairs sourced from Kraken exchange via REST + WebSocket.
# These use a KRAKEN: prefix internally so the cache layer and engine can
# distinguish them from CME futures tickers.
#
# "point" = value of a $1 move per 1 unit (spot crypto = 1.0).
# "tick"  = minimum price increment on Kraken.
# "margin"= notional margin to risk-size 1 unit (informal, for dashboard display).
#
# Enable/disable Kraken crypto in the pipeline via ENABLE_KRAKEN_CRYPTO env var.
# ---------------------------------------------------------------------------
ENABLE_KRAKEN_CRYPTO = os.getenv("ENABLE_KRAKEN_CRYPTO", "1").strip().lower() in ("1", "true", "yes")

KRAKEN_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    "BTC/USD": {
        "ticker": "KRAKEN:XBTUSD",
        "data_ticker": "KRAKEN:XBTUSD",
        "point": 1.0,
        "tick": 0.1,
        "margin": 5_000,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "ETH/USD": {
        "ticker": "KRAKEN:ETHUSD",
        "data_ticker": "KRAKEN:ETHUSD",
        "point": 1.0,
        "tick": 0.01,
        "margin": 500,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "SOL/USD": {
        "ticker": "KRAKEN:SOLUSD",
        "data_ticker": "KRAKEN:SOLUSD",
        "point": 1.0,
        "tick": 0.001,
        "margin": 50,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "LINK/USD": {
        "ticker": "KRAKEN:LINKUSD",
        "data_ticker": "KRAKEN:LINKUSD",
        "point": 1.0,
        "tick": 0.001,
        "margin": 25,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "AVAX/USD": {
        "ticker": "KRAKEN:AVAXUSD",
        "data_ticker": "KRAKEN:AVAXUSD",
        "point": 1.0,
        "tick": 0.001,
        "margin": 30,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "DOT/USD": {
        "ticker": "KRAKEN:DOTUSD",
        "data_ticker": "KRAKEN:DOTUSD",
        "point": 1.0,
        "tick": 0.0001,
        "margin": 15,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "ADA/USD": {
        "ticker": "KRAKEN:ADAUSD",
        "data_ticker": "KRAKEN:ADAUSD",
        "point": 1.0,
        "tick": 0.00001,
        "margin": 10,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "POL/USD": {
        "ticker": "KRAKEN:POLUSD",
        "data_ticker": "KRAKEN:POLUSD",
        "point": 1.0,
        "tick": 0.0001,
        "margin": 10,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
    "XRP/USD": {
        "ticker": "KRAKEN:XRPUSD",
        "data_ticker": "KRAKEN:XRPUSD",
        "point": 1.0,
        "tick": 0.0001,
        "margin": 10,
        "exchange": "kraken",
        "asset_class": "crypto",
    },
}

# ---------------------------------------------------------------------------
# Active contract specs — selected by CONTRACT_MODE env var
# ---------------------------------------------------------------------------
CONTRACT_SPECS = MICRO_CONTRACT_SPECS if CONTRACT_MODE == "micro" else FULL_CONTRACT_SPECS

# Convenience: name → data ticker (for data fetching).
# When data_ticker differs from ticker, data is fetched using data_ticker.
# Gold uses MGC=F directly to avoid front-month mismatch between GC and MGC.
ASSETS: dict[str, str] = {name: str(spec.get("data_ticker", spec["ticker"])) for name, spec in CONTRACT_SPECS.items()}

# Merge in Kraken crypto pairs when enabled
if ENABLE_KRAKEN_CRYPTO:
    ASSETS.update({name: str(spec["data_ticker"]) for name, spec in KRAKEN_CONTRACT_SPECS.items()})

# Reverse lookup: data ticker → name
TICKER_TO_NAME: dict[str, str] = {
    str(spec.get("data_ticker", spec["ticker"])): name for name, spec in CONTRACT_SPECS.items()
}
if ENABLE_KRAKEN_CRYPTO:
    TICKER_TO_NAME.update({str(spec["data_ticker"]): name for name, spec in KRAKEN_CONTRACT_SPECS.items()})

# ---------------------------------------------------------------------------
# Flat ticker sets — useful for quick membership checks
# ---------------------------------------------------------------------------
# All micro-contract tickers (for filtering signals to tradeable instruments)
MICRO_TICKERS: frozenset[str] = frozenset(
    str(spec.get("data_ticker", spec["ticker"])) for spec in MICRO_CONTRACT_SPECS.values()
)

# FX futures tickers subset (data tickers)
FX_TICKERS: frozenset[str] = frozenset({"6E=F", "6B=F", "6J=F", "6A=F", "6C=F", "6S=F"})

# Interest rate futures tickers subset
RATES_TICKERS: frozenset[str] = frozenset({"ZN=F", "ZB=F"})

# Agricultural futures tickers subset
AG_TICKERS: frozenset[str] = frozenset({"ZC=F", "ZS=F", "ZW=F"})

# Kraken crypto tickers subset (data tickers)
CRYPTO_TICKERS: frozenset[str] = frozenset(str(spec["data_ticker"]) for spec in KRAKEN_CONTRACT_SPECS.values())

# Overnight-session relevant tickers (traded when US equity markets are closed)
OVERNIGHT_TICKERS: frozenset[str] = frozenset(
    {
        # Metals
        "MGC=F",
        "SIL=F",
        "MHG=F",
        # Energy
        "CL=F",
        "NG=F",
        # FX
        "6E=F",
        "6B=F",
        "6J=F",
        "6A=F",
        "6C=F",
        "6S=F",
        # CME Crypto futures
        "MBT=F",
        "MET=F",
        # Kraken spot crypto (24/7 markets — always "overnight")
        *(str(spec["data_ticker"]) for spec in KRAKEN_CONTRACT_SPECS.values()),
    }
)

# ---------------------------------------------------------------------------
# Watchlists — curated asset tiers for strategy & scanning
# ---------------------------------------------------------------------------
# CORE_WATCHLIST: 5 assets actively traded with 1-lot micro stop-and-reverse.
#   Selected for: tight spreads, deep liquidity, clean breakout patterns,
#   sufficient ATR for meaningful R-multiples, low margin requirement.
#   Total margin for 5 positions ≈ $5,680 (11.4% of $50K account).
CORE_WATCHLIST: dict[str, str] = {
    "Gold": "MGC=F",  # Best breakout patterns, 23h trading, $10/pt, $1,100 margin
    "Crude Oil": "MCL=F",  # Clean ORB setups, strong session opens, $100/pt, $700 margin
    "S&P": "MES=F",  # Highest liquidity micro, trends cleanly intraday, $5/pt, $1,500 margin
    "Nasdaq": "MNQ=F",  # Highest ATR index micro, big moves = big R, $2/pt, $2,100 margin
    "Euro FX": "M6E=F",  # Best FX micro for breakouts, London open textbook ORB, $280 margin
}

# EXTENDED_WATCHLIST: 5 additional assets for signal scanning & CNN predictions.
#   Not actively traded with stop-and-reverse unless core assets aren't setting up.
#   Still generate breakout signals, CNN inferences, and dashboard alerts.
EXTENDED_WATCHLIST: dict[str, str] = {
    "Silver": "SIL=F",  # Correlated with Gold; trade when MGC is choppy but SIL trends
    "Russell 2000": "M2K=F",  # Small-cap divergence plays; trade when MES/MNQ are range-bound
    "British Pound": "M6B=F",  # Trade during London session when 6E is flat
    "Micro Bitcoin": "MBT=F",  # 24/7 market; weekend breakout setups when futures are closed
    "10Y T-Note": "ZN=F",  # Macro context + flight-to-safety; trade on FOMC/CPI days
}

# ACTIVE_WATCHLIST: Union of core + extended — all assets the engine should
# actively monitor for breakout signals, CNN inference, and dashboard display.
ACTIVE_WATCHLIST: dict[str, str] = {**CORE_WATCHLIST, **EXTENDED_WATCHLIST}

# CORE_TICKERS / EXTENDED_TICKERS / ACTIVE_TICKERS: frozenset variants for
# quick membership checks (keyed by data_ticker, e.g. "MGC=F").
CORE_TICKERS: frozenset[str] = frozenset(CORE_WATCHLIST.values())
EXTENDED_TICKERS: frozenset[str] = frozenset(EXTENDED_WATCHLIST.values())
ACTIVE_TICKERS: frozenset[str] = frozenset(ACTIVE_WATCHLIST.values())

# Point values for quick P&L calculation (per contract, per full point move)
# Keyed by data_ticker so callers can look up by the ticker they actually trade.
POINT_VALUE: dict[str, float] = {
    str(spec.get("data_ticker", spec["ticker"])): float(spec["point"]) for spec in MICRO_CONTRACT_SPECS.values()
}
# Add Kraken crypto point values
if ENABLE_KRAKEN_CRYPTO:
    POINT_VALUE.update({str(spec["data_ticker"]): float(spec["point"]) for spec in KRAKEN_CONTRACT_SPECS.values()})

# Tick sizes keyed by data_ticker
TICK_SIZE: dict[str, float] = {
    str(spec.get("data_ticker", spec["ticker"])): float(spec["tick"]) for spec in MICRO_CONTRACT_SPECS.values()
}
# Add Kraken crypto tick sizes
if ENABLE_KRAKEN_CRYPTO:
    TICK_SIZE.update({str(spec["data_ticker"]): float(spec["tick"]) for spec in KRAKEN_CONTRACT_SPECS.values()})


def set_contract_mode(mode: str) -> dict:
    """Switch between 'micro' and 'full' contract specs at runtime.

    Updates the module-level CONTRACT_SPECS, ASSETS, and TICKER_TO_NAME dicts
    in place so all importers see the change.

    Returns the newly active CONTRACT_SPECS.
    """
    global CONTRACT_MODE
    mode = mode.lower()
    if mode not in ("micro", "full"):
        raise ValueError(f"Invalid contract mode '{mode}'. Use 'micro' or 'full'.")

    CONTRACT_MODE = mode
    source = MICRO_CONTRACT_SPECS if mode == "micro" else FULL_CONTRACT_SPECS

    CONTRACT_SPECS.clear()
    CONTRACT_SPECS.update(source)

    ASSETS.clear()
    ASSETS.update({name: str(spec.get("data_ticker", spec["ticker"])) for name, spec in CONTRACT_SPECS.items()})

    TICKER_TO_NAME.clear()
    TICKER_TO_NAME.update({str(spec.get("data_ticker", spec["ticker"])): name for name, spec in CONTRACT_SPECS.items()})

    return CONTRACT_SPECS


# ---------------------------------------------------------------------------
# Trade statuses
# ---------------------------------------------------------------------------
STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"
STATUS_CANCELLED = "CANCELLED"


# ═══════════════════════════════════════════════════════════════════════════
# Database abstraction layer
# ═══════════════════════════════════════════════════════════════════════════
#
# Two backends, same interface:
#   - SQLite: uses sqlite3 directly (zero dependencies beyond stdlib)
#   - Postgres: uses psycopg via SQLAlchemy's raw connection interface
#
# The key difference is parameter placeholders: SQLite uses `?` while
# Postgres uses `%s`.  We normalise by writing SQL with `?` and
# converting at execution time when using Postgres.
# ═══════════════════════════════════════════════════════════════════════════


def _get_sa_engine():
    """Lazily create the SQLAlchemy engine for Postgres."""
    global _sa_engine
    if _sa_engine is None:
        try:
            from sqlalchemy import create_engine

            _sa_engine = create_engine(
                DATABASE_URL,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            logger.info("Postgres engine created: %s", DATABASE_URL.split("@")[-1])
        except Exception as exc:
            logger.error("Failed to create Postgres engine: %s", exc)
            raise
    return _sa_engine


def _convert_placeholders(sql: str) -> str:
    """Convert SQLite-style `?` placeholders to Postgres-style `%s`."""
    return sql.replace("?", "%s")


class _RowProxy:
    """Lightweight dict-like wrapper around a Postgres row tuple.

    Provides item access by column name (row["col"]) and dict()
    conversion, matching sqlite3.Row behaviour.
    """

    __slots__ = ("_data",)

    def __init__(self, columns: list[str], values: tuple):
        self._data = dict(zip(columns, values, strict=False))

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str):
        return key in self._data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)


class _PgCursorWrapper:
    """Wraps a psycopg/DBAPI cursor to auto-convert `?` → `%s`
    and return _RowProxy objects for dict-like access."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql: str, params=None):
        converted = _convert_placeholders(sql)
        if params:
            self._cursor.execute(converted, params)
        else:
            self._cursor.execute(converted)
        return self

    def executescript(self, sql: str):
        """Execute a multi-statement script.  Postgres doesn't have
        executescript, so we just execute it as a single string."""
        # Replace SQLite-specific syntax if present
        pg_sql = sql.replace("AUTOINCREMENT", "")
        # SERIAL PRIMARY KEY already handles auto-increment in Postgres
        self._cursor.execute(pg_sql)
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        if self._cursor.description:
            columns = [desc[0] for desc in self._cursor.description]
            return _RowProxy(columns, row)
        return row

    def fetchall(self):
        rows = self._cursor.fetchall()
        if not rows or not self._cursor.description:
            return rows
        columns = [desc[0] for desc in self._cursor.description]
        return [_RowProxy(columns, r) for r in rows]

    @property
    def lastrowid(self):
        # psycopg3 doesn't always set lastrowid; we handle this
        # in the CRUD functions with RETURNING clauses
        return getattr(self._cursor, "lastrowid", None)

    @property
    def description(self):
        return self._cursor.description


class _PgConnectionWrapper:
    """Wraps a SQLAlchemy raw connection to match sqlite3.Connection API.

    Provides execute(), executescript(), commit(), close(), and
    row_factory-like behaviour via _RowProxy.
    """

    def __init__(self, raw_conn):
        self._conn = raw_conn
        self._cursor = raw_conn.cursor()

    def execute(self, sql: str, params=None):
        wrapper = _PgCursorWrapper(self._cursor)
        wrapper.execute(sql, params)
        return wrapper

    def executescript(self, sql: str):
        wrapper = _PgCursorWrapper(self._cursor)
        wrapper.executescript(sql)
        return wrapper

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        with contextlib.suppress(Exception):
            self._cursor.close()
        with contextlib.suppress(Exception):
            self._conn.close()


def _get_sqlite_conn() -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_conn():
    """Get a database connection (Postgres or SQLite).

    Returns an object with execute(), executescript(), commit(), close()
    methods.  Rows are accessible by column name via dict-style access.
    """
    if _USE_POSTGRES:
        try:
            engine = _get_sa_engine()
            raw = engine.raw_connection()
            return _PgConnectionWrapper(raw)
        except Exception as exc:
            logger.warning("Postgres connection failed, falling back to SQLite: %s", exc)
            return _get_sqlite_conn()
    return _get_sqlite_conn()


def _is_using_postgres() -> bool:
    """Return True if Postgres is the active backend."""
    return _USE_POSTGRES


# ---------------------------------------------------------------------------
# Trades schema
# ---------------------------------------------------------------------------

_SCHEMA_V2_SQLITE = """
CREATE TABLE IF NOT EXISTS trades_v2 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    account_size    INTEGER NOT NULL,
    asset           TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry           REAL    NOT NULL,
    sl              REAL,
    tp              REAL,
    contracts       INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL DEFAULT 'OPEN',
    close_price     REAL,
    close_time      TEXT,
    pnl             REAL,
    rr              REAL,
    notes           TEXT    DEFAULT '',
    strategy        TEXT    DEFAULT '',
    grade           TEXT    DEFAULT '',
    source          TEXT    DEFAULT 'manual'
);
"""

_SCHEMA_V2_PG = """
CREATE TABLE IF NOT EXISTS trades_v2 (
    id              SERIAL PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    account_size    INTEGER NOT NULL,
    asset           TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry           DOUBLE PRECISION NOT NULL,
    sl              DOUBLE PRECISION,
    tp              DOUBLE PRECISION,
    contracts       INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL DEFAULT 'OPEN',
    close_price     DOUBLE PRECISION,
    close_time      TEXT,
    pnl             DOUBLE PRECISION,
    rr              DOUBLE PRECISION,
    notes           TEXT    DEFAULT '',
    strategy        TEXT    DEFAULT '',
    grade           TEXT    DEFAULT '',
    source          TEXT    DEFAULT 'manual'
);
"""

# ---------------------------------------------------------------------------
# Migration: add grade + source columns to existing trades_v2 tables
# ---------------------------------------------------------------------------

# Postgres supports IF NOT EXISTS on ALTER TABLE ADD COLUMN (PG 9.6+)
_MIGRATE_ADD_GRADE_PG = [
    "ALTER TABLE trades_v2 ADD COLUMN IF NOT EXISTS grade TEXT DEFAULT '';",
    "ALTER TABLE trades_v2 ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'manual';",
]

# SQLite does NOT support IF NOT EXISTS on ALTER TABLE — use try/except in init_db
_MIGRATE_ADD_GRADE_SQLITE = [
    "ALTER TABLE trades_v2 ADD COLUMN grade TEXT DEFAULT '';",
    "ALTER TABLE trades_v2 ADD COLUMN source TEXT DEFAULT 'manual';",
]

_MIGRATE_V1 = """
INSERT INTO trades_v2
    (id, created_at, account_size, asset, direction, entry, sl, tp,
     contracts, status, close_price, close_time, pnl, rr, notes, strategy)
SELECT
    id,
    date,
    150000,
    asset,
    direction,
    entry,
    sl,
    tp,
    contracts,
    CASE WHEN exit_price IS NOT NULL AND exit_price != 0 THEN 'CLOSED' ELSE 'OPEN' END,
    CASE WHEN exit_price != 0 THEN exit_price ELSE NULL END,
    CASE WHEN exit_price IS NOT NULL AND exit_price != 0 THEN date ELSE NULL END,
    pnl,
    rr,
    COALESCE(notes, ''),
    COALESCE(strategy, '')
FROM trades;
"""


# ---------------------------------------------------------------------------
# Reddit posts schema
# ---------------------------------------------------------------------------

_SCHEMA_REDDIT_POSTS_SQLITE = """
CREATE TABLE IF NOT EXISTS reddit_posts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id         TEXT        NOT NULL UNIQUE,
    subreddit       TEXT        NOT NULL,
    author          TEXT,
    title           TEXT,
    body            TEXT,
    url             TEXT,
    score           INTEGER     DEFAULT 0,
    num_comments    INTEGER     DEFAULT 0,
    is_comment      INTEGER     DEFAULT 0,
    asset           TEXT        NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    upvote_ratio    REAL,
    created_utc     TEXT        NOT NULL,
    fetched_at      TEXT        NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset      ON reddit_posts (asset);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_created    ON reddit_posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset_time ON reddit_posts (asset, created_utc DESC);
"""

_SCHEMA_REDDIT_POSTS_PG = """
CREATE TABLE IF NOT EXISTS reddit_posts (
    id              SERIAL PRIMARY KEY,
    post_id         TEXT        NOT NULL UNIQUE,
    subreddit       TEXT        NOT NULL,
    author          TEXT,
    title           TEXT,
    body            TEXT,
    url             TEXT,
    score           INTEGER     DEFAULT 0,
    num_comments    INTEGER     DEFAULT 0,
    is_comment      BOOLEAN     DEFAULT FALSE,
    asset           TEXT        NOT NULL,
    sentiment_score DOUBLE PRECISION,
    sentiment_label TEXT,
    upvote_ratio    DOUBLE PRECISION,
    created_utc     TIMESTAMPTZ NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset      ON reddit_posts (asset);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_created    ON reddit_posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_asset_time ON reddit_posts (asset, created_utc DESC);
"""

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------


def _init_audit_tables(conn, use_postgres: bool) -> None:
    """Create audit event tables (risk_events, orb_events) idempotently.

    Called from init_db() after the core tables are created.
    Also runs non-destructive column migrations so existing deployments
    pick up the new breakout_type / mtf_score / macd_slope / divergence
    columns without requiring a manual ALTER TABLE.
    """
    if use_postgres:
        for schema in (_SCHEMA_RISK_EVENTS_PG, _SCHEMA_ORB_EVENTS_PG):
            for stmt in schema.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except Exception as exc:
                        # Index may already exist — not fatal
                        logger.debug("Audit table DDL (pg, non-fatal): %s", exc)
        conn.commit()
        # Run column-level migrations (ADD COLUMN IF NOT EXISTS — idempotent)
        for stmt in _MIGRATE_ORB_EVENTS_ADD_COLS_PG:
            try:
                conn.execute(stmt)
            except Exception as exc:
                logger.debug("ORB events column migration (pg, non-fatal): %s", exc)
        with contextlib.suppress(Exception):
            conn.commit()
    else:
        conn.executescript(_SCHEMA_RISK_EVENTS_SQLITE)
        conn.executescript(_SCHEMA_ORB_EVENTS_SQLITE)
        # SQLite: ALTER TABLE ADD COLUMN is idempotent only if we catch the error
        for stmt in _MIGRATE_ORB_EVENTS_ADD_COLS_SQLITE.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception as exc:
                    # "duplicate column name" is expected on existing DBs — ignore
                    if "duplicate column" not in str(exc).lower():
                        logger.debug("ORB events column migration (sqlite, non-fatal): %s", exc)
        with contextlib.suppress(Exception):
            conn.commit()

    logger.info("Audit tables initialised (risk_events, orb_events)")


def init_db() -> None:
    """Initialise the trades_v2, daily_journal, and audit event tables.

    For SQLite: migrates from v1 schema if needed.
    For Postgres: creates tables idempotently (CREATE TABLE IF NOT EXISTS).
    """
    conn = _get_conn()

    if _USE_POSTGRES:
        try:
            conn.executescript(_SCHEMA_V2_PG)
            conn.executescript(_SCHEMA_DAILY_JOURNAL_PG)
            conn.commit()
            logger.info("Postgres tables initialised (trades_v2, daily_journal)")
        except Exception as exc:
            logger.error("Postgres init_db failed: %s", exc)
            conn.rollback()

        # Run grade/source column migrations (ADD COLUMN IF NOT EXISTS — idempotent on PG)
        for stmt in _MIGRATE_ADD_GRADE_PG:
            try:
                conn.execute(stmt)
            except Exception as exc:
                logger.debug("grade/source column migration (pg, non-fatal): %s", exc)
        with contextlib.suppress(Exception):
            conn.commit()

        # Create audit tables (separate try so core tables aren't rolled back)
        try:
            _init_audit_tables(conn, use_postgres=True)
        except Exception as exc:
            logger.error("Audit table init failed (non-fatal): %s", exc)
            with contextlib.suppress(Exception):
                conn.rollback()

        # Create reddit_posts table (separate try — non-fatal if praw not used)
        try:
            conn.executescript(_SCHEMA_REDDIT_POSTS_PG)
            conn.commit()
            logger.info("Postgres tables initialised (reddit_posts)")
        except Exception as exc:
            logger.warning("reddit_posts table init failed (non-fatal): %s", exc)
            with contextlib.suppress(Exception):
                conn.rollback()

        conn.close()
        return

    # --- SQLite path ---
    # Check if new table already exists
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades_v2'")
    if cur.fetchone() is not None:
        # trades_v2 exists — ensure daily_journal + audit tables exist, then run column migrations
        conn.executescript(_SCHEMA_DAILY_JOURNAL_SQLITE)
        _init_audit_tables(conn, use_postgres=False)
        # Run grade/source column migrations (SQLite doesn't support IF NOT EXISTS — use try/except)
        for stmt in _MIGRATE_ADD_GRADE_SQLITE:
            try:
                conn.execute(stmt)
            except Exception as exc:
                # "duplicate column name" is expected on already-migrated DBs — ignore
                if "duplicate column" not in str(exc).lower():
                    logger.debug("grade/source column migration (sqlite, non-fatal): %s", exc)
        with contextlib.suppress(Exception):
            conn.commit()
        conn.close()
        return

    # Create v2 schema
    conn.executescript(_SCHEMA_V2_SQLITE)

    # Create daily journal schema
    conn.executescript(_SCHEMA_DAILY_JOURNAL_SQLITE)

    # Create audit tables
    _init_audit_tables(conn, use_postgres=False)

    # Create reddit_posts table
    try:
        conn.executescript(_SCHEMA_REDDIT_POSTS_SQLITE)
    except Exception as exc:
        logger.warning("reddit_posts table init failed (non-fatal): %s", exc)

    # Migrate from v1 if it exists
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    if cur.fetchone() is not None:
        try:
            conn.executescript(_MIGRATE_V1)
            conn.execute("ALTER TABLE trades RENAME TO trades_v1_backup")
            conn.commit()
        except Exception:
            conn.rollback()

    conn.close()


# ---------------------------------------------------------------------------
# Helper: convert row to dict (works for both sqlite3.Row and _RowProxy)
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> dict:
    """Convert a database row to a plain dict."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}  # noqa: SIM118
    return dict(row)


# ---------------------------------------------------------------------------
# Helper: insert with RETURNING for Postgres, lastrowid for SQLite
# ---------------------------------------------------------------------------


def _insert_returning_id(conn, sql: str, params: tuple, table: str = "trades_v2") -> int:
    """Execute an INSERT and return the new row's id.

    For Postgres, appends RETURNING id to the SQL.
    For SQLite, uses cursor.lastrowid.
    """
    if _USE_POSTGRES:
        pg_sql = _convert_placeholders(sql) + " RETURNING id"
        cur = conn._cursor if hasattr(conn, "_cursor") else conn.execute(pg_sql, params)
        if hasattr(conn, "_cursor"):
            conn._cursor.execute(pg_sql, params)
            row = conn._cursor.fetchone()
        else:
            row = cur.fetchone()
        return row[0] if row else 0
    else:
        cur = conn.execute(sql, params)
        return cur.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def create_trade(
    account_size: int,
    asset: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    contracts: int,
    strategy: str = "",
    notes: str = "",
) -> int:
    """Insert a new OPEN trade. Returns the new trade id."""
    conn = _get_conn()
    now = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")

    sql = """INSERT INTO trades_v2
           (created_at, account_size, asset, direction, entry, sl, tp,
            contracts, status, strategy, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    params = (
        now,
        account_size,
        asset,
        direction,
        entry,
        sl,
        tp,
        contracts,
        STATUS_OPEN,
        strategy,
        notes,
    )

    trade_id = _insert_returning_id(conn, sql, params, "trades_v2")
    conn.commit()
    conn.close()
    return trade_id


def upsert_trade_from_fill(
    account_key: str,
    symbol: str,
    direction: str,
    entry_price: float,
    close_price: float | None,
    contracts: int,
    pnl: float | None,
    fill_time: str,
    strategy: str = "rithmic_sync",
    notes: str = "",
    source: str = "rithmic_sync",
) -> int:
    """Insert or update a trade record from a Rithmic fill.

    Uses fill_time + symbol + account_key as a natural dedup key:
      - created_at matches fill_time (date prefix)
      - asset matches symbol
      - notes contains account_key

    Returns the trade id (int).
    """
    conn = _get_conn()

    # Build the notes field so it embeds the account key for dedup/filtering
    full_notes = notes or f"rithmic_sync:{account_key}"
    if account_key and account_key not in full_notes:
        full_notes = f"{full_notes} [{account_key}]"

    # Normalise fill_time to "YYYY-MM-DD HH:MM:SS" if possible
    created_at = fill_time
    try:
        # Accept ISO strings, epoch strings, or already-formatted timestamps
        if fill_time and fill_time.strip():
            # Try parsing common formats
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    created_at = datetime.strptime(fill_time.strip()[:19], fmt).strftime("%Y-%m-%d %H:%M:%S")
                    break
                except ValueError:
                    continue
    except Exception:
        pass
    if not created_at:
        created_at = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")

    # Determine status: if we have a close_price it's already closed
    status = STATUS_CLOSED if close_price is not None else STATUS_OPEN

    # Check for an existing record with matching fill_time prefix + symbol + account_key in notes
    date_prefix = created_at[:10]  # "YYYY-MM-DD"
    existing = conn.execute(
        """SELECT id FROM trades_v2
           WHERE created_at LIKE ?
             AND asset = ?
             AND notes LIKE ?
           LIMIT 1""",
        (f"{date_prefix}%", symbol, f"%{account_key}%"),
    ).fetchone()

    if existing is not None:
        trade_id = int(_row_to_dict(existing).get("id") or 0)
        # Update the existing record with latest fill data
        conn.execute(
            """UPDATE trades_v2
               SET entry      = ?,
                   close_price = ?,
                   pnl         = ?,
                   status      = ?,
                   contracts   = ?,
                   direction   = ?,
                   strategy    = ?,
                   notes       = ?,
                   source      = ?,
                   close_time  = CASE WHEN ? IS NOT NULL THEN ? ELSE close_time END
               WHERE id = ?""",
            (
                entry_price,
                close_price,
                pnl,
                status,
                contracts,
                direction.upper(),
                strategy,
                full_notes,
                source,
                close_price,
                created_at if close_price is not None else None,
                trade_id,
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(
            "upsert_trade_from_fill: updated existing trade id=%d symbol=%s account=%s",
            trade_id,
            symbol,
            account_key,
        )
        return trade_id

    # No existing record — insert a new one
    sql = """INSERT INTO trades_v2
           (created_at, account_size, asset, direction, entry, sl, tp,
            contracts, status, close_price, close_time, pnl,
            strategy, notes, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    params = (
        created_at,
        150_000,  # default account_size — callers can patch if needed
        symbol,
        direction.upper(),
        entry_price,
        None,  # sl — not available from fill data
        None,  # tp — not available from fill data
        contracts,
        status,
        close_price,
        created_at if close_price is not None else None,
        pnl,
        strategy,
        full_notes,
        source,
    )
    trade_id = _insert_returning_id(conn, sql, params, "trades_v2")
    conn.commit()
    conn.close()
    logger.debug(
        "upsert_trade_from_fill: inserted new trade id=%d symbol=%s account=%s",
        trade_id,
        symbol,
        account_key,
    )
    return trade_id


def close_trade(trade_id: int, close_price: float) -> dict:
    """Close an open trade and calculate realised P&L.

    Returns a dict with the trade details including pnl.
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM trades_v2 WHERE id = ?", (trade_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Trade {trade_id} not found")
    if row["status"] != STATUS_OPEN:
        conn.close()
        raise ValueError(f"Trade {trade_id} is already {row['status']}")

    asset = row["asset"]
    direction = row["direction"]
    entry = row["entry"]
    contracts = row["contracts"]

    spec = CONTRACT_SPECS.get(asset)
    point_value = float(spec["point"]) if spec else 1.0

    if direction.upper() == "LONG":
        pnl = (close_price - entry) * point_value * contracts
        rr = abs((close_price - entry) / (entry - row["sl"])) if row["sl"] and row["sl"] != entry else 0.0
    else:
        pnl = (entry - close_price) * point_value * contracts
        rr = abs((entry - close_price) / (row["sl"] - entry)) if row["sl"] and row["sl"] != entry else 0.0

    close_time = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")

    conn.execute(
        """UPDATE trades_v2
           SET status = ?, close_price = ?, close_time = ?, pnl = ?, rr = ?
           WHERE id = ?""",
        (STATUS_CLOSED, close_price, close_time, round(pnl, 2), round(rr, 2), trade_id),
    )
    conn.commit()

    result = _row_to_dict(row)
    result.update(
        status=STATUS_CLOSED,
        close_price=close_price,
        close_time=close_time,
        pnl=round(pnl, 2),
        rr=round(rr, 2),
    )
    conn.close()
    return result


def cancel_trade(trade_id: int) -> None:
    """Cancel an open trade (never filled)."""
    conn = _get_conn()
    conn.execute(
        "UPDATE trades_v2 SET status = ? WHERE id = ? AND status = ?",
        (STATUS_CANCELLED, trade_id, STATUS_OPEN),
    )
    conn.commit()
    conn.close()


def _query_to_list(conn, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return a list of dicts.

    Works with both SQLite and Postgres backends.  For SQLite, we use
    pd.read_sql for convenience.  For Postgres, we fetch rows directly
    and convert to dicts, avoiding pd.read_sql connection issues.
    """
    if _USE_POSTGRES:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]
    else:
        df = pd.read_sql(sql, conn, params=params)
        return df.to_dict(orient="records")


def get_open_trades(account_size: int | None = None) -> list[dict]:
    """Return all OPEN trades, optionally filtered by account size."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY created_at DESC",
            (STATUS_OPEN, account_size),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY created_at DESC",
            (STATUS_OPEN,),
        )
    conn.close()
    return results


def get_closed_trades(account_size: int | None = None) -> list[dict]:
    """Return all CLOSED trades for the journal."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? AND account_size = ? ORDER BY close_time DESC",
            (STATUS_CLOSED, account_size),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE status = ? ORDER BY close_time DESC",
            (STATUS_CLOSED,),
        )
    conn.close()
    return results


def get_all_trades(account_size: int | None = None) -> list[dict]:
    """Return all trades regardless of status."""
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 WHERE account_size = ? ORDER BY created_at DESC",
            (account_size,),
        )
    else:
        results = _query_to_list(
            conn,
            "SELECT * FROM trades_v2 ORDER BY created_at DESC",
        )
    conn.close()
    return results


def get_today_pnl(account_size: int | None = None) -> float:
    """Sum of realised P&L for trades closed today."""
    today = datetime.now(tz=_EST).strftime("%Y-%m-%d")
    conn = _get_conn()
    if account_size:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades_v2 WHERE status = ? AND close_time LIKE ? AND account_size = ?",
            (STATUS_CLOSED, f"{today}%", account_size),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades_v2 WHERE status = ? AND close_time LIKE ?",
            (STATUS_CLOSED, f"{today}%"),
        ).fetchone()
    conn.close()
    # For Postgres _RowProxy, access by index via values; for SQLite Row, use index
    if row is None:
        return 0.0
    if hasattr(row, "values"):
        vals = list(row.values())
        return float(vals[0]) if vals else 0.0
    return float(row[0]) if row else 0.0  # type: ignore[index]


def get_today_trades(account_size: int | None = None) -> list[dict]:
    """Return all trades created or closed today."""
    today = datetime.now(tz=_EST).strftime("%Y-%m-%d")
    conn = _get_conn()
    if account_size:
        results = _query_to_list(
            conn,
            """SELECT * FROM trades_v2
               WHERE (created_at LIKE ? OR close_time LIKE ?)
                 AND account_size = ?
               ORDER BY created_at DESC""",
            (f"{today}%", f"{today}%", account_size),
        )
    else:
        results = _query_to_list(
            conn,
            """SELECT * FROM trades_v2
               WHERE created_at LIKE ? OR close_time LIKE ?
               ORDER BY created_at DESC""",
            (f"{today}%", f"{today}%"),
        )
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------


def calc_max_contracts(
    entry: float,
    sl: float,
    asset: str,
    risk_dollars: float,
    hard_max: int,
) -> int:
    """Calculate max contracts respecting risk-per-trade and account cap.

    Uses the currently active CONTRACT_SPECS (micro or full).
    The hard_max should come from account profile's max_contracts_micro
    when trading micros, or max_contracts when trading full-size.
    """
    spec = CONTRACT_SPECS.get(asset)
    if spec is None:
        return 1
    risk_per_contract = abs(entry - sl) * float(spec["point"])
    if risk_per_contract <= 0:
        return 1
    raw = int(risk_dollars // risk_per_contract)
    return max(1, min(raw, hard_max))


def get_max_contracts_for_profile(profile_key: str) -> int:
    """Return the appropriate max contracts limit for the active contract mode."""
    profile = ACCOUNT_PROFILES.get(profile_key)
    if profile is None:
        return 4
    if CONTRACT_MODE == "micro":
        return int(profile.get("max_contracts_micro", profile["max_contracts"]))
    return int(profile["max_contracts"])


def calc_pnl(
    asset: str,
    direction: str,
    entry: float,
    close_price: float,
    contracts: int,
) -> float:
    """Calculate P&L for a given trade."""
    spec = CONTRACT_SPECS.get(asset)
    point_value = float(spec["point"]) if spec else 1.0
    if direction.upper() == "LONG":
        return (close_price - entry) * point_value * contracts
    else:
        return (entry - close_price) * point_value * contracts


# ---------------------------------------------------------------------------
# Daily Journal CRUD
# ---------------------------------------------------------------------------


def save_daily_journal(
    trade_date: str,
    account_size: int,
    gross_pnl: float,
    net_pnl: float,
    num_contracts: int = 0,
    instruments: str = "",
    notes: str = "",
) -> int:
    """Save or update a daily journal entry.

    Commissions are auto-calculated as gross_pnl - net_pnl.
    If an entry already exists for the given date, it is updated.
    Returns the row id.
    """
    commissions = round(gross_pnl - net_pnl, 2)
    now = datetime.now(tz=_EST).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_conn()

    # Check if entry exists for this date
    existing = conn.execute("SELECT id FROM daily_journal WHERE trade_date = ?", (trade_date,)).fetchone()

    if existing:
        conn.execute(
            """UPDATE daily_journal
               SET account_size = ?, gross_pnl = ?, net_pnl = ?,
                   commissions = ?, num_contracts = ?, instruments = ?,
                   notes = ?, created_at = ?
               WHERE trade_date = ?""",
            (
                account_size,
                round(gross_pnl, 2),
                round(net_pnl, 2),
                commissions,
                num_contracts,
                instruments,
                notes,
                now,
                trade_date,
            ),
        )
        row_id = existing["id"]
    else:
        insert_sql = """INSERT INTO daily_journal
               (trade_date, account_size, gross_pnl, net_pnl, commissions,
                num_contracts, instruments, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        insert_params = (
            trade_date,
            account_size,
            round(gross_pnl, 2),
            round(net_pnl, 2),
            commissions,
            num_contracts,
            instruments,
            notes,
            now,
        )
        row_id = _insert_returning_id(conn, insert_sql, insert_params, "daily_journal")

    conn.commit()
    conn.close()
    return row_id  # type: ignore[return-value]


def get_daily_journal(
    limit: int = 30,
    account_size: int | None = None,
) -> pd.DataFrame:
    """Return recent daily journal entries as a DataFrame."""
    conn = _get_conn()

    if _USE_POSTGRES:
        # For Postgres, query and convert to DataFrame manually
        if account_size:
            rows = _query_to_list(
                conn,
                """SELECT * FROM daily_journal
                   WHERE account_size = ?
                   ORDER BY trade_date DESC LIMIT ?""",
                (account_size, limit),
            )
        else:
            rows = _query_to_list(
                conn,
                "SELECT * FROM daily_journal ORDER BY trade_date DESC LIMIT ?",
                (limit,),
            )
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        # SQLite: use pd.read_sql directly
        if account_size:
            df = pd.read_sql(
                """SELECT * FROM daily_journal
                   WHERE account_size = ?
                   ORDER BY trade_date DESC LIMIT ?""",
                conn,
                params=(account_size, limit),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM daily_journal ORDER BY trade_date DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        conn.close()
        return df


def get_journal_stats(account_size: int | None = None) -> dict:
    """Compute aggregate stats from the daily journal.

    Returns dict with total_days, total_gross, total_net, total_commissions,
    win_days, loss_days, win_rate, avg_daily_pnl, best_day, worst_day,
    current_streak.
    """
    df = get_daily_journal(limit=9999, account_size=account_size)
    if df.empty:
        return {
            "total_days": 0,
            "total_gross": 0.0,
            "total_net": 0.0,
            "total_commissions": 0.0,
            "win_days": 0,
            "loss_days": 0,
            "break_even_days": 0,
            "win_rate": 0.0,
            "avg_daily_net": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "current_streak": 0,
        }

    total_days = len(df)
    total_gross = float(df["gross_pnl"].sum())  # type: ignore[arg-type]
    total_net = float(df["net_pnl"].sum())  # type: ignore[arg-type]
    total_commissions = float(df["commissions"].sum())  # type: ignore[arg-type]
    win_days = int((df["net_pnl"] > 0).sum())
    loss_days = int((df["net_pnl"] < 0).sum())
    break_even_days = int((df["net_pnl"] == 0).sum())
    win_rate = win_days / total_days * 100 if total_days > 0 else 0.0
    avg_daily_net = total_net / total_days if total_days > 0 else 0.0
    best_day = float(df["net_pnl"].max())  # type: ignore[arg-type]
    worst_day = float(df["net_pnl"].min())  # type: ignore[arg-type]

    # Current streak (sorted by date ascending for streak calc)
    sorted_df = df.sort_values("trade_date", ascending=True)
    streak = 0
    for pnl in reversed(sorted_df["net_pnl"].tolist()):
        if pnl > 0:
            if streak >= 0:
                streak += 1
            else:
                break
        elif pnl < 0:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break

    return {
        "total_days": total_days,
        "total_gross": round(total_gross, 2),
        "total_net": round(total_net, 2),
        "total_commissions": round(total_commissions, 2),
        "win_days": win_days,
        "loss_days": loss_days,
        "break_even_days": break_even_days,
        "win_rate": round(win_rate, 1),
        "avg_daily_net": round(avg_daily_net, 2),
        "best_day": round(best_day, 2),
        "worst_day": round(worst_day, 2),
        "current_streak": streak,
    }


# ---------------------------------------------------------------------------
# SQLite → Postgres one-time migration helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audit event CRUD — persistent storage for risk blocks & ORB detections
# ---------------------------------------------------------------------------


def record_risk_event(
    event_type: str,
    symbol: str = "",
    side: str = "",
    reason: str = "",
    daily_pnl: float = 0.0,
    open_trades: int = 0,
    account_size: int = 0,
    risk_pct: float = 0.0,
    session: str = "",
    metadata: dict | None = None,
) -> int | None:
    """Persist a risk event to the database for permanent audit trail.

    Args:
        event_type: "block", "warning", "clear", "circuit_breaker", etc.
        symbol: Instrument symbol (e.g. "MGC", "MNQ")
        side: "LONG" or "SHORT"
        reason: Human-readable reason string
        daily_pnl: Current daily P&L at the time of the event
        open_trades: Number of open trades at the time
        account_size: Account size at the time
        risk_pct: Risk as percentage of account
        session: Session mode ("pre_market", "active", "off_hours")
        metadata: Optional extra data (serialised to JSON)

    Returns:
        The inserted row ID, or None on failure.
    """
    now_str = datetime.now(tz=_EST).isoformat()
    meta_json = json.dumps(metadata or {})

    try:
        conn = _get_conn()
        if _USE_POSTGRES:
            cur = conn.execute(
                "INSERT INTO risk_events "
                "(timestamp, event_type, symbol, side, reason, daily_pnl, "
                " open_trades, account_size, risk_pct, session, metadata_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (
                    now_str,
                    event_type,
                    symbol,
                    side,
                    reason,
                    daily_pnl,
                    open_trades,
                    account_size,
                    risk_pct,
                    session,
                    meta_json,
                ),
            )
            row = cur.fetchone()
            row_id = row["id"] if row else None
        else:
            cur = conn.execute(
                "INSERT INTO risk_events "
                "(timestamp, event_type, symbol, side, reason, daily_pnl, "
                " open_trades, account_size, risk_pct, session, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now_str,
                    event_type,
                    symbol,
                    side,
                    reason,
                    daily_pnl,
                    open_trades,
                    account_size,
                    risk_pct,
                    session,
                    meta_json,
                ),
            )
            row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to record risk event: %s", exc)
        return None


def get_risk_events(
    limit: int = 100,
    event_type: str | None = None,
    symbol: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Query risk events from the database.

    Args:
        limit: Maximum number of events to return.
        event_type: Filter by event type (e.g. "block").
        symbol: Filter by symbol.
        since: ISO timestamp — only return events after this time.

    Returns:
        List of event dicts, most recent first.
    """
    ph = "%s" if _USE_POSTGRES else "?"
    conditions: list[str] = []
    params: list[Any] = []

    if event_type:
        conditions.append(f"event_type = {ph}")
        params.append(event_type)
    if symbol:
        conditions.append(f"symbol = {ph}")
        params.append(symbol)
    if since:
        conditions.append(f"timestamp >= {ph}")
        params.append(since)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    sql = f"SELECT * FROM risk_events {where} ORDER BY timestamp DESC LIMIT {ph}"
    params.append(limit)

    try:
        conn = _get_conn()
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to query risk events: %s", exc)
        return []


def record_orb_event(
    symbol: str,
    or_high: float = 0.0,
    or_low: float = 0.0,
    or_range: float = 0.0,
    atr_value: float = 0.0,
    breakout_detected: bool = False,
    direction: str = "",
    trigger_price: float = 0.0,
    long_trigger: float = 0.0,
    short_trigger: float = 0.0,
    bar_count: int = 0,
    session: str = "",
    metadata: dict | None = None,
    breakout_type: str = "ORB",
    mtf_score: float | None = None,
    macd_slope: float | None = None,
    divergence: str = "",
) -> int | None:
    """Persist an ORB evaluation result to the database.

    Args:
        symbol: Instrument symbol.
        or_high: Opening range high price.
        or_low: Opening range low price.
        or_range: OR high − OR low.
        atr_value: ATR value used for breakout threshold.
        breakout_detected: True if a breakout was detected.
        direction: "LONG", "SHORT", or "".
        trigger_price: Price that triggered the breakout.
        long_trigger: Upper breakout level.
        short_trigger: Lower breakout level.
        bar_count: Number of bars in the opening range.
        session: Session mode.
        metadata: Optional extra data (serialised to JSON).
        breakout_type: Breakout type string — "ORB", "PDR", "IB", or "CONS".
        mtf_score: Aggregate multi-timeframe score (0.0–1.0), or None.
        macd_slope: MACD histogram slope from MTF analysis, or None.
        divergence: Divergence classification — "confirming", "opposing", or "".

    Returns:
        The inserted row ID, or None on failure.
    """
    now_str = datetime.now(tz=_EST).isoformat()
    meta_json = json.dumps(metadata or {})
    bd_int = 1 if breakout_detected else 0

    try:
        conn = _get_conn()
        if _USE_POSTGRES:
            cur = conn.execute(
                "INSERT INTO orb_events "
                "(timestamp, symbol, or_high, or_low, or_range, atr_value, "
                " breakout_detected, direction, trigger_price, long_trigger, "
                " short_trigger, bar_count, session, metadata_json, "
                " breakout_type, mtf_score, macd_slope, divergence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (
                    now_str,
                    symbol,
                    or_high,
                    or_low,
                    or_range,
                    atr_value,
                    bd_int,
                    direction,
                    trigger_price,
                    long_trigger,
                    short_trigger,
                    bar_count,
                    session,
                    meta_json,
                    breakout_type,
                    mtf_score,
                    macd_slope,
                    divergence,
                ),
            )
            row = cur.fetchone()
            row_id = row["id"] if row else None
        else:
            cur = conn.execute(
                "INSERT INTO orb_events "
                "(timestamp, symbol, or_high, or_low, or_range, atr_value, "
                " breakout_detected, direction, trigger_price, long_trigger, "
                " short_trigger, bar_count, session, metadata_json, "
                " breakout_type, mtf_score, macd_slope, divergence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now_str,
                    symbol,
                    or_high,
                    or_low,
                    or_range,
                    atr_value,
                    bd_int,
                    direction,
                    trigger_price,
                    long_trigger,
                    short_trigger,
                    bar_count,
                    session,
                    meta_json,
                    breakout_type,
                    mtf_score,
                    macd_slope,
                    divergence,
                ),
            )
            row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to record ORB event: %s", exc)
        return None


def get_orb_events(
    limit: int = 100,
    symbol: str | None = None,
    breakout_only: bool = False,
    since: str | None = None,
    breakout_type: str | None = None,
) -> list[dict]:
    """Query ORB events from the database.

    Args:
        limit: Maximum number of events to return.
        symbol: Filter by symbol.
        breakout_only: If True, only return events where breakout was detected.
        since: ISO timestamp — only return events after this time.
        breakout_type: Filter by breakout type string (e.g. "ORB", "PDR", "IB",
            "CONS", "WEEKLY", etc.).  Case-insensitive.  None = all types.

    Returns:
        List of event dicts, most recent first.
    """
    ph = "%s" if _USE_POSTGRES else "?"
    conditions: list[str] = []
    params: list[Any] = []

    if symbol:
        conditions.append(f"symbol = {ph}")
        params.append(symbol)
    if breakout_only:
        conditions.append("breakout_detected = 1")
    if since:
        conditions.append(f"timestamp >= {ph}")
        params.append(since)
    if breakout_type and breakout_type.upper() != "ALL":
        conditions.append(f"breakout_type = {ph}")
        params.append(breakout_type.upper())

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    sql = f"SELECT * FROM orb_events {where} ORDER BY timestamp DESC LIMIT {ph}"
    params.append(limit)

    try:
        conn = _get_conn()
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to query ORB events: %s", exc)
        return []


def get_audit_summary(days_back: int = 7) -> dict:
    """Get a summary of audit events for the last N days.

    Returns:
        Dict with counts and breakdowns for risk and ORB events.
    """
    from datetime import timedelta

    cutoff = (datetime.now(tz=_EST) - timedelta(days=days_back)).isoformat()
    ph = "%s" if _USE_POSTGRES else "?"

    summary: dict[str, Any] = {
        "days_back": days_back,
        "cutoff": cutoff,
        "risk_events": {"total": 0, "blocks": 0, "warnings": 0, "by_symbol": {}},
        "orb_events": {"total": 0, "breakouts": 0, "by_symbol": {}},
    }

    try:
        conn = _get_conn()

        # Risk events summary
        cur = conn.execute(
            f"SELECT event_type, symbol, COUNT(*) as cnt "
            f"FROM risk_events WHERE timestamp >= {ph} "
            f"GROUP BY event_type, symbol ORDER BY cnt DESC",
            (cutoff,),
        )
        for row in cur.fetchall():
            d = _row_to_dict(row)
            cnt = int(d.get("cnt", 0))
            summary["risk_events"]["total"] += cnt
            et = d.get("event_type", "")
            sym = d.get("symbol", "")
            if et == "block":
                summary["risk_events"]["blocks"] += cnt
            elif et == "warning":
                summary["risk_events"]["warnings"] += cnt
            if sym:
                by_sym: dict[str, int] = summary["risk_events"]["by_symbol"]
                by_sym[sym] = by_sym.get(sym, 0) + cnt

        # ORB events summary
        cur = conn.execute(
            f"SELECT symbol, breakout_detected, COUNT(*) as cnt "
            f"FROM orb_events WHERE timestamp >= {ph} "
            f"GROUP BY symbol, breakout_detected ORDER BY cnt DESC",
            (cutoff,),
        )
        for row in cur.fetchall():
            d = _row_to_dict(row)
            cnt = int(d.get("cnt", 0))
            summary["orb_events"]["total"] += cnt
            if d.get("breakout_detected", 0) == 1:
                summary["orb_events"]["breakouts"] += cnt
            sym = d.get("symbol", "")
            if sym:
                by_sym_orb: dict[str, int] = summary["orb_events"]["by_symbol"]
                by_sym_orb[sym] = by_sym_orb.get(sym, 0) + cnt

        conn.close()
    except Exception as exc:
        logger.error("Failed to build audit summary: %s", exc)
        summary["error"] = str(exc)

    return summary


# ---------------------------------------------------------------------------
# SQLite → Postgres one-time migration helper
# ---------------------------------------------------------------------------


def migrate_sqlite_to_postgres(
    sqlite_path: str | None = None,
    pg_url: str | None = None,
) -> dict:
    """One-time migration: copy all data from SQLite to Postgres.

    Call this manually when transitioning from SQLite to Postgres:
        from lib.models import migrate_sqlite_to_postgres
        migrate_sqlite_to_postgres("data/futures_journal.db")

    Returns a dict with counts of migrated records.
    """
    import sqlite3 as _sqlite3

    src_path = sqlite_path or DB_PATH
    target_url = pg_url or DATABASE_URL

    if not target_url.startswith("postgresql"):
        raise ValueError("DATABASE_URL must point to a Postgres database")

    # Read from SQLite
    src_conn = _sqlite3.connect(src_path)
    src_conn.row_factory = _sqlite3.Row

    result: dict[str, Any] = {"trades": 0, "journal": 0, "errors": []}

    # Check source tables
    tables = [r[0] for r in src_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    from sqlalchemy import create_engine, text

    pg_engine = create_engine(target_url)

    with pg_engine.connect() as pg_conn:
        # Migrate trades_v2
        if "trades_v2" in tables:
            rows = src_conn.execute("SELECT * FROM trades_v2").fetchall()
            for row in rows:
                d = dict(row)
                d.pop("id", None)
                try:
                    cols = ", ".join(d.keys())
                    placeholders = ", ".join(f":{k}" for k in d)
                    pg_conn.execute(
                        text(f"INSERT INTO trades_v2 ({cols}) VALUES ({placeholders})"),
                        d,
                    )
                    result["trades"] += 1
                except Exception as exc:
                    result["errors"].append(f"Trade: {exc}")

        # Migrate daily_journal
        if "daily_journal" in tables:
            rows = src_conn.execute("SELECT * FROM daily_journal").fetchall()
            for row in rows:
                d = dict(row)
                d.pop("id", None)
                try:
                    cols = ", ".join(d.keys())
                    placeholders = ", ".join(f":{k}" for k in d)
                    pg_conn.execute(
                        text(
                            f"INSERT INTO daily_journal ({cols}) VALUES ({placeholders}) "
                            f"ON CONFLICT (trade_date) DO NOTHING"
                        ),
                        d,
                    )
                    result["journal"] += 1
                except Exception as exc:
                    result["errors"].append(f"Journal: {exc}")

        pg_conn.commit()

    src_conn.close()
    logger.info(
        "Migration complete: %d trades, %d journal entries, %d errors",
        result["trades"],
        result["journal"],
        len(result["errors"]),
    )
    return result
