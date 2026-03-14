"""
Unified Asset Registry
======================
Single source of truth for all tradeable assets across micro, full-size,
and spot variants.  Replaces the scattered MICRO_CONTRACT_SPECS,
FULL_CONTRACT_SPECS, KRAKEN_CONTRACT_SPECS lookups with a unified
registry keyed by generalized asset name ("Gold", "S&P", "Bitcoin", etc.).

Usage:
    from lib.core.asset_registry import ASSET_REGISTRY, get_asset, get_asset_by_ticker

    gold = get_asset("Gold")
    gold.micro.ticker        # "MGC=F"
    gold.full.ticker         # "GC=F"
    gold.micro.point_value   # 10.0
    gold.full.point_value    # 100.0
    gold.asset_class         # AssetClass.METALS

    # Reverse lookup from any ticker variant
    asset = get_asset_by_ticker("MGC=F")   # → Gold Asset
    asset = get_asset_by_ticker("GC=F")    # → Gold Asset
    asset = get_asset_by_ticker("KRAKEN:XBTUSD")  # → Bitcoin Asset

    # Get all assets in a class
    metals = get_asset_group(AssetClass.METALS)  # [Gold, Silver, Copper]

    # Backward-compatible CONTRACT_SPECS access
    from lib.core.asset_registry import get_micro_spec, get_full_spec
    spec = get_micro_spec("Gold")  # {"ticker": "MGC=F", "point": 10, ...}
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

logger = logging.getLogger("core.asset_registry")


# ---------------------------------------------------------------------------
# Asset class taxonomy
# ---------------------------------------------------------------------------
class AssetClass(enum.Enum):
    """High-level asset class grouping."""

    METALS = "metals"
    ENERGY = "energy"
    EQUITY_INDEX = "equity_index"
    FX = "fx"
    TREASURIES = "treasuries"
    AGRICULTURE = "agriculture"
    CRYPTO = "crypto"


# CNN ordinal mapping (matches breakout_cnn.py ASSET_CLASS_ORDINALS)
ASSET_CLASS_ORDINAL: dict[AssetClass, int] = {
    AssetClass.EQUITY_INDEX: 0,
    AssetClass.FX: 1,
    AssetClass.METALS: 2,
    AssetClass.ENERGY: 2,  # metals/energy share ordinal in CNN v6
    AssetClass.TREASURIES: 3,
    AssetClass.AGRICULTURE: 3,  # treasuries/ags share ordinal in CNN v6
    AssetClass.CRYPTO: 4,
}


# ---------------------------------------------------------------------------
# Contract variant — one specific tradeable instrument
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ContractVariant:
    """A single tradeable contract variant of an asset."""

    ticker: str  # Trading ticker: "MGC=F", "GC=F", "KRAKEN:XBTUSD"
    data_ticker: str  # Data-fetch ticker (may differ from trading ticker)
    point_value: float  # Dollar value per 1.0 price point move
    tick_size: float  # Minimum price increment
    margin: float  # Approximate initial margin requirement ($)
    exchange: str = "CME"  # Exchange identifier
    full_micro_ratio: float = 1.0  # Ratio to micro (1.0 for micro, 10.0 for GC vs MGC)

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick move."""
        return self.tick_size * self.point_value


# ---------------------------------------------------------------------------
# Asset — generalized asset linking all variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    """Unified asset representation linking micro, full, and spot variants."""

    name: str  # Human-readable: "Gold", "S&P", "Bitcoin"
    asset_class: AssetClass
    micro: ContractVariant | None = None  # Micro futures contract
    full: ContractVariant | None = None  # Full-size futures contract
    spot: ContractVariant | None = None  # Spot (Kraken crypto, etc.)
    peers: tuple[str, ...] = ()  # Related asset names for cross-correlation
    primary_sessions: tuple[str, ...] = ()  # Best trading sessions

    @property
    def cnn_ordinal(self) -> int:
        """CNN asset_class_id ordinal for feature vector."""
        return ASSET_CLASS_ORDINAL.get(self.asset_class, 0)

    @property
    def primary_ticker(self) -> str:
        """Preferred ticker: micro > full > spot."""
        if self.micro:
            return self.micro.ticker
        if self.full:
            return self.full.ticker
        if self.spot:
            return self.spot.ticker
        return ""

    @property
    def data_ticker(self) -> str:
        """Preferred data ticker: micro > full > spot."""
        if self.micro:
            return self.micro.data_ticker
        if self.full:
            return self.full.data_ticker
        if self.spot:
            return self.spot.data_ticker
        return ""

    @property
    def variants(self) -> dict[str, ContractVariant]:
        """All available contract variants."""
        result: dict[str, ContractVariant] = {}
        if self.micro:
            result["micro"] = self.micro
        if self.full:
            result["full"] = self.full
        if self.spot:
            result["spot"] = self.spot
        return result

    def all_tickers(self) -> set[str]:
        """Return set of ALL tickers across all variants (trading + data)."""
        tickers: set[str] = set()
        for v in self.variants.values():
            tickers.add(v.ticker)
            tickers.add(v.data_ticker)
        return tickers

    def compute_position_size(
        self,
        variant_key: str,
        entry_price: float,
        stop_price: float,
        max_risk_dollars: float,
    ) -> tuple[int, float]:
        """Compute number of contracts and actual risk dollars.

        Args:
            variant_key: "micro", "full", or "spot"
            entry_price: Intended entry price
            stop_price: Stop loss price
            max_risk_dollars: Maximum dollars to risk

        Returns:
            (num_contracts, risk_dollars)
        """
        variant = self.variants.get(variant_key)
        if not variant or variant.tick_size <= 0 or variant.point_value <= 0:
            return 0, 0.0

        stop_distance = abs(entry_price - stop_price)
        risk_per_contract = stop_distance * variant.point_value
        if risk_per_contract <= 0:
            return 1, 0.0

        num_contracts = max(1, int(max_risk_dollars / risk_per_contract))
        actual_risk = num_contracts * risk_per_contract
        return num_contracts, round(actual_risk, 2)

    def dual_sizing(
        self,
        entry_price: float,
        stop_price: float,
        max_risk_dollars: float,
    ) -> dict[str, dict]:
        """Compute position size for BOTH micro and full contracts side by side.

        Returns dict with "micro" and "full" keys, each containing:
          contracts, risk_dollars, tp1_dollars (at 2R), tp2_dollars (at 3R)
        """
        result = {}
        for key in ("micro", "full"):
            variant = self.variants.get(key)
            if not variant:
                continue
            contracts, risk = self.compute_position_size(key, entry_price, stop_price, max_risk_dollars)
            stop_dist = abs(entry_price - stop_price)
            pnl_per_contract_per_point = variant.point_value
            result[key] = {
                "symbol": variant.ticker,
                "contracts": contracts,
                "risk_dollars": risk,
                "tp1_dollars": round(contracts * stop_dist * 2 * pnl_per_contract_per_point, 2),
                "tp2_dollars": round(contracts * stop_dist * 3 * pnl_per_contract_per_point, 2),
            }
        return result


# ---------------------------------------------------------------------------
# Registry definition — all assets
# ---------------------------------------------------------------------------
def _build_registry() -> dict[str, Asset]:
    """Build the complete asset registry."""
    registry: dict[str, Asset] = {}

    # ── Metals ──────────────────────────────────────────────────────────
    registry["Gold"] = Asset(
        name="Gold",
        asset_class=AssetClass.METALS,
        micro=ContractVariant(
            ticker="MGC=F",
            data_ticker="MGC=F",
            point_value=10.0,
            tick_size=0.10,
            margin=1_100,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="GC=F",
            data_ticker="GC=F",
            point_value=100.0,
            tick_size=0.10,
            margin=11_000,
            full_micro_ratio=10.0,
        ),
        peers=("Silver", "Copper"),
        primary_sessions=("london", "us"),
    )

    registry["Silver"] = Asset(
        name="Silver",
        asset_class=AssetClass.METALS,
        micro=ContractVariant(
            ticker="SIL=F",
            data_ticker="SI=F",
            point_value=1_000.0,
            tick_size=0.005,
            margin=1_800,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="SI=F",
            data_ticker="SI=F",
            point_value=5_000.0,
            tick_size=0.005,
            margin=9_000,
            full_micro_ratio=5.0,
        ),
        peers=("Gold", "Copper"),
        primary_sessions=("london", "us"),
    )

    registry["Copper"] = Asset(
        name="Copper",
        asset_class=AssetClass.METALS,
        micro=ContractVariant(
            ticker="MHG=F",
            data_ticker="HG=F",
            point_value=2_500.0,
            tick_size=0.0005,
            margin=600,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="HG=F",
            data_ticker="HG=F",
            point_value=25_000.0,
            tick_size=0.0005,
            margin=6_000,
            full_micro_ratio=10.0,
        ),
        peers=("Gold", "Silver"),
        primary_sessions=("london", "us"),
    )

    # ── Energy ──────────────────────────────────────────────────────────
    registry["Crude Oil"] = Asset(
        name="Crude Oil",
        asset_class=AssetClass.ENERGY,
        micro=ContractVariant(
            ticker="MCL=F",
            data_ticker="CL=F",
            point_value=100.0,
            tick_size=0.01,
            margin=700,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="CL=F",
            data_ticker="CL=F",
            point_value=1_000.0,
            tick_size=0.01,
            margin=7_000,
            full_micro_ratio=10.0,
        ),
        peers=("Natural Gas",),
        primary_sessions=("us", "london_ny"),
    )

    registry["Natural Gas"] = Asset(
        name="Natural Gas",
        asset_class=AssetClass.ENERGY,
        micro=ContractVariant(
            ticker="MNG=F",
            data_ticker="NG=F",
            point_value=1_000.0,
            tick_size=0.001,
            margin=350,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="NG=F",
            data_ticker="NG=F",
            point_value=10_000.0,
            tick_size=0.001,
            margin=3_500,
            full_micro_ratio=10.0,
        ),
        peers=("Crude Oil",),
        primary_sessions=("us",),
    )

    # ── Equity Index ────────────────────────────────────────────────────
    registry["S&P"] = Asset(
        name="S&P",
        asset_class=AssetClass.EQUITY_INDEX,
        micro=ContractVariant(
            ticker="MES=F",
            data_ticker="ES=F",
            point_value=5.0,
            tick_size=0.25,
            margin=1_500,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ES=F",
            data_ticker="ES=F",
            point_value=50.0,
            tick_size=0.25,
            margin=12_000,
            full_micro_ratio=10.0,
        ),
        peers=("Nasdaq", "Russell 2000", "Dow Jones"),
        primary_sessions=("us", "cme"),
    )

    registry["Nasdaq"] = Asset(
        name="Nasdaq",
        asset_class=AssetClass.EQUITY_INDEX,
        micro=ContractVariant(
            ticker="MNQ=F",
            data_ticker="NQ=F",
            point_value=2.0,
            tick_size=0.25,
            margin=2_100,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="NQ=F",
            data_ticker="NQ=F",
            point_value=20.0,
            tick_size=0.25,
            margin=17_000,
            full_micro_ratio=10.0,
        ),
        peers=("S&P", "Russell 2000", "Dow Jones"),
        primary_sessions=("us", "cme"),
    )

    registry["Russell 2000"] = Asset(
        name="Russell 2000",
        asset_class=AssetClass.EQUITY_INDEX,
        micro=ContractVariant(
            ticker="M2K=F",
            data_ticker="RTY=F",
            point_value=5.0,
            tick_size=0.10,
            margin=1_200,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="RTY=F",
            data_ticker="RTY=F",
            point_value=50.0,
            tick_size=0.10,
            margin=8_000,
            full_micro_ratio=10.0,
        ),
        peers=("S&P", "Nasdaq", "Dow Jones"),
        primary_sessions=("us",),
    )

    registry["Dow Jones"] = Asset(
        name="Dow Jones",
        asset_class=AssetClass.EQUITY_INDEX,
        micro=ContractVariant(
            ticker="MYM=F",
            data_ticker="YM=F",
            point_value=0.5,
            tick_size=1.0,
            margin=1_100,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="YM=F",
            data_ticker="YM=F",
            point_value=5.0,
            tick_size=1.0,
            margin=9_000,
            full_micro_ratio=10.0,
        ),
        peers=("S&P", "Nasdaq", "Russell 2000"),
        primary_sessions=("us",),
    )

    # ── FX Futures ──────────────────────────────────────────────────────
    registry["Euro FX"] = Asset(
        name="Euro FX",
        asset_class=AssetClass.FX,
        micro=ContractVariant(
            ticker="M6E=F",
            data_ticker="6E=F",
            point_value=12_500.0,
            tick_size=0.0001,
            margin=280,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6E=F",
            data_ticker="6E=F",
            point_value=125_000.0,
            tick_size=0.00005,
            margin=2_800,
            full_micro_ratio=10.0,
        ),
        peers=("British Pound", "Swiss Franc"),
        primary_sessions=("london", "frankfurt", "london_ny"),
    )

    registry["British Pound"] = Asset(
        name="British Pound",
        asset_class=AssetClass.FX,
        micro=ContractVariant(
            ticker="M6B=F",
            data_ticker="6B=F",
            point_value=6_250.0,
            tick_size=0.0001,
            margin=260,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6B=F",
            data_ticker="6B=F",
            point_value=62_500.0,
            tick_size=0.0001,
            margin=2_600,
            full_micro_ratio=10.0,
        ),
        peers=("Euro FX",),
        primary_sessions=("london", "london_ny"),
    )

    registry["Japanese Yen"] = Asset(
        name="Japanese Yen",
        asset_class=AssetClass.FX,
        # No genuine micro — use standard contract as both
        micro=ContractVariant(
            ticker="6J=F",
            data_ticker="6J=F",
            point_value=12_500_000.0,
            tick_size=0.0000005,
            margin=2_400,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6J=F",
            data_ticker="6J=F",
            point_value=12_500_000.0,
            tick_size=0.0000005,
            margin=2_400,
            full_micro_ratio=1.0,
        ),
        peers=("Euro FX", "Australian Dollar"),
        primary_sessions=("tokyo", "london"),
    )

    registry["Australian Dollar"] = Asset(
        name="Australian Dollar",
        asset_class=AssetClass.FX,
        micro=ContractVariant(
            ticker="6A=F",
            data_ticker="6A=F",
            point_value=100_000.0,
            tick_size=0.0001,
            margin=1_800,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6A=F",
            data_ticker="6A=F",
            point_value=100_000.0,
            tick_size=0.0001,
            margin=1_800,
            full_micro_ratio=1.0,
        ),
        peers=("Canadian Dollar",),
        primary_sessions=("sydney", "tokyo", "london"),
    )

    registry["Canadian Dollar"] = Asset(
        name="Canadian Dollar",
        asset_class=AssetClass.FX,
        micro=ContractVariant(
            ticker="6C=F",
            data_ticker="6C=F",
            point_value=100_000.0,
            tick_size=0.0001,
            margin=1_600,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6C=F",
            data_ticker="6C=F",
            point_value=100_000.0,
            tick_size=0.0001,
            margin=1_600,
            full_micro_ratio=1.0,
        ),
        peers=("Australian Dollar",),
        primary_sessions=("us", "london_ny"),
    )

    registry["Swiss Franc"] = Asset(
        name="Swiss Franc",
        asset_class=AssetClass.FX,
        micro=ContractVariant(
            ticker="6S=F",
            data_ticker="6S=F",
            point_value=125_000.0,
            tick_size=0.0001,
            margin=3_000,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="6S=F",
            data_ticker="6S=F",
            point_value=125_000.0,
            tick_size=0.0001,
            margin=3_000,
            full_micro_ratio=1.0,
        ),
        peers=("Euro FX",),
        primary_sessions=("london", "frankfurt"),
    )

    # ── Treasuries ──────────────────────────────────────────────────────
    registry["10Y T-Note"] = Asset(
        name="10Y T-Note",
        asset_class=AssetClass.TREASURIES,
        # No micro — use standard contract
        micro=ContractVariant(
            ticker="ZN=F",
            data_ticker="ZN=F",
            point_value=1_000.0,
            tick_size=0.015625,
            margin=1_800,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ZN=F",
            data_ticker="ZN=F",
            point_value=1_000.0,
            tick_size=0.015625,
            margin=1_800,
            full_micro_ratio=1.0,
        ),
        peers=("30Y T-Bond",),
        primary_sessions=("us",),
    )

    registry["30Y T-Bond"] = Asset(
        name="30Y T-Bond",
        asset_class=AssetClass.TREASURIES,
        micro=ContractVariant(
            ticker="ZB=F",
            data_ticker="ZB=F",
            point_value=1_000.0,
            tick_size=0.03125,
            margin=3_200,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ZB=F",
            data_ticker="ZB=F",
            point_value=1_000.0,
            tick_size=0.03125,
            margin=3_200,
            full_micro_ratio=1.0,
        ),
        peers=("10Y T-Note",),
        primary_sessions=("us",),
    )

    # ── Agriculture ─────────────────────────────────────────────────────
    registry["Corn"] = Asset(
        name="Corn",
        asset_class=AssetClass.AGRICULTURE,
        micro=ContractVariant(
            ticker="ZC=F",
            data_ticker="ZC=F",
            point_value=50.0,
            tick_size=0.25,
            margin=1_200,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ZC=F",
            data_ticker="ZC=F",
            point_value=50.0,
            tick_size=0.25,
            margin=1_200,
            full_micro_ratio=1.0,
        ),
        peers=("Soybeans", "Wheat"),
        primary_sessions=("us",),
    )

    registry["Soybeans"] = Asset(
        name="Soybeans",
        asset_class=AssetClass.AGRICULTURE,
        micro=ContractVariant(
            ticker="ZS=F",
            data_ticker="ZS=F",
            point_value=50.0,
            tick_size=0.25,
            margin=2_200,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ZS=F",
            data_ticker="ZS=F",
            point_value=50.0,
            tick_size=0.25,
            margin=2_200,
            full_micro_ratio=1.0,
        ),
        peers=("Corn", "Wheat"),
        primary_sessions=("us",),
    )

    registry["Wheat"] = Asset(
        name="Wheat",
        asset_class=AssetClass.AGRICULTURE,
        micro=ContractVariant(
            ticker="ZW=F",
            data_ticker="ZW=F",
            point_value=50.0,
            tick_size=0.25,
            margin=1_700,
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ZW=F",
            data_ticker="ZW=F",
            point_value=50.0,
            tick_size=0.25,
            margin=1_700,
            full_micro_ratio=1.0,
        ),
        peers=("Corn", "Soybeans"),
        primary_sessions=("us",),
    )

    # ── Crypto (CME Futures + Kraken Spot) ──────────────────────────────
    registry["Bitcoin"] = Asset(
        name="Bitcoin",
        asset_class=AssetClass.CRYPTO,
        micro=ContractVariant(
            ticker="MBT=F",
            data_ticker="MBT=F",
            point_value=0.1,
            tick_size=5.0,
            margin=8_000,
            exchange="CME",
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="BTC=F",
            data_ticker="BTC=F",
            point_value=5.0,
            tick_size=5.0,
            margin=80_000,
            exchange="CME",
            full_micro_ratio=50.0,
        ),
        spot=ContractVariant(
            ticker="KRAKEN:XBTUSD",
            data_ticker="KRAKEN:XBTUSD",
            point_value=1.0,
            tick_size=0.1,
            margin=5_000,
            exchange="Kraken",
        ),
        peers=("Ethereum",),
        primary_sessions=("us", "london", "tokyo"),
    )

    registry["Ethereum"] = Asset(
        name="Ethereum",
        asset_class=AssetClass.CRYPTO,
        micro=ContractVariant(
            ticker="MET=F",
            data_ticker="MET=F",
            point_value=0.1,
            tick_size=0.25,
            margin=700,
            exchange="CME",
            full_micro_ratio=1.0,
        ),
        full=ContractVariant(
            ticker="ETH=F",
            data_ticker="ETH=F",
            point_value=50.0,
            tick_size=0.25,
            margin=6_500,
            exchange="CME",
            full_micro_ratio=500.0,
        ),
        spot=ContractVariant(
            ticker="KRAKEN:ETHUSD",
            data_ticker="KRAKEN:ETHUSD",
            point_value=1.0,
            tick_size=0.01,
            margin=500,
            exchange="Kraken",
        ),
        peers=("Bitcoin",),
        primary_sessions=("us", "london", "tokyo"),
    )

    # ── Kraken-only crypto (no CME futures equivalent) ──────────────────
    _kraken_only: list[tuple[str, str, str, float, float, tuple[str, ...]]] = [
        # (name, kraken_ticker, kraken_pair, tick_size, margin, peers)
        ("Solana", "KRAKEN:SOLUSD", "SOL/USD", 0.001, 50, ("Bitcoin", "Ethereum")),
        ("Chainlink", "KRAKEN:LINKUSD", "LINK/USD", 0.001, 25, ("Ethereum",)),
        ("Avalanche", "KRAKEN:AVAXUSD", "AVAX/USD", 0.001, 30, ("Ethereum", "Solana")),
        ("Polkadot", "KRAKEN:DOTUSD", "DOT/USD", 0.0001, 15, ("Ethereum",)),
        ("Cardano", "KRAKEN:ADAUSD", "ADA/USD", 0.00001, 10, ("Ethereum", "Polkadot")),
        ("Polygon", "KRAKEN:POLUSD", "POL/USD", 0.0001, 10, ("Ethereum",)),
        ("XRP", "KRAKEN:XRPUSD", "XRP/USD", 0.0001, 10, ("Bitcoin",)),
    ]

    for name, ticker, _pair, tick, margin, peers in _kraken_only:
        registry[name] = Asset(
            name=name,
            asset_class=AssetClass.CRYPTO,
            spot=ContractVariant(
                ticker=ticker,
                data_ticker=ticker,
                point_value=1.0,
                tick_size=tick,
                margin=margin,
                exchange="Kraken",
            ),
            peers=peers,
            primary_sessions=("us", "london", "tokyo"),
        )

    return registry


# ---------------------------------------------------------------------------
# Module-level singleton registry
# ---------------------------------------------------------------------------
ASSET_REGISTRY: dict[str, Asset] = _build_registry()

# Build reverse lookup: any ticker → asset name
_TICKER_TO_NAME: dict[str, str] = {}
for _asset_name, _asset in ASSET_REGISTRY.items():
    for _t in _asset.all_tickers():
        if _t:
            _TICKER_TO_NAME[_t] = _asset_name

# Also add Kraken short aliases used by the training pipeline
_KRAKEN_SHORT_ALIASES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "LINK": "Chainlink",
    "AVAX": "Avalanche",
    "DOT": "Polkadot",
    "ADA": "Cardano",
    "POL": "Polygon",
    "XRP": "XRP",
}
_TICKER_TO_NAME.update(_KRAKEN_SHORT_ALIASES)

# Also add training-pipeline short symbols for CME micros
_TRAINING_SHORT_ALIASES: dict[str, str] = {
    "MGC": "Gold",
    "SIL": "Silver",
    "MHG": "Copper",
    "MCL": "Crude Oil",
    "MNG": "Natural Gas",
    "MES": "S&P",
    "MNQ": "Nasdaq",
    "M2K": "Russell 2000",
    "MYM": "Dow Jones",
    "6E": "Euro FX",
    "6B": "British Pound",
    "6J": "Japanese Yen",
    "6A": "Australian Dollar",
    "6C": "Canadian Dollar",
    "6S": "Swiss Franc",
    "ZN": "10Y T-Note",
    "ZB": "30Y T-Bond",
    "ZC": "Corn",
    "ZS": "Soybeans",
    "ZW": "Wheat",
    "MBT": "Bitcoin",
    "MET": "Ethereum",
}
_TICKER_TO_NAME.update(_TRAINING_SHORT_ALIASES)

# Legacy contract-month aliases (used by data connectors)
_LEGACY_ALIASES: dict[str, str] = {
    "MGCZ4": "Gold",
    "MGCH5": "Gold",
    "MGCM5": "Gold",
    "MGCQ5": "Gold",
    "MESZ4": "S&P",
    "MESH5": "S&P",
    "MESM5": "S&P",
    "MESQ5": "S&P",
    "MNQZ4": "Nasdaq",
    "MNQH5": "Nasdaq",
    "MNQM5": "Nasdaq",
    "MNQQ5": "Nasdaq",
    "MYMZ4": "Dow Jones",
    "MYMH5": "Dow Jones",
    "MYMM5": "Dow Jones",
    "M2KZ4": "Russell 2000",
    "M2KH5": "Russell 2000",
    "M2KM5": "Russell 2000",
    "MCLZ4": "Crude Oil",
    "MCLF5": "Crude Oil",
    "MCLG5": "Crude Oil",
    "SILZ4": "Silver",
    "SILH5": "Silver",
    "6EZ4": "Euro FX",
    "6EH5": "Euro FX",
    "6EM5": "Euro FX",
    "6BZ4": "British Pound",
    "6BH5": "British Pound",
}
_TICKER_TO_NAME.update(_LEGACY_ALIASES)


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------
def get_asset(name: str) -> Asset | None:
    """Look up an asset by its generalized name (e.g. 'Gold', 'S&P')."""
    return ASSET_REGISTRY.get(name)


def get_asset_by_ticker(ticker: str) -> Asset | None:
    """Reverse lookup: find the Asset for ANY ticker variant.

    Works with: MGC=F, GC=F, KRAKEN:XBTUSD, MGC, MES, BTC, etc.
    """
    name = _TICKER_TO_NAME.get(ticker)
    if name:
        return ASSET_REGISTRY.get(name)

    # Try stripping =F suffix
    if ticker.endswith("=F"):
        name = _TICKER_TO_NAME.get(ticker[:-2])
        if name:
            return ASSET_REGISTRY.get(name)

    return None


def get_asset_name_by_ticker(ticker: str) -> str:
    """Return the generalized name for a ticker, or the ticker itself if not found."""
    asset = get_asset_by_ticker(ticker)
    return asset.name if asset else ticker


def get_asset_group(asset_class: AssetClass) -> list[Asset]:
    """Return all assets in a given asset class."""
    return [a for a in ASSET_REGISTRY.values() if a.asset_class == asset_class]


def get_asset_group_names(asset_class: AssetClass) -> list[str]:
    """Return names of all assets in a given asset class."""
    return [a.name for a in ASSET_REGISTRY.values() if a.asset_class == asset_class]


def get_futures_assets() -> list[Asset]:
    """Return all assets that have at least a micro or full futures variant."""
    return [a for a in ASSET_REGISTRY.values() if a.micro or a.full]


def get_crypto_spot_assets() -> list[Asset]:
    """Return all assets that have a spot (Kraken) variant."""
    return [a for a in ASSET_REGISTRY.values() if a.spot]


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------
def get_micro_spec(name: str) -> dict | None:
    """Return micro contract spec dict, matching old MICRO_CONTRACT_SPECS format."""
    asset = ASSET_REGISTRY.get(name)
    if not asset or not asset.micro:
        return None
    v = asset.micro
    return {
        "ticker": v.ticker,
        "data_ticker": v.data_ticker,
        "point": v.point_value,
        "tick": v.tick_size,
        "margin": v.margin,
    }


def get_full_spec(name: str) -> dict | None:
    """Return full-size contract spec dict, matching old FULL_CONTRACT_SPECS format."""
    asset = ASSET_REGISTRY.get(name)
    if not asset or not asset.full:
        return None
    v = asset.full
    return {
        "ticker": v.ticker,
        "data_ticker": v.data_ticker,
        "point": v.point_value,
        "tick": v.tick_size,
        "margin": v.margin,
    }


def get_all_data_tickers() -> set[str]:
    """Return set of all data tickers across all assets and variants."""
    tickers: set[str] = set()
    for asset in ASSET_REGISTRY.values():
        for v in asset.variants.values():
            tickers.add(v.data_ticker)
    return tickers


def total_margin_for_assets(names: list[str], variant: str = "micro") -> float:
    """Sum of margin requirements for a list of assets."""
    total = 0.0
    for name in names:
        asset = ASSET_REGISTRY.get(name)
        if asset:
            v = asset.variants.get(variant)
            if v:
                total += v.margin
    return total
