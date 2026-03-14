"""
Realistic CME futures slippage and commission model.

Per the todo.md blueprint, this module provides:
  - 1-tick slippage per side during RTH for retail order sizes
  - Time-of-day multipliers (RTH=1x, ETH/overnight=2x, events=1.5x)
  - Instrument-specific tick values and round-turn commission estimates
  - Break-even tick calculations per instrument
  - A unified cost calculator for backtesting and live P&L estimation

Slippage values (per side, 1 tick):
  ES  $12.50    MES  $1.25
  NQ  $5.00     MNQ  $0.50
  CL  $10.00    MCL  $1.00
  GC  $10.00    MGC  $1.00

Conservative all-in round-turn (commission + exchange + NFA + clearing + slippage):
  ES  ~$28-30   MES  ~$3.50
  NQ  ~$28      MNQ  ~$3.50
  CL  ~$30      MCL  ~$4.00
  GC  ~$30      MGC  ~$4.00

Usage:
    from lib.costs import get_cost_model, estimate_trade_costs, slippage_commission_rate

    # Get the cost model for an asset
    model = get_cost_model("Gold")
    print(model["slippage_per_side"])       # $1.00 for MGC
    print(model["round_turn_all_in"])       # $4.00 for MGC
    print(model["break_even_ticks"])        # ~4.0 ticks for MGC

    # Estimate full costs for a trade
    costs = estimate_trade_costs("Gold", contracts=5, session="rth")
    print(costs["total_cost"])              # total slippage + commissions

    # Get a commission rate suitable for backtesting.py
    rate = slippage_commission_rate("Gold")  # fractional cost per trade value
"""

from datetime import datetime
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Cost specifications per instrument (full-size contracts)
# ---------------------------------------------------------------------------

FULL_COST_SPECS = {
    "Gold": {
        "ticker": "GC=F",
        "tick_size": 0.10,
        "tick_value": 10.00,  # $10 per tick (100 per point, tick=0.10)
        "slippage_ticks": 1,  # 1 tick per side during RTH
        "slippage_per_side": 10.00,  # tick_value * slippage_ticks
        "commission_rt": 4.60,  # round-turn commission (broker)
        "exchange_rt": 3.40,  # exchange + NFA + clearing fees
        "round_turn_all_in": 28.00,  # commission + exchange + 2x slippage
        "break_even_ticks": 2.8,  # round_turn_all_in / tick_value
        "point_value": 100,
    },
    "Silver": {
        "ticker": "SI=F",
        "tick_size": 0.005,
        "tick_value": 25.00,  # $25 per tick (5000 per point, tick=0.005)
        "slippage_ticks": 1,
        "slippage_per_side": 25.00,
        "commission_rt": 4.60,
        "exchange_rt": 3.40,
        "round_turn_all_in": 58.00,
        "break_even_ticks": 2.3,
        "point_value": 5000,
    },
    "Copper": {
        "ticker": "HG=F",
        "tick_size": 0.0005,
        "tick_value": 12.50,  # $12.50 per tick (25000 per point, tick=0.0005)
        "slippage_ticks": 1,
        "slippage_per_side": 12.50,
        "commission_rt": 4.60,
        "exchange_rt": 3.40,
        "round_turn_all_in": 33.00,
        "break_even_ticks": 2.6,
        "point_value": 25000,
    },
    "Crude Oil": {
        "ticker": "CL=F",
        "tick_size": 0.01,
        "tick_value": 10.00,  # $10 per tick (1000 per point, tick=0.01)
        "slippage_ticks": 1,
        "slippage_per_side": 10.00,
        "commission_rt": 4.60,
        "exchange_rt": 5.40,
        "round_turn_all_in": 30.00,
        "break_even_ticks": 3.0,
        "point_value": 1000,
    },
    "S&P": {
        "ticker": "ES=F",
        "tick_size": 0.25,
        "tick_value": 12.50,  # $12.50 per tick (50 per point, tick=0.25)
        "slippage_ticks": 1,
        "slippage_per_side": 12.50,
        "commission_rt": 4.60,
        "exchange_rt": 3.16,
        "round_turn_all_in": 32.76,
        "break_even_ticks": 2.6,
        "point_value": 50,
    },
    "Nasdaq": {
        "ticker": "NQ=F",
        "tick_size": 0.25,
        "tick_value": 5.00,  # $5 per tick (20 per point, tick=0.25)
        "slippage_ticks": 1,
        "slippage_per_side": 5.00,
        "commission_rt": 4.60,
        "exchange_rt": 3.16,
        "round_turn_all_in": 17.76,
        "break_even_ticks": 3.6,
        "point_value": 20,
    },
}

# ---------------------------------------------------------------------------
# Cost specifications per instrument (micro contracts)
# ---------------------------------------------------------------------------

MICRO_COST_SPECS = {
    "Gold": {
        "ticker": "MGC=F",
        "tick_size": 0.10,
        "tick_value": 1.00,  # $1 per tick (10 per point, tick=0.10)
        "slippage_ticks": 1,
        "slippage_per_side": 1.00,
        "commission_rt": 1.60,
        "exchange_rt": 0.90,
        "round_turn_all_in": 4.50,
        "break_even_ticks": 4.5,
        "point_value": 10,
    },
    "Silver": {
        "ticker": "SIL=F",
        "tick_size": 0.005,
        "tick_value": 5.00,  # $5 per tick (1000 per point, tick=0.005)
        "slippage_ticks": 1,
        "slippage_per_side": 5.00,
        "commission_rt": 1.60,
        "exchange_rt": 0.90,
        "round_turn_all_in": 12.50,
        "break_even_ticks": 2.5,
        "point_value": 1000,
    },
    "Copper": {
        "ticker": "MHG=F",
        "tick_size": 0.0005,
        "tick_value": 1.25,  # $1.25 per tick (2500 per point, tick=0.0005)
        "slippage_ticks": 1,
        "slippage_per_side": 1.25,
        "commission_rt": 1.60,
        "exchange_rt": 0.90,
        "round_turn_all_in": 5.00,
        "break_even_ticks": 4.0,
        "point_value": 2500,
    },
    "Crude Oil": {
        "ticker": "MCL=F",
        "tick_size": 0.01,
        "tick_value": 1.00,  # $1 per tick (100 per point, tick=0.01)
        "slippage_ticks": 1,
        "slippage_per_side": 1.00,
        "commission_rt": 1.60,
        "exchange_rt": 0.90,
        "round_turn_all_in": 4.50,
        "break_even_ticks": 4.5,
        "point_value": 100,
    },
    "S&P": {
        "ticker": "MES=F",
        "tick_size": 0.25,
        "tick_value": 1.25,  # $1.25 per tick (5 per point, tick=0.25)
        "slippage_ticks": 1,
        "slippage_per_side": 1.25,
        "commission_rt": 1.30,
        "exchange_rt": 0.72,
        "round_turn_all_in": 4.52,
        "break_even_ticks": 3.6,
        "point_value": 5,
    },
    "Nasdaq": {
        "ticker": "MNQ=F",
        "tick_size": 0.25,
        "tick_value": 0.50,  # $0.50 per tick (2 per point, tick=0.25)
        "slippage_ticks": 1,
        "slippage_per_side": 0.50,
        "commission_rt": 1.30,
        "exchange_rt": 0.72,
        "round_turn_all_in": 3.02,
        "break_even_ticks": 6.0,
        "point_value": 2,
    },
}


# ---------------------------------------------------------------------------
# Time-of-day slippage multipliers
# ---------------------------------------------------------------------------

# RTH = Regular Trading Hours, ETH = Extended Trading Hours
SLIPPAGE_MULTIPLIERS = {
    "rth": 1.0,  # Regular session — tightest spreads
    "eth": 2.0,  # Overnight/extended — wider spreads
    "event": 1.5,  # High-volatility events (CPI, FOMC, NFP, EIA)
}

# Session boundaries (EST hours)
# RTH varies by product but these are reasonable defaults:
#   CME equity futures: 9:30 AM - 4:00 PM ET
#   CME metals/energy: varies, but RTH roughly 8:30 AM - 1:30 PM ET
# For simplicity, define RTH as 8:00 AM - 4:00 PM ET (covers most products)
RTH_START_HOUR = 8
RTH_END_HOUR = 16


def _get_session_type(dt: datetime | None = None) -> str:
    """Determine the current session type based on time of day.

    Returns "rth" during regular trading hours, "eth" otherwise.
    Event detection should be handled externally and passed explicitly.
    """
    if dt is None:
        dt = datetime.now(tz=_EST)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=_EST)
    else:
        dt = dt.astimezone(_EST)

    hour = dt.hour
    if RTH_START_HOUR <= hour < RTH_END_HOUR:
        return "rth"
    return "eth"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cost_model(
    asset_name: str,
    contract_mode: str = "micro",
) -> dict:
    """Return the cost model for a given asset.

    Args:
        asset_name: Asset name (e.g. "Gold", "S&P", "Crude Oil")
        contract_mode: "micro" or "full"

    Returns:
        Dict with tick_size, tick_value, slippage_per_side, commission_rt,
        exchange_rt, round_turn_all_in, break_even_ticks, point_value.
    """
    specs = MICRO_COST_SPECS if contract_mode == "micro" else FULL_COST_SPECS
    model = specs.get(asset_name)
    if model is None:
        # Fallback: generic conservative estimate
        return {
            "ticker": "UNKNOWN",
            "tick_size": 0.01,
            "tick_value": 1.00,
            "slippage_ticks": 1,
            "slippage_per_side": 1.00,
            "commission_rt": 2.00,
            "exchange_rt": 1.00,
            "round_turn_all_in": 5.00,
            "break_even_ticks": 5.0,
            "point_value": 1,
        }
    return dict(model)  # return a copy


def estimate_trade_costs(
    asset_name: str,
    contracts: int = 1,
    session: str = "rth",
    contract_mode: str = "micro",
) -> dict:
    """Estimate the full costs for a trade (entry + exit).

    Args:
        asset_name: Asset name
        contracts: Number of contracts
        session: "rth", "eth", or "event"
        contract_mode: "micro" or "full"

    Returns:
        Dict with slippage_total, commission_total, total_cost,
        cost_per_contract, break_even_move (in price units).
    """
    model = get_cost_model(asset_name, contract_mode)
    multiplier = SLIPPAGE_MULTIPLIERS.get(session, 1.0)

    # Slippage: per side × 2 sides × multiplier × contracts
    slippage_total = model["slippage_per_side"] * 2.0 * multiplier * contracts

    # Commission + exchange fees: round-turn per contract (already includes
    # slippage in the all-in figure, so subtract slippage to avoid double-counting)
    base_commission = (model["commission_rt"] + model["exchange_rt"]) * contracts
    total_cost = slippage_total + base_commission

    # Break-even move in price units
    point_value = model["point_value"]
    break_even_move = total_cost / (point_value * contracts) if point_value > 0 else 0

    return {
        "slippage_total": round(slippage_total, 2),
        "commission_total": round(base_commission, 2),
        "total_cost": round(total_cost, 2),
        "cost_per_contract": round(total_cost / max(contracts, 1), 2),
        "break_even_move": round(break_even_move, 4),
        "break_even_ticks": round(break_even_move / model["tick_size"] if model["tick_size"] > 0 else 0, 1),
        "session": session,
        "multiplier": multiplier,
    }


def slippage_commission_rate(
    asset_name: str,
    contract_mode: str = "micro",
    reference_price: float | None = None,
) -> float:
    """Compute a fractional commission rate for use with backtesting.py.

    backtesting.py's `commission` parameter is a fraction of trade value.
    We compute: all_in_cost / (reference_price * point_value) per side.

    If reference_price is not provided, uses approximate current prices.

    Returns a float suitable for `Backtest(..., commission=rate)`.
    """
    model = get_cost_model(asset_name, contract_mode)

    # Approximate reference prices for each instrument (mid-2025 ballpark)
    _approx_prices = {
        "Gold": 2700.0,
        "Silver": 32.0,
        "Copper": 4.5,
        "Crude Oil": 70.0,
        "S&P": 5800.0,
        "Nasdaq": 20000.0,
    }

    if reference_price is None:
        reference_price = _approx_prices.get(asset_name, 1000.0)

    # Trade notional value = price * point_value
    notional = reference_price * model["point_value"]
    if notional <= 0:
        return 0.0002  # fallback

    # Per-side cost = half of round-turn all-in
    per_side_cost = model["round_turn_all_in"] / 2.0

    # Commission rate as fraction of notional
    rate = per_side_cost / notional

    # Clamp to reasonable bounds
    return max(0.0001, min(rate, 0.01))


def format_cost_summary(asset_name: str, contract_mode: str = "micro") -> str:
    """Return a human-readable cost summary for display in the dashboard."""
    model = get_cost_model(asset_name, contract_mode)
    costs_rth = estimate_trade_costs(asset_name, 1, "rth", contract_mode)
    costs_eth = estimate_trade_costs(asset_name, 1, "eth", contract_mode)

    return (
        f"{asset_name} ({model['ticker']}): "
        f"Tick=${model['tick_value']:.2f}, "
        f"Slip/side=${model['slippage_per_side']:.2f}, "
        f"RT All-in=${costs_rth['total_cost']:.2f} (RTH) / "
        f"${costs_eth['total_cost']:.2f} (ETH), "
        f"Break-even={costs_rth['break_even_ticks']:.1f} ticks"
    )


def should_use_full_contracts(
    asset_name: str,
    micro_contracts: int,
) -> dict:
    """Advise whether to switch from micros to full-size contracts.

    Per the todo blueprint: switch to full contracts when consistently
    trading 10+ micros, because 10 MES costs ~$18 RT vs ~$5.76 for 1 ES.

    Returns a dict with recommendation and cost comparison.
    """
    micro_costs = estimate_trade_costs(asset_name, micro_contracts, "rth", "micro")

    # How many full contracts would this be? (micros are 1/10 of full, except Silver 1/5)
    micro_to_full = {"Silver": 5}.get(asset_name, 10)
    full_contracts = micro_contracts // micro_to_full
    remainder = micro_contracts % micro_to_full

    if full_contracts == 0:
        return {
            "recommend_full": False,
            "reason": f"Only {micro_contracts} micros — stay with micros",
            "micro_cost": micro_costs["total_cost"],
            "full_cost": None,
            "savings": 0,
        }

    full_costs = estimate_trade_costs(asset_name, full_contracts, "rth", "full")
    remainder_costs = (
        estimate_trade_costs(asset_name, remainder, "rth", "micro") if remainder > 0 else {"total_cost": 0}
    )

    mixed_cost = full_costs["total_cost"] + remainder_costs["total_cost"]
    savings = micro_costs["total_cost"] - mixed_cost

    return {
        "recommend_full": savings > 0 and full_contracts >= 1,
        "reason": (
            f"Switch to {full_contracts} full + {remainder} micro: save ${savings:.2f}/trade"
            if savings > 0
            else "Micro contracts are more cost-effective at this size"
        ),
        "micro_cost": micro_costs["total_cost"],
        "full_cost": mixed_cost,
        "savings": round(savings, 2),
        "full_contracts": full_contracts,
        "remainder_micros": remainder,
    }
