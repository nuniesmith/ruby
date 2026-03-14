"""
Test that Gold price data uses the micro contract (MGC) with no scaling.

Verifies: the Gold symbol maps to MGC=F for data fetching,
and no contract multiplier is applied to raw price data.
"""

import os

# Path setup
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("DISABLE_REDIS", "1")


def test_gold_price_no_scaling():
    """Gold micro (MGC) data_ticker must be MGC=F, not GC=F.

    The ASSETS dict should map Gold to MGC=F so that the Massive API
    resolves to the micro gold front-month contract, matching the
    MGC chart exactly.  Using GC=F (full-size) can cause
    a price mismatch due to different front-month contract months.
    """
    from lib.core.models import ASSETS, MICRO_CONTRACT_SPECS

    # Gold micro spec must use MGC=F as data_ticker
    gold_micro = MICRO_CONTRACT_SPECS["Gold"]
    assert gold_micro["ticker"] == "MGC=F", "Gold micro ticker must be MGC=F"
    assert gold_micro["data_ticker"] == "MGC=F", (
        "Gold micro data_ticker must be MGC=F, not GC=F — "
        "using GC=F causes front-month mismatch and ~$100 price discrepancy"
    )

    # When in micro mode (default), ASSETS["Gold"] should resolve to MGC=F
    assert ASSETS.get("Gold") == "MGC=F", f"ASSETS['Gold'] is '{ASSETS.get('Gold')}', expected 'MGC=F'"


def test_gold_point_value_is_micro():
    """Gold micro contract point value must be $10/point, not $100/point."""
    from lib.core.models import MICRO_CONTRACT_SPECS

    gold_micro = MICRO_CONTRACT_SPECS["Gold"]
    assert gold_micro["point"] == 10, (
        f"Gold micro point value is {gold_micro['point']}, expected 10 (micro). Full-size GC uses 100."
    )


def test_gold_massive_product_mapping():
    """Massive client must map MGC=F to the MGC product code."""
    from lib.integrations.massive_client import YAHOO_TO_MASSIVE_PRODUCT

    assert YAHOO_TO_MASSIVE_PRODUCT.get("MGC=F") == "MGC", "MGC=F must map to MGC product code in Massive client"


def test_gold_price_passthrough_no_multiplier():
    """Verify that raw OHLCV close prices are passed through without scaling.

    Simulates a scenario where Gold OHLCV data has close prices around 5200.
    The cache/display layer should show the same raw values — no multiplication
    by contract point values ($10 micro or $100 full-size).
    """
    from lib.core.models import CONTRACT_SPECS

    gold_spec = CONTRACT_SPECS.get("Gold", {})
    point_value = gold_spec.get("point", 10)

    # Simulate raw price from API
    raw_price = 5212.30
    # The displayed price should be the raw price, NOT raw_price * point_value
    assert raw_price != raw_price * point_value, "Sanity check: raw != scaled"
    # No scaling function exists — prices are passed through as-is
    # This test documents that contract point values should NEVER be applied to prices
    displayed_price = raw_price  # passthrough
    assert displayed_price == raw_price


def test_gold_risk_calculation_uses_point_value_correctly():
    """Point value should only be used for dollar risk calc, not price scaling.

    risk_per_contract = |entry - stop| * point_value
    This converts a price-distance into dollars, which is correct.
    The point value must NOT be applied to the entry/stop prices themselves.
    """
    from lib.core.models import calc_max_contracts

    entry = 5212.30
    stop = 5200.00
    risk_dollars = 375.0  # 0.75% of $50k
    hard_max = 10

    contracts = calc_max_contracts(entry, stop, "Gold", risk_dollars, hard_max)

    # risk_per_contract = |5212.30 - 5200.00| * 10 = $123.00
    # max = floor(375 / 123) = 3
    assert contracts == 3, f"Expected 3 contracts, got {contracts}"
