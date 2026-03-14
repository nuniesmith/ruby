# Indicators Library

A modular technical indicator library for futures/options trading analysis.

## Package Structure

```
indicators/
├── base.py              — abstract Indicator base class
├── registry.py          — IndicatorRegistry + @register_indicator decorator
├── factory.py           — IndicatorFactory (create by name/config)
├── manager.py           — IndicatorManager (batch calc, serialization)
├── helpers.py           — functional wrappers (one-liner API)
├── presets.py           — pre-configured indicator groups
├── trend/               — EMA, SMA, WMA, VWAP, MACD, ADLine
│   └── volatility/      — ATR, BollingerBands
├── momentum/            — RSI, Stochastic
├── volume/              — VolumeZoneOscillator, VWAPIndicator
├── other/               — 10 additional indicators
├── candle_patterns.py   — candlestick pattern identification
├── areas_of_interest.py — FVG, S/D zones, key levels
├── patterns.py          — PatternDetector class
├── indicators.py        — crypto-specific extensions
└── market_timing.py     — session analysis + should-trade-now logic
```

## Usage Examples

### Single indicator (direct)

```python
import pandas as pd
from lib.indicators.momentum.rsi import RSI

rsi = RSI(name="RSI", params={"period": 14})
result_df = rsi.calculate(df)  # returns DataFrame with RSI_14 column
print(rsi.get_value())         # latest RSI value
```

### Via factory

```python
from lib.indicators.factory import IndicatorFactory

ind = IndicatorFactory.create("rsi", period=14)
result = ind.calculate(df)
```

### Via IndicatorManager (batch)

```python
from lib.indicators.manager import IndicatorManager
from lib.indicators.momentum.rsi import RSI
from lib.indicators.trend.volatility.atr import ATR

mgr = IndicatorManager()
mgr.add_indicator(RSI(name="RSI", params={"period": 14}))
mgr.add_indicator(ATR(name="ATR", params={"period": 14}))
results = mgr.calculate_all(df)  # dict of {name: DataFrame}
```

### Functional helpers (one-liner API)

```python
from lib.indicators.helpers import ema, sma, rsi, atr, macd, bollinger, vwap

ema_series  = ema(df["close"], 21)
sma_series  = sma(df["close"], 50)
rsi_series  = rsi(df["close"], 14)
atr_series  = atr(df, 14)              # df needs high/low/close cols
macd_dict   = macd(df["close"])        # {"macd_line", "signal_line", "histogram"}
bb_dict     = bollinger(df["close"])   # {"middle", "upper", "lower", "bandwidth", "percent_b"}
vwap_series = vwap(df)                 # df needs high/low/close/volume cols
```

### Presets (pre-configured groups)

```python
from lib.indicators.presets import SCALP_PRESET, SWING_PRESET, REGIME_PRESET, build_manager

mgr = build_manager(SCALP_PRESET)
results = mgr.calculate_all(df)
```

Available presets:

| Preset | Bars | Indicators |
|---|---|---|
| `SCALP_PRESET` | 1m / 5m | EMA(9), EMA(21), RSI(14), ATR(14), VWAP |
| `SWING_PRESET` | 15m / 1H | EMA(21/50/200), MACD, RSI(14), ATR(14), BollingerBands(20) |
| `REGIME_PRESET` | Any | ATR(14), BollingerBands(20), ChoppinessIndex(14) |

## Column Naming

Registry-based indicators (`RSI`, `EMA`, `ATR`, etc.) use **lowercase** columns:

- `close`, `high`, `low`, `open`, `volume`

Standalone indicators (`EMAIndicator`, `VWAPIndicator`, `ChoppinessIndex`) use **capitalised** columns:

- `Close`, `High`, `Low`, `Open`, `Volume`

The `helpers.py` functions accept both (they do case-insensitive column lookup for OHLCV).

When using `build_manager()` with a preset, all indicator classes must follow the
`Indicator.__init__(name, params)` signature. Standalone classes that do not conform
(e.g. the raw `ChoppinessIndex`) require an adapter wrapper — see the
`_ChoppinessIndexAdapter` in `presets.py` for a reference implementation.

## Adding a New Indicator

1. Create `indicators/<category>/my_indicator.py`
2. Extend `Indicator` from `lib.indicators.base`
3. Decorate with `@register_indicator`
4. Implement `calculate(self, data, price_column="close") -> pd.DataFrame`
5. Add import to `indicators/__init__.py`

Example skeleton:

```python
from typing import Any, Dict, List, Optional
import pandas as pd
from lib.indicators.base import Indicator
from lib.indicators.registry import register_indicator

@register_indicator
class MyIndicator(Indicator):
    def __init__(self, name: str = "MyIndicator", params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)
        self.period = self.params.get("period", 14)

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        result = data[price_column].rolling(self.period).mean()  # placeholder
        return pd.DataFrame({f"{self.name}_{self.period}": result}, index=data.index)

    @classmethod
    def required_columns(cls) -> List[str]:
        return ["close"]
```
