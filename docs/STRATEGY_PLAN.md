# Futures Breakout System — Strategy Plan

## Table of Contents

1. [Asset Review & Recommendation](#1-asset-review--recommendation)
2. [Model Versatility Architecture](#2-model-versatility-architecture)
3. [Stop-and-Reverse Strategy with Micro Contracts](#3-stop-and-reverse-strategy-with-micro-contracts)
4. [Order Management & Take Profit Ladder](#4-order-management--take-profit-ladder)
5. [EMA9 Trailing Logic](#5-ema9-trailing-logic)
6. [Multi-Timeframe Support](#6-multi-timeframe-support)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Asset Review & Recommendation

### Current Universe (22 Micro Futures + 9 Kraken Crypto)

| # | Asset | Ticker | Margin | Sector | Liquidity | Spread | ORB Quality | Rec |
|---|-------|--------|--------|--------|-----------|--------|-------------|-----|
| 1 | Gold | MGC | $1,100 | Metals | ★★★★★ | Tight | Excellent | ✅ **KEEP** |
| 2 | Silver | SIL | $1,800 | Metals | ★★★☆☆ | Medium | Good | ⚠️ Watch |
| 3 | Copper | MHG | $600 | Metals | ★★☆☆☆ | Wide | Fair | ❌ Drop |
| 4 | Crude Oil | MCL | $700 | Energy | ★★★★★ | Tight | Excellent | ✅ **KEEP** |
| 5 | Natural Gas | MNG | $350 | Energy | ★★☆☆☆ | Wide | Poor (gaps) | ❌ Drop |
| 6 | S&P 500 | MES | $1,500 | Index | ★★★★★ | Tight | Excellent | ✅ **KEEP** |
| 7 | Nasdaq | MNQ | $2,100 | Index | ★★★★★ | Tight | Excellent | ✅ **KEEP** |
| 8 | Russell 2000 | M2K | $1,200 | Index | ★★★★☆ | Tight | Good | ⚠️ Watch |
| 9 | Dow Jones | MYM | $1,100 | Index | ★★★★☆ | Tight | Good | ⚠️ Watch |
| 10 | Euro FX | M6E | $280 | FX | ★★★★☆ | Tight | Good | ✅ **KEEP** |
| 11 | British Pound | M6B | $260 | FX | ★★★☆☆ | Medium | Good | ⚠️ Watch |
| 12 | Japanese Yen | 6J | $2,400 | FX | ★★★★☆ | Tight | Good | ⚠️ Watch |
| 13 | Australian Dollar | 6A | $1,800 | FX | ★★☆☆☆ | Medium | Fair | ❌ Drop |
| 14 | Canadian Dollar | 6C | $1,600 | FX | ★★☆☆☆ | Medium | Fair | ❌ Drop |
| 15 | Swiss Franc | 6S | $3,000 | FX | ★★☆☆☆ | Medium | Fair | ❌ Drop |
| 16 | 10Y T-Note | ZN | $1,800 | Rates | ★★★★☆ | Tight | Good | ⚠️ Watch |
| 17 | 30Y T-Bond | ZB | $3,200 | Rates | ★★★☆☆ | Medium | Fair | ❌ Drop |
| 18 | Corn | ZC | $1,200 | Ags | ★★☆☆☆ | Medium | Poor | ❌ Drop |
| 19 | Soybeans | ZS | $2,200 | Ags | ★★☆☆☆ | Medium | Poor | ❌ Drop |
| 20 | Wheat | ZW | $1,700 | Ags | ★★☆☆☆ | Wide | Poor | ❌ Drop |
| 21 | Micro Bitcoin | MBT | $8,000 | Crypto | ★★★☆☆ | Medium | Good | ⚠️ Watch |
| 22 | Micro Ether | MET | $700 | Crypto | ★★★☆☆ | Medium | Good | ⚠️ Watch |

### Recommended Core Watchlist (5 Assets for Live Trading)

These are the assets you should actively trade with the stop-and-reverse
micro contract strategy. They share these properties:

- **Tight spreads** — cost of entry/exit is minimal
- **Deep liquidity** — fills at quoted price, minimal slippage on micros
- **Clean breakout patterns** — range formations are well-defined, follow-through is consistent
- **Sufficient volatility** — enough ATR to generate meaningful R-multiples
- **Low margin** — micro contracts keep capital requirements manageable

| # | Asset | Ticker (Micro) | Why |
|---|-------|----------------|-----|
| 1 | **Gold** | MGC | Best-in-class breakout patterns. Trades 23 hours. Responds to macro catalysts. Asian + London + US sessions all active. $10/point, $1,100 margin. |
| 2 | **Crude Oil** | MCL | Very clean ORB setups. Strong session opens (London energy, US open). News-driven follow-through. $100/point, $700 margin. |
| 3 | **S&P 500** | MES | Highest liquidity micro future in existence. Perfect for stop-and-reverse — trends cleanly intraday. $5/point, $1,500 margin. |
| 4 | **Nasdaq** | MNQ | Highest ATR of the index micros — bigger moves = bigger R-multiples. Tech-driven catalysts create clean directional days. $2/point, $2,100 margin. |
| 5 | **Euro FX** | M6E | Best FX micro for breakouts. London open is a textbook ORB session. Low margin ($280) makes stop-and-reverse very capital-efficient. |

**Total margin for 5 positions (1 micro each): ~$5,680**
At $50K account, that's 11.4% margin utilization — very comfortable.

### Extended Watchlist (5 More for Scanning / Signals Only)

These assets should still generate breakout signals and CNN predictions,
but you wouldn't run the stop-and-reverse strategy on them unless the
core 5 aren't setting up.

| Asset | Ticker | Reason for extended list |
|-------|--------|------------------------|
| Silver | SIL | Correlated with Gold; trade when MGC is choppy but SIL is trending |
| Russell 2000 | M2K | Small-cap divergence plays; trade when MES/MNQ are range-bound |
| British Pound | M6B | Trade during London session when 6E is flat |
| Micro Bitcoin | MBT | 24/7 market; weekend breakout setups when futures are closed |
| 10Y T-Note | ZN | Macro context + flight-to-safety signals; trade on FOMC/CPI days |

### Assets to Remove

Drop these from the live watchlist entirely. They add noise without
edge — wide spreads, poor liquidity, erratic breakout behavior:

- **Copper (MHG)** — Too thin, gaps constantly, spread eats the edge
- **Natural Gas (MNG)** — Wild gaps, inventory report spikes destroy stops
- **Australian Dollar (6A)** — No micro, low volume, follows 6E anyway
- **Canadian Dollar (6C)** — No micro, oil-correlated (just trade MCL)
- **Swiss Franc (6S)** — No micro, high margin, redundant with 6E
- **30Y T-Bond (ZB)** — High margin, redundant with ZN
- **Corn (ZC)** — Ags are seasonal/news-driven, not breakout-friendly
- **Soybeans (ZS)** — Same issues as corn
- **Wheat (ZW)** — Widest ag spread, worst breakout follow-through

> **Action**: Update `MICRO_CONTRACT_SPECS` to add an `active: bool` field
> or create a `CORE_WATCHLIST` / `EXTENDED_WATCHLIST` constant so the engine,
> dashboard, and strategy can filter appropriately without removing the
> specs entirely (they're still useful for data fetching/display).

---

## 2. Model Versatility Architecture

### The Key Insight: Breakouts Are Universal

You're right that most breakout setups look fundamentally the same across
assets. The CNN sees:

1. **A range box** (gold, silver, blue, purple... whatever color)
2. **A breakout bar** punching through the range
3. **Volume confirmation** in the volume panel
4. **VWAP/EMA9 context** — is the breakout with or against the trend?

The *pattern* is identical whether it's Gold at $2,400 or Euro FX at 1.0850.
What varies is:

- **Volatility regime** — captured by `atr_pct` tabular feature
- **Volume characteristics** — captured by `volume_ratio`
- **Session context** — captured by `session_flag` / `session_ordinal`
- **Breakout type** — captured by `breakout_type_ord`

This is exactly why the hybrid CNN + tabular architecture works: the image
backbone learns the universal visual pattern, and the tabular head learns
the asset/session/type-specific context.

### Current Model: 8 Tabular Features (v5)

```
[quality_pct, volume_ratio, atr_pct, cvd_delta, nr7_flag,
 direction_flag, session_flag, london_overlap_flag]
```

### Proposed Model: 12 Tabular Features (v6)

To make the model truly versatile across all assets, types, and sessions,
we should expand the tabular head to include asset-aware and type-aware
features:

```
Feature                  Idx  Range     Source
─────────────────────────────────────────────────────────────
quality_pct              [0]  0.0–1.0   Signal quality / 100
volume_ratio             [1]  0.0–1.0   log1p normalised breakout vol / avg vol
atr_pct                  [2]  0.0–1.0   ATR / price × 100, clamped
cvd_delta                [3]  -1.0–1.0  Buy vol − sell vol / total vol
nr7_flag                 [4]  0.0|1.0   Narrowest range in 7 days
direction_flag           [5]  0.0|1.0   1.0 = LONG, 0.0 = SHORT
session_ordinal          [6]  0.0–1.0   Session position in Globex day (9 sessions)
london_overlap_flag      [7]  0.0|1.0   1.0 if 08:00–09:00 ET overlap
─── NEW ───────────────────────────────────────────────────────
breakout_type_ord        [8]  0.0–1.0   BreakoutType ordinal / 12 (13 types)
asset_volatility_class   [9]  0.0–1.0   Low=0, Med=0.5, High=1.0 from vol clusters
range_atr_ratio         [10]  0.0–1.0   Range size / ATR, normalised
hour_of_day             [11]  0.0–1.0   Hour / 23 (captures intraday timing)
```

### Why NOT Add a Per-Asset Embedding

You might think: "add a `symbol_id` feature so the model knows it's
trading Gold vs Nasdaq." **Don't do this.** Here's why:

1. **Overfitting risk** — The model would learn "Gold breakouts are good,
   Wheat breakouts are bad" instead of learning the *pattern*.
2. **Generalization death** — Can't predict on a new asset it's never seen.
3. **The tabular features already encode what matters** — `atr_pct` captures
   volatility profile, `volume_ratio` captures liquidity, `session_ordinal`
   captures timing. The model doesn't need to know the asset name.

The breakout pattern IS the same. The model should learn "tight range +
volume spike + trend alignment = good breakout" regardless of asset.

### Training Strategy: All Assets, All Types, All Sessions

Train on the full universe so the model sees maximum diversity:

```
Symbols:   MGC, MCL, MES, MNQ, M6E, SIL, M2K, M6B, MBT, ZN
Types:     all (13 breakout types)
Sessions:  all (9 Globex sessions)
Days:      90 (3 months of history)
```

This gives the model:
- ~10 assets × 13 types × 9 sessions × ~90 days = massive diversity
- The model learns the UNIVERSAL breakout pattern
- At inference time, it works on ANY asset (even ones not in training)
  because it's learned the pattern, not the ticker

### Per-Asset Inference: Already Works

When the engine runs `predict_breakout()` for a specific asset, it passes
that asset's *actual* tabular features (real ATR, real volume ratio, real
session). The model doesn't need to know the asset name — it knows the
*characteristics*.

---

## 3. Stop-and-Reverse Strategy with Micro Contracts

### Concept

For each of the 5 core watchlist assets, maintain a **persistent 1-lot
micro position** that is always either LONG or SHORT. The strategy never
goes flat (except during maintenance windows or when all signals are
below threshold).

This is a **regime-following approach**: the system detects the prevailing
breakout direction and stays positioned accordingly. When the direction
flips, it reverses (close current + open opposite = "stop and reverse").

### State Machine

```
                ┌──────────────┐
                │   FLAT       │
                │ (no position)│
                └──────┬───────┘
                       │
            CNN signal ≥ threshold
            + quality gate passed
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────┐     ┌──────────┐
        │  LONG 1  │     │ SHORT 1  │
        │  micro   │◄───►│  micro   │
        └────┬─────┘     └────┬─────┘
             │                │
     SHORT signal ≥ T   LONG signal ≥ T
     + reversal gate    + reversal gate
             │                │
             └───►REVERSE◄────┘
                  (close + open opposite)
```

### Reversal Gate Logic

A stop-and-reverse is expensive (you pay spread twice), so we add gates
to prevent whipsawing:

```python
def should_reverse(current_direction: str, new_signal: BreakoutResult) -> bool:
    """Decide if we should reverse the position."""
    # 1. New signal must be opposite direction
    if new_signal.direction == current_direction:
        return False  # same direction — hold or add, don't reverse

    # 2. CNN confidence must be high (not just above threshold)
    if new_signal.cnn_prob is not None and new_signal.cnn_prob < 0.85:
        return False  # need strong conviction to pay reversal cost

    # 3. Quality gate
    if new_signal.filter_passed is False:
        return False

    # 4. Minimum time in position (avoid whipsaw)
    # Don't reverse within 30 minutes of last entry
    if minutes_since_last_entry < 30:
        return False

    # 5. MTF alignment — higher timeframe must agree with new direction
    if new_signal.mtf_score is not None and new_signal.mtf_score < 0.6:
        return False

    # 6. Minimum adverse excursion — current position should be losing
    # Don't flip a winning position unless the new signal is exceptional
    if current_unrealized_pnl > 0 and new_signal.cnn_prob < 0.92:
        return False

    return True
```

### Position Tracking

Each asset needs a position state object:

```python
@dataclass
class MicroPosition:
    """Tracks a single micro contract position for stop-and-reverse."""
    symbol: str
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    contracts: int = 1       # always 1 micro
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0

    # Bracket phase tracking
    phase: str = "initial"   # "initial" | "tp1_hit" | "tp2_hit" | "trailing"
    tp1_hit: bool = False
    tp2_hit: bool = False
    breakeven_set: bool = False

    # EMA9 trailing state
    ema9_trail_active: bool = False
    ema9_trail_price: float = 0.0

    # Metrics
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    reversal_count: int = 0  # how many times this position was reversed into

    # Source signal
    breakout_type: str = ""
    cnn_prob: float = 0.0
    session_key: str = ""
```

### Entry Placement

Rather than market-ordering at the breakout trigger, use **limit orders
at favorable prices** relative to the range:

```
LONG entry logic:
  - Preferred: Limit buy at range_high + 1 tick (just above breakout level)
  - Acceptable: Limit buy at VWAP if VWAP < range_high
  - Worst case: Market order if price already 0.5×ATR above range_high
    and CNN prob > 0.90 (chasing with high conviction only)

SHORT entry logic:
  - Preferred: Limit sell at range_low − 1 tick
  - Acceptable: Limit sell at VWAP if VWAP > range_low
  - Worst case: Market order if price already 0.5×ATR below range_low
    and CNN prob > 0.90
```

For **stop-and-reverse**, the reversal itself is a market order (you need
immediate execution to flip), but subsequent entries after the initial
reverse can use limit placement.

---

## 4. Order Management & Take Profit Ladder

### 3-Phase Bracket (Already in RangeConfig)

The bracket system is already defined in `breakout_types.py` for all 13
types. Here's how it works in the context of the stop-and-reverse strategy:

```
Phase 1: INITIAL
├── Stop Loss: entry ∓ (SL × ATR)     — typically 1.0–1.5× ATR
├── TP1:       entry ± (TP1 × ATR)    — typically 1.5–2.0× ATR
└── Action at TP1:
    ├── With multi-lot: Take partial profit (sell 1 of N contracts)
    └── With 1-lot (our case): Move stop to breakeven, hold for TP2

Phase 2: BREAKEVEN (after TP1 hit)
├── Stop Loss: moved to entry price (breakeven)
├── TP2:       entry ± (TP2 × ATR)    — typically 2.5–3.5× ATR
└── Action at TP2:
    └── Activate EMA9 trailing stop (Phase 3)

Phase 3: EMA9 TRAILING (after TP2 hit)
├── Stop Loss: EMA9 value (updated every bar close)
├── TP3:       entry ± (TP3 × ATR)    — typically 4.0–5.0× ATR (hard cap)
└── Exit conditions:
    ├── Price touches EMA9 from the profitable side → exit
    ├── Price reaches TP3 → exit (full target hit)
    └── End of session → exit (no overnight hold for day-session types)
```

### With 1 Micro Contract: The Bracket Walk

Since we're always holding exactly 1 contract, we can't scale out. Instead:

```
TP1 hit → DON'T exit. Move stop to breakeven. Hold for bigger move.
TP2 hit → DON'T exit. Activate EMA9 trailing. Let it run.
TP3 hit → EXIT. Full target achieved.
EMA9 touch after TP2 → EXIT. The trend is weakening.
Stop hit at any phase → EXIT (or REVERSE if opposite signal fires).
```

This means most trades have only two outcomes:
1. **Stopped out** at original SL (Phase 1 loss) → ~1.0–1.5R loss
2. **Trailed out** via EMA9 after TP2 (Phase 3 win) → ~2.5–4.0R win

The TP1→breakeven move eliminates the "small wins that don't cover the
losses" problem. You're either losing 1R or winning 2.5R+.

### Bracket Multipliers by Type

From the existing `RangeConfig` entries:

| Type | SL | TP1 | TP2 | TP3 | EMA Trail | Style |
|------|-----|-----|-----|-----|-----------|-------|
| ORB | 1.5 | 2.0 | 3.0 | 4.5 | ✅ EMA9 | Aggressive |
| PrevDay | 1.5 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Conservative |
| InitialBalance | 1.0 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Tight |
| Consolidation | 1.0 | 2.0 | 3.5 | 5.0 | ✅ EMA9 | Spring-loaded |
| Weekly | 1.5 | 2.0 | 3.5 | 5.0 | ✅ EMA9 | HTF wide |
| Monthly | 2.0 | 2.5 | 4.0 | 6.0 | ✅ EMA9 | HTF widest |
| Asian | 1.0 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Overnight |
| BollingerSqueeze | 1.0 | 2.0 | 3.5 | 5.0 | ✅ EMA9 | Squeeze pop |
| ValueArea | 1.0 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Auction |
| InsideDay | 1.0 | 2.0 | 3.0 | 4.5 | ✅ EMA9 | Coiled |
| GapRejection | 1.0 | 1.5 | 2.5 | 3.5 | ✅ EMA9 | Gap fill |
| PivotPoints | 1.0 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Floor pivot |
| Fibonacci | 1.0 | 1.5 | 2.5 | 4.0 | ✅ EMA9 | Fib zone |

---

## 5. EMA9 Trailing Logic

### How It Works

After TP2 is hit, the stop-loss becomes the EMA9 value on the trading
timeframe (1-minute bars). The trailing stop is updated on every bar close.

```python
def update_ema9_trail(position: MicroPosition, bars_1m: pd.DataFrame) -> float | None:
    """Update the EMA9 trailing stop. Returns exit price if triggered."""
    if not position.ema9_trail_active:
        return None

    # Compute EMA9 on most recent bars
    ema9 = bars_1m["Close"].ewm(span=9, adjust=False).mean().iloc[-1]

    current_price = bars_1m["Close"].iloc[-1]

    if position.direction == "LONG":
        # Trail below price — exit if price closes below EMA9
        position.ema9_trail_price = ema9
        if current_price < ema9:
            return ema9  # exit signal

    elif position.direction == "SHORT":
        # Trail above price — exit if price closes above EMA9
        position.ema9_trail_price = ema9
        if current_price > ema9:
            return ema9  # exit signal

    return None  # still trailing, no exit
```

### EMA9 Trailing Rules

1. **Only activate after TP2 is hit** — never trail with EMA9 before TP2
2. **Use bar CLOSE, not intrabar wick** — a wick through EMA9 is not an exit;
   the bar must close beyond it
3. **Update every 1-minute bar close** — not tick-by-tick
4. **Hard cap at TP3** — if price reaches TP3 while trailing, exit immediately
   regardless of EMA9 position
5. **Session end** — if the position is an intraday type (ORB, IB, Consolidation),
   exit at session end even if EMA9 hasn't triggered

### EMA9 vs Higher Timeframe

For multi-timeframe awareness, compute EMA9 on both 1m and 5m:

- **1m EMA9**: Primary trailing stop (tight, responsive)
- **5m EMA9**: Confirmation — if 5m EMA9 is also breached, exit is higher conviction
- **15m EMA9**: Context only — if 15m EMA9 is still supportive, the trend may resume
  after a 1m EMA9 touch (use this for re-entry logic, not exit)

---

## 6. Multi-Timeframe Support

### Current State

The engine already computes:
- **1m bars** — primary breakout detection and ORB range building
- **5m bars** — focus computation (wave analysis, signal quality)
- **15m bars** — MTF analyzer (EMA bias, MACD slope, divergence)
- **Daily bars** — NR7 detection, daily range context

### Proposed MTF Integration for Strategy

```
Timeframe    Role in Strategy                     Data Source
──────────────────────────────────────────────────────────────
1m           Entry timing, EMA9 trail, range box   engine:bars_1m:{ticker}
5m           Trend confirmation, EMA20/50 slope    engine:bars_5m:{ticker}
15m          MTF score, MACD, regime filter         engine:bars_15m:{ticker}
1H           HTF bias, key S/R levels               Resample from 5m
Daily        NR7, inside day, gap, weekly/monthly   engine:bars_daily:{ticker}
```

### MTF Confluence Score (Already Exists)

The MTF analyzer (`lib/analysis/mtf_analyzer.py`) already computes:

```python
class MTFResult:
    mtf_score: float        # 0.0–1.0 aggregate score
    ema_slope_direction: str # "bullish", "bearish", "neutral"
    macd_histogram_slope: float
    divergence_type: str    # "bullish_div", "bearish_div", or ""
```

This is already wired into the ORB signal flow. For the stop-and-reverse
strategy, we use it as a **reversal gate** (Section 3): don't reverse
into a position that the higher timeframes disagree with.

### New: Intrabar MTF Update

For the persistent position strategy, we need **continuous MTF updates**,
not just at breakout detection time. Add a periodic MTF recomputation:

```
Every 5 minutes:
  For each active micro position:
    1. Fetch fresh 15m bars
    2. Recompute MTF score
    3. If MTF score drops below 0.3 AND position is losing:
       → Consider early exit (don't wait for stop)
    4. If MTF score rises above 0.8 AND position is winning:
       → Widen the trailing stop (use 5m EMA9 instead of 1m)
```

---

## 7. Implementation Plan

### Phase 1: Model Update (v6 Features + Retrain)

**Files to change:**

| File | Change |
|------|--------|
| `src/lib/analysis/breakout_cnn.py` | Expand `TABULAR_FEATURES` from 8 → 12, update `NUM_TABULAR`, update `BreakoutDataset.__getitem__`, update `_normalise_tabular_for_inference` |
| `src/lib/training/dataset_generator.py` | Add `breakout_type_ord`, `asset_volatility_class`, `range_atr_ratio`, `hour_of_day` columns to CSV output |
| `src/lib/services/engine/main.py` | Pass 12 features (instead of 8) to `predict_breakout()` at inference time |
| `src/lib/core/breakout_types.py` | No change needed — ordinals already defined |

**Note on backward compatibility:** The model architecture
(`HybridBreakoutCNN`) takes `num_tabular` as a constructor parameter.
Old 8-feature models won't load into a 12-feature architecture. We handle
this by versioning: models trained with v6 features have
`num_tabular_features: 12` in their metadata JSON. The loader checks this
and instantiates the correct architecture.

### Phase 2: Asset Configuration

**Files to change:**

| File | Change |
|------|--------|
| `src/lib/core/models.py` | Add `CORE_WATCHLIST` and `EXTENDED_WATCHLIST` constants. Add `active` flag to contract specs. |
| `src/lib/services/engine/focus.py` | Filter `compute_daily_focus` to only core + extended assets |
| `src/lib/training/trainer_server.py` | Default training symbols list uses all 10 (core + extended) |

### Phase 3: Stop-and-Reverse Engine Module

**New file:** `src/lib/services/engine/position_manager.py`

This is the core of the new strategy. It manages persistent micro positions:

```python
# Responsibilities:
# 1. Track active micro positions per asset (in Redis for persistence)
# 2. Process incoming breakout signals → decide: hold / reverse / exit
# 3. Manage bracket phases (initial → breakeven → trailing)
# 4. Compute EMA9 trail updates every bar
# 5. Emit order commands (for Rithmic integration to execute)
# 6. Persist position history for analysis
```

Key classes and functions:
- `MicroPosition` — position state dataclass
- `PositionManager` — manages all active positions
- `should_reverse()` — reversal gate logic
- `update_bracket_phase()` — phase transitions (TP1→BE, TP2→trail)
- `update_ema9_trail()` — trailing stop computation


**Files in `futures` repo:**

| File | Change |
|------|--------|
| `src/lib/services/engine/position_manager.py` | 13-value `BreakoutType` enum, 3-phase bracket, EMA9 trailing, stop-and-reverse logic |
| `src/lib/analysis/breakout_cnn.py` | Update to pass 12 tabular features (v6 contract) |
| `src/lib/services/engine/rithmic_client.py` | Position state sync with Rithmic execution layer |

### Phase 5: Full Retrain & Deploy

```bash
# 1. Generate dataset with all 13 types, all sessions, 10 assets
python -m lib.services.training.dataset_generator generate \
  --symbols MGC MCL MES MNQ M6E SIL M2K M6B MBT ZN \
  --session all \
  --breakout-type all \
  --days 90 \
  --output-dir ~/github/rb/dataset

# 2. Train on GPU rig (RTX 2070 SUPER)
python scripts/smoke_test_trainer.py \
  --symbols MGC MCL MES MNQ M6E SIL M2K M6B MBT ZN \
  --days 90 \
  --epochs 25

# 3. Evaluate + promote champion
# (automated by trainer_server pipeline)



# 5. Verify
python -c "from lib.analysis.breakout_cnn import model_info; print(model_info())"
```

---

## Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Core assets | 5 (MGC, MCL, MES, MNQ, M6E) | Best liquidity + breakout quality + margin efficiency |
| Extended assets | 5 more (SIL, M2K, M6B, MBT, ZN) | Signal scanning, context, fallback |
| Drop assets | 12 removed | Noise, wide spreads, poor patterns |
| Model approach | Universal — train on all, infer on any | Breakout patterns are the same; tabular features encode context |
| Position strategy | 1-lot micro, always-in stop-and-reverse | Low capital, persistent exposure, trend-following |
| Bracket | 3-phase: SL/TP1 → breakeven → EMA9 trail to TP3 | Eliminates small wins, lets winners run |
| Entry placement | Limit at range edge, market only for reversals | Reduce slippage, improve fill quality |
| MTF integration | 1m (entry/trail) + 5m (confirm) + 15m (bias) + daily (context) | Already built; add continuous updates for position management |
| Feature contract | v5 (8 features) → v6 (12 features) | Add breakout_type_ord, vol class, range ratio, hour |