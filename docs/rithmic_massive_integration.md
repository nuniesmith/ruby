# Rithmic + Massive Integration Design
## Full capability review, system mapping, and phased build plan

---

## 1. What we have today

### 1.1 `rithmic_client.py` — current state

The existing integration is a **read-only, short-lived session poller**. When you press
"Refresh" in Settings → Prop Accounts it:

1. Opens a connection to the Order Plant + PnL Plant
2. Calls `list_accounts()`, `list_positions()`, `get_pnl_info()`
3. Stores the snapshot in Redis under `rithmic:account_status:{key}`
4. Disconnects immediately

What it does **not** do:
- Hold a persistent connection between refreshes
- Stream real-time ticks, time bars, or order book data
- Use the Ticker Plant at all (market data)
- Use the History Plant at all (historical bars)
- Feed any computed metrics back into the engine or dashboard

### 1.2 `massive_client.py` — current state

Massive is the primary live market data source. It provides:

| Feature | Status |
|---|---|
| REST historical OHLCV (1s / 1m / 1d) | ✅ Active |
| Front-month contract resolution | ✅ Active |
| Real-time snapshots (price, vol, OI) | ✅ Active |
| WebSocket minute bars (AM channel) | ✅ Active, feeds engine |
| WebSocket trades / CVD buffer (T channel) | ✅ Active |
| WebSocket quotes / BBO (Q channel) | ✅ Wired, opt-in |
| Per-second aggregates (A channel) | ✅ Wired, upgrade path exists |
| L2 order book | ❌ Not available from Massive |

Massive's WebSocket has no L2/depth-of-market channel. That is a hard
limitation of the Massive/Polygon futures product — it only provides
aggregates, trades, and BBO quotes.

**Rithmic is therefore the only source we have for true L2 order book data.**

---

## 2. Full `async_rithmic` capability inventory

The installed version exposes four plants. Here is every callable, what it
returns, and how it maps to our system.

### 2.1 Ticker Plant (real-time market data)

```python
# All called on the connected RithmicClient instance

# One-shot lookups
await client.list_exchanges()
# → list of exchange objects (exchange_name, exchange_id, etc.)

await client.get_front_month_contract(symbol="ES", exchange="CME")
# → str, e.g. "ESM5" — the active front-month ticker

await client.search_symbols(search_text="ES", exchange="CME")
# → list of matching instrument objects

await client.request_market_depth(symbol="ESM5", exchange="CME", depth_price=5200.0)
# → one-shot L2 snapshot at a given price level

# Streaming subscriptions (fires client.on_tick event)
await client.subscribe_to_market_data(
    symbol="ESM5",
    exchange="CME",
    data_type=DataType.LAST_TRADE,   # template 150 → last trade tick
)
await client.subscribe_to_market_data(
    symbol="ESM5",
    exchange="CME",
    data_type=DataType.BBO,          # template 151 → best bid/offer
)
await client.subscribe_to_market_data(
    symbol="ESM5",
    exchange="CME",
    data_type=DataType.ORDER_BOOK,   # template 156 → full order book (L2)
    # fires client.on_order_book instead of on_tick
)

# Streaming L2 market depth (depth-by-order updates)
await client.subscribe_to_market_depth(
    symbol="ESM5",
    exchange="CME",
    depth_price=5200.0,              # fires client.on_market_depth
)
await client.unsubscribe_from_market_depth(...)
```

**Event hooks (register async callbacks):**

```python
client.on_tick           += async_callback   # LAST_TRADE and BBO ticks
client.on_order_book     += async_callback   # L2 order book snapshots
client.on_market_depth   += async_callback   # depth-by-order updates
```

`on_order_book` fires with the raw protobuf response from template 156
(`order_book_pb2`). It contains multiple price levels with bid/ask sizes.

`on_market_depth` fires with template 160 (`depth_by_order_pb2`). This is
the full depth-by-order stream — each individual order in the book, not just
aggregated sizes per level. It is the highest-resolution order flow data
Rithmic provides.

### 2.2 History Plant (historical data)

```python
# Historical tick data (every trade, microsecond resolution)
ticks = await client.get_historical_tick_data(
    symbol="ESM5",
    exchange="CME",
    start_time=datetime(2025, 3, 8, 9, 30, tzinfo=UTC),
    end_time=datetime(2025, 3, 8, 10, 0, tzinfo=UTC),
)
# → list of dicts: {symbol, price, volume, datetime, ...}

# Historical OHLCV time bars
bars = await client.get_historical_time_bars(
    symbol="ESM5",
    exchange="CME",
    start_time=...,
    end_time=...,
    bar_type=TimeBarType.MINUTE_BAR,  # or DAILY_BAR, WEEKLY_BAR, etc.
    bar_type_periods=5,               # 5-minute bars
)
# → list of dicts: {symbol, open, high, low, close, volume, marker, bar_end_datetime}

# Live time bar subscription (fires on each completed bar)
await client.subscribe_to_time_bar_data(
    symbol="ESM5",
    exchange="CME",
    bar_type=TimeBarType.MINUTE_BAR,
    bar_type_periods=1,
)
# fires client.on_time_bar
client.on_time_bar += async_callback
```

`TimeBarType` includes: `MINUTE_BAR`, `DAILY_BAR`, `WEEKLY_BAR`,
`MONTHLY_BAR`, `SECOND_BAR` (if available on account).

### 2.3 Order Plant (account + orders — read-only subset)

```python
# After connect(), client.accounts is populated automatically
client.accounts   # list of account objects with .account_id
client.fcm_id     # FCM ID from login
client.ib_id      # IB ID from login

# Order inspection (read-only — we never place orders)
await client.list_orders(...)
await client.show_order_history_summary(...)
```

> **Policy note:** We connect with `OrderPlacement.MANUAL` which is
> read-only — the library will not allow programmatic order submission.
> This is intentional and must never change.

### 2.4 PnL Plant (live P&L streaming)

```python
# One-shot snapshot
await client.list_positions(account_id="1234")
# → list of instrument_pnl_position_update objects

await client.list_account_summary(account_id="1234")
# → list of account_pnl_position_update objects

# Live streaming (fires on every position change)
await client.subscribe_to_pnl_updates()
# fires client.on_instrument_pnl_update — per-instrument P&L
# fires client.on_account_pnl_update    — account-level P&L

client.on_instrument_pnl_update += async_callback
client.on_account_pnl_update    += async_callback
```

Key fields from `instrument_pnl_position_update.proto`:
- `symbol`, `exchange`, `is_snapshot`
- `open_position_qty`, `close_position_qty`
- `net_position` — current net size
- `open_trade_equity` — unrealized P&L
- `fill_buy_qty`, `fill_sell_qty` — session fill counts
- `fill_buy_cost`, `fill_sell_cost`
- `average_open_price`

Key fields from `account_pnl_position_update.proto`:
- `account_id`, `fcm_id`
- `net_liq_value` — net liquidation value
- `open_position_pnl` — total unrealized
- `option_closed_pnl`, `futures_closed_pnl`
- `min_account_balance` — daily drawdown floor
- `account_balance` — current balance
- `excess_buy_margin` / `available_buying_power`

---

## 3. What Rithmic gives us that Massive cannot

| Capability | Massive | Rithmic |
|---|---|---|
| Real-time OHLCV bars | ✅ 1m/1s WebSocket | ✅ via History Plant |
| Trade tick stream | ✅ T.* channel | ✅ `on_tick` (LAST_TRADE) |
| Best bid/offer (BBO) | ✅ Q.* channel | ✅ `on_tick` (BBO) |
| **L2 order book (full depth)** | ❌ Not available | ✅ `on_order_book` |
| **Depth-by-order (individual orders)** | ❌ Not available | ✅ `on_market_depth` |
| **Historical tick data** | ❌ Not available | ✅ `get_historical_tick_data` |
| **Historical time bars (Rithmic native)** | ❌ limited history | ✅ `get_historical_time_bars` |
| **Live P&L streaming** | ❌ Not available | ✅ `on_instrument_pnl_update` |
| **Account balance / buying power** | ❌ Not available | ✅ `on_account_pnl_update` |
| **Position net size (real-time)** | ❌ Not available | ✅ `net_position` field |
| Front-month resolution | ✅ /contracts endpoint | ✅ `get_front_month_contract` |
| Exchange listing | ❌ | ✅ `list_exchanges` |

The three capabilities that have the most direct impact on the existing
system are:

1. **L2 order book** — feeds CVD delta, bid/ask imbalance, absorption
   detection, and the planned order flow features in `breakout_filters.py`
2. **Live P&L streaming** — replaces the 5-second Redis poll from
   `live_risk.py` with a push-based update that fires on every fill
3. **Historical tick data** — allows us to back-fill high-resolution trade
   data for volume profile and CVD reconstruction beyond what Massive stores

---

## 4. Computed metrics from L2 + ticks

Once we have `on_order_book`, `on_market_depth`, and `on_tick` streaming,
we can compute the following in real time. Each metric and where it plugs
into the existing system is listed below.

### 4.1 Cumulative Volume Delta (CVD)

**Source:** `on_tick` (LAST_TRADE) — each tick has price + size + aggressor side.

```
cvd += size   if aggressor == BUY
cvd -= size   if aggressor == SELL
```

**Current state:** `MassiveFeedManager` already has a `_trade_buffer` and
`get_trade_buffer()` for CVD computation, but we only know price and size —
we do not reliably know the aggressor side from Massive because it does not
tag the side on trade ticks.

**With Rithmic `on_tick`:** The LAST_TRADE message from Rithmic includes
`aggressor_side` (bid-initiated vs ask-initiated). This is the ground truth
for CVD. We compute it in the streaming handler and publish to Redis as
`rithmic:cvd:{symbol}`.

The dashboard already reads CVD from the volume profile panel
(`_render_volume_profile_panel` in `dashboard.py`). We plug Rithmic CVD
directly into that Redis key.

### 4.2 Bid/Ask Imbalance (BAI)

**Source:** `on_order_book` (template 156) — full order book snapshot.

```
total_bid_size = sum(level.bid_size for level in top_N_levels)
total_ask_size = sum(level.ask_size for level in top_N_levels)
bai = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
# range [-1, +1], positive = bid heavy, negative = ask heavy
```

We compute this over the top 5 and top 10 levels independently.
`bai_5` is useful for ORB entry timing; `bai_10` is useful for regime
confirmation.

**Feeds into:** `breakout_filters.py` → new `check_order_book_imbalance()`
filter. When a long ORB fires and `bai_5 < -0.15` (ask-side dominated) we
add a soft warning flag rather than blocking the signal outright.

### 4.3 Stacked Imbalances

**Source:** `on_order_book` — compare consecutive price levels for
large size clusters.

A stacked imbalance exists when 3+ consecutive bid levels each have size
≥ 1.5× the corresponding ask level (or vice versa). This is an institutional
footprint — large resting limit orders that create a magnetic price level.

```
for i in range(len(levels) - 2):
    bid_ratio = levels[i].bid / max(levels[i].ask, 1)
    if bid_ratio >= 1.5 and bid_ratio_next >= 1.5 and bid_ratio_next2 >= 1.5:
        stacked_bid_imbalances.append(levels[i].price)
```

**Feeds into:** `handlers.py` → `handle_breakout_check()` gets a new
`order_book_context` parameter. The ORB panel on the dashboard shows
stacked imbalance levels as horizontal lines on the mini-chart.

### 4.4 Absorption Detection

**Source:** `on_tick` (LAST_TRADE) cross-referenced with `on_order_book`.

Absorption = large volume trading through a price level WITHOUT moving
the price. The ask side is absorbing selling, or the bid side is absorbing
buying.

```
if tick.price == prior_best_ask and tick.size > absorption_threshold:
    if best_ask has NOT moved after this tick:
        absorption_event = "ASK_ABSORBING_SELL"
```

This is one of the most reliable signals for ORB breakout failure. If we
see 500+ contracts absorbed at the ORB high, the breakout is likely to
fail.

**Feeds into:** `breakout_filters.py` → new `check_absorption()` filter.
Also feeds the `_render_asset_card()` in `dashboard.py` — a small
absorption indicator next to the bid/ask spread display.

### 4.5 Delta Divergence

**Source:** CVD (from ticks) vs price direction.

If price is making a new ORB high but CVD is declining or negative, the
breakout has weak buying conviction — delta divergence. This is a high-value
filter for ORB false breakouts.

```
price_direction = +1 if current_price > orb_high_reference else -1
cvd_direction   = +1 if current_cvd > prior_cvd else -1
divergence = (price_direction != cvd_direction)
```

**Feeds into:** `breakout_filters.py` → new `check_delta_divergence()`
filter. Has its own CNN feature slot.

### 4.6 Order Flow Momentum (OFM)

**Source:** Rolling window of LAST_TRADE ticks.

```
buy_volume  = sum of tick.size where aggressor == BUY in last N seconds
sell_volume = sum of tick.size where aggressor == SELL in last N seconds
ofm = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)
```

Where N = 30 seconds for ORB entry confirmation, 5 minutes for session
regime. OFM > +0.2 during a long ORB breakout confirms institutional buying.

**Feeds into:** `live_risk.py` → added to `LiveRiskState` as
`order_flow_momentum` per active symbol. Displayed in the position overlay.

### 4.7 VWAP with Volume Distribution

**Source:** All LAST_TRADE ticks for the session.

Rithmic ticks let us compute true intraday VWAP from first principles:

```
vwap = cumulative(price * size) / cumulative(size)
```

Unlike Massive's minute-bar VWAP approximation (which uses bar midpoint ×
volume), Rithmic tick-based VWAP is exact. The deviation from VWAP is a
core feature in the existing CNN model (`feature_contract.json`).

**Feeds into:** `cache.py` → `rithmic:vwap:{symbol}` key replaces the
Massive-derived VWAP when Rithmic is connected. The engine reads VWAP from
this key via `cache_get` — no other changes needed.

### 4.8 Large Order Detection (Iceberg / Sweep)

**Source:** `on_market_depth` (template 160 — depth-by-order stream).

Each individual order in the book has an order ID, size, and side. We can
detect:

- **Iceberg orders:** same price level repeatedly refilled to the same size
  after partial fills — indicates a large participant hiding size
- **Sweeps:** a sequence of aggressive market orders hitting multiple price
  levels in rapid succession — signals institutional urgency

These are stored as `rithmic:sweep_alert:{symbol}` in Redis with a 60-second
TTL and surfaced in the `_get_gap_alerts()` feed.

### 4.9 Session POC from Tick Volume

**Source:** All LAST_TRADE ticks, bucketed by price.

The existing `_render_vp_svg()` computes volume profile from OHLCV bars
(using a bell-curve approximation of volume distribution within each bar).
With Rithmic ticks, we can compute an exact TPO-style session profile:

```
price_bucket = round(tick.price / tick_size) * tick_size
volume_at_price[price_bucket] += tick.size
poc = max(volume_at_price, key=volume_at_price.get)
value_area_high, value_area_low = compute_70pct_va(volume_at_price)
```

**Feeds into:** `_render_volume_profile_panel()` — the SVG VP renderer
already has a `session_profiles` path. We add a Rithmic-sourced profile
that overlays on top of the Massive-bar-derived profile for comparison.

---

## 5. System integration map

```
async_rithmic library
        │
        ├── TickerPlant
        │       ├── on_tick (LAST_TRADE)  ──► RithmicStreamProcessor.on_trade()
        │       │                                    ├── CVD accumulator
        │       │                                    ├── OFM rolling window
        │       │                                    ├── Session VWAP
        │       │                                    └── Tick VP buckets
        │       │
        │       ├── on_tick (BBO)         ──► RithmicStreamProcessor.on_bbo()
        │       │                                    └── Spread monitor
        │       │
        │       ├── on_order_book         ──► RithmicStreamProcessor.on_l2()
        │       │                                    ├── BAI (bid/ask imbalance)
        │       │                                    ├── Stacked imbalances
        │       │                                    └── Absorption detector
        │       │
        │       └── on_market_depth       ──► RithmicStreamProcessor.on_depth()
        │                                        ├── Iceberg detector
        │                                        └── Sweep detector
        │
        ├── HistoryPlant
        │       ├── on_time_bar           ──► replaces Massive WebSocket bars
        │       └── get_historical_*      ──► backfill_rithmic() in backfill.py
        │
        └── PnlPlant
                ├── on_instrument_pnl_update ──► live_risk.py position updates
                └── on_account_pnl_update    ──► live_risk.py account balance

All metrics published to Redis:
    rithmic:cvd:{symbol}              ← float, session CVD
    rithmic:ofm:{symbol}              ← float [-1,+1], 30s order flow momentum
    rithmic:vwap:{symbol}             ← float, tick-based VWAP
    rithmic:bai:{symbol}              ← {bai_5, bai_10, updated_at}
    rithmic:l2_snapshot:{symbol}      ← {bids: [...], asks: [...], ts}
    rithmic:stacked:{symbol}          ← {bid_levels: [...], ask_levels: [...]}
    rithmic:absorption:{symbol}       ← {side, price, volume, ts} or null
    rithmic:sweep:{symbol}            ← {direction, levels_swept, volume, ts} or null
    rithmic:vp_session:{symbol}       ← {poc, vah, val, profile: {price: vol}}
    rithmic:pnl:{account_key}         ← live P&L from PnlPlant
    rithmic:position:{account_key}    ← live net positions
```

---

## 6. Phased build plan

### Phase R1 — Persistent streaming connection (foundation)

**Goal:** Replace the short-lived poll-and-disconnect model with a
long-lived `RithmicStreamManager` that holds a persistent connection and
reconnects automatically using the library's built-in backoff.

**New file:** `src/lib/integrations/rithmic_stream.py`

```python
class RithmicStreamManager:
    """
    Long-lived Rithmic connection. Manages one RithmicClient per account
    config that has streaming enabled. Registers event handlers for ticks,
    L2, time bars, and P&L. Publishes all computed metrics to Redis.
    """
    async def start(self, account_key: str) -> None: ...
    async def stop(self, account_key: str) -> None: ...
    async def add_symbols(self, symbols: list[str]) -> None: ...
    async def remove_symbols(self, symbols: list[str]) -> None: ...
    def get_status(self) -> dict: ...
```

The `RithmicStreamManager` is started by the **data service** (`lifespan`
in `data/main.py`) alongside the existing `MassiveFeedManager`. They run
concurrently — Massive provides aggregates and broad market coverage;
Rithmic provides L2, tick-accurate CVD, and live P&L for the connected
account's traded symbols.

**Connection scope:** connect to `TICKER_PLANT` + `PNL_PLANT` only on
initial start. `HISTORY_PLANT` is connected on-demand for backfill requests.
`ORDER_PLANT` is never used (read-only policy).

**Plant selection at connect time:**

```python
await client.connect(plants=[
    SysInfraType.TICKER_PLANT,
    SysInfraType.PNL_PLANT,
])
```

**Deliverables:**
- `RithmicStreamManager` class
- Engine scheduler calls `manager.start()` in `SessionMode.EVENING` transition
- Engine scheduler calls `manager.stop()` in `SessionMode.OFF_HOURS` transition
  (Rithmic sessions should not be held during off-hours to avoid idle disconnect)
- Redis key `rithmic:stream_status` with connection state, uptime, msg counts

---

### Phase R2 — Live ticks + CVD + VWAP

**Goal:** Stream `LAST_TRADE` ticks for all focus symbols, compute CVD,
OFM, and session VWAP, publish to Redis.

**New file:** `src/lib/integrations/rithmic_metrics.py`

```python
class TickMetricsAccumulator:
    """
    Stateful accumulator for one symbol's tick stream.
    Maintains CVD, OFM rolling window, and session VWAP in memory.
    Flushes to Redis every 1 second (or on significant change).
    """
    def on_tick(self, tick: dict) -> None: ...
    def get_cvd(self) -> float: ...
    def get_vwap(self) -> float: ...
    def get_ofm(self, window_secs: int = 30) -> float: ...
    def flush_to_redis(self) -> None: ...
    def reset_session(self) -> None: ...  # called at 18:00 ET daily
```

**Symbol selection:** We subscribe to the same symbols as the `MassiveFeedManager`
(derived from `ASSETS` in `models.py`). The Rithmic-native symbol is resolved
via `get_front_month_contract(symbol, exchange)` and cached in Redis under
`rithmic:front_month:{root}` with a 4-hour TTL.

**Integration with engine:** The engine's `handle_breakout_check()` in
`handlers.py` already reads `premarket_cvd` from focus data. We add a
fallback read of `rithmic:cvd:{symbol}` when Rithmic is connected.
When both Massive and Rithmic are streaming, Rithmic CVD takes precedence
because it has aggressor-side ground truth.

**Dashboard integration:** The volume profile panel's CVD bar chart
already exists. Its data source is updated to prefer `rithmic:cvd:*` when
available, falling back to the Massive trade buffer computation.

**Deliverables:**
- `TickMetricsAccumulator` per symbol
- CVD, OFM, and VWAP published to Redis every second
- Session reset hook at 18:00 ET
- Dashboard VP panel reads from `rithmic:cvd:*`

---

### Phase R3 — L2 order book + derived signals

**Goal:** Stream the full order book, compute BAI, stacked imbalances, and
absorption, publish to Redis, and wire into the breakout filter pipeline.

**Subscribe:**

```python
await client.subscribe_to_market_data(
    symbol=front_month,
    exchange=exchange,
    data_type=DataType.ORDER_BOOK,   # fires on_order_book (template 156)
)
```

**New processor in `rithmic_metrics.py`:**

```python
class L2BookProcessor:
    """
    Maintains a local order book state from on_order_book snapshots.
    Computes BAI, stacked imbalances, and absorption on each update.
    Publishes to Redis at most every 250ms (throttled — L2 can be noisy).
    """
    def on_l2_update(self, response) -> None: ...
    def get_bai(self, levels: int = 5) -> float: ...
    def get_stacked_imbalances(self, threshold: float = 1.5) -> dict: ...
    def get_absorption_state(self) -> dict | None: ...
    def get_snapshot(self) -> dict: ...  # {bids: [...], asks: [...]}
```

**New breakout filter in `breakout_filters.py`:**

```python
def check_order_book_context(
    signal_time: datetime,
    symbol: str,
    signal_direction: str,       # "long" or "short"
    min_bai_agreement: float = 0.05,
    absorption_block: bool = True,
) -> FilterResult:
    """
    Reads live L2 metrics from Redis and applies soft filters.
    - BAI disagreement: warns but does not block
    - Absorption at breakout level: blocks (if absorption_block=True)
    - Stacked imbalance nearby: adds context note
    """
```

This filter is added to the filter chain in `apply_all_filters()` as an
**optional** gate, disabled by default, enabled when `RITHMIC_STREAMING=true`
in env and at least one account is connected and streaming.

**New dashboard panel:** The existing `_render_asset_card()` function in
`dashboard.py` gets a new L2 micro-display: a compact bid/ask ladder showing
the top 5 levels with size bars, the current BAI indicator, and an absorption
warning badge. This renders only when `rithmic:l2_snapshot:{symbol}` exists
in Redis and is fresh (< 2 seconds old).

**Deliverables:**
- `L2BookProcessor` class, 250ms Redis flush
- `check_order_book_context()` filter
- L2 mini-ladder in asset cards (5 rows, inline SVG)
- BAI and absorption badges in focus cards

---

### Phase R4 — Live P&L streaming (replace poll)

**Goal:** Replace the 5-second `refresh_account()` poll with push-based
`on_instrument_pnl_update` and `on_account_pnl_update` events.

**Current flow:**
```
Scheduler → refresh_account() → connect → list_positions → disconnect
```

**New flow:**
```
RithmicStreamManager.start() → subscribe_to_pnl_updates()
    → on_instrument_pnl_update → LiveRiskPublisher.on_rithmic_position()
    → on_account_pnl_update    → LiveRiskPublisher.on_rithmic_account()
```

**Changes to `live_risk.py`:**

```python
# New fields on LiveRiskState:
rithmic_net_position: dict[str, int]   = field(default_factory=dict)
# symbol → net_position (from net_position field in proto)

rithmic_unrealized_pnl: float | None   = None
# from open_position_pnl on account_pnl_update

rithmic_account_balance: float | None  = None
# from account_balance on account_pnl_update

rithmic_buying_power: float | None     = None
# from excess_buy_margin on account_pnl_update

rithmic_min_balance: float | None      = None
# from min_account_balance — this is the trailing drawdown floor for prop firms
```

The `min_account_balance` field is critically important for prop firm
accounts. Rithmic reports the current trailing drawdown floor in this field.
We compare `account_balance` against `min_account_balance` to compute
**available drawdown budget**, which is displayed prominently in the
risk strip.

```
available_drawdown = account_balance - min_account_balance
drawdown_pct       = (account_balance - min_account_balance) / initial_account_size
```

When `available_drawdown` drops below a configurable threshold
(default: `max_daily_loss * 2`), the risk manager raises a
`DRAWDOWN_WARNING` event which flows through the existing alert pipeline.

**Deliverables:**
- P&L streaming wired into `LiveRiskState`
- Trailing drawdown floor computed and displayed
- `DRAWDOWN_WARNING` alert event
- Settings page shows per-account balance + drawdown bar

---

### Phase R5 — Historical tick backfill

**Goal:** Use the History Plant to backfill high-resolution tick data for
volume profile reconstruction and CNN feature generation.

**Trigger:** When the engine starts a new Globex day (18:00 ET) and detects
that the previous session's tick VP is missing from Redis, it requests a
backfill for the prior session's trading hours.

**New function in `backfill.py`:**

```python
async def backfill_rithmic_ticks(
    symbol: str,
    exchange: str,
    start_time: datetime,
    end_time: datetime,
    client: RithmicClient,
) -> pd.DataFrame:
    """
    Fetch historical tick data from Rithmic History Plant.
    Returns a DataFrame with columns: datetime, price, volume, aggressor_side.
    Stores computed session VP, VWAP, and CVD in Redis.
    """
```

This directly improves the CNN dataset generator's tick VP features, which
currently approximate intrabar volume distribution from OHLCV. With actual
tick data, the VP profiles in `feature_contract.json` become exact.

**Deliverables:**
- `backfill_rithmic_ticks()` function
- Scheduler hook at 18:00 ET to trigger prior-session backfill
- Backfilled tick VP stored under `rithmic:vp_historical:{symbol}:{date}`

---

### Phase R6 — Depth-by-order streaming (iceberg + sweep)

**Goal:** Stream the full depth-by-order feed (`on_market_depth`, template
160) for the 2–3 most active focus symbols during Active session.

This is the most data-intensive stream — every individual order in the book
fires an event. We limit it to the top focus symbols during the
`SessionMode.ACTIVE` window only.

```python
await client.subscribe_to_market_depth(
    symbol=front_month,
    exchange=exchange,
    depth_price=current_price,   # centre the depth window around current price
)
```

**New processor:**

```python
class DepthByOrderProcessor:
    """
    Tracks individual orders in the book. Detects iceberg refills
    (same price level repeatedly partially filled and refilled) and
    sweep events (consecutive aggressive orders across multiple levels).
    """
    def on_depth_update(self, response) -> None: ...
    def get_iceberg_levels(self) -> list[dict]: ...
    def get_recent_sweeps(self, window_secs: int = 60) -> list[dict]: ...
```

Iceberg levels and sweep events are published to Redis and surfaced in
the `_get_gap_alerts()` feed as high-priority alerts with a ⚡ badge.

**Deliverables:**
- `DepthByOrderProcessor` class (Active session only, top 2 symbols)
- Iceberg and sweep alerts in the dashboard alert feed
- Depth-by-order enabled/disabled per-symbol in Settings

---

## 7. Massive + Rithmic data fusion

When both Massive WebSocket and Rithmic streaming are active, we operate
them in **complementary** mode rather than competitive mode:

| Data type | Primary source | Fallback |
|---|---|---|
| OHLCV bars (1m) | Massive AM.* channel | Rithmic `on_time_bar` |
| Trade CVD | Rithmic `on_tick` (has side) | Massive T.* buffer (no side, estimate only) |
| BBO quotes | Either (Rithmic preferred, lower latency) | Other |
| L2 order book | Rithmic only | N/A (Massive has no L2) |
| Session VP | Rithmic tick-based (exact) | Massive bar-based (approx) |
| Live VWAP | Rithmic tick-based (exact) | Massive VWAP field in bar |
| P&L / positions | Rithmic PnL Plant (live push) | Massive N/A, manual journal |
| Historical OHLCV | Massive REST (2y+ history) | Rithmic (limited lookback) |
| Front-month ticker | Rithmic (exchange-authoritative) | Massive /contracts endpoint |

The fusion logic lives in a new `data_bus.py` module that the engine and
dashboard query instead of hitting Redis keys directly. `data_bus.py` reads
the appropriate key based on which sources are currently connected.

---

## 8. Redis key schema (full list)

```
# Rithmic stream state
rithmic:stream_status                  hash  {connected, accounts, uptime, msg_count}
rithmic:front_month:{root}             string  e.g. "ESM5"  (TTL 4h)

# Tick metrics (updated every ~1s while streaming)
rithmic:cvd:{symbol}                   string  float
rithmic:ofm:{symbol}                   string  float [-1, +1]
rithmic:vwap:{symbol}                  string  float
rithmic:tick_count:{symbol}            string  int (session total)

# L2 order book (updated every ~250ms while streaming)
rithmic:l2_snapshot:{symbol}           string  JSON {bids, asks, ts}  (TTL 5s)
rithmic:bai:{symbol}                   string  JSON {bai_5, bai_10, ts}  (TTL 5s)
rithmic:stacked:{symbol}               string  JSON {bid_levels, ask_levels, ts}  (TTL 10s)
rithmic:absorption:{symbol}            string  JSON {side, price, volume, ts} or null  (TTL 30s)

# Depth-by-order (Active session only)
rithmic:sweep:{symbol}                 string  JSON {direction, levels, volume, ts}  (TTL 60s)
rithmic:iceberg:{symbol}               string  JSON [{price, est_size, refill_count}]  (TTL 60s)

# Session VP from ticks (updated continuously, finalised at session end)
rithmic:vp_session:{symbol}            string  JSON {poc, vah, val, profile}
rithmic:vp_historical:{symbol}:{date}  string  JSON {poc, vah, val, profile}  (TTL 7d)

# Live P&L (updated on every fill event)
rithmic:pnl:{account_key}             string  JSON (instrument_pnl fields)
rithmic:account:{account_key}         string  JSON (account_pnl fields incl. min_balance)
```

---

## 9. Prop firm account notes

Each prop firm has specific rules that are directly observable through the
PnL Plant fields:

| Field | Prop firm usage |
|---|---|
| `account_balance` | Current account value |
| `min_account_balance` | Trailing drawdown floor (Apex, Topstep, TPT) — **do not let account_balance touch this** |
| `excess_buy_margin` | Available buying power for new positions |
| `open_position_pnl` | Real-time unrealized — counts against daily drawdown in some firms |
| `futures_closed_pnl` | Realised P&L for the day — target for consistency rules |

For Apex specifically: `min_account_balance` is a trailing maximum drawdown.
It tracks the highest account value and subtracts the drawdown limit from it.
If `account_balance <= min_account_balance` you are in violation.

The `DRAWDOWN_WARNING` alert (Phase R4) must fire early enough to exit any
open positions before touching the floor. The recommended threshold is when
`available_drawdown < max_daily_loss_setting` — i.e., you have less
remaining drawdown buffer than your configured per-day risk limit.

---

## 10. Implementation priorities (recommended order)

1. **R1** — Persistent streaming connection (blocks everything else)
2. **R4** — Live P&L streaming (highest immediate trading value)
3. **R2** — Live ticks + CVD + VWAP (replaces Massive trade buffer)
4. **R3** — L2 order book + BAI (new signal source for filters)
5. **R5** — Historical tick backfill (CNN dataset quality improvement)
6. **R6** — Depth-by-order iceberg/sweep (most complex, lowest urgency)

R1 and R4 together are the minimum viable Rithmic integration that delivers
real trading value: you have a live connection, you see live P&L, and the
risk system knows your exact balance and trailing drawdown floor at all times.

R2 and R3 are the order flow layer that makes the ORB signal quality
meaningfully better. With true aggressor-side CVD and L2 absorption
detection, the false breakout rate on ORB signals should drop noticeably
during the London and US sessions.

R5 and R6 are the research/enhancement tier — valuable but not blocking
for live operation.