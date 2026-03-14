# Ruby Futures — Logging Standard

> **Last updated**: 2026-03-13
> **Status**: Adopted — all new code MUST follow this guide. Existing code migrated incrementally.

---

## TL;DR — The One Pattern Every File Should Use

```python
# At the top of every module:
from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Then use structured key-value logging everywhere:
logger.info("order_placed", symbol="MES", qty=2, account="tpt_eval")
logger.warning("rate_limit_approaching", calls=48, max=50)
logger.error("connection_failed", host="rithmic.com", retries=3, exc_info=True)
```

That's it. No `import logging`. No `from loguru import logger`. No custom Logger classes.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Application Code                          │
│                                                              │
│  from lib.core.logging_config import get_logger              │
│  logger = get_logger(__name__)                               │
│  logger.info("event_name", key=value, ...)                   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                    structlog pipeline                         │
│                                                              │
│  contextvars → add_logger_name → add_log_level → timestamp  │
│  → stack_info → unicode_decode → format_exc_info             │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              stdlib logging (root handler)                    │
│                                                              │
│  ProcessorFormatter bridges structlog → stdlib so that       │
│  third-party libs (uvicorn, httpx, sqlalchemy) also get      │
│  the same structured formatting.                             │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                     Output                                    │
│                                                              │
│  LOG_FORMAT=console  →  coloured key-value lines (dev)       │
│  LOG_FORMAT=json     →  JSON lines (Docker / production)     │
└──────────────────────────────────────────────────────────────┘
```

### Why structlog + stdlib?

| Approach | Pros | Cons |
|----------|------|------|
| **structlog + stdlib** ✅ | Universal bus — uvicorn/httpx/sqlalchemy/FastAPI all emit stdlib logs and get formatted consistently. Key-value structured logging. JSON mode for production. Already wired in `logging_config.py`. | Need `structlog` dependency |
| loguru | Nice API, coloured output | Separate sink system — doesn't go through stdlib root handler, so third-party logs look different. Can't be bridged cleanly. |
| stdlib alone | Zero dependencies | No structured key-value logging. Verbose setup. |

---

## Setup (Service Entry Points Only)

Each service calls `setup_logging()` **once** at process startup. This is already done in the existing entry points:

```python
# src/lib/services/data/main.py
from lib.core.logging_config import setup_logging
setup_logging(service="data-service")

# src/lib/services/engine/main.py
from lib.core.logging_config import setup_logging
setup_logging(service="engine")

# src/lib/services/training/trainer_server.py
from lib.core.logging_config import setup_logging
setup_logging(service="trainer")

# src/lib/services/web/main.py
from lib.core.logging_config import setup_logging
setup_logging(service="web")
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Root log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `console` | `console` for coloured dev output, `json` for Docker/production |

---

## Module-Level Logger Pattern

Every Python module in the project should follow this exact pattern:

```python
"""Module docstring."""

# ... stdlib imports ...
# ... third-party imports ...

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Now use logger throughout the module:

def do_something(symbol: str, qty: int) -> None:
    logger.info("processing_order", symbol=symbol, qty=qty)
    try:
        result = execute(symbol, qty)
        logger.info("order_filled", symbol=symbol, fill_price=result.price)
    except ConnectionError:
        logger.error("order_failed", symbol=symbol, qty=qty, exc_info=True)
```

### Key Rules

1. **One `logger` per module** — always `get_logger(__name__)` at module level
2. **Event names, not sentences** — `"order_placed"` not `"An order was placed"`
3. **Key-value context, not f-strings** — `logger.info("bar_loaded", symbol=sym, count=n)` not `logger.info(f"Loaded {n} bars for {sym}")`
4. **`exc_info=True` for errors** — never `logger.error(str(e))`, always `logger.error("what_failed", exc_info=True)` or `logger.exception("what_failed")`
5. **No `print()` statements** — ever

---

## Structured Logging Best Practices

### ✅ Do This

```python
# Event name as first arg, context as kwargs
logger.info("trade_executed", symbol="MES", side="BUY", qty=2, price=5432.50)

# Bind persistent context for a subsystem
log = logger.bind(account="tpt_eval_1", session="us")
log.info("position_opened", symbol="MES", qty=1)
log.info("position_closed", symbol="MES", pnl=125.00)

# Timing
import time
t0 = time.monotonic()
result = expensive_operation()
logger.info("operation_complete", duration_ms=round((time.monotonic() - t0) * 1000))

# Exception logging
try:
    risky_call()
except Exception:
    logger.exception("risky_call_failed", symbol=sym)  # auto-attaches traceback
```

### ❌ Don't Do This

```python
# Don't use f-strings for log messages
logger.info(f"Loaded {n} bars for {sym}")           # BAD — can't parse/filter

# Don't use print
print(f"Debug: {value}")                             # BAD — goes to stdout, no level

# Don't import logging directly
import logging                                        # BAD — use get_logger()
logger = logging.getLogger(__name__)                  # BAD

# Don't import loguru directly
from loguru import logger                             # BAD — bypasses structlog pipeline

# Don't use the custom Logger class from logging_utils.py
from lib.utils.logging_utils import get_logger        # BAD — different get_logger()
from lib.utils.setup_logging import setup_logging      # BAD — different setup_logging()

# Don't catch and stringify exceptions
except Exception as e:
    logger.error(f"Failed: {e}")                      # BAD — loses traceback
```

---

## Log Levels Guide

| Level | When to Use | Example |
|-------|------------|---------|
| `DEBUG` | Internal state useful during development only | `logger.debug("cache_hit", key="bars:MES", age_s=12)` |
| `INFO` | Normal operations worth recording | `logger.info("engine_started", symbols=9, interval="1m")` |
| `WARNING` | Something unexpected but recoverable | `logger.warning("stale_data", symbol="ZW", age_min=15)` |
| `ERROR` | Something failed and needs attention | `logger.error("rithmic_disconnect", account="acc1", exc_info=True)` |
| `CRITICAL` | System is unusable | `logger.critical("database_unreachable", host=db_host)` |

**Rule of thumb**: If you wouldn't want to be woken up for it, it's not `ERROR`. If the system can keep running, it's `WARNING`.

---

## Contextual Logging with `bind()`

Use `bind()` to attach persistent context to a logger instance. This is especially useful in classes or long-running loops:

```python
class CopyTrader:
    def __init__(self, main_account: str):
        self.log = get_logger(__name__).bind(component="copy_trader", main=main_account)

    async def execute_copy(self, order, slave_accounts):
        for acc in slave_accounts:
            acc_log = self.log.bind(slave=acc.key)
            acc_log.info("copying_order", symbol=order.symbol, qty=order.qty)
            try:
                result = await acc.place_order(order)
                acc_log.info("copy_filled", fill_price=result.price)
            except Exception:
                acc_log.exception("copy_failed")
```

Every log line from `acc_log` will automatically include `component=copy_trader`, `main=<account>`, and `slave=<key>`.

---

## File Locations & What Goes Where

| File | Purpose | Who Imports It |
|------|---------|---------------|
| `src/lib/core/logging_config.py` | **THE** logging config. `setup_logging()` + `get_logger()` | Every module in the project |
| `src/lib/utils/logging_utils.py` | **LEGACY** — loguru-based Logger class | Being phased out. Do NOT use in new code. |
| `src/lib/utils/setup_logging.py` | **LEGACY** — wrapper around logging_utils.py | Being phased out. Do NOT use in new code. |

### Migration Status

The legacy `logging_utils.py` and `setup_logging.py` will remain in the codebase until all consumers are migrated. They still work — they just output through loguru's pipeline instead of structlog's.

---

## Migration Guide — Converting Existing Code

### From `import logging` (60+ files)

This is the most common pattern. The migration is mechanical:

```python
# BEFORE
import logging
logger = logging.getLogger("api.dashboard")
logger.info("Dashboard rendered for %s", symbol)

# AFTER
from lib.core.logging_config import get_logger
logger = get_logger(__name__)
logger.info("dashboard_rendered", symbol=symbol)
```

Since `setup_logging()` wires structlog's formatter onto the stdlib root handler, existing `logging.getLogger()` calls already get structlog formatting. So this migration is **low urgency** — it just makes the code consistent and enables key-value logging.

### From `from loguru import logger` (36 files)

Loguru logs bypass the structlog pipeline entirely. These should be migrated with higher priority:

```python
# BEFORE
from loguru import logger
logger.info(f"Loaded {n} bars for {symbol}")
logger.bind(name="lib.core.db").debug("Connection registered")

# AFTER
from lib.core.logging_config import get_logger
logger = get_logger(__name__)
logger.info("bars_loaded", symbol=symbol, count=n)
logger.debug("connection_registered")
```

### From `lib.utils.logging_utils` / `lib.utils.setup_logging` (3 files)

```python
# BEFORE
from lib.utils.setup_logging import get_module_logger, setup_logging
logger = get_module_logger("my_module")
setup_logging(service_name="engine", log_level="DEBUG")

# AFTER
from lib.core.logging_config import get_logger, setup_logging
logger = get_logger(__name__)
setup_logging(service="engine", level="DEBUG")
```

---

## Priority Migration Order

| Priority | Files | Why |
|----------|-------|-----|
| 🔴 High | `src/lib/core/db/__init__.py`, `src/lib/core/db/base.py`, `src/lib/core/db/postgres.py` | loguru in core infra — bypasses structlog |
| 🔴 High | `src/lib/core/base.py`, `src/lib/core/service.py`, `src/lib/core/runner.py` | Import legacy `setup_logging.py` |
| 🟡 Medium | `src/lib/integrations/*.py` | loguru mix — `rithmic_client.py`, `massive_client.py`, etc. |
| 🟢 Low | `src/lib/services/data/api/*.py` | Already use stdlib `logging.getLogger()` which works through structlog — just needs keyword cleanup |
| 🟢 Low | `src/lib/services/engine/*.py` | Same as above |

---

## Docker / Production

In `docker-compose.yml`, set JSON logging for all services:

```yaml
environment:
  LOG_LEVEL: INFO
  LOG_FORMAT: json
```

JSON output looks like:

```json
{"event": "order_placed", "symbol": "MES", "qty": 2, "level": "info", "timestamp": "2026-03-13T14:23:01Z", "service": "engine", "logger": "engine.copy_trader"}
```

This is parseable by Grafana Loki, Datadog, CloudWatch, `jq`, etc.

---

## Suppressed Libraries

`setup_logging()` in `logging_config.py` automatically suppresses noisy third-party loggers to `WARNING` level:

- `uvicorn.access`
- `httpx`, `httpcore`
- `websockets`, `urllib3`
- `sqlalchemy.engine`
- `hmmlearn`
- `matplotlib`

To add more, edit the `noisy` list in `logging_config.py`.

---

## Testing

In tests, logging works normally. To assert on log output, use `structlog.testing`:

```python
import structlog
from structlog.testing import capture_logs

def test_order_logging():
    with capture_logs() as logs:
        place_order("MES", 2)

    assert any(
        log["event"] == "order_placed" and log["symbol"] == "MES"
        for log in logs
    )
```

---

## Quick Reference

```python
# ─── Import ───────────────────────────────────────────
from lib.core.logging_config import get_logger
logger = get_logger(__name__)

# ─── Basic logging ────────────────────────────────────
logger.debug("cache_checked", key="bars:MES")
logger.info("bars_loaded", symbol="MES", count=500)
logger.warning("data_stale", symbol="ZW", age_min=15)
logger.error("fetch_failed", url=url, exc_info=True)
logger.exception("unhandled_error")  # auto-attaches traceback

# ─── Bind persistent context ─────────────────────────
log = logger.bind(account="tpt_eval", session="us")
log.info("position_opened", symbol="MES")

# ─── In service entry points only ─────────────────────
from lib.core.logging_config import setup_logging
setup_logging(service="data-service")          # call once at startup
setup_logging(service="engine", level="DEBUG") # override level
```
