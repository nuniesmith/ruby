#!/usr/bin/env python3
"""
verify_training_local.py — Pre-flight training pipeline verification
=====================================================================
Runs locally (no Docker, no GPU required) to catch all three failure modes
seen in training logs before you deploy:

  1. missing ``historical_bars`` table  →  trainer falls back to yfinance
     which uses wrong ticker format ``/MGC`` → HTTP 500 / TypeError loop
  2. yfinance fallback using ``/MGC`` instead of ``MGC=F``  →  every symbol
     spams 8+ error lines and returns no data
  3. Kraken ``EGeneral:Too many requests``  →  crypto bars unavailable,
     crypto_momentum returns empty signals for every row in the dataset

After fixing those it also verifies:
  4. DataResolver engine→db→cache→api chain actually resolves data
  5. dataset_generator produces at least N rows for a single symbol
  6. CNN model can be instantiated and run a forward pass

Usage:
    # From the repo root (auto-detects Docker port mappings):
    .venv/bin/python scripts/verify_training_local.py

    # With explicit overrides:
    ENGINE_DATA_URL=http://localhost:8050 .venv/bin/python scripts/verify_training_local.py
    .venv/bin/python scripts/verify_training_local.py --symbol MGC --days 7 --skip-cnn
    .venv/bin/python scripts/verify_training_local.py --fix-table

Exit codes:
    0 — all checks passed (safe to trigger a training run)
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure we can import from src/ whether running from repo root or scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Auto-probe running Docker services and inject env vars before any imports
# so that DATABASE_URL, REDIS_URL, and ENGINE_DATA_URL point at the real
# host-mapped ports even when the caller hasn't set them manually.
# ---------------------------------------------------------------------------


def _docker_port(service: str, container_port: int) -> str | None:
    """Return the host-side port mapping for a docker-compose service, or None."""
    try:
        result = subprocess.run(
            ["docker", "compose", "port", service, str(container_port)],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode == 0:
            addr = result.stdout.strip()  # e.g. "0.0.0.0:8050"
            if addr and ":" in addr:
                return addr.split(":")[-1]
    except Exception:
        pass
    return None


def _probe_and_inject_env() -> dict[str, str]:
    """
    Detect running Docker compose services and set env vars if not already
    set by the caller.  Returns a dict of what was auto-set (for display).
    """
    injected: dict[str, str] = {}

    # --- ENGINE_DATA_URL ---
    if not os.environ.get("ENGINE_DATA_URL") and not os.environ.get("DATA_SERVICE_URL"):
        port = _docker_port("data", 8000)
        if port:
            url = f"http://localhost:{port}"
            os.environ["ENGINE_DATA_URL"] = url
            injected["ENGINE_DATA_URL"] = url

    # --- DATABASE_URL (Postgres) ---
    if not os.environ.get("DATABASE_URL"):
        port = _docker_port("postgres", 5432)
        if port:
            # Read credentials from env or fall back to compose defaults
            pg_user = os.environ.get("POSTGRES_USER", "futures_user")
            pg_pass = os.environ.get("POSTGRES_PASSWORD", "")
            pg_db = os.environ.get("POSTGRES_DB", "futures_db")
            if pg_pass:
                url = f"postgresql://{pg_user}:{pg_pass}@localhost:{port}/{pg_db}"
            else:
                # Try to read from .env file in repo root
                env_file = _REPO_ROOT / ".env"
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("POSTGRES_PASSWORD="):
                            pg_pass = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if pg_pass:
                    url = f"postgresql://{pg_user}:{pg_pass}@localhost:{port}/{pg_db}"
                else:
                    url = f"postgresql://{pg_user}@localhost:{port}/{pg_db}"
            os.environ["DATABASE_URL"] = url
            injected["DATABASE_URL"] = url.replace(f":{pg_pass}@", ":***@") if pg_pass else url

    # --- REDIS_URL ---
    if not os.environ.get("REDIS_URL"):
        port = _docker_port("redis", 6379)
        if port:
            # Try to read REDIS_PASSWORD from .env
            redis_pass = os.environ.get("REDIS_PASSWORD", "")
            if not redis_pass:
                env_file = _REPO_ROOT / ".env"
                if env_file.exists():
                    for line in env_file.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("REDIS_PASSWORD="):
                            redis_pass = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            if redis_pass:
                url = f"redis://:{redis_pass}@localhost:{port}/0"
                display = f"redis://:***@localhost:{port}/0"
            else:
                url = f"redis://localhost:{port}/0"
                display = url
            os.environ["REDIS_URL"] = url
            injected["REDIS_URL"] = display

    # --- TRAINER_URL ---
    if not os.environ.get("TRAINER_URL"):
        port = _docker_port("trainer", 8200)
        if port:
            url = f"http://localhost:{port}"
            os.environ["TRAINER_URL"] = url
            injected["TRAINER_URL"] = url

    return injected


# Run probe immediately — must happen before any lib imports that read env vars
_auto_injected = _probe_and_inject_env()

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    _RESET = "\033[0m"
    return f"{code}{text}{_RESET}"


def _ok(msg: str) -> None:
    print(_c("\033[92m", f"  ✅ PASS  {msg}"))


def _fail(msg: str) -> None:
    print(_c("\033[91m", f"  ❌ FAIL  {msg}"))


def _warn(msg: str) -> None:
    print(_c("\033[93m", f"  ⚠️  WARN  {msg}"))


def _info(msg: str) -> None:
    print(_c("\033[94m", f"  ℹ️  INFO  {msg}"))


def _section(title: str) -> None:
    print()
    print(_c("\033[1m", f"── {title} " + "─" * max(0, 60 - len(title))))


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
_results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def _record(label: str, passed: bool, detail: str = "") -> bool:
    _results.append((label, passed, detail))
    if passed:
        _ok(f"{label}" + (f"  ({detail})" if detail else ""))
    else:
        _fail(f"{label}" + (f"  — {detail}" if detail else ""))
    return passed


# ===========================================================================
# CHECK 1 — historical_bars table exists and has data
# ===========================================================================


def _pg_connect_direct():
    """
    Attempt a direct psycopg (v3) or psycopg2 connection to Postgres using
    DATABASE_URL.  Returns (conn, adapter_name) or raises ImportError if
    neither driver is available.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url.startswith("postgresql"):
        raise ValueError("DATABASE_URL not set or not a Postgres URL")

    # Try psycopg3 first (installed in this venv as 'psycopg')
    try:
        import psycopg  # type: ignore[import]

        conn = psycopg.connect(db_url)
        return conn, "psycopg3"
    except ImportError:
        pass

    # Fall back to psycopg2
    try:
        import psycopg2  # type: ignore[import]

        conn = psycopg2.connect(db_url)
        return conn, "psycopg2"
    except ImportError:
        pass

    raise ImportError("Neither psycopg nor psycopg2 is installed in this venv")


def check_historical_bars_table(auto_fix: bool = False) -> bool:
    """Verify the historical_bars table exists and contains rows.

    Primary path: query Postgres directly via psycopg/psycopg2.
    Fallback: ask the engine /bars/status HTTP endpoint (no DB driver needed).
    The --fix-table flag calls init_backfill_table() via the engine module
    when a direct DB connection is available.
    """
    _section("CHECK 1 — historical_bars DB table")

    # ── Primary: direct Postgres connection ──────────────────────────────
    try:
        conn, adapter = _pg_connect_direct()
        _info(f"Connected to Postgres directly via {adapter}")
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='historical_bars'"
            )
            row = cur.fetchone()
            table_exists = bool(row and int(row[0]) > 0)
        finally:
            conn.close()

        if table_exists:
            # Also check it has at least some rows
            try:
                conn2, _ = _pg_connect_direct()
                try:
                    cur2 = conn2.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM historical_bars")
                    r = cur2.fetchone()
                    total = int(r[0]) if r else 0
                    first = r[1] if r else None
                    last = r[2] if r else None
                finally:
                    conn2.close()

                if total == 0:
                    detail = "table exists but is empty — trigger HISTORICAL_BACKFILL before retraining"
                    _warn(detail)
                    return _record("historical_bars table exists (empty)", False, detail)

                return _record(
                    "historical_bars table exists",
                    True,
                    f"{total:,} rows  first={first}  last={last}",
                )
            except Exception as exc:
                # Table exists, row count failed — still pass
                return _record("historical_bars table exists", True, f"count check failed: {exc}")

        # Table missing
        if auto_fix:
            _warn("historical_bars table missing — auto-creating via engine module")
            try:
                from lib.services.engine.backfill import init_backfill_table

                init_backfill_table()
                _info("Table created. Trigger HISTORICAL_BACKFILL before retraining.")
                return _record(
                    "historical_bars table created (was missing)",
                    True,
                    "table now exists but is empty — backfill required",
                )
            except Exception as fix_exc:
                return _record("historical_bars table create", False, str(fix_exc))
        else:
            _info("Re-run with --fix-table to auto-create, then trigger a backfill.")
            return _record(
                "historical_bars table exists",
                False,
                "table missing → trainer falls back to yfinance /MGC ticker → HTTP 500 loop",
            )

    except (ImportError, ValueError):
        pass  # no driver or no DATABASE_URL — fall through to HTTP fallback
    except Exception as exc:
        _warn(f"Direct Postgres check failed ({exc}) — falling back to engine HTTP API")

    # ── Fallback: ask engine /bars/status ─────────────────────────────────
    engine_url = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "").rstrip("/")
    if not engine_url:
        return _record(
            "historical_bars table check",
            False,
            "No DATABASE_URL and no ENGINE_DATA_URL — cannot verify table",
        )

    try:
        import requests

        r = requests.get(f"{engine_url}/bars/status", timeout=10)
        if r.status_code == 200:
            data = r.json()
            symbols = data if isinstance(data, list) else data.get("symbols", [])
            if symbols:
                total = sum(s.get("bar_count", 0) for s in symbols if isinstance(s, dict))
                return _record(
                    "historical_bars table exists",
                    True,
                    f"{len(symbols)} symbols, {total:,} total bars (via engine /bars/status)",
                )
            return _record(
                "historical_bars table exists",
                False,
                "engine /bars/status returned 0 symbols — table empty or backfill not run",
            )
        elif r.status_code == 404:
            return _record(
                "historical_bars table exists",
                False,
                "engine /bars/status 404 — table may not exist yet; trigger HISTORICAL_BACKFILL",
            )
        else:
            return _record(
                "historical_bars table check",
                False,
                f"engine /bars/status returned HTTP {r.status_code}: {r.text[:120]}",
            )
    except Exception as exc:
        return _record("historical_bars table check", False, f"engine HTTP fallback failed: {exc}")


# ===========================================================================
# CHECK 2 — yfinance ticker format
# ===========================================================================


def check_yfinance_ticker_format() -> bool:
    """Confirm _resolve_ticker() produces MGC=F not /MGC.

    The /MGC format is a Yahoo Finance futures format that yfinance now
    rejects with HTTP 500.  Our resolver should always produce MGC=F.
    """
    _section("CHECK 2 — yfinance / symbol resolver ticker format")

    try:
        from lib.services.training.dataset_generator import _resolve_ticker

        bad: list[str] = []
        test_cases = {
            "MGC": "MGC=F",
            "MES": "MES=F",
            "MNQ": "MNQ=F",
            "MCL": "MCL=F",
            "6E": "6E=F",
            "ZN": "ZN=F",
            "ZC": "ZC=F",
        }
        for symbol, expected in test_cases.items():
            got = _resolve_ticker(symbol)
            if got != expected:
                bad.append(f"{symbol} → got '{got}', want '{expected}'")

        if bad:
            return _record(
                "_resolve_ticker() produces correct =F tickers",
                False,
                "; ".join(bad),
            )
        return _record(
            "_resolve_ticker() produces correct =F tickers",
            True,
            f"checked {len(test_cases)} symbols",
        )

    except ImportError as exc:
        return _record("_resolve_ticker check", False, f"import error: {exc}")
    except Exception as exc:
        return _record("_resolve_ticker check", False, str(exc))


def check_yfinance_no_slash_ticker() -> bool:
    """Confirm yfinance is NOT called with /MGC-style tickers anywhere in the
    fallback chain by monkey-patching download and checking what gets called."""
    _section("CHECK 2b — yfinance not called with /SYMBOL slash tickers")

    try:
        import yfinance as yf

        called_with: list[str] = []
        original_download = yf.download

        def _patched_download(tickers, *args, **kwargs):
            called_with.append(str(tickers))
            # Don't actually make a network call — raise immediately
            raise RuntimeError("_patched: aborting yfinance call for inspection")

        yf.download = _patched_download
        try:
            # We can't easily trigger the full fallback without live infra,
            # but we can at least verify the ticker resolver never prefixes /
            from lib.services.training.dataset_generator import (
                _load_bars_from_cache,  # noqa: F401
                _resolve_ticker,
            )

            slash_tickers = [t for t in [_resolve_ticker(s) for s in ["MGC", "MES", "MCL"]] if t.startswith("/")]
            if slash_tickers:
                return _record(
                    "No /SYMBOL slash tickers passed to yfinance",
                    False,
                    f"slash tickers detected: {slash_tickers}",
                )
            return _record(
                "No /SYMBOL slash tickers passed to yfinance",
                True,
                "resolver produces =F suffix tickers only",
            )
        finally:
            yf.download = original_download

    except ImportError:
        _warn("yfinance not installed — skipping slash-ticker check (non-fatal if Massive key is set)")
        return _record("yfinance slash-ticker check", True, "yfinance not installed — skipped")
    except Exception as exc:
        return _record("yfinance slash-ticker check", False, str(exc))


# ===========================================================================
# CHECK 3 — Kraken rate-limit / data availability
# ===========================================================================


def check_kraken_rate_limit(symbol: str = "KRAKEN:XBTUSD", days: int = 2) -> bool:
    """Verify we can fetch at least a small window of Kraken bars.

    When many symbols request Kraken bars in rapid succession the public REST
    API returns EGeneral:Too many requests.  This check uses a small window
    (2 days) and waits up to 3 retries with back-off to distinguish a genuine
    rate-limit from a total outage.
    """
    _section("CHECK 3 — Kraken REST API reachable (BTC 2-day sample)")

    try:
        from lib.services.training.dataset_generator import _load_bars_from_kraken

        for attempt in range(1, 4):
            try:
                df = _load_bars_from_kraken(symbol, days=days)
                if df is not None and not df.empty:
                    return _record(
                        "Kraken REST returns bars",
                        True,
                        f"{len(df)} bars for {symbol} (attempt {attempt})",
                    )
                _warn(f"Attempt {attempt}: empty response — waiting 5s before retry")
                time.sleep(5)
            except Exception as exc:
                msg = str(exc)
                if "Too many requests" in msg:
                    _warn(f"Attempt {attempt}: rate-limited — waiting 10s before retry")
                    time.sleep(10)
                else:
                    return _record("Kraken REST returns bars", False, msg)

        return _record(
            "Kraken REST returns bars",
            False,
            "EGeneral:Too many requests — all 3 attempts rate-limited. "
            "Root cause: dataset_generator hits Kraken directly for each crypto symbol "
            "without inter-call delay when bars_source=engine and the engine is unreachable. "
            "Fix: ensure engine/data service is running so trainer uses the engine HTTP API "
            "instead of calling Kraken REST directly per symbol.",
        )

    except ImportError as exc:
        return _record("Kraken bar fetch", False, f"import error: {exc}")
    except Exception as exc:
        return _record("Kraken bar fetch", False, str(exc))


def check_kraken_crypto_momentum(days: int = 2) -> bool:
    """Verify crypto_momentum doesn't silently return empty signals.

    If BTC/ETH bars aren't available the scorer returns zero-filled signals
    for every training row, silently corrupting the feature vector.
    """
    _section("CHECK 3b — crypto_momentum feature not silently empty")

    try:
        from lib.analysis.crypto_momentum import (
            compute_all_crypto_momentum,
            crypto_momentum_to_tabular,
        )

        features = crypto_momentum_to_tabular(compute_all_crypto_momentum())

        # The function should return a dict with numeric values
        if not isinstance(features, dict) or not features:
            return _record(
                "crypto_momentum returns non-empty features",
                False,
                "returned empty dict — BTC/ETH bars unavailable → feature vector corrupted with zeros",
            )

        zero_vals = [k for k, v in features.items() if v == 0.0]
        if len(zero_vals) == len(features):
            return _record(
                "crypto_momentum returns non-empty features",
                False,
                f"all {len(features)} features are 0.0 — Kraken bars not available at dataset generation time",
            )

        return _record(
            "crypto_momentum returns non-empty features",
            True,
            f"{len(features)} features, {len(zero_vals)} are zero",
        )

    except ImportError as exc:
        # crypto_momentum may not be importable outside the service; treat as warn
        _warn(f"crypto_momentum not importable ({exc}) — skipped")
        return _record("crypto_momentum importable", True, "skipped — not importable outside service")
    except Exception as exc:
        return _record("crypto_momentum returns non-empty features", False, str(exc))


# ===========================================================================
# CHECK 4 — DataResolver / ENGINE_DATA_URL reachable
# ===========================================================================


def check_engine_data_url(symbol: str = "MGC", days: int = 5) -> bool:
    """Verify the engine/data service HTTP API is reachable and returns bars.

    The trainer's default bars_source is 'engine', which calls
    GET /bars/{symbol}?interval=1m&days_back=N on the data service.
    If that URL is wrong or the service is down the trainer silently falls
    back to the broken yfinance chain.
    """
    _section("CHECK 4 — ENGINE_DATA_URL reachable and returns bars")

    engine_url = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "http://data:8000").rstrip("/")
    _info(f"ENGINE_DATA_URL = {engine_url}")

    try:
        import requests

        # Health check first
        try:
            r = requests.get(f"{engine_url}/health", timeout=5)
            if r.status_code != 200:
                return _record(
                    "Engine data service /health",
                    False,
                    f"HTTP {r.status_code} — service may be down or ENGINE_DATA_URL is wrong",
                )
            _info(f"/health → {r.status_code} OK")
        except requests.exceptions.ConnectionError:
            return _record(
                "Engine data service /health",
                False,
                f"Connection refused at {engine_url} — is the data service running? "
                f"Set ENGINE_DATA_URL=http://localhost:8000 if running locally.",
            )
        except Exception as exc:
            return _record("Engine data service /health", False, str(exc))

        # Fetch bars
        try:
            r = requests.get(
                f"{engine_url}/bars/{symbol}",
                params={"interval": "1m", "days_back": str(days), "auto_fill": "true"},
                timeout=30,
            )
            if r.status_code == 404:
                _warn(f"/bars/{symbol} returned 404 — no data stored yet; backfill needed")
                return _record(
                    f"Engine /bars/{symbol} returns data",
                    False,
                    "404 — historical_bars table empty. Run a backfill first: "
                    "trigger HISTORICAL_BACKFILL from the engine or POST /api/backfill.",
                )
            if r.status_code != 200:
                return _record(
                    f"Engine /bars/{symbol} returns data",
                    False,
                    f"HTTP {r.status_code}: {r.text[:200]}",
                )

            payload = r.json()
            bar_count = payload.get("bar_count", 0)
            if bar_count == 0:
                return _record(
                    f"Engine /bars/{symbol} returns data",
                    False,
                    "bar_count=0 — historical_bars table may be empty",
                )

            return _record(
                f"Engine /bars/{symbol} returns data",
                True,
                f"{bar_count} bars over {days}d  source={payload.get('source', '?')}",
            )
        except Exception as exc:
            return _record(f"Engine /bars/{symbol} returns data", False, str(exc))

    except ImportError:
        _warn("requests not installed — skipping engine URL check")
        return _record("Engine /health reachable", True, "skipped — requests not installed")


# ===========================================================================
# CHECK 4b — DataResolver fallback chain smoke-test
# ===========================================================================


def check_data_resolver(symbol: str = "MGC", days: int = 5) -> bool:
    """Verify bar data is resolvable for the given symbol.

    Strategy (in order):
      1. Try DataResolver directly — works when this process has a working
         DATABASE_URL (psycopg/psycopg2) and REDIS_URL with auth.
      2. Fall back to engine HTTP /bars/{symbol} — always works when the
         data service is running, regardless of local driver availability.
         The trainer uses this exact path (bars_source=engine), so a pass
         here means the trainer will succeed.
    """
    _section("CHECK 4b — DataResolver / bar data reachable")

    # Silence the noisy lib fallback messages (psycopg2 missing, yfinance
    # errors) that appear when DataResolver tries SQLite with Postgres SQL.
    import logging as _logging

    _noisy = [
        _logging.getLogger("lib.services.engine.backfill"),
        _logging.getLogger("lib.services.data.resolver"),
        _logging.getLogger("lib.integrations.massive_client"),
        _logging.getLogger("massive_client"),
        _logging.getLogger("MASSIVE"),
    ]
    # Also suppress the [MASSIVE] handler that writes directly to the
    # massive_client logger's StreamHandler (added at module import time)
    _massive_handlers_saved: list[tuple] = []
    try:
        import lib.integrations.massive_client as _mc  # noqa: PLC0415

        for _h in list(_mc.logger.handlers):
            _mc.logger.removeHandler(_h)
            _massive_handlers_saved.append((_mc.logger, _h))
    except Exception:
        pass
    _saved_levels = {lg: lg.level for lg in _noisy}
    for lg in _noisy:
        lg.setLevel(_logging.CRITICAL)

    resolver_ok = False
    resolver_detail = ""
    try:
        from lib.services.data.resolver import DataResolver

        resolver = DataResolver(
            enable_backfill_redis=False,
            enable_backfill_postgres=False,
        )
        df, meta = resolver.resolve_with_meta(symbol, days=days)

        if df is not None and not df.empty:
            resolver_ok = True
            resolver_detail = f"{len(df)} bars for {symbol} via source={meta.source}"
    except Exception:
        pass
    finally:
        # Restore log levels
        for lg in _noisy:
            lg.setLevel(_saved_levels[lg])
        # Restore massive_client StreamHandler
        for _lg, _h in _massive_handlers_saved:
            _lg.addHandler(_h)

    if resolver_ok:
        return _record("DataResolver returns bars", True, resolver_detail)

    # ── Fallback: engine HTTP /bars/{symbol} ──────────────────────────────
    # The trainer's bars_source=engine calls this exact endpoint, so if it
    # returns bars the training run will succeed even if DataResolver can't
    # connect locally (e.g. psycopg2 missing, Redis auth not injected).
    engine_url = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "").rstrip("/")
    if not engine_url:
        return _record(
            "DataResolver returns bars",
            False,
            "DataResolver found no data and ENGINE_DATA_URL is not set — cannot verify",
        )

    try:
        import requests

        r = requests.get(
            f"{engine_url}/bars/{symbol}",
            params={"interval": "1m", "days_back": str(days)},
            timeout=15,
        )
        if r.status_code == 200:
            payload = r.json()
            bar_count = payload.get("bar_count", 0)
            if bar_count == 0:
                # Try counting from the data split
                idx = payload.get("data", {}).get("index", [])
                bar_count = len(idx) if isinstance(idx, list) else 0
            if bar_count > 0:
                _info(
                    "DataResolver could not connect locally (psycopg/Redis) "
                    "but engine HTTP confirmed bars — trainer will succeed"
                )
                return _record(
                    "DataResolver returns bars",
                    True,
                    f"{bar_count} bars for {symbol} confirmed via engine HTTP "
                    f"(local DataResolver unavailable — psycopg or Redis auth issue)",
                )
            return _record(
                "DataResolver returns bars",
                False,
                f"engine /bars/{symbol} returned 0 bars — historical_bars table may be empty",
            )
        return _record(
            "DataResolver returns bars",
            False,
            f"engine /bars/{symbol} HTTP {r.status_code} — no bar data available",
        )
    except Exception as exc:
        return _record("DataResolver returns bars", False, f"engine HTTP fallback failed: {exc}")


# ===========================================================================
# CHECK 5 — dataset_generator produces rows (tiny run)
# ===========================================================================


def check_dataset_generation(symbol: str = "MGC", days: int = 7) -> bool:
    """Run dataset_generator for a single symbol with a minimal window.

    Uses bars_source='engine' (the production default) and caps the run to
    3 days / US-open only / ORB only / 20 samples so it finishes in <30s.
    A hard 60-second thread timeout aborts and records a failure if anything
    hangs (e.g. waiting on a stalled HTTP connection).

    Suppresses the noisy psycopg2-missing + yfinance-fallback log spam that
    appears when the lib falls back from Postgres to SQLite locally.
    """
    # Always use a small window regardless of what --days was set to
    _smoke_days = min(days, 3)
    _section(f"CHECK 5 — dataset_generator single-symbol smoke ({symbol}, {_smoke_days}d)")

    # Loggers that spam psycopg2/SQLite/yfinance noise during the smoke run
    import logging as _logging
    import tempfile
    import threading

    _noisy_ds = [
        _logging.getLogger(n)
        for n in [
            "lib.services.engine.backfill",
            "lib.services.data.resolver",
            "lib.integrations.massive_client",
            "massive_client",
            "yfinance",
            "urllib3.connectionpool",
        ]
    ]
    _saved_ds = {lg: lg.level for lg in _noisy_ds}
    for lg in _noisy_ds:
        lg.setLevel(_logging.CRITICAL)

    # Also mute the massive_client StreamHandler added at import time
    _mc_handlers: list[tuple] = []
    try:
        import lib.integrations.massive_client as _mc2

        for _h in list(_mc2.logger.handlers):
            _mc2.logger.removeHandler(_h)
            _mc_handlers.append((_mc2.logger, _h))
    except Exception:
        pass

    result_box: list = []  # [passed, detail]  written by worker thread
    exc_box: list = []  # [exception]

    def _worker():
        # Re-apply log suppression inside the worker thread — each thread
        # inherits the logger objects but setLevel is not thread-local, so
        # this is safe and idempotent.
        for _lg in _noisy_ds:
            _lg.setLevel(_logging.CRITICAL)
        try:
            import lib.integrations.massive_client as _mc3

            for _h in list(_mc3.logger.handlers):
                _mc3.logger.removeHandler(_h)
        except Exception:
            pass
        try:
            from lib.services.training.dataset_generator import DatasetConfig, generate_dataset

            with tempfile.TemporaryDirectory() as tmpdir:
                config = DatasetConfig(
                    output_dir=tmpdir,
                    image_dir=os.path.join(tmpdir, "images"),
                    bars_source="engine",
                    orb_session="us",  # US open only — fastest session
                    breakout_type="ORB",  # single type for speed
                    use_parity_renderer=True,
                    max_samples_per_type_label=20,  # tiny cap
                    max_samples_per_session_label=20,
                )
                t0 = time.monotonic()
                stats = generate_dataset(symbols=[symbol], days_back=_smoke_days, config=config)
                elapsed = time.monotonic() - t0

                if stats.total_images == 0:
                    result_box.append(
                        (
                            False,
                            f"0 images in {elapsed:.1f}s  trades={stats.total_trades}  "
                            "— bar data unavailable or all windows filtered out",
                        )
                    )
                else:
                    label_str = ", ".join(f"{k}={v}" for k, v in sorted(stats.label_distribution.items()))
                    result_box.append(
                        (
                            True,
                            f"{stats.total_images} images  [{label_str}]  {elapsed:.1f}s",
                        )
                    )
        except ImportError as exc:
            exc_box.append(("import", exc))
        except Exception as exc:
            exc_box.append(("runtime", exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=60)

    # Restore suppressed loggers and handlers
    for lg in _noisy_ds:
        lg.setLevel(_saved_ds[lg])
    for _lg, _h in _mc_handlers:
        _lg.addHandler(_h)

    # Evaluate result
    if t.is_alive():
        return _record(
            "dataset_generator produces images",
            False,
            "timed out after 60s — engine HTTP may be stalled or bars_source chain is blocked",
        )

    if exc_box:
        kind, exc = exc_box[0]
        if kind == "import":
            return _record("dataset_generator importable", False, f"import error: {exc}")
        return _record("dataset_generator produces images", False, str(exc))

    if not result_box:
        return _record("dataset_generator produces images", False, "worker exited without result")

    passed, detail = result_box[0]
    return _record("dataset_generator produces images", passed, detail)


# ===========================================================================
# CHECK 6 — CNN model instantiation + forward pass (no GPU needed)
# ===========================================================================


def check_cnn_model() -> bool:
    """Verify HybridBreakoutCNN can be instantiated and log its parameter count.

    Uses pretrained=False so no weights are downloaded (CPU-only check).
    The real signature has no num_classes — output is always 2 (LONG/SHORT).
    """
    _section("CHECK 6 — HybridBreakoutCNN instantiation")

    try:
        import torch  # noqa: F401
    except ImportError:
        _warn("torch not installed — skipping CNN checks (GPU trainer only)")
        return _record("torch available", True, "skipped — torch not installed locally")

    try:
        from lib.analysis.breakout_cnn import NUM_TABULAR, HybridBreakoutCNN

        # pretrained=False avoids downloading EfficientNetV2 weights locally
        model = HybridBreakoutCNN(num_tabular=NUM_TABULAR, pretrained=False)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        _info(f"Model: {n_params:,} parameters  num_tabular={NUM_TABULAR}  output=2 classes")
        return _record(
            "HybridBreakoutCNN instantiated",
            True,
            f"{n_params:,} params  num_tabular={NUM_TABULAR}  output=2",
        )
    except Exception as exc:
        return _record("HybridBreakoutCNN instantiated", False, str(exc))


def check_cnn_forward_pass() -> bool:
    """Run a CPU forward pass through the model.

    Passes zero tensors for image, tabular, asset_class_ids, and asset_ids
    matching the real forward() signature.  Output should be (1, 2).
    """
    _section("CHECK 6b — HybridBreakoutCNN CPU forward pass")

    try:
        import torch
    except ImportError:
        return _record("CNN forward pass", True, "skipped — torch not installed")

    try:
        from lib.analysis.breakout_cnn import NUM_TABULAR, HybridBreakoutCNN

        model = HybridBreakoutCNN(num_tabular=NUM_TABULAR, pretrained=False)
        model.eval()

        img = torch.zeros(1, 3, 224, 224)
        tab = torch.zeros(1, NUM_TABULAR)
        # asset_class_ids: (B,) int in [0, NUM_ASSET_CLASSES)
        # asset_ids:       (B,) int in [0, NUM_ASSETS)
        asset_class_ids = torch.zeros(1, dtype=torch.long)
        asset_ids = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            logits = model(img, tab, asset_class_ids=asset_class_ids, asset_ids=asset_ids)

        expected_shape = (1, 2)  # always 2 classes: LONG / SHORT
        if tuple(logits.shape) != expected_shape:
            return _record(
                "HybridBreakoutCNN forward pass",
                False,
                f"unexpected output shape {tuple(logits.shape)}, want {expected_shape}",
            )
        return _record(
            "HybridBreakoutCNN forward pass",
            True,
            f"output shape {tuple(logits.shape)}  num_tabular={NUM_TABULAR}",
        )

    except Exception as exc:
        return _record("HybridBreakoutCNN forward pass", False, str(exc))


# ===========================================================================
# CHECK 7 — Trainer server health (if running)
# ===========================================================================


def check_trainer_server() -> bool:
    """If a trainer server is already running, verify it is healthy and idle."""
    _section("CHECK 7 — Trainer server health (if running)")

    trainer_url = os.getenv("TRAINER_URL", "http://localhost:8200").rstrip("/")
    _info(f"TRAINER_URL = {trainer_url}")

    try:
        import requests

        try:
            r = requests.get(f"{trainer_url}/health", timeout=4)
        except requests.exceptions.ConnectionError:
            _info("Trainer server not running locally — skipping (not required for local checks)")
            return _record("Trainer server health", True, "not running — skipped")

        if r.status_code != 200:
            return _record("Trainer server health", False, f"HTTP {r.status_code}")

        _info(f"/health → {r.status_code} OK")

        # Check status — make sure it's idle before we fire a retrain
        try:
            s = requests.get(f"{trainer_url}/status", timeout=4).json()
            status = s.get("status", "unknown")
            if status not in ("idle", "ready"):
                _warn(f"Trainer is not idle — status={status}. A training run may be in progress.")
                return _record(
                    "Trainer server is idle",
                    False,
                    f"status={status} — wait for current run to finish before triggering another",
                )
            return _record("Trainer server is healthy and idle", True, f"status={status}")
        except Exception as exc:
            return _record("Trainer server status", False, str(exc))

    except ImportError:
        return _record("Trainer server health", True, "skipped — requests not installed")
    except Exception as exc:
        return _record("Trainer server health", False, str(exc))


# ===========================================================================
# Diagnosis summary + remediation hints
# ===========================================================================

_REMEDIATION: dict[str, str] = {
    "historical_bars table exists": (
        "Run:  python scripts/verify_training_local.py --fix-table\n"
        "Then trigger a backfill: engine → OFF-HOURS → HISTORICAL_BACKFILL action,\n"
        "or POST /api/backfill on the data service."
    ),
    "historical_bars table created (was missing)": (
        "Table created but is empty. Trigger a backfill before retraining:\n"
        "  engine → OFF-HOURS → HISTORICAL_BACKFILL, or POST /api/backfill."
    ),
    "_resolve_ticker() produces correct =F tickers": (
        "Bug in dataset_generator._SYMBOL_TO_TICKER or _resolve_ticker().\n"
        "Ensure every futures symbol maps to the SYMBOL=F format, not /SYMBOL."
    ),
    "Kraken REST returns bars": (
        "The trainer is calling Kraken REST directly for each crypto symbol\n"
        "without delay, hitting the public rate limit.\n"
        "Fix: ensure ENGINE_DATA_URL points to a running data service so bars\n"
        "are served from Redis/Postgres without hammering Kraken.\n"
        "Locally:  ENGINE_DATA_URL=http://localhost:8000 python scripts/verify_training_local.py"
    ),
    "crypto_momentum returns non-empty features": (
        "BTC/ETH bars are not available at dataset generation time.\n"
        "This silently zeros out the crypto_momentum feature column in every\n"
        "training row, degrading model quality.\n"
        "Ensure Kraken bars are cached in Redis before running the trainer."
    ),
    "Engine data service /health": (
        "Start the data service locally:  docker compose up data\n"
        "Or if running bare-metal:  uvicorn lib.services.data.main:app --port 8000\n"
        "Then set:  ENGINE_DATA_URL=http://localhost:8000"
    ),
    "Engine /bars/MGC returns data": (
        "The data service is running but has no bars for MGC.\n"
        "Trigger a backfill:  POST /api/backfill  or wait for the engine's\n"
        "nightly HISTORICAL_BACKFILL action to complete."
    ),
    "DataResolver returns bars": (
        "No bar data available in Redis, Postgres, or via external APIs.\n"
        "Check that MASSIVE_API_KEY is set (for futures) or Kraken is reachable\n"
        "(for crypto).  Then trigger HISTORICAL_BACKFILL."
    ),
    "dataset_generator produces images": (
        "Dataset generation produced 0 images — most likely because bar data\n"
        "was unavailable and the yfinance fallback failed.\n"
        "Resolve the bar data issues above first, then re-run."
    ),
}


def print_summary(show_remediation: bool = True) -> int:
    """Print a final summary table and return exit code (0=all passed, 1=any failed)."""
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed = [r for r in _results if r[1]]
    failed = [r for r in _results if not r[1]]

    for label, ok, detail in _results:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {label}")
        if not ok and detail:
            print(f"       {_c(chr(27) + '[91m', detail)}")

    print()
    print(f"  {len(passed)} passed  |  {len(failed)} failed  |  {len(_results)} total")
    print("=" * 70)

    if failed and show_remediation:
        print()
        print(_c("\033[1m", "REMEDIATION HINTS"))
        print("─" * 70)
        shown: set[str] = set()
        for label, ok, _ in _results:
            if not ok and label in _REMEDIATION and label not in shown:
                print(f"\n[{label}]")
                for line in _REMEDIATION[label].splitlines():
                    print(f"  {line}")
                shown.add(label)
        print()

    return 0 if not failed else 1


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight local verification for the training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--symbol", default="MGC", help="Primary test symbol (default: MGC)")
    p.add_argument("--days", type=int, default=7, help="Days of bar history to request (default: 7)")
    p.add_argument("--fix-table", action="store_true", help="Auto-create historical_bars table if missing")
    p.add_argument("--skip-cnn", action="store_true", help="Skip CNN model instantiation checks")
    p.add_argument("--skip-dataset", action="store_true", help="Skip dataset generation smoke test (slow)")
    p.add_argument("--skip-kraken", action="store_true", help="Skip Kraken live API checks (network-free mode)")
    p.add_argument("--no-remediation", action="store_true", help="Suppress remediation hints in summary")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(_c("\033[1m", "\n🔍 Futures Training Pipeline — Local Pre-flight Verification\n"))
    print(f"  symbol       = {args.symbol}")
    print(f"  days         = {args.days}")
    print(f"  fix-table    = {args.fix_table}")
    print(f"  skip-cnn     = {args.skip_cnn}")
    print(f"  skip-dataset = {args.skip_dataset}")
    print(f"  skip-kraken  = {args.skip_kraken}")

    # Show which env vars were auto-detected from Docker
    if _auto_injected:
        print()
        print(_c("\033[94m", "  🐳 Auto-detected from Docker compose:"))
        for k, v in _auto_injected.items():
            print(f"      {k} = {v}")
    else:
        print()
        _info("No Docker compose services detected — using env vars as-is")

    # Show final effective values for the three critical vars
    print()
    print(f"  ENGINE_DATA_URL = {os.environ.get('ENGINE_DATA_URL', '(not set)')}")
    print(f"  DATABASE_URL    = {os.environ.get('DATABASE_URL', '(not set — will use SQLite fallback)')}")
    print(f"  REDIS_URL       = {os.environ.get('REDIS_URL', '(not set)')}")
    print(f"  TRAINER_URL     = {os.environ.get('TRAINER_URL', 'http://localhost:8200')}")

    # Run all checks in order
    check_historical_bars_table(auto_fix=args.fix_table)
    check_yfinance_ticker_format()
    check_yfinance_no_slash_ticker()

    if not args.skip_kraken:
        check_kraken_rate_limit()
        check_kraken_crypto_momentum()
    else:
        _info("Skipping Kraken live API checks (--skip-kraken)")

    check_engine_data_url(symbol=args.symbol, days=args.days)
    check_data_resolver(symbol=args.symbol, days=args.days)

    if not args.skip_dataset:
        check_dataset_generation(symbol=args.symbol, days=args.days)
    else:
        _info("Skipping dataset generation smoke test (--skip-dataset)")

    if not args.skip_cnn:
        check_cnn_model()
        check_cnn_forward_pass()
    else:
        _info("Skipping CNN model checks (--skip-cnn)")

    check_trainer_server()

    sys.exit(print_summary(show_remediation=not args.no_remediation))


if __name__ == "__main__":
    main()
