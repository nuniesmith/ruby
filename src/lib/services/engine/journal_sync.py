"""
Journal Sync — Rithmic fill → trades_v2 auto-population
=========================================================
Implements the JOURNAL-SYNC scheduled background task.

Fill → Round-Trip Matching
--------------------------
Rithmic returns individual order fills (each buy/sell leg separately).
This module groups them into complete round-trip trades by:
  1. Sorting fills by (symbol, fill_time)
  2. Pairing each SELL fill with the most recent unmatched BUY fill for
     the same symbol (long trades), or BUY with SELL (short trades).
  3. Calculating P&L, hold time, and contracts from the matched pair.
  4. Writing the completed round-trip to ``trades_v2`` via
     ``upsert_trade_from_fill()``.
  5. For fills with no matching counterpart yet (still open), writing an
     OPEN trade record that gets updated on the next sync cycle.

Scheduler Integration
---------------------
``run_journal_sync()`` is the top-level entry point called by the engine
scheduler (``ActionType.JOURNAL_SYNC``).  It is async-safe and designed to
run every 5 minutes during the ACTIVE session.

Usage (engine main.py):
    from lib.services.engine.journal_sync import run_journal_sync
    # ... in action_handlers:
    ActionType.JOURNAL_SYNC: lambda: asyncio.run(run_journal_sync()),

Manual trigger via API:
    POST /journal/sync
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("journal_sync")

# ---------------------------------------------------------------------------
# Contract specs — tick value per point for P&L calculation
# Matches CONTRACT_SPECS in models.py; kept here to avoid a circular import.
# ---------------------------------------------------------------------------
_POINT_VALUES: dict[str, float] = {
    # CME Micro E-mini Equity Index
    "MES": 5.0,
    "MNQ": 2.0,
    "MYM": 0.50,
    "M2K": 5.0,
    # CME E-mini Equity Index
    "ES": 50.0,
    "NQ": 20.0,
    "YM": 5.0,
    "RTY": 50.0,
    # CME Micro Metals
    "MGC": 10.0,
    "SIL": 1000.0,
    # CME Full Metals
    "GC": 100.0,
    "SI": 5000.0,
    # CME Energy
    "CL": 1000.0,
    "QM": 500.0,
    # CME Interest Rates / Treasuries
    "ZN": 1000.0,
    "ZB": 1000.0,
    "ZF": 1000.0,
    "ZT": 2000.0,
    # Crypto (Kraken — quoted in USD per coin, 1-contract = 1 unit)
    "BTC": 1.0,
    "ETH": 1.0,
    "SOL": 1.0,
    "XRP": 1.0,
    "ADA": 1.0,
    "DOT": 1.0,
    "LINK": 1.0,
    "AVAX": 1.0,
    "MATIC": 1.0,
    "ATOM": 1.0,
}

_DEFAULT_POINT_VALUE = 1.0


def _point_value(symbol: str) -> float:
    """Return the dollar value per point for a symbol.

    Strips exchange prefix (e.g. ``KRAKEN:BTC`` → ``BTC``) and CME
    contract month suffixes (e.g. ``MESH5`` → ``MES``).
    """
    sym = symbol.upper().strip()
    # Strip exchange prefix
    if ":" in sym:
        sym = sym.split(":")[-1]
    # Strip trailing month/year codes (letters + digits at end, e.g. H5, Z25)
    import re

    base = re.sub(r"[A-Z]\d{1,2}$", "", sym)
    return _POINT_VALUES.get(base, _POINT_VALUES.get(sym, _DEFAULT_POINT_VALUE))


# ---------------------------------------------------------------------------
# Fill normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_side(buy_sell: str) -> str:
    """Normalise Rithmic buy_sell_type to 'BUY' or 'SELL'."""
    val = str(buy_sell).upper().strip()
    if val in ("BUY", "B", "BUY_TO_OPEN", "BUY_TO_CLOSE", "1"):
        return "BUY"
    if val in ("SELL", "S", "SELL_TO_OPEN", "SELL_TO_CLOSE", "2"):
        return "SELL"
    # async_rithmic TransactionType enum names
    if "BUY" in val:
        return "BUY"
    if "SELL" in val:
        return "SELL"
    return val  # pass through unknown values


def _is_filled(status: str) -> bool:
    """Return True when the order status indicates a completed fill."""
    val = str(status).upper().strip()
    return val in ("FILLED", "FILL", "COMPLETE", "COMPLETED", "PARTIAL_FILL", "PARTIAL")


# ---------------------------------------------------------------------------
# Round-trip matching
# ---------------------------------------------------------------------------


def match_fills_to_trades(fills: list[dict]) -> list[dict]:
    """Convert a flat list of individual fills into round-trip trade dicts.

    Algorithm
    ---------
    1. Filter to filled orders only.
    2. Group fills by symbol (and optionally account_key).
    3. Within each group, sort by fill_time ascending.
    4. Walk fills in order; maintain a per-symbol stack of open legs.
       - BUY fill → push onto the open-long stack.
       - SELL fill → pop the oldest open-long fill and pair as a
         completed short-exit (LONG trade closed).
       - SELL fill with no open longs → push onto the open-short stack
         (short entry).
       - BUY fill with open shorts → pop oldest open-short and pair as
         a completed long-exit (SHORT trade closed).
    5. Any unpaired fills at the end are returned as OPEN trades (entry
       only, no exit yet).

    Returns a list of normalised trade dicts ready for ``upsert_trade_from_fill``.
    """
    # Only process actually filled orders
    filled = [f for f in fills if _is_filled(f.get("status", ""))]
    if not filled:
        return []

    # Group by (account_key, symbol)
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in filled:
        key = (str(f.get("account_key", "")), str(f.get("symbol", "")))
        groups.setdefault(key, []).append(f)

    trades: list[dict] = []

    for (account_key, symbol), group in groups.items():
        # Sort by fill_time ascending (handle missing / malformed times)
        def _sort_key(f: dict) -> str:
            t = str(f.get("fill_time", ""))
            return t if t else "0000-00-00 00:00:00"

        sorted_fills = sorted(group, key=_sort_key)

        open_longs: list[dict] = []  # BUY fills awaiting a closing SELL
        open_shorts: list[dict] = []  # SELL fills awaiting a closing BUY

        for fill in sorted_fills:
            side = _normalise_side(fill.get("buy_sell", ""))
            qty = int(fill.get("qty", 1) or 1)
            price = float(fill.get("fill_price", 0.0) or 0.0)
            fill_time = str(fill.get("fill_time", ""))
            commission = float(fill.get("commission", 0.0) or 0.0)

            if side == "BUY":
                if open_shorts:
                    # Close a short position
                    entry_fill = open_shorts.pop(0)
                    entry_price = float(entry_fill.get("fill_price", 0.0) or 0.0)
                    entry_qty = int(entry_fill.get("qty", 1) or 1)
                    entry_comm = float(entry_fill.get("commission", 0.0) or 0.0)
                    pv = _point_value(symbol)
                    gross_pnl = (entry_price - price) * min(qty, entry_qty) * pv
                    total_comm = commission + entry_comm
                    trades.append(
                        _build_trade_dict(
                            account_key=account_key,
                            symbol=symbol,
                            direction="SHORT",
                            entry_price=entry_price,
                            close_price=price,
                            contracts=min(qty, entry_qty),
                            gross_pnl=gross_pnl,
                            net_pnl=gross_pnl - total_comm,
                            entry_time=str(entry_fill.get("fill_time", "")),
                            close_time=fill_time,
                            commission=total_comm,
                        )
                    )
                    # If more qty remains, push remainder back as open short
                    remainder = entry_qty - qty
                    if remainder > 0:
                        leftover = dict(entry_fill)
                        leftover["qty"] = remainder
                        open_shorts.insert(0, leftover)
                else:
                    # Open a new long position
                    open_longs.append(fill)

            elif side == "SELL":
                if open_longs:
                    # Close a long position
                    entry_fill = open_longs.pop(0)
                    entry_price = float(entry_fill.get("fill_price", 0.0) or 0.0)
                    entry_qty = int(entry_fill.get("qty", 1) or 1)
                    entry_comm = float(entry_fill.get("commission", 0.0) or 0.0)
                    pv = _point_value(symbol)
                    gross_pnl = (price - entry_price) * min(qty, entry_qty) * pv
                    total_comm = commission + entry_comm
                    trades.append(
                        _build_trade_dict(
                            account_key=account_key,
                            symbol=symbol,
                            direction="LONG",
                            entry_price=entry_price,
                            close_price=price,
                            contracts=min(qty, entry_qty),
                            gross_pnl=gross_pnl,
                            net_pnl=gross_pnl - total_comm,
                            entry_time=str(entry_fill.get("fill_time", "")),
                            close_time=fill_time,
                            commission=total_comm,
                        )
                    )
                    remainder = entry_qty - qty
                    if remainder > 0:
                        leftover = dict(entry_fill)
                        leftover["qty"] = remainder
                        open_longs.insert(0, leftover)
                else:
                    # Open a new short position
                    open_shorts.append(fill)

        # Remaining unpaired fills → OPEN trades (entry without exit)
        for fill in open_longs:
            trades.append(
                _build_trade_dict(
                    account_key=account_key,
                    symbol=symbol,
                    direction="LONG",
                    entry_price=float(fill.get("fill_price", 0.0) or 0.0),
                    close_price=None,
                    contracts=int(fill.get("qty", 1) or 1),
                    gross_pnl=None,
                    net_pnl=None,
                    entry_time=str(fill.get("fill_time", "")),
                    close_time=None,
                    commission=float(fill.get("commission", 0.0) or 0.0),
                )
            )
        for fill in open_shorts:
            trades.append(
                _build_trade_dict(
                    account_key=account_key,
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=float(fill.get("fill_price", 0.0) or 0.0),
                    close_price=None,
                    contracts=int(fill.get("qty", 1) or 1),
                    gross_pnl=None,
                    net_pnl=None,
                    entry_time=str(fill.get("fill_time", "")),
                    close_time=None,
                    commission=float(fill.get("commission", 0.0) or 0.0),
                )
            )

    return trades


def _build_trade_dict(
    *,
    account_key: str,
    symbol: str,
    direction: str,
    entry_price: float,
    close_price: float | None,
    contracts: int,
    gross_pnl: float | None,
    net_pnl: float | None,
    entry_time: str,
    close_time: str | None,
    commission: float,
) -> dict[str, Any]:
    """Build a normalised trade dict for upsert_trade_from_fill."""
    return {
        "account_key": account_key,
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "close_price": close_price,
        "contracts": contracts,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "pnl": net_pnl,  # alias used by upsert_trade_from_fill
        "entry_time": entry_time,
        "close_time": close_time,
        "fill_time": entry_time,
        "commission": commission,
    }


# ---------------------------------------------------------------------------
# Hold-time helper
# ---------------------------------------------------------------------------


def _hold_minutes(entry_time: str, close_time: str) -> float | None:
    """Return hold time in minutes between two timestamp strings, or None."""
    if not entry_time or not close_time:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            t0 = datetime.strptime(entry_time.strip()[:19], fmt)
            t1 = datetime.strptime(close_time.strip()[:19], fmt)
            return max(0.0, (t1 - t0).total_seconds() / 60.0)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------


def _write_trades_to_db(trades: list[dict]) -> tuple[int, int]:
    """Upsert a list of matched-trade dicts into trades_v2.

    Returns (inserted_count, updated_count) — approximate, based on
    whether a record already existed for the same date+symbol+account.
    """
    from lib.core.models import upsert_trade_from_fill

    inserted = 0
    updated = 0

    for t in trades:
        try:
            trade_id = upsert_trade_from_fill(
                account_key=t["account_key"],
                symbol=t["symbol"],
                direction=t["direction"],
                entry_price=float(t["entry_price"] or 0.0),
                close_price=t.get("close_price"),
                contracts=int(t.get("contracts") or 1),
                pnl=t.get("net_pnl"),
                fill_time=str(t.get("fill_time") or t.get("entry_time") or ""),
                strategy="rithmic_sync",
                notes=f"rithmic_sync:{t['account_key']}",
                source="rithmic_sync",
            )
            if trade_id > 0:
                inserted += 1
        except Exception as exc:
            logger.warning(
                "journal_sync: failed to upsert trade %s %s: %s",
                t.get("symbol"),
                t.get("direction"),
                exc,
            )

    return inserted, updated


# ---------------------------------------------------------------------------
# Daily journal summary updater
# ---------------------------------------------------------------------------


def _refresh_daily_journal_summary(account_key: str | None = None) -> None:
    """Recompute today's daily_journal row from trades_v2 closed trades.

    Aggregates all today's rithmic_sync closed trades (optionally filtered
    by account_key) and writes a summary row to daily_journal via
    ``save_daily_journal()``.

    This is a best-effort operation — failures are logged but not re-raised.
    """
    try:
        from lib.core.models import _get_conn, save_daily_journal

        today = datetime.now(tz=_EST).strftime("%Y-%m-%d")
        conn = _get_conn()

        conditions = ["source = 'rithmic_sync'", "status = 'CLOSED'", "created_at LIKE ?"]
        params: list[Any] = [f"{today}%"]

        if account_key:
            conditions.append("notes LIKE ?")
            params.append(f"%{account_key}%")

        where = " AND ".join(conditions)
        sql = f"""
            SELECT
                SUM(COALESCE(pnl, 0))       AS net_pnl,
                SUM(COALESCE(contracts, 0)) AS num_contracts,
                GROUP_CONCAT(DISTINCT asset) AS instruments
            FROM trades_v2
            WHERE {where}
        """
        row = conn.execute(sql, tuple(params)).fetchone()
        conn.close()

        if row is None:
            return

        if hasattr(row, "keys"):
            d = {k: row[k] for k in row}
        else:
            d = {"net_pnl": row[0], "num_contracts": row[1], "instruments": row[2]}

        net_pnl = float(d.get("net_pnl") or 0.0)
        num_contracts = int(d.get("num_contracts") or 0)
        instruments = str(d.get("instruments") or "")

        if net_pnl == 0.0 and num_contracts == 0:
            # Nothing to summarise yet
            return

        save_daily_journal(
            trade_date=today,
            account_size=150_000,
            gross_pnl=net_pnl,  # best estimate without separate commission tracking
            net_pnl=net_pnl,
            commissions=0.0,
            num_contracts=num_contracts,
            instruments=instruments,
            notes=f"auto-synced from rithmic fills ({account_key or 'all accounts'})",
        )
        logger.info(
            "journal_sync: daily_journal updated — date=%s net_pnl=%.2f contracts=%d",
            today,
            net_pnl,
            num_contracts,
        )
    except Exception as exc:
        logger.warning("journal_sync: daily_journal refresh failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------


async def run_journal_sync(account_key: str | None = None) -> dict[str, Any]:
    """Pull today's fills from Rithmic and upsert matched trades into trades_v2.

    Parameters
    ----------
    account_key : str | None
        When provided, sync only this account.  When ``None`` (default),
        sync all enabled accounts via ``get_all_today_fills()``.

    Returns a status dict with counts and any errors encountered.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "fills_retrieved": 0,
        "trades_matched": 0,
        "trades_written": 0,
        "errors": [],
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }

    # 1. Retrieve fills from Rithmic -----------------------------------------
    fills: list[dict] = []
    try:
        from lib.integrations.rithmic_client import get_manager

        manager = get_manager()

        if account_key:
            fills = await manager.get_today_fills(account_key)
        else:
            fills = await manager.get_all_today_fills()

        result["fills_retrieved"] = len(fills)
        logger.info(
            "journal_sync: retrieved %d fill(s) for %s",
            len(fills),
            account_key or "all accounts",
        )
    except Exception as exc:
        msg = f"fill retrieval error: {exc}"
        logger.error("journal_sync: %s", msg, exc_info=True)
        result["errors"].append(msg)
        result["status"] = "error"
        return result

    if not fills:
        logger.info("journal_sync: no fills to process — nothing to sync")
        return result

    # 2. Match fills → round-trip trades ------------------------------------
    try:
        trades = match_fills_to_trades(fills)
        result["trades_matched"] = len(trades)
        logger.info(
            "journal_sync: matched %d fill(s) → %d trade(s)",
            len(fills),
            len(trades),
        )
    except Exception as exc:
        msg = f"fill matching error: {exc}"
        logger.error("journal_sync: %s", msg, exc_info=True)
        result["errors"].append(msg)
        result["status"] = "error"
        return result

    if not trades:
        logger.info("journal_sync: no complete round-trips found yet")
        return result

    # 3. Write to DB -----------------------------------------------------------
    try:
        written, _ = _write_trades_to_db(trades)
        result["trades_written"] = written
        logger.info("journal_sync: wrote %d trade record(s) to trades_v2", written)
    except Exception as exc:
        msg = f"db write error: {exc}"
        logger.error("journal_sync: %s", msg, exc_info=True)
        result["errors"].append(msg)
        result["status"] = "partial"

    # 4. Update daily journal summary ----------------------------------------
    with contextlib.suppress(Exception):
        _refresh_daily_journal_summary(account_key=account_key)

    # 5. Publish sync status to Redis (non-fatal) -----------------------------
    _publish_sync_status(result)

    return result


def _publish_sync_status(result: dict[str, Any]) -> None:
    """Publish the sync result to Redis for dashboard visibility."""
    with contextlib.suppress(Exception):
        import json

        from lib.core.cache import cache_set

        cache_set(
            "journal:last_sync",
            json.dumps(result, default=str).encode(),
            ttl=86400,  # 24 h
        )


# ---------------------------------------------------------------------------
# Sync status reader (used by the API endpoint)
# ---------------------------------------------------------------------------


def get_last_sync_status() -> dict[str, Any] | None:
    """Return the most recent sync result from Redis, or None."""
    with contextlib.suppress(Exception):
        import json

        from lib.core.cache import cache_get

        raw = cache_get("journal:last_sync")
        if raw:
            return json.loads(raw)
    return None


# ---------------------------------------------------------------------------
# Convenience: run sync synchronously (for non-async callers)
# ---------------------------------------------------------------------------


def run_journal_sync_sync(account_key: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper around ``run_journal_sync`` for use in the engine
    scheduler loop (which is synchronous).

    Creates a new event loop for the async call rather than using
    ``asyncio.run()`` to avoid issues when called from within an existing
    event loop.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(run_journal_sync(account_key=account_key))
        finally:
            loop.close()
    except Exception as exc:
        logger.error("journal_sync: sync wrapper error: %s", exc, exc_info=True)
        return {
            "status": "error",
            "fills_retrieved": 0,
            "trades_matched": 0,
            "trades_written": 0,
            "errors": [str(exc)],
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
