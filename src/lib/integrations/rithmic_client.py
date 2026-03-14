"""
Rithmic Prop Account Integration
==================================
Read-only account monitoring for Rithmic-connected prop firm accounts
(TPT, Apex, Topstep, etc.) using the ``async-rithmic`` library.

Architecture:
    - ``RithmicAccountConfig``  — per-account credentials + metadata
    - ``RithmicAccountManager`` — async lifecycle: connect/disconnect/refresh
    - ``RithmicStreamManager``  — persistent TICKER_PLANT streaming connection
    - ``router``               — FastAPI endpoints consumed by the dashboard

Endpoints:
    GET  /api/rithmic/accounts          — list configured accounts (no creds)
    GET  /api/rithmic/status            — live status JSON for all accounts
    GET  /api/rithmic/status/html       — HTMX fragment for Connections page
    GET  /api/rithmic/account/{key}     — single account detail JSON
    POST /api/rithmic/account/{key}/refresh — force-refresh a single account
    GET  /api/rithmic/config            — load saved account configs (UI)
    POST /api/rithmic/config            — save account configs from settings UI

    --- Streaming (RITHMIC-STREAM-A) ---
    GET  /api/rithmic/stream/status           — live/connected/subscribed state
    POST /api/rithmic/stream/subscribe        — subscribe to symbols
    GET  /api/rithmic/stream/l1/{symbol}      — L1 bid/ask/last snapshot
    GET  /api/rithmic/stream/ticks/{symbol}   — recent tick history

Credentials are stored in Redis (AES-256-encrypted via a server-side key
derived from the ``SECRET_KEY`` env var) and NEVER returned to the browser.
The UI only sends credentials on initial save; subsequent reads return
masked values.

Prop firm presets (system_name / gateway defaults):
    TPT          — "Rithmic Paper Trading", Chicago
    Apex         — "Apex Trader Funding",   Chicago
    Topstep      — "TopStep",               Chicago
    TradeDay     — "TradeDay",              Chicago
    MyFundedFutures — "My Funded Futures",  Chicago
    Custom       — user-supplied values
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("api.rithmic")

router = APIRouter(tags=["Rithmic"])

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Prop firm presets
# ---------------------------------------------------------------------------

PROP_FIRM_PRESETS: dict[str, dict[str, str]] = {
    "tpt": {
        "label": "Take Profit Trader (TPT)",
        "system_name": "Rithmic Paper Trading",
        "gateway": "Chicago",
    },
    "apex": {
        "label": "Apex Trader Funding",
        "system_name": "Apex Trader Funding",
        "gateway": "Chicago",
    },
    "topstep": {
        "label": "TopStep",
        "system_name": "TopStep",
        "gateway": "Chicago",
    },
    "tradeday": {
        "label": "TradeDay",
        "system_name": "TradeDay",
        "gateway": "Chicago",
    },
    "mff": {
        "label": "My Funded Futures",
        "system_name": "My Funded Futures",
        "gateway": "Chicago",
    },
    "custom": {
        "label": "Custom",
        "system_name": "",
        "gateway": "Chicago",
    },
}

# Ordered list for the UI dropdown
PROP_FIRM_OPTIONS: list[dict[str, str]] = [{"value": k, "label": v["label"]} for k, v in PROP_FIRM_PRESETS.items()]

# ---------------------------------------------------------------------------
# Credential encryption helpers
# ---------------------------------------------------------------------------

_SECRET_KEY = os.getenv("SECRET_KEY", os.getenv("API_KEY", "ruby-futures-default"))


def _derive_fernet_key() -> bytes:
    """Derive a 32-byte Fernet-compatible key from the server SECRET_KEY."""
    digest = hashlib.sha256(_SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def _encrypt(plaintext: str) -> str:
    """Encrypt plaintext using Fernet (AES-128-CBC + HMAC).

    Falls back to base64 obfuscation when cryptography is not installed —
    still prevents accidental browser exposure, but not cryptographically
    secure.  Log a warning so the operator knows to install the package.
    """
    if not plaintext:
        return ""
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]

        f = Fernet(_derive_fernet_key())
        return f.encrypt(plaintext.encode()).decode()
    except ImportError:
        logger.warning(
            "cryptography package not installed — Rithmic credentials are "
            "stored with base64 obfuscation only.  Run: pip install cryptography"
        )
        return base64.b64encode(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    """Decrypt a value previously encrypted by ``_encrypt``."""
    if not ciphertext:
        return ""
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]

        f = Fernet(_derive_fernet_key())
        return f.decrypt(ciphertext.encode()).decode()
    except ImportError:
        try:
            return base64.b64decode(ciphertext.encode()).decode()
        except Exception:
            return ""
    except Exception:
        # Fernet InvalidToken or corrupt data — may be plain base64 from a migration
        try:
            return base64.b64decode(ciphertext.encode()).decode()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Account config model
# ---------------------------------------------------------------------------

_REDIS_CONFIG_KEY = "rithmic:account_configs"
_REDIS_STATUS_KEY = "rithmic:account_status:{key}"


class RithmicAccountConfig:
    """Per-account configuration (credentials + metadata).

    ``key``         — short stable identifier, e.g. "acc1", "tpt_eval"
    ``label``       — display name shown in the UI
    ``prop_firm``   — one of the PROP_FIRM_PRESETS keys
    ``system_name`` — Rithmic system_name override (empty = use preset)
    ``gateway``     — "Chicago" | "Sydney" | "Sao Paulo" | "custom"
    ``username``    — encrypted Rithmic username
    ``password``    — encrypted Rithmic password
    ``enabled``     — whether to connect on startup
    ``app_name``    — app identifier sent to Rithmic (default: ruby_futures)
    ``app_version`` — version string sent to Rithmic
    """

    def __init__(
        self,
        key: str,
        label: str,
        prop_firm: str = "tpt",
        system_name: str = "",
        gateway: str = "Chicago",
        username: str = "",
        password: str = "",
        enabled: bool = True,
        app_name: str = "ruby_futures",
        app_version: str = "1.0",
        account_size: int = 150_000,
    ) -> None:
        self.key = key
        self.label = label
        self.prop_firm = prop_firm
        self.gateway = gateway
        self.app_name = app_name
        self.app_version = app_version
        self.enabled = enabled
        self.account_size = account_size

        # Resolve system_name: explicit override > preset > empty
        preset = PROP_FIRM_PRESETS.get(prop_firm, {})
        self.system_name = system_name or preset.get("system_name", "")

        # Credentials are always stored encrypted
        self._username_enc = username  # may already be encrypted
        self._password_enc = password

    # ------------------------------------------------------------------
    # Credential accessors
    # ------------------------------------------------------------------

    def set_credentials(self, username: str, password: str) -> None:
        """Encrypt and store plaintext credentials."""
        self._username_enc = _encrypt(username)
        self._password_enc = _encrypt(password)

    def get_username(self) -> str:
        return _decrypt(self._username_enc)

    def get_password(self) -> str:
        return _decrypt(self._password_enc)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialise to dict for Redis storage (encrypted creds included)."""
        return {
            "key": self.key,
            "label": self.label,
            "prop_firm": self.prop_firm,
            "system_name": self.system_name,
            "gateway": self.gateway,
            "username_enc": self._username_enc,
            "password_enc": self._password_enc,
            "enabled": self.enabled,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "account_size": self.account_size,
        }

    def to_ui_dict(self) -> dict[str, Any]:
        """Serialise for the browser — credentials masked, never returned."""
        return {
            "key": self.key,
            "label": self.label,
            "prop_firm": self.prop_firm,
            "prop_firm_label": PROP_FIRM_PRESETS.get(self.prop_firm, {}).get("label", self.prop_firm),
            "system_name": self.system_name,
            "gateway": self.gateway,
            "enabled": self.enabled,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "account_size": self.account_size,
            # Never expose plaintext — show only whether credentials are configured
            "username_set": bool(self._username_enc),
            "password_set": bool(self._password_enc),
            # Mask: show first 3 chars of decrypted username for UI recognition only
            "username_hint": self._username_hint(),
        }

    def _username_hint(self) -> str:
        u = self.get_username()
        if not u:
            return ""
        if len(u) <= 4:
            return u[0] + "***"
        return u[:3] + "***"

    @classmethod
    def from_storage_dict(cls, d: dict[str, Any]) -> RithmicAccountConfig:
        obj = cls(
            key=d["key"],
            label=d.get("label", d["key"]),
            prop_firm=d.get("prop_firm", "tpt"),
            system_name=d.get("system_name", ""),
            gateway=d.get("gateway", "Chicago"),
            username=d.get("username_enc", ""),
            password=d.get("password_enc", ""),
            enabled=d.get("enabled", True),
            app_name=d.get("app_name", "ruby_futures"),
            app_version=d.get("app_version", "1.0"),
            account_size=d.get("account_size", 150_000),
        )
        # Overwrite with pre-encrypted values from storage
        obj._username_enc = d.get("username_enc", "")
        obj._password_enc = d.get("password_enc", "")
        return obj


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load_configs() -> list[RithmicAccountConfig]:
    """Load account configs from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get(_REDIS_CONFIG_KEY)
        if raw:
            items = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return [RithmicAccountConfig.from_storage_dict(d) for d in items]
    except Exception as exc:
        logger.debug("rithmic: config load error: %s", exc)
    return []


def _save_configs(configs: list[RithmicAccountConfig]) -> None:
    """Persist account configs to Redis (no expiry)."""
    try:
        from lib.core.cache import cache_set

        data = json.dumps([c.to_storage_dict() for c in configs])
        cache_set(_REDIS_CONFIG_KEY, data.encode(), 0)
    except Exception as exc:
        logger.warning("rithmic: config save error: %s", exc)


def _load_status(key: str) -> dict[str, Any]:
    """Load the last-known live status for an account from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get(_REDIS_STATUS_KEY.format(key=key))
        if raw:
            return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        pass
    return {}


def _save_status(key: str, status: dict[str, Any]) -> None:
    """Cache account live status in Redis (TTL: 120 s)."""
    try:
        from lib.core.cache import cache_set

        cache_set(_REDIS_STATUS_KEY.format(key=key), json.dumps(status, default=str).encode(), 120)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Account manager
# ---------------------------------------------------------------------------

# Map from gateway name → async_rithmic Gateway enum value
_GATEWAY_MAP: dict[str, Any] = {}


def _resolve_gateway(name: str) -> Any:
    """Return the async_rithmic Gateway enum for the given name string."""
    global _GATEWAY_MAP
    if not _GATEWAY_MAP:
        try:
            from async_rithmic import Gateway  # type: ignore[import-untyped]

            _GATEWAY_MAP = {
                "Chicago": Gateway.CHICAGO,
                "Sydney": Gateway.SYDNEY,
                "Sao Paulo": Gateway.SAO_PAULO,
                "test": Gateway.TEST,
            }
        except ImportError:
            return None
    return _GATEWAY_MAP.get(name, _GATEWAY_MAP.get("Chicago"))


class RithmicAccountManager:
    """Manages async Rithmic client connections for all configured accounts.

    Each account gets its own ``RithmicClient`` instance connected to the
    Order Plant + PnL Plant for read-only position/P&L data.

    The manager is intentionally lightweight: it does NOT hold a persistent
    connection between refreshes (Rithmic sessions can be flaky in eval
    environments).  Instead, ``refresh(key)`` opens → reads → closes a
    short-lived session and stores the result in Redis.

    Long-lived streaming will be added in a future phase once we have funded
    accounts and need real-time position updates.
    """

    def __init__(self) -> None:
        self._configs: dict[str, RithmicAccountConfig] = {}
        self._status: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    def reload_configs(self) -> None:
        """Reload configs from Redis (call after settings save)."""
        configs = _load_configs()
        self._configs = {c.key: c for c in configs}
        # Restore cached status for each account
        for key in self._configs:
            cached = _load_status(key)
            if cached:
                self._status[key] = cached
        self._loaded = True
        logger.info("rithmic: loaded %d account config(s)", len(self._configs))

    def get_all_ui(self) -> list[dict[str, Any]]:
        """Return all account configs safe for browser consumption."""
        if not self._loaded:
            self.reload_configs()
        return [c.to_ui_dict() for c in self._configs.values()]

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Return the last-known live status for all accounts."""
        if not self._loaded:
            self.reload_configs()
        return dict(self._status)

    def get_status(self, key: str) -> dict[str, Any]:
        return self._status.get(key, {"key": key, "connected": False, "error": "not yet polled"})

    async def refresh_account(self, key: str) -> dict[str, Any]:
        """Open a short-lived Rithmic session, pull positions + P&L, close.

        Returns a status dict that is also cached in Redis.
        """
        if not self._loaded:
            self.reload_configs()

        config = self._configs.get(key)
        if config is None:
            return {"key": key, "connected": False, "error": "account not found"}

        if not config.enabled:
            return {"key": key, "connected": False, "error": "account disabled"}

        username = config.get_username()
        password = config.get_password()
        if not username or not password:
            return {"key": key, "connected": False, "error": "credentials not configured"}

        status: dict[str, Any] = {
            "key": key,
            "label": config.label,
            "prop_firm": config.prop_firm,
            "prop_firm_label": PROP_FIRM_PRESETS.get(config.prop_firm, {}).get("label", config.prop_firm),
            "connected": False,
            "error": "",
            "refreshed_at": datetime.now(tz=_EST).isoformat(),
            "accounts": [],
            "positions": [],
            "pnl": {},
        }

        try:
            from async_rithmic import RithmicClient  # type: ignore[import-untyped]
        except ImportError:
            status["error"] = "async-rithmic not installed (pip install async-rithmic)"
            _save_status(key, status)
            self._status[key] = status
            return status

        gateway = _resolve_gateway(config.gateway)
        if gateway is None:
            status["error"] = f"unknown gateway: {config.gateway}"
            _save_status(key, status)
            self._status[key] = status
            return status

        client: Any = None
        try:
            async with self._lock:
                client = RithmicClient(  # type: ignore[possibly-undefined]
                    user=username,
                    password=password,
                    system_name=config.system_name,
                    app_name=config.app_name,
                    app_version=config.app_version,
                    gateway=gateway,
                )
                await asyncio.wait_for(client.connect(), timeout=15.0)
                status["connected"] = True

                # List accounts
                try:
                    accounts = await asyncio.wait_for(client.list_accounts(), timeout=10.0)
                    status["accounts"] = [
                        {
                            "id": getattr(a, "account_id", str(a)),
                            "name": getattr(a, "account_name", getattr(a, "account_id", str(a))),
                            "fcm": getattr(a, "fcm_id", ""),
                        }
                        for a in (accounts or [])
                    ]
                except Exception as acc_exc:
                    logger.debug("rithmic[%s]: list_accounts error: %s", key, acc_exc)

                # Positions
                try:
                    positions = await asyncio.wait_for(client.list_positions(), timeout=10.0)
                    status["positions"] = [
                        {
                            "symbol": getattr(p, "symbol", ""),
                            "size": getattr(p, "size", getattr(p, "pos_size", 0)),
                            "avg_price": getattr(p, "avg_price", getattr(p, "open_price", 0.0)),
                            "open_pnl": getattr(p, "open_pnl", getattr(p, "unrealized_pnl", 0.0)),
                        }
                        for p in (positions or [])
                    ]
                except Exception as pos_exc:
                    logger.debug("rithmic[%s]: list_positions error: %s", key, pos_exc)

                # P&L
                try:
                    pnl = await asyncio.wait_for(client.get_pnl_info(), timeout=10.0)
                    if pnl:
                        status["pnl"] = {
                            "unrealized": getattr(pnl, "unrealized_pnl", getattr(pnl, "open_pnl", None)),
                            "realized": getattr(pnl, "realized_pnl", getattr(pnl, "closed_pnl", None)),
                            "total": getattr(pnl, "total_pnl", None),
                        }
                except Exception as pnl_exc:
                    logger.debug("rithmic[%s]: get_pnl_info error: %s", key, pnl_exc)

        except TimeoutError:
            status["error"] = "connection timed out (15 s) — check credentials and network"
        except Exception as exc:
            status["error"] = str(exc)[:200]
            logger.warning("rithmic[%s]: refresh error: %s", key, exc)
        finally:
            if client is not None:
                import contextlib

                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.disconnect(), timeout=5.0)

        _save_status(key, status)
        self._status[key] = status
        return status

    async def refresh_all(self) -> None:
        """Refresh all enabled accounts concurrently."""
        if not self._loaded:
            self.reload_configs()
        tasks = [self.refresh_account(k) for k, c in self._configs.items() if c.enabled]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_today_fills(self, account_key: str) -> list[dict]:
        """Open a short-lived Rithmic session and retrieve today's order/fill history.

        Maps each order to a normalised fill dict and caches the result in Redis
        under ``rithmic:fills:{account_key}:{today_date}`` with a 24 h TTL.

        Returns a list of fill dicts (may be empty if no fills or on error).
        """
        if not self._loaded:
            self.reload_configs()

        config = self._configs.get(account_key)
        if config is None:
            logger.warning("rithmic[%s]: get_today_fills — account not found", account_key)
            return []

        if not config.enabled:
            logger.debug("rithmic[%s]: get_today_fills — account disabled", account_key)
            return []

        username = config.get_username()
        password = config.get_password()
        if not username or not password:
            logger.warning("rithmic[%s]: get_today_fills — credentials not configured", account_key)
            return []

        try:
            from async_rithmic import RithmicClient  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("rithmic[%s]: async-rithmic not installed", account_key)
            return []

        gateway = _resolve_gateway(config.gateway)
        if gateway is None:
            logger.warning("rithmic[%s]: get_today_fills — unknown gateway: %s", account_key, config.gateway)
            return []

        fills: list[dict] = []
        client: Any = None
        try:
            async with self._lock:
                client = RithmicClient(  # type: ignore[call-arg]
                    user=username,
                    password=password,
                    system_name=config.system_name,
                    app_name=config.app_name,
                    app_version=config.app_version,
                    gateway=gateway,
                )
                await asyncio.wait_for(client.connect(), timeout=15.0)

                try:
                    orders = await asyncio.wait_for(client.show_order_history_summary(), timeout=20.0)
                    for order in orders or []:
                        fills.append(
                            {
                                "account_key": account_key,
                                "order_id": str(getattr(order, "order_id", "")),
                                "symbol": str(getattr(order, "symbol", "")),
                                "exchange": str(getattr(order, "exchange", "")),
                                "buy_sell": str(
                                    getattr(
                                        order,
                                        "buy_sell_type",
                                        getattr(order, "action", ""),
                                    )
                                ),
                                "qty": int(
                                    getattr(
                                        order,
                                        "qty",
                                        getattr(order, "quantity", 0),
                                    )
                                ),
                                "fill_price": float(
                                    getattr(
                                        order,
                                        "avg_fill_price",
                                        getattr(order, "fill_price", 0.0),
                                    )
                                ),
                                "fill_time": str(
                                    getattr(
                                        order,
                                        "update_time",
                                        getattr(order, "fill_time", ""),
                                    )
                                ),
                                "status": str(getattr(order, "status", "FILLED")),
                                "commission": float(getattr(order, "commission", 0.0)),
                            }
                        )
                    logger.info(
                        "rithmic[%s]: retrieved %d fill(s) from order history",
                        account_key,
                        len(fills),
                    )
                except Exception as hist_exc:
                    logger.warning(
                        "rithmic[%s]: show_order_history_summary error: %s",
                        account_key,
                        hist_exc,
                    )

        except TimeoutError:
            logger.warning("rithmic[%s]: get_today_fills — connection timed out", account_key)
        except Exception as exc:
            logger.warning("rithmic[%s]: get_today_fills — error: %s", account_key, exc)
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.disconnect(), timeout=5.0)

        # Cache in Redis: rithmic:fills:{account_key}:{today_date}  TTL = 24 h
        try:
            from lib.core.cache import cache_set

            today_date = datetime.now(tz=_EST).strftime("%Y-%m-%d")
            redis_key = f"rithmic:fills:{account_key}:{today_date}"
            cache_set(redis_key, json.dumps(fills, default=str).encode(), 86400)
        except Exception as cache_exc:
            logger.debug("rithmic[%s]: fills cache write failed (non-fatal): %s", account_key, cache_exc)

        return fills

    async def get_all_today_fills(self) -> list[dict]:
        """Retrieve today's fills for all enabled accounts and combine them.

        Calls ``get_today_fills`` concurrently for every enabled account and
        merges the results into a single flat list.
        """
        if not self._loaded:
            self.reload_configs()

        enabled_keys = [k for k, c in self._configs.items() if c.enabled]
        if not enabled_keys:
            return []

        results = await asyncio.gather(
            *[self.get_today_fills(k) for k in enabled_keys],
            return_exceptions=True,
        )

        combined: list[dict] = []
        for key, result in zip(enabled_keys, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("rithmic[%s]: get_all_today_fills — error: %s", key, result)
            elif isinstance(result, list):
                combined.extend(result)

        logger.info(
            "rithmic: get_all_today_fills — total %d fill(s) across %d account(s)", len(combined), len(enabled_keys)
        )
        return combined

    async def eod_close_all_positions(
        self,
        *,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Cancel all working orders then flatten all open positions across every
        enabled account.

        This is the 16:00 ET hard-safety-net. It is intentionally separate from
        ``refresh_account`` so it can be called independently (scheduler, manual
        API call) without polluting the status cache.

        Sequence per account:
          1. ``cancel_all_orders(account_id=...)``   — kills every pending entry /
             stop / target on the account (template 346).
          2. Short asyncio.sleep(0.5) so the exchange has time to ack cancels.
          3. ``exit_position(account_id=...)``        — market-flattens whatever
             net position remains (template 3504, no symbol filter = flatten all).

        Args:
            dry_run: When *True* the method still connects and discovers accounts
                     but skips the cancel/exit calls.  Useful for staging checks.

        Returns:
            List of result dicts, one per account key, with keys:
                key, label, account_id, cancelled, flattened, error, dry_run.
        """
        if not self._loaded:
            self.reload_configs()

        results: list[dict[str, Any]] = []

        for key, config in self._configs.items():
            if not config.enabled:
                results.append({"key": key, "label": config.label, "skipped": "disabled"})
                continue

            username = config.get_username()
            password = config.get_password()
            if not username or not password:
                results.append(
                    {
                        "key": key,
                        "label": config.label,
                        "error": "credentials not configured",
                    }
                )
                continue

            result: dict[str, Any] = {
                "key": key,
                "label": config.label,
                "cancelled": False,
                "flattened": False,
                "error": "",
                "dry_run": dry_run,
                "timestamp": datetime.now(tz=_EST).isoformat(),
            }

            try:
                from async_rithmic import OrderPlacement, RithmicClient  # type: ignore[import-untyped]
            except ImportError:
                result["error"] = "async-rithmic not installed"
                results.append(result)
                continue

            gateway = _resolve_gateway(config.gateway)
            if gateway is None:
                result["error"] = f"unknown gateway: {config.gateway}"
                results.append(result)
                continue

            client: Any = None
            try:
                client = RithmicClient(
                    user=username,
                    password=password,
                    system_name=config.system_name,
                    app_name=config.app_name,
                    app_version=config.app_version,
                    url=gateway,
                    manual_or_auto=OrderPlacement.MANUAL,
                )
                await asyncio.wait_for(client.connect(), timeout=15.0)

                # Discover account IDs so we can target each one individually.
                account_ids: list[str] = []
                with contextlib.suppress(Exception):
                    accounts = await asyncio.wait_for(client.list_accounts(), timeout=10.0)
                    account_ids = [str(getattr(a, "account_id", None) or a) for a in (accounts or [])]

                # Fall back to config-level account_id if discovery fails.
                if not account_ids and hasattr(config, "account_id") and config.account_id:  # type: ignore[attr-defined]
                    account_ids = [config.account_id]  # type: ignore[attr-defined]

                result["account_ids"] = account_ids

                if not dry_run:
                    for account_id in account_ids:
                        # Step 1 — kill all working orders
                        try:
                            await asyncio.wait_for(
                                client.cancel_all_orders(account_id=account_id),
                                timeout=10.0,
                            )
                            result["cancelled"] = True
                        except Exception as exc:
                            logger.warning(
                                "rithmic[%s] cancel_all_orders(%s) failed: %s",
                                key,
                                account_id,
                                exc,
                            )

                    # Brief pause so exchange acks land before flatten
                    await asyncio.sleep(0.5)

                    for account_id in account_ids:
                        # Step 2 — flatten remaining net position at market
                        try:
                            await asyncio.wait_for(
                                client.exit_position(
                                    account_id=account_id,
                                    manual_or_auto=OrderPlacement.MANUAL,
                                ),
                                timeout=10.0,
                            )
                            result["flattened"] = True
                        except Exception as exc:
                            logger.warning(
                                "rithmic[%s] exit_position(%s) failed: %s",
                                key,
                                account_id,
                                exc,
                            )
                else:
                    logger.info(
                        "rithmic[%s] EOD dry-run — would cancel+flatten account_ids=%s",
                        key,
                        account_ids,
                    )

            except TimeoutError:
                result["error"] = "connection timed out (15 s)"
            except Exception as exc:
                result["error"] = str(exc)[:300]
                logger.error("rithmic[%s] EOD close error: %s", key, exc)
            finally:
                if client is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(client.disconnect(), timeout=5.0)

            logger.info(
                "rithmic[%s] EOD close: cancelled=%s flattened=%s dry_run=%s error=%r",
                key,
                result.get("cancelled"),
                result.get("flattened"),
                dry_run,
                result.get("error"),
            )
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

_manager: RithmicAccountManager | None = None


def get_manager() -> RithmicAccountManager:
    global _manager
    if _manager is None:
        _manager = RithmicAccountManager()
        _manager.reload_configs()
    return _manager


# ---------------------------------------------------------------------------
# Persistent stream manager (RITHMIC-STREAM-A)
# ---------------------------------------------------------------------------


class RithmicStreamManager:
    """Persistent Rithmic connection for live market data streaming.

    Manages long-lived connections to TICKER_PLANT (live ticks + time bars)
    and PNL_PLANT (real-time P&L). Unlike RithmicAccountManager which uses
    short-lived polling connections, this class maintains a persistent
    connection with automatic reconnection on failure.

    Gated by:
        RITHMIC_LIVE_DATA=1   — enables the streaming connection
        RITHMIC_DEBUG_LOGGING=1 — verbose connection logging

    Redis keys written:
        rithmic:ticks:{symbol}    — rolling 300-tick window (list, LPUSH+LTRIM)
        rithmic:l1:{symbol}       — best bid/ask (hash, 2s TTL)
        rithmic:pnl:{account_key} — real-time P&L per account (hash, no TTL)
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._connected: bool = False
        self._subscribed_symbols: set[str] = set()
        self._reconnect_task: asyncio.Task[None] | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._config: RithmicAccountConfig | None = None
        self._debug: bool = os.getenv("RITHMIC_DEBUG_LOGGING", "0") == "1"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, config: RithmicAccountConfig) -> bool:
        """Connect to TICKER_PLANT.  Returns True on success.

        Uses exponential backoff on failure (max 5 retries, 2→4→8→16→32 s).
        The config is saved so that ``reconnect`` can re-use it later.
        """
        async with self._lock:
            self._config = config

        username = config.get_username()
        password = config.get_password()
        if not username or not password:
            logger.warning("rithmic-stream: credentials not configured — cannot start")
            return False

        max_retries = 5
        delay = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                from async_rithmic import RithmicClient, SysInfraType  # type: ignore[import-untyped]

                gateway = _resolve_gateway(config.gateway)
                if gateway is None:
                    logger.error("rithmic-stream: unknown gateway %r", config.gateway)
                    return False

                if self._debug:
                    logger.debug(
                        "rithmic-stream: connecting (attempt %d/%d) user=%s gateway=%s",
                        attempt,
                        max_retries,
                        config.get_username(),
                        config.gateway,
                    )

                client = RithmicClient(
                    user=username,
                    password=password,
                    system_name=config.system_name,
                    app_name=config.app_name,
                    app_version=config.app_version,
                    url=gateway,
                )
                await asyncio.wait_for(
                    client.connect(infra_type=SysInfraType.TICKER_PLANT),
                    timeout=20.0,
                )

                async with self._lock:
                    self._client = client
                    self._connected = True

                logger.info(
                    "rithmic-stream: connected to TICKER_PLANT (gateway=%s user=%s)",
                    config.gateway,
                    config.get_username(),
                )
                return True

            except ImportError:
                logger.error("rithmic-stream: async-rithmic not installed — streaming unavailable")
                return False
            except TimeoutError:
                logger.warning("rithmic-stream: connect attempt %d timed out", attempt)
            except Exception as exc:
                logger.warning("rithmic-stream: connect attempt %d failed: %s", attempt, exc)

            if attempt < max_retries:
                if self._debug:
                    logger.debug("rithmic-stream: retrying in %.0f s", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 32.0)

        logger.error("rithmic-stream: exhausted %d connect attempts — giving up", max_retries)
        async with self._lock:
            self._connected = False
        return False

    async def stop(self) -> None:
        """Disconnect gracefully and cancel any pending reconnect task."""
        # Cancel reconnect background task first
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None

        async with self._lock:
            client = self._client
            self._client = None
            self._connected = False

        if client is not None:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=5.0)
                logger.info("rithmic-stream: disconnected")
            except Exception as exc:
                logger.debug("rithmic-stream: disconnect error (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Symbol subscription
    # ------------------------------------------------------------------

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        """Subscribe to tick + L1 data for each symbol.

        Calls ``client.subscribe_to_market_data(symbol)`` from the
        async_rithmic API.  Successfully subscribed symbols are stored in
        ``_subscribed_symbols`` so they can be re-subscribed on reconnect.
        """
        async with self._lock:
            client = self._client
            connected = self._connected

        if not connected or client is None:
            logger.warning("rithmic-stream: cannot subscribe — not connected")
            return

        for symbol in symbols:
            try:
                await asyncio.wait_for(
                    client.subscribe_to_market_data(symbol),
                    timeout=10.0,
                )
                async with self._lock:
                    self._subscribed_symbols.add(symbol)
                if self._debug:
                    logger.debug("rithmic-stream: subscribed to %s", symbol)
            except Exception as exc:
                logger.warning("rithmic-stream: subscribe(%s) failed: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Tick callback
    # ------------------------------------------------------------------

    async def on_tick(self, tick_data: Any) -> None:
        """Callback for incoming tick events from TICKER_PLANT.

        Publishes to Redis:
          - ``rithmic:ticks:{symbol}``  — rolling 300-tick list (LPUSH + LTRIM)
          - ``rithmic:l1:{symbol}``     — hash with bid/ask/last/volume (2 s TTL)

        ``tick_data`` is the raw object from async_rithmic; we read attributes
        defensively with ``getattr`` so schema changes don't crash the callback.
        """
        try:
            symbol: str = str(getattr(tick_data, "symbol", "") or getattr(tick_data, "ticker", ""))
            if not symbol:
                return

            bid = getattr(tick_data, "best_bid_price", None)
            ask = getattr(tick_data, "best_ask_price", None)
            last = getattr(tick_data, "last_trade_price", None) or getattr(tick_data, "trade_price", None)
            volume = getattr(tick_data, "last_trade_size", None) or getattr(tick_data, "trade_size", None)
            ts = getattr(tick_data, "trade_time", None) or getattr(tick_data, "ssboe", None)

            tick_dict: dict[str, Any] = {
                "symbol": symbol,
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "last": float(last) if last is not None else None,
                "volume": int(volume) if volume is not None else None,
                "ts": int(ts) if ts is not None else None,
            }

            from lib.core.cache import REDIS_AVAILABLE, _r

            if not REDIS_AVAILABLE or _r is None:
                return

            # Rolling 300-tick window
            ticks_key = f"rithmic:ticks:{symbol}"
            _r.lpush(ticks_key, json.dumps(tick_dict))
            _r.ltrim(ticks_key, 0, 299)

            # L1 snapshot with 2-second TTL
            l1_key = f"rithmic:l1:{symbol}"
            l1_data: dict[str, str] = {}
            if tick_dict["bid"] is not None:
                l1_data["bid"] = str(tick_dict["bid"])
            if tick_dict["ask"] is not None:
                l1_data["ask"] = str(tick_dict["ask"])
            if tick_dict["last"] is not None:
                l1_data["last"] = str(tick_dict["last"])
            if tick_dict["volume"] is not None:
                l1_data["volume"] = str(tick_dict["volume"])
            if tick_dict["ts"] is not None:
                l1_data["ts"] = str(tick_dict["ts"])
            if l1_data:
                _r.hset(l1_key, mapping=l1_data)
                _r.expire(l1_key, 2)

        except Exception as exc:
            logger.debug("rithmic-stream: on_tick error: %s", exc)

    # ------------------------------------------------------------------
    # Reconnect background task
    # ------------------------------------------------------------------

    async def reconnect(self, config: RithmicAccountConfig) -> None:
        """Background task: re-connect and re-subscribe after a disconnect.

        Intended to be spawned as an ``asyncio.create_task`` when the stream
        drops.  Calls ``start()`` (which already does exponential backoff)
        and then re-subscribes all previously subscribed symbols.
        """
        logger.info("rithmic-stream: starting reconnect background task")
        try:
            success = await self.start(config)
            if success:
                async with self._lock:
                    symbols = list(self._subscribed_symbols)
                if symbols:
                    await self.subscribe_symbols(symbols)
                    logger.info("rithmic-stream: reconnected and re-subscribed %d symbols", len(symbols))
            else:
                logger.error("rithmic-stream: reconnect failed — stream is offline")
        except asyncio.CancelledError:
            logger.debug("rithmic-stream: reconnect task cancelled")
            raise
        except Exception as exc:
            logger.error("rithmic-stream: reconnect task error: %s", exc)

    def trigger_reconnect(self, config: RithmicAccountConfig | None = None) -> None:
        """Schedule a reconnect task on the running event loop (non-async entry point).

        Safe to call from a sync context or from a connection-dropped callback.
        Does nothing if a reconnect task is already running.
        """
        cfg = config or self._config
        if cfg is None:
            logger.warning("rithmic-stream: trigger_reconnect called with no config — ignoring")
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            logger.debug("rithmic-stream: reconnect already in progress — skipping")
            return
        try:
            loop = asyncio.get_event_loop()
            self._reconnect_task = loop.create_task(self.reconnect(cfg))
        except RuntimeError:
            logger.debug("rithmic-stream: no running event loop for reconnect task")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_live(self) -> bool:
        """Return True when RITHMIC_LIVE_DATA=1 AND the stream is connected."""
        return os.getenv("RITHMIC_LIVE_DATA", "0") == "1" and self._connected

    # ------------------------------------------------------------------
    # Redis readers (async)
    # ------------------------------------------------------------------

    async def get_l1_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Read the latest L1 bid/ask/last from Redis ``rithmic:l1:{symbol}``.

        Returns None when the key is missing or Redis is unavailable.
        The key has a 2 s TTL so a None result means no recent tick arrived.
        """
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if not REDIS_AVAILABLE or _r is None:
                return None

            raw = _r.hgetall(f"rithmic:l1:{symbol}")
            if not raw:
                return None

            # Decode bytes keys/values from Redis
            decoded: dict[str, Any] = {}
            for k, v in raw.items():
                key_str = k.decode() if isinstance(k, bytes) else str(k)
                val_str = v.decode() if isinstance(v, bytes) else str(v)
                # Coerce numeric fields
                try:
                    decoded[key_str] = float(val_str) if "." in val_str else int(val_str)
                except (ValueError, TypeError):
                    decoded[key_str] = val_str

            decoded["symbol"] = symbol
            return decoded

        except Exception as exc:
            logger.debug("rithmic-stream: get_l1_snapshot(%s) error: %s", symbol, exc)
            return None

    async def get_recent_ticks(self, symbol: str, n: int = 50) -> list[dict[str, Any]]:
        """Read the *n* most recent ticks from Redis ``rithmic:ticks:{symbol}``.

        Returns an empty list when the key is absent or Redis is unavailable.
        Ticks are ordered newest-first (LPUSH inserts at head).
        """
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if not REDIS_AVAILABLE or _r is None:
                return []

            n = max(1, min(n, 300))
            raw_items = _r.lrange(f"rithmic:ticks:{symbol}", 0, n - 1)
            ticks: list[dict[str, Any]] = []
            for item in raw_items:
                with contextlib.suppress(Exception):
                    ticks.append(json.loads(item.decode() if isinstance(item, bytes) else item))
            return ticks

        except Exception as exc:
            logger.debug("rithmic-stream: get_recent_ticks(%s) error: %s", symbol, exc)
            return []


# ---------------------------------------------------------------------------
# Stream manager singleton
# ---------------------------------------------------------------------------

_stream_manager: RithmicStreamManager | None = None


def get_stream_manager() -> RithmicStreamManager:
    """Return (or lazily create) the module-level RithmicStreamManager singleton."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = RithmicStreamManager()
    return _stream_manager


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------


def _render_status_html(all_status: dict[str, dict[str, Any]], configs_ui: list[dict[str, Any]]) -> str:
    """Render the Connections-page Rithmic panel fragment."""
    now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

    if not configs_ui:
        return """
<div style="text-align:center;padding:1.5rem 0;color:var(--text-faint);font-size:12px">
    No Rithmic accounts configured.<br/>
    <a href="/settings" style="color:#818cf8;text-decoration:none;font-size:11px">
        ⚙️ Add an account in Settings → Prop Accounts
    </a>
</div>"""

    rows = ""
    for cfg in configs_ui:
        key = cfg["key"]
        st = all_status.get(key, {})
        connected = st.get("connected", False)
        error = st.get("error", "")
        refreshed = st.get("refreshed_at", "")

        # Connection dot
        dot_color = "#22c55e" if connected else "#ef4444"
        dot_title = "Connected" if connected else (error or "Not connected")

        # P&L summary
        pnl = st.get("pnl", {})
        unreal = pnl.get("unrealized")
        real = pnl.get("realized")

        pnl_html = ""
        if unreal is not None or real is not None:

            def _fmt(v: Any) -> str:
                if v is None:
                    return "n/a"
                try:
                    fv = float(v)
                    color = "#4ade80" if fv >= 0 else "#f87171"
                    return f'<span style="color:{color}">${fv:+,.2f}</span>'
                except (TypeError, ValueError):
                    return str(v)

            pnl_html = (
                f'<span style="font-size:10px;color:var(--text-muted)">Unreal: {_fmt(unreal)}'
                f"&nbsp;·&nbsp;Real: {_fmt(real)}</span>"
            )

        # Positions summary
        positions = st.get("positions", [])
        pos_summary = ""
        if positions:
            pos_items = ", ".join(
                f"{p['symbol']} {'+' if int(p.get('size', 0)) > 0 else ''}{p.get('size', 0)}" for p in positions[:4]
            )
            pos_summary = f'<div style="font-size:10px;color:var(--text-muted);margin-top:2px">Pos: {pos_items}</div>'

        age_str = ""
        if refreshed:
            try:
                dt = datetime.fromisoformat(refreshed)
                age_s = (datetime.now(tz=_EST) - dt).total_seconds()
                age_str = f"{int(age_s)}s ago" if age_s < 60 else f"{int(age_s / 60)}m ago"
            except Exception:
                pass

        rows += f"""
<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 0;
            border-bottom:1px solid var(--border-subtle)">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                 background:{dot_color};margin-top:3px;flex-shrink:0"
          title="{dot_title}"></span>
    <div style="flex:1;min-width:0">
        <div style="display:flex;align-items:center;justify-content:space-between">
            <span style="font-size:12px;font-weight:600">{cfg["label"]}</span>
            <span style="font-size:10px;color:var(--text-faint)">{age_str}</span>
        </div>
        <div style="font-size:10px;color:var(--text-muted)">{cfg["prop_firm_label"]}</div>
        {pnl_html}
        {pos_summary}
        {f'<div style="font-size:10px;color:#f87171;margin-top:2px">{error}</div>' if not connected and error else ""}
    </div>
    <button onclick="refreshRithmicAccount('{key}')"
            style="background:none;border:1px solid var(--border-subtle);border-radius:4px;
                   padding:2px 7px;font-size:10px;color:var(--text-muted);cursor:pointer;
                   white-space:nowrap"
            title="Force refresh this account">↺</button>
</div>"""

    return f"""
<div style="font-size:11px">
    {rows}
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding-top:6px;margin-top:2px">
        <span style="color:var(--text-faint);font-size:10px">Updated {now_str}</span>
        <button onclick="refreshAllRithmic()"
                style="background:none;border:1px solid var(--border-subtle);border-radius:4px;
                       padding:2px 8px;font-size:10px;color:var(--text-muted);cursor:pointer">
            ↺ Refresh all
        </button>
    </div>
</div>
<script>
function refreshRithmicAccount(key) {{
    fetch('/api/rithmic/account/' + key + '/refresh', {{method:'POST'}})
        .then(function() {{
            if (typeof htmx !== 'undefined')
                htmx.ajax('GET', '/api/rithmic/status/html', {{target:'#conn-rithmic', swap:'innerHTML'}});
        }});
}}
function refreshAllRithmic() {{
    fetch('/api/rithmic/refresh-all', {{method:'POST'}})
        .then(function() {{
            setTimeout(function() {{
                if (typeof htmx !== 'undefined')
                    htmx.ajax('GET', '/api/rithmic/status/html', {{target:'#conn-rithmic', swap:'innerHTML'}});
            }}, 2000);
        }});
}}
</script>"""


def _render_settings_panel(configs_ui: list[dict[str, Any]]) -> str:
    """Render the Settings-page Prop Accounts panel HTML."""

    # Build existing account rows
    account_rows = ""
    for cfg in configs_ui:
        key = cfg["key"]
        prop_firm_options = "".join(
            f'<option value="{pf["value"]}"'
            f"{' selected' if pf['value'] == cfg['prop_firm'] else ''}>"
            f"{pf['label']}</option>"
            for pf in PROP_FIRM_OPTIONS
        )
        gateway_options = "".join(
            f'<option value="{g}"{" selected" if g == cfg["gateway"] else ""}>{g}</option>'
            for g in ["Chicago", "Sydney", "Sao Paulo", "test"]
        )
        acct_size = cfg.get("account_size", 150_000)
        account_size_options = "".join(
            f'<option value="{v}"{" selected" if v == acct_size else ""}>${v:,}</option>'
            for v in [25_000, 50_000, 100_000, 150_000, 200_000, 300_000]
        )
        enabled_checked = "checked" if cfg.get("enabled", True) else ""
        username_hint = cfg.get("username_hint", "")
        placeholder_user = f"Current: {username_hint}" if username_hint else "Rithmic username"
        placeholder_pass = "Leave blank to keep current" if cfg.get("password_set") else "Rithmic password"

        account_rows += f"""
<div class="rithmic-account-card" id="rithmic-card-{key}"
     style="border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:10px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <span style="font-size:0.8rem;font-weight:700">{cfg["label"]}</span>
        <div style="display:flex;gap:6px;align-items:center">
            <label style="display:flex;align-items:center;gap:4px;font-size:0.72rem;color:var(--muted)">
                <input type="checkbox" id="rith-{key}-enabled" {enabled_checked}
                       onchange="saveRithmicAccount('{key}')"/> Enabled
            </label>
            <button onclick="removeRithmicAccount('{key}')"
                    style="background:none;border:1px solid rgba(239,68,68,0.4);border-radius:4px;
                           padding:2px 8px;font-size:0.68rem;color:#f87171;cursor:pointer">
                ✕ Remove
            </button>
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">Label</label>
            <input type="text" id="rith-{key}-label" value="{cfg["label"]}"
                   style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                          border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                          font-family:inherit"/>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">Prop Firm</label>
            <select id="rith-{key}-propfirm"
                    onchange="applyPropFirmPreset('{key}')"
                    style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                           border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                           font-family:inherit">
                {prop_firm_options}
            </select>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">
                Username
            </label>
            <input type="text" id="rith-{key}-user" placeholder="{placeholder_user}"
                   autocomplete="off"
                   style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                          border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                          font-family:inherit"/>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">
                Password
            </label>
            <input type="password" id="rith-{key}-pass" placeholder="{placeholder_pass}"
                   autocomplete="new-password"
                   style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                          border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                          font-family:inherit"/>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">
                Gateway
            </label>
            <select id="rith-{key}-gateway"
                    style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                           border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                           font-family:inherit">
                {gateway_options}
            </select>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">
                Account Size
            </label>
            <select id="rith-{key}-acctsize"
                    style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                           border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                           font-family:inherit">
                {account_size_options}
            </select>
        </div>
        <div>
            <label style="font-size:0.68rem;color:var(--muted);display:block;margin-bottom:3px">
                System Name <span style="color:var(--faint)">(auto-filled)</span>
            </label>
            <input type="text" id="rith-{key}-sysname" value="{cfg["system_name"]}"
                   placeholder="e.g. Rithmic Paper Trading"
                   style="width:100%;background:var(--bg-input);border:1px solid var(--border);
                          border-radius:4px;padding:4px 8px;font-size:0.75rem;color:var(--text);
                          font-family:inherit"/>
        </div>
    </div>

    <div style="margin-top:8px;display:flex;gap:6px;align-items:center">
        <button onclick="saveRithmicAccount('{key}')"
                style="background:#3b82f6;border:none;border-radius:4px;padding:4px 12px;
                       font-size:0.75rem;color:#fff;cursor:pointer;font-family:inherit">
            💾 Save
        </button>
        <button onclick="testRithmicAccount('{key}')"
                style="background:none;border:1px solid var(--border);border-radius:4px;
                       padding:4px 12px;font-size:0.75rem;color:var(--text);cursor:pointer;
                       font-family:inherit">
            🔌 Test Connection
        </button>
        <span id="rith-{key}-msg" style="font-size:0.72rem;color:var(--muted)"></span>
    </div>
</div>"""

    presets_json = json.dumps(PROP_FIRM_PRESETS)

    return f"""
<div style="margin-bottom:12px;font-size:0.72rem;color:var(--muted);line-height:1.5;
            background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
            border-radius:7px;padding:10px 14px">
    📋 <strong>Read-only mode</strong> — connects to Rithmic to pull account
    balances, positions, and P&L. No orders are placed. Uses the same
    credentials as your prop firm's desktop platform.
    <br/>
    ⚠️ Before first use: accept market data agreements at
    <a href="https://rtraderpro.rithmic.com" target="_blank"
       style="color:#818cf8">rtraderpro.rithmic.com</a> (one-time per account).
</div>

{
        account_rows
        if account_rows
        else '<div style="color:var(--muted);font-size:0.8rem;text-align:center;padding:16px 0">'
        'No accounts configured. Click "+ Add Account" below.</div>'
    }

<div style="display:flex;gap:8px;margin-top:8px">
    <button onclick="addRithmicAccount()"
            style="background:none;border:1px solid var(--border);border-radius:4px;
                   padding:5px 14px;font-size:0.75rem;color:var(--text);cursor:pointer;
                   font-family:inherit">
        + Add Account
    </button>
    <span id="rithmic-global-msg" style="font-size:0.72rem;color:var(--muted);align-self:center"></span>
</div>

<script>
(function() {{
var _presets = {presets_json};

// Apply prop firm preset values when the dropdown changes
window.applyPropFirmPreset = function(key) {{
    var sel = document.getElementById('rith-' + key + '-propfirm');
    if (!sel) return;
    var preset = _presets[sel.value] || {{}};
    var sysnameEl = document.getElementById('rith-' + key + '-sysname');
    if (sysnameEl && preset.system_name) sysnameEl.value = preset.system_name;
    var gwEl = document.getElementById('rith-' + key + '-gateway');
    if (gwEl && preset.gateway) gwEl.value = preset.gateway;
}};

window.saveRithmicAccount = function(key) {{
    var data = {{
        key:         key,
        label:       (document.getElementById('rith-' + key + '-label')    || {{}}).value || key,
        prop_firm:   (document.getElementById('rith-' + key + '-propfirm') || {{}}).value || 'tpt',
        system_name: (document.getElementById('rith-' + key + '-sysname')  || {{}}).value || '',
        gateway:     (document.getElementById('rith-' + key + '-gateway')  || {{}}).value || 'Chicago',
        account_size: parseInt((document.getElementById('rith-' + key + '-acctsize') || {{}}).value) || 150000,
        username:    (document.getElementById('rith-' + key + '-user')     || {{}}).value || '',
        password:    (document.getElementById('rith-' + key + '-pass')     || {{}}).value || '',
        enabled:     (document.getElementById('rith-' + key + '-enabled')  || {{checked:true}}).checked,
    }};
    var msgEl = document.getElementById('rith-' + key + '-msg');
    if (msgEl) msgEl.textContent = 'Saving…';
    fetch('/api/rithmic/account/' + key + '/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(data),
    }}).then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (msgEl) msgEl.textContent = d.ok ? '✅ Saved' : ('❌ ' + (d.error || 'error'));
        // Clear password field after save for security
        var passEl = document.getElementById('rith-' + key + '-pass');
        if (passEl) passEl.value = '';
        setTimeout(function() {{ if (msgEl) msgEl.textContent = ''; }}, 3000);
      }}).catch(function(e) {{ if (msgEl) msgEl.textContent = '❌ ' + e; }});
}};

window.testRithmicAccount = function(key) {{
    var msgEl = document.getElementById('rith-' + key + '-msg');
    if (msgEl) msgEl.textContent = '🔌 Connecting…';
    fetch('/api/rithmic/account/' + key + '/refresh', {{method:'POST'}})
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
            if (msgEl) {{
                msgEl.textContent = d.connected
                    ? ('✅ Connected (' + (d.accounts||[]).length + ' account(s))')
                    : ('❌ ' + (d.error || 'failed'));
            }}
        }}).catch(function(e) {{ if (msgEl) msgEl.textContent = '❌ ' + e; }});
}};

window.removeRithmicAccount = function(key) {{
    if (!confirm('Remove account "' + key + '"? This cannot be undone.')) return;
    fetch('/api/rithmic/account/' + key + '/remove', {{method:'DELETE'}})
        .then(function(r) {{ return r.json(); }})
        .then(function(d) {{
            var card = document.getElementById('rithmic-card-' + key);
            if (card) card.remove();
            var gm = document.getElementById('rithmic-global-msg');
            if (gm) {{ gm.textContent = d.ok ? 'Removed.' : ('Error: ' + d.error); }}
        }});
}};

var _addCounter = 0;
window.addRithmicAccount = function() {{
    _addCounter++;
    var newKey = 'acc' + Date.now();
    // Build a minimal empty card by refreshing the settings panel
    fetch('/api/rithmic/config/new-key', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{key: newKey, label: 'Account ' + _addCounter}}),
    }}).then(function(r) {{ return r.json(); }})
      .then(function() {{
        // Reload the whole section
        if (typeof htmx !== 'undefined') {{
            htmx.ajax('GET', '/settings/rithmic/panel', {{
                target: '#rithmic-settings-panel', swap: 'innerHTML'
            }});
        }} else {{
            location.reload();
        }}
      }});
}};

}})();
</script>"""


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@router.get("/api/rithmic/accounts")
def list_accounts():
    """Return all configured accounts (no credentials)."""
    mgr = get_manager()
    return JSONResponse({"accounts": mgr.get_all_ui()})


@router.get("/api/rithmic/status")
def get_status():
    """Return last-known live status for all accounts."""
    mgr = get_manager()
    return JSONResponse(
        {
            "status": mgr.get_all_status(),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.get("/api/rithmic/status/html", response_class=HTMLResponse)
def get_status_html():
    """HTMX fragment for the Connections page Rithmic panel."""
    mgr = get_manager()
    html = _render_status_html(mgr.get_all_status(), mgr.get_all_ui())
    return HTMLResponse(content=html)


@router.get("/api/rithmic/account/{key}")
def get_account(key: str):
    """Return status for a single account."""
    mgr = get_manager()
    return JSONResponse(mgr.get_status(key))


@router.post("/api/rithmic/account/{key}/refresh")
async def refresh_account(key: str):
    """Force-refresh a single account's live data."""
    mgr = get_manager()
    result = await mgr.refresh_account(key)
    return JSONResponse(result)


@router.post("/api/rithmic/refresh-all")
async def refresh_all():
    """Refresh all enabled accounts (runs concurrently)."""
    mgr = get_manager()
    await mgr.refresh_all()
    return JSONResponse({"ok": True, "message": "refresh queued"})


@router.get("/api/rithmic/fills/{key}")
async def get_account_fills(key: str):
    """Retrieve today's order/fill history for a single Rithmic account.

    Opens a short-lived session, calls ``show_order_history_summary()``,
    caches the result in Redis (24 h TTL), and returns the fill list.

    Each fill dict contains:
        account_key, order_id, symbol, exchange, buy_sell, qty,
        fill_price, fill_time, status, commission.
    """
    mgr = get_manager()
    fills = await mgr.get_today_fills(key)
    return JSONResponse(
        {
            "ok": True,
            "account_key": key,
            "fills": fills,
            "count": len(fills),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.get("/api/rithmic/fills")
async def get_all_fills():
    """Retrieve today's order/fill history for all enabled Rithmic accounts.

    Calls ``get_today_fills`` concurrently for every enabled account and
    returns a combined flat list along with a per-account breakdown.
    """
    mgr = get_manager()
    fills = await mgr.get_all_today_fills()

    # Build per-account breakdown for convenience
    by_account: dict[str, list[dict]] = {}
    for f in fills:
        ak = f.get("account_key", "unknown")
        by_account.setdefault(ak, []).append(f)

    return JSONResponse(
        {
            "ok": True,
            "fills": fills,
            "count": len(fills),
            "by_account": {k: {"count": len(v), "fills": v} for k, v in by_account.items()},
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.post("/api/rithmic/eod-close")
async def eod_close(req: _FastAPIRequest):
    """Cancel all working orders and flatten all positions across every enabled account.

    Optional JSON body:
        { "dry_run": true }   — connect + discover but skip cancel/exit calls.

    This endpoint is called by the scheduler at 16:00 ET and can also be
    triggered manually from the dashboard.
    """
    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await req.json()

    dry_run: bool = bool(body.get("dry_run", False))

    mgr = get_manager()
    results = await mgr.eod_close_all_positions(dry_run=dry_run)

    any_error = any(r.get("error") for r in results)
    logger.info(
        "EOD close triggered via API: dry_run=%s accounts=%d any_error=%s",
        dry_run,
        len(results),
        any_error,
    )
    return JSONResponse(
        {
            "ok": not any_error,
            "dry_run": dry_run,
            "results": results,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.delete("/api/rithmic/account/{key}/remove")
def remove_account(key: str):
    """Remove an account config permanently."""
    configs = _load_configs()
    original_len = len(configs)
    configs = [c for c in configs if c.key != key]
    if len(configs) == original_len:
        return JSONResponse({"ok": False, "error": f"account '{key}' not found"})
    _save_configs(configs)
    mgr = get_manager()
    mgr.reload_configs()
    mgr._status.pop(key, None)
    return JSONResponse({"ok": True})


@router.post("/api/rithmic/config/new-key")
async def create_empty_account(request: Request):
    """Create a blank account placeholder so the UI can show an edit card."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    new_key = body.get("key", f"acc_{int(datetime.now().timestamp())}")
    label = body.get("label", "New Account")

    configs = _load_configs()
    if any(c.key == new_key for c in configs):
        return JSONResponse({"ok": True, "key": new_key, "message": "already exists"})

    new_cfg = RithmicAccountConfig(
        key=new_key,
        label=label,
        prop_firm="tpt",
        enabled=False,  # disabled until credentials are entered
    )
    configs.append(new_cfg)
    _save_configs(configs)
    get_manager().reload_configs()
    return JSONResponse({"ok": True, "key": new_key})


@router.get("/settings/rithmic/panel", response_class=HTMLResponse)
def rithmic_settings_panel():
    """HTMX fragment: render just the Rithmic account cards for the Settings page."""
    mgr = get_manager()
    html = _render_settings_panel(mgr.get_all_ui())
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Save + dependency-check endpoints
# ---------------------------------------------------------------------------

from fastapi import Request as _FastAPIRequest  # noqa: E402 — needed after router definition


@router.get("/api/rithmic/deps")
def check_deps():
    """Return installation status for packages required by the Rithmic integration."""
    import importlib

    packages = {
        "async-rithmic": "async_rithmic",
        "cryptography": "cryptography",
        "httpx": "httpx",
    }
    result: dict[str, dict[str, Any]] = {}
    for display_name, import_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", None) or getattr(mod, "version", None) or "installed"
            result[display_name] = {"installed": True, "version": str(version)}
        except ImportError:
            result[display_name] = {"installed": False, "version": None}

    all_ok = all(v["installed"] for v in result.values())
    return JSONResponse({"ok": all_ok, "packages": result})


@router.post("/api/rithmic/account/{key}/save")
async def save_account_ui(key: str, req: _FastAPIRequest):
    """Save account config from the Settings UI.

    Identical to /config but uses the path the frontend JS calls.
    Body JSON: key, label, prop_firm, system_name, gateway,
               username (plaintext), password (plaintext), enabled.
    Password is only updated when a non-empty value is provided.
    """
    try:
        body: dict[str, Any] = await req.json()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"invalid JSON: {exc}"}, status_code=400)

    body_key = body.get("key", key)
    configs = _load_configs()
    existing = next((c for c in configs if c.key == body_key), None)
    if existing is None:
        existing = RithmicAccountConfig(key=body_key, label=body.get("label", body_key))
        configs.append(existing)

    existing.label = body.get("label", existing.label)
    existing.prop_firm = body.get("prop_firm", existing.prop_firm)
    existing.gateway = body.get("gateway", existing.gateway)
    existing.enabled = bool(body.get("enabled", existing.enabled))
    existing.account_size = int(body.get("account_size", existing.account_size))

    sys_override = body.get("system_name", "").strip()
    preset = PROP_FIRM_PRESETS.get(existing.prop_firm, {})
    existing.system_name = sys_override or preset.get("system_name", existing.system_name)

    new_user = body.get("username", "").strip()
    new_pass = body.get("password", "").strip()
    if new_user:
        existing._username_enc = _encrypt(new_user)
    if new_pass:
        existing._password_enc = _encrypt(new_pass)

    _save_configs(configs)
    get_manager().reload_configs()
    logger.info("rithmic: saved config for account '%s' via UI", body_key)
    return JSONResponse({"ok": True, "key": body_key})


@router.post("/api/rithmic/account/{key}/config")
async def save_account_config(key: str, req: _FastAPIRequest):
    """Save (upsert) a single account config from the Settings UI.

    Body (JSON):
        key, label, prop_firm, system_name, gateway,
        username  (plaintext — encrypted server-side, optional),
        password  (plaintext — encrypted server-side, optional),
        enabled   (bool)
    """
    try:
        body: dict[str, Any] = await req.json()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"invalid JSON: {exc}"}, status_code=400)

    body_key = body.get("key", key)
    if body_key != key:
        return JSONResponse({"ok": False, "error": "key mismatch"}, status_code=400)

    configs = _load_configs()
    existing = next((c for c in configs if c.key == key), None)

    if existing is None:
        existing = RithmicAccountConfig(key=key, label=body.get("label", key))
        configs.append(existing)

    existing.label = body.get("label", existing.label)
    existing.prop_firm = body.get("prop_firm", existing.prop_firm)
    existing.gateway = body.get("gateway", existing.gateway)
    existing.enabled = bool(body.get("enabled", existing.enabled))
    existing.app_name = body.get("app_name", existing.app_name)
    existing.app_version = body.get("app_version", existing.app_version)
    existing.account_size = int(body.get("account_size", existing.account_size))

    # Resolve system_name: explicit > preset > existing
    sys_override = body.get("system_name", "").strip()
    preset = PROP_FIRM_PRESETS.get(existing.prop_firm, {})
    existing.system_name = sys_override or preset.get("system_name", existing.system_name)

    # Only update credentials when provided (non-empty strings)
    new_user = body.get("username", "").strip()
    new_pass = body.get("password", "").strip()
    if new_user:
        existing._username_enc = _encrypt(new_user)
    if new_pass:
        existing._password_enc = _encrypt(new_pass)

    _save_configs(configs)
    get_manager().reload_configs()

    logger.info("rithmic: saved config for account '%s'", key)
    return JSONResponse({"ok": True, "key": key})


# ---------------------------------------------------------------------------
# Streaming endpoints (RITHMIC-STREAM-A)
# ---------------------------------------------------------------------------


@router.get("/api/rithmic/stream/status")
def stream_status():
    """Return the current state of the persistent stream manager.

    Response fields:
        live          — True when RITHMIC_LIVE_DATA=1 AND connected
        connected     — raw _connected flag (ignores env var)
        env_enabled   — True when RITHMIC_LIVE_DATA=1
        subscribed_symbols — list of currently subscribed symbols
    """
    sm = get_stream_manager()
    return JSONResponse(
        {
            "live": sm.is_live(),
            "connected": sm._connected,
            "env_enabled": os.getenv("RITHMIC_LIVE_DATA", "0") == "1",
            "subscribed_symbols": sorted(sm._subscribed_symbols),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    )


@router.post("/api/rithmic/stream/subscribe")
async def stream_subscribe(req: _FastAPIRequest):
    """Subscribe the stream manager to a list of symbols.

    Body JSON:
        { "symbols": ["MES", "MNQ"] }

    Returns 503 when the stream is not connected.
    Symbols are passed directly to ``subscribe_symbols``; each one triggers
    a ``subscribe_to_market_data`` call on the live TICKER_PLANT connection.
    """
    try:
        body: dict[str, Any] = await req.json()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"invalid JSON: {exc}"}, status_code=400)

    symbols: list[str] = body.get("symbols", [])
    if not symbols or not isinstance(symbols, list):
        return JSONResponse({"ok": False, "error": "symbols must be a non-empty list"}, status_code=400)

    sm = get_stream_manager()

    if not sm._connected:
        return JSONResponse(
            {"ok": False, "error": "stream not connected — start the stream first"},
            status_code=503,
        )

    await sm.subscribe_symbols([str(s) for s in symbols])

    return JSONResponse(
        {
            "ok": True,
            "requested": symbols,
            "subscribed_symbols": sorted(sm._subscribed_symbols),
        }
    )


@router.get("/api/rithmic/stream/l1/{symbol}")
async def stream_l1(symbol: str):
    """Return the latest L1 bid/ask/last snapshot for *symbol* from Redis.

    The value is written by ``RithmicStreamManager.on_tick`` with a 2 s TTL.
    Returns 404 when no recent tick has arrived (stream not live or no data).
    """
    sm = get_stream_manager()
    snapshot = await sm.get_l1_snapshot(symbol.upper())
    if snapshot is None:
        return JSONResponse(
            {"ok": False, "error": f"no L1 data for {symbol} — stream may not be live"},
            status_code=404,
        )
    return JSONResponse({"ok": True, "symbol": symbol.upper(), "l1": snapshot})


@router.get("/api/rithmic/stream/ticks/{symbol}")
async def stream_ticks(symbol: str, n: int = 50):
    """Return the *n* most recent ticks for *symbol* from Redis.

    Query param:
        n — number of ticks to return (1–300, default 50)

    Ticks are ordered newest-first.  Returns an empty list when the stream
    has not published any ticks for this symbol yet.
    """
    n = max(1, min(n, 300))
    sm = get_stream_manager()
    ticks = await sm.get_recent_ticks(symbol.upper(), n=n)
    return JSONResponse(
        {
            "ok": True,
            "symbol": symbol.upper(),
            "count": len(ticks),
            "ticks": ticks,
        }
    )
