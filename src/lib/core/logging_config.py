"""
Structured logging configuration for Ruby Futures.

Provides a unified logging setup using ``structlog`` across all services
(data-service, engine, background tasks).  Call ``setup_logging()``
once at process startup — every subsequent ``structlog.get_logger()`` or
``logging.getLogger()`` call will emit structured, key-value log lines.

Usage::

    from lib.logging_config import setup_logging, get_logger

    setup_logging(service="data-service")
    logger = get_logger()

    logger.info("engine_started", account_size=150_000, interval="5m")
    # => 2025-06-01T14:23:01Z [info] engine_started  account_size=150000 interval=5m service=data-service

In development (LOG_FORMAT=console, the default) output is coloured and
human-friendly.  Set LOG_FORMAT=json for machine-parseable JSON lines
(recommended in Docker / production).
"""

import datetime
import logging
import os
import sys
from collections.abc import MutableMapping  # noqa: TC003
from typing import Any
from zoneinfo import ZoneInfo

import structlog

_ET = ZoneInfo("America/New_York")


def _et_timestamper(
    logger: Any,
    method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """structlog processor that stamps log events with Eastern Time (ET).

    Automatically reflects EDT (UTC-4) or EST (UTC-5) based on DST.
    Replaces the default UTC ``TimeStamper(fmt="iso")``.
    """
    event_dict["timestamp"] = datetime.datetime.now(_ET).strftime("%H:%M:%S")
    return event_dict


def setup_logging(
    *,
    service: str = "futures",
    level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure ``structlog`` and stdlib ``logging`` for the whole process.

    Parameters
    ----------
    service:
        Name bound to every log event (e.g. ``"data-service"``,
        ``"engine"``, ``"background-tasks"``).
    level:
        Root log level.  Falls back to the ``LOG_LEVEL`` env var, then
        ``"INFO"``.
    log_format:
        ``"console"`` for human-readable coloured output (default in dev),
        ``"json"`` for structured JSON lines.  Falls back to the
        ``LOG_FORMAT`` env var, then ``"console"``.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_format is None:
        log_format = os.getenv("LOG_FORMAT", "console").lower()

    numeric_level = getattr(logging, level, logging.INFO)

    # ------------------------------------------------------------------
    # Shared structlog processors (run on every log event)
    # ------------------------------------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _et_timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # ----- JSON output (production / Docker) -----
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        # ----- Console output (local development) -----
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
            pad_event_to=35,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )

    # ------------------------------------------------------------------
    # Wire up stdlib logging so third-party libraries (uvicorn, httpx,
    # sqlalchemy, etc.) also go through structlog formatting.
    # ------------------------------------------------------------------
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Preserve handlers that are NOT StreamHandlers (e.g. _RingBufferHandler
    # from trainer_server).  Without this, setup_logging() drops the ring
    # buffer handler, which then gets re-added on the next import — stacking
    # duplicates and causing every log line to appear multiple times.
    _keep = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]
    root_logger.handlers.clear()
    for h in _keep:
        root_logger.addHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party loggers
    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "websockets",
        "urllib3",
        "sqlalchemy.engine",
        "hmmlearn",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Configure structlog itself
    # ------------------------------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bind the service name to every future log event from structlog
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service)


def get_logger(name: str | None = None, **initial_binds: Any) -> structlog.stdlib.BoundLogger:
    """Return a structured logger, optionally bound with extra context.

    Parameters
    ----------
    name:
        Logger name (shown in ``logger_name`` field).  When ``None``,
        structlog infers the caller's module name.
    **initial_binds:
        Key-value pairs permanently bound to this logger instance.

    Examples
    --------
    >>> logger = get_logger("engine", ticker="ES")
    >>> logger.info("refresh_complete", bars=500)
    """
    log = structlog.get_logger(name)
    if initial_binds:
        log = log.bind(**initial_binds)
    return log
