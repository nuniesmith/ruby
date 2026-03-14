"""
ORM engine management for the Ruby Futures System.

This module provides SQLAlchemy ORM engine initialization, management
and cleanup functionality for the application.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from lib.core.logging_config import get_logger

# Type-checking-only imports so Pyright sees the real types without
# runtime side-effects when SQLAlchemy is absent.
if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.orm import DeclarativeMeta, Session

# Runtime availability check
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Module logger
_logger = get_logger(__name__)

# Base ORM model
Base: DeclarativeMeta | None = None  # type: ignore[assignment]
if HAS_SQLALCHEMY:
    Base = declarative_base()  # type: ignore[assignment]

# Engine and session tracking
_engine_lock = threading.RLock()
_engines: dict[str, Any] = {}
_session_factories: dict[str, Any] = {}
_active_sessions: list[Any] = []


def init_engine(
    db_url: str,
    engine_name: str = "default",
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    **engine_kwargs: Any,
) -> Engine | None:
    """
    Initialize a SQLAlchemy engine.

    Args:
        db_url: Database connection URL
        engine_name: Name for this engine instance
        echo: Whether to echo SQL statements
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        pool_timeout: Pool timeout in seconds
        **engine_kwargs: Additional engine arguments

    Returns:
        SQLAlchemy engine or None if SQLAlchemy is not available
    """
    if not HAS_SQLALCHEMY:
        _logger.warning("sqlalchemy_not_installed")
        return None

    with _engine_lock:
        if engine_name in _engines:
            _logger.debug("engine_already_initialized", engine_name=engine_name)
            return _engines[engine_name]

        try:
            engine = create_engine(  # type: ignore[possibly-undefined]
                db_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                **engine_kwargs,
            )

            _engines[engine_name] = engine

            session_factory = sessionmaker(bind=engine)  # type: ignore[possibly-undefined]
            _session_factories[engine_name] = scoped_session(session_factory)  # type: ignore[possibly-undefined]

            _logger.info("engine_initialized", engine_name=engine_name)
            return engine  # type: ignore[return-value]

        except Exception as e:
            _logger.error("engine_init_failed", engine_name=engine_name, error=str(e))
            return None


def get_engine(engine_name: str = "default") -> Any | None:
    """
    Get a SQLAlchemy engine by name.

    Args:
        engine_name: Name of the engine

    Returns:
        SQLAlchemy engine or None if not found
    """
    with _engine_lock:
        return _engines.get(engine_name)


def get_session(engine_name: str = "default") -> Session | None:
    """
    Get a SQLAlchemy session.

    The session will be tracked for cleanup during system shutdown.

    Args:
        engine_name: Name of the engine to use

    Returns:
        SQLAlchemy session or None if engine not found
    """
    with _engine_lock:
        session_factory = _session_factories.get(engine_name)
        if not session_factory:
            return None

        session = session_factory()
        _active_sessions.append(session)
        return session  # type: ignore[return-value]


def close_session(session: Any) -> bool:
    """
    Close a SQLAlchemy session.

    Args:
        session: The session to close

    Returns:
        True if successful, False otherwise
    """
    if not session:
        return True

    try:
        session.close()
        with _engine_lock:
            if session in _active_sessions:
                _active_sessions.remove(session)
        return True
    except Exception as e:
        _logger.error("session_close_failed", error=str(e))
        return False


def shutdown_engine(engine_name: str | None = None) -> bool:
    """
    Shutdown SQLAlchemy engine(s).

    This function properly disposes of SQLAlchemy engines and ensures
    all database connections are closed.

    Args:
        engine_name: Name of engine to shut down, or None for all engines

    Returns:
        True if successful, False otherwise
    """
    if not HAS_SQLALCHEMY:
        return True  # Nothing to do

    success = True

    # Close all active sessions first
    with _engine_lock:
        active_sessions = _active_sessions.copy()

    for session in active_sessions:
        try:
            session.close()
            with _engine_lock:
                if session in _active_sessions:
                    _active_sessions.remove(session)
        except Exception as e:
            _logger.error("session_close_failed_during_shutdown", error=str(e))
            success = False

    # Dispose engines
    with _engine_lock:
        engines_to_shutdown: dict[str, Any] = {}

        if engine_name is not None:
            # Shut down specific engine
            if engine_name in _engines:
                engines_to_shutdown[engine_name] = _engines[engine_name]
            else:
                _logger.warning("engine_not_found_for_shutdown", engine_name=engine_name)
        else:
            # Shut down all engines
            engines_to_shutdown = _engines.copy()

        # Dispose each engine
        for name, engine in engines_to_shutdown.items():
            try:
                _logger.info("engine_shutting_down", engine_name=name)
                engine.dispose()

                # Remove from tracking
                if name in _engines:
                    del _engines[name]

                if name in _session_factories:
                    _session_factories[name].remove()
                    del _session_factories[name]

            except Exception as e:
                _logger.error("engine_shutdown_failed", engine_name=name, error=str(e))
                success = False

    if success:
        _logger.info("engines_shutdown_complete")
    else:
        _logger.warning("engines_shutdown_partial_failure")

    return success
