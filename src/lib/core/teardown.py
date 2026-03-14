#!/usr/bin/env python3
"""
Core teardown functionality.

This module handles the orderly shutdown of all core system components,
ensuring proper resource cleanup and graceful termination of services.
It provides both component-specific teardown functions and a main orchestrator.
"""

import importlib
import importlib.util
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Core components that require specific teardown procedures
CORE_COMPONENTS = [
    "database",
    "cache",
    "scheduler",
    "event_bus",
    "pubsub",
    "filesystem",
    "security",
]

# Standard timeout for teardown operations (in seconds)
DEFAULT_TEARDOWN_TIMEOUT = 30

# Thread-safe teardown state tracking
_teardown_lock = threading.RLock()
_teardown_state: dict[str, Any] = {
    "in_progress": False,
    "completed": False,
    "start_time": None,
    "end_time": None,
    "successful_components": [],
    "failed_components": {},
    "skipped_components": [],
}

# Get logger for this module
_logger = logger
# NOTE: _logger is an alias kept for compatibility within this module


def _update_teardown_state(**kwargs) -> None:
    """
    Update the teardown state in a thread-safe manner.

    Args:
        **kwargs: Key-value pairs to update in the teardown state
    """
    global _teardown_state
    with _teardown_lock:
        for key, value in kwargs.items():
            if key in _teardown_state:
                _teardown_state[key] = value


def get_teardown_state() -> dict[str, Any]:
    """
    Get the current teardown state.

    Returns:
        Dict containing teardown state information
    """
    global _teardown_state
    with _teardown_lock:
        return _teardown_state.copy()


def _shutdown_component(component_name: str, timeout: float = DEFAULT_TEARDOWN_TIMEOUT) -> tuple[bool, str | None]:
    """
    Shutdown a specific core component with timeout.

    Args:
        component_name: Name of the component to shut down
        timeout: Maximum time to wait for shutdown (seconds)

    Returns:
        Tuple of (success, error_message)
    """
    start_time = time.time()

    try:
        # Try to dynamically import the component's teardown module
        module_path = f"core.{component_name}.teardown"

        # Check if the module exists before trying to import
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            # Try a secondary path
            module_path = f"core.components.{component_name}.teardown"
            spec = importlib.util.find_spec(module_path)

            if spec is None:
                # Last attempt - direct component teardown
                module_path = f"core.{component_name}"
                spec = importlib.util.find_spec(module_path)

                if spec is None:
                    _logger.debug("No teardown module found", component=component_name)
                    return True, None  # Assume success if component doesn't exist

        # Import the module
        module = importlib.import_module(module_path)

        # Look for shutdown/teardown function
        if hasattr(module, "teardown"):
            teardown_func = module.teardown
        elif hasattr(module, "shutdown"):
            teardown_func = module.shutdown
        else:
            _logger.debug("No teardown/shutdown function found", module_path=module_path)
            return True, None  # Assume success if no teardown function

        # If we have a background thread version, use it with timeout
        if hasattr(module, "teardown_async") or hasattr(module, "shutdown_async"):
            async_func = getattr(module, "teardown_async", None) or module.shutdown_async

            # Create a thread to run the teardown
            teardown_thread = threading.Thread(target=async_func, daemon=True, name=f"teardown-{component_name}")

            # Start the thread and wait with timeout
            teardown_thread.start()
            teardown_thread.join(timeout=timeout)

            # Check if thread completed
            if teardown_thread.is_alive():
                elapsed = time.time() - start_time
                _logger.warning("Teardown timed out", component=component_name, elapsed_s=round(elapsed, 2))
                return False, f"Timed out after {elapsed:.2f}s"

        else:
            # Call the teardown function directly
            teardown_func()

        elapsed = time.time() - start_time
        _logger.info("Component shut down", component=component_name, elapsed_s=round(elapsed, 3))
        return True, None

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        _logger.error(
            "Error shutting down component", component=component_name, elapsed_s=round(elapsed, 3), error=error_msg
        )
        return False, error_msg


def shutdown_databases() -> bool:
    """
    Shutdown all database connections.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Shutting down database connections")

    try:
        # Try to import database module - handle imports with try/except
        success = True

        # Close connections
        try:
            # Use conditional import to prevent linting errors
            if TYPE_CHECKING:
                from lib.core.db import close_all_connections
            else:
                # Dynamic import at runtime
                try:
                    db_module = importlib.import_module("lib.core.db")
                    if hasattr(db_module, "close_all_connections"):
                        close_all_connections = db_module.close_all_connections
                        close_all_connections()
                        _logger.info("All database connections closed")
                except ImportError:
                    # Try alternative import paths
                    try:
                        db_module = importlib.import_module("lib.core.database")
                        if hasattr(db_module, "close_all_connections"):
                            close_all_connections = db_module.close_all_connections
                            close_all_connections()
                            _logger.info("All database connections closed")
                    except ImportError:
                        _logger.debug("Database module not found, skipping connection cleanup")
        except AttributeError:
            _logger.debug("close_all_connections function not found in database module")
        except Exception as e:
            _logger.error("Error closing database connections", error=str(e))
            success = False

        # Shut down ORM engine
        try:
            if TYPE_CHECKING:
                from lib.core.db.orm import shutdown_engine
            else:
                try:
                    orm_module = importlib.import_module("lib.core.db.orm")
                    if hasattr(orm_module, "shutdown_engine"):
                        shutdown_engine = orm_module.shutdown_engine
                        shutdown_engine()
                        _logger.info("ORM engine shut down")
                except ImportError:
                    # Try alternative paths
                    try:
                        orm_module = importlib.import_module("lib.core.database.orm")
                        if hasattr(orm_module, "shutdown_engine"):
                            shutdown_engine = orm_module.shutdown_engine
                            shutdown_engine()
                            _logger.info("ORM engine shut down")
                    except ImportError:
                        _logger.debug("ORM module not found, skipping engine shutdown")
        except AttributeError:
            _logger.debug("shutdown_engine function not found in ORM module")
        except Exception as e:
            _logger.error("Error shutting down ORM engine", error=str(e))
            success = False

        return success
    except Exception as e:
        _logger.error("Unexpected error in database shutdown", error=str(e))
        return False


def shutdown_cache() -> bool:
    """
    Shutdown cache systems.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Shutting down cache systems")

    try:
        # Try to import cache module
        try:
            cache_module = importlib.import_module("lib.core.data.cache")
            if hasattr(cache_module, "flush_and_close"):
                cache_module.flush_and_close()
                _logger.info("Cache systems flushed and closed")
            else:
                _logger.debug("flush_and_close function not found in cache module")
        except ImportError:
            # Try alternative paths
            try:
                cache_module = importlib.import_module("lib.core.cache")
                if hasattr(cache_module, "flush_and_close"):
                    cache_module.flush_and_close()
                    _logger.info("Cache systems flushed and closed")
                else:
                    _logger.debug("flush_and_close function not found in cache module")
            except ImportError:
                _logger.debug("Cache module not found, skipping cache shutdown")
            except Exception as e:
                _logger.error("Error shutting down cache", error=str(e))
                return False

        return True
    except Exception as e:
        _logger.error("Unexpected error in cache shutdown", error=str(e))
        return False


def shutdown_event_system() -> bool:
    """
    Shutdown event bus and pubsub systems.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Shutting down event systems")
    success = True

    try:
        # Try to shut down event bus
        try:
            event_bus_module = importlib.import_module("lib.core.event_bus")
            if hasattr(event_bus_module, "shutdown"):
                event_bus_module.shutdown()
                _logger.info("Event bus shut down")
            else:
                _logger.debug("shutdown function not found in event_bus module")
        except ImportError:
            _logger.debug("Event bus module not found, skipping shutdown")
        except Exception as e:
            _logger.error("Error shutting down event bus", error=str(e))
            success = False

        # Try to shut down pubsub
        try:
            pubsub_module = importlib.import_module("lib.core.pubsub")
            if hasattr(pubsub_module, "disconnect_all"):
                pubsub_module.disconnect_all()
                _logger.info("Pubsub connections closed")
            else:
                _logger.debug("disconnect_all function not found in pubsub module")
        except ImportError:
            _logger.debug("Pubsub module not found, skipping shutdown")
        except Exception as e:
            _logger.error("Error disconnecting from pubsub", error=str(e))
            success = False

        return success
    except Exception as e:
        _logger.error("Unexpected error in event system shutdown", error=str(e))
        return False


def shutdown_scheduler() -> bool:
    """
    Shutdown task scheduler.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Shutting down task scheduler")

    try:
        try:
            scheduler_module = importlib.import_module("lib.core.scheduler")
            if hasattr(scheduler_module, "shutdown"):
                scheduler_module.shutdown()
                _logger.info("Task scheduler shut down")
            else:
                _logger.debug("shutdown function not found in scheduler module")
        except ImportError:
            _logger.debug("Scheduler module not found, skipping shutdown")
        except Exception as e:
            _logger.error("Error shutting down scheduler", error=str(e))
            return False

        return True
    except Exception as e:
        _logger.error("Unexpected error in scheduler shutdown", error=str(e))
        return False


def shutdown_filesystem() -> bool:
    """
    Close file handles and finalize filesystem operations.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Finalizing filesystem operations")

    try:
        try:
            fs_module = importlib.import_module("lib.core.filesystem")
            if hasattr(fs_module, "close_all_handles"):
                fs_module.close_all_handles()
                _logger.info("All file handles closed")
            else:
                _logger.debug("close_all_handles function not found in filesystem module")
        except ImportError:
            _logger.debug("Filesystem module not found, skipping cleanup")
        except Exception as e:
            _logger.error("Error closing file handles", error=str(e))
            return False

        return True
    except Exception as e:
        _logger.error("Unexpected error in filesystem shutdown", error=str(e))
        return False


def shutdown_security() -> bool:
    """
    Clean up security and encryption resources.

    Returns:
        True if successful, False otherwise
    """
    _logger.info("Cleaning up security resources")

    try:
        try:
            security_module = importlib.import_module("lib.core.security")
            if hasattr(security_module, "cleanup"):
                security_module.cleanup()
                _logger.info("Security resources cleaned up")
            else:
                _logger.debug("cleanup function not found in security module")
        except ImportError:
            _logger.debug("Security module not found, skipping cleanup")
        except Exception as e:
            _logger.error("Error cleaning up security resources", error=str(e))
            return False

        return True
    except Exception as e:
        _logger.error("Unexpected error in security shutdown", error=str(e))
        return False


def teardown(components: list[str] | None = None, timeout: float = DEFAULT_TEARDOWN_TIMEOUT) -> bool:
    """
    Orchestrate the teardown of all core components.

    This function manages the orderly shutdown of all system components,
    ensuring resources are released properly. Components are shut down in a
    specific order to ensure dependencies are respected.

    Args:
        components: List of specific components to shut down (defaults to all)
        timeout: Maximum time to wait for each component shutdown (seconds)

    Returns:
        True if all components shut down successfully, False otherwise
    """
    global _teardown_state

    # Check if teardown already in progress
    with _teardown_lock:
        if _teardown_state["in_progress"]:
            _logger.warning("Teardown already in progress, ignoring duplicate request")
            return False

        if _teardown_state["completed"]:
            _logger.warning("Teardown already completed, ignoring duplicate request")
            return True

        # Mark teardown as started
        _teardown_state["in_progress"] = True
        _teardown_state["start_time"] = time.time()
        _teardown_state["successful_components"] = []
        _teardown_state["failed_components"] = {}
        _teardown_state["skipped_components"] = []

    _logger.info("Beginning core system teardown")

    # Determine which components to shut down
    shutdown_components = list(components) if components is not None else list(CORE_COMPONENTS)

    # Components to shut down in order (specific components with dedicated functions)
    ordered_teardown = [
        # High-level services first
        ("scheduler", shutdown_scheduler),
        # Then communication systems
        ("event_bus", shutdown_event_system),
        ("pubsub", None),  # Handled by event_system
        ("messaging", None),  # Will use generic mechanism
        # Then data systems
        ("database", shutdown_databases),
        ("cache", shutdown_cache),
        # Then supporting systems
        ("computation", None),  # Will use generic mechanism
        ("filesystem", shutdown_filesystem),
        # Finally, security
        ("security", shutdown_security),
    ]

    success = True

    # First, shut down components with specific teardown sequences
    for component_name, teardown_func in ordered_teardown:
        if component_name not in shutdown_components:
            with _teardown_lock:
                _teardown_state["skipped_components"].append(component_name)
            continue

        try:
            if teardown_func is not None:
                # Use the specialized teardown function
                component_success = teardown_func()
                if component_success:
                    with _teardown_lock:
                        _teardown_state["successful_components"].append(component_name)
                else:
                    with _teardown_lock:
                        _teardown_state["failed_components"][component_name] = "Failed to shut down properly"
                    success = False
            else:
                # Use the generic component shutdown
                component_success, error = _shutdown_component(component_name, timeout)
                if component_success:
                    with _teardown_lock:
                        _teardown_state["successful_components"].append(component_name)
                else:
                    with _teardown_lock:
                        _teardown_state["failed_components"][component_name] = error or "Unknown error"
                    success = False

            # Remove from list of components to process
            if component_name in shutdown_components:
                shutdown_components.remove(component_name)

        except Exception as e:
            _logger.error("Error in teardown of component", component=component_name, error=str(e))
            with _teardown_lock:
                _teardown_state["failed_components"][component_name] = str(e)
            success = False

    # Then handle any remaining components with the generic shutdown
    for component_name in shutdown_components:
        try:
            component_success, error = _shutdown_component(component_name, timeout)
            if component_success:
                with _teardown_lock:
                    _teardown_state["successful_components"].append(component_name)
            else:
                with _teardown_lock:
                    _teardown_state["failed_components"][component_name] = error or "Unknown error"
                success = False
        except Exception as e:
            _logger.error("Error in teardown of component", component=component_name, error=str(e))
            with _teardown_lock:
                _teardown_state["failed_components"][component_name] = str(e)
            success = False

    # Clean up system resources
    try:
        # Flush all loggers before completing
        # [additional cleanup here if needed]
        pass

    except Exception as e:
        _logger.error("Error during final cleanup", error=str(e))
        success = False

    # Mark teardown as completed
    end_time = time.time()
    with _teardown_lock:
        _teardown_state["in_progress"] = False
        _teardown_state["completed"] = True
        _teardown_state["end_time"] = end_time

    # Calculate elapsed time
    elapsed = end_time - _teardown_state["start_time"]

    # Log summary
    with _teardown_lock:
        successful = len(_teardown_state["successful_components"])
        failed = len(_teardown_state["failed_components"])
        skipped = len(_teardown_state["skipped_components"])

    _logger.info(
        "Core teardown completed",
        elapsed_s=round(elapsed, 2),
        successful=successful,
        failed=failed,
        skipped=skipped,
    )

    return success


def emergency_shutdown() -> None:
    """
    Perform an emergency shutdown of the system.

    This is a more aggressive shutdown that should only be used when
    the normal teardown process fails or in critical situations.
    """
    _logger.warning("EMERGENCY SHUTDOWN INITIATED")

    try:
        # Try to stop any critical services with minimal cleanup
        # Database connections
        try:
            db_module = importlib.import_module("lib.core.database")
            if hasattr(db_module, "emergency_close"):
                db_module.emergency_close()
                _logger.info("Database emergency shutdown complete")
            else:
                _logger.warning("emergency_close function not found in database module")
        except ImportError:
            _logger.debug("Database module not found, skipping emergency shutdown")
        except Exception as e:
            _logger.error("Error in emergency database shutdown", error=str(e))

        # File handles
        try:
            fs_module = importlib.import_module("lib.core.filesystem")
            if hasattr(fs_module, "emergency_close"):
                fs_module.emergency_close()
                _logger.info("Filesystem emergency shutdown complete")
            else:
                _logger.warning("emergency_close function not found in filesystem module")
        except ImportError:
            _logger.debug("Filesystem module not found, skipping emergency shutdown")
        except Exception as e:
            _logger.error("Error in emergency filesystem shutdown", error=str(e))

        # Other critical cleanup
        # [add as needed]

    except Exception as e:
        _logger.critical("Critical error during emergency shutdown", error=str(e))

    # Force flush all loggers
    # (No logger.complete() method; consider logger.remove() or logger.shutdown() if needed)

    # Force flush all loggers
    _logger.warning("Emergency shutdown complete")
    success = teardown()
    sys.exit(0 if success else 1)
