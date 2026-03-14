"""
Service Runner Module.

This module provides the ServiceRunner class that handles the lifecycle
of a service including startup, signal handling, and shutdown.
"""

import json
import os
import signal
import sys
import traceback
from pathlib import Path

from lib.core.config import load_configuration
from lib.core.logging_config import get_logger, setup_logging
from lib.utils.discover import find_config_file


class ServiceRunner:
    """
    Runner class that handles service lifecycle.
    """

    def __init__(self, service):
        """
        Initialize the service runner.

        Args:
            service: The service to run
        """
        self.service = service
        # Logger is already set up by BaseService.__init__
        self.logger = service.logger
        self._stopping = False

    def handle_shutdown(self, sig, frame) -> None:
        """
        Handle shutdown signals gracefully.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        # Prevent multiple shutdown attempts
        if self._stopping:
            self.logger.info("Shutdown already in progress")
            return

        self._stopping = True
        self.logger.info(f"Received signal {sig}, shutting down {self.service.service_name} service...")

        try:
            self.service.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

        sys.exit(0)

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        # Set up additional signals on Unix platforms
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self.handle_shutdown)

    def run(self) -> None:
        """Run the service."""
        try:
            # Setup environment
            base_dir = self.service.setup_environment()
            self.logger.info(f"Starting {self.service.service_name} service from {base_dir}")

            # Parse arguments
            args = self.service.parse_arguments()

            # If diagnostics mode, run diagnostics and exit
            if hasattr(args, "diagnose") and args.diagnose:
                diagnostics = self.service.run_diagnostics()
                print(f"=== Diagnostics for {self.service.service_name} service ===")
                print(json.dumps(diagnostics, indent=2, default=str))
                sys.exit(0)

            # Update logging level if specified via command line
            # Re-setup logging ONLY if the level changed from the initial setup
            initial_log_level = os.environ.get(f"{self.service.service_name.upper()}_LOG_LEVEL", "INFO").upper()

            if args.log_level.upper() != initial_log_level:
                self.logger.info(f"Log level changed via argument to: {args.log_level.upper()}")

                # Re-initialize logger with new log level
                try:
                    # Check if the logger has a set_level method (from the comprehensive logging system)
                    if hasattr(self.service.logger, "set_level"):
                        self.service.logger.set_level(args.log_level.upper())
                        self.logger.info(f"Updated log level to: {args.log_level.upper()}")
                    else:
                        # Otherwise reconfigure logging with the new level and get a fresh logger
                        setup_logging(service=self.service.service_name, level=args.log_level)
                        self.service.logger = get_logger(self.service.service_name)
                        self.logger = self.service.logger
                except Exception as e:
                    self.logger.warning(f"Failed to update log level: {e}")

            # Setup signal handlers
            self.setup_signal_handlers()

            # Find and load configuration
            config_path = args.config or find_config_file(self.service.service_name, base_dir)
            if config_path and Path(config_path).exists():  # Check existence before loading
                config_success, config = load_configuration(config_path)

                if not config_success:
                    self.logger.warning(f"Failed to load configuration from {config_path}, using defaults")
                    config = {}
                else:
                    self.logger.info(f"Loaded configuration from {config_path}")
            else:
                self.logger.warning(f"Configuration path not found or specified ({config_path}), using defaults")
                config = {}

            # Load settings if available and override args
            settings = self.service.load_settings()
            if settings:
                self.logger.info("Using settings module")

                # Apply settings to args if attributes exist
                for attr in ["port", "host", "workers"]:
                    env_attr = attr.upper()
                    setting_value = None
                    if hasattr(settings, env_attr):
                        setting_value = getattr(settings, env_attr)
                    elif hasattr(settings, "server") and hasattr(settings.server, env_attr):
                        setting_value = getattr(settings.server, env_attr)

                    if hasattr(args, attr) and setting_value is not None:
                        setattr(args, attr, setting_value)
                        self.logger.debug(f"Overriding arg '{attr}' with setting value: {setting_value}")

            # Initialize components
            self.service.initialize_components(config)

            # Preprocess configuration
            config = self.service.preprocess_config(config)

            # Create app
            app = self.service.create_app(config)
            if app is None:
                self.logger.error("Failed to create application")
                sys.exit(1)

            # Run app
            self.logger.info(f"Starting {self.service.service_name} service on {args.host}:{args.port}")

            # Filter out explicitly passed parameters to avoid duplicates
            filtered_kwargs = {k: v for k, v in vars(args).items() if k not in ("host", "port", "debug", "workers")}

            self.service.run_app(
                app,
                args.host,
                args.port,
                debug=args.debug,
                workers=args.workers,
                **filtered_kwargs,  # Pass filtered kwargs to avoid duplicates
            )

        except KeyboardInterrupt:
            self.logger.info(f"Shutting down {self.service.service_name} service")
            self.service.cleanup()
            sys.exit(0)
        except Exception as e:
            # Enhanced error handling in exception case
            try:
                # Try to use the logger, but wrap in another try/except
                print(f"Error running {self.service.service_name} service: {type(e).__name__}: {e}", file=sys.stderr)

                # Safely log the exception, accounting for potential logger issues
                if hasattr(self.logger, "error") and callable(self.logger.error):
                    self.logger.error(f"Error running {self.service.service_name} service: {e}", exc_info=True)
                else:
                    print("Logger error method unavailable. Traceback:", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            except Exception as log_err:
                print(f"CRITICAL: Failed to log exception: {log_err}", file=sys.stderr)
                print("Original exception traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            sys.exit(1)
