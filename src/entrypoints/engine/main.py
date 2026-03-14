"""
Engine Service — entrypoint.

Thin wrapper that delegates to the real implementation in
``lib.services.engine.main``.

Usage:
    python -m entrypoints.engine.main

Docker:
    CMD ["python", "-m", "entrypoints.engine.main"]
"""

from lib.services.engine.main import main

if __name__ == "__main__":
    main()
