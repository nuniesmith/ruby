"""
Data Service — entrypoint.

Thin wrapper that delegates to the real implementation in
``lib.services.data.main``.

Usage:
    python -m entrypoints.data.main

Docker:
    CMD ["python", "-m", "entrypoints.data.main"]
"""

from lib.services.data.main import main

if __name__ == "__main__":
    main()
