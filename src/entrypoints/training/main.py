"""
Training Service — entrypoint.

Thin wrapper that delegates to the real implementation in
``lib.services.training.trainer_server``.

Usage:
    python -m entrypoints.training.main

Docker:
    CMD ["python", "-m", "entrypoints.training.main"]
"""

from lib.services.training.trainer_server import main

if __name__ == "__main__":
    main()
