"""
Web Service — entrypoint.

Thin wrapper that re-exports the FastAPI app from
``lib.services.web.main`` so that uvicorn and ``python -m`` invocations
work from the entrypoints package.

Usage:
    PYTHONPATH=src uvicorn entrypoints.web.main:app --host 0.0.0.0 --port 8080

Docker:
    CMD ["uvicorn", "entrypoints.web.main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

from lib.services.web.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    from lib.services.web.main import LOG_LEVEL, WEB_HOST, WEB_PORT

    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level=LOG_LEVEL,
    )
