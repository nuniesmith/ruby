"""
lib.services — Service-layer modules for Ruby Futures.

Sub-packages:
    engine   — Background computation worker (scheduling, breakout detection,
               focus computation, risk management, position management, etc.)
               Contains the nested ``engine.data`` sub-package which provides
               the FastAPI REST API, SSE endpoints, and background tasks that
               run inside the same container as the engine worker.
    training — GPU training pipeline (dataset generation, ORB simulation,
               CNN trainer server). Runs on a dedicated GPU machine.
    web      — HTMX dashboard frontend (reverse proxy to data service).
"""
