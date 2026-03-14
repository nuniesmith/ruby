"""
Web Service — HTMX Dashboard Frontend
=======================================
Thin frontend service that serves the HTMX dashboard and proxies
API/SSE requests to the data service backend.

This service is stateless — it reads no Redis or Postgres directly.
All data flows through the data service API.
"""
