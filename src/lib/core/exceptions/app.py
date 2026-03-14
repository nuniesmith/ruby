"""
Exception handlers for the FastAPI application.

This module provides standardized exception handlers for:
- HTTP exceptions (404, 500, etc.)
- Validation exceptions (invalid input data)
- Unhandled general exceptions

These handlers ensure consistent error responses throughout the application.
"""

import logging
import traceback
from typing import Any

from fastapi import Request, status  # type: ignore[attr-defined]
from fastapi.exceptions import RequestValidationError  # type: ignore[attr-defined]
from fastapi.responses import JSONResponse  # type: ignore[attr-defined]
from starlette.exceptions import HTTPException as StarletteHTTPException  # type: ignore[attr-defined]

from lib.core.exceptions.base import FrameworkException

logger = logging.getLogger("app.exceptions")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions and return standardized JSON responses.

    Args:
        request: The incoming request
        exc: The HTTP exception

    Returns:
        JSONResponse with proper status code and formatted error message
    """
    logger.debug(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
            "details": getattr(exc, "details", {}),
            "path": request.url.path,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation exceptions from request parsing.

    Args:
        request: The incoming request
        exc: The validation exception

    Returns:
        JSONResponse with 422 status and detailed validation errors
    """
    errors = exc.errors()
    logger.debug(f"Validation error: {len(errors)} issues found")

    # Format the validation errors for better readability
    formatted_errors = []
    for error in errors:
        formatted_errors.append(
            {"loc": error.get("loc", []), "msg": error.get("msg", ""), "type": error.get("type", "")}
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {"errors": formatted_errors},
            "path": request.url.path,
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle any unhandled exceptions and return a standardized error response.

    Args:
        request: The incoming request
        exc: The unhandled exception

    Returns:
        JSONResponse with 500 status and error details (limited in production)
    """
    # Get application state for debug mode
    debug_mode = getattr(request.app.state, "debug", False)

    # Handle our custom exceptions
    if isinstance(exc, FrameworkException):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        # Map certain exception types to specific status codes
        if exc.code == "NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND
        elif exc.code == "VALIDATION_ERROR":
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif exc.code == "AUTH_ERROR":
            status_code = status.HTTP_401_UNAUTHORIZED

        return JSONResponse(status_code=status_code, content=exc.to_dict() | {"path": request.url.path})

    # For generic exceptions, generate a proper internal error
    log_msg = f"Unhandled exception: {str(exc)}"
    if debug_mode:
        logger.error(log_msg, exc_info=True)
    else:
        logger.error(log_msg)

    error_content: dict[str, Any] = {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "path": request.url.path,
    }

    # Include stack trace in debug mode only
    if debug_mode:
        error_content["details"] = {
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        }

    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_content)
