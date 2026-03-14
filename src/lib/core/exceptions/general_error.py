import traceback
from typing import Any

from prometheus_client import CollectorRegistry, Counter

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Custom Prometheus registry to avoid global duplication
REGISTRY = CollectorRegistry()


def get_exception_counter():
    """
    Lazily initialize and return the Prometheus exception counter.

    Returns:
        Counter: Prometheus Counter for exceptions.
    """
    return Counter("app_exceptions_total", "Count of exceptions by type", ["exception_type"], registry=REGISTRY)


class GeneralError(BaseException):
    """
    Base class for all custom exceptions.
    """

    DEFAULT_ERROR_CODE = 1000
    DEFAULT_MESSAGE = "An unspecified error occurred."
    _EXCEPTION_COUNTER = None  # Lazily initialized exception counter

    def __init__(
        self,
        message: str | None = None,
        code: int | None = None,
        details: str | dict[str, Any] | None = None,
        context: dict[str, Any] | Exception | None = None,
        log_traceback: bool = True,
    ):
        """
        Initialize the exception with optional details and context.

        Args:
            message (Optional[str]): Human-readable error message.
            code (Optional[int]): Application-specific error code.
            details (Optional[Union[str, Dict[str, Any]]]): Additional details about the error.
            context (Optional[Union[Dict[str, Any], Exception]]): Contextual information or an exception instance.
            log_traceback (bool): Whether to log the traceback.
        """
        self.code = code or self.DEFAULT_ERROR_CODE
        self.message = message or self.DEFAULT_MESSAGE
        self.details = details or {}
        self.context = self._extract_context(context)
        self.log_traceback = log_traceback

        super().__init__(self.message)

        # Initialize the counter lazily
        if GeneralError._EXCEPTION_COUNTER is None:
            GeneralError._EXCEPTION_COUNTER = get_exception_counter()

        # Increment the Prometheus exception counter
        GeneralError._EXCEPTION_COUNTER.labels(exception_type=self.__class__.__name__).inc()

        # Log the error upon initialization
        self.log_error()

    def _extract_context(self, context: dict[str, Any] | Exception | None) -> dict[str, Any]:
        """
        Extract contextual information from the given context.

        Args:
            context (Optional[Union[Dict[str, Any], Exception]]): Contextual information or an exception instance.

        Returns:
            Dict[str, Any]: Extracted context information.
        """
        if isinstance(context, dict):
            return context
        if isinstance(context, Exception):
            extracted_context = {"exception": str(context)}
            if context.__cause__ and isinstance(context.__cause__, Exception):
                extracted_context["cause"] = str(self._extract_context(context.__cause__))
            return extracted_context
        return {}

    def log_error(self):
        """
        Log the error message and details using structlog.
        """
        log_kwargs = {"error_code": self.code, "message": self.message}
        if self.details:
            log_kwargs["details"] = self.details
        if self.context:
            log_kwargs["context"] = self.context

        logger.error("general_error_raised", **log_kwargs)

        if self.log_traceback:
            formatted_traceback = self._format_traceback()
            if formatted_traceback:
                logger.error("general_error_traceback", traceback=formatted_traceback)

    def _build_log_message(self) -> str:
        """
        Build a structured log message for the error.

        Returns:
            str: The log message.
        """
        log_message = f"[Error Code: {self.code}] {self.message}"
        if self.details:
            log_message += f" | Details: {self.details}"
        if self.context:
            log_message += f" | Context: {self.context}"
        return log_message

    @staticmethod
    def _format_traceback() -> str | None:
        """
        Format the current traceback if available.

        Returns:
            Optional[str]: The formatted traceback, or None if not available.
        """
        return "".join(traceback.format_exc()) if traceback.format_exc().strip() else None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the error to a dictionary representation.

        Returns:
            Dict[str, Any]: The dictionary representation of the error.
        """
        error_data = {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "context": self.context,
        }
        if self.log_traceback:
            formatted_traceback = self._format_traceback()
            if formatted_traceback:
                error_data["traceback"] = formatted_traceback
        return error_data

    def __str__(self) -> str:
        """
        Return the string representation of the error.

        Returns:
            str: The error message.
        """
        return self._build_log_message()
