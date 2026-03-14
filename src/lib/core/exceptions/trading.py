from lib.core.exceptions.base import FrameworkException


class TradingException(FrameworkException):
    """
    Base exception class for trading-related errors.

    This includes:
    - Order errors
    - Strategy errors
    - Execution errors
    - Market condition errors
    """

    def __init__(self, message: str = "", code: str = "", details: dict | None = None):
        """
        Initialize a new trading exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
        """
        if details is None:
            details = {}
        code = code or "TRADING_ERROR"
        message = message or "A trading error occurred"
        super().__init__(message=message, code=code, details=details)


class OrderException(TradingException):
    """
    Exception raised for order-related errors.

    This includes:
    - Invalid order parameters
    - Order execution failures
    - Order status errors
    """

    def __init__(self, message: str = "", code: str = "", details: dict | None = None, order_id: str = ""):
        """
        Initialize a new order exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            order_id: ID of the order that raised the exception
        """
        if details is None:
            details = {}
        code = code or "ORDER_ERROR"

        if order_id:
            details["order_id"] = order_id

        message = message or (f"Order error for order '{order_id}'" if order_id else "Order error")

        super().__init__(message=message, code=code, details=details)


class StrategyException(TradingException):
    """
    Exception raised for strategy-related errors.

    This includes:
    - Strategy execution errors
    - Signal generation errors
    - Parameter validation errors
    """

    def __init__(self, message: str = "", code: str = "", details: dict | None = None, strategy_id: str = ""):
        """
        Initialize a new strategy exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            strategy_id: ID of the strategy that raised the exception
        """
        if details is None:
            details = {}
        code = code or "STRATEGY_ERROR"

        if strategy_id:
            details["strategy_id"] = strategy_id

        if strategy_id:
            message = message or f"Strategy error for strategy '{strategy_id}'"
        else:
            message = message or "Strategy error"

        super().__init__(message=message, code=code, details=details)


class ExecutionException(TradingException):
    """
    Exception raised for trade execution errors.

    This includes:
    - Execution service errors
    - Exchange connectivity errors
    - Order execution errors
    """

    def __init__(self, message: str = "", code: str = "", details: dict | None = None, exchange: str = ""):
        """
        Initialize a new execution exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            exchange: Name of the exchange where the execution was attempted
        """
        if details is None:
            details = {}
        code = code or "EXECUTION_ERROR"

        if exchange:
            details["exchange"] = exchange

        message = message or (f"Execution error on exchange '{exchange}'" if exchange else "Execution error")

        super().__init__(message=message, code=code, details=details)


class MarketConditionException(TradingException):
    """
    Exception raised for adverse market condition errors.

    This includes:
    - Market closed
    - Extreme volatility
    - Liquidity issues
    """

    def __init__(
        self, message: str = "", code: str = "", details: dict | None = None, market: str = "", condition: str = ""
    ):
        """
        Initialize a new market condition exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            market: Market identifier
            condition: Specific market condition
        """
        if details is None:
            details = {}
        code = code or "MARKET_CONDITION_ERROR"

        if market:
            details["market"] = market
        if condition:
            details["condition"] = condition

        if market and condition:
            message = message or f"Adverse market condition: {condition} on {market}"
        elif condition:
            message = message or f"Adverse market condition: {condition}"
        elif market:
            message = message or f"Adverse market condition on {market}"
        else:
            message = message or "Adverse market condition"

        super().__init__(message=message, code=code, details=details)


class InsufficientFundsException(TradingException):
    """
    Exception raised when there are insufficient funds for a trade.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "",
        details: dict | None = None,
        required: float = 0.0,
        available: float = 0.0,
        asset: str = "",
    ):
        """
        Initialize a new insufficient funds exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            required: Required amount
            available: Available amount
            asset: Asset identifier
        """
        if details is None:
            details = {}
        code = code or "INSUFFICIENT_FUNDS"

        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available
        if asset:
            details["asset"] = asset

        if asset and required is not None and available is not None:
            message = message or f"Insufficient {asset}: required {required}, available {available}"
        elif required is not None and available is not None:
            message = message or f"Insufficient funds: required {required}, available {available}"
        elif asset:
            message = message or f"Insufficient {asset}"
        else:
            message = message or "Insufficient funds"

        super().__init__(message=message, code=code, details=details)


class PositionLimitException(TradingException):
    """
    Exception raised when a position limit would be exceeded by a trade.
    """

    def __init__(
        self,
        message: str = "",
        code: str = "",
        details: dict | None = None,
        limit: float = 0.0,
        attempted: float = 0.0,
        asset: str = "",
    ):
        """
        Initialize a new position limit exception.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            details: Additional details about the error
            limit: Position limit
            attempted: Attempted position size
            asset: Asset identifier
        """
        if details is None:
            details = {}
        code = code or "POSITION_LIMIT_EXCEEDED"

        if limit is not None:
            details["limit"] = limit
        if attempted is not None:
            details["attempted"] = attempted
        if asset:
            details["asset"] = asset

        if asset and limit is not None and attempted is not None:
            message = message or f"Position limit exceeded for {asset}: limit {limit}, attempted {attempted}"
        elif limit is not None and attempted is not None:
            message = message or f"Position limit exceeded: limit {limit}, attempted {attempted}"
        elif asset:
            message = message or f"Position limit exceeded for {asset}"
        else:
            message = message or "Position limit exceeded"

        super().__init__(message=message, code=code, details=details)
