import math
import statistics

from lib.core.exceptions.base import ValidationException


def safe_divide(numerator: int | float, denominator: int | float, default: int | float = 0) -> int | float:
    """
    Safely divide two numbers, returning default if denominator is 0.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is 0

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def round_to_precision(value: float, precision: int) -> float:
    """
    Round a value to a specific number of decimal places.

    Args:
        value: Value to round
        precision: Number of decimal places

    Returns:
        Rounded value
    """
    multiplier = 10**precision
    return round(value * multiplier) / multiplier


def round_to_tick(value: float, tick_size: float) -> float:
    """
    Round a value to the nearest tick.

    Args:
        value: Value to round
        tick_size: Tick size

    Returns:
        Value rounded to the nearest tick
    """
    return round(value / tick_size) * tick_size


def truncate_to_precision(value: float, precision: int) -> float:
    """
    Truncate a value to a specific number of decimal places.

    Args:
        value: Value to truncate
        precision: Number of decimal places

    Returns:
        Truncated value
    """
    multiplier = 10**precision
    return math.floor(value * multiplier) / multiplier


def truncate_to_tick(value: float, tick_size: float) -> float:
    """
    Truncate a value to the nearest tick.

    Args:
        value: Value to truncate
        tick_size: Tick size

    Returns:
        Value truncated to the nearest tick
    """
    return math.floor(value / tick_size) * tick_size


def percent_change(initial: float, final: float) -> float:
    """
    Calculate the percentage change between two values.

    Args:
        initial: Initial value
        final: Final value

    Returns:
        Percentage change
    """
    if initial == 0:
        return 0
    return (final - initial) / abs(initial) * 100


def log_returns(prices: list[float]) -> list[float]:
    """
    Calculate logarithmic returns from a list of prices.

    Args:
        prices: List of prices

    Returns:
        List of logarithmic returns
    """
    if len(prices) < 2:
        return []

    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def simple_returns(prices: list[float]) -> list[float]:
    """
    Calculate simple returns from a list of prices.

    Args:
        prices: List of prices

    Returns:
        List of simple returns
    """
    if len(prices) < 2:
        return []

    return [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]


def annualized_returns(returns: list[float], periods_per_year: int) -> float:
    """
    Calculate annualized returns from a list of returns.

    Args:
        returns: List of returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized return
    """
    if not returns:
        return 0

    total_return: float = 1
    for r in returns:
        total_return *= 1 + r

    # Convert to annualized return
    return (total_return ** (periods_per_year / len(returns))) - 1


def annualized_volatility(returns: list[float], periods_per_year: int) -> float:
    """
    Calculate annualized volatility from a list of returns.

    Args:
        returns: List of returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility
    """
    if not returns:
        return 0

    std_dev = statistics.stdev(returns)
    return std_dev * math.sqrt(periods_per_year)


def sharpe_ratio(returns: list[float], risk_free_rate: float, periods_per_year: int) -> float:
    """
    Calculate the Sharpe ratio from a list of returns.

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0

    # Convert annual risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = [r - rf_per_period for r in returns]
    mean_excess_return = statistics.mean(excess_returns)

    if len(returns) < 2:
        return 0

    std_dev = statistics.stdev(returns)
    if std_dev == 0:
        return 0

    return (mean_excess_return / std_dev) * math.sqrt(periods_per_year)


def sortino_ratio(returns: list[float], risk_free_rate: float, periods_per_year: int) -> float:
    """
    Calculate the Sortino ratio from a list of returns.

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if not returns:
        return 0

    # Convert annual risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = [r - rf_per_period for r in returns]
    mean_excess_return = statistics.mean(excess_returns)

    # Calculate downside deviation (standard deviation of negative returns)
    downside_returns = [r for r in excess_returns if r < 0]
    if not downside_returns:
        return float("inf")  # No downside risk

    downside_deviation = math.sqrt(sum(r**2 for r in downside_returns) / len(downside_returns))

    if downside_deviation == 0:
        return float("inf")  # No downside risk

    return (mean_excess_return / downside_deviation) * math.sqrt(periods_per_year)


def maximum_drawdown(prices: list[float]) -> float:
    """
    Calculate the maximum drawdown from a list of prices.

    Args:
        prices: List of prices

    Returns:
        Maximum drawdown as a percentage
    """
    if not prices:
        return 0

    max_drawdown: float = 0
    peak = prices[0]

    for price in prices:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown * 100


def calmar_ratio(returns: list[float], prices: list[float], periods_per_year: int) -> float:
    """
    Calculate the Calmar ratio.

    Args:
        returns: List of returns
        prices: List of prices
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if not returns or not prices:
        return 0

    ann_return = annualized_returns(returns, periods_per_year)
    max_dd = maximum_drawdown(prices) / 100  # Convert to decimal

    if max_dd == 0:
        return float("inf")  # No drawdown

    return ann_return / max_dd


def value_at_risk(returns: list[float], confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) from a list of returns.

    Args:
        returns: List of returns
        confidence: Confidence level (default 0.95)

    Returns:
        Value at Risk
    """
    if not returns:
        return 0

    # Sort returns
    sorted_returns = sorted(returns)

    # Find the index at the specified confidence level
    index = int(len(sorted_returns) * (1 - confidence))

    # Ensure index is within bounds
    index = max(0, min(index, len(sorted_returns) - 1))

    return -sorted_returns[index]


def conditional_value_at_risk(returns: list[float], confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) from a list of returns.

    Args:
        returns: List of returns
        confidence: Confidence level (default 0.95)

    Returns:
        Conditional Value at Risk
    """
    if not returns:
        return 0

    # Calculate VaR
    var = value_at_risk(returns, confidence)

    # Find returns below VaR
    tail_returns = [r for r in returns if r <= -var]

    if not tail_returns:
        return var

    # Calculate mean of tail returns
    return -statistics.mean(tail_returns)


def moving_average(data: list[float], window: int) -> list[float]:
    """
    Calculate the simple moving average.

    Args:
        data: List of values
        window: Moving average window size

    Returns:
        List of moving averages
    """
    if window <= 0:
        raise ValidationException(message="Window size must be positive", details={"window": window})

    if len(data) < window:
        return []

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        result.append(sum(window_data) / window)

    return result


def exponential_moving_average(data: list[float], window: int) -> list[float]:
    """
    Calculate the exponential moving average.

    Args:
        data: List of values
        window: Window size for EMA calculation

    Returns:
        List of exponential moving averages
    """
    if window <= 0:
        raise ValidationException(message="Window size must be positive", details={"window": window})

    if not data:
        return []

    # Calculate smoothing factor
    alpha = 2 / (window + 1)

    # Initialize EMA with first value
    ema = [data[0]]

    # Calculate EMA for the rest of the values
    for i in range(1, len(data)):
        ema.append(data[i] * alpha + ema[i - 1] * (1 - alpha))

    return ema


def weighted_moving_average(data: list[float], weights: list[float]) -> list[float]:
    """
    Calculate the weighted moving average.

    Args:
        data: List of values
        weights: List of weights (should sum to 1)

    Returns:
        List of weighted moving averages
    """
    if not weights:
        raise ValidationException(message="Weights list cannot be empty", details={"weights": weights})

    if abs(sum(weights) - 1.0) > 1e-10:
        raise ValidationException(message="Weights must sum to 1", details={"weights": weights, "sum": sum(weights)})

    window = len(weights)
    if len(data) < window:
        return []

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        weighted_sum = sum(w * v for w, v in zip(weights, window_data, strict=False))
        result.append(weighted_sum)

    return result


def bollinger_bands(
    data: list[float], window: int, num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate Bollinger Bands.

    Args:
        data: List of values
        window: Moving average window size
        num_std: Number of standard deviations for the bands

    Returns:
        Tuple of (middle band, upper band, lower band)
    """
    if window <= 1:
        raise ValidationException(message="Window size must be greater than 1", details={"window": window})

    if len(data) < window:
        return [], [], []

    # Calculate middle band (simple moving average)
    middle_band = moving_average(data, window)

    # Calculate standard deviations for each window
    std_devs = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        std_devs.append(statistics.stdev(window_data))

    # Calculate upper and lower bands
    upper_band = [mid + num_std * std for mid, std in zip(middle_band, std_devs, strict=False)]
    lower_band = [mid - num_std * std for mid, std in zip(middle_band, std_devs, strict=False)]

    return middle_band, upper_band, lower_band


def relative_strength_index(data: list[float], window: int) -> list[float]:
    """
    Calculate the Relative Strength Index (RSI).

    Args:
        data: List of values
        window: RSI window size

    Returns:
        List of RSI values
    """
    if window < 2:
        raise ValidationException(message="Window size must be at least 2", details={"window": window})

    if len(data) <= window:
        return []

    # Calculate price changes
    changes = [data[i] - data[i - 1] for i in range(1, len(data))]

    # Initialize lists for gains and losses
    gains = [max(0, change) for change in changes]
    losses = [max(0, -change) for change in changes]

    # Calculate initial average gain and loss
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window

    # Calculate RSI values
    rsi_values: list[float] = []

    for i in range(window, len(changes)):
        # Update average gain and loss
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        # Calculate RS and RSI
        if avg_loss == 0:
            rsi: float = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi_values.append(rsi)

    return rsi_values


def macd(
    data: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate the Moving Average Convergence Divergence (MACD).

    Args:
        data: List of values
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    if fast_period >= slow_period:
        raise ValidationException(
            message="Fast period must be less than slow period",
            details={"fast_period": fast_period, "slow_period": slow_period},
        )

    if len(data) <= slow_period:
        return [], [], []

    # Calculate fast and slow EMAs
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)

    # Calculate MACD line (fast EMA - slow EMA)
    macd_line = [fast - slow for fast, slow in zip(fast_ema[slow_period - fast_period :], slow_ema, strict=False)]

    # Calculate signal line (EMA of MACD line)
    signal_line = exponential_moving_average(macd_line, signal_period)

    # Calculate histogram (MACD line - signal line)
    histogram = [macd - signal for macd, signal in zip(macd_line[signal_period - 1 :], signal_line, strict=False)]

    # Align the lengths
    macd_line = macd_line[signal_period - 1 :]

    return macd_line, signal_line, histogram


def average_true_range(highs: list[float], lows: list[float], closes: list[float], window: int) -> list[float]:
    """
    Calculate the Average True Range (ATR).

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        window: ATR window size

    Returns:
        List of ATR values
    """
    if window < 1:
        raise ValidationException(message="Window size must be positive", details={"window": window})

    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValidationException(
            message="Input lists must have the same length",
            details={"highs_len": len(highs), "lows_len": len(lows), "closes_len": len(closes)},
        )

    if len(highs) <= 1:
        return []

    # Calculate true ranges
    true_ranges = []

    for i in range(1, len(highs)):
        # Current high - current low
        range1 = highs[i] - lows[i]
        # Current high - previous close
        range2 = abs(highs[i] - closes[i - 1])
        # Current low - previous close
        range3 = abs(lows[i] - closes[i - 1])

        true_ranges.append(max(range1, range2, range3))

    if len(true_ranges) < window:
        return []

    # Calculate initial ATR (simple average of first window true ranges)
    atr = [sum(true_ranges[:window]) / window]

    # Calculate subsequent ATR values
    for i in range(window, len(true_ranges)):
        atr.append((atr[-1] * (window - 1) + true_ranges[i]) / window)

    return atr
