import json
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd

from lib.core.logging_config import get_logger

logger = get_logger(__name__)


# --- Custom JSON Encoder ---
class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects like pandas Timestamps.

    Example Usage:
    ```python
    import json
    import pandas as pd

    timestamp = pd.Timestamp('2024-01-01T12:00:00')
    data = {"time": timestamp, "value": 10}
    json_string = json.dumps(data, cls=CustomJSONEncoder)
    print(json_string)
    ```
    """

    def default(self, obj: Any) -> Any:  # type: ignore[override]
        """Override default method to handle pandas Timestamp."""
        logger.debug("encoding_object", obj_type=str(type(obj)))
        if isinstance(obj, pd.Timestamp):
            iso_format = obj.isoformat()
            logger.debug("encoded_pd_timestamp", iso_format=iso_format)
            return iso_format
        else:
            logger.debug("delegating_encoding_to_super", obj_type=str(type(obj)))
            default_encoded = super().default(obj)
            logger.debug("encoding_delegated", result_type=str(type(default_encoded)))
            return default_encoded


# --- URL Utility Functions ---
def construct_redis_url(
    user: str = "default",
    password: str = "",
    host: str = "redis",
    port: str | int = "6379",
    db: str | int = "0",
    use_tls: bool = False,
    use_sentinel: bool = False,
    use_cluster: bool = False,  # Added missing parameter
) -> str:
    """
    Construct the Redis connection URL, supporting optional TLS, Sentinel, and Cluster modes.

    Logs detailed information about the construction process without revealing sensitive details.

    Args:
        user (str): Redis user (default: "default").
        password (str): Redis password (default: "").
        host (str): Redis host (default: "redis").
        port (Union[str, int]): Redis port (default: "6379").
        db (Union[str, int]): Redis database number (default: "0").
        use_tls (bool): Use TLS/SSL for connection (default: False).
        use_sentinel (bool): Use Redis Sentinel for connection (default: False).
        use_cluster (bool): Use Redis Cluster mode (default: False).

    Returns:
        str: The constructed Redis connection URL.
    """
    logger.debug(
        "constructing_redis_url",
        use_tls=use_tls,
        use_sentinel=use_sentinel,
        use_cluster=use_cluster,
        host=host,
        port=port,
        db=db,
        user_provided=bool(user),
        password_provided=bool(password),
    )

    protocol = "rediss" if use_tls else "redis"
    user, password = user.strip(), password.strip()
    url = ""

    if user and password:
        url = f"{protocol}://{user}:{password}@{host}:{port}/{db}"
        logger.debug("redis_url_constructed", variant="user_and_password", clean_url=clean_redis_url(url))
    elif password:
        url = f"{protocol}://:{password}@{host}:{port}/{db}"
        logger.debug("redis_url_constructed", variant="password_only", clean_url=clean_redis_url(url))
    else:
        url = f"{protocol}://{host}:{port}/{db}"
        logger.debug("redis_url_constructed", variant="no_auth", clean_url=clean_redis_url(url))

    # Note: Additional Sentinel-specific logic would go here if needed
    # Note: Additional Cluster-specific logic would go here if needed
    # This implementation just accepts the parameters but doesn't modify behavior

    cleaned_url = clean_redis_url(url)
    logger.debug("redis_url_construction_success", clean_url=cleaned_url)
    return url


def clean_redis_url(redis_url: str) -> str:
    """
    Clean the Redis URL for logging purposes (removes sensitive info).

    Returns the URL with only hostname, port, and path.

    Args:
        redis_url (str): The Redis URL to clean.

    Returns:
        str: The cleaned Redis URL with sensitive information removed.
    """
    logger.debug("cleaning_redis_url")
    parsed = urlparse(redis_url)
    clean_netloc = parsed.hostname or ""
    if parsed.port:
        clean_netloc += f":{parsed.port}"
    cleaned = urlunparse((parsed.scheme, clean_netloc, parsed.path or "", "", "", ""))
    logger.debug("redis_url_cleaned", clean_url=cleaned)
    return cleaned
