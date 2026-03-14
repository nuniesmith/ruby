import csv
import datetime
import decimal
import enum
import io
import json
import uuid
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles additional Python types:
    - datetime: Converted to ISO format
    - date: Converted to ISO format
    - UUID: Converted to string
    - Decimal: Converted to float
    - Enum: Converted to value
    - Path: Converted to string
    - Sets: Converted to lists
    """

    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, datetime.time):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()

        return super().default(obj)


def to_json(obj: Any, pretty: bool = False) -> str:
    """
    Convert an object to a JSON string.

    Args:
        obj: Object to convert
        pretty: Whether to format the JSON for readability

    Returns:
        JSON string
    """
    indent = 2 if pretty else None
    return json.dumps(obj, cls=JSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Convert a JSON string to an object.

    Args:
        json_str: JSON string to convert

    Returns:
        Parsed object

    Raises:
        ValueError: If the JSON is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def to_dict(obj: Any) -> Any:
    """
    Convert an object to a dictionary.

    If the object has a to_dict method, it will be used.
    Otherwise, attempt to convert the object to a dict.

    Args:
        obj: Object to convert

    Returns:
        Dictionary representation
    """
    if obj is None:
        return {}

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_dict(item) for item in obj]

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Attempt to convert to dict using __dict__
    if hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}

    # As a last resort, convert to string
    return str(obj)


def from_dict(data: dict[str, Any], cls: type[T]) -> T:
    """
    Create an instance of a class from a dictionary.

    If the class has a from_dict method, it will be used.
    Otherwise, a new instance will be created passing the dict as kwargs.

    Args:
        data: Dictionary to convert
        cls: Class to create

    Returns:
        Instance of cls
    """
    if data is None:
        return None

    if hasattr(cls, "from_dict") and callable(cls.from_dict):  # type: ignore[attr-defined]
        return cls.from_dict(data)  # type: ignore[attr-defined]

    return cls(**data)


def to_csv(data: list[dict[str, Any]], dialect: str = "excel") -> str:
    """
    Convert a list of dictionaries to a CSV string.

    Args:
        data: List of dictionaries to convert
        dialect: CSV dialect to use

    Returns:
        CSV string
    """
    if not data:
        return ""

    output = io.StringIO()
    fieldnames = data[0].keys()

    writer = csv.DictWriter(output, fieldnames=fieldnames, dialect=dialect)
    writer.writeheader()
    writer.writerows(data)

    return output.getvalue()


def from_csv(csv_str: str, dialect: str = "excel") -> list[dict[str, Any]]:
    """
    Convert a CSV string to a list of dictionaries.

    Args:
        csv_str: CSV string to convert
        dialect: CSV dialect to use

    Returns:
        List of dictionaries
    """
    if not csv_str:
        return []

    input_file = io.StringIO(csv_str)
    reader = csv.DictReader(input_file, dialect=dialect)

    return list(reader)


def is_jsonable(obj: Any) -> bool:
    """
    Check if an object can be serialized to JSON.

    Args:
        obj: Object to check

    Returns:
        True if the object can be serialized to JSON, False otherwise
    """
    try:
        json.dumps(obj, cls=JSONEncoder)
        return True
    except Exception:
        return False


def mask_sensitive_data(data: dict[str, Any], sensitive_fields: list[str]) -> dict[str, Any]:
    """
    Mask sensitive data in a dictionary.

    Args:
        data: Dictionary containing data to mask
        sensitive_fields: List of field names to mask

    Returns:
        Dictionary with sensitive data masked
    """
    if not data or not sensitive_fields:
        return data

    result: dict[str, Any] = {}

    for key, value in data.items():
        if key in sensitive_fields:
            result[key] = "****"
        elif isinstance(value, dict):
            result[key] = mask_sensitive_data(value, sensitive_fields)
        elif isinstance(value, list):
            result[key] = [
                mask_sensitive_data(item, sensitive_fields) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value

    return result


def serialize_file(file_path: str | Path, binary: bool = False) -> dict[str, Any]:
    """
    Serialize a file to a dictionary.

    Args:
        file_path: Path to the file
        binary: Whether to read the file in binary mode

    Returns:
        Dictionary containing file metadata and content
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mode = "rb" if binary else "r"
    with open(path, mode) as f:
        content = f.read()

    return {
        "name": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "modified": datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "content": content,
        "binary": binary,
    }


def deserialize_file(data: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """
    Deserialize a file from a dictionary.

    Args:
        data: Dictionary containing file data
        output_dir: Directory to write the file to (optional)

    Returns:
        Path to the created file
    """
    output_path = Path(output_dir) / data["name"] if output_dir else Path(data["path"])

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if data.get("binary", False) else "w"
    with open(output_path, mode) as f:
        f.write(data["content"])

    return output_path


def deep_merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    The second dictionary takes precedence over the first.

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator to use for keys

    Returns:
        Flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """
    Unflatten a dictionary with keys separated by a separator.

    Args:
        d: Flattened dictionary
        sep: Separator used for keys

    Returns:
        Nested dictionary
    """
    result: dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)

        # Navigate through the parts
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value at the final part
        current[parts[-1]] = value

    return result
