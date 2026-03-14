"""
Permission and File System Utilities.

This module provides functions for checking permissions on files and directories.
"""

import os
import stat
from pathlib import Path
from typing import Any


def verify_permissions(path: str | Path | None, write_access: bool = False) -> dict[str, Any]:
    """
    Verify permissions on a path.

    Args:
        path: Path to verify
        write_access: Whether to check for write access

    Returns:
        Dict[str, Any]: Permission information
    """
    if path is None:
        return {"path": None, "exists": False, "errors": ["Path is None"]}

    path_obj = Path(path) if isinstance(path, str) else path
    result: dict[str, Any] = {"path": str(path_obj), "exists": path_obj.exists(), "errors": []}

    if not path_obj.exists():
        result["errors"].append("Path does not exist")

        # Check if parent exists and is writable
        parent = path_obj.parent
        result["parent_exists"] = parent.exists()
        if write_access:
            result["parent_writable"] = parent.exists() and os.access(parent, os.W_OK)
            if parent.exists() and not os.access(parent, os.W_OK):
                result["errors"].append("Parent directory is not writable")

        return result

    # Check if it's a file or directory
    result["is_file"] = path_obj.is_file()
    result["is_dir"] = path_obj.is_dir()

    # Check permissions
    result["readable"] = os.access(path_obj, os.R_OK)

    if write_access:
        result["writable"] = os.access(path_obj, os.W_OK)
        if not result["writable"]:
            result["errors"].append("Path is not writable")

    if not result["readable"]:
        result["errors"].append("Path is not readable")

    # Get detailed stats
    try:
        stat_info = os.stat(path_obj)
        result["permissions"] = stat.filemode(stat_info.st_mode)
        result["uid"] = stat_info.st_uid
        result["gid"] = stat_info.st_gid
        if path_obj.is_file():
            result["size"] = stat_info.st_size
        else:
            result["size"] = None
    except Exception as e:
        result["errors"].append(f"Could not get stat info: {e}")

    return result
