"""
System Information Utilities.

This module provides functions for getting system information and
detecting container environments.
"""

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


def get_system_info() -> dict[str, Any]:
    """
    Get basic system information for diagnostics.

    Returns:
        Dict[str, Any]: System information dictionary
    """
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "current_directory": str(Path.cwd()),
        "pid": os.getpid(),
    }


def detect_container_environment() -> dict[str, Any]:
    """
    Detect if running in a container environment.

    Returns:
        Dict[str, Any]: Container detection results
    """
    results: dict[str, Any] = {"is_docker": False, "is_kubernetes": False, "container_id": None, "container_info": {}}

    # Check for Docker
    if Path("/.dockerenv").exists():
        results["is_docker"] = True

    # Check for Docker cgroup
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content:
                results["is_docker"] = True

                # Try to extract container ID
                import re

                container_id_match = re.search(r"[0-9a-f]{64}", content)
                if container_id_match:
                    results["container_id"] = container_id_match.group(0)
    except (FileNotFoundError, PermissionError, ImportError):
        pass

    # Check for Kubernetes
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        results["is_kubernetes"] = True

        # Get additional info from env vars
        k8s_vars = ["KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT", "HOSTNAME", "KUBERNETES_PORT"]
        results["container_info"] = {key: os.environ.get(key) for key in k8s_vars if key in os.environ}

    return results
