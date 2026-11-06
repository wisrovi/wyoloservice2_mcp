"""Cluster credentials storage and retrieval utilities."""

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE: Path = Path.home() / ".wyolo_mcp_config.json"


def get_credentials() -> Dict[str, str]:
    """Retrieve saved cluster credentials from the configuration file.

    Returns:
        Dict[str, str]: Dictionary containing api_url, control_host, cifs_user, cifs_pass.

    Raises:
        ValueError: If configuration file does not exist.
    """
    if not CONFIG_FILE.exists():
        raise ValueError("Cluster credentials not configured. Please use the 'set_cluster_credentials' tool first.")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def set_cluster_credentials(ip: str, cifs_user: str, cifs_pass: str) -> Dict[str, Any]:
    """Save the cluster IP and Samba CIFS credentials to a local configuration file.

    Args:
        ip: Physical or local network IP of the control host.
        cifs_user: CIFS Samba username.
        cifs_pass: CIFS Samba password.

    Returns:
        Dict[str, Any]: Success status and destination file path or error message.
    """
    config_data = {
        "api_url": f"http://{ip}:23442",
        "control_host": ip,
        "cifs_user": cifs_user,
        "cifs_pass": cifs_pass,
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        return {"success": True, "message": f"Credentials saved successfully to {CONFIG_FILE}"}
    except Exception as exc:
        return {"error": f"Failed to save credentials: {exc}"}
