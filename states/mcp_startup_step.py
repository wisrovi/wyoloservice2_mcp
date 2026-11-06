"""MCP server startup step in WPipe pipeline."""

from typing import Any, Dict
from wpipe import step

from wyolo_mcp.utils.credentials import CONFIG_FILE


@step(name="mcp_startup_step", version="v1.0", tags=["mcp", "init"])
def mcp_startup_step(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate MCP execution environment and configuration prerequisites.

    Args:
        input_data (Dict[str, Any]): Context dictionary.

    Returns:
        Dict[str, Any]: Context dictionary with credentials status.
    """
    input_data["config_file_exists"] = CONFIG_FILE.exists()
    input_data["mcp_ready"] = True
    return input_data
