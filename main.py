"""Main entrypoint for wyoloservice2_mcp server (WPipe orchestrated)."""

from typing import Any, Dict
from pipelines import mcp_server_pipeline
from wyolo_mcp.server import main as run_server


def main() -> Dict[str, Any]:
    """Execute startup pipeline and launch FastMCP server on stdio.

    Returns:
        Dict[str, Any]: Context dictionary resulting from pipeline execution.
    """
    pipe = mcp_server_pipeline()
    context = pipe.run({})
    run_server()
    return context


if __name__ == "__main__":
    main()
