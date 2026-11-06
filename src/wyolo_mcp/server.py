"""NeuralForgeAI MCP Server module for Antigravity and Claude agents."""

import os
import sys
from typing import Any, Callable, Dict, List
from mcp.server.fastmcp import FastMCP

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dto.training_config_dto import TrainingConfig
from wyolo_mcp.utils.artifact_tools import (
    download_mlflow_study_artifacts,
    download_study_artifact,
)
from wyolo_mcp.utils.cluster_tools import (
    get_cluster_status,
    preflight_check_node,
    recommend_training_node,
)
from wyolo_mcp.utils.credentials import (
    set_cluster_credentials,
)
from wyolo_mcp.utils.dataset_tools import (
    check_dataset_path,
    validate_dataset_advanced,
)
from wyolo_mcp.utils.estimator_tools import (
    estimate_study_duration,
)
from wyolo_mcp.utils.invoker_tools import (
    manage_invoker_queues,
    trigger_broadcast_docker_pull,
)
from wyolo_mcp.utils.study_tools import (
    cancel_study,
    get_study_details,
)
from wyolo_mcp.utils.test_launcher import (
    launch_private_test_training,
)
from wyolo_mcp.utils.training_tools import (
    generate_training_yaml,
    launch_training,
)
from wyolo_mcp.utils.xai_tools import (
    get_study_heatmaps_summary,
    get_study_xai_metrics,
)

mcp: FastMCP = FastMCP("NeuralForgeAI-MCP")

# Register all modular tools with FastMCP
ALL_TOOLS: List[Callable[..., Any]] = [
    set_cluster_credentials,
    get_cluster_status,
    get_study_details,
    cancel_study,
    generate_training_yaml,
    launch_training,
    check_dataset_path,
    validate_dataset_advanced,
    manage_invoker_queues,
    launch_private_test_training,
    trigger_broadcast_docker_pull,
    download_mlflow_study_artifacts,
    recommend_training_node,
    estimate_study_duration,
    preflight_check_node,
    get_study_xai_metrics,
    get_study_heatmaps_summary,
    download_study_artifact,
]

for tool_fn in ALL_TOOLS:
    mcp.tool()(tool_fn)


def main() -> None:
    """Entry point for executing the FastMCP stdio server."""
    print("Starting NeuralForgeAI MCP server on stdio...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
