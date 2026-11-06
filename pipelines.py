"""WPipe DAG definition module for NeuralForge MCP Server."""

from wpipe import Pipeline

from states.mcp_startup_step import mcp_startup_step

VERSION: str = "0.5.2"


def mcp_server_pipeline() -> Pipeline:
    """Builds and returns the primary MCP server initialization pipeline.

    DAG: mcp_startup_step

    Returns:
        Pipeline: Configured WPipe pipeline instance for MCP initialization.
    """
    pipeline = Pipeline(
        pipeline_name="mcp_server_pipeline",
        pipeline_version=VERSION,
        show_progress=False,
        verbose=False,
        max_retries=0,
    )
    pipeline.set_steps([mcp_startup_step])
    return pipeline
