import os
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("NeuralForgeAI-MCP")

# Environment configuration
API_URL = os.getenv("NEURALFORGE_API_URL", "http://localhost:8000")

class TrainingConfig(BaseModel):
    name: str = Field(..., description="Name of the study")
    dataset: str = Field(..., description="Absolute path to the dataset.yaml file")
    epochs: int = Field(100, description="Number of epochs")
    models: List[str] = Field(default=["yolov8n.pt"], description="List of models to try")
    batch_sizes: List[int] = Field(default=[16], description="List of batch sizes")

@mcp.tool()
async def get_cluster_status() -> Dict[str, Any]:
    """
    Get the overall status of the NeuralForgeAI cluster, including active celery workers 
    and general health of the API.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/")
            if response.status_code == 200:
                return {"status": "online", "api_details": response.json()}
            return {"status": "error", "message": f"API returned {response.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}

@mcp.tool()
async def get_study_details(study_id: str) -> Dict[str, Any]:
    """
    Get detailed telemetry and status of a specific YOLO training study.
    Returns progress, active invoker, and current trial metrics.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/study/{study_id}")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Study not found or API error: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

@mcp.tool()
async def cancel_study(study_id: str) -> Dict[str, Any]:
    """
    Cancel a running training study by its ID. This will stop the active trials 
    and terminate the executor containers.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_URL}/study/{study_id}/cancel")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to cancel: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

@mcp.tool()
async def launch_training(config: TrainingConfig) -> Dict[str, Any]:
    """
    Launch a new YOLO hyperparameter optimization study on the cluster.
    """
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "name": config.name,
                "dataset": config.dataset,
                "epochs": config.epochs,
                "models": config.models,
                "batch_sizes": config.batch_sizes
            }
            response = await client.post(f"{API_URL}/train/yolo", json=payload)
            if response.status_code == 200:
                return {"success": True, "details": response.json()}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"error": str(e)}

@mcp.tool()
def check_dataset_path(path: str) -> Dict[str, Any]:
    """
    Locally verify if a dataset path exists on the host machine and list its contents.
    Useful for preventing FileNotFoundError before launching a training.
    """
    if not os.path.exists(path):
        return {"exists": False, "message": f"Path '{path}' does not exist on the MCP host."}
    
    is_dir = os.path.isdir(path)
    contents = os.listdir(path) if is_dir else []
    
    return {
        "exists": True,
        "is_directory": is_dir,
        "contents": contents[:20] if contents else [],
        "message": "Path found. Showing up to 20 items." if contents else "Path found."
    }

def main():
    # Run the FastMCP server via stdio (standard for Claude Desktop / MCP clients)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
