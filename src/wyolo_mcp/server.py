import os
import httpx
import psutil
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP

# Intentar importar la nueva librería de data_prep (si está instalada en el mismo entorno)
try:
    from wyolo_data_prep import check_yolo_dataset
    HAS_DATA_PREP = True
except ImportError:
    HAS_DATA_PREP = False

# Initialize FastMCP server
mcp = FastMCP("NeuralForgeAI-MCP")

# Environment configuration
API_URL = os.getenv("NEURALFORGE_API_URL", "http://192.168.10.252:23442")

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
            response = await client.get(f"{API_URL}/health")
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
            import yaml
            
            # Generate the YAML config expected by NeuralForgeAI
            yaml_config = {
                "study_name": config.name,
                "executor": "yolo_v8",
                "dataset": config.dataset,
                "hyperparameters": {
                    "epochs": config.epochs,
                    "models": config.models,
                    "batch_sizes": config.batch_sizes
                }
            }
            yaml_content = yaml.dump(yaml_config)
            
            # The API expects a multipart form-data upload with a file named 'config_file'
            files = {
                "config_file": (f"{config.name}.yaml", yaml_content.encode("utf-8"), "application/x-yaml")
            }
            data = {
                "mode": "public",
                "priority": "medium"
            }
            response = await client.post(f"{API_URL}/train", files=files, data=data)
            
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

@mcp.tool()
def get_host_metrics() -> Dict[str, Any]:
    """
    Get CPU, RAM, and Disk metrics of the host machine running the MCP server.
    Useful for diagnosing if the cluster manager is out of resources.
    """
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
    }

@mcp.tool()
def validate_dataset_advanced(yaml_path: str) -> Dict[str, Any]:
    """
    Validates a YOLO dataset profoundly using the new wyoloservice2_data_prep library.
    Checks inside the YAML for structural correctness.
    """
    if not HAS_DATA_PREP:
        return {"error": "wyoloservice2_data_prep library is not installed in the MCP environment."}
    
    return check_yolo_dataset(yaml_path)

import sys

def main():
    print("Starting NeuralForgeAI MCP server on stdio...", file=sys.stderr)
    # Run the FastMCP server via stdio (standard for Claude Desktop / MCP clients)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
