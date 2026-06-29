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
    task: str = Field(..., description="Task type: MUST be 'detect' (Detection), 'segment' (Segmentation), or 'classify' (Classification)")
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
            
            # Determine the appropriate fitness metric based on task type
            fitness_mapping = {
                "detect": "metrics/mAP50-95(B)",
                "segment": "metrics/mAP50-95(M)",
                "classify": "metrics/accuracy_top1"
            }
            fitness_metric = fitness_mapping.get(config.task, "metrics/mAP50-95(B)")
            
            # Generate the YAML config expected by NeuralForgeAI
            yaml_config = {
                "study_name": config.name,
                "executor": "yolo_v8",
                "dataset": config.dataset,
                "task": config.task,
                "hyperparameters": {
                    "epochs": config.epochs,
                    "models": config.models,
                    "batch_sizes": config.batch_sizes
                },
                "optuna": {
                    "fitness_metric": fitness_metric,
                    "direction": "maximize"
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

import subprocess
import json
import shlex

@mcp.tool()
def check_dataset_path(path: str, control_host: str, cifs_user: str, cifs_pass: str) -> Dict[str, Any]:
    """
    Verify if a dataset path exists on the remote Samba share by spinning up a lightweight Docker container.
    The agent must retrieve the Samba credentials (control_host, cifs_user, cifs_pass) from its memory or ask the user.
    """
    cmd = f"""
    /usr/local/bin/mount-cifs.sh >/dev/null 2>&1
    if [ ! -e "{path}" ]; then
        echo '{{"exists": false, "message": "Path does not exist on the CIFS share."}}'
        exit 0
    fi
    if [ -d "{path}" ]; then
        contents=$(ls -1 "{path}" | head -n 20 | tr '\n' ',' | sed 's/,$//')
        echo '{{"exists": true, "is_directory": true, "contents": "'"$contents"'", "message": "Path found on CIFS."}}'
    else
        echo '{{"exists": true, "is_directory": false, "contents": [], "message": "File found on CIFS."}}'
    fi
    """
    
    docker_cmd = [
        "docker", "run", "--rm", "--privileged",
        "-e", f"CONTROL_HOST={control_host}",
        "-e", f"CIFS_USER={cifs_user}",
        "-e", f"CIFS_PASS={cifs_pass}",
        "wisrovi/train_service:worker_executor_v1.0.0",
        "bash", "-c", cmd
    ]
    
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"error": f"Failed to execute docker check: {str(e)}"}

@mcp.tool()
def validate_dataset_advanced(dataset_path: str, control_host: str, cifs_user: str, cifs_pass: str, task: str = "detect") -> Dict[str, Any]:
    """
    Validates a YOLO dataset structure by running an inspection script inside a Docker container 
    connected to the remote CIFS share. Supports detect/segment (yaml) and classify (directory).
    The agent must retrieve the Samba credentials (control_host, cifs_user, cifs_pass) from its memory or ask the user.
    """
    python_script = f"""
import os
import yaml
import json

path = "{dataset_path}"
task = "{task}"
result = {{"valid": False, "error": "Unknown"}}

if task == "classify":
    if not os.path.isdir(path):
        result["error"] = "For classification, dataset must be a directory"
    else:
        train_dir = os.path.join(path, "train")
        val_dir = os.path.join(path, "val")
        if not os.path.isdir(train_dir):
            result["error"] = "Missing 'train' directory"
        elif not os.path.isdir(val_dir):
            result["error"] = "Missing 'val' directory"
        else:
            classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            result = {{"valid": True, "task": task, "classes": len(classes), "names": classes, "train_path": train_dir}}
else:
    if not os.path.isfile(path):
        result["error"] = "YAML file not found or is not a file"
    else:
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            missing = [f for f in ['train', 'val', 'nc', 'names'] if f not in data]
            if missing:
                result["error"] = f"Missing required fields: {{missing}}"
            else:
                train_path = os.path.join(os.path.dirname(path), data['train']) if not os.path.isabs(data['train']) else data['train']
                if not os.path.exists(train_path):
                    result["error"] = f"Train path does not exist on CIFS: {{train_path}}"
                else:
                    result = {{"valid": True, "task": task, "classes": data['nc'], "names": data['names'], "train_path": train_path}}
        except Exception as e:
            result["error"] = str(e)

print(json.dumps(result))
"""
    
    cmd = f"/usr/local/bin/mount-cifs.sh >/dev/null 2>&1 && python3 -c {shlex.quote(python_script)}"
    
    docker_cmd = [
        "docker", "run", "--rm", "--privileged",
        "-e", f"CONTROL_HOST={control_host}",
        "-e", f"CIFS_USER={cifs_user}",
        "-e", f"CIFS_PASS={cifs_pass}",
        "wisrovi/train_service:worker_executor_v1.0.0",
        "bash", "-c", cmd
    ]
    
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        # Extract the JSON line from stdout (ignoring any other print noise)
        for line in result.stdout.strip().split('\\n'):
            if line.startswith('{"valid"'):
                return json.loads(line)
        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"error": f"Failed to execute docker validation: {str(e)}"}

import sys

def main():
    print("Starting NeuralForgeAI MCP server on stdio...", file=sys.stderr)
    # Run the FastMCP server via stdio (standard for Claude Desktop / MCP clients)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
