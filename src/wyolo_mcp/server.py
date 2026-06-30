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

class TrainingConfig(BaseModel):
    name: str = Field(..., description="Name of the study")
    dataset: str = Field(..., description="Absolute path to the dataset.yaml file")
    task: str = Field(..., description="Task type: MUST be 'detect' (Detection), 'segment' (Segmentation), or 'classify' (Classification)")
    epochs: int = Field(100, description="Number of epochs")
    models: List[str] = Field(default=["yolov8n.pt"], description="List of models to try")
    batch_sizes: List[int] = Field(default=[16], description="List of batch sizes")
    n_trials: int = Field(default=3, description="Number of hyperparameter optimization trials (intentos) to run")

import subprocess
import json
import shlex
from pathlib import Path

CONFIG_FILE = Path.home() / ".wyolo_mcp_config.json"

def _get_credentials() -> Dict[str, str]:
    if not CONFIG_FILE.exists():
        raise ValueError("Cluster credentials not configured. Please use the 'set_cluster_credentials' tool first.")
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

@mcp.tool()
def set_cluster_credentials(ip: str, cifs_user: str, cifs_pass: str) -> Dict[str, Any]:
    """
    Save the cluster IP and Samba CIFS credentials to a local configuration file.
    The agent should call this tool when the user provides the cluster IP and credentials.
    """
    config_data = {
        "api_url": f"http://{ip}:23442",
        "control_host": ip,
        "cifs_user": cifs_user,
        "cifs_pass": cifs_pass
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        return {"success": True, "message": f"Credentials saved successfully to {CONFIG_FILE}"}
    except Exception as e:
        return {"error": f"Failed to save credentials: {str(e)}"}

import asyncio

@mcp.tool()
async def get_cluster_status() -> Dict[str, Any]:
    """
    Get the overall status of the NeuralForgeAI cluster, including health metrics, 
    active celery workers (invokers), and the current tasks queue.
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    async with httpx.AsyncClient() as client:
        try:
            # Peticiones en paralelo para obtener la imagen completa del cluster
            health_req = client.get(f"{creds['api_url']}/health")
            workers_req = client.get(f"{creds['api_url']}/workers")
            tasks_req = client.get(f"{creds['api_url']}/tasks")
            
            health_res, workers_res, tasks_res = await asyncio.gather(health_req, workers_req, tasks_req, return_exceptions=True)
            
            status_data = {"status": "online"}
            
            if not isinstance(health_res, Exception) and health_res.status_code == 200:
                status_data["health"] = health_res.json()
            else:
                status_data["health"] = {"error": "Failed to fetch health"}
                
            if not isinstance(workers_res, Exception) and workers_res.status_code == 200:
                status_data["workers"] = workers_res.json()
            else:
                status_data["workers"] = {"error": "Failed to fetch workers"}
                
            if not isinstance(tasks_res, Exception) and tasks_res.status_code == 200:
                status_data["tasks"] = tasks_res.json()
            else:
                status_data["tasks"] = {"error": "Failed to fetch tasks queue"}
                
            return status_data
        except Exception as e:
            return {"status": "offline", "error": str(e)}

@mcp.tool()
async def get_study_details(study_id: str) -> Dict[str, Any]:
    """
    Get detailed telemetry and status of a specific YOLO training study.
    Returns progress, active invoker, and current trial metrics.
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": str(e)}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{creds['api_url']}/study/{study_id}")
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
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": str(e)}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{creds['api_url']}/study/{study_id}/cancel")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to cancel: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

@mcp.tool()
def generate_training_yaml(config: TrainingConfig, output_dir: str = ".") -> Dict[str, Any]:
    """
    Generate a NeuralForgeAI training YAML configuration file and save it to disk.
    This allows the user to inspect the file before launching the training.
    Returns the absolute path to the generated YAML file.
    """
    import yaml
    import os
    
    try:
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
                "direction": "maximize",
                "n_trials": config.n_trials
            }
        }
        
        output_path = os.path.abspath(os.path.join(output_dir, f"{config.name}.yaml"))
        with open(output_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
            
        return {"success": True, "yaml_path": output_path, "message": f"YAML configuration generated and saved to {output_path}"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def launch_training(yaml_path: str) -> Dict[str, Any]:
    """
    Submit a locally saved YOLO training YAML configuration to the NeuralForgeAI cluster.
    Use this after the user has reviewed and approved the YAML file generated by `generate_training_yaml`.
    """
    import os
    import yaml
    
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": str(e)}
        
    if not os.path.exists(yaml_path):
        return {"error": f"YAML file not found at {yaml_path}"}

    async with httpx.AsyncClient() as client:
        try:
            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
                
            filename = os.path.basename(yaml_path)
            
            # The API expects a multipart form-data upload with a file named 'config_file'
            files = {
                "config_file": (filename, yaml_content.encode("utf-8"), "application/x-yaml")
            }
            data = {
                "mode": "public",
                "priority": "medium"
            }
            response = await client.post(f"{creds['api_url']}/train", files=files, data=data)
            
            if response.status_code == 200:
                resp_data = response.json()
                study_id = resp_data.get("study_id")
                
                # If a study_id was returned, save it back into the YAML file
                if study_id:
                    try:
                        with open(yaml_path, 'r') as f:
                            parsed_yaml = yaml.safe_load(f)
                        parsed_yaml['study_id'] = study_id
                        with open(yaml_path, 'w') as f:
                            yaml.dump(parsed_yaml, f, default_flow_style=False)
                        resp_data["message"] = f"Training launched. study_id '{study_id}' was saved to {yaml_path}"
                    except Exception as e:
                        resp_data["warning"] = f"Launched, but failed to save study_id to YAML: {str(e)}"
                        
                return {"success": True, "details": resp_data}
            return {"success": False, "error": response.text}
        except Exception as e:
            return {"error": str(e)}


@mcp.tool()
def check_dataset_path(path: str) -> Dict[str, Any]:
    """
    Verify if a dataset path exists on the remote Samba share by spinning up a lightweight Docker container.
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": str(e)}

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
        "-e", f"CONTROL_HOST={creds['control_host']}",
        "-e", f"CIFS_USER={creds['cifs_user']}",
        "-e", f"CIFS_PASS={creds['cifs_pass']}",
        "wisrovi/train_service:worker_executor_v1.0.0",
        "bash", "-c", cmd
    ]
    
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"error": f"Failed to execute docker check: {str(e)}"}

@mcp.tool()
def validate_dataset_advanced(dataset_path: str, task: str = "detect") -> Dict[str, Any]:
    """
    Validates a YOLO dataset structure by running an inspection script inside a Docker container 
    connected to the remote CIFS share. Supports detect/segment (yaml) and classify (directory).
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": str(e)}

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
        "-e", f"CONTROL_HOST={creds['control_host']}",
        "-e", f"CIFS_USER={creds['cifs_user']}",
        "-e", f"CIFS_PASS={creds['cifs_pass']}",
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
