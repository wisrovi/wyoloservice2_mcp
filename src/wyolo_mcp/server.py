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
    name: str = Field(..., description="Name of the study. MUST follow the format <project_name>_<dataset_name> (e.g. arepo_cicatrices). If the user does not provide a project name, you MUST ask for it before calling this tool. IMPORTANT: Do NOT mention this formatting rule to the user, just ask for the project name naturally.")
    dataset: str = Field(..., description="Absolute path to the dataset.yaml file")
    task: str = Field(..., description="Task type: MUST be 'detect' (Detection), 'segment' (Segmentation), or 'classify' (Classification)")
    epochs: int = Field(100, description="Number of epochs")
    models: List[str] = Field(default=["yolov8n.pt"], description="List of models to try")
    imgsz: List[int] = Field(default=[640], description="List of image sizes (imgsz) to try")
    n_trials: int = Field(default=3, description="Number of hyperparameter optimization trials (intentos) to run")
    metadata_content: str = Field(default="YOLO training dataset", description="A short description of what the dataset classifies or detects. Determine this by running validate_dataset_advanced first.")
    metadata_documentation: str = Field(default="Auto-generated training study", description="Longer documentation about the dataset classes and analysis. Determine this by running validate_dataset_advanced first.")

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
                tasks_data = tasks_res.json()
                
                # Enrich active running tasks with resolved IP addresses and SSH logging commands
                import re
                workers_dict = status_data["workers"].get("workers", {}) if "workers" in status_data else {}
                running_tasks = tasks_data.get("running", [])
                
                for task in running_tasks:
                    worker_name = task.get("worker")
                    task_name = task.get("name", "")
                    
                    # Extract IP address from the worker
                    invoker_ip = "Unknown"
                    if workers_dict and worker_name in workers_dict:
                        info = workers_dict[worker_name]
                        if isinstance(info, list) and len(info) > 0:
                            invoker_ip = info[0]
                    else:
                        match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', worker_name or "")
                        if match:
                            invoker_ip = match.group(1)
                    
                    task["worker_ip"] = invoker_ip
                    
                    if "train_on_gpu" in task_name and invoker_ip not in ("Unknown", "managers"):
                        executor_name = f"wyolo_executor_{invoker_ip}"
                        task["log_commands"] = {
                            "docker_logs": f"ssh -t wyolo@{invoker_ip} \"docker logs -f {executor_name}\"",
                            "tail_log_file": f"ssh -t wyolo@{invoker_ip} \"tail -f /home/wyolo/train_service_results/logs_{executor_name}.txt\""
                        }
                    else:
                        task["log_commands"] = None
                
                status_data["tasks"] = tasks_data
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
    
    IMPORTANT WORKFLOW FOR AGENTS: 
    When the user asks 'how is my training going?' (or similar) without providing a study_id:
    1. DO NOT ask the user for the study_id immediately.
    2. First, search the current working directory for `.yaml` files.
    3. Read the discovered `.yaml` files to check if they contain a `study_id` field.
    4. If you find a `study_id` inside a YAML file, automatically use this tool with that ID to check its status.
    5. Only ask the user for the ID or file path if you cannot find any `.yaml` files with a `study_id` in the current directory.
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
    import getpass
    import socket
    
    try:
        fitness_mapping = {
            "detect": "metrics/mAP50-95(B)",
            "segment": "metrics/mAP50-95(M)",
            "classify": "metrics/accuracy_top1"
        }
        fitness_metric = fitness_mapping.get(config.task, "metrics/mAP50-95(B)")
        
        try:
            author = getpass.getuser()
        except:
            author = socket.gethostname()
            
        yaml_config = {
            "model": config.models[0] if config.models else "yolov8n.pt",
            "type": "yolo",
            "metadata": {
                "content": config.metadata_content,
                "author": author,
                "documentation": config.metadata_documentation
            },
            "train": {
                "task": config.task,
                "data": config.dataset,
                "epochs": config.epochs,
                "imgsz": config.imgsz[0] if config.imgsz else 640
            },
            "sweeper": {
                "version": 2,
                "study_name": config.name,
                "direction": "maximize",
                "fitness": fitness_metric,
                "algorithm": "optuna",
                "tune": False,
                "n_trials": config.n_trials,
                "search_space": {
                    "model": ["choice"] + config.models if config.models else ["choice", "yolov8n.pt"],
                    "train": {
                        "imgsz": ["choice"] + config.imgsz if config.imgsz else ["choice", 640]
                    }
                }
            }
        }
        
        output_path = os.path.abspath(os.path.join(output_dir, f"{config.name}.yaml"))
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
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
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{"valid"'):
                return json.loads(line)
        return json.loads(result.stdout.strip())
    except Exception as e:
        return {"error": f"Failed to execute docker validation: {str(e)}"}


@mcp.tool()
async def manage_invoker_queues(
    worker_ip: str = Field(..., description="IP address of the target invoker worker (e.g., 192.168.1.54)"),
    action: str = Field(..., description="Action to perform: 'pause' (out of public queues) or 'resume' (back to public queues)"),
    mode: str = Field("temporal", description="Pause mode: 'temporal' (resume after X hours) or 'perpetual' (wait for manual resume)"),
    hours: float = Field(4.0, description="Number of hours for temporal pause (default: 4.0)")
) -> Dict[str, Any]:
    """
    Control the public queue consumption mode of a specific invoker node remotely.
    This will update the persistent status in Redis and send instant Celery signals.
    """
    import time
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"error": f"Failed to retrieve cluster credentials: {str(e)}"}

    redis_host = creds.get("control_host")
    redis_port = 23437  # default port
    redis_db = 0        # default db

    # 1. Update State in Redis
    try:
        import redis
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        state_key = f"invoker:{worker_ip}:pause_state"
        until_key = f"invoker:{worker_ip}:pause_until"

        if action == "pause":
            if mode == "temporal":
                pause_until = time.time() + (hours * 3600)
                redis_client.set(state_key, "paused_temporal")
                redis_client.set(until_key, str(pause_until))
                redis_msg = f"Set state to 'paused_temporal' until timestamp {pause_until} ({hours} hours)"
            else:
                redis_client.set(state_key, "paused_perpetual")
                redis_client.delete(until_key)
                redis_msg = "Set state to 'paused_perpetual'"
        else:
            redis_client.set(state_key, "active")
            redis_client.delete(until_key)
            redis_msg = "Set state to 'active' (resumed)"
    except ImportError:
        return {"error": "Required library 'redis' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to update state in Redis: {str(e)}"}

    # 2. Send instant control command to destination node via Celery
    try:
        from celery import Celery
        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        app = Celery("ml_cluster", broker=redis_url, backend=redis_url)
        
        node_name = f"celery@wyolo_invoker_{worker_ip}"
        public_queues = ["gpus_high", "gpus_medium", "gpus_low"]
        
        results = []
        for queue in public_queues:
            if action == "pause":
                response = app.control.cancel_consumer(
                    queue,
                    destination=[node_name],
                    reply=True,
                    timeout=2.0
                )
            else:
                response = app.control.add_consumer(
                    queue,
                    destination=[node_name],
                    reply=True,
                    timeout=2.0
                )
            results.append((queue, response))
            
        success = False
        node_responses = {}
        for queue, response in results:
            if response:
                for item in response:
                    if isinstance(item, dict):
                        for node, status in item.items():
                            node_responses[f"{node}:{queue}"] = status.get('ok', status)
                            success = True
                    else:
                        node_responses[queue] = item
                        success = True
            else:
                node_responses[queue] = "No response (worker offline)"
                
        return {
            "success": True,
            "redis_update": redis_msg,
            "celery_signals": {
                "sent_to": node_name,
                "node_status": "responsive" if success else "offline (state saved in Redis for startup)",
                "responses": node_responses
            }
        }
    except ImportError:
        return {
            "success": True,
            "redis_update": redis_msg,
            "celery_signals": {
                "warning": "Required library 'celery' is not installed in the MCP environment, but state was saved to Redis."
            }
        }
    except Exception as e:
        return {
            "success": True,
            "redis_update": redis_msg,
            "celery_signals": {
                "error": f"Failed to send immediate Celery signals: {str(e)} (state was saved in Redis)"
            }
        }
@mcp.tool()
async def launch_private_test_training(task_type: str, ip_address: str) -> Dict[str, Any]:
    """
    Launch a private base test training study targeted directly to a specific worker IP.
    This tool is fully self-contained and does not require local config files.
    
    Args:
        task_type: Type of YOLO task to test: 'detection', 'classification', or 'segmentation'
        ip_address: Physical IP address of the target invoker worker (e.g. '192.168.1.39')
    """
    import yaml
    import time
    
    # Embedded configurations for absolute portability
    configs_map = {
        "segmentation": """
model: "yolo26n-seg.pt"
type: "yolo"
train:
  batch: -1
  data: "/examples/ArchitecturePlan/data.yaml"
  epochs: 2
  imgsz: 640
  plots: true
sweeper:
  version: 1
  algorithm: optuna
  direction: maximize
  study_name: "architecture_segmentation"
  fitness: "metrics/mAP50(M)"
  tune: false
  sampler: "TPESampler"
  n_trials: 1
  search_space:
    model: [ "choice", "yolov8n-seg.pt" ]
    train:
      imgsz: [ "choice", 416 ]
      lr0: [ "loguniform", 1e-5, 1e-2 ]
extras:
  gpu:
    id: 0
    limit: 0.95
metadata:
  content: "Este es un experimento de clasificacion de imagenes."
  author: "William Rodriguez"
  documentation: "Este modelo fue entrenado con datos del 2025."
""",
        "classification": """
model: "yolo26n-cls.pt"
type: "yolo"
train:
  batch: -1
  data: "/examples/colorball.v8i.multiclass/"
  epochs: 5
  imgsz: 640
  plots: false
sweeper:
  version: 1
  algorithm: optuna
  direction: maximize
  study_name: "color_ball_classification"
  fitness: "metrics/accuracy_top1"
  tune: false
  sampler: "TPESampler"
  n_trials: 1
  search_space:
    model: [ "choice", "yolov8n-cls.pt" ]
    train:
      imgsz: [ "choice", 416 ]
      lr0: [ "loguniform", 1e-5, 1e-2 ]
extras:
  gpu:
    id: 0
    limit: 0.95
metadata:
  content: "Este es un experimento de clasificacion de imagenes."
  author: "William Rodriguez"
  documentation: "Este modelo fue entrenado con datos del 2025."
""",
        "detection": """
model: "yolo26n.pt"
type: "yolo"
train:
  batch: -1
  data: "/examples/Deteksi_komponen_elektronik.v1i/data.yaml"
  epochs: 2
  imgsz: 640
  plots: true
sweeper:
  version: 1
  algorithm: optuna
  direction: maximize
  study_name: "component_detection"
  fitness: "metrics/mAP50"
  tune: false
  sampler: "TPESampler"
  n_trials: 1
  search_space:
    model: [ "choice", "yolov8n.pt" ]
    train:
      imgsz: [ "choice", 416 ]
      lr0: [ "loguniform", 1e-5, 1e-2 ]
extras:
  gpu:
    id: 0
    limit: 0.95
metadata:
  content: "Este es un experimento de clasificacion de imagenes."
  author: "William Rodriguez"
  documentation: "Este modelo fue entrenado con datos del 2025."
"""
    }
    
    task_type = task_type.lower().strip()
    if task_type not in configs_map:
        return {
            "success": False,
            "error": f"Invalid task_type '{task_type}'. Supported types: 'detection', 'classification', 'segmentation'"
        }
        
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"success": False, "error": f"Credentials error: {str(e)}"}
        
    try:
        # Parse the embedded YAML config
        config_data = yaml.safe_load(configs_map[task_type])
        
        # Enrich config data for forced private execution
        if "sweeper" not in config_data:
            config_data["sweeper"] = {}
            
        timestamp = int(time.time())
        config_data["sweeper"]["debug"] = ip_address
        config_data["sweeper"]["study_name"] = f"test_private_{task_type}_{ip_address.replace('.', '_')}"
        
        # Serialize the modified config data back to YAML string
        yaml_content = yaml.dump(config_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        filename = f"test_private_{task_type}_{ip_address}_{timestamp}.yaml"
        
        # Send the file to the NeuralForge API
        async with httpx.AsyncClient() as client:
            files = {
                "config_file": (filename, yaml_content.encode("utf-8"), "application/x-yaml")
            }
            data = {
                "mode": "private",
                "worker_name": ip_address
            }
            
            response = await client.post(f"{creds['api_url']}/train", files=files, data=data)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "study_type": task_type,
                    "target_ip": ip_address,
                    "study_name": config_data["sweeper"]["study_name"],
                    "api_response": response.json()
                }
            return {
                "success": False,
                "error": f"API responded with status code {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {"success": False, "error": f"Unexpected error during launch: {str(e)}"}
@mcp.tool()
async def trigger_broadcast_docker_pull(image_name: str = "wisrovi/train_service:worker_executor_v1.0.0") -> Dict[str, Any]:
    """
    Send a broadcast remote control command to all active Celery worker invokers 
    to force them to execute 'docker pull' on the specified image.
    
    Args:
        image_name: The Docker image to pull (e.g. 'wisrovi/train_service:worker_executor_v1.0.0')
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"success": False, "error": f"Credentials error: {str(e)}"}
        
    redis_url = f"redis://{creds['control_host']}:23437/0"
    
    try:
        from celery import Celery
        # Initialize celery application pointing to the broker
        app = Celery('tasks', broker=redis_url)
        
        # Send broadcast control command
        # Celery control broadcast returns a list of responses from the workers
        responses = app.control.broadcast(
            "force_docker_pull",
            arguments={"image_name": image_name},
            reply=True,
            timeout=8.0
        )
        
        if not responses:
            return {
                "success": False,
                "message": "No active workers responded. Make sure the invoker nodes are online."
            }
            
        formatted_responses = {}
        success_count = 0
        failed_count = 0
        
        for response in responses:
            for node, details in response.items():
                if isinstance(details, dict):
                    status = details.get("status", "unknown")
                    if status == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                    formatted_responses[node] = details
                else:
                    failed_count += 1
                    formatted_responses[node] = {"status": "error", "error": str(details)}
                    
        return {
            "success": True,
            "image": image_name,
            "summary": f"Broadcast pull sent. {success_count} nodes succeeded, {failed_count} nodes failed/timed out.",
            "responses": formatted_responses
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "Required library 'celery' is not installed in the MCP execution environment."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to send Celery broadcast command: {str(e)}"
        }


@mcp.tool()
def download_mlflow_trial_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Download all artifacts (EDA, weights, reports) for a specific MLflow Run ID, 
    zip them into a single archive, and return the absolute path to the zip file.
    """
    import os
    import tempfile
    import shutil
    try:
        creds = _get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as e:
        return {"error": f"Failed to retrieve cluster credentials: {str(e)}"}
        
    try:
        from mlflow.client import MlflowClient
        
        client = MlflowClient(tracking_uri=mlflow_uri)
        temp_dir = tempfile.mkdtemp(prefix=f"mlflow_run_{run_id}_")
        
        # Download all artifacts for the run
        local_dir = client.download_artifacts(run_id, "", dst_path=temp_dir)
        
        # Zip the contents
        zip_path = os.path.join(tempfile.gettempdir(), f"artifacts_{run_id}")
        shutil.make_archive(zip_path, 'zip', local_dir)
        
        # Clean up the unzipped directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "success": True, 
            "zip_path": f"{zip_path}.zip",
            "message": f"Artifacts downloaded and zipped successfully."
        }
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to download artifacts from MLflow: {str(e)}"}

import sys

def main():
    print("Starting NeuralForgeAI MCP server on stdio...", file=sys.stderr)
    # Run the FastMCP server via stdio (standard for Claude Desktop / MCP clients)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
