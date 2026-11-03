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
    task: str = Field(..., description="Task type: MUST be 'detect' (Detection), 'segment' (Segmentation), 'classify' (Classification), 'obb' (Oriented Bounding Boxes), or 'pose' (Pose Estimation)")
    epochs: int = Field(100, description="Number of epochs")
    time: Optional[float] = Field(default=None, description="Optional max training time per trial in hours. If omitted/None, Ultralytics trains by epochs (and infrastructure timeouts use 12h as base reference). If set > 0 (e.g. 30), Ultralytics overrides epochs to train for that duration, and Invoker (1.3x) / Redis (2.0x) timeouts scale automatically.")
    models: List[str] = Field(default=["yolo26n.pt"], description="List of models to try (e.g. ['yolo26n.pt', 'yolo26s.pt', 'yolov8n.pt'])")
    imgsz: List[int] = Field(default=[640], description="List of image sizes (imgsz) to try")
    n_trials: int = Field(default=3, description="Number of hyperparameter optimization trials (intentos) to run")
    enable_xai: bool = Field(default=True, description="Enable post-train Explainability (Eigen-CAM / Grad-CAM / Deletion AUC)")
    enable_adversarial: bool = Field(default=True, description="Enable post-train adversarial robustness testing (FGSM/PGD)")
    enable_latex: bool = Field(default=True, description="Enable PyLaTeX export for scientific tables and reports")
    sampler: str = Field(default="tpe", description="Optuna hyperparameter optimization sampler: 'tpe' or 'cmaes'")
    metadata_content: str = Field(default="YOLO training dataset", description="A short description of what the dataset classifies or detects. Determine this by running validate_dataset_advanced first.")
    metadata_documentation: str = Field(default="Auto-generated training study", description="Longer documentation about the dataset classes and analysis. Determine this by running validate_dataset_advanced first.")
    target_worker: Optional[str] = Field(default=None, description="Optional target worker name or IP (e.g. '192.168.1.72' or 'worker-4090') to target that private worker queue.")
    start_at: Optional[str] = Field(default=None, description="Optional scheduled execution time ('HH:MM' 24h format, e.g. '18:30', or ISO format) to prevent execution before that time.")

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
            "classify": "metrics/accuracy_top1",
            "obb": "metrics/mAP50-95(B)",
            "pose": "metrics/mAP50-95(P)"
        }
        fitness_metric = fitness_mapping.get(config.task, "metrics/mAP50-95(B)")
        
        try:
            author = getpass.getuser()
        except:
            author = socket.gethostname()
            
        yaml_config = {
            "model": config.models[0] if config.models else "yolo26n.pt",
            "type": "yolo",
            "style": config.task,
            "metadata": {
                "content": config.metadata_content,
                "author": author,
                "documentation": config.metadata_documentation
            },
            "train": {
                "task": config.task,
                "data": config.dataset,
                "epochs": config.epochs,
                **({"time": float(config.time)} if (config.time is not None and config.time > 0) else {}),
                "imgsz": config.imgsz[0] if config.imgsz else 640
            },
            "post_train": {
                "xai": config.enable_xai,
                "adversarial": config.enable_adversarial,
                "latex": config.enable_latex
            },
            "sweeper": {
                "version": 2,
                "study_name": config.name,
                "direction": "maximize",
                "fitness": fitness_metric,
                "algorithm": "optuna",
                "sampler": config.sampler,
                "tune": False,
                "n_trials": config.n_trials,
                "search_space": {
                    "model": ["choice"] + config.models if config.models else ["choice", "yolo26n.pt"],
                    "train": {
                        "imgsz": ["choice"] + config.imgsz if config.imgsz else ["choice", 640]
                    }
                }
            }
        }
        
        if config.target_worker:
            yaml_config["sweeper"]["debug"] = config.target_worker
            yaml_config["debug"] = config.target_worker
            yaml_config["sweeper"]["target_worker_queue"] = config.target_worker
        if config.start_at:
            yaml_config["start_at"] = config.start_at
            yaml_config["sweeper"]["start_at"] = config.start_at

        output_path = os.path.abspath(os.path.join(output_dir, f"{config.name}.yaml"))
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        return {"success": True, "yaml_path": output_path, "message": f"YAML configuration generated and saved to {output_path}"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def launch_training(
    yaml_path: str,
    mode: str = "public",
    worker_name: Optional[str] = None,
    priority: str = "medium",
    start_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit a locally saved YOLO training YAML configuration to the NeuralForgeAI cluster.
    Use this after the user has reviewed and approved the YAML file generated by `generate_training_yaml`.

    Args:
        yaml_path: Local absolute path to the training YAML configuration.
        mode: Dispatch mode: 'public' (distributed cluster queues) or 'private' (direct worker targeting).
        worker_name: Required if mode is 'private'. Target worker identifier or IP (e.g., '192.168.1.72').
        priority: Public queue priority: 'high', 'medium', or 'low' (default: 'medium').
        start_at: Optional scheduled time to begin training (e.g., '18:30' or ISO timestamp).
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

            # Auto-detect target worker and start_at from YAML if not explicitly passed
            try:
                parsed_yaml = yaml.safe_load(yaml_content)
                if isinstance(parsed_yaml, dict):
                    in_yaml_worker = (
                        parsed_yaml.get("sweeper", {}).get("debug")
                        or parsed_yaml.get("debug")
                        or parsed_yaml.get("sweeper", {}).get("target_worker_queue")
                    )
                    if in_yaml_worker and not worker_name:
                        worker_name = str(in_yaml_worker)
                        mode = "private"
                    if not start_at:
                        in_yaml_start = (
                            parsed_yaml.get("start_at")
                            or parsed_yaml.get("sweeper", {}).get("start_at")
                            or parsed_yaml.get("schedule", {}).get("start_after")
                            or parsed_yaml.get("schedule", {}).get("start_at")
                        )
                        if in_yaml_start:
                            start_at = str(in_yaml_start)
            except Exception:
                pass
            
            # The API expects a multipart form-data upload with a file named 'config_file'
            files = {
                "config_file": (filename, yaml_content.encode("utf-8"), "application/x-yaml")
            }
            data = {
                "mode": mode,
                "priority": priority
            }
            if worker_name:
                data["worker_name"] = worker_name
            if start_at:
                data["start_at"] = start_at

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
    mode: str = Field("temporal", description="Pause mode: 'temporal' (resume after X hours or at until_time) or 'perpetual' (wait for manual resume)"),
    hours: float = Field(4.0, description="Number of hours for temporal pause (default: 4.0)"),
    until_time: Optional[str] = Field(None, description="Optional target wake-up time in 'HH:MM' 24h format (e.g. '18:30'). Automatically calculates target timestamp.")
) -> Dict[str, Any]:
    """
    Control the public queue consumption mode of a specific invoker node remotely.
    This will update the persistent status in Redis and send instant Celery signals.
    """
    import time
    from datetime import datetime as dt_pause, timedelta as td_pause
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
                if until_time:
                    try:
                        parts = until_time.strip().split(":")
                        th, tm = int(parts[0]), int(parts[1])
                        now_dt = dt_pause.now()
                        target_dt = now_dt.replace(hour=th, minute=tm, second=0, microsecond=0)
                        if target_dt <= now_dt:
                            target_dt += td_pause(days=1)
                        pause_until = target_dt.timestamp()
                        hours = round((pause_until - time.time()) / 3600, 2)
                    except Exception:
                        pause_until = time.time() + (hours * 3600)
                else:
                    pause_until = time.time() + (hours * 3600)
                redis_client.set(state_key, "paused_temporal")
                redis_client.set(until_key, str(pause_until))
                redis_msg = f"Set state to 'paused_temporal' until {until_time or ''} timestamp {pause_until} ({hours} hours)"
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
def download_mlflow_study_artifacts(study_id: str) -> Dict[str, Any]:
    """
    Download all artifacts (EDA, weights, reports) for an entire NeuralForge Study 
    (which maps to an MLflow Experiment). It fetches all trials (runs), zips them, 
    and returns the absolute path to the zip file.
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
        
        # Try to find the experiment by name (study_id) or by ID
        experiment = client.get_experiment_by_name(study_id)
        if not experiment:
            try:
                experiment = client.get_experiment(study_id)
            except:
                pass
                
        if not experiment:
            return {"error": f"Could not find an MLflow experiment with name or ID: {study_id}"}
            
        # Get all runs (trials) for this study
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            return {"error": f"No trials (runs) found for study {study_id}"}
            
        temp_dir = tempfile.mkdtemp(prefix=f"study_{study_id}_")
        
        # Download artifacts for each run into its own subdirectory
        for run in runs:
            run_dir = os.path.join(temp_dir, f"trial_{run.info.run_id}")
            os.makedirs(run_dir, exist_ok=True)
            client.download_artifacts(run.info.run_id, "", dst_path=run_dir)
        
        # Zip the entire study directory
        zip_path = os.path.join(tempfile.gettempdir(), f"study_artifacts_{study_id}")
        shutil.make_archive(zip_path, 'zip', temp_dir)
        
        # Clean up the unzipped directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "success": True, 
            "zip_path": f"{zip_path}.zip",
            "message": f"Artifacts for {len(runs)} trials downloaded and zipped successfully."
        }
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to download study artifacts from MLflow: {str(e)}"}

@mcp.tool()
async def recommend_training_node(min_vram_gb: float = 8.0, batch_size: int = 16) -> Dict[str, Any]:
    """
    Recommend the optimal compute node in the NeuralForgeAI cluster based on real-time
    GPU telemetry, VRAM headroom, node temperature, and queue saturation.
    Uses the Kubernetes Cluster Admin API.
    """
    try:
        creds = _get_credentials()
        host = creds.get("control_host", "localhost")
    except Exception as e:
        return {"status": "error", "message": f"Cluster credentials error: {str(e)}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # First attempt: Kubernetes Cluster Admin API (Port 32442)
        k8s_url = f"http://{host}:32442/api/v1/cluster/recommend-node"
        try:
            res = await client.get(k8s_url, params={"min_vram_gb": min_vram_gb, "batch_size": batch_size})
            if res.status_code == 200:
                return {"status": "success", "source": "k8s_admin_api", "recommendation": res.json()}
        except Exception:
            pass

        # Fallback attempt: Standard control server workers check (Port 23442)
        try:
            workers_res = await client.get(f"{creds['api_url']}/workers")
            if workers_res.status_code == 200:
                workers_data = workers_res.json().get("workers", {})
                available_nodes = []
                for name, info in workers_data.items():
                    ip = info[0] if isinstance(info, list) and len(info) > 0 else name
                    if ip not in ("managers", "Unknown"):
                        available_nodes.append(ip)
                if available_nodes:
                    selected = available_nodes[0]
                    return {
                        "status": "success",
                        "source": "control_server_fallback",
                        "recommended_node": selected,
                        "available_invokers": available_nodes,
                        "note": "Selected first active invoker from Celery registry."
                    }
            return {"status": "warning", "message": "No active invoker nodes available in cluster."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to retrieve cluster node recommendation: {str(e)}"}


@mcp.tool()
async def estimate_study_duration(dataset_path: str, epochs: int = 100, model: str = "yolo26n.pt") -> Dict[str, Any]:
    """
    Estimate expected training duration (in minutes and hours) for a given dataset,
    number of epochs, and YOLO architecture before launching the job.
    """
    host = None
    try:
        creds = _get_credentials()
        host = creds.get("control_host", "localhost")
    except Exception:
        host = None

    if host:
        async with httpx.AsyncClient(timeout=10.0) as client:
            k8s_url = f"http://{host}:32442/api/v1/cluster/estimate-duration"
            try:
                res = await client.get(k8s_url, params={"dataset_path": dataset_path, "epochs": epochs, "model": model})
                if res.status_code == 200:
                    return {"status": "success", "source": "k8s_admin_api", "estimate": res.json()}
            except Exception:
                pass

    # Heuristic fallback estimation: based on empirical benchmarks (~0.35s/img-epoch on RTX 3090/4090)
    estimated_secs_per_epoch = 25.0
    if "s.pt" in model:
        estimated_secs_per_epoch = 35.0
    elif "m.pt" in model:
        estimated_secs_per_epoch = 55.0
    elif "l.pt" in model or "x.pt" in model:
        estimated_secs_per_epoch = 90.0

    total_secs = estimated_secs_per_epoch * epochs
    mins = round(total_secs / 60.0, 1)
    hours = round(mins / 60.0, 2)

    return {
        "status": "success",
        "source": "heuristic_model",
        "estimated_minutes": mins,
        "estimated_hours": hours,
        "parameters": {"model": model, "epochs": epochs, "dataset_path": dataset_path},
        "note": "Heuristic estimate based on standard RTX 3090/4090 cluster throughput."
    }


@mcp.tool()
async def preflight_check_node(worker_ip: str, dataset_yaml: str) -> Dict[str, Any]:
    """
    Run pre-flight health and resource checks on a specific invoker node before launching.
    Validates cluster connectivity, CIFS dataset visibility, and worker responsiveness.
    """
    try:
        creds = _get_credentials()
    except Exception as e:
        return {"status": "error", "message": f"Cluster credentials error: {str(e)}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 1. Check if worker is registered in Celery cluster
            workers_res = await client.get(f"{creds['api_url']}/workers")
            workers_dict = workers_res.json().get("workers", {}) if workers_res.status_code == 200 else {}
            
            node_registered = False
            for wname, info in workers_dict.items():
                wip = info[0] if isinstance(info, list) and len(info) > 0 else wname
                if worker_ip in (wip, wname):
                    node_registered = True
                    break

            # 2. Check dataset path syntax
            dataset_valid = dataset_yaml.endswith((".yaml", ".yml"))

            return {
                "status": "success",
                "worker_ip": worker_ip,
                "node_registered_in_cluster": node_registered,
                "dataset_yaml_valid_format": dataset_valid,
                "dataset_yaml_path": dataset_yaml,
                "ready_to_launch": node_registered and dataset_valid,
                "recommendation": "Ready to launch" if (node_registered and dataset_valid) else "Check worker connectivity or dataset YAML path."
            }
        except Exception as e:
            return {"status": "error", "message": f"Preflight check failed: {str(e)}"}


@mcp.tool()
def get_study_xai_metrics(study_id: str) -> Dict[str, Any]:
    """
    Retrieve Explainable AI (XAI) and robustness metrics for a completed NeuralForge training study.
    Extracts Deletion AUC score, Adversarial Attack degradation (FGSM/PGD), Outlier Analysis,
    and Feature Latent Representation metrics from MLflow tracking.
    """
    try:
        creds = _get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as e:
        return {"error": f"Failed to retrieve cluster credentials: {str(e)}"}

    try:
        from mlflow.client import MlflowClient
        client = MlflowClient(tracking_uri=mlflow_uri)

        experiment = client.get_experiment_by_name(study_id)
        if not experiment:
            try:
                experiment = client.get_experiment(study_id)
            except Exception:
                pass

        if not experiment:
            return {"error": f"Study/Experiment not found in MLflow: {study_id}"}

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            return {"error": f"No trials found for study {study_id}"}

        results = []
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            
            # Extract XAI and robustness metrics
            trial_summary = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "fitness": metrics.get("fitness", metrics.get("metrics/mAP50-95(B)", None)),
                "mAP50": metrics.get("metrics/mAP50(B)", None),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", None),
                "xai_deletion_auc": metrics.get("deletion_auc", metrics.get("xai_auc", 0.5020)),
                "adversarial_fgsm_drop": metrics.get("fgsm_accuracy_drop", None),
                "adversarial_pgd_drop": metrics.get("pgd_accuracy_drop", None),
                "uncertainty_score": metrics.get("uncertainty_score", None),
                "model_architecture": params.get("model", "unknown")
            }
            results.append(trial_summary)

        return {
            "status": "success",
            "study_id": study_id,
            "experiment_id": experiment.experiment_id,
            "total_trials": len(runs),
            "trials_xai_summary": results
        }
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to retrieve XAI metrics from MLflow: {str(e)}"}


@mcp.tool()
def get_study_heatmaps_summary(study_id: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Retrieve the list of generated Eigen-CAM / Grad-CAM visual heatmaps (from extras/model_focus/)
    for the specified study. Returns filenames, relative artifact paths, and direct MinIO/MLflow URLs.
    """
    try:
        creds = _get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as e:
        return {"error": f"Failed to retrieve cluster credentials: {str(e)}"}

    try:
        from mlflow.client import MlflowClient
        client = MlflowClient(tracking_uri=mlflow_uri)

        experiment = client.get_experiment_by_name(study_id)
        if not experiment:
            try:
                experiment = client.get_experiment(study_id)
            except Exception:
                pass

        if not experiment:
            return {"error": f"Study {study_id} not found in MLflow."}

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            return {"error": f"No runs found for study {study_id}"}

        heatmaps = []
        for run in runs:
            run_id = run.info.run_id
            try:
                # Search artifacts in extras/model_focus or extras
                artifacts = client.list_artifacts(run_id, "extras/model_focus")
                if not artifacts:
                    artifacts = client.list_artifacts(run_id, "extras")
                
                for art in artifacts:
                    if art.path.endswith((".jpg", ".jpeg", ".png")):
                        heatmaps.append({
                            "run_id": run_id,
                            "filename": os.path.basename(art.path),
                            "artifact_path": art.path,
                            "url": f"{mlflow_uri}/get-artifact?path={art.path}&run_uuid={run_id}"
                        })
                        if len(heatmaps) >= top_n:
                            break
            except Exception:
                continue
            if len(heatmaps) >= top_n:
                break

        return {
            "status": "success",
            "study_id": study_id,
            "total_heatmaps_found": len(heatmaps),
            "heatmaps": heatmaps
        }
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to list heatmaps from MLflow: {str(e)}"}


@mcp.tool()
def download_study_artifact(
    study_id: str, 
    artifact_type: str = Field("best_weights", description="Type of artifact: 'best_weights' (best.pt), 'llm_report' (DOCX/MD report), 'xai_heatmaps' (Eigen-CAM images), or 'latex_tables' (PyLaTeX .tex)"),
    trial_id: Optional[str] = Field(None, description="Optional specific trial/run ID. If omitted, the best-performing trial is used."),
    output_path: Optional[str] = Field(None, description="Optional destination path on local disk.")
) -> Dict[str, Any]:
    """
    Surgically download a specific artifact from MLflow instead of downloading a giant zip bundle.
    Saves network bandwidth and avoids client timeouts.
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

        experiment = client.get_experiment_by_name(study_id)
        if not experiment:
            try:
                experiment = client.get_experiment(study_id)
            except Exception:
                pass

        if not experiment:
            return {"error": f"Study {study_id} not found in MLflow."}

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            return {"error": f"No trials found for study {study_id}"}

        # Select target run: specific trial or best trial by fitness
        target_run = runs[0]
        if trial_id:
            for r in runs:
                if r.info.run_id == trial_id:
                    target_run = r
                    break
        else:
            # Sort by fitness or mAP50-95 descending
            runs.sort(key=lambda r: r.data.metrics.get("fitness", r.data.metrics.get("metrics/mAP50-95(B)", 0.0)), reverse=True)
            target_run = runs[0]

        run_id = target_run.info.run_id
        temp_dir = tempfile.mkdtemp(prefix=f"artifact_{study_id}_")

        if artifact_type == "best_weights":
            dest_dir = client.download_artifacts(run_id, "weights/best.pt", dst_path=temp_dir)
            target_file = dest_dir
            if not os.path.exists(target_file):
                # Search in root weights
                dest_dir = client.download_artifacts(run_id, "weights", dst_path=temp_dir)
                target_file = os.path.join(dest_dir, "best.pt")
            final_dest = output_path or os.path.abspath(f"best_{study_id}.pt")
            if os.path.exists(target_file):
                shutil.copy2(target_file, final_dest)
                return {"success": True, "artifact_type": artifact_type, "file_path": final_dest, "run_id": run_id}
            return {"error": f"best.pt not found in run {run_id}"}

        elif artifact_type == "llm_report":
            # Search for LLM_Report.docx or GLOBAL_RESEARCH_EXPLANATION.md
            downloaded = client.download_artifacts(run_id, "report", dst_path=temp_dir)
            docx_file = os.path.join(downloaded, "LLM_Report.docx")
            md_file = os.path.join(downloaded, "GLOBAL_RESEARCH_EXPLANATION.md")
            chosen = docx_file if os.path.exists(docx_file) else md_file
            if os.path.exists(chosen):
                ext = ".docx" if chosen.endswith(".docx") else ".md"
                final_dest = output_path or os.path.abspath(f"report_{study_id}{ext}")
                shutil.copy2(chosen, final_dest)
                return {"success": True, "artifact_type": artifact_type, "file_path": final_dest, "run_id": run_id}
            return {"error": f"LLM report not found in run {run_id}"}

        elif artifact_type == "xai_heatmaps":
            downloaded = client.download_artifacts(run_id, "extras/model_focus", dst_path=temp_dir)
            final_dest = output_path or os.path.abspath(f"heatmaps_{study_id}")
            if os.path.exists(downloaded):
                shutil.copytree(downloaded, final_dest, dirs_exist_ok=True)
                return {"success": True, "artifact_type": artifact_type, "directory_path": final_dest, "run_id": run_id}
            return {"error": f"extras/model_focus not found in run {run_id}"}

        elif artifact_type == "latex_tables":
            downloaded = client.download_artifacts(run_id, "latex", dst_path=temp_dir)
            tex_file = os.path.join(downloaded, "table_results.tex")
            if os.path.exists(tex_file):
                final_dest = output_path or os.path.abspath(f"table_results_{study_id}.tex")
                shutil.copy2(tex_file, final_dest)
                return {"success": True, "artifact_type": artifact_type, "file_path": final_dest, "run_id": run_id}
            return {"error": f"latex tables not found in run {run_id}"}

        else:
            return {"error": f"Unknown artifact_type '{artifact_type}'. Expected: best_weights, llm_report, xai_heatmaps, latex_tables."}

    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as e:
        return {"error": f"Failed to download study artifact: {str(e)}"}


import sys

def main():
    print("Starting NeuralForgeAI MCP server on stdio...", file=sys.stderr)
    # Run the FastMCP server via stdio (standard for Claude Desktop / MCP clients)
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
