"""Training YAML generation and submission tools."""

import getpass
import os
import socket
import sys
from typing import Any, Dict, Optional
import httpx
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dto.training_config_dto import TrainingConfig
from wyolo_mcp.utils.credentials import get_credentials


def generate_training_yaml(config: TrainingConfig, output_dir: str = ".") -> Dict[str, Any]:
    """Generate a NeuralForgeAI training YAML configuration file and save it to disk.

    Args:
        config: TrainingConfig instance defining study parameters.
        output_dir: Directory where the YAML configuration will be written.

    Returns:
        Dict[str, Any]: Generation status, saved file path, and summary message.
    """
    try:
        fitness_mapping = {
            "detect": "metrics/mAP50-95(B)",
            "segment": "metrics/mAP50-95(M)",
            "classify": "metrics/accuracy_top1",
            "obb": "metrics/mAP50-95(B)",
            "pose": "metrics/mAP50-95(P)",
        }
        fitness_metric = fitness_mapping.get(config.task, "metrics/mAP50-95(B)")

        try:
            author = getpass.getuser()
        except Exception:
            author = socket.gethostname()

        yaml_config = {
            "model": config.models[0] if config.models else "yolo26n.pt",
            "type": "yolo",
            "style": config.task,
            "metadata": {
                "content": config.metadata_content,
                "author": author,
                "documentation": config.metadata_documentation,
            },
            "train": {
                "task": config.task,
                "data": config.dataset,
                "epochs": config.epochs,
                **({"time": float(config.time)} if (config.time is not None and config.time > 0) else {}),
                "imgsz": config.imgsz[0] if config.imgsz else 640,
            },
            "post_train": {
                "xai": config.enable_xai,
                "adversarial": config.enable_adversarial,
                "latex": config.enable_latex,
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
                        "imgsz": ["choice"] + config.imgsz if config.imgsz else ["choice", 640],
                    },
                },
            },
        }

        if config.target_worker:
            yaml_config["sweeper"]["debug"] = config.target_worker
            yaml_config["debug"] = config.target_worker
            yaml_config["sweeper"]["target_worker_queue"] = config.target_worker
        if config.start_at:
            yaml_config["start_at"] = config.start_at
            yaml_config["sweeper"]["start_at"] = config.start_at

        output_path = os.path.abspath(os.path.join(output_dir, f"{config.name}.yaml"))
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return {"success": True, "yaml_path": output_path, "message": f"YAML configuration saved to {output_path}"}
    except Exception as exc:
        return {"error": str(exc)}


async def launch_training(
    yaml_path: str,
    mode: str = "public",
    worker_name: Optional[str] = None,
    priority: str = "medium",
    start_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit a locally saved YOLO training YAML configuration to the cluster API.

    Args:
        yaml_path: Absolute local file path to YAML file.
        mode: Execution mode ('public' or 'private').
        worker_name: Specific worker name or IP for private queue.
        priority: Priority in public queue ('high', 'medium', 'low').
        start_at: Optional delayed start time string.

    Returns:
        Dict[str, Any]: Submission response with study_id or error detail.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

    if not os.path.exists(yaml_path):
        return {"error": f"YAML file not found at {yaml_path}"}

    async with httpx.AsyncClient() as client:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            filename = os.path.basename(yaml_path)
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

            files = {"config_file": (filename, yaml_content.encode("utf-8"), "application/x-yaml")}
            data = {"mode": mode, "priority": priority}
            if worker_name:
                data["worker_name"] = worker_name
            if start_at:
                data["start_at"] = start_at

            response = await client.post(f"{creds['api_url']}/train", files=files, data=data)
            if response.status_code == 200:
                resp_data = response.json()
                study_id = resp_data.get("study_id")
                if study_id:
                    try:
                        with open(yaml_path, "r", encoding="utf-8") as f:
                            pyaml = yaml.safe_load(f)
                        pyaml["study_id"] = study_id
                        with open(yaml_path, "w", encoding="utf-8") as f:
                            yaml.dump(pyaml, f, default_flow_style=False)
                        resp_data["message"] = f"Training launched. study_id '{study_id}' saved to {yaml_path}"
                    except Exception as exc:
                        resp_data["warning"] = f"Launched, but failed to save study_id: {exc}"
                return {"success": True, "details": resp_data}
            return {"success": False, "error": response.text}
        except Exception as exc:
            return {"error": str(exc)}
