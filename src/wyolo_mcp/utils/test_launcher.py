"""Private test training launcher for invoker verification."""

from typing import Any, Dict
import httpx
import yaml

from wyolo_mcp.utils.credentials import get_credentials

CONFIG_SEGMENTATION: str = """
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
    model: ["choice", "yolo26n-seg.pt"]
    train:
      imgsz: ["choice", 640]
"""

CONFIG_DETECTION: str = """
model: "yolo26n.pt"
type: "yolo"
train:
  batch: -1
  data: "/examples/coco8/coco8.yaml"
  epochs: 2
  imgsz: 640
sweeper:
  version: 1
  algorithm: optuna
  direction: maximize
  study_name: "coco8_detection"
  fitness: "metrics/mAP50-95(B)"
  tune: false
  sampler: "TPESampler"
  n_trials: 1
"""


async def launch_private_test_training(task_type: str, ip_address: str) -> Dict[str, Any]:
    """Launch a private base test training study targeted directly to a specific worker IP.

    Args:
        task_type: Task to test ('detection' or 'segmentation').
        ip_address: Physical IP address of the target invoker worker.

    Returns:
        Dict[str, Any]: Submission result from API.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

    raw_yaml = CONFIG_SEGMENTATION if task_type == "segmentation" else CONFIG_DETECTION
    config_dict = yaml.safe_load(raw_yaml)
    config_dict["sweeper"]["debug"] = ip_address
    config_dict["debug"] = ip_address
    config_dict["sweeper"]["target_worker_queue"] = ip_address

    final_yaml = yaml.dump(config_dict, default_flow_style=False)
    filename = f"test_{task_type}_{ip_address.replace('.', '_')}.yaml"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            files = {"config_file": (filename, final_yaml.encode("utf-8"), "application/x-yaml")}
            data = {"mode": "private", "worker_name": ip_address}
            res = await client.post(f"{creds['api_url']}/train", files=files, data=data)
            if res.status_code == 200:
                return {"success": True, "details": res.json()}
            return {"success": False, "error": res.text}
        except Exception as exc:
            return {"error": str(exc)}
