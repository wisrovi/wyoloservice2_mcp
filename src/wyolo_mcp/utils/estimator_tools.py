"""Study duration estimation utilities for NeuralForge MCP."""

from typing import Any, Dict
import httpx

from wyolo_mcp.utils.credentials import get_credentials


async def estimate_study_duration(dataset_path: str, epochs: int = 100, model: str = "yolo26n.pt") -> Dict[str, Any]:
    """Estimate expected training duration for given dataset and model.

    Args:
        dataset_path: Dataset path string.
        epochs: Number of epochs.
        model: Target model architecture.

    Returns:
        Dict[str, Any]: Duration estimation in minutes and hours.
    """
    host = None
    try:
        creds = get_credentials()
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
    }
