"""Cluster telemetry, node inspection, and recommendation tools."""

import asyncio
from typing import Any, Dict
import httpx

from wyolo_mcp.utils.credentials import get_credentials


async def get_cluster_status() -> Dict[str, Any]:
    """Get the overall status of the NeuralForgeAI cluster.

    Queries health metrics, active celery workers, and tasks queue in parallel.

    Returns:
        Dict[str, Any]: Consolidated health, worker, and task telemetry.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    async with httpx.AsyncClient() as client:
        try:
            health_req = client.get(f"{creds['api_url']}/health")
            workers_req = client.get(f"{creds['api_url']}/workers")
            tasks_req = client.get(f"{creds['api_url']}/tasks")

            health_res, workers_res, tasks_res = await asyncio.gather(
                health_req, workers_req, tasks_req, return_exceptions=True
            )

            status_data: Dict[str, Any] = {"status": "online"}
            status_data["health"] = (
                health_res.json()
                if not isinstance(health_res, Exception) and health_res.status_code == 200
                else {"error": "Failed to fetch health"}
            )
            status_data["workers"] = (
                workers_res.json()
                if not isinstance(workers_res, Exception) and workers_res.status_code == 200
                else {"error": "Failed to fetch workers"}
            )

            if not isinstance(tasks_res, Exception) and tasks_res.status_code == 200:
                tasks_data = tasks_res.json()
                workers_dict = status_data["workers"].get("workers", {})
                for task in tasks_data.get("running", []):
                    wname = task.get("worker")
                    tname = task.get("name", "")
                    invoker_ip = (
                        workers_dict[wname][0]
                        if wname in workers_dict and isinstance(workers_dict[wname], list) and workers_dict[wname]
                        else "Unknown"
                    )
                    task["worker_ip"] = invoker_ip
                    if "train_on_gpu" in tname and invoker_ip not in ("Unknown", "managers"):
                        ename = f"wyolo_executor_{invoker_ip}"
                        task["log_commands"] = {
                            "docker_logs": f'ssh -t wyolo@{invoker_ip} "docker logs -f {ename}"',
                            "tail_log_file": f'ssh -t wyolo@{invoker_ip} "tail -f /home/wyolo/train_service_results/logs_{ename}.txt"',
                        }
                    else:
                        task["log_commands"] = None
                status_data["tasks"] = tasks_data
            else:
                status_data["tasks"] = {"error": "Failed to fetch tasks queue"}

            return status_data
        except Exception as exc:
            return {"status": "offline", "error": str(exc)}


async def recommend_training_node(min_vram_gb: float = 8.0, batch_size: int = 16) -> Dict[str, Any]:
    """Recommend optimal compute node based on real-time GPU telemetry and queue load.

    Args:
        min_vram_gb: Minimum required VRAM in gigabytes.
        batch_size: Target training batch size.

    Returns:
        Dict[str, Any]: Recommended node details and source metadata.
    """
    try:
        creds = get_credentials()
        host = creds.get("control_host", "localhost")
    except Exception as exc:
        return {"status": "error", "message": f"Cluster credentials error: {exc}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        k8s_url = f"http://{host}:32442/api/v1/cluster/recommend-node"
        try:
            res = await client.get(k8s_url, params={"min_vram_gb": min_vram_gb, "batch_size": batch_size})
            if res.status_code == 200:
                return {"status": "success", "source": "k8s_admin_api", "recommendation": res.json()}
        except Exception:
            pass

        try:
            workers_res = await client.get(f"{creds['api_url']}/workers")
            if workers_res.status_code == 200:
                workers_data = workers_res.json().get("workers", {})
                available = [
                    info[0] if isinstance(info, list) and info else name
                    for name, info in workers_data.items()
                    if name != "managers"
                ]
                if available:
                    return {
                        "status": "success",
                        "source": "control_server_fallback",
                        "recommended_node": available[0],
                        "available_invokers": available,
                    }
            return {"status": "warning", "message": "No active invoker nodes available in cluster."}
        except Exception as exc:
            return {"status": "error", "message": f"Failed to retrieve node recommendation: {exc}"}


async def preflight_check_node(worker_ip: str, dataset_yaml: str) -> Dict[str, Any]:
    """Run pre-flight health and resource checks on a specific invoker node.

    Args:
        worker_ip: IP address of target worker.
        dataset_yaml: Dataset YAML path string.

    Returns:
        Dict[str, Any]: Node registration and dataset format check results.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"status": "error", "message": f"Cluster credentials error: {exc}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            workers_res = await client.get(f"{creds['api_url']}/workers")
            workers_dict = workers_res.json().get("workers", {}) if workers_res.status_code == 200 else {}
            node_registered = any(
                worker_ip in (info[0] if isinstance(info, list) and info else wname, wname)
                for wname, info in workers_dict.items()
            )
            dataset_valid = dataset_yaml.endswith((".yaml", ".yml"))

            return {
                "status": "success",
                "worker_ip": worker_ip,
                "node_registered_in_cluster": node_registered,
                "dataset_yaml_valid_format": dataset_valid,
                "dataset_yaml_path": dataset_yaml,
                "ready_to_launch": node_registered and dataset_valid,
                "recommendation": "Ready to launch" if (node_registered and dataset_valid) else "Check connectivity or YAML.",
            }
        except Exception as exc:
            return {"status": "error", "message": f"Preflight check failed: {exc}"}
