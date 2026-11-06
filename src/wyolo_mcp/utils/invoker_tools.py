"""Invoker queue management and broadcast pull tools."""

import time
from datetime import datetime as dt_pause, timedelta as td_pause
from typing import Any, Dict, Optional

from wyolo_mcp.utils.credentials import get_credentials


async def manage_invoker_queues(
    worker_ip: str,
    action: str,
    mode: str = "temporal",
    hours: float = 4.0,
    until_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Control the public queue consumption mode of a specific invoker node remotely.

    Args:
        worker_ip: IP address of the target invoker worker.
        action: Action to perform ('pause' or 'resume').
        mode: Pause mode ('temporal' or 'perpetual').
        hours: Number of hours for temporal pause.
        until_time: Optional target wake-up time in 'HH:MM' 24h format.

    Returns:
        Dict[str, Any]: Result of Redis update and Celery signaling.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": f"Failed to retrieve cluster credentials: {exc}"}

    redis_host = creds.get("control_host")
    redis_port = 23437
    redis_db = 0

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
                redis_msg = f"Set state to 'paused_temporal' until {until_time or ''} ({hours} hours)"
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
    except Exception as exc:
        return {"error": f"Failed to update state in Redis: {exc}"}

    try:
        from celery import Celery

        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        app = Celery("ml_cluster", broker=redis_url, backend=redis_url)
        node_name = f"celery@wyolo_invoker_{worker_ip}"
        public_queues = ["gpus_high", "gpus_medium", "gpus_low"]
        results = []
        for queue in public_queues:
            if action == "pause":
                resp = app.control.cancel_consumer(queue, destination=[node_name], reply=True, timeout=2.0)
            else:
                resp = app.control.add_consumer(queue, destination=[node_name], reply=True, timeout=2.0)
            results.append((queue, resp))

        success = any(bool(r) for _, r in results)
        return {
            "success": True,
            "redis_update": redis_msg,
            "celery_signals": {"sent_to": node_name, "status": "responsive" if success else "offline"},
        }
    except Exception as exc:
        return {"success": True, "redis_update": redis_msg, "celery_signals": {"warning": str(exc)}}


def trigger_broadcast_docker_pull(image_name: str) -> Dict[str, Any]:
    """Broadcast force_docker_pull command to all worker nodes via Celery.

    Args:
        image_name: Docker image name with tag to pull.

    Returns:
        Dict[str, Any]: Broadcast responses from workers.
    """
    try:
        creds = get_credentials()
        redis_host = creds.get("control_host")
        redis_url = f"redis://{redis_host}:23437/0"
    except Exception as exc:
        return {"success": False, "error": f"Failed to retrieve cluster credentials: {exc}"}

    try:
        from celery import Celery

        app = Celery("tasks", broker=redis_url)
        responses = app.control.broadcast("force_docker_pull", arguments={"image_name": image_name}, reply=True, timeout=8.0)
        if not responses:
            return {"success": False, "message": "No active workers responded."}

        formatted = {}
        for resp in responses:
            for node, details in resp.items():
                formatted[node] = details
        return {"success": True, "image": image_name, "responses": formatted}
    except ImportError:
        return {"success": False, "error": "Required library 'celery' is not installed."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
