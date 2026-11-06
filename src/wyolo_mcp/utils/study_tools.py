"""Study telemetry and cancellation tools."""

from typing import Any, Dict
import httpx

from wyolo_mcp.utils.credentials import get_credentials


async def get_study_details(study_id: str) -> Dict[str, Any]:
    """Get detailed telemetry and status of a specific YOLO training study.

    Args:
        study_id: Study UUID string.

    Returns:
        Dict[str, Any]: Study progress, active worker, and trial metrics.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{creds['api_url']}/study/{study_id}")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Study not found or API error: {response.text}"}
        except Exception as exc:
            return {"error": str(exc)}


async def cancel_study(study_id: str) -> Dict[str, Any]:
    """Cancel a running training study by its ID.

    Args:
        study_id: Target study UUID.

    Returns:
        Dict[str, Any]: Confirmation response or error.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{creds['api_url']}/study/{study_id}/cancel")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to cancel: {response.text}"}
        except Exception as exc:
            return {"error": str(exc)}
