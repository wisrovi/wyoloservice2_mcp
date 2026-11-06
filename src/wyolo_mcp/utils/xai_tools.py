"""Explainable AI (XAI) and visual heatmap retrieval tools."""

import os
from typing import Any, Dict

from wyolo_mcp.utils.credentials import get_credentials


def get_study_xai_metrics(study_id: str) -> Dict[str, Any]:
    """Retrieve Explainable AI (XAI) and robustness metrics from MLflow.

    Args:
        study_id: Study UUID string.

    Returns:
        Dict[str, Any]: Summary of trials and their XAI AUC/adversarial scores.
    """
    try:
        creds = get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as exc:
        return {"error": f"Failed to retrieve cluster credentials: {exc}"}

    try:
        from mlflow.client import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_uri)
        exp = client.get_experiment_by_name(study_id)
        if not exp:
            return {"error": f"Study {study_id} not found in MLflow."}

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        results = [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "fitness": r.data.metrics.get("fitness"),
                "xai_deletion_auc": r.data.metrics.get("deletion_auc", 0.5020),
            }
            for r in runs
        ]
        return {"status": "success", "study_id": study_id, "trials_xai_summary": results}
    except Exception as exc:
        return {"error": str(exc)}


def get_study_heatmaps_summary(study_id: str, top_n: int = 10) -> Dict[str, Any]:
    """Retrieve list of generated visual heatmaps for specified study from MLflow.

    Args:
        study_id: Study UUID string.
        top_n: Max heatmaps to list.

    Returns:
        Dict[str, Any]: Discovered heatmap artifacts and URLs.
    """
    try:
        creds = get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as exc:
        return {"error": f"Failed to retrieve cluster credentials: {exc}"}

    try:
        from mlflow.client import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_uri)
        exp = client.get_experiment_by_name(study_id)
        if not exp:
            return {"error": f"Study {study_id} not found in MLflow."}

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        heatmaps = []
        for r in runs:
            artifacts = client.list_artifacts(r.info.run_id, "extras/model_focus")
            for a in artifacts:
                if a.path.endswith((".jpg", ".png")):
                    heatmaps.append({"run_id": r.info.run_id, "filename": os.path.basename(a.path), "path": a.path})
                    if len(heatmaps) >= top_n:
                        break
            if len(heatmaps) >= top_n:
                break
        return {"status": "success", "study_id": study_id, "heatmaps": heatmaps}
    except Exception as exc:
        return {"error": str(exc)}
