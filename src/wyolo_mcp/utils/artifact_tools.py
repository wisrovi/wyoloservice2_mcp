"""Study artifact download tools using MLflow client."""

import os
import shutil
import tempfile
from typing import Any, Dict, Optional

from wyolo_mcp.utils.credentials import get_credentials


def download_mlflow_study_artifacts(study_id: str) -> Dict[str, Any]:
    """Download all artifacts for an entire NeuralForge Study from MLflow.

    Args:
        study_id: Target study identifier.

    Returns:
        Dict[str, Any]: Destination zip path and execution summary.
    """
    try:
        creds = get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as exc:
        return {"error": f"Failed to retrieve cluster credentials: {exc}"}

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
            return {"error": f"Could not find MLflow experiment: {study_id}"}

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            return {"error": f"No trials found for study {study_id}"}

        temp_dir = tempfile.mkdtemp(prefix=f"study_{study_id}_")
        for run in runs:
            run_dir = os.path.join(temp_dir, f"trial_{run.info.run_id}")
            os.makedirs(run_dir, exist_ok=True)
            client.download_artifacts(run.info.run_id, "", dst_path=run_dir)

        zip_path = os.path.join(tempfile.gettempdir(), f"study_artifacts_{study_id}")
        shutil.make_archive(zip_path, "zip", temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": True,
            "zip_path": f"{zip_path}.zip",
            "message": f"Artifacts for {len(runs)} trials downloaded and zipped successfully.",
        }
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as exc:
        return {"error": f"Failed to download study artifacts from MLflow: {exc}"}


def download_study_artifact(
    study_id: str,
    artifact_type: str = "best_weights",
    trial_id: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Download a specific artifact from MLflow without downloading full zip.

    Args:
        study_id: Target study UUID.
        artifact_type: Type of artifact ('best_weights', 'llm_report', 'xai_heatmaps', 'latex_tables').
        trial_id: Optional specific trial/run ID.
        output_path: Optional destination path on local disk.

    Returns:
        Dict[str, Any]: Download status and local file path.
    """
    try:
        creds = get_credentials()
        mlflow_uri = f"http://{creds['control_host']}:5000"
    except Exception as exc:
        return {"error": f"Failed to retrieve cluster credentials: {exc}"}

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

        target_run = runs[0]
        if trial_id:
            for r in runs:
                if r.info.run_id == trial_id:
                    target_run = r
                    break
        else:
            runs.sort(
                key=lambda r: r.data.metrics.get("fitness", r.data.metrics.get("metrics/mAP50-95(B)", 0.0)),
                reverse=True,
            )
            target_run = runs[0]

        run_id = target_run.info.run_id
        temp_dir = tempfile.mkdtemp(prefix=f"artifact_{study_id}_")

        if artifact_type == "best_weights":
            dest_dir = client.download_artifacts(run_id, "weights/best.pt", dst_path=temp_dir)
            target_file = dest_dir if os.path.exists(dest_dir) else os.path.join(client.download_artifacts(run_id, "weights", dst_path=temp_dir), "best.pt")
            final_dest = output_path or os.path.abspath(f"best_{study_id}.pt")
            if os.path.exists(target_file):
                shutil.copy2(target_file, final_dest)
                return {"success": True, "artifact_type": artifact_type, "file_path": final_dest, "run_id": run_id}
            return {"error": f"best.pt not found in run {run_id}"}

        elif artifact_type == "llm_report":
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

        return {"error": f"Unknown artifact_type '{artifact_type}'."}
    except ImportError:
        return {"error": "Required library 'mlflow' is not installed in the MCP environment."}
    except Exception as exc:
        return {"error": f"Failed to download study artifact: {exc}"}
