"""Dataset verification and validation tools using Dockerized inspection."""

import json
import shlex
import subprocess
from typing import Any, Dict

from wyolo_mcp.utils.credentials import get_credentials


def check_dataset_path(path: str) -> Dict[str, Any]:
    """Verify if a dataset path exists on the remote Samba CIFS share.

    Args:
        path: Absolute dataset path on CIFS share.

    Returns:
        Dict[str, Any]: Existence flag, directory status, contents preview.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

    cmd = f"""
    /usr/local/bin/mount-cifs.sh >/dev/null 2>&1
    if [ ! -e "{path}" ]; then
        echo '{{"exists": false, "message": "Path does not exist on the CIFS share."}}'
        exit 0
    fi
    if [ -d "{path}" ]; then
        contents=$(ls -1 "{path}" | head -n 20 | tr '\\n' ',' | sed 's/,$//')
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
        "bash", "-c", cmd,
    ]

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except Exception as exc:
        return {"error": f"Failed to execute docker check: {exc}"}


def validate_dataset_advanced(dataset_path: str, task: str = "detect") -> Dict[str, Any]:
    """Validates YOLO dataset structure and format via Docker container.

    Args:
        dataset_path: Absolute dataset path on CIFS share.
        task: Target YOLO task ('detect', 'segment', 'classify', 'obb', 'pose').

    Returns:
        Dict[str, Any]: Validation status, classes count, and names list.
    """
    try:
        creds = get_credentials()
    except Exception as exc:
        return {"error": str(exc)}

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
        "bash", "-c", cmd,
    ]

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split("\n"):
            if line.startswith('{"valid"'):
                return json.loads(line)
        return json.loads(result.stdout.strip())
    except Exception as exc:
        return {"error": f"Failed to execute docker validation: {exc}"}
