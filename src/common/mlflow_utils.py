from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import yaml

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_DATASET_POINTER = "datasets/Market-1501-v15.09.15.dvc"


def get_tracking_uri(config: dict[str, Any] | None = None) -> str:
    configured_uri = None
    if config is not None:
        configured_uri = config.get("logging", {}).get("tracking_uri")
    return os.environ.get("MLFLOW_TRACKING_URI") or configured_uri or DEFAULT_TRACKING_URI


def get_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def get_dvc_revision(pointer_path: str = DEFAULT_DATASET_POINTER) -> str:
    pointer = ROOT / pointer_path
    metadata = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    outputs = metadata.get("outs", [])
    if len(outputs) != 1 or not outputs[0].get("md5"):
        raise ValueError(f"Invalid DVC pointer: {pointer}")
    return str(outputs[0]["md5"])


def collect_hardware_info(device: torch.device | None = None) -> dict[str, Any]:
    import torch

    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
    }
    if selected_device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(selected_device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


def configure_mlflow(experiment_name: str, config: dict[str, Any] | None = None) -> str:
    tracking_uri = get_tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def lineage_tags(
    *,
    git_commit: str | None = None,
    dvc_revision: str | None = None,
) -> dict[str, str]:
    resolved_git_commit = git_commit or get_git_commit()
    resolved_dvc_revision = dvc_revision or get_dvc_revision()
    return {
        "git.commit": resolved_git_commit,
        "mlflow.source.git.commit": resolved_git_commit,
        "dvc.dataset_revision": resolved_dvc_revision,
    }
