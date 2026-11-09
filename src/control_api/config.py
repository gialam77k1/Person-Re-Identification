from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _url(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


@dataclass(frozen=True)
class ControlSettings:
    database_path: Path
    github_repository: str
    github_token: str | None
    github_api_url: str
    preapproval_workflow: str
    retraining_workflow: str
    release_workflow: str
    rollback_workflow: str
    workflow_ref: str
    mlflow_url: str
    prometheus_url: str
    qdrant_url: str
    qdrant_collection: str
    triton_url: str
    triton_model: str
    inference_api_url: str
    public_mlflow_url: str
    public_grafana_url: str
    public_minio_url: str
    public_qdrant_url: str
    public_triton_url: str
    public_inference_api_url: str
    public_github_actions_url: str
    cors_origins: tuple[str, ...]
    callback_token: str
    allowed_datasets: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "ControlSettings":
        database = Path(os.getenv("CONTROL_DATABASE_PATH", "artifacts/control/control.db"))
        if not database.is_absolute():
            database = ROOT / database
        datasets = tuple(
            value.strip()
            for value in os.getenv("CONTROL_ALLOWED_DATASETS", "MSMT17").split(",")
            if value.strip()
        )
        if not datasets:
            raise ValueError("CONTROL_ALLOWED_DATASETS must contain at least one dataset.")
        return cls(
            database_path=database,
            github_repository=os.getenv(
                "CONTROL_GITHUB_REPOSITORY", "gialam77k1/Person-Re-Identification"
            ),
            github_token=os.getenv("CONTROL_GITHUB_TOKEN") or None,
            github_api_url=_url("CONTROL_GITHUB_API_URL", "https://api.github.com"),
            preapproval_workflow=os.getenv(
                "CONTROL_PREAPPROVAL_WORKFLOW", "cross-domain-drift.yml"
            ),
            retraining_workflow=os.getenv(
                "CONTROL_RETRAINING_WORKFLOW", "retrain-v2.yml"
            ),
            release_workflow=os.getenv("CONTROL_RELEASE_WORKFLOW", "release.yml"),
            rollback_workflow=os.getenv("CONTROL_ROLLBACK_WORKFLOW", "rollback.yml"),
            workflow_ref=os.getenv("CONTROL_WORKFLOW_REF", "main"),
            mlflow_url=_url("CONTROL_MLFLOW_URL", "http://mlflow:5000"),
            prometheus_url=_url("CONTROL_PROMETHEUS_URL", "http://prometheus:9090"),
            qdrant_url=_url("CONTROL_QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.getenv("CONTROL_QDRANT_COLLECTION", "reid_reference_v1"),
            triton_url=_url("CONTROL_TRITON_URL", "http://triton-server:8000"),
            triton_model=os.getenv("CONTROL_TRITON_MODEL", "reid_embedding"),
            inference_api_url=_url("CONTROL_INFERENCE_API_URL", "http://inference-api:8080"),
            public_mlflow_url=_url("CONTROL_PUBLIC_MLFLOW_URL", "http://localhost:5000"),
            public_grafana_url=_url("CONTROL_PUBLIC_GRAFANA_URL", "http://localhost:3000"),
            public_minio_url=_url("CONTROL_PUBLIC_MINIO_URL", "http://localhost:9001"),
            public_qdrant_url=_url("CONTROL_PUBLIC_QDRANT_URL", "http://localhost:6333/dashboard"),
            public_triton_url=_url(
                "CONTROL_PUBLIC_TRITON_URL",
                "http://localhost:8000/v2/health/ready",
            ),
            public_inference_api_url=_url(
                "CONTROL_PUBLIC_INFERENCE_API_URL",
                "http://localhost:8080/docs",
            ),
            public_github_actions_url=_url(
                "CONTROL_PUBLIC_GITHUB_ACTIONS_URL",
                "https://github.com/gialam77k1/Person-Re-Identification/actions",
            ),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "CONTROL_CORS_ORIGINS", "http://localhost:3001"
                ).split(",")
                if origin.strip()
            ),
            callback_token=os.getenv(
                "CONTROL_CALLBACK_TOKEN", "change-me-control-callback-token"
            ),
            allowed_datasets=datasets,
        )
