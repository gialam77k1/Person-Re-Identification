from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.mlflow_utils import configure_mlflow

REQUIRED_METRICS = {"rank1", "rank5", "mAP", "mINP", "train_duration_seconds"}
REQUIRED_ARTIFACTS = {
    "config/vit_reid.yaml",
    "config/effective_config.json",
    "metrics/metrics_v1.json",
    "checkpoints/best_model.pth",
    "model/model_embedding.onnx",
    "model/model_embedding.onnx.data",
    "reports/onnx_parity.json",
    "release/preprocessing.json",
    "release/model_manifest.json",
    "provenance/runtime.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MLflow model v1 lineage and artifacts.")
    parser.add_argument("--experiment", default="market1501-vit-bnneck")
    parser.add_argument("--model", default="market1501-vit-bnneck")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--release", default="market1501-vit-bnneck:v1")
    return parser.parse_args()


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    results: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            results.update(list_artifacts_recursive(client, run_id, item.path))
        else:
            results.add(item.path)
    return results


def main() -> None:
    args = parse_args()
    configure_mlflow(args.experiment)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(args.model, args.alias)
    run = client.get_run(model_version.run_id)
    artifact_paths = list_artifacts_recursive(client, run.info.run_id)
    missing_metrics = sorted(REQUIRED_METRICS - set(run.data.metrics))
    missing_artifacts = sorted(REQUIRED_ARTIFACTS - artifact_paths)
    lineage = {
        "git_commit": run.data.tags.get("git.commit"),
        "dvc_dataset_revision": run.data.tags.get("dvc.dataset_revision"),
        "release": run.data.tags.get("release.id"),
    }
    version_lineage = {
        "git_commit": model_version.tags.get("git.commit"),
        "dvc_dataset_revision": model_version.tags.get("dvc.dataset_revision"),
        "release": model_version.tags.get("release.id"),
    }

    with tempfile.TemporaryDirectory(prefix="mlflow-minio-verify-") as temp_dir:
        downloaded_report = client.download_artifacts(
            run.info.run_id,
            "reports/onnx_parity.json",
            temp_dir,
        )
        artifact_downloaded = Path(downloaded_report).is_file()

    passed = (
        str(model_version.version) == "1"
        and model_version.aliases == [args.alias]
        and not missing_metrics
        and not missing_artifacts
        and all(lineage.values())
        and lineage == version_lineage
        and lineage["release"] == args.release
        and artifact_downloaded
    )
    result = {
        "passed": passed,
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_id": run.info.run_id,
        "run_status": run.info.status,
        "registered_model": args.model,
        "version": model_version.version,
        "aliases": model_version.aliases,
        "lineage": lineage,
        "missing_metrics": missing_metrics,
        "missing_artifacts": missing_artifacts,
        "artifact_downloaded": artifact_downloaded,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
