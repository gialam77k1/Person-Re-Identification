from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.mlflow_utils import collect_hardware_info, configure_mlflow, lineage_tags
from src.common.utils import save_json

DEFAULT_MANIFEST = "artifacts/pipeline/market1501-vit-bnneck-v1-dvc/release/model_manifest.json"
METRICS_PATH = "artifacts/market1501/market1501-vit-staged-v2/metrics/metrics_v1.json"
EFFECTIVE_CONFIG_PATH = "artifacts/market1501/market1501-vit-staged-v2/logs/effective_config.json"
TRAIN_LOG_PATH = "artifacts/market1501/market1501-vit-staged-v2/logs/train.log"
CHECKPOINT_PATH = "artifacts/market1501/market1501-vit-bnneck-v1/checkpoints/best_model.pth"
ONNX_DIR = "artifacts/market1501/market1501-vit-bnneck-v1-onnx/exports"
PARITY_PATH = "artifacts/market1501/market1501-vit-bnneck-v1-onnx-parity/metrics/onnx_parity.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish frozen model v1 to MLflow.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--validation-report",
        required=True,
        help="Validation gate report. The release is logged regardless of PASS/FAIL.",
    )
    parser.add_argument("--experiment", default="market1501-vit-bnneck")
    parser.add_argument("--alias", default="champion")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run-id", help="Attach release artifacts to an existing training run.")
    target.add_argument(
        "--run-metadata",
        help="Read the target training run ID from its mlflow_run.json file.",
    )
    return parser.parse_args()


def resolve(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def estimate_legacy_training_duration() -> float:
    start_path = resolve(EFFECTIVE_CONFIG_PATH)
    end_path = resolve(TRAIN_LOG_PATH)
    return max(0.0, end_path.stat().st_mtime - start_path.stat().st_mtime)


def release_metrics(manifest: dict) -> dict[str, float]:
    quality = manifest["quality_metrics"]
    if "base" not in quality or "reranked" not in quality:
        return {
            key: float(quality[key])
            for key in ("rank1", "rank5", "mAP", "mINP")
            if quality.get(key) is not None
        }
    base = quality["base"]
    reranked = quality["reranked"]
    return {
        "rank1": float(reranked["rank1"]),
        "rank5": float(reranked["rank5"]),
        "mAP": float(reranked["mAP"]),
        "mINP": float(reranked["mINP"]),
        "rank1_base": float(base["rank1"]),
        "rank5_base": float(base["rank5"]),
        "mAP_base": float(base["mAP"]),
        "mINP_base": float(base["mINP"]),
        "rank1_rerank": float(reranked["rank1"]),
        "rank5_rerank": float(reranked["rank5"]),
        "mAP_rerank": float(reranked["mAP"]),
        "mINP_rerank": float(reranked["mINP"]),
    }


def find_release_run(client: MlflowClient, experiment_id: str, release_id: str):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`release.id` = '{release_id}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0] if runs else None


def validation_allows_promotion(manifest: dict, report: dict) -> bool:
    manifest_release = manifest["release"]["release_id"]
    if report.get("release_id") != manifest_release:
        raise ValueError(
            f"Validation report release {report.get('release_id')!r} does not match {manifest_release!r}."
        )
    return report.get("status") == "PASS" and report.get("passed") is True


def record_validation_result(
    run_id: str,
    report_path: Path,
    report: dict,
) -> None:
    client = MlflowClient()
    client.set_tag(run_id, "validation.status", str(report.get("status", "UNKNOWN")))
    client.set_tag(run_id, "validation.release_id", str(report.get("release_id", "")))
    client.log_metric(run_id, "validation_passed", 1.0 if report.get("passed") is True else 0.0)
    client.log_metric(run_id, "validation_failed_checks", float(len(report.get("failed_checks", []))))
    client.log_artifact(run_id, str(report_path), artifact_path="validation")


def publish_release_run(
    manifest: dict,
    manifest_path: Path,
    experiment_name: str,
    target_run_id: str | None = None,
) -> str:
    configure_mlflow(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")

    release_id = manifest["release"]["release_id"]
    client = MlflowClient()
    existing_run = (
        client.get_run(target_run_id)
        if target_run_id is not None
        else find_release_run(client, experiment.experiment_id, release_id)
    )
    if existing_run is not None:
        if existing_run.data.tags.get("release.id") == release_id:
            return existing_run.info.run_id
        if target_run_id is None:
            return existing_run.info.run_id

    source = manifest["source"]
    git_commit = source["git_commit"]
    dvc_revision = source.get("dataset_dvc_revision")
    if not dvc_revision:
        dvc_revision = next(
            asset["dvc_md5"]
            for asset in manifest["data_versioning"]["assets"]
            if asset["role"] == "market1501_dataset"
        )
    try:
        duration_seconds = estimate_legacy_training_duration()
    except FileNotFoundError:
        duration_seconds = 0.0
    runtime = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": duration_seconds,
        "training_duration_source": "artifact_mtime_estimate",
        "hardware": collect_hardware_info(),
        "hardware_source": "publication_host; legacy training GPU was not recorded",
    }
    runtime_path = save_json(runtime, "artifacts/mlflow-publish/v1/runtime.json")

    run_context = (
        mlflow.start_run(run_id=target_run_id)
        if target_run_id is not None
        else mlflow.start_run(run_name="market1501-vit-bnneck-v1-release")
    )
    with run_context as run:
        mlflow.set_tags(
            {
                **lineage_tags(git_commit=git_commit, dvc_revision=dvc_revision),
                "run.kind": "model-release",
                "release.id": release_id,
                "release.version": manifest["release"].get("version", "unversioned"),
                "checkpoint.epoch": str(source["checkpoint_epoch"]),
                "training.code_commit": source.get("training_code_commit", git_commit),
                "training.duration_source": runtime["training_duration_source"],
                "hardware.source": runtime["hardware_source"],
            }
        )
        mlflow.log_params(
            {
                "architecture": manifest["model"]["architecture"],
                "embedding_dim": manifest["model"]["embedding_dim"],
                "dataset": manifest["model"]["dataset"],
                "checkpoint_epoch": source["checkpoint_epoch"],
                "onnx_opset": manifest["model"].get("onnx_opset"),
                "input_shape": "x".join(map(str, manifest["contract"]["input_shape"])),
                "output_shape": "x".join(map(str, manifest["contract"]["output_shape"])),
            }
        )
        mlflow.log_metrics(
            {
                **release_metrics(manifest),
                "train_duration_seconds": duration_seconds,
            }
        )
        mlflow.log_artifact(str(resolve(source["config_path"])), artifact_path="config")
        artifact_paths = {
            "pytorch_checkpoint": "checkpoints",
            "onnx_graph": "model",
            "onnx_external_weights": "model",
            "preprocessing_contract": "release",
            "reference_embeddings": "reference",
            "reference_pids": "reference",
            "reference_camids": "reference",
            "reference_paths": "reference",
        }
        for item in manifest["files"]:
            artifact_path = artifact_paths.get(item["role"])
            if artifact_path:
                mlflow.log_artifact(
                    str(resolve(item["repository_path"])),
                    artifact_path=artifact_path,
                )
        mlflow.log_artifact(str(manifest_path), artifact_path="release")
        mlflow.log_artifact(str(runtime_path), artifact_path="provenance")
        return run.info.run_id


def register_release(
    manifest: dict,
    run_id: str,
    alias: str,
    validation_report: dict,
) -> tuple[str, bool]:
    client = MlflowClient()
    model_name = manifest["release"]["name"]
    release_id = manifest["release"]["release_id"]
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            model_name,
            description="ViT-B/16 BNNeck model for Market-1501 person re-identification.",
            tags={"task": "person-re-identification"},
        )

    existing_versions = client.search_model_versions(f"name = '{model_name}'")
    model_version = next(
        (item for item in existing_versions if item.tags.get("release.id") == release_id),
        None,
    )
    if model_version is None:
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            tags={
                "release.id": release_id,
                "git.commit": manifest["source"]["git_commit"],
                "validation.status": validation_report["status"],
                "dvc.dataset_revision": manifest["source"].get("dataset_dvc_revision", "unknown"),
            },
            description="Frozen market1501-vit-bnneck:v1 ONNX release.",
        )
    passed = validation_allows_promotion(manifest, validation_report)
    client.set_model_version_tag(
        model_name,
        model_version.version,
        "validation.status",
        validation_report["status"],
    )
    if passed:
        client.set_registered_model_alias(model_name, alias, model_version.version)
    return str(model_version.version), passed


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_report_path = resolve(args.validation_report)
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    validation_allows_promotion(manifest, validation_report)
    target_run_id = args.run_id
    if args.run_metadata:
        run_metadata = json.loads(resolve(args.run_metadata).read_text(encoding="utf-8"))
        target_run_id = run_metadata["run_id"]
    run_id = publish_release_run(
        manifest,
        manifest_path,
        args.experiment,
        target_run_id=target_run_id,
    )
    record_validation_result(run_id, validation_report_path, validation_report)
    version, promoted = register_release(
        manifest,
        run_id,
        args.alias,
        validation_report,
    )
    result = {
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_id": run_id,
        "registered_model": manifest["release"]["name"],
        "version": version,
        "alias": args.alias if promoted else None,
        "promoted": promoted,
        "validation_status": validation_report["status"],
        "release": manifest["release"]["release_id"],
    }
    save_json(result, "artifacts/mlflow-publish/v1/publish_result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
