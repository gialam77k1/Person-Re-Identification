from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import save_json


REQUIRED_ARTIFACT_ROLES = {
    "pytorch_checkpoint",
    "onnx_graph",
    "onnx_external_weights",
    "preprocessing_contract",
    "reference_embeddings",
    "reference_pids",
    "reference_camids",
    "reference_paths",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a model candidate before packaging or promotion.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--thresholds", default="configs/model_validation.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipeline-params", default="params.yaml")
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--evaluation-metrics", required=True)
    parser.add_argument("--onnx-manifest", required=True)
    parser.add_argument("--parity-report", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--preprocessing-output")
    parser.add_argument("--deployment-metrics")
    parser.add_argument(
        "--mlflow-run-metadata",
        help="Log the PASS/FAIL report to the MLflow run referenced by this JSON file.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Write a FAIL report but return exit code 0; intended only for gate tests.",
    )
    return parser.parse_args()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def repository_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_spec(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Artifact is empty: {path}")
    return {
        "role": role,
        "repository_path": repository_path(path),
        "size_bytes": size,
        "sha256": sha256_file(path),
    }


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    actual: Any = None,
    expected: Any = None,
    message: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "message": message,
        }
    )


def metric_at_least(
    checks: list[dict[str, Any]], name: str, value: Any, minimum: float
) -> None:
    passed = finite_number(value) and float(value) >= minimum
    add_check(checks, name, passed, actual=value, expected={"min": minimum})


def validate_deployment_metrics(
    checks: list[dict[str, Any]],
    deployment_metrics: dict[str, Any] | None,
    evaluation_metrics: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    if deployment_metrics is None:
        add_check(
            checks,
            "deployment_metrics_present",
            not bool(policy.get("required", True)),
            actual=False,
            expected={"required": bool(policy.get("required", True))},
            message="Deployment metrics report is required before promotion.",
        )
        return

    add_check(checks, "deployment_metrics_present", True, actual=True, expected=True)
    mappings = (
        ("rank1", "rank1_base", "rank1_max_drop"),
        ("mAP", "mAP_base", "map_max_drop"),
        ("mINP", "mINP_base", "minp_max_drop"),
    )
    for deployed_key, candidate_key, policy_key in mappings:
        deployed_baseline = deployment_metrics.get(deployed_key)
        candidate = evaluation_metrics.get(candidate_key, evaluation_metrics.get(deployed_key))
        allowed_drop = float(policy[policy_key])
        drop = (
            float(deployed_baseline) - float(candidate)
            if finite_number(deployed_baseline) and finite_number(candidate)
            else None
        )
        passed = drop is not None and drop <= allowed_drop
        add_check(
            checks,
            f"deployment_{deployed_key}_regression",
            passed,
            actual={
                "candidate": candidate,
                "deployment_baseline": deployed_baseline,
                "drop": drop,
            },
            expected={"max_drop": allowed_drop},
        )


def build_validation(
    *,
    release_id: str,
    source_git_commit: str,
    thresholds_path: Path,
    config_path: Path,
    pipeline_params_path: Path,
    data_manifest_path: Path,
    training_metrics_path: Path,
    evaluation_metrics_path: Path,
    onnx_manifest_path: Path,
    parity_report_path: Path,
    reference_manifest_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    preprocessing_path: Path,
    deployment_metrics_path: Path | None,
    report_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = read_yaml(thresholds_path)
    data_manifest = read_json(data_manifest_path)
    training_metrics = read_json(training_metrics_path)
    evaluation = read_json(evaluation_metrics_path)
    onnx_manifest = read_json(onnx_manifest_path)
    parity = read_json(parity_report_path)
    reference = read_json(reference_manifest_path)
    preprocessing = read_json(preprocessing_path)
    deployment = read_json(deployment_metrics_path) if deployment_metrics_path else None
    checks: list[dict[str, Any]] = []

    quality = thresholds["quality"]
    metric_at_least(checks, "quality_rank1", evaluation.get("rank1"), float(quality["rank1_min"]))
    metric_at_least(checks, "quality_mAP", evaluation.get("mAP"), float(quality["map_min"]))
    metric_at_least(checks, "quality_mINP", evaluation.get("mINP"), float(quality["minp_min"]))

    add_check(checks, "dataset_validation", bool(data_manifest.get("passed")), actual=data_manifest.get("passed"), expected=True)
    onnx_policy = thresholds["onnx"]
    add_check(checks, "pytorch_onnx_parity", bool(parity.get("passed")), actual=parity.get("passed"), expected=True)
    metric_at_least(
        checks,
        "onnx_cosine_similarity",
        parity.get("min_cosine_similarity"),
        float(onnx_policy["cosine_min"]),
    )
    max_abs_error = parity.get("max_abs_error")
    add_check(
        checks,
        "onnx_max_abs_error",
        finite_number(max_abs_error) and float(max_abs_error) <= float(onnx_policy["max_abs_error"]),
        actual=max_abs_error,
        expected={"max": float(onnx_policy["max_abs_error"])},
    )

    contract = thresholds["contract"]
    expected_input = list(contract["input_shape"])
    expected_output = list(contract["output_shape"])
    observed_output = list(parity.get("output_shape", []))
    add_check(checks, "input_contract", onnx_manifest.get("input_shape") == expected_input and preprocessing.get("input", {}).get("shape") == expected_input, actual={"onnx": onnx_manifest.get("input_shape"), "preprocessing": preprocessing.get("input", {}).get("shape")}, expected=expected_input)
    add_check(checks, "output_contract", observed_output == expected_output and onnx_manifest.get("embedding_dim") == contract["embedding_dim"] and parity.get("output_name") == contract["output_name"], actual={"shape": observed_output, "dimension": onnx_manifest.get("embedding_dim"), "name": parity.get("output_name")}, expected={"shape": expected_output, "dimension": contract["embedding_dim"], "name": contract["output_name"]})
    add_check(checks, "input_name_contract", parity.get("input_name") == contract["input_name"], actual=parity.get("input_name"), expected=contract["input_name"])

    embeddings_path = resolve(reference["embeddings_path"])
    pids_path = resolve(reference["pids_path"])
    camids_path = resolve(reference["camids_path"])
    paths_path = resolve(reference["paths_path"])
    embeddings = np.load(embeddings_path, mmap_mode="r")
    pids = np.load(pids_path, mmap_mode="r")
    camids = np.load(camids_path, mmap_mode="r")
    paths = np.load(paths_path, mmap_mode="r")
    expected_dim = int(contract["embedding_dim"])
    expected_samples = int(reference["num_samples"])
    add_check(checks, "embedding_shape", embeddings.ndim == 2 and embeddings.shape == (expected_samples, expected_dim) and int(reference.get("feature_dim", -1)) == expected_dim, actual={"array": list(embeddings.shape), "manifest_feature_dim": reference.get("feature_dim")}, expected=[expected_samples, expected_dim])
    finite_embeddings = bool(np.isfinite(embeddings).all())
    add_check(checks, "embedding_finite", finite_embeddings, actual="all finite" if finite_embeddings else "contains NaN or Inf", expected="all finite")
    add_check(checks, "reference_metadata_alignment", pids.shape == camids.shape == paths.shape == (expected_samples,), actual={"pids": list(pids.shape), "camids": list(camids.shape), "paths": list(paths.shape)}, expected=[expected_samples])

    external_weights = onnx_path.with_name(f"{onnx_path.name}.data")
    artifacts = [
        artifact_spec("pytorch_checkpoint", checkpoint_path),
        artifact_spec("onnx_graph", onnx_path),
        artifact_spec("onnx_external_weights", external_weights),
        artifact_spec("preprocessing_contract", preprocessing_path),
        artifact_spec("reference_embeddings", embeddings_path),
        artifact_spec("reference_pids", pids_path),
        artifact_spec("reference_camids", camids_path),
        artifact_spec("reference_paths", paths_path),
    ]
    artifact_roles = {item["role"] for item in artifacts}
    checksum_complete = artifact_roles == REQUIRED_ARTIFACT_ROLES and all(
        len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in artifacts
    )
    add_check(checks, "manifest_checksums_complete", checksum_complete, actual=sorted(artifact_roles), expected=sorted(REQUIRED_ARTIFACT_ROLES))
    validate_deployment_metrics(checks, deployment, evaluation, thresholds["deployment"])

    passed = all(check["passed"] for check in checks)
    release_name, _, release_version = release_id.partition(":")
    manifest = {
        "schema_version": 2,
        "release": {
            "name": release_name,
            "version": release_version or "unversioned",
            "release_id": release_id,
            "status": "validated" if passed else "rejected",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "source": {
            "git_commit": source_git_commit,
            "dataset_dvc_revision": data_manifest.get("dataset_dvc_revision"),
            "checkpoint_epoch": training_metrics.get("best_epoch"),
            "config_path": repository_path(config_path),
            "config_sha256": sha256_file(config_path),
            "pipeline_params_path": repository_path(pipeline_params_path),
            "pipeline_params_sha256": sha256_file(pipeline_params_path),
            "validation_policy_path": repository_path(thresholds_path),
            "validation_policy_sha256": sha256_file(thresholds_path),
        },
        "contract": {
            "input_name": contract["input_name"],
            "input_shape": expected_input,
            "output_name": contract["output_name"],
            "output_shape": expected_output,
            "embedding_dim": expected_dim,
            "preprocessing_path": repository_path(preprocessing_path),
        },
        "model": {
            "task": "person-re-identification",
            "dataset": evaluation.get("dataset", "market1501"),
            "architecture": "vit_b_16",
            "embedding_dim": expected_dim,
            "onnx_opset": onnx_manifest.get("opset"),
        },
        "quality_metrics": {key: evaluation.get(key) for key in ("rank1", "rank5", "mAP", "mINP")},
        "deployment_metrics": deployment,
        "files": artifacts,
        "validation": {
            "status": "PASS" if passed else "FAIL",
            "report_path": repository_path(report_path),
            "thresholds_path": repository_path(thresholds_path),
        },
    }
    report = {
        "schema_version": 1,
        "release_id": release_id,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "validated_artifacts": artifacts,
        "manifest_path": repository_path(manifest_path),
    }
    return report, manifest


def log_validation_to_mlflow(
    report_path: Path,
    report: dict[str, Any],
    run_metadata_path: Path,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    metadata = read_json(run_metadata_path)
    tracking_uri = metadata.get("tracking_uri")
    run_id = metadata.get("run_id")
    if not tracking_uri or not run_id:
        raise ValueError("MLflow run metadata must contain tracking_uri and run_id.")

    mlflow.set_tracking_uri(str(tracking_uri))
    client = MlflowClient()
    client.set_tag(run_id, "validation.status", report["status"])
    client.set_tag(run_id, "validation.release_id", report["release_id"])
    client.log_metric(run_id, "validation_passed", 1.0 if report["passed"] else 0.0)
    client.log_metric(run_id, "validation_failed_checks", float(len(report["failed_checks"])))
    client.log_artifact(run_id, str(report_path), artifact_path="validation")


def main() -> None:
    args = parse_args()
    report_path = resolve(args.output)
    manifest_path = resolve(args.manifest_output)
    preprocessing_path = resolve(args.preprocessing)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.preprocessing_output:
            preprocessing_output = resolve(args.preprocessing_output)
            preprocessing_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preprocessing_path, preprocessing_output)
            preprocessing_path = preprocessing_output
        report, manifest = build_validation(
            release_id=args.release_id,
            source_git_commit=args.source_git_commit,
            thresholds_path=resolve(args.thresholds),
            config_path=resolve(args.config),
            pipeline_params_path=resolve(args.pipeline_params),
            data_manifest_path=resolve(args.data_manifest),
            training_metrics_path=resolve(args.training_metrics),
            evaluation_metrics_path=resolve(args.evaluation_metrics),
            onnx_manifest_path=resolve(args.onnx_manifest),
            parity_report_path=resolve(args.parity_report),
            reference_manifest_path=resolve(args.reference_manifest),
            checkpoint_path=resolve(args.checkpoint),
            onnx_path=resolve(args.onnx),
            preprocessing_path=preprocessing_path,
            deployment_metrics_path=resolve(args.deployment_metrics) if args.deployment_metrics else None,
            report_path=report_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "release_id": args.release_id,
            "status": "FAIL",
            "passed": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": [
                {
                    "name": "gate_execution",
                    "status": "FAIL",
                    "passed": False,
                    "actual": f"{type(exc).__name__}: {exc}",
                    "expected": "validation completes without missing or malformed inputs",
                    "message": "Validation could not complete.",
                }
            ],
            "failed_checks": ["gate_execution"],
            "manifest_path": repository_path(manifest_path),
        }
        manifest = {
            "schema_version": 2,
            "release": {
                "name": args.release_id.partition(":")[0],
                "version": args.release_id.partition(":")[2] or "unversioned",
                "release_id": args.release_id,
                "status": "rejected",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "files": [],
            "validation": {
                "status": "FAIL",
                "report_path": repository_path(report_path),
            },
        }
    save_json(manifest, manifest_path)
    save_json(report, report_path)
    if args.mlflow_run_metadata:
        try:
            log_validation_to_mlflow(
                report_path,
                report,
                resolve(args.mlflow_run_metadata),
            )
        except Exception as exc:
            report["status"] = "FAIL"
            report["passed"] = False
            report.setdefault("checks", []).append(
                {
                    "name": "mlflow_report_storage",
                    "status": "FAIL",
                    "passed": False,
                    "actual": f"{type(exc).__name__}: {exc}",
                    "expected": "validation report stored in MLflow",
                    "message": "Promotion is blocked because the validation report was not persisted.",
                }
            )
            report.setdefault("failed_checks", []).append("mlflow_report_storage")
            manifest["release"]["status"] = "rejected"
            manifest.setdefault("validation", {})["status"] = "FAIL"
            save_json(manifest, manifest_path)
            save_json(report, report_path)
    print(json.dumps(report, indent=2))
    if not report["passed"] and not args.no_fail_exit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
