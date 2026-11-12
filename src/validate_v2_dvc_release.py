from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "market1501-duke-vit-bnneck:v2"
METRICS = ("rank1", "rank5", "mAP", "mINP")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def artifact(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing release artifact {role}: {path}")
    return {"role": role, "repository_path": repository_path(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_metrics(report: dict[str, Any]) -> bool:
    return all(finite_number(report.get(key)) for key in METRICS)


def build_release(
    *,
    checkpoint_manifest_path: Path,
    data_manifest_path: Path,
    duke_evaluation_path: Path,
    market1501_evaluation_path: Path,
    forgetting_report_path: Path,
    onnx_manifest_path: Path,
    parity_report_path: Path,
    reference_manifest_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    reference_files: list[tuple[str, Path]],
    preprocessing_source_path: Path,
    preprocessing_output_path: Path,
    policy_path: Path,
    report_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = read_json(checkpoint_manifest_path)
    data = read_json(data_manifest_path)
    duke = read_json(duke_evaluation_path)
    market1501 = read_json(market1501_evaluation_path)
    forgetting = read_json(forgetting_report_path)
    onnx = read_json(onnx_manifest_path)
    parity = read_json(parity_report_path)
    reference = read_json(reference_manifest_path)
    preprocessing = read_json(preprocessing_source_path)
    policy = read_yaml(policy_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected})

    release_id = str(checkpoint.get("release_id") or policy.get("release_id") or "")
    check("validation_policy_release", bool(release_id) and policy.get("release_id") == release_id, policy.get("release_id"), release_id)
    check("data_validation", data.get("passed") is True, data.get("passed"), True)
    check("checkpoint_release", checkpoint.get("release_id") == release_id, checkpoint.get("release_id"), release_id)
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    check("checkpoint_checksum", checkpoint_sha == checkpoint.get("checkpoint_sha256"), checkpoint_sha, checkpoint.get("checkpoint_sha256"))
    check("mlflow_run_id", bool(checkpoint.get("mlflow_run_id")), checkpoint.get("mlflow_run_id"), "non-empty")
    check("git_commit", bool(checkpoint.get("git_commit")) and bool(checkpoint.get("training_git_commit")), {"release": checkpoint.get("git_commit"), "training": checkpoint.get("training_git_commit")}, "both non-empty")
    check("training_config_hash", len(str(checkpoint.get("training_config_sha256", ""))) == 64, checkpoint.get("training_config_sha256"), "SHA256")
    revisions = checkpoint.get("dataset_revisions", {})
    check("dataset_revisions", bool(revisions.get("duke")) and bool(revisions.get("market1501")), revisions, "both non-empty")
    check("evaluation_release", duke.get("release_id") == market1501.get("release_id") == release_id, {"duke": duke.get("release_id"), "market1501": market1501.get("release_id")}, release_id)
    check("evaluation_metrics_finite", finite_metrics(duke) and finite_metrics(market1501), {"duke": {key: duke.get(key) for key in METRICS}, "market1501": {key: market1501.get(key) for key in METRICS}}, "all finite")
    threshold_keys = {"rank1": "rank1_min", "rank5": "rank5_min", "mAP": "map_min", "mINP": "minp_min"}
    for dataset_name, evaluation in (("duke", duke), ("market1501", market1501)):
        thresholds = policy.get("quality", {}).get(dataset_name, {})
        for metric, threshold_key in threshold_keys.items():
            value, minimum = evaluation.get(metric), thresholds.get(threshold_key)
            check(f"quality_{dataset_name}_{metric.lower()}", finite_number(value) and finite_number(minimum) and float(value) >= float(minimum), value, {"min": minimum})
    check("same_evaluated_checkpoint", duke.get("checkpoint_sha256") == market1501.get("checkpoint_sha256") == checkpoint_sha, {"duke": duke.get("checkpoint_sha256"), "market1501": market1501.get("checkpoint_sha256"), "release": checkpoint_sha}, "identical")
    check("catastrophic_forgetting", forgetting.get("status") == "PASS" and forgetting.get("passed") is True, forgetting.get("status"), "PASS")

    contract = policy.get("contract", {})
    expected_input_shape = contract.get("input_shape")
    onnx_contract = (
        onnx.get("release_id") == release_id
        and onnx.get("input_name") == contract.get("input_name")
        and onnx.get("input_dtype") == contract.get("input_dtype")
        and onnx.get("input_shape") == expected_input_shape
        and onnx.get("output_name") == contract.get("output_name")
        and onnx.get("output_dtype") == contract.get("output_dtype")
        and int(onnx.get("embedding_dim", 0)) == int(contract.get("embedding_dim", 0))
        and onnx.get("preserve_aspect_ratio") is contract.get("preserve_aspect_ratio")
    )
    check("onnx_export_contract", onnx_contract, {key: onnx.get(key) for key in ("release_id", "input_name", "input_dtype", "input_shape", "output_name", "output_dtype", "embedding_dim", "preserve_aspect_ratio")}, contract)
    parity_input_shape = parity.get("input_shape", [])
    parity_shape = parity.get("output_shape", [])
    parity_contract = (
        parity.get("input_name") == contract.get("input_name")
        and isinstance(parity_input_shape, list)
        and len(parity_input_shape) == 4
        and parity_input_shape[0] in (1, "batch_size")
        and parity_input_shape[1:] == expected_input_shape[1:]
        and parity.get("output_name") == contract.get("output_name")
        and isinstance(parity_shape, list)
        and len(parity_shape) == 2
        and parity_shape[0] in (1, "batch_size")
        and int(parity_shape[1]) == int(contract.get("embedding_dim", 0))
    )
    check("onnx_runtime_contract", parity_contract, {"input_name": parity.get("input_name"), "input_shape": parity.get("input_shape"), "output_name": parity.get("output_name"), "output_shape": parity_shape}, {"input_name": contract.get("input_name"), "input_shape": expected_input_shape, "output_name": contract.get("output_name"), "embedding_dim": contract.get("embedding_dim")})
    parity_policy = policy.get("onnx_parity", {})
    cosine, max_error = parity.get("min_cosine_similarity"), parity.get("max_abs_error")
    check("onnx_parity_cosine", parity.get("passed") is True and finite_number(cosine) and float(cosine) >= float(parity_policy.get("cosine_min", float("inf"))), cosine, {"min": parity_policy.get("cosine_min")})
    check("onnx_parity_max_abs_error", parity.get("passed") is True and finite_number(max_error) and float(max_error) <= float(parity_policy.get("max_abs_error", -1.0)), max_error, {"max": parity_policy.get("max_abs_error")})

    preprocessing_passed = (
        preprocessing.get("input", {}).get("name") == contract.get("input_name")
        and preprocessing.get("input", {}).get("dtype") == contract.get("input_dtype")
        and preprocessing.get("input", {}).get("layout") == contract.get("input_layout")
        and preprocessing.get("input", {}).get("shape") == expected_input_shape
        and preprocessing.get("resize", {}).get("preserve_aspect_ratio") is contract.get("preserve_aspect_ratio")
        and preprocessing.get("resize", {}).get("target_height") == expected_input_shape[2]
        and preprocessing.get("resize", {}).get("target_width") == expected_input_shape[3]
        and len(preprocessing.get("normalization", {}).get("mean", [])) == 3
        and len(preprocessing.get("normalization", {}).get("std", [])) == 3
        and preprocessing.get("inference", {}).get("reference_embeddings_l2_normalized") is True
    )
    check("preprocessing_contract", preprocessing_passed, preprocessing, contract)

    reference_by_role = dict(reference_files)
    embeddings = np.load(reference_by_role["reference_embeddings"], allow_pickle=False)
    pids = np.load(reference_by_role["reference_pids"], allow_pickle=False)
    camids = np.load(reference_by_role["reference_camids"], allow_pickle=False)
    paths = np.load(reference_by_role["reference_paths"], allow_pickle=False)
    expected_dim = int(contract.get("embedding_dim", 0))
    check("reference_shape", embeddings.ndim == 2 and embeddings.shape[1] == expected_dim and embeddings.shape[0] > 0, list(embeddings.shape), [">0", expected_dim])
    check("reference_metadata_alignment", len(pids) == len(camids) == len(paths) == len(embeddings), {"embeddings": len(embeddings), "pids": len(pids), "camids": len(camids), "paths": len(paths)}, "all identical")
    finite_embeddings = bool(np.isfinite(embeddings).all())
    check("reference_finite", finite_embeddings, finite_embeddings, True)
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.ndim == 2 else np.asarray([])
    norm_error = float(np.max(np.abs(norms - 1.0))) if norms.size and np.isfinite(norms).all() else float("inf")
    norm_atol = float(contract.get("reference_l2_norm_atol", 0.0))
    check("reference_l2_normalized", finite_embeddings and bool(np.allclose(norms, 1.0, atol=norm_atol, rtol=0.0)), norm_error, {"max_abs_error": norm_atol})
    check("reference_manifest_contract", reference.get("release_id") == release_id and int(reference.get("feature_dim", 0)) == expected_dim and int(reference.get("num_samples", 0)) == len(embeddings) and reference.get("finite") is True and reference.get("l2_normalized") is True, {key: reference.get(key) for key in ("release_id", "feature_dim", "num_samples", "finite", "l2_normalized")}, {"release_id": release_id, "feature_dim": expected_dim, "num_samples": len(embeddings), "finite": True, "l2_normalized": True})

    preprocessing["release_id"] = release_id
    preprocessing_output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_output_path.write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    files = [artifact("pytorch_checkpoint", checkpoint_path), artifact("onnx_graph", onnx_path), artifact("onnx_external_weights", onnx_path.with_name(f"{onnx_path.name}.data")), artifact("preprocessing_contract", preprocessing_output_path), artifact("validation_policy", policy_path), *[artifact(role, path) for role, path in reference_files]]
    export_files = onnx.get("files", {})
    check("onnx_artifact_checksums", export_files.get("graph", {}).get("sha256") == files[1]["sha256"] and export_files.get("external_weights", {}).get("sha256") == files[2]["sha256"], export_files, {"graph": files[1]["sha256"], "external_weights": files[2]["sha256"]})
    reference_checksums = reference.get("files", {})
    generated_reference_checksums = {item["role"].removeprefix("reference_"): item["sha256"] for item in files if item["role"].startswith("reference_")}
    check("reference_artifact_checksums", all(reference_checksums.get(role, {}).get("sha256") == digest for role, digest in generated_reference_checksums.items()), {role: value.get("sha256") for role, value in reference_checksums.items()}, generated_reference_checksums)
    check("artifact_checksums", all(len(item["sha256"]) == 64 for item in files), len(files), len(files))

    passed = all(item["passed"] for item in checks)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 2,
        "release": {"name": release_id.partition(":")[0], "version": release_id.partition(":")[2], "release_id": release_id, "status": "validated" if passed else "rejected", "created_at": now},
        "source": {"git_commit": checkpoint.get("git_commit"), "training_git_commit": checkpoint.get("training_git_commit"), "dataset_revisions": revisions, "mlflow_run_id": checkpoint.get("mlflow_run_id"), "config_sha256": checkpoint.get("training_config_sha256"), "checkpoint_epoch": checkpoint.get("best_epoch")},
        "model": {"task": "person-re-identification", "architecture": "vit_b_16", "embedding_dim": expected_dim, "training_dataset": "duke"},
        "quality_metrics": {
            "duke": {key: duke.get(key) for key in ("rank1", "rank5", "rank10", "rank20", "mAP", "mINP")},
            "market1501": {key: market1501.get(key) for key in ("rank1", "rank5", "rank10", "rank20", "mAP", "mINP")},
        },
        "contract": {"input_name": contract.get("input_name"), "input_shape": expected_input_shape, "output_name": contract.get("output_name"), "output_shape": ["batch_size", expected_dim], "embedding_dim": expected_dim, "preprocessing_path": repository_path(preprocessing_output_path), "reference_embeddings_l2_normalized": True},
        "reference": {"num_samples": len(embeddings), "embedding_dim": expected_dim, "norm_max_abs_error": norm_error},
        "policy": {"path": repository_path(policy_path), "sha256": sha256_file(policy_path)},
        "files": files,
        "validation": {"status": "PASS" if passed else "FAIL", "report_path": repository_path(report_path), "catastrophic_forgetting_report": repository_path(forgetting_report_path), "parity_report": repository_path(parity_report_path)},
    }
    report = {"schema_version": 2, "release_id": release_id, "status": "PASS" if passed else "FAIL", "passed": passed, "checked_at": now, "policy": manifest["policy"], "checks": checks, "failed_checks": [item["name"] for item in checks if not item["passed"]], "validated_artifacts": files, "manifest_path": repository_path(manifest_path)}
    return report, manifest


def log_validation_to_mlflow(report_path: Path, manifest_path: Path, policy_path: Path, report: dict[str, Any], run_metadata_path: Path) -> None:
    import mlflow
    from mlflow import MlflowClient

    metadata = read_json(run_metadata_path)
    tracking_uri, run_id = metadata.get("tracking_uri"), metadata.get("run_id")
    if not tracking_uri or not run_id:
        raise ValueError("MLflow run metadata must contain tracking_uri and run_id")
    mlflow.set_tracking_uri(str(tracking_uri))
    client = MlflowClient()
    client.get_run(str(run_id))
    client.set_tag(str(run_id), "validation.v2.status", report["status"])
    client.set_tag(str(run_id), "validation.v2.release_id", str(report.get("release_id")))
    client.log_metric(str(run_id), "validation.v2.passed", 1.0 if report["passed"] else 0.0)
    client.log_metric(str(run_id), "validation.v2.failed_checks", float(len(report["failed_checks"])))
    for path in (report_path, manifest_path, policy_path):
        client.log_artifact(str(run_id), str(path), artifact_path="validation/v2")


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint-manifest", "data-manifest", "duke-evaluation", "market1501-evaluation", "forgetting-report", "onnx-manifest", "parity-report", "reference-manifest", "checkpoint", "onnx", "preprocessing-source", "preprocessing-output", "policy", "mlflow-run", "output", "manifest-output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--reference-root", required=True)
    args = parser.parse_args()
    reference_root = Path(args.reference_root)
    report_path, manifest_path, policy_path = Path(args.output), Path(args.manifest_output), Path(args.policy)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, manifest = build_release(
            checkpoint_manifest_path=Path(args.checkpoint_manifest), data_manifest_path=Path(args.data_manifest), duke_evaluation_path=Path(args.duke_evaluation), market1501_evaluation_path=Path(args.market1501_evaluation), forgetting_report_path=Path(args.forgetting_report), onnx_manifest_path=Path(args.onnx_manifest), parity_report_path=Path(args.parity_report), reference_manifest_path=Path(args.reference_manifest), checkpoint_path=Path(args.checkpoint), onnx_path=Path(args.onnx),
            reference_files=[("reference_embeddings", reference_root / "reference_embeddings.npy"), ("reference_pids", reference_root / "reference_pids.npy"), ("reference_camids", reference_root / "reference_camids.npy"), ("reference_paths", reference_root / "reference_paths.npy")],
            preprocessing_source_path=Path(args.preprocessing_source), preprocessing_output_path=Path(args.preprocessing_output), policy_path=policy_path, report_path=report_path, manifest_path=manifest_path,
        )
    except Exception as error:
        now = datetime.now(timezone.utc).isoformat()
        failed_release = str(read_yaml(policy_path).get("release_id") or RELEASE_ID)
        report = {"schema_version": 2, "release_id": failed_release, "status": "FAIL", "passed": False, "checked_at": now, "checks": [], "failed_checks": ["validation_exception"], "error": f"{type(error).__name__}: {error}", "manifest_path": repository_path(manifest_path)}
        manifest = {"schema_version": 2, "release": {"release_id": failed_release, "status": "rejected", "created_at": now}, "validation": {"status": "FAIL", "report_path": repository_path(report_path)}, "files": []}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        log_validation_to_mlflow(report_path, manifest_path, policy_path, report, Path(args.mlflow_run))
    except Exception as error:
        report["status"], report["passed"] = "FAIL", False
        report.setdefault("failed_checks", []).append("mlflow_validation_logging")
        report["mlflow_error"] = f"{type(error).__name__}: {error}"
        manifest.setdefault("validation", {})["status"] = "FAIL"
        manifest.setdefault("release", {})["status"] = "rejected"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
