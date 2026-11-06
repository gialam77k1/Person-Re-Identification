from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "market1501-msmt17-vit-bnneck:v2"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
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


def artifact(role: str, path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing release artifact {role}: {path}")
    return {
        "role": role,
        "repository_path": repository_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def finite_metrics(report: dict) -> bool:
    return all(
        isinstance(report.get(key), (int, float))
        and not isinstance(report.get(key), bool)
        and math.isfinite(float(report[key]))
        for key in ("rank1", "rank5", "mAP", "mINP")
    )


def build_release(
    *,
    checkpoint_manifest_path: Path,
    data_manifest_path: Path,
    msmt17_evaluation_path: Path,
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
    report_path: Path,
    manifest_path: Path,
) -> tuple[dict, dict]:
    checkpoint = read_json(checkpoint_manifest_path)
    data = read_json(data_manifest_path)
    msmt17 = read_json(msmt17_evaluation_path)
    market1501 = read_json(market1501_evaluation_path)
    forgetting = read_json(forgetting_report_path)
    onnx = read_json(onnx_manifest_path)
    parity = read_json(parity_report_path)
    reference = read_json(reference_manifest_path)
    preprocessing = read_json(preprocessing_source_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected})

    check("data_validation", data.get("passed") is True, data.get("passed"), True)
    check("checkpoint_release", checkpoint.get("release_id") == RELEASE_ID, checkpoint.get("release_id"), RELEASE_ID)
    actual_checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    check("checkpoint_checksum", actual_checkpoint_sha == checkpoint.get("checkpoint_sha256"), actual_checkpoint_sha, checkpoint.get("checkpoint_sha256"))
    check("mlflow_run_id", bool(checkpoint.get("mlflow_run_id")), checkpoint.get("mlflow_run_id"), "non-empty")
    check("training_config_hash", len(str(checkpoint.get("training_config_sha256", ""))) == 64, checkpoint.get("training_config_sha256"), "SHA256")
    revisions = checkpoint.get("dataset_revisions", {})
    check("dataset_revisions", bool(revisions.get("msmt17")) and bool(revisions.get("market1501")), revisions, "both non-empty")
    check("evaluation_release", msmt17.get("release_id") == market1501.get("release_id") == RELEASE_ID, {"msmt17": msmt17.get("release_id"), "market1501": market1501.get("release_id")}, RELEASE_ID)
    check("evaluation_metrics_finite", finite_metrics(msmt17) and finite_metrics(market1501), {"msmt17": {key: msmt17.get(key) for key in ("rank1", "rank5", "mAP", "mINP")}, "market1501": {key: market1501.get(key) for key in ("rank1", "rank5", "mAP", "mINP")}}, "all finite")
    check("same_evaluated_checkpoint", msmt17.get("checkpoint_sha256") == market1501.get("checkpoint_sha256") == actual_checkpoint_sha, {"msmt17": msmt17.get("checkpoint_sha256"), "market1501": market1501.get("checkpoint_sha256"), "release": actual_checkpoint_sha}, "identical")
    check("catastrophic_forgetting", forgetting.get("status") == "PASS" and forgetting.get("passed") is True, forgetting.get("status"), "PASS")
    check("onnx_contract", onnx.get("input_shape") == [1, 3, 224, 224] and int(onnx.get("embedding_dim", 0)) == 512, {"input": onnx.get("input_shape"), "dim": onnx.get("embedding_dim")}, {"input": [1, 3, 224, 224], "dim": 512})
    check("onnx_parity", parity.get("passed") is True, parity.get("passed"), True)
    check("reference_contract", int(reference.get("feature_dim", 0)) == 512 and int(reference.get("num_samples", 0)) > 0, {"dim": reference.get("feature_dim"), "samples": reference.get("num_samples")}, {"dim": 512, "samples": ">0"})

    preprocessing["release_id"] = RELEASE_ID
    preprocessing_output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_output_path.write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    files = [
        artifact("pytorch_checkpoint", checkpoint_path),
        artifact("onnx_graph", onnx_path),
        artifact("onnx_external_weights", onnx_path.with_name(f"{onnx_path.name}.data")),
        artifact("preprocessing_contract", preprocessing_output_path),
        *[artifact(role, path) for role, path in reference_files],
    ]
    check("artifact_checksums", all(len(item["sha256"]) == 64 for item in files), len(files), len(files))
    passed = all(item["passed"] for item in checks)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "release": {"name": "market1501-msmt17-vit-bnneck", "version": "v2", "release_id": RELEASE_ID, "status": "validated" if passed else "rejected", "created_at": now},
        "source": {
            "git_commit": checkpoint.get("git_commit"),
            "training_git_commit": checkpoint.get("training_git_commit"),
            "dataset_revisions": revisions,
            "mlflow_run_id": checkpoint.get("mlflow_run_id"),
            "config_sha256": checkpoint.get("training_config_sha256"),
            "checkpoint_epoch": checkpoint.get("best_epoch"),
        },
        "model": {"task": "person-re-identification", "architecture": "vit_b_16", "embedding_dim": 512, "training_dataset": "msmt17"},
        "quality_metrics": {"msmt17": {key: msmt17.get(key) for key in ("rank1", "rank5", "mAP", "mINP")}, "market1501": {key: market1501.get(key) for key in ("rank1", "rank5", "mAP", "mINP")}},
        "contract": {"input_shape": [1, 3, 224, 224], "output_shape": [1, 512], "preprocessing_path": repository_path(preprocessing_output_path)},
        "files": files,
        "validation": {"status": "PASS" if passed else "FAIL", "report_path": repository_path(report_path), "catastrophic_forgetting_report": repository_path(forgetting_report_path), "parity_report": repository_path(parity_report_path)},
    }
    report = {"schema_version": 1, "release_id": RELEASE_ID, "status": "PASS" if passed else "FAIL", "passed": passed, "checked_at": now, "checks": checks, "failed_checks": [item["name"] for item in checks if not item["passed"]], "validated_artifacts": files, "manifest_path": repository_path(manifest_path)}
    return report, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint-manifest", "data-manifest", "msmt17-evaluation", "market1501-evaluation", "forgetting-report", "onnx-manifest", "parity-report", "reference-manifest", "checkpoint", "onnx", "preprocessing-source", "preprocessing-output", "output", "manifest-output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--reference-root", required=True)
    args = parser.parse_args()
    reference_root = Path(args.reference_root)
    report_path, manifest_path = Path(args.output), Path(args.manifest_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report, manifest = build_release(
        checkpoint_manifest_path=Path(args.checkpoint_manifest), data_manifest_path=Path(args.data_manifest), msmt17_evaluation_path=Path(args.msmt17_evaluation), market1501_evaluation_path=Path(args.market1501_evaluation), forgetting_report_path=Path(args.forgetting_report), onnx_manifest_path=Path(args.onnx_manifest), parity_report_path=Path(args.parity_report), reference_manifest_path=Path(args.reference_manifest), checkpoint_path=Path(args.checkpoint), onnx_path=Path(args.onnx),
        reference_files=[("reference_embeddings", reference_root / "reference_embeddings.npy"), ("reference_pids", reference_root / "reference_pids.npy"), ("reference_camids", reference_root / "reference_camids.npy"), ("reference_paths", reference_root / "reference_paths.npy")],
        preprocessing_source_path=Path(args.preprocessing_source), preprocessing_output_path=Path(args.preprocessing_output), report_path=report_path, manifest_path=manifest_path,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
