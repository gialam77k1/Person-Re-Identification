from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_publication_report(
    manifest_path: Path,
    validation_path: Path,
    dvc_status_path: Path,
    checkpoint_pointers: list[Path],
    remote: str,
    stage: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    dvc_status = read_json(dvc_status_path)
    release_id = manifest.get("release", {}).get("release_id")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})

    check("release_validated", manifest.get("release", {}).get("status") == "validated", manifest.get("release", {}).get("status"), "validated")
    check("validation_pass", validation.get("status") == "PASS" and validation.get("passed") is True, validation.get("status"), "PASS")
    check("release_id_match", validation.get("release_id") == release_id and bool(release_id), validation.get("release_id"), release_id)
    check("dvc_remote_clean", not dvc_status, dvc_status, {})

    published_artifacts: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        path = Path(str(item.get("repository_path", "")))
        exists = path.is_file()
        actual_sha = sha256(path) if exists else None
        expected_sha = item.get("sha256")
        check(f"artifact_{item.get('role', 'unknown')}", exists and actual_sha == expected_sha, actual_sha, expected_sha)
        published_artifacts.append(
            {
                "role": item.get("role"),
                "repository_path": str(path),
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size if exists else None,
            }
        )

    pointer_entries = []
    for pointer in checkpoint_pointers:
        exists = pointer.is_file()
        check(f"dvc_pointer_{pointer.name}", exists, exists, True)
        pointer_entries.append({"path": str(pointer), "sha256": sha256(pointer) if exists else None})

    source = manifest.get("source", {})
    for key in ("git_commit", "training_git_commit", "mlflow_run_id", "config_sha256"):
        check(f"lineage_{key}", bool(source.get(key)), source.get(key), "non-empty")
    revisions = source.get("dataset_revisions", {})
    check("lineage_dataset_revisions", bool(revisions.get("duke")) and bool(revisions.get("market1501")), revisions, "duke and market1501")

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "release_id": release_id,
        "status": "PASS" if passed else "FAIL",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "dvc": {"remote": remote, "stage": stage, "status": "clean" if not dvc_status else "dirty"},
        "lineage": source,
        "model_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "validation_report": {"path": str(validation_path), "sha256": sha256(validation_path)},
        "checkpoint_pointers": pointer_entries,
        "artifacts": published_artifacts,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that the complete v2 release is present in DVC remote storage.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--dvc-status", required=True)
    parser.add_argument("--checkpoint-pointer", action="append", default=[])
    parser.add_argument("--remote", default="minio")
    parser.add_argument("--stage", default="v2_package_triton")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build_publication_report(
            Path(args.manifest),
            Path(args.validation_report),
            Path(args.dvc_status),
            [Path(path) for path in args.checkpoint_pointer],
            args.remote,
            args.stage,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
            "failed_checks": ["publication_exception"],
        }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
