from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def artifact(path: Path, bundle_dir: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Evidence file is missing or empty: {path}")
    destination = bundle_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return {"path": name, "size_bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def build_bundle(
    *,
    cycle_id: str,
    release_id: str,
    dataset: str,
    drift_path: Path,
    evaluation_path: Path,
    decision_path: Path,
    base_manifest_path: Path,
    output_path: Path,
    source_commit: str | None = None,
    workflow_run_id: str = "",
) -> dict[str, Any]:
    drift = read_json(drift_path)
    evaluation = read_json(evaluation_path)
    decision = read_json(decision_path)
    base_manifest = read_json(base_manifest_path)
    actual_release = base_manifest.get("release", {}).get("release_id")
    if actual_release != release_id:
        raise ValueError(f"Base manifest release mismatch: {actual_release!r} != {release_id!r}")
    if drift.get("release_id") != release_id or evaluation.get("release_id") != release_id:
        raise ValueError("Drift/evaluation reports do not belong to the requested base release.")
    if str(evaluation.get("dataset", "")).lower() != dataset.lower():
        raise ValueError("Evaluation report does not belong to the requested dataset.")
    dvc_revision = evaluation.get("dataset_dvc_revision")
    if not dvc_revision:
        raise ValueError("Evaluation report is missing dataset_dvc_revision.")
    if decision.get("status") not in {"REVIEW_REQUIRED", "RETRAIN_APPROVED"}:
        raise ValueError(f"Evidence is not eligible for approval: {decision.get('status')}")
    checkpoint = next(
        (item for item in base_manifest.get("files", []) if item.get("role") == "pytorch_checkpoint"),
        None,
    )
    if not checkpoint:
        raise ValueError("Base manifest is missing the PyTorch checkpoint lineage.")
    bundle_dir = output_path.parent
    payload = {
        "schema_version": 1,
        "cycle_id": cycle_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow_run_id": str(workflow_run_id),
        "source_git_commit": source_commit or git_commit(),
        "release_id": release_id,
        "dataset": dataset.lower(),
        "dataset_dvc_revision": dvc_revision,
        "base_checkpoint": checkpoint,
        "base_manifest": artifact(base_manifest_path, bundle_dir, "base_model_manifest.json"),
        "evidence": {
            "drift_report": artifact(drift_path, bundle_dir, "drift_report.json"),
            "evaluation_report": artifact(evaluation_path, bundle_dir, "evaluation_report.json"),
            "retraining_decision": artifact(decision_path, bundle_dir, "decision.json"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_bundle(
    manifest_path: Path,
    *,
    cycle_id: str,
    release_id: str,
    dataset: str,
    current_commit: str | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    expected = {
        "cycle_id": cycle_id,
        "release_id": release_id,
        "dataset": dataset.lower(),
        "source_git_commit": current_commit or git_commit(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Evidence lineage mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    for spec in [manifest["base_manifest"], *manifest["evidence"].values()]:
        path = manifest_path.parent / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {path}")
    decision_path = manifest_path.parent / manifest["evidence"]["retraining_decision"]["path"]
    decision = read_json(decision_path)
    if decision.get("status") not in {"REVIEW_REQUIRED", "RETRAIN_APPROVED"}:
        raise ValueError(f"Invalid retraining decision: {decision.get('status')}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify an immutable retraining evidence bundle.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cycle-id", required=True)
    build.add_argument("--release-id", required=True)
    build.add_argument("--dataset", required=True)
    build.add_argument("--drift-report", required=True)
    build.add_argument("--evaluation-report", required=True)
    build.add_argument("--decision", required=True)
    build.add_argument("--base-manifest", required=True)
    build.add_argument("--source-commit")
    build.add_argument("--workflow-run-id", default="")
    build.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--cycle-id", required=True)
    verify.add_argument("--release-id", required=True)
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--current-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build":
            payload = build_bundle(
                cycle_id=args.cycle_id,
                release_id=args.release_id,
                dataset=args.dataset,
                drift_path=Path(args.drift_report),
                evaluation_path=Path(args.evaluation_report),
                decision_path=Path(args.decision),
                base_manifest_path=Path(args.base_manifest),
                output_path=Path(args.output),
                source_commit=args.source_commit,
                workflow_run_id=args.workflow_run_id,
            )
        else:
            payload = verify_bundle(
                Path(args.manifest),
                cycle_id=args.cycle_id,
                release_id=args.release_id,
                dataset=args.dataset,
                current_commit=args.current_commit,
            )
        print(json.dumps(payload, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
