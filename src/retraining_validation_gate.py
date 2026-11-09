from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_gate(
    *, evidence: dict[str, Any], decision: dict[str, Any], training: dict[str, Any],
    evaluation: dict[str, Any], baseline: dict[str, Any], mlflow_run: dict[str, Any],
    config: dict[str, Any], policy: dict[str, Any], base_checkpoint: Path,
    candidate_checkpoint: Path, git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "actual": actual, "expected": expected})

    release_id = str(config["release"]["id"])
    base_release = str(config["release"]["base_release_id"])
    dataset = str(config["data"]["dataset"]["name"]).lower()
    check("approval", decision.get("status") == policy["contract"]["approved_status"], decision.get("status"), policy["contract"]["approved_status"])
    check("git_commit", evidence.get("source_git_commit") == git_commit, git_commit, evidence.get("source_git_commit"))
    check("base_release", evidence.get("release_id") == base_release == decision.get("base_release_id"), {"evidence": evidence.get("release_id"), "config": base_release, "decision": decision.get("base_release_id")}, base_release)
    check("dataset", evidence.get("dataset") == dataset == str(decision.get("target_dataset", "")).lower(), {"evidence": evidence.get("dataset"), "config": dataset, "decision": decision.get("target_dataset")}, dataset)
    check("dvc_revision", bool(evidence.get("dataset_dvc_revision")) and baseline.get("dataset_dvc_revision") == evidence.get("dataset_dvc_revision"), baseline.get("dataset_dvc_revision"), evidence.get("dataset_dvc_revision"))
    expected_base_sha = evidence.get("base_checkpoint", {}).get("sha256")
    check("base_checkpoint_checksum", base_checkpoint.is_file() and sha256(base_checkpoint) == expected_base_sha, sha256(base_checkpoint) if base_checkpoint.is_file() else None, expected_base_sha)
    check("candidate_checkpoint", candidate_checkpoint.is_file() and candidate_checkpoint.stat().st_size > 0, candidate_checkpoint.stat().st_size if candidate_checkpoint.is_file() else 0, ">0 bytes")
    check("training_release", training.get("release_id") == release_id and training.get("base_release_id") == base_release, {"release": training.get("release_id"), "base": training.get("base_release_id")}, {"release": release_id, "base": base_release})
    check("mlflow_run", bool(mlflow_run.get("run_id")) and mlflow_run.get("status") == "FINISHED", {"run_id": mlflow_run.get("run_id"), "status": mlflow_run.get("status")}, {"run_id": "non-empty", "status": "FINISHED"})
    check("evaluation_release", evaluation.get("release_id") == release_id, evaluation.get("release_id"), release_id)
    check("valid_queries", int(evaluation.get("valid_queries", 0)) >= int(policy["quality"]["minimum_valid_queries"]), evaluation.get("valid_queries"), {"min": policy["quality"]["minimum_valid_queries"]})
    metric_policies = (("rank1", "rank1_max_drop_from_v1"), ("mAP", "map_max_drop_from_v1"), ("mINP", "minp_max_drop_from_v1"))
    for metric, policy_key in metric_policies:
        value = evaluation.get(metric)
        baseline_value = baseline.get(metric)
        finite = isinstance(value, (int, float)) and math.isfinite(float(value))
        threshold = float(baseline_value) - float(policy["quality"][policy_key]) if isinstance(baseline_value, (int, float)) else None
        check(f"quality_{metric}", finite and threshold is not None and float(value) >= threshold, value, {"min": threshold, "baseline": baseline_value})
    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": 1, "release_id": release_id, "status": "PASS" if passed else "FAIL",
        "passed": passed, "checked_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "mlflow_run_id": mlflow_run.get("run_id"), "dataset_dvc_revision": evidence.get("dataset_dvc_revision"),
    }
    manifest = {
        "schema_version": 1,
        "release": {"name": release_id.partition(":")[0], "version": release_id.partition(":")[2], "release_id": release_id, "status": "validated" if passed else "rejected"},
        "source": {"git_commit": git_commit, "dataset_dvc_revision": evidence.get("dataset_dvc_revision"), "base_release_id": base_release, "base_checkpoint_sha256": expected_base_sha, "mlflow_run_id": mlflow_run.get("run_id")},
        "model": {"task": "person-re-identification", "dataset": dataset, "architecture": config["model"]["backbone"], "embedding_dim": config["model"]["embedding_dim"]},
        "quality_metrics": {key: evaluation.get(key) for key in ("rank1", "rank5", "mAP", "mINP")},
        "files": [{"role": "pytorch_checkpoint", "repository_path": str(candidate_checkpoint), "sha256": sha256(candidate_checkpoint), "size_bytes": candidate_checkpoint.stat().st_size}] if candidate_checkpoint.is_file() else [],
        "validation": {"status": report["status"]},
    }
    return report, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate an approved DukeMTMC-reID v2 retraining candidate.")
    for name in ("evidence", "decision", "training-metrics", "evaluation-metrics", "baseline-evaluation", "mlflow-run", "config", "policy", "base-checkpoint", "candidate-checkpoint", "output", "manifest-output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    output, manifest_output = Path(args.output), Path(args.manifest_output)
    try:
        report, manifest = build_gate(
            evidence=read_json(Path(args.evidence)), decision=read_json(Path(args.decision)), training=read_json(Path(args.training_metrics)),
            evaluation=read_json(Path(args.evaluation_metrics)), baseline=read_json(Path(args.baseline_evaluation)), mlflow_run=read_json(Path(args.mlflow_run)),
            config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), policy=yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")),
            base_checkpoint=Path(args.base_checkpoint), candidate_checkpoint=Path(args.candidate_checkpoint), git_commit=args.git_commit or current_commit(),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        report = {"schema_version": 1, "release_id": "unknown", "status": "FAIL", "passed": False, "failed_checks": ["gate_execution"], "error": f"{type(exc).__name__}: {exc}"}
        manifest = {"schema_version": 1, "release": {"release_id": "unknown", "status": "rejected"}, "validation": {"status": "FAIL"}, "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
