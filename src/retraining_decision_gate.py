from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _check(actual: Any, expected: Any, passed: bool) -> dict[str, Any]:
    return {"actual": actual, "expected": expected, "passed": bool(passed)}


def build_retraining_decision(
    drift_report: dict[str, Any],
    evaluation_report: dict[str, Any],
    policy: dict[str, Any],
    approved_by: str = "",
) -> dict[str, Any]:
    candidate = policy["candidate"]
    evidence_policy = policy["evidence"]
    approval_policy = policy["approval"]
    expected_release = str(candidate["base_release_id"])
    expected_dataset = str(candidate["target_dataset"])

    evidence_checks = {
        "drift_release": _check(
            drift_report.get("release_id"),
            expected_release,
            drift_report.get("release_id") == expected_release,
        ),
        "evaluation_release": _check(
            evaluation_report.get("release_id"),
            expected_release,
            evaluation_report.get("release_id") == expected_release,
        ),
        "dataset": _check(
            evaluation_report.get("dataset"),
            expected_dataset,
            evaluation_report.get("dataset") == expected_dataset,
        ),
        "query_samples": _check(
            evaluation_report.get("num_query_samples"),
            f">={evidence_policy['minimum_query_samples']}",
            int(evaluation_report.get("num_query_samples", 0)) >= int(evidence_policy["minimum_query_samples"]),
        ),
        "gallery_samples": _check(
            evaluation_report.get("num_gallery_samples"),
            f">={evidence_policy['minimum_gallery_samples']}",
            int(evaluation_report.get("num_gallery_samples", 0))
            >= int(evidence_policy["minimum_gallery_samples"]),
        ),
        "query_coverage": _check(
            evaluation_report.get("query_coverage"),
            f">={evidence_policy['minimum_query_coverage']}",
            float(evaluation_report.get("query_coverage", 0.0))
            >= float(evidence_policy["minimum_query_coverage"]),
        ),
        "embedding_dim": _check(
            evaluation_report.get("embedding_dim"),
            evidence_policy["required_embedding_dim"],
            int(evaluation_report.get("embedding_dim", 0)) == int(evidence_policy["required_embedding_dim"]),
        ),
        "embeddings_finite": _check(
            evaluation_report.get("embeddings_finite"),
            True,
            evaluation_report.get("embeddings_finite") is True,
        ),
        "dataset_dvc_revision": _check(
            evaluation_report.get("dataset_dvc_revision"),
            "non-empty DVC revision",
            bool(evaluation_report.get("dataset_dvc_revision")),
        ),
    }
    evidence_valid = all(item["passed"] for item in evidence_checks.values())

    drift_required = bool(evidence_policy.get("require_drift_detected", True))
    drift_detected = drift_report.get("drift_detected") is True
    drift_triggered = drift_detected if drift_required else True

    quality_checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in policy["quality"]["retrain_when_any_below"].items():
        actual = float(evaluation_report.get(metric, 1.0))
        maximum = float(threshold)
        quality_checks[metric] = {
            "actual": actual,
            "retrain_below": maximum,
            "triggered": actual < maximum,
        }
    quality_triggered = any(item["triggered"] for item in quality_checks.values())
    retraining_signal = evidence_valid and drift_triggered and quality_triggered

    minimum_name_length = int(approval_policy.get("minimum_approver_name_length", 1))
    approval_required = bool(approval_policy.get("manual_approval_required", True))
    approval_valid = (not approval_required) or len(approved_by.strip()) >= minimum_name_length

    if not evidence_valid:
        status = "INSUFFICIENT_EVIDENCE"
    elif not retraining_signal:
        status = "NO_RETRAIN"
    elif not approval_valid:
        status = "REVIEW_REQUIRED"
    else:
        status = "RETRAIN_APPROVED"

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_name": policy["policy_name"],
        "status": status,
        "base_release_id": expected_release,
        "target_dataset": expected_dataset,
        "target_dataset_version": candidate["target_dataset_version"],
        "target_epochs": int(candidate["target_epochs"]),
        "evidence_valid": evidence_valid,
        "drift_triggered": drift_triggered,
        "quality_triggered": quality_triggered,
        "retraining_signal": retraining_signal,
        "manual_approval_required": approval_required,
        "approved_by": approved_by.strip() or None,
        "approval_valid": approval_valid,
        "evidence_checks": evidence_checks,
        "quality_checks": quality_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether cross-domain evidence permits retraining.")
    parser.add_argument("--drift-report", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--config", default="configs/retraining_decision.yaml")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-approved", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        drift_report = json.loads(Path(args.drift_report).read_text(encoding="utf-8"))
        evaluation_report = json.loads(Path(args.evaluation_report).read_text(encoding="utf-8"))
        policy = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        result = build_retraining_decision(drift_report, evaluation_report, policy, args.approved_by)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Retraining decision: {result['status']}")
        if args.require_approved and result["status"] != "RETRAIN_APPROVED":
            return 2
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
