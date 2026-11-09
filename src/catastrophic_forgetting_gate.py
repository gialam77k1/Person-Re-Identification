from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METRICS = ("rank1", "rank5", "mAP", "mINP")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _finite_metrics(report: dict[str, Any]) -> bool:
    return all(
        isinstance(report.get(metric), (int, float))
        and not isinstance(report.get(metric), bool)
        and math.isfinite(float(report[metric]))
        for metric in METRICS
    )


def build_forgetting_gate(
    *,
    duke_v1: dict[str, Any],
    duke_v2: dict[str, Any],
    market1501_v1_manifest: dict[str, Any],
    market1501_v2: dict[str, Any],
    policy: dict[str, Any],
    policy_sha256: str,
    mlflow_run_id: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    release_id = str(policy["release_id"])
    protocol = str(policy["market1501_retention"]["baseline_protocol"])
    market_v1 = market1501_v1_manifest.get("quality_metrics", {}).get(protocol, {})
    check("release_duke", duke_v2.get("release_id") == release_id, duke_v2.get("release_id"), release_id)
    check("release_market1501", market1501_v2.get("release_id") == release_id, market1501_v2.get("release_id"), release_id)
    check(
        "dataset_duke",
        str(duke_v2.get("dataset", "")).lower() == "dukemtmc-reid",
        duke_v2.get("dataset"),
        "dukemtmc-reid",
    )
    check("dataset_market1501", str(market1501_v2.get("dataset", "")).lower() == "market1501", market1501_v2.get("dataset"), "market1501")
    check("finite_duke_v1", _finite_metrics(duke_v1), {key: duke_v1.get(key) for key in METRICS}, "all finite")
    check("finite_duke_v2", _finite_metrics(duke_v2), {key: duke_v2.get(key) for key in METRICS}, "all finite")
    check("finite_market1501_v1", _finite_metrics(market_v1), {key: market_v1.get(key) for key in METRICS}, "all finite")
    check("finite_market1501_v2", _finite_metrics(market1501_v2), {key: market1501_v2.get(key) for key in METRICS}, "all finite")

    same_checkpoint = (
        bool(duke_v2.get("checkpoint_sha256"))
        and duke_v2.get("checkpoint_sha256") == market1501_v2.get("checkpoint_sha256")
    )
    check(
        "same_candidate_checkpoint",
        same_checkpoint,
        {
            "duke": duke_v2.get("checkpoint_sha256"),
            "market1501": market1501_v2.get("checkpoint_sha256"),
        },
        "identical non-empty SHA256",
    )
    check(
        "same_git_commit",
        bool(duke_v2.get("git_commit"))
        and duke_v2.get("git_commit") == market1501_v2.get("git_commit"),
        {"duke": duke_v2.get("git_commit"), "market1501": market1501_v2.get("git_commit")},
        "identical non-empty commit",
    )
    check(
        "dataset_dvc_revisions",
        bool(duke_v2.get("dataset_dvc_revision"))
        and bool(market1501_v2.get("dataset_dvc_revision")),
        {
            "duke": duke_v2.get("dataset_dvc_revision"),
            "market1501": market1501_v2.get("dataset_dvc_revision"),
        },
        "both non-empty",
    )

    minimum = policy["minimum_counts"]
    for dataset, report in (("duke", duke_v2), ("market1501", market1501_v2)):
        for key in ("query_count", "gallery_count", "valid_queries"):
            threshold_key = "valid_queries" if key == "valid_queries" else key.removesuffix("_count")
            threshold = int(minimum[threshold_key])
            actual = int(report.get(key, 0))
            check(f"{dataset}_{key}", actual >= threshold, actual, {"minimum": threshold})

    improvement = policy["duke_improvement"]
    for metric, threshold_key in (
        ("rank1", "rank1_min_absolute_gain"),
        ("mAP", "map_min_absolute_gain"),
    ):
        baseline = duke_v1.get(metric)
        candidate = duke_v2.get(metric)
        delta = (
            float(candidate) - float(baseline)
            if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float))
            else None
        )
        threshold = float(improvement[threshold_key])
        check(
            f"duke_{metric}_improvement",
            delta is not None and math.isfinite(delta) and delta >= threshold,
            {"baseline": baseline, "candidate": candidate, "absolute_gain": delta},
            {"minimum_absolute_gain": threshold},
        )

    retention = policy["market1501_retention"]
    for metric, threshold_key in (
        ("rank1", "rank1_max_absolute_drop"),
        ("rank5", "rank5_max_absolute_drop"),
        ("mAP", "map_max_absolute_drop"),
        ("mINP", "minp_max_absolute_drop"),
    ):
        baseline = market_v1.get(metric)
        candidate = market1501_v2.get(metric)
        drop = (
            float(baseline) - float(candidate)
            if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float))
            else None
        )
        maximum = float(retention[threshold_key])
        check(
            f"market1501_{metric}_retention",
            drop is not None and math.isfinite(drop) and drop <= maximum,
            {"baseline": baseline, "candidate": candidate, "absolute_drop": drop},
            {"maximum_absolute_drop": maximum},
        )

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "release_id": release_id,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha256,
        "mlflow_run_id": mlflow_run_id,
        "candidate_checkpoint_sha256": duke_v2.get("checkpoint_sha256"),
        "reports": {
            "duke": {key: duke_v2.get(key) for key in METRICS},
            "market1501": {key: market1501_v2.get(key) for key in METRICS},
        },
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duke-v1", required=True)
    parser.add_argument("--duke-v2", required=True)
    parser.add_argument("--market1501-v1-manifest", required=True)
    parser.add_argument("--market1501-v2", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--mlflow-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    policy_path = Path(args.policy)
    mlflow_run = read_json(Path(args.mlflow_run))
    report = build_forgetting_gate(
        duke_v1=read_json(Path(args.duke_v1)),
        duke_v2=read_json(Path(args.duke_v2)),
        market1501_v1_manifest=read_json(Path(args.market1501_v1_manifest)),
        market1501_v2=read_json(Path(args.market1501_v2)),
        policy=yaml.safe_load(policy_path.read_text(encoding="utf-8")),
        policy_sha256=file_sha256(policy_path),
        mlflow_run_id=str(mlflow_run.get("run_id", "")),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
