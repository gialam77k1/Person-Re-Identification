from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.validate_v2_evaluation import canonical_dataset_name


def validate_config(
    path: Path,
    *,
    release_id: str,
    dataset: str,
    report: Path,
) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config is not a mapping: {path}")
    actual_release = str(payload.get("release", {}).get("id", ""))
    if actual_release != release_id:
        raise ValueError(
            f"Evaluation config release mismatch in {path}: "
            f"{actual_release!r} != {release_id!r}"
        )
    actual_dataset = canonical_dataset_name(
        str(payload.get("data", {}).get("dataset", {}).get("name", ""))
    )
    expected_dataset = canonical_dataset_name(dataset)
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"Evaluation config dataset mismatch in {path}: "
            f"{actual_dataset!r} != {expected_dataset!r}"
        )
    run_root = Path(str(payload.get("artifacts", {}).get("run_root", "")))
    filename = str(payload.get("evaluation", {}).get("report_filename", ""))
    configured_report = run_root / "metrics" / filename
    if configured_report != report:
        raise ValueError(
            f"Evaluation report path mismatch in {path}: "
            f"{configured_report} != {report}"
        )
    return {
        "config": str(path),
        "dataset": actual_dataset,
        "report": str(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail before GPU training if post-training evaluation wiring is inconsistent."
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--duke-config", required=True)
    parser.add_argument("--duke-report", required=True)
    parser.add_argument("--market-config", required=True)
    parser.add_argument("--market-report", required=True)
    parser.add_argument("--forgetting-policy", required=True)
    parser.add_argument("--validation-policy", required=True)
    parser.add_argument("--target-dataset", default="dukemtmc-reid")
    args = parser.parse_args()
    try:
        checks = [
            validate_config(
                Path(args.duke_config),
                release_id=args.release_id,
                dataset=args.target_dataset,
                report=Path(args.duke_report),
            ),
            validate_config(
                Path(args.market_config),
                release_id=args.release_id,
                dataset="market1501",
                report=Path(args.market_report),
            ),
        ]
        for policy in (Path(args.forgetting_policy), Path(args.validation_policy)):
            payload = yaml.safe_load(policy.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload:
                raise ValueError(f"Policy is empty or invalid: {policy}")
        print(json.dumps({"status": "PASS", "checks": checks}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
