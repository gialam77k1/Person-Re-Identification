from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_report(path: Path, release_id: str, dataset: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("release_id") != release_id:
        raise ValueError("Evaluation release does not match the requested v2 release.")
    if str(report.get("dataset", "")).lower() != dataset.lower():
        raise ValueError("Evaluation dataset does not match the requested dataset.")
    for key in ("rank1", "rank5", "mAP", "mINP"):
        value = report.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Evaluation metric {key} is missing or non-finite.")
    if int(report.get("valid_queries", 0)) <= 0:
        raise ValueError("Evaluation did not produce any valid queries.")
    for key in ("query_count", "gallery_count"):
        if int(report.get(key, 0)) <= 0:
            raise ValueError(f"Evaluation {key} is missing or invalid.")
    for key in ("checkpoint_sha256", "git_commit", "dataset_dvc_revision"):
        if not str(report.get(key, "")).strip():
            raise ValueError(f"Evaluation lineage field {key} is missing.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the DukeMTMC-reID v2 evaluation output.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--dataset", default="duke")
    args = parser.parse_args()
    try:
        report = validate_report(Path(args.report), args.release_id, args.dataset)
        print(json.dumps({"status": "PASS", "metrics": {key: report[key] for key in ("rank1", "rank5", "mAP", "mINP")}}, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
