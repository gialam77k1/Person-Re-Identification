from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deployed retrieval quality against offline metrics.")
    parser.add_argument("--offline-metrics", required=True)
    parser.add_argument("--deployment-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank1-max-drop", type=float, default=0.005)
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def build_quality_report(
    offline: dict[str, Any],
    deployment: dict[str, Any],
    *,
    rank1_max_drop: float,
    embedding_dim: int,
) -> dict[str, Any]:
    offline_rank1 = float(offline.get("rank1_base", offline.get("rank1")))
    deployment_rank1 = float(deployment["rank1"])
    rank1_drop = offline_rank1 - deployment_rank1
    checks = [
        {
            "name": "dataset_match",
            "passed": deployment.get("dataset") == offline.get("dataset"),
            "actual": deployment.get("dataset"),
            "expected": offline.get("dataset"),
        },
        {
            "name": "valid_query_count_match",
            "passed": deployment.get("valid_queries") == offline.get("valid_queries"),
            "actual": deployment.get("valid_queries"),
            "expected": offline.get("valid_queries"),
        },
        {
            "name": "rank1_drop",
            "passed": rank1_drop <= rank1_max_drop,
            "actual": rank1_drop,
            "maximum": rank1_max_drop,
        },
        {
            "name": "embedding_dim",
            "passed": deployment.get("embedding_dim") == embedding_dim,
            "actual": deployment.get("embedding_dim"),
            "expected": embedding_dim,
        },
        {
            "name": "embeddings_finite",
            "passed": deployment.get("embeddings_finite") is True,
            "actual": deployment.get("embeddings_finite"),
            "expected": True,
        },
        {
            "name": "embeddings_l2_normalized",
            "passed": deployment.get("embeddings_l2_normalized") is True,
            "actual": deployment.get("embeddings_l2_normalized"),
            "expected": True,
        },
    ]
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "offline_protocol": {
            "rank1_metric": "rank1_base",
            "rank1": offline_rank1,
            "flip_test": offline.get("flip_test"),
            "reranking": False,
        },
        "deployment_protocol": deployment.get("protocol"),
        "deployment_rank1": deployment_rank1,
        "rank1_drop": rank1_drop,
        "rank1_max_drop": rank1_max_drop,
        "checks": checks,
        "failed_checks": failed,
    }


def main() -> None:
    args = parse_args()
    report = build_quality_report(
        read_json(args.offline_metrics),
        read_json(args.deployment_metrics),
        rank1_max_drop=args.rank1_max_drop,
        embedding_dim=args.embedding_dim,
    )
    save_json(report, args.output)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
