from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update a Control Center pipeline stage.")
    parser.add_argument("cycle_id")
    parser.add_argument("stage")
    parser.add_argument("status")
    parser.add_argument("--progress", type=float)
    parser.add_argument("--detail", default="")
    parser.add_argument("--metrics", default="{}")
    parser.add_argument(
        "--metrics-file",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="Merge a JSON report, or attach it below NAME. May be repeated.",
    )
    return parser.parse_args()


def load_metrics(inline: str, metrics_files: str | list[str] | None) -> dict:
    metrics = json.loads(inline)
    if isinstance(metrics_files, str):
        metrics_files = [metrics_files]
    for spec in metrics_files or []:
        name = None
        path = spec
        if "=" in spec:
            name, path = spec.split("=", 1)
            if not name or not path:
                raise ValueError(f"Invalid named metrics file: {spec}")
        file_metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(file_metrics, dict):
            raise ValueError(f"Metrics file must contain a JSON object: {path}")
        if name:
            metrics[name] = file_metrics
        else:
            file_metrics.update(metrics)
            metrics = file_metrics
    return metrics


def main() -> int:
    args = parse_args()
    if args.cycle_id in {"", "manual"}:
        print("Control callback skipped for manual workflow run.")
        return 0
    base_url = os.getenv("CONTROL_API_URL", "http://localhost:8090").rstrip("/")
    token = os.getenv("CONTROL_CALLBACK_TOKEN", "")
    if not token:
        print("ERROR: CONTROL_CALLBACK_TOKEN is required", file=sys.stderr)
        return 1
    try:
        metrics = load_metrics(args.metrics, args.metrics_file)
        payload = json.dumps(
            {
                "status": args.status,
                "progress": args.progress,
                "detail": args.detail or None,
                "metrics": metrics,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/control/cycles/{args.cycle_id}/stages/{args.stage}",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Control-Callback-Token": token,
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"callback returned HTTP {response.status}")
        return 0
    except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
