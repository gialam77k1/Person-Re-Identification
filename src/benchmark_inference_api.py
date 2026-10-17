from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import save_json
from src.deployment_http import post_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI embedding or search latency.")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--endpoint", choices=("embed", "search"), default="search")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="artifacts/deployment-tests/latency_benchmark.json")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def validate_response(endpoint: str, response: dict[str, Any], top_k: int) -> None:
    if endpoint == "embed":
        embedding = np.asarray(response.get("embedding"), dtype=np.float32)
        if embedding.shape != (512,) or not np.all(np.isfinite(embedding)):
            raise ValueError("Embed response must contain 512 finite values.")
    else:
        results = response.get("results")
        if not isinstance(results, list) or len(results) != top_k:
            raise ValueError(f"Search response must contain exactly {top_k} results.")


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.workers < 1:
        raise ValueError("iterations/workers must be positive and warmup must be non-negative.")
    endpoint_url = f"{args.api_url.rstrip('/')}/{args.endpoint}"
    query = {"top_k": args.top_k} if args.endpoint == "search" else None

    def request_once() -> float:
        started = time.perf_counter()
        response = post_image_path(endpoint_url, args.image_path, query=query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        validate_response(args.endpoint, response, args.top_k)
        return elapsed_ms

    for _ in range(args.warmup):
        request_once()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        latencies = list(executor.map(lambda _: request_once(), range(args.iterations)))
    wall_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "endpoint": f"/{args.endpoint}",
        "image_path": str(Path(args.image_path).resolve()),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "workers": args.workers,
        "successful_requests": len(latencies),
        "failed_requests": 0,
        "latency_ms": {
            "min": min(latencies),
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies),
        },
        "throughput_requests_per_second": len(latencies) / max(wall_seconds, 1e-12),
    }
    if not all(math.isfinite(value) for value in report["latency_ms"].values()):
        raise RuntimeError("Benchmark produced a non-finite latency.")
    save_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
