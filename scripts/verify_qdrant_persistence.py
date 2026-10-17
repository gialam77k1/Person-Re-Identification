from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def snapshot(base_url: str, collection: str) -> dict[str, Any]:
    info = request_json("GET", f"{base_url}/collections/{collection}")["result"]
    scroll = request_json(
        "POST",
        f"{base_url}/collections/{collection}/points/scroll",
        {"limit": 1, "with_payload": True, "with_vector": True},
    )["result"]["points"]
    if not scroll:
        raise RuntimeError("Qdrant collection is empty.")
    point_fingerprint = hashlib.sha256(
        json.dumps(scroll[0], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": info.get("status"),
        "points_count": info.get("points_count"),
        "indexed_vectors_count": info.get("indexed_vectors_count"),
        "vectors": info.get("config", {}).get("params", {}).get("vectors"),
        "sample_point_sha256": point_fingerprint,
    }


def wait_ready(base_url: str, attempts: int = 30) -> None:
    for _ in range(attempts):
        try:
            request = urllib.request.Request(f"{base_url}/healthz", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError("Qdrant did not become ready after restart.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qdrant data survives a container restart.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="reid_reference_v1")
    parser.add_argument("--compose-file", default="docker-compose.mlops.yml")
    parser.add_argument("--env-file", default=".env.mlops")
    parser.add_argument("--output", default="artifacts/deployment-tests/qdrant_persistence.json")
    args = parser.parse_args()

    before = snapshot(args.qdrant_url, args.collection)
    subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            args.env_file,
            "-f",
            args.compose_file,
            "restart",
            "qdrant",
        ],
        check=True,
    )
    wait_ready(args.qdrant_url)
    after = snapshot(args.qdrant_url, args.collection)
    passed = before == after and after["status"] == "green" and after["points_count"] > 0
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "collection": args.collection,
        "before": before,
        "after": after,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
