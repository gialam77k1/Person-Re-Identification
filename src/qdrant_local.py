from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import ensure_dir, save_json, tee_output


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant HTTP error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Qdrant at {url}") from exc


def create_collection(args: argparse.Namespace) -> None:
    url = f"{args.qdrant_url.rstrip('/')}/collections/{args.collection_name}"
    payload = {
        "vectors": {
            "size": args.vector_size,
            "distance": args.distance,
        }
    }
    result = _request_json("PUT", url, payload)
    print(json.dumps(result, indent=2))


def upsert_reference_embeddings(args: argparse.Namespace) -> None:
    embeddings = np.load(args.embeddings_path)
    pids = np.load(args.pids_path)
    camids = np.load(args.camids_path)

    if embeddings.ndim != 2:
        raise ValueError("Reference embeddings must have shape [N, D].")
    if len(embeddings) != len(pids) or len(embeddings) != len(camids):
        raise ValueError("Embeddings, pids, and camids must have the same length.")

    points = []
    for index, (vector, pid, camid) in enumerate(zip(embeddings, pids, camids, strict=True)):
        points.append(
            {
                "id": index + 1,
                "vector": vector.astype(float).tolist(),
                "payload": {
                    "pid": int(pid),
                    "camid": int(camid),
                },
            }
        )

    url = f"{args.qdrant_url.rstrip('/')}/collections/{args.collection_name}/points"
    batch_results = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        result = _request_json("PUT", url, {"points": batch_points})
        batch_results.append(
            {
                "start_index": start,
                "end_index_exclusive": start + len(batch_points),
                "result": result,
            }
        )

    manifest = {
        "collection_name": args.collection_name,
        "num_points": len(points),
        "batch_size": batch_size,
        "num_batches": len(batch_results),
        "embeddings_path": str(Path(args.embeddings_path).resolve()),
        "pids_path": str(Path(args.pids_path).resolve()),
        "camids_path": str(Path(args.camids_path).resolve()),
        "qdrant_result": batch_results[-1]["result"] if batch_results else {},
    }
    save_json(manifest, Path(args.output_root) / "qdrant_upsert_manifest.json")
    print(json.dumps(manifest, indent=2))


def query_embedding(args: argparse.Namespace) -> None:
    query_vector = np.load(args.embedding_path)
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector[0]
    if query_vector.ndim != 1:
        raise ValueError("Query embedding must have shape [D] or [1, D].")

    url = f"{args.qdrant_url.rstrip('/')}/collections/{args.collection_name}/points/query"
    payload = {
        "query": query_vector.astype(float).tolist(),
        "limit": args.limit,
        "with_payload": True,
        "with_vector": False,
    }
    result = _request_json("POST", url, payload)
    manifest = {
        "collection_name": args.collection_name,
        "embedding_path": str(Path(args.embedding_path).resolve()),
        "limit": args.limit,
        "result": result,
    }
    stem = Path(args.embedding_path).stem
    save_json(manifest, Path(args.output_root) / f"{stem}_qdrant_query.json")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--output-root", default="artifacts/qdrant/local")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-collection")
    create_parser.add_argument("--collection-name", default="reid_reference")
    create_parser.add_argument("--vector-size", type=int, default=512)
    create_parser.add_argument("--distance", default="Cosine")
    create_parser.set_defaults(handler=create_collection)

    upsert_parser = subparsers.add_parser("upsert-reference")
    upsert_parser.add_argument("--collection-name", default="reid_reference")
    upsert_parser.add_argument("--embeddings-path", required=True)
    upsert_parser.add_argument("--pids-path", required=True)
    upsert_parser.add_argument("--camids-path", required=True)
    upsert_parser.add_argument("--batch-size", type=int, default=512)
    upsert_parser.set_defaults(handler=upsert_reference_embeddings)

    query_parser = subparsers.add_parser("query-embedding")
    query_parser.add_argument("--collection-name", default="reid_reference")
    query_parser.add_argument("--embedding-path", required=True)
    query_parser.add_argument("--limit", type=int, default=5)
    query_parser.set_defaults(handler=query_embedding)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = ensure_dir(args.output_root)
    log_path = output_root / "qdrant_local.log"
    with tee_output(log_path):
        print(f"Logging console output to {log_path}")
        args.handler(args)


if __name__ == "__main__":
    main()
