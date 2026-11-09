from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.utils import save_json
from src.evaluate_deployment_retrieval import infer_dataset_embeddings
from src.validate_data import dvc_revision


def deterministic_sample(samples: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(samples):
        return list(samples)
    generator = random.Random(seed)
    selected_indices = sorted(generator.sample(range(len(samples)), limit))
    return [samples[index] for index in selected_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture labeled external-dataset embeddings from Triton.")
    parser.add_argument("--config", default="configs/duke_v1_evaluation.yaml")
    parser.add_argument("--dataset-pointer", default="datasets/dukemtmc-reid.dvc")
    parser.add_argument("--split", choices=("train", "query", "gallery"), default="query")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--release-id", default="market1501-vit-bnneck:v1-dvc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--jsonl-output", default="")
    return parser.parse_args()


def main() -> None:
    from src.data.dataset import build_dataset

    args = parse_args()
    config = load_runtime_config(args.config, command_name="capture-external-embeddings")
    dataset = build_dataset(config, args.split, transform=None, relabel=False)
    dataset.samples = deterministic_sample(dataset.samples, args.max_samples, args.seed)
    embeddings, person_ids, camera_ids, paths = infer_dataset_embeddings(
        dataset=dataset,
        split_name=args.split,
        server_url=args.server_url,
        model_name="reid_embedding",
        input_name="images",
        output_name="embeddings",
        input_height=int(config["data"]["image_height"]),
        input_width=int(config["data"]["image_width"]),
        max_samples=0,
        preserve_aspect_ratio=bool(config["data"].get("preserve_aspect_ratio", False)),
        workers=args.workers,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings, allow_pickle=False)
    pointer_path = (ROOT / args.dataset_pointer).resolve()
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_id": args.release_id,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": config["data"]["dataset"]["name"],
        "dataset_dvc_revision": dvc_revision(pointer_path),
        "split": args.split,
        "sample_seed": args.seed,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_finite": bool(np.isfinite(embeddings).all()),
        "paths": paths,
        "person_ids": person_ids.tolist(),
        "camera_ids": camera_ids.tolist(),
    }
    save_json(metadata, args.metadata_output)

    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for index, embedding in enumerate(embeddings):
                record = {
                    "release_id": args.release_id,
                    "dataset": metadata["dataset"],
                    "path": paths[index],
                    "person_id": int(person_ids[index]),
                    "camera_id": int(camera_ids[index]),
                    "embedding": embedding.tolist(),
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(json.dumps({key: metadata[key] for key in metadata if key not in {"paths", "person_ids", "camera_ids"}}, indent=2))


if __name__ == "__main__":
    main()
