from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.utils import ensure_dir, save_json, tee_output
from src.reid.cross_domain import evaluate_cross_domain_retrieval
from src.reid.evaluation import compute_distance_matrix
from src.triton_infer import preprocess_image
from src.triton_infer import infer_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_reid.yaml")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="reid_embedding")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-name", default="embeddings")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-gallery", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--release-id", default="market1501-vit-bnneck:v1-dvc")
    parser.add_argument("--dataset-pointer", default="datasets/MSMT17_V1.dvc")
    parser.add_argument("--output", default="")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--gallery-match-query-pids-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent Triton requests. Keep 1 for serial evaluation.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.dataset.name=market1501",
    )
    return parser.parse_args()


def infer_dataset_embeddings(
    dataset,
    split_name: str,
    server_url: str,
    model_name: str,
    input_name: str,
    output_name: str,
    input_height: int,
    input_width: int,
    max_samples: int,
    preserve_aspect_ratio: bool = False,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    selected_samples = dataset.samples[:max_samples] if max_samples > 0 else dataset.samples
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def infer_sample(sample: dict) -> np.ndarray:
        tensor = preprocess_image(
            sample["img_path"],
            input_height=input_height,
            input_width=input_width,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        return np.asarray(
            infer_embedding(
                server_url=server_url,
                model_name=model_name,
                input_name=input_name,
                output_name=output_name,
                tensor=tensor,
            ),
            dtype=np.float32,
        )

    embeddings: list[np.ndarray] = []
    person_ids = []
    camera_ids = []
    paths: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inferred = executor.map(infer_sample, selected_samples)
        progress = tqdm(
            zip(selected_samples, inferred),
            total=len(selected_samples),
            desc=f"Triton {split_name} infer",
            dynamic_ncols=True,
        )
        for sample, embedding in progress:
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            if embedding.ndim != 1:
                raise ValueError(
                    f"Triton embedding for {sample['img_path']} must have shape [D] or [1, D], got {embedding.shape}."
                )

            norm = np.linalg.norm(embedding)
            if not np.isfinite(embedding).all():
                raise ValueError(f"Triton embedding is not finite for {split_name} image {sample['img_path']}.")
            if norm <= 1e-12:
                raise ValueError(f"Triton embedding norm is zero for {split_name} image {sample['img_path']}.")

            embeddings.append((embedding / norm).astype(np.float32))
            person_ids.append(int(sample["pid"]))
            camera_ids.append(int(sample["camid"]))
            paths.append(str(sample["img_path"]))

    if not embeddings:
        raise RuntimeError(f"No sample was selected from split '{split_name}' for deployment retrieval evaluation.")

    return (
        np.stack(embeddings, axis=0),
        np.asarray(person_ids, dtype=np.int32),
        np.asarray(camera_ids, dtype=np.int32),
        paths,
    )


def filter_samples_by_pid(samples: list[dict[str, int | str]], allowed_pids: set[int]) -> list[dict[str, int | str]]:
    return [sample for sample in samples if int(sample["pid"]) in allowed_pids]


def rank_at(cmc: np.ndarray, rank: int) -> float:
    if cmc.size == 0:
        return 0.0
    index = min(rank - 1, cmc.shape[0] - 1)
    return float(cmc[index])


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def dvc_revision(pointer_path: Path) -> str:
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    outputs = pointer.get("outs", []) if isinstance(pointer, dict) else []
    if len(outputs) != 1 or not outputs[0].get("md5"):
        raise ValueError(f"Invalid DVC pointer: {pointer_path}")
    return str(outputs[0]["md5"])


def build_topk_preview(
    distance_matrix: np.ndarray,
    query_paths: list[str],
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    gallery_paths: list[str],
    topk: int = 5,
    limit_queries: int = 5,
) -> list[dict]:
    ranked_indices = np.argsort(distance_matrix, axis=1)
    previews = []
    for query_index in range(min(limit_queries, distance_matrix.shape[0])):
        matches = []
        for gallery_index in ranked_indices[query_index, :topk]:
            matches.append(
                {
                    "gallery_path": str(Path(gallery_paths[gallery_index]).resolve()),
                    "gallery_pid": int(gallery_pids[gallery_index]),
                    "gallery_camid": int(gallery_camids[gallery_index]),
                    "distance": float(distance_matrix[query_index, gallery_index]),
                }
            )

        previews.append(
            {
                "query_path": str(Path(query_paths[query_index]).resolve()),
                "query_pid": int(query_pids[query_index]),
                "top_matches": matches,
            }
        )
    return previews


def main() -> None:
    from src.data.dataset import build_dataset

    args = parse_args()
    config = load_runtime_config(args.config, args.set, command_name="deployment-retrieval-evaluate")
    ensure_dir(config["artifacts"]["logs_dir"])
    ensure_dir(config["artifacts"]["metrics_dir"])

    log_path = Path(config["artifacts"]["logs_dir"]) / "deployment_retrieval_evaluate.log"
    with tee_output(log_path):
        print(f"Logging console output to {log_path}")

        query_dataset = build_dataset(config, "query", transform=None, relabel=False)
        gallery_dataset = build_dataset(config, "gallery", transform=None, relabel=False)

        query_vectors, query_pids, query_camids, query_paths = infer_dataset_embeddings(
            dataset=query_dataset,
            split_name="query",
            server_url=args.server_url,
            model_name=args.model_name,
            input_name=args.input_name,
            output_name=args.output_name,
            input_height=int(config["data"]["image_height"]),
            input_width=int(config["data"]["image_width"]),
            max_samples=args.max_queries,
            preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
            workers=args.workers,
        )
        if args.gallery_match_query_pids_only:
            selected_query_pids = set(query_pids.tolist())
            gallery_dataset.samples = filter_samples_by_pid(gallery_dataset.samples, selected_query_pids)
            if not gallery_dataset.samples:
                raise RuntimeError("No gallery sample remained after filtering by query pid set.")

        gallery_vectors, gallery_pids, gallery_camids, gallery_paths = infer_dataset_embeddings(
            dataset=gallery_dataset,
            split_name="gallery",
            server_url=args.server_url,
            model_name=args.model_name,
            input_name=args.input_name,
            output_name=args.output_name,
            input_height=int(config["data"]["image_height"]),
            input_width=int(config["data"]["image_width"]),
            max_samples=args.max_gallery,
            preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
            workers=args.workers,
        )

        retrieval_metrics = evaluate_cross_domain_retrieval(
            query_vectors,
            gallery_vectors,
            query_pids,
            gallery_pids,
            query_camids,
            gallery_camids,
            query_batch_size=args.query_batch_size,
        )

        preview_query_count = min(5, query_vectors.shape[0])
        preview_distances = compute_distance_matrix(query_vectors[:preview_query_count], gallery_vectors)
        pointer_path = (ROOT / args.dataset_pointer).resolve()

        results = {
            "schema_version": 3,
            "release_id": args.release_id,
            "git_commit": git_commit(),
            "dataset_dvc_revision": dvc_revision(pointer_path),
            "run_slug": config["runtime"]["run_slug"],
            "run_root": config["runtime"]["run_root"],
            "dataset": config["data"]["dataset"]["name"],
            "server_url": args.server_url,
            "model_name": args.model_name,
            "protocol": "Triton HTTP, letterbox preprocessing, L2-normalized embeddings, cosine distance, no flip test, no re-ranking",
            "workers": args.workers,
            "num_query_samples": int(query_vectors.shape[0]),
            "num_gallery_samples": int(gallery_vectors.shape[0]),
            "query_batch_size": args.query_batch_size,
            "embeddings_finite": bool(
                np.all(np.isfinite(query_vectors)) and np.all(np.isfinite(gallery_vectors))
            ),
            "embeddings_l2_normalized": bool(
                np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-5)
                and np.allclose(np.linalg.norm(gallery_vectors, axis=1), 1.0, atol=1e-5)
            ),
            **retrieval_metrics,
            "topk_preview": build_topk_preview(
                distance_matrix=preview_distances,
                query_paths=query_paths[:preview_query_count],
                query_pids=query_pids[:preview_query_count],
                gallery_pids=gallery_pids,
                gallery_camids=gallery_camids,
                gallery_paths=gallery_paths,
            ),
        }

        if args.save_embeddings:
            embeddings_dir = ensure_dir(config["artifacts"]["embeddings_dir"])
            np.save(embeddings_dir / "query_embeddings.npy", query_vectors, allow_pickle=False)
            np.save(embeddings_dir / "gallery_embeddings.npy", gallery_vectors, allow_pickle=False)
            save_json(
                {
                    "release_id": args.release_id,
                    "dataset": config["data"]["dataset"]["name"],
                    "dataset_dvc_revision": results["dataset_dvc_revision"],
                    "query_paths": query_paths,
                    "query_pids": query_pids.tolist(),
                    "query_camids": query_camids.tolist(),
                    "gallery_paths": gallery_paths,
                    "gallery_pids": gallery_pids.tolist(),
                    "gallery_camids": gallery_camids.tolist(),
                },
                embeddings_dir / "metadata.json",
            )

        output_path = (
            Path(args.output)
            if args.output
            else Path(config["artifacts"]["metrics_dir"]) / "deployment_retrieval_latest.json"
        )
        saved_output = save_json(results, output_path)
        save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")
        if args.log_mlflow:
            import mlflow

            mlflow.set_tracking_uri(config["logging"]["tracking_uri"])
            mlflow.set_experiment(config["experiment_name"])
            with mlflow.start_run(run_name=f"{args.release_id}-on-{config['data']['dataset']['name']}"):
                mlflow.log_params(
                    {
                        "release_id": args.release_id,
                        "dataset": config["data"]["dataset"]["name"],
                        "dataset_dvc_revision": results["dataset_dvc_revision"],
                        "git_commit": results["git_commit"],
                        "num_query_samples": results["num_query_samples"],
                        "num_gallery_samples": results["num_gallery_samples"],
                    }
                )
                mlflow.log_metrics(
                    {
                        name: float(results[name])
                        for name in ("rank1", "rank5", "rank10", "rank20", "mAP", "mINP")
                    }
                )
                mlflow.log_artifact(str(saved_output), artifact_path="cross-domain-evaluation")
        print(results)


if __name__ == "__main__":
    main()
