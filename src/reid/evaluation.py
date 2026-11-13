from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import numpy as np


def _embedded_preview_asset(value: str, *, max_size: tuple[int, int] = (192, 384)) -> dict[str, str]:
    """Return a small, content-addressed JPEG that survives an ephemeral runner.

    Inbox evaluation views are symlinks below ``RUNNER_TEMP``.  Persisting only
    their paths makes the Control UI lose every preview as soon as the workflow
    exits.  The thumbnail is intentionally stored with the evaluation evidence
    so the report is portable across runners, datasets and future model versions.
    """
    from PIL import Image, ImageOps

    source = Path(value)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(max_size)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=78, optimize=True)
    content = output.getvalue()
    return {
        "asset_id": hashlib.sha256(content).hexdigest(),
        "asset_mime": "image/jpeg",
        "asset_data": base64.b64encode(content).decode("ascii"),
    }


def extract_features(model, loader, device, flip_test: bool = False):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    features = []
    person_ids = []
    camera_ids = []
    paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract", leave=True, dynamic_ncols=True, disable=False):
            images = batch["image"].to(device)
            _, embeddings = model(images)
            if flip_test:
                flipped_images = torch.flip(images, dims=[3])
                _, flipped_embeddings = model(flipped_images)
                embeddings = 0.5 * (embeddings + flipped_embeddings)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            features.append(embeddings.cpu())
            person_ids.extend(batch["pid"].tolist())
            camera_ids.extend(batch["camid"].tolist())
            paths.extend(batch["path"])

    stacked = torch.cat(features, dim=0).numpy()
    return stacked, np.asarray(person_ids), np.asarray(camera_ids), paths


def compute_distance_matrix(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    return 1 - np.matmul(query_features, gallery_features.T)


def _portable_dataset_path(value: str, dataset_root: str | Path | None = None) -> str:
    path = Path(value)
    if dataset_root is not None:
        try:
            return path.resolve().relative_to(Path(dataset_root).resolve()).as_posix()
        except ValueError:
            pass
    for split in ("query", "bounding_box_test", "bounding_box_train"):
        if split in path.parts:
            return Path(*path.parts[path.parts.index(split) :]).as_posix()
    return path.name


def build_retrieval_preview(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_paths: list[str],
    query_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_paths: list[str],
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    *,
    dataset_root: str | Path | None = None,
    topk: int = 5,
    limit_queries: int = 6,
    embed_images: bool = False,
) -> list[dict]:
    """Build deterministic, auditable query-to-gallery examples for the UI.

    The ranking follows the same Market-1501 protocol as aggregate evaluation:
    gallery images with the same identity and camera as the query are excluded.
    Paths are stored relative to the dataset root so reports remain portable from
    a self-hosted runner to the Control API.
    """
    if limit_queries <= 0 or topk <= 0:
        return []
    selected = min(limit_queries, int(query_features.shape[0]))
    distances = compute_distance_matrix(query_features[:selected], gallery_features)
    previews: list[dict] = []
    for query_index in range(selected):
        query_pid = int(query_pids[query_index])
        query_camid = int(query_camids[query_index])
        ranked = np.argsort(distances[query_index])
        predictions: list[dict] = []
        for gallery_index in ranked:
            gallery_pid = int(gallery_pids[gallery_index])
            gallery_camid = int(gallery_camids[gallery_index])
            if gallery_pid == query_pid and gallery_camid == query_camid:
                continue
            distance = float(distances[query_index, gallery_index])
            predictions.append(
                {
                    "rank": len(predictions) + 1,
                    "image_path": _portable_dataset_path(
                        gallery_paths[gallery_index], dataset_root
                    ),
                    "person_id": gallery_pid,
                    "camera_id": gallery_camid,
                    "distance": distance,
                    "similarity": float(1.0 - distance),
                    "correct": gallery_pid == query_pid,
                    **(
                        _embedded_preview_asset(gallery_paths[gallery_index])
                        if embed_images
                        else {}
                    ),
                }
            )
            if len(predictions) >= topk:
                break
        previews.append(
            {
                "query": {
                    "image_path": _portable_dataset_path(
                        query_paths[query_index], dataset_root
                    ),
                    "person_id": query_pid,
                    "camera_id": query_camid,
                    **(
                        _embedded_preview_asset(query_paths[query_index])
                        if embed_images
                        else {}
                    ),
                },
                "predictions": predictions,
                "top1_correct": bool(predictions and predictions[0]["correct"]),
            }
        )
    return previews


def re_rank_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    all_features = np.concatenate([query_features, gallery_features], axis=0).astype(np.float32)
    all_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
    original_dist = 2.0 - 2.0 * np.matmul(all_features, all_features.T)
    original_dist = np.clip(original_dist, 0.0, None)
    original_dist = np.transpose(original_dist / np.maximum(np.max(original_dist, axis=0), 1e-12))

    all_num = original_dist.shape[0]
    query_num = query_features.shape[0]
    v = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    for i in range(all_num):
        forward_neighbors = initial_rank[i, : k1 + 1]
        backward_neighbors = initial_rank[forward_neighbors, : k1 + 1]
        reciprocal = forward_neighbors[np.where(backward_neighbors == i)[0]]
        reciprocal_expansion = reciprocal.copy()

        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2)) + 1]
            candidate_reciprocal = candidate_forward[np.where(candidate_backward == candidate)[0]]
            if len(np.intersect1d(candidate_reciprocal, reciprocal)) > (2.0 / 3.0) * len(candidate_reciprocal):
                reciprocal_expansion = np.append(reciprocal_expansion, candidate_reciprocal)

        reciprocal_expansion = np.unique(reciprocal_expansion)
        weights = np.exp(-original_dist[i, reciprocal_expansion])
        v[i, reciprocal_expansion] = weights / np.sum(weights)

    if k2 > 1:
        v_qe = np.zeros_like(v, dtype=np.float32)
        for i in range(all_num):
            v_qe[i, :] = np.mean(v[initial_rank[i, :k2], :], axis=0)
        v = v_qe

    inv_index = [np.where(v[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        non_zero = np.where(v[i, :] != 0)[0]
        related = [inv_index[idx] for idx in non_zero]
        for j, related_images in enumerate(related):
            temp_min[0, related_images] += np.minimum(v[i, non_zero[j]], v[related_images, non_zero[j]])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:query_num, :] * lambda_value
    return final_dist[:, query_num:]


def evaluate_market1501(
    distance_matrix: np.ndarray,
    query_pid: np.ndarray,
    gallery_pid: np.ndarray,
    query_cam: np.ndarray,
    gallery_cam: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float, float, int]:
    indices = np.argsort(distance_matrix, axis=1)
    matches = (gallery_pid[indices] == query_pid[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    all_inp = []

    for query_idx in range(distance_matrix.shape[0]):
        q_pid = query_pid[query_idx]
        q_cam = query_cam[query_idx]
        order = indices[query_idx]

        remove = (gallery_pid[order] == q_pid) & (gallery_cam[order] == q_cam)
        keep = np.invert(remove)
        raw_cmc = matches[query_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = raw_cmc.sum()
        precision = raw_cmc.cumsum() / (np.arange(raw_cmc.shape[0]) + 1)
        ap = (precision * raw_cmc).sum() / num_rel
        all_ap.append(ap)

        hardest_match_rank = np.flatnonzero(raw_cmc)[-1] + 1
        inp = num_rel / hardest_match_rank
        all_inp.append(float(inp))

    if not all_cmc:
        raise RuntimeError("No valid query samples were found during evaluation.")

    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    mean_ap = float(np.mean(all_ap))
    mean_inp = float(np.mean(all_inp))
    return cmc, mean_ap, mean_inp, len(all_cmc)
