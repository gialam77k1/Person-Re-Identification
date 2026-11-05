from __future__ import annotations

from typing import Any

import numpy as np

from src.reid.evaluation import compute_distance_matrix, evaluate_market1501


def _validate_embeddings(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} embeddings must be a non-empty 2D array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} embeddings contain NaN or Inf.")
    if np.any(np.linalg.norm(array, axis=1) <= 1e-12):
        raise ValueError(f"{name} embeddings contain a zero-norm row.")
    return array


def evaluate_cross_domain_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    query_batch_size: int = 128,
    max_rank: int = 50,
) -> dict[str, Any]:
    """Evaluate retrieval in query batches to avoid a full MSMT17 distance matrix."""
    query = _validate_embeddings("query", query_embeddings)
    gallery = _validate_embeddings("gallery", gallery_embeddings)
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery embedding dimensions differ.")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be at least 1.")

    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    query_camids = np.asarray(query_camids)
    gallery_camids = np.asarray(gallery_camids)
    if query.shape[0] != query_pids.size or query.shape[0] != query_camids.size:
        raise ValueError("Query metadata length does not match query embeddings.")
    if gallery.shape[0] != gallery_pids.size or gallery.shape[0] != gallery_camids.size:
        raise ValueError("Gallery metadata length does not match gallery embeddings.")

    weighted_cmc = np.zeros(min(max_rank, gallery.shape[0]), dtype=np.float64)
    weighted_map = 0.0
    weighted_minp = 0.0
    valid_queries = 0
    for start in range(0, query.shape[0], query_batch_size):
        stop = min(start + query_batch_size, query.shape[0])
        distances = compute_distance_matrix(query[start:stop], gallery)
        try:
            cmc, mean_ap, mean_inp, batch_valid = evaluate_market1501(
                distances,
                query_pids[start:stop],
                gallery_pids,
                query_camids[start:stop],
                gallery_camids,
                max_rank=max_rank,
            )
        except RuntimeError as exc:
            if "No valid query samples" in str(exc):
                continue
            raise
        weighted_cmc[: cmc.size] += cmc.astype(np.float64) * batch_valid
        weighted_map += mean_ap * batch_valid
        weighted_minp += mean_inp * batch_valid
        valid_queries += batch_valid

    if valid_queries == 0:
        raise RuntimeError("No valid query samples were found during cross-domain evaluation.")

    cmc = weighted_cmc / valid_queries

    def rank_at(rank: int) -> float:
        return float(cmc[min(rank - 1, cmc.size - 1)])

    return {
        "rank1": rank_at(1),
        "rank5": rank_at(5),
        "rank10": rank_at(10),
        "rank20": rank_at(20),
        "mAP": float(weighted_map / valid_queries),
        "mINP": float(weighted_minp / valid_queries),
        "valid_queries": int(valid_queries),
        "query_coverage": float(valid_queries / query.shape[0]),
        "embedding_dim": int(query.shape[1]),
    }
