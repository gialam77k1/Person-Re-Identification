from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    person_ids = []
    camera_ids = []
    paths = []

    for batch in tqdm(loader, desc="Extract", leave=True, dynamic_ncols=True, disable=False):
        images = batch["image"].to(device)
        _, embeddings = model(images)
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
) -> tuple[np.ndarray, float]:
    indices = np.argsort(distance_matrix, axis=1)
    matches = (gallery_pid[indices] == query_pid[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []

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

    if not all_cmc:
        raise RuntimeError("No valid query samples were found during evaluation.")

    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    mean_ap = float(np.mean(all_ap))
    return cmc, mean_ap
