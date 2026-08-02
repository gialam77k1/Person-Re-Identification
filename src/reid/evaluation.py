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

    for batch in tqdm(loader, desc="Extract", leave=False):
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
