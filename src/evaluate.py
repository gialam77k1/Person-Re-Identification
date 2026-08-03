from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.utils import configure_torch_home, infer_device, load_checkpoint, save_json
from src.data.dataset import Market1501Dataset, build_transforms
from src.models.reid_model import build_model_from_config
from src.reid.evaluation import (
    compute_distance_matrix,
    evaluate_market1501,
    extract_features,
    re_rank_distance_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_torch_home()
    device = infer_device(config["device"])
    _, test_transform = build_transforms(config["data"]["image_height"], config["data"]["image_width"])

    train_dataset = Market1501Dataset(config["data"]["train_dir"], transform=test_transform, relabel=True)
    query_dataset = Market1501Dataset(config["data"]["query_dir"], transform=test_transform, relabel=False)
    gallery_dataset = Market1501Dataset(config["data"]["gallery_dir"], transform=test_transform, relabel=False)

    query_loader = DataLoader(
        query_dataset,
        batch_size=config["data"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config["data"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = build_model_from_config(
        config,
        num_classes=train_dataset.num_classes,
        pretrained=False,
    ).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)

    query_features, query_pid, query_cam, _ = extract_features(model, query_loader, device)
    gallery_features, gallery_pid, gallery_cam, _ = extract_features(model, gallery_loader, device)
    distance_matrix = compute_distance_matrix(query_features, gallery_features)
    if config["evaluation"].get("use_rerank", False):
        distance_matrix = re_rank_distance_matrix(
            query_features,
            gallery_features,
            k1=config["evaluation"].get("rerank_k1", 20),
            k2=config["evaluation"].get("rerank_k2", 6),
            lambda_value=config["evaluation"].get("rerank_lambda", 0.3),
        )
    cmc, mean_ap = evaluate_market1501(distance_matrix, query_pid, gallery_pid, query_cam, gallery_cam)

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "loaded_epoch": checkpoint.get("epoch"),
        "use_rerank": bool(config["evaluation"].get("use_rerank", False)),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mean_ap),
    }
    print(results)

    output_path = Path(config["artifacts"]["metrics_dir"]) / "evaluation_latest.json"
    save_json(results, output_path)


if __name__ == "__main__":
    main()
