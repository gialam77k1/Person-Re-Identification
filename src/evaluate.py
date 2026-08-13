from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.utils import configure_torch_home, infer_device, load_checkpoint, save_json
from src.data.dataset import build_dataset_splits, build_transforms
from src.models.reid_model import build_model_from_config
from src.reid.evaluation import (
    compute_distance_matrix,
    evaluate_market1501,
    extract_features,
    re_rank_distance_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dadnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set evaluation.use_rerank=true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config, args.set, command_name="evaluate")
    configure_torch_home()
    device = infer_device(config["device"])
    _, test_transform = build_transforms(config["data"]["image_height"], config["data"]["image_width"])

    train_dataset, query_dataset, gallery_dataset = build_dataset_splits(
        config,
        train_transform=test_transform,
        test_transform=test_transform,
    )

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

    flip_test = config["evaluation"].get("flip_test", False)
    query_features, query_pid, query_cam, _ = extract_features(model, query_loader, device, flip_test=flip_test)
    gallery_features, gallery_pid, gallery_cam, _ = extract_features(
        model,
        gallery_loader,
        device,
        flip_test=flip_test,
    )
    base_distance_matrix = compute_distance_matrix(query_features, gallery_features)
    base_cmc, base_mean_ap, base_mean_inp, valid_queries = evaluate_market1501(
        base_distance_matrix,
        query_pid,
        gallery_pid,
        query_cam,
        gallery_cam,
    )

    results = {
        "run_slug": config["runtime"]["run_slug"],
        "run_root": config["runtime"]["run_root"],
        "dataset": config["data"]["dataset"]["name"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "loaded_epoch": checkpoint.get("epoch"),
        "flip_test": bool(flip_test),
        "use_rerank": bool(config["evaluation"].get("use_rerank", False)),
        "valid_queries": int(valid_queries),
        "rank1_base": float(base_cmc[0]),
        "rank5_base": float(base_cmc[4]),
        "rank10_base": float(base_cmc[9]),
        "rank20_base": float(base_cmc[19]),
        "mAP_base": float(base_mean_ap),
        "mINP_base": float(base_mean_inp),
    }

    if config["evaluation"].get("use_rerank", False):
        rerank_distance_matrix = re_rank_distance_matrix(
            query_features,
            gallery_features,
            k1=config["evaluation"].get("rerank_k1", 20),
            k2=config["evaluation"].get("rerank_k2", 6),
            lambda_value=config["evaluation"].get("rerank_lambda", 0.3),
        )
        rerank_cmc, rerank_mean_ap, rerank_mean_inp, _ = evaluate_market1501(
            rerank_distance_matrix,
            query_pid,
            gallery_pid,
            query_cam,
            gallery_cam,
        )
        results.update(
            {
                "rank1_rerank": float(rerank_cmc[0]),
                "rank5_rerank": float(rerank_cmc[4]),
                "rank10_rerank": float(rerank_cmc[9]),
                "rank20_rerank": float(rerank_cmc[19]),
                "mAP_rerank": float(rerank_mean_ap),
                "mINP_rerank": float(rerank_mean_inp),
                "rank1": float(rerank_cmc[0]),
                "rank5": float(rerank_cmc[4]),
                "rank10": float(rerank_cmc[9]),
                "rank20": float(rerank_cmc[19]),
                "mAP": float(rerank_mean_ap),
                "mINP": float(rerank_mean_inp),
            }
        )
    else:
        results.update(
            {
                "rank1": float(base_cmc[0]),
                "rank5": float(base_cmc[4]),
                "rank10": float(base_cmc[9]),
                "rank20": float(base_cmc[19]),
                "mAP": float(base_mean_ap),
                "mINP": float(base_mean_inp),
            }
        )
    print(results)

    output_path = Path(config["artifacts"]["metrics_dir"]) / "evaluation_latest.json"
    save_json(results, output_path)
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")


if __name__ == "__main__":
    main()
