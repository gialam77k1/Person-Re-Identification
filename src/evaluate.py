from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.mlflow_utils import get_dvc_revision, get_git_commit
from src.common.utils import configure_torch_home, ensure_dir, infer_device, save_json, tee_output
from src.data.dataset import build_dataset_splits, build_transforms
from src.models.reid_model import build_model_from_config
from src.reid.cross_domain import evaluate_cross_domain_retrieval
from src.reid.evaluation import (
    build_retrieval_preview,
    evaluate_market1501,
    extract_features,
    re_rank_distance_matrix,
)


TASK_HEAD_PREFIXES = ("classifier.", "auxiliary_classifiers.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict, dict]:
    resolved = Path(checkpoint_path).resolve()
    checkpoint = torch.load(resolved, map_location=device)
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("Evaluation checkpoint is missing model_state_dict")

    target_state = model.state_dict()
    compatible = {}
    skipped = {}
    for name, value in source_state.items():
        if name not in target_state:
            skipped[name] = "not_in_target_model"
        elif tuple(value.shape) != tuple(target_state[name].shape):
            if not name.startswith(TASK_HEAD_PREFIXES):
                raise ValueError(f"Non-task tensor shape mismatch for {name}")
            skipped[name] = (
                f"task_head_shape_mismatch:{tuple(value.shape)}->{tuple(target_state[name].shape)}"
            )
        else:
            compatible[name] = value

    if not any(name.startswith("backbone.") for name in compatible):
        raise ValueError("Evaluation checkpoint has no compatible backbone tensors")
    incompatible = model.load_state_dict(compatible, strict=False)
    non_task_missing = [
        name for name in incompatible.missing_keys if not name.startswith(TASK_HEAD_PREFIXES)
    ]
    if non_task_missing:
        raise ValueError(f"Evaluation checkpoint is missing non-task tensors: {non_task_missing}")
    return checkpoint, {
        "loaded_parameter_tensors": len(compatible),
        "skipped_task_tensors": skipped,
        "missing_task_tensors": list(incompatible.missing_keys),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_reid.yaml")
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
    ensure_dir(config["artifacts"]["logs_dir"])
    ensure_dir(config["artifacts"]["metrics_dir"])

    log_path = Path(config["artifacts"]["logs_dir"]) / "evaluate.log"
    with tee_output(log_path):
        run_evaluation_command(config, args.checkpoint)


def run_evaluation_command(config: dict, checkpoint_path: str) -> None:
    print(f"Logging console output to {Path(config['artifacts']['logs_dir']) / 'evaluate.log'}")

    device = infer_device(config["device"])
    _, test_transform = build_transforms(
        config["data"]["image_height"],
        config["data"]["image_width"],
        preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
    )

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
    checkpoint, checkpoint_load = load_evaluation_checkpoint(model, checkpoint_path, device)
    resolved_checkpoint = Path(checkpoint_path).resolve()

    flip_test = config["evaluation"].get("flip_test", False)
    query_features, query_pid, query_cam, query_paths = extract_features(model, query_loader, device, flip_test=flip_test)
    gallery_features, gallery_pid, gallery_cam, gallery_paths = extract_features(
        model,
        gallery_loader,
        device,
        flip_test=flip_test,
    )
    base_results = evaluate_cross_domain_retrieval(
        query_features,
        gallery_features,
        query_pid,
        gallery_pid,
        query_cam,
        gallery_cam,
        query_batch_size=int(config["evaluation"].get("query_batch_size", 64)),
    )

    results = {
        "run_slug": config["runtime"]["run_slug"],
        "run_root": config["runtime"]["run_root"],
        "dataset": config["data"]["dataset"]["name"],
        "release_id": config.get("release", {}).get("id"),
        "base_release_id": config.get("release", {}).get("base_release_id"),
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": sha256_file(resolved_checkpoint),
        "loaded_epoch": checkpoint.get("epoch"),
        "checkpoint_load": checkpoint_load,
        "git_commit": get_git_commit(),
        "dataset_dvc_revision": get_dvc_revision(
            config["evaluation"]["dataset_pointer"]
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flip_test": bool(flip_test),
        "use_rerank": bool(config["evaluation"].get("use_rerank", False)),
        "query_count": len(query_dataset),
        "gallery_count": len(gallery_dataset),
        "valid_queries": base_results["valid_queries"],
        "rank1_base": base_results["rank1"],
        "rank5_base": base_results["rank5"],
        "rank10_base": base_results["rank10"],
        "rank20_base": base_results["rank20"],
        "mAP_base": base_results["mAP"],
        "mINP_base": base_results["mINP"],
        "retrieval_preview": build_retrieval_preview(
            query_features=query_features,
            gallery_features=gallery_features,
            query_paths=query_paths,
            query_pids=query_pid,
            query_camids=query_cam,
            gallery_paths=gallery_paths,
            gallery_pids=gallery_pid,
            gallery_camids=gallery_cam,
            dataset_root=config["data"]["location"]["root"],
        ),
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
                "rank1": base_results["rank1"],
                "rank5": base_results["rank5"],
                "rank10": base_results["rank10"],
                "rank20": base_results["rank20"],
                "mAP": base_results["mAP"],
                "mINP": base_results["mINP"],
            }
        )
    print(results)

    output_path = Path(config["artifacts"]["metrics_dir"]) / config["evaluation"].get(
        "report_filename", "evaluation_latest.json"
    )
    save_json(results, output_path)
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")


if __name__ == "__main__":
    main()
