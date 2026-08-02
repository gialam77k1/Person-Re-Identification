from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mlflow
import torch
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.utils import (
    configure_torch_home,
    ensure_dir,
    infer_device,
    save_checkpoint,
    save_json,
    seed_everything,
)
from src.data.dataset import Market1501Dataset, RandomIdentitySampler, build_transforms
from src.models.reid_model import build_model_from_config
from src.reid.evaluation import compute_distance_matrix, evaluate_market1501, extract_features
from src.reid.losses import ReIDLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def maybe_init_mlflow(config: dict) -> bool:
    if not config["logging"].get("enable_mlflow", False):
        return False

    tracking_uri = config["logging"].get("tracking_uri")
    artifact_location = ensure_dir(config["logging"].get("artifact_location", "artifacts/mlflow"))
    ensure_dir("mlruns")
    if not tracking_uri:
        tracking_uri = "sqlite:///mlruns/mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if experiment is None:
        mlflow.create_experiment(
            name=config["experiment_name"],
            artifact_location=artifact_location.as_uri(),
        )
    mlflow.set_experiment(config["experiment_name"])
    mlflow.start_run(run_name="baseline-v1")
    mlflow.log_artifact(config["config_path"])
    return True


def close_mlflow(active: bool) -> None:
    if active and mlflow.active_run() is not None:
        mlflow.end_run()


def build_loaders(config: dict):
    train_transform, test_transform = build_transforms(
        config["data"]["image_height"],
        config["data"]["image_width"],
    )

    train_dataset = Market1501Dataset(config["data"]["train_dir"], transform=train_transform, relabel=True)
    query_dataset = Market1501Dataset(config["data"]["query_dir"], transform=test_transform, relabel=False)
    gallery_dataset = Market1501Dataset(config["data"]["gallery_dir"], transform=test_transform, relabel=False)

    sampler = RandomIdentitySampler(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        instances_per_identity=config["data"]["instances_per_identity"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
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
    return train_dataset, train_loader, query_loader, gallery_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_triplet = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["pid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            logits, embeddings = model(images)
            loss, ce_loss, triplet_loss = criterion(logits, embeddings, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        running_ce += float(ce_loss.item())
        running_triplet += float(triplet_loss.item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss / max(1, progress.n):.4f}",
            acc=f"{100.0 * correct / max(1, total):.2f}%",
        )

    return {
        "train_loss": running_loss / len(loader),
        "train_ce_loss": running_ce / len(loader),
        "train_triplet_loss": running_triplet / len(loader),
        "train_accuracy": correct / max(1, total),
    }


def run_evaluation(model, query_loader, gallery_loader, device):
    query_features, query_pid, query_cam, _ = extract_features(model, query_loader, device)
    gallery_features, gallery_pid, gallery_cam, _ = extract_features(model, gallery_loader, device)
    distance_matrix = compute_distance_matrix(query_features, gallery_features)
    cmc, mean_ap = evaluate_market1501(distance_matrix, query_pid, gallery_pid, query_cam, gallery_cam)
    return {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": mean_ap,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    configure_torch_home()

    for key in ("root", "checkpoints_dir", "metrics_dir", "embeddings_dir", "logs_dir"):
        if key in config["artifacts"]:
            ensure_dir(config["artifacts"][key])

    device = infer_device(config["device"])
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")

    if device.type != "cuda" and config["train"]["amp"]:
        warnings.warn("AMP was requested but CUDA is not available. Training will run in FP32.")

    train_dataset, train_loader, query_loader, gallery_loader = build_loaders(config)
    model = build_model_from_config(
        config,
        num_classes=train_dataset.num_classes,
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = ReIDLoss(
        ce_weight=config["train"]["ce_weight"],
        triplet_weight=config["train"]["triplet_weight"],
        triplet_margin=config["train"]["triplet_margin"],
        label_smoothing=config["train"]["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    scaler = amp.GradScaler(device.type, enabled=use_amp)

    mlflow_active = maybe_init_mlflow(config)
    if mlflow_active:
        mlflow.log_params(
            {
                "dataset": "Market-1501",
                "batch_size": config["data"]["batch_size"],
                "eval_batch_size": config["data"]["eval_batch_size"],
                "instances_per_identity": config["data"]["instances_per_identity"],
                "epochs": config["train"]["epochs"],
                "learning_rate": config["train"]["learning_rate"],
                "weight_decay": config["train"]["weight_decay"],
                "embedding_dim": config["model"]["embedding_dim"],
                "pretrained": config["model"]["pretrained"],
                "model_variant": config["model"].get("variant", "baseline"),
                "device": str(device),
            }
        )

    best_map = float("-inf")
    history = []

    try:
        for epoch in range(1, config["train"]["epochs"] + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            eval_metrics = run_evaluation(model, query_loader, gallery_loader, device)
            scheduler.step()

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **eval_metrics,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_metrics)

            print(
                f"Epoch {epoch}/{config['train']['epochs']} "
                f"| loss={epoch_metrics['train_loss']:.4f} "
                f"| rank1={epoch_metrics['rank1'] * 100:.2f}% "
                f"| mAP={epoch_metrics['mAP'] * 100:.2f}%"
            )

            if mlflow_active:
                mlflow.log_metrics(epoch_metrics, step=epoch)

            last_checkpoint = Path(config["artifacts"]["checkpoints_dir"]) / "last_model.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, last_checkpoint)

            if epoch_metrics["mAP"] > best_map:
                best_map = epoch_metrics["mAP"]
                best_checkpoint = Path(config["artifacts"]["checkpoints_dir"]) / "best_model.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, best_checkpoint)

        summary = {
            "best_epoch": max(history, key=lambda item: item["mAP"])["epoch"],
            "best_rank1": max(history, key=lambda item: item["mAP"])["rank1"],
            "best_rank5": max(history, key=lambda item: item["mAP"])["rank5"],
            "best_rank10": max(history, key=lambda item: item["mAP"])["rank10"],
            "best_mAP": max(history, key=lambda item: item["mAP"])["mAP"],
            "history": history,
        }
        metrics_path = Path(config["artifacts"]["metrics_dir"]) / "metrics_v1.json"
        saved_metrics = save_json(summary, metrics_path)
        if mlflow_active:
            mlflow.log_artifact(str(saved_metrics))
            mlflow.log_artifact(str(Path(config["artifacts"]["checkpoints_dir"]) / "best_model.pth"))
    finally:
        close_mlflow(mlflow_active)


if __name__ == "__main__":
    main()
