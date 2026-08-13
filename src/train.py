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

from src.common.config import load_runtime_config
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
from src.reid.evaluation import (
    compute_distance_matrix,
    evaluate_market1501,
    extract_features,
    re_rank_distance_matrix,
)
from src.reid.losses import ReIDLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.batch_size=32",
    )
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
        color_jitter=config["augmentation"].get("color_jitter", False),
        random_erasing=config["augmentation"].get("random_erasing", False),
        random_grayscale_p=config["augmentation"].get("random_grayscale_p", 0.0),
        random_affine_degrees=config["augmentation"].get("random_affine_degrees", 0.0),
        random_occlusion_p=config["augmentation"].get("random_occlusion_p", 0.0),
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, grad_clip_norm: float | None = None):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_triplet = 0.0
    running_center = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=True, dynamic_ncols=True, disable=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["pid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            logits, embeddings = model(images)
            loss, ce_loss, triplet_loss, center_loss = criterion(logits, embeddings, labels)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        running_ce += float(ce_loss.item())
        running_triplet += float(triplet_loss.item())
        running_center += float(center_loss.item())
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
        "train_center_loss": running_center / len(loader),
        "train_accuracy": correct / max(1, total),
    }


def run_evaluation(model, query_loader, gallery_loader, device, config):
    flip_test = config["evaluation"].get("flip_test", False)
    query_features, query_pid, query_cam, _ = extract_features(model, query_loader, device, flip_test=flip_test)
    gallery_features, gallery_pid, gallery_cam, _ = extract_features(
        model,
        gallery_loader,
        device,
        flip_test=flip_test,
    )
    base_distance_matrix = compute_distance_matrix(query_features, gallery_features)
    cmc, mean_ap, mean_inp, valid_queries = evaluate_market1501(
        base_distance_matrix,
        query_pid,
        gallery_pid,
        query_cam,
        gallery_cam,
    )
    results = {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "rank20": float(cmc[19]),
        "mAP": mean_ap,
        "mINP": mean_inp,
        "valid_queries": int(valid_queries),
        "rank1_base": float(cmc[0]),
        "rank5_base": float(cmc[4]),
        "rank10_base": float(cmc[9]),
        "rank20_base": float(cmc[19]),
        "mAP_base": mean_ap,
        "mINP_base": mean_inp,
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
                "mAP_rerank": rerank_mean_ap,
                "mINP_rerank": rerank_mean_inp,
            }
        )
        results["rank1"] = results["rank1_rerank"]
        results["rank5"] = results["rank5_rerank"]
        results["rank10"] = results["rank10_rerank"]
        results["rank20"] = results["rank20_rerank"]
        results["mAP"] = results["mAP_rerank"]
        results["mINP"] = results["mINP_rerank"]

    return results


def build_optimizer(config: dict, model: torch.nn.Module, criterion: ReIDLoss) -> torch.optim.Optimizer:
    base_lr = config["train"]["learning_rate"]
    backbone_lr_factor = config["train"].get("backbone_lr_factor", 0.1)
    weight_decay = config["train"]["weight_decay"]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    parameter_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_lr_factor},
        {"params": head_params, "lr": base_lr},
    ]
    if criterion.center is not None:
        parameter_groups.append(
            {
                "params": criterion.center.parameters(),
                "lr": config["train"].get("center_loss_lr", 0.25),
                "weight_decay": 0.0,
            }
        )

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    scheduler_type = config["train"].get("scheduler_type", "cosine").lower()
    if scheduler_type == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=config["train"].get("lr_reduce_factor", 0.5),
                patience=config["train"].get("lr_reduce_patience", 5),
                threshold=config["train"].get("lr_reduce_threshold", 1e-3),
                min_lr=config["train"].get("min_lr", 1e-6),
            ),
            "metric",
        )

    warmup_epochs = config["train"].get("warmup_epochs", 0)

    def lr_lambda(epoch: int) -> float:
        total_epochs = max(1, config["train"]["epochs"])
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        min_lr_scale = config["train"].get("min_lr_scale", 0.01)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "epoch"


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config, args.set)
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
        num_classes=train_dataset.num_classes,
        embedding_dim=config["model"]["embedding_dim"],
        ce_weight=config["train"]["ce_weight"],
        triplet_weight=config["train"]["triplet_weight"],
        triplet_margin=config["train"]["triplet_margin"],
        label_smoothing=config["train"]["label_smoothing"],
        center_loss_weight=config["train"].get("center_loss_weight", 0.0),
    )
    optimizer = build_optimizer(config, model, criterion)
    scheduler, scheduler_step_mode = build_scheduler(config, optimizer)
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
                "backbone_lr_factor": config["train"].get("backbone_lr_factor", 0.1),
                "center_loss_weight": config["train"].get("center_loss_weight", 0.0),
                "center_loss_lr": config["train"].get("center_loss_lr", 0.25),
                "scheduler_type": config["train"].get("scheduler_type", "cosine"),
                "lr_reduce_factor": config["train"].get("lr_reduce_factor", 0.5),
                "lr_reduce_patience": config["train"].get("lr_reduce_patience", 5),
                "min_lr": config["train"].get("min_lr", 1e-6),
                "warmup_epochs": config["train"].get("warmup_epochs", 0),
                "grad_clip_norm": config["train"].get("grad_clip_norm", 0.0),
                "embedding_dim": config["model"]["embedding_dim"],
                "pretrained": config["model"]["pretrained"],
                "model_variant": config["model"].get("variant", "baseline"),
                "flip_test": config["evaluation"].get("flip_test", False),
                "use_rerank": config["evaluation"].get("use_rerank", False),
                "device": str(device),
            }
        )

    best_map = float("-inf")
    early_stopping = config["train"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    early_stopping_patience = int(early_stopping.get("patience", 10))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    early_stopping_monitor = early_stopping.get("monitor", "mAP")
    best_monitored_metric = float("-inf")
    bad_epochs = 0
    history = []

    try:
        for epoch in range(1, config["train"]["epochs"] + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                grad_clip_norm=config["train"].get("grad_clip_norm"),
            )
            eval_metrics = run_evaluation(model, query_loader, gallery_loader, device, config)

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
                f"| rank5={epoch_metrics['rank5'] * 100:.2f}% "
                f"| rank10={epoch_metrics['rank10'] * 100:.2f}% "
                f"| rank20={epoch_metrics['rank20'] * 100:.2f}% "
                f"| mAP={epoch_metrics['mAP'] * 100:.2f}% "
                f"| mINP={epoch_metrics['mINP'] * 100:.2f}%"
            )
            if config["evaluation"].get("use_rerank", False):
                print(
                    f"  Base metrics    "
                    f"| rank1={epoch_metrics['rank1_base'] * 100:.2f}% "
                    f"| rank5={epoch_metrics['rank5_base'] * 100:.2f}% "
                    f"| rank10={epoch_metrics['rank10_base'] * 100:.2f}% "
                    f"| rank20={epoch_metrics['rank20_base'] * 100:.2f}% "
                    f"| mAP={epoch_metrics['mAP_base'] * 100:.2f}% "
                    f"| mINP={epoch_metrics['mINP_base'] * 100:.2f}%"
                )
                print(
                    f"  Rerank metrics  "
                    f"| rank1={epoch_metrics['rank1_rerank'] * 100:.2f}% "
                    f"| rank5={epoch_metrics['rank5_rerank'] * 100:.2f}% "
                    f"| rank10={epoch_metrics['rank10_rerank'] * 100:.2f}% "
                    f"| rank20={epoch_metrics['rank20_rerank'] * 100:.2f}% "
                    f"| mAP={epoch_metrics['mAP_rerank'] * 100:.2f}% "
                    f"| mINP={epoch_metrics['mINP_rerank'] * 100:.2f}%"
                )

            if mlflow_active:
                mlflow.log_metrics(epoch_metrics, step=epoch)

            last_checkpoint = Path(config["artifacts"]["checkpoints_dir"]) / "last_model.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, last_checkpoint)

            current_monitored_metric = float(epoch_metrics[early_stopping_monitor])
            if scheduler_step_mode == "metric":
                scheduler.step(current_monitored_metric)
            else:
                scheduler.step()

            if epoch_metrics["mAP"] > best_map:
                best_map = epoch_metrics["mAP"]
                best_checkpoint = Path(config["artifacts"]["checkpoints_dir"]) / "best_model.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, best_checkpoint)

            if current_monitored_metric > (best_monitored_metric + early_stopping_min_delta):
                best_monitored_metric = current_monitored_metric
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stopping_enabled and bad_epochs >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"after {bad_epochs} epochs without improvement in {early_stopping_monitor}."
                )
                break

        summary = {
            "best_epoch": max(history, key=lambda item: item["mAP"])["epoch"],
            "best_rank1": max(history, key=lambda item: item["mAP"])["rank1"],
            "best_rank5": max(history, key=lambda item: item["mAP"])["rank5"],
            "best_rank10": max(history, key=lambda item: item["mAP"])["rank10"],
            "best_rank20": max(history, key=lambda item: item["mAP"])["rank20"],
            "best_mAP": max(history, key=lambda item: item["mAP"])["mAP"],
            "best_mINP": max(history, key=lambda item: item["mAP"])["mINP"],
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
