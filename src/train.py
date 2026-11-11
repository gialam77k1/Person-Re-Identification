from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
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
from src.common.mlflow_utils import (
    collect_hardware_info,
    configure_mlflow,
    get_dvc_revision,
    get_git_commit,
    lineage_tags,
)
from src.common.utils import (
    configure_torch_home,
    ensure_dir,
    infer_device,
    save_checkpoint,
    save_json,
    seed_everything,
    tee_output,
)
from src.data.dataset import RandomIdentitySampler, build_dataset_splits, build_transforms
from src.models.reid_model import build_model_from_config
from src.models.transfer import load_transfer_checkpoint, set_vit_lower_blocks_trainable
from src.reid.cross_domain import evaluate_cross_domain_retrieval
from src.reid.evaluation import (
    evaluate_market1501,
    extract_features,
    re_rank_distance_matrix,
)
from src.reid.losses import ReIDLoss


def report_control_progress(
    stage: str,
    status: str,
    *,
    progress: float | None = None,
    detail: str = "",
    metrics: dict | None = None,
) -> None:
    cycle_id = os.getenv("CONTROL_CYCLE_ID", "")
    token = os.getenv("CONTROL_CALLBACK_TOKEN", "")
    if not cycle_id or cycle_id == "manual" or not token:
        return
    base_url = os.getenv("CONTROL_API_URL", "http://localhost:8090").rstrip("/")
    payload = json.dumps(
        {
            "status": status,
            "progress": progress,
            "detail": detail or None,
            "metrics": metrics or {},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/control/cycles/{cycle_id}/stages/{stage}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "X-Control-Callback-Token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARNING: Control Center progress callback failed: {exc}", file=sys.stderr)


def collect_gpu_telemetry(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    telemetry = {
        "gpu_memory_allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 2),
        "gpu_memory_reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 2),
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device.index or 0}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        telemetry["gpu_utilization_percent"] = float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        # CUDA memory remains useful telemetry when nvidia-smi is unavailable.
        pass
    return telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_reid.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.batch_size=32",
    )
    return parser.parse_args()


def maybe_init_mlflow(config: dict) -> tuple[bool, dict]:
    if not config["logging"].get("enable_mlflow", False):
        return False, {}

    tracking_uri = configure_mlflow(config["experiment_name"], config)
    git_commit = get_git_commit()
    dvc_revision = get_dvc_revision()
    run = mlflow.start_run(run_name=config["runtime"]["run_slug"])
    mlflow.set_tags(
        {
            **lineage_tags(git_commit=git_commit, dvc_revision=dvc_revision),
            "run.kind": "training",
            "dataset": config["data"]["dataset"]["name"],
        }
    )
    metadata = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "tracking_uri": tracking_uri,
        "git_commit": git_commit,
        "dvc_dataset_revision": dvc_revision,
    }
    return True, metadata


def close_mlflow(active: bool, status: str = "FINISHED") -> None:
    if active and mlflow.active_run() is not None:
        mlflow.end_run(status=status)


def build_loaders(config: dict):
    train_transform, test_transform = build_transforms(
        config["data"]["image_height"],
        config["data"]["image_width"],
        color_jitter=config["augmentation"].get("color_jitter", False),
        random_erasing=config["augmentation"].get("random_erasing", False),
        random_grayscale_p=config["augmentation"].get("random_grayscale_p", 0.0),
        random_affine_degrees=config["augmentation"].get("random_affine_degrees", 0.0),
        random_occlusion_p=config["augmentation"].get("random_occlusion_p", 0.0),
        preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
    )

    train_dataset, query_dataset, gallery_dataset = build_dataset_splits(
        config,
        train_transform=train_transform,
        test_transform=test_transform,
    )

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


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("Model does not expose a 'backbone' module")

    for parameter in backbone.parameters():
        parameter.requires_grad = trainable


def validate_training_dataset(config: dict, train_dataset) -> None:
    expected_images = config["data"].get("expected_train_images")
    expected_identities = config["data"].get("expected_train_identities")
    if expected_images is not None and len(train_dataset) != int(expected_images):
        raise ValueError(
            f"Expected {expected_images} training images, received {len(train_dataset)}"
        )
    if expected_identities is not None and train_dataset.num_classes != int(expected_identities):
        raise ValueError(
            f"Expected {expected_identities} training identities, "
            f"received {train_dataset.num_classes}"
        )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    grad_clip_norm: float | None = None,
    backbone_frozen: bool = False,
):
    model.train()
    if backbone_frozen:
        # Keep dropout and other training-time behavior disabled while the
        # pretrained backbone is frozen. Newly added ReID heads stay in train mode.
        model.backbone.eval()
    running_loss = 0.0
    running_ce = 0.0
    running_triplet = 0.0
    running_center = 0.0
    running_auxiliary = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=True, dynamic_ncols=True, disable=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["pid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            if hasattr(model, "forward_training"):
                outputs = model.forward_training(images)
                logits = outputs["logits"]
                embeddings = outputs["embedding"]
                metric_embeddings = outputs.get("metric_embedding")
                auxiliary_logits = outputs.get("auxiliary_logits")
                auxiliary_embeddings = outputs.get("auxiliary_embeddings")
            else:
                logits, embeddings = model(images)
                metric_embeddings = None
                auxiliary_logits = None
                auxiliary_embeddings = None

            loss, ce_loss, triplet_loss, center_loss, auxiliary_loss = criterion(
                logits,
                embeddings,
                labels,
                metric_embeddings=metric_embeddings,
                auxiliary_logits=auxiliary_logits,
                auxiliary_embeddings=auxiliary_embeddings,
            )

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
        running_auxiliary += float(auxiliary_loss.item())
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
        "train_auxiliary_loss": running_auxiliary / len(loader),
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
        "rank1": base_results["rank1"],
        "rank5": base_results["rank5"],
        "rank10": base_results["rank10"],
        "rank20": base_results["rank20"],
        "mAP": base_results["mAP"],
        "mINP": base_results["mINP"],
        "valid_queries": base_results["valid_queries"],
        "rank1_base": base_results["rank1"],
        "rank5_base": base_results["rank5"],
        "rank10_base": base_results["rank10"],
        "rank20_base": base_results["rank20"],
        "mAP_base": base_results["mAP"],
        "mINP_base": base_results["mINP"],
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
    config = load_runtime_config(args.config, args.set, command_name="train")
    seed_everything(config["seed"])
    configure_torch_home()

    for key in ("root", "checkpoints_dir", "metrics_dir", "embeddings_dir", "logs_dir"):
        if key in config["artifacts"]:
            ensure_dir(config["artifacts"][key])

    log_path = Path(config["artifacts"]["logs_dir"]) / "train.log"
    with tee_output(log_path):
        run_training(config)


def run_training(config: dict) -> None:
    print(f"Logging console output to {Path(config['artifacts']['logs_dir']) / 'train.log'}")
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()

    device = infer_device(config["device"])
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")

    if device.type != "cuda" and config["train"]["amp"]:
        warnings.warn("AMP was requested but CUDA is not available. Training will run in FP32.")

    train_dataset, train_loader, query_loader, gallery_loader = build_loaders(config)
    validate_training_dataset(config, train_dataset)
    model = build_model_from_config(
        config,
        num_classes=train_dataset.num_classes,
        pretrained=config["model"]["pretrained"],
    ).to(device)

    transfer_config = config.get("transfer", {})
    transfer_report: dict = {"enabled": False}
    if transfer_config.get("enabled", False):
        transfer_report = {
            "enabled": True,
            **load_transfer_checkpoint(
                model,
                transfer_config["base_checkpoint"],
                device,
                reset_classifier_heads=bool(
                    transfer_config.get("reset_classifier_heads", True)
                ),
                expected_epoch=transfer_config.get("expected_checkpoint_epoch"),
            ),
        }
        print(
            "Loaded transfer checkpoint "
            f"epoch {transfer_report['source_epoch']} with "
            f"{transfer_report['loaded_backbone_tensors']} backbone tensors; "
            "DukeMTMC-reID classifier heads remain newly initialized."
        )

    transfer_report_path = save_json(
        transfer_report,
        Path(config["artifacts"]["logs_dir"]) / "transfer_report.json",
    )

    criterion = ReIDLoss(
        num_classes=train_dataset.num_classes,
        embedding_dim=config["model"]["embedding_dim"],
        ce_weight=config["train"]["ce_weight"],
        triplet_weight=config["train"]["triplet_weight"],
        triplet_margin=config["train"]["triplet_margin"],
        label_smoothing=config["train"]["label_smoothing"],
        center_loss_weight=config["train"].get("center_loss_weight", 0.0),
        auxiliary_loss_weight=config["train"].get("auxiliary_loss_weight", 0.0),
        normalize_triplet_embeddings=config["train"].get("normalize_triplet_embeddings", True),
    )
    optimizer = build_optimizer(config, model, criterion)
    scheduler, scheduler_step_mode = build_scheduler(config, optimizer)
    scaler = amp.GradScaler(device.type, enabled=use_amp)

    freeze_backbone_epochs = max(0, int(config["train"].get("freeze_backbone_epochs", 0)))
    progressive_config = config["train"].get("progressive_unfreeze", {})
    progressive_enabled = bool(progressive_config.get("enabled", False))
    if freeze_backbone_epochs > 0 and progressive_enabled:
        raise ValueError(
            "train.freeze_backbone_epochs and train.progressive_unfreeze cannot both be enabled"
        )

    frozen_lower_blocks = 0
    full_unfreeze_epoch = 1
    if progressive_enabled:
        frozen_lower_blocks = int(progressive_config.get("freeze_lower_blocks", 0))
        full_unfreeze_epoch = int(progressive_config.get("full_unfreeze_epoch", 1))
        if full_unfreeze_epoch < 1 or full_unfreeze_epoch > int(config["train"]["epochs"]):
            raise ValueError("progressive_unfreeze.full_unfreeze_epoch is outside the training range")
        phase = set_vit_lower_blocks_trainable(
            model,
            frozen_lower_blocks=frozen_lower_blocks,
        )
        print(
            "Progressive unfreeze enabled: "
            f"{phase['frozen_lower_blocks']}/{phase['total_blocks']} lower ViT blocks frozen "
            f"until epoch {full_unfreeze_epoch - 1}; "
            f"{phase['trainable_upper_blocks']} upper blocks and ReID heads train immediately."
        )
    if freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print(
            f"Backbone frozen for the first {freeze_backbone_epochs} epoch(s); "
            "only ReID heads will be updated."
        )

    mlflow_active, mlflow_metadata = maybe_init_mlflow(config)
    effective_config_path = Path(config["artifacts"]["logs_dir"]) / "effective_config.json"
    save_json(config, effective_config_path)
    if mlflow_active:
        mlflow_metadata.update(
            {
                "started_at": started_at.isoformat(),
                "hardware": collect_hardware_info(device),
            }
        )
        mlflow_run_path = save_json(
            mlflow_metadata,
            Path(config["artifacts"]["logs_dir"]) / "mlflow_run.json",
        )
        mlflow.log_artifact(config["config_path"], artifact_path="config")
        mlflow.log_artifact(str(effective_config_path), artifact_path="config")
        mlflow.log_artifact(str(mlflow_run_path), artifact_path="provenance")
        mlflow.log_artifact(str(transfer_report_path), artifact_path="provenance")
        mlflow.log_params(
            {
                "dataset": config["data"]["dataset"]["name"],
                "run_slug": config["runtime"]["run_slug"],
                "batch_size": config["data"]["batch_size"],
                "eval_batch_size": config["data"]["eval_batch_size"],
                "instances_per_identity": config["data"]["instances_per_identity"],
                "epochs": config["train"]["epochs"],
                "learning_rate": config["train"]["learning_rate"],
                "weight_decay": config["train"]["weight_decay"],
                "backbone_lr_factor": config["train"].get("backbone_lr_factor", 0.1),
                "center_loss_weight": config["train"].get("center_loss_weight", 0.0),
                "center_loss_lr": config["train"].get("center_loss_lr", 0.25),
                "auxiliary_loss_weight": config["train"].get("auxiliary_loss_weight", 0.0),
                "normalize_triplet_embeddings": config["train"].get(
                    "normalize_triplet_embeddings", True
                ),
                "scheduler_type": config["train"].get("scheduler_type", "cosine"),
                "lr_reduce_factor": config["train"].get("lr_reduce_factor", 0.5),
                "lr_reduce_patience": config["train"].get("lr_reduce_patience", 5),
                "min_lr": config["train"].get("min_lr", 1e-6),
                "warmup_epochs": config["train"].get("warmup_epochs", 0),
                "early_stopping_enabled": config["train"].get("early_stopping", {}).get(
                    "enabled", False
                ),
                "early_stopping_monitor": config["train"].get("early_stopping", {}).get(
                    "monitor", "mAP"
                ),
                "early_stopping_patience": config["train"].get("early_stopping", {}).get(
                    "patience", 10
                ),
                "early_stopping_min_delta": config["train"].get("early_stopping", {}).get(
                    "min_delta", 0.0
                ),
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "progressive_unfreeze": progressive_enabled,
                "frozen_lower_blocks": frozen_lower_blocks,
                "full_unfreeze_epoch": full_unfreeze_epoch,
                "grad_clip_norm": config["train"].get("grad_clip_norm", 0.0),
                "embedding_dim": config["model"]["embedding_dim"],
                "pretrained": config["model"]["pretrained"],
                "model_variant": config["model"].get("variant", "vit"),
                "flip_test": config["evaluation"].get("flip_test", False),
                "use_rerank": config["evaluation"].get("use_rerank", False),
                "device": str(device),
                "release_id": config.get("release", {}).get("id", ""),
                "base_release_id": config.get("release", {}).get("base_release_id", ""),
                "transfer_checkpoint": transfer_report.get("checkpoint", ""),
                "transfer_checkpoint_epoch": transfer_report.get("source_epoch", ""),
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
    run_status = "FINISHED"
    stopped_early = False
    stop_reason = "max_epochs_completed"
    completed_epochs = 0

    try:
        for epoch in range(1, config["train"]["epochs"] + 1):
            backbone_frozen = epoch <= freeze_backbone_epochs
            if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
                set_backbone_trainable(model, True)
                print(
                    f"Backbone unfrozen at epoch {epoch}; "
                    "full-model fine-tuning is now enabled."
                )
            if progressive_enabled and epoch == full_unfreeze_epoch:
                set_backbone_trainable(model, True)
                print(
                    f"All ViT blocks unfrozen at epoch {epoch}; "
                    "full-model fine-tuning is now enabled with the backbone LR group."
                )

            lower_blocks_frozen = progressive_enabled and epoch < full_unfreeze_epoch

            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                use_amp,
                grad_clip_norm=config["train"].get("grad_clip_norm"),
                backbone_frozen=backbone_frozen,
            )
            eval_metrics = run_evaluation(model, query_loader, gallery_loader, device, config)

            epoch_metrics = {
                "epoch": epoch,
                "backbone_frozen": int(backbone_frozen),
                "lower_blocks_frozen": int(lower_blocks_frozen),
                "full_model_finetune": int(not backbone_frozen and not lower_blocks_frozen),
                **train_metrics,
                **eval_metrics,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
            }
            history.append(epoch_metrics)
            completed_epochs = epoch
            latest_log = (
                f"Epoch {epoch}/{config['train']['epochs']} | "
                f"loss={epoch_metrics['train_loss']:.4f} | "
                f"mAP={epoch_metrics['mAP']:.4f} | "
                f"rank1={epoch_metrics['rank1']:.4f}"
            )
            report_control_progress(
                "train_v2",
                "RUNNING",
                progress=epoch / config["train"]["epochs"],
                detail=f"Epoch {epoch}/{config['train']['epochs']}",
                metrics={
                    "epoch": epoch,
                    "epochs": config["train"]["epochs"],
                    "train_loss": epoch_metrics["train_loss"],
                    "rank1": epoch_metrics["rank1"],
                    "mAP": epoch_metrics["mAP"],
                    "mINP": epoch_metrics["mINP"],
                    "backbone_lr": epoch_metrics["lr"],
                    "head_lr": epoch_metrics["head_lr"],
                    "elapsed_seconds": round(time.perf_counter() - started_monotonic, 1),
                    "latest_log": latest_log,
                    "mlflow_run_id": mlflow_metadata.get("run_id", ""),
                    "mlflow_run_url": (
                        f"{str(mlflow_metadata.get('tracking_uri', '')).rstrip('/')}"
                        f"/#/experiments/{mlflow_metadata.get('experiment_id', '')}"
                        f"/runs/{mlflow_metadata.get('run_id', '')}"
                        if mlflow_metadata.get("run_id")
                        else ""
                    ),
                    **collect_gpu_telemetry(device),
                },
            )

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
                stopped_early = True
                stop_reason = (
                    f"early_stopping:{early_stopping_monitor}:"
                    f"no_improvement_{bad_epochs}_epochs"
                )
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"after {bad_epochs} epochs without improvement in {early_stopping_monitor}."
                )
                break

        summary = {
            "run_slug": config["runtime"]["run_slug"],
            "run_root": config["runtime"]["run_root"],
            "dataset": config["data"]["dataset"]["name"],
            "release_id": config.get("release", {}).get("id"),
            "base_release_id": config.get("release", {}).get("base_release_id"),
            "transfer": transfer_report,
            "configured_epochs": int(config["train"]["epochs"]),
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "early_stopping": {
                "enabled": early_stopping_enabled,
                "monitor": early_stopping_monitor,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "bad_epochs_at_stop": bad_epochs,
                "best_monitored_value": best_monitored_metric,
            },
            "best_epoch": max(history, key=lambda item: item["mAP"])["epoch"],
            "best_rank1": max(history, key=lambda item: item["mAP"])["rank1"],
            "best_rank5": max(history, key=lambda item: item["mAP"])["rank5"],
            "best_rank10": max(history, key=lambda item: item["mAP"])["rank10"],
            "best_rank20": max(history, key=lambda item: item["mAP"])["rank20"],
            "best_mAP": max(history, key=lambda item: item["mAP"])["mAP"],
            "best_mINP": max(history, key=lambda item: item["mAP"])["mINP"],
            "history": history,
        }
        metrics_filename = config["artifacts"].get("metrics_filename", "metrics_v1.json")
        metrics_path = Path(config["artifacts"]["metrics_dir"]) / metrics_filename
        saved_metrics = save_json(summary, metrics_path)
        if mlflow_active:
            mlflow.log_metrics(
                {
                    "best_rank1": summary["best_rank1"],
                    "best_rank5": summary["best_rank5"],
                    "best_mAP": summary["best_mAP"],
                    "best_mINP": summary["best_mINP"],
                    "best_epoch": summary["best_epoch"],
                    "completed_epochs": summary["completed_epochs"],
                    "stopped_early": int(summary["stopped_early"]),
                }
            )
            mlflow.set_tags(
                {
                    "training.stop_reason": summary["stop_reason"],
                    "training.stopped_early": str(summary["stopped_early"]).lower(),
                }
            )
            mlflow.log_artifact(str(saved_metrics), artifact_path="metrics")
            mlflow.log_artifact(
                str(Path(config["artifacts"]["checkpoints_dir"]) / "best_model.pth"),
                artifact_path="checkpoints",
            )
            mlflow.log_artifact(
                str(Path(config["artifacts"]["checkpoints_dir"]) / "last_model.pth"),
                artifact_path="checkpoints",
            )
    except Exception:
        run_status = "FAILED"
        raise
    finally:
        if mlflow_active:
            duration_seconds = time.perf_counter() - started_monotonic
            mlflow.log_metric("train_duration_seconds", duration_seconds)
            mlflow_metadata.update(
                {
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": duration_seconds,
                    "status": run_status,
                    "configured_epochs": int(config["train"]["epochs"]),
                    "completed_epochs": completed_epochs,
                    "stopped_early": stopped_early,
                    "stop_reason": stop_reason if run_status == "FINISHED" else "training_failed",
                }
            )
            mlflow_run_path = save_json(
                mlflow_metadata,
                Path(config["artifacts"]["logs_dir"]) / "mlflow_run.json",
            )
            mlflow.log_artifact(str(mlflow_run_path), artifact_path="provenance")
        close_mlflow(mlflow_active, status=run_status)


if __name__ == "__main__":
    main()
