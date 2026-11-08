from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


EXPECTED_TRAIN_IMAGES = 30248
EXPECTED_TRAIN_IDENTITIES = 1041
MINIMUM_FREE_BYTES = 12 * 1024**3
EXPECTED_RELEASE_ID = "market1501-msmt17-vit-bnneck:v2"
EXPECTED_BASE_RELEASE_ID = "market1501-vit-bnneck:v1-dvc"
EXPECTED_DATASET = "msmt17"
EXPECTED_DATASET_VERSION = "MSMT17_V1"
EXPECTED_CHECKPOINT_EPOCH = 48
EXPECTED_EPOCHS = 30
EXPECTED_TRAIN_BATCH_SIZE = 16
EXPECTED_EVAL_BATCH_SIZE = 16
EXPECTED_QUERY_BATCH_SIZE = 32


def validate_training_contract(config: dict, decision: dict) -> None:
    release = config.get("release", {})
    data = config.get("data", {})
    dataset = data.get("dataset", {})
    transfer = config.get("transfer", {})
    train = config.get("train", {})
    progressive = train.get("progressive_unfreeze", {})
    early_stopping = train.get("early_stopping", {})

    expected_values = {
        "release.id": (release.get("id"), EXPECTED_RELEASE_ID),
        "release.base_release_id": (
            release.get("base_release_id"),
            EXPECTED_BASE_RELEASE_ID,
        ),
        "data.dataset.name": (str(dataset.get("name", "")).lower(), EXPECTED_DATASET),
        "data.dataset.version": (dataset.get("version"), EXPECTED_DATASET_VERSION),
        "data.expected_train_images": (
            data.get("expected_train_images"),
            EXPECTED_TRAIN_IMAGES,
        ),
        "data.expected_train_identities": (
            data.get("expected_train_identities"),
            EXPECTED_TRAIN_IDENTITIES,
        ),
        "data.batch_size": (data.get("batch_size"), EXPECTED_TRAIN_BATCH_SIZE),
        "data.eval_batch_size": (
            data.get("eval_batch_size"),
            EXPECTED_EVAL_BATCH_SIZE,
        ),
        "transfer.expected_checkpoint_epoch": (
            transfer.get("expected_checkpoint_epoch"),
            EXPECTED_CHECKPOINT_EPOCH,
        ),
        "train.epochs": (train.get("epochs"), EXPECTED_EPOCHS),
        "train.progressive_unfreeze.freeze_lower_blocks": (
            progressive.get("freeze_lower_blocks"),
            8,
        ),
        "train.progressive_unfreeze.full_unfreeze_epoch": (
            progressive.get("full_unfreeze_epoch"),
            6,
        ),
        "train.warmup_epochs": (train.get("warmup_epochs"), 3),
        "train.scheduler_type": (str(train.get("scheduler_type", "")).lower(), "cosine"),
        "train.early_stopping.monitor": (
            early_stopping.get("monitor"),
            "mAP",
        ),
        "train.early_stopping.patience": (
            early_stopping.get("patience"),
            5,
        ),
        "evaluation.query_batch_size": (
            config.get("evaluation", {}).get("query_batch_size"),
            EXPECTED_QUERY_BATCH_SIZE,
        ),
    }
    for field, (actual, expected) in expected_values.items():
        if actual != expected:
            raise ValueError(
                f"Training contract mismatch for {field}: expected {expected!r}, received {actual!r}"
            )

    required_flags = {
        "transfer.enabled": transfer.get("enabled"),
        "transfer.reset_classifier_heads": transfer.get("reset_classifier_heads"),
        "train.progressive_unfreeze.enabled": progressive.get("enabled"),
        "train.early_stopping.enabled": early_stopping.get("enabled"),
    }
    for field, enabled in required_flags.items():
        if enabled is not True:
            raise ValueError(f"Training contract requires {field}=true")

    if int(train.get("freeze_backbone_epochs", -1)) != 0:
        raise ValueError("Training contract requires train.freeze_backbone_epochs=0")
    if float(train.get("backbone_lr_factor", -1)) != 0.05:
        raise ValueError("Training contract requires train.backbone_lr_factor=0.05")

    decision_values = {
        "base_release_id": EXPECTED_BASE_RELEASE_ID,
        "target_dataset": EXPECTED_DATASET,
        "target_dataset_version": EXPECTED_DATASET_VERSION,
        "target_epochs": EXPECTED_EPOCHS,
    }
    for field, expected in decision_values.items():
        actual = decision.get(field)
        if field == "target_dataset":
            actual = str(actual or "").lower()
        if actual != expected:
            raise ValueError(
                f"Approved decision mismatch for {field}: expected {expected!r}, received {actual!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def validate_inputs(
    decision_path: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    workspace: Path,
    config_path: Path,
    *,
    require_cuda: bool = True,
) -> dict[str, object]:
    if not decision_path.is_file():
        raise FileNotFoundError(f"Missing retraining decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_status = decision.get("status")
    if decision_status != "RETRAIN_APPROVED":
        raise ValueError("Retraining gate is not RETRAIN_APPROVED")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing v2 training config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_training_contract(config, decision)
    configured_epochs = int(config.get("train", {}).get("epochs", 0))
    approved_epochs = int(decision.get("target_epochs", 0))
    if configured_epochs != approved_epochs:
        raise ValueError(
            "Approved retraining epochs do not match config: "
            f"decision={approved_epochs}, config={configured_epochs}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing v1 checkpoint: {checkpoint_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Missing MSMT17 dataset: {dataset_path}")

    train_list = dataset_path / "list_train.txt"
    if not train_list.is_file():
        raise FileNotFoundError(f"Missing MSMT17 train list: {train_list}")
    train_lines = [
        line
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_ids = {int(line.split()[1]) for line in train_lines}
    if len(train_lines) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(
            f"MSMT17 train must contain {EXPECTED_TRAIN_IMAGES} images, "
            f"received {len(train_lines)}"
        )
    if len(train_ids) != EXPECTED_TRAIN_IDENTITIES:
        raise ValueError(
            f"MSMT17 train must contain {EXPECTED_TRAIN_IDENTITIES} identities, "
            f"received {len(train_ids)}"
        )

    free_bytes = shutil.disk_usage(workspace).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise ValueError(
            f"At least 12 GiB free disk is required, received {free_bytes / 1024**3:.1f} GiB"
        )

    gpu_name = "not-required"
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "decision": decision_status,
        "target_epochs": configured_epochs,
        "train_images": len(train_lines),
        "train_identities": len(train_ids),
        "free_disk_gib": round(free_bytes / 1024**3, 1),
        "gpu": gpu_name,
    }


def main() -> None:
    args = parse_args()
    result = validate_inputs(
        Path(args.decision),
        Path(args.checkpoint),
        Path(args.dataset),
        Path(args.workspace),
        Path(args.config),
    )
    print("GPU:", result["gpu"])
    print(
        "MSMT17 train:",
        result["train_images"],
        "images /",
        result["train_identities"],
        "identities",
    )
    print("Retraining gate:", result["decision"])
    print("Approved epochs:", result["target_epochs"])
    print("Free disk:", f"{result['free_disk_gib']} GiB")


if __name__ == "__main__":
    main()
