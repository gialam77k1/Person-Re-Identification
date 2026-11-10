from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml


EXPECTED_TRAIN_IMAGES = 16522
EXPECTED_TRAIN_IDENTITIES = 702
MINIMUM_FREE_BYTES = 12 * 1024**3
EXPECTED_RELEASE_ID = "market1501-duke-vit-bnneck:v2"
EXPECTED_BASE_RELEASE_ID = "market1501-vit-bnneck:v1-dvc"
EXPECTED_DATASET = "dukemtmc-reid"
EXPECTED_DATASET_VERSION = "dukemtmc-reid"
EXPECTED_CHECKPOINT_EPOCH = 48
EXPECTED_EPOCHS = 50
EXPECTED_TRAIN_BATCH_SIZE = 16
EXPECTED_EVAL_BATCH_SIZE = 16
EXPECTED_QUERY_BATCH_SIZE = 64
DUKE_FILENAME = re.compile(r"^(?P<pid>-?\d+)_c(?P<camera>\d+)_")


def validate_training_contract(config: dict, decision: dict) -> None:
    release = config.get("release", {})
    data = config.get("data", {})
    dataset = data.get("dataset", {})
    transfer = config.get("transfer", {})
    train = config.get("train", {})
    progressive = train.get("progressive_unfreeze", {})
    early_stopping = train.get("early_stopping", {})

    expected_release = decision.get("target_release_id", EXPECTED_RELEASE_ID)
    expected_base = decision.get("base_release_id", EXPECTED_BASE_RELEASE_ID)
    expected_values = {
        "release.id": (release.get("id"), expected_release),
        "release.base_release_id": (
            release.get("base_release_id"),
            expected_base,
        ),
        "data.dataset.name": (str(dataset.get("name", "")).lower(), EXPECTED_DATASET),
        "data.dataset.version": (dataset.get("version"), EXPECTED_DATASET_VERSION),
        "data.batch_size": (data.get("batch_size"), EXPECTED_TRAIN_BATCH_SIZE),
        "data.eval_batch_size": (
            data.get("eval_batch_size"),
            EXPECTED_EVAL_BATCH_SIZE,
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
        "base_release_id": expected_base,
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
        raise FileNotFoundError(f"Missing iterative training config: {config_path}")
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
        raise FileNotFoundError(f"Missing source champion checkpoint: {checkpoint_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Missing DukeMTMC-reID dataset: {dataset_path}")

    train_dir = dataset_path / "bounding_box_train"
    query_dir = dataset_path / "query"
    gallery_dir = dataset_path / "bounding_box_test"
    for split_dir in (train_dir, query_dir, gallery_dir):
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing DukeMTMC-reID split: {split_dir}")

    train_images = sorted(train_dir.glob("*.jpg"))
    parsed = [DUKE_FILENAME.match(path.name) for path in train_images]
    if any(match is None for match in parsed):
        raise ValueError("DukeMTMC-reID train contains an invalid filename")
    train_ids = {int(match.group("pid")) for match in parsed if match and int(match.group("pid")) >= 0}
    train_cameras = {int(match.group("camera")) for match in parsed if match}
    expected_train_images = int(config.get("data", {}).get("expected_train_images", 0))
    expected_train_identities = int(
        config.get("data", {}).get("expected_train_identities", 0)
    )
    if expected_train_images < 1000:
        raise ValueError("Training config requires at least 1000 labeled images")
    if expected_train_identities < 50:
        raise ValueError("Training config requires at least 50 identities")
    if len(train_images) != expected_train_images:
        raise ValueError(
            f"DukeMTMC-reID train must contain {expected_train_images} images, "
            f"received {len(train_images)}"
        )
    if len(train_ids) != expected_train_identities:
        raise ValueError(
            f"DukeMTMC-reID train must contain {expected_train_identities} identities, "
            f"received {len(train_ids)}"
        )
    if len(train_cameras) < 2:
        raise ValueError(
            "DukeMTMC-reID train must contain at least 2 cameras, "
            f"received {sorted(train_cameras)}"
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
        "train_images": len(train_images),
        "train_identities": len(train_ids),
        "train_cameras": len(train_cameras),
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
        "DukeMTMC-reID train:",
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
