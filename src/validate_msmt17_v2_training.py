from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


EXPECTED_TRAIN_IMAGES = 30248
EXPECTED_TRAIN_IDENTITIES = 1041
MINIMUM_FREE_BYTES = 12 * 1024**3


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
