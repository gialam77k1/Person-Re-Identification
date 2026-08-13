from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.utils import (
    configure_torch_home,
    ensure_dir,
    infer_device,
    load_checkpoint,
    resolve_path,
    save_json,
    tee_output,
)
from src.data.dataset import build_dataset, build_transforms
from src.models.reid_model import build_model_from_config
from src.reid.evaluation import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dadnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set data.eval_batch_size=64",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config, args.set, command_name="extract")
    configure_torch_home()
    ensure_dir(config["artifacts"]["logs_dir"])
    ensure_dir(config["artifacts"]["embeddings_dir"])

    log_path = Path(config["artifacts"]["logs_dir"]) / "extract.log"
    with tee_output(log_path):
        run_extract_command(config, args.checkpoint)


def run_extract_command(config: dict, checkpoint_path: str) -> None:
    print(f"Logging console output to {Path(config['artifacts']['logs_dir']) / 'extract.log'}")

    device = infer_device(config["device"])
    _, test_transform = build_transforms(config["data"]["image_height"], config["data"]["image_width"])

    train_dataset = build_dataset(config, "train", transform=test_transform, relabel=True)
    train_loader = DataLoader(
        train_dataset,
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
    load_checkpoint(model, checkpoint_path, device)

    features, pids, camids, paths = extract_features(model, train_loader, device)
    embeddings_path = resolve_path(Path(config["artifacts"]["embeddings_dir"]) / "reference_embeddings.npy")
    pids_path = resolve_path(Path(config["artifacts"]["embeddings_dir"]) / "reference_pids.npy")
    camids_path = resolve_path(Path(config["artifacts"]["embeddings_dir"]) / "reference_camids.npy")

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, features)
    np.save(pids_path, pids)
    np.save(camids_path, camids)

    manifest = {
        "run_slug": config["runtime"]["run_slug"],
        "run_root": config["runtime"]["run_root"],
        "dataset": config["data"]["dataset"]["name"],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "embeddings_path": str(embeddings_path),
        "pids_path": str(pids_path),
        "camids_path": str(camids_path),
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "first_paths": paths[:10],
    }
    save_json(manifest, Path(config["artifacts"]["embeddings_dir"]) / "reference_manifest.json")
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")
    print(manifest)


if __name__ == "__main__":
    main()
