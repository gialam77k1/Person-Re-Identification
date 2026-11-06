from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_runtime_config
from src.common.mlflow_utils import get_dvc_revision, get_git_commit
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_reid.yaml")
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
    _, test_transform = build_transforms(
        config["data"]["image_height"],
        config["data"]["image_width"],
        preserve_aspect_ratio=config["data"].get("preserve_aspect_ratio", False),
    )

    train_dataset = build_dataset(config, "train", transform=test_transform, relabel=False)
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
    paths_path = resolve_path(Path(config["artifacts"]["embeddings_dir"]) / "reference_paths.npy")

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, features)
    np.save(pids_path, pids)
    np.save(camids_path, camids)
    np.save(paths_path, np.asarray(paths))

    if features.ndim != 2 or features.shape[1] != int(config["model"]["embedding_dim"]):
        raise ValueError(f"Unexpected reference embedding shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Reference embeddings contain NaN or Inf")
    norms = np.linalg.norm(features, axis=1)
    norm_max_abs_error = float(np.max(np.abs(norms - 1.0))) if norms.size else float("inf")
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(
            "Reference embeddings are not L2-normalized: "
            f"maximum norm error is {norm_max_abs_error:.8f}"
        )

    dataset_pointer = config.get("evaluation", {}).get("dataset_pointer")
    dataset_revision = get_dvc_revision(dataset_pointer) if dataset_pointer else None
    checkpoint_resolved = resolve_path(checkpoint_path)
    generated_files = {
        "embeddings": embeddings_path,
        "pids": pids_path,
        "camids": camids_path,
        "paths": paths_path,
    }

    manifest = {
        "schema_version": 2,
        "release_id": config.get("release", {}).get("id"),
        "run_slug": config["runtime"]["run_slug"],
        "run_root": config["runtime"]["run_root"],
        "dataset": config["data"]["dataset"]["name"],
        "dataset_revision": dataset_revision,
        "git_commit": get_git_commit(),
        "checkpoint": repository_path(checkpoint_resolved),
        "checkpoint_sha256": sha256_file(checkpoint_resolved),
        "embeddings_path": repository_path(embeddings_path),
        "pids_path": repository_path(pids_path),
        "camids_path": repository_path(camids_path),
        "paths_path": repository_path(paths_path),
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "finite": True,
        "l2_normalized": True,
        "l2_norm": {
            "minimum": float(norms.min()),
            "maximum": float(norms.max()),
            "mean": float(norms.mean()),
            "max_abs_error_from_one": norm_max_abs_error,
            "atol": 1e-4,
        },
        "files": {
            role: {
                "path": repository_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in generated_files.items()
        },
        "first_paths": paths[:10],
    }
    save_json(manifest, Path(config["artifacts"]["embeddings_dir"]) / "reference_manifest.json")
    save_json(config, Path(config["artifacts"]["logs_dir"]) / "effective_config.json")
    print(manifest)


if __name__ == "__main__":
    main()
