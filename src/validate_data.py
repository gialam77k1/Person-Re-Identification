from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, normalize_data_config
from src.common.utils import save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MARKET1501_NAME = re.compile(r"^(-?\d+)_c(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an image-folder ReID dataset.")
    parser.add_argument("--config", default="configs/vit_reid.yaml")
    parser.add_argument("--dataset-pointer", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def dvc_revision(pointer_path: Path) -> str:
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    outputs = pointer.get("outs", []) if isinstance(pointer, dict) else []
    if len(outputs) != 1 or not outputs[0].get("md5"):
        raise ValueError(f"Invalid DVC pointer: {pointer_path}")
    return str(outputs[0]["md5"])


def inspect_split(split_path: Path) -> dict:
    if not split_path.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_path}")

    images = sorted(
        path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"Dataset split contains no images: {split_path}")

    person_ids: set[int] = set()
    camera_ids: set[int] = set()
    invalid_names: list[str] = []
    for image_path in images:
        match = MARKET1501_NAME.match(image_path.name)
        if match is None:
            invalid_names.append(image_path.name)
            continue
        person_id = int(match.group(1))
        if person_id != -1:
            person_ids.add(person_id)
        camera_ids.add(int(match.group(2)))

    if invalid_names:
        preview = ", ".join(invalid_names[:5])
        raise ValueError(f"Invalid Market-1501 filenames in {split_path}: {preview}")

    return {
        "path": str(split_path.resolve()),
        "num_images": len(images),
        "num_person_ids": len(person_ids),
        "num_cameras": len(camera_ids),
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    pointer_path = (ROOT / args.dataset_pointer).resolve()
    config = normalize_data_config(load_config(config_path))
    splits = {
        split_name: inspect_split(Path(config["data"][f"{split_name}_dir"]))
        for split_name in ("train", "query", "gallery")
    }
    payload = {
        "passed": True,
        "dataset": config["data"]["dataset"]["name"],
        "dataset_root": str(Path(config["data"]["location"]["root"]).resolve()),
        "dataset_dvc_revision": dvc_revision(pointer_path),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "splits": splits,
        "total_images": sum(item["num_images"] for item in splits.values()),
    }
    save_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
