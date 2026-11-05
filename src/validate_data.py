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


def inspect_msmt17_split(dataset_root: Path, split_name: str, list_path: Path) -> dict:
    if not list_path.is_file():
        raise FileNotFoundError(f"Missing MSMT17 list file: {list_path}")

    split_dir_name = "train" if split_name == "train" else "test"
    split_dir = dataset_root / split_dir_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing MSMT17 split directory: {split_dir}")

    person_ids: set[int] = set()
    camera_ids: set[int] = set()
    listed_paths: set[Path] = set()
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid MSMT17 entry at {list_path}:{line_number}: {raw_line}")

        relative_path = Path(fields[0])
        person_id = int(fields[1])
        image_path = split_dir / relative_path
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"Missing MSMT17 image listed at {list_path}:{line_number}: {image_path}")
        if image_path in listed_paths:
            raise ValueError(f"Duplicate MSMT17 image in {list_path}: {relative_path}")

        name_parts = relative_path.stem.split("_")
        if len(name_parts) < 3:
            raise ValueError(f"Invalid MSMT17 filename at {list_path}:{line_number}: {relative_path.name}")
        filename_pid = int(name_parts[0])
        if filename_pid != person_id:
            raise ValueError(
                f"MSMT17 person ID mismatch at {list_path}:{line_number}: "
                f"filename={filename_pid}, label={person_id}"
            )

        listed_paths.add(image_path)
        person_ids.add(person_id)
        camera_ids.add(int(name_parts[2]))

    if not listed_paths:
        raise RuntimeError(f"MSMT17 list contains no images: {list_path}")

    return {
        "path": str(split_dir.resolve()),
        "list_file": str(list_path.resolve()),
        "num_images": len(listed_paths),
        "num_person_ids": len(person_ids),
        "num_cameras": len(camera_ids),
    }


def inspect_dataset_splits(config: dict) -> dict[str, dict]:
    dataset_name = str(config["data"]["dataset"]["name"]).strip().lower()
    if dataset_name != "msmt17":
        return {
            split_name: inspect_split(Path(config["data"][f"{split_name}_dir"]))
            for split_name in ("train", "query", "gallery")
        }

    dataset_root = Path(config["data"]["location"]["root"])
    list_files = config["data"]["location"].get("list_files", {})
    if not isinstance(list_files, dict):
        raise ValueError("MSMT17 config requires data.location.list_files.")

    def resolve_config_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    return {
        split_name: inspect_msmt17_split(
            resolve_config_path(str(dataset_root)),
            split_name,
            resolve_config_path(str(list_files[split_name])),
        )
        for split_name in ("train", "query", "gallery")
    }


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    pointer_path = (ROOT / args.dataset_pointer).resolve()
    config = normalize_data_config(load_config(config_path))
    splits = inspect_dataset_splits(config)
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
