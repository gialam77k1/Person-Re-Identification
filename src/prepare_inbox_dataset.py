from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import yaml


REID_FILE = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an immutable training view for a Data Inbox batch.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-pointer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-release")
    parser.add_argument("--target-release")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--run-root")
    return parser.parse_args()


def load_samples(source: Path) -> list[tuple[Path, int, int]]:
    manifest_path = source / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = []
        for sample in manifest["samples"]:
            path = (source / sample["image"]).resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise ValueError(f"Invalid manifest image: {sample['image']}")
            result.append((path, int(sample["person_id"]), int(sample["camera_id"])))
        return result

    result = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        match = REID_FILE.match(path.name)
        if match and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result.append((path.resolve(), int(match.group("pid")), int(match.group("cam"))))
    return result


def validate_samples(
    source: Path, samples: list[tuple[Path, int, int]]
) -> tuple[list[tuple[Path, int, int]], str]:
    accepted: list[tuple[Path, int, int]] = []
    seen: set[str] = set()
    aggregate = hashlib.sha256()
    for path, pid, cam in samples:
        content = path.read_bytes()
        valid_magic = content.startswith(b"\xff\xd8\xff") or content.startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        content_sha = hashlib.sha256(content).hexdigest()
        if pid < 0 or cam < 1 or not valid_magic or content_sha in seen:
            continue
        seen.add(content_sha)
        relative = path.relative_to(source).as_posix()
        aggregate.update(
            f"{relative}\0{pid}\0{cam}\0{len(content)}\0{content_sha}\n".encode()
        )
        accepted.append((path, pid, cam))
    return accepted, aggregate.hexdigest()


def build_view(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source).resolve()
    base_dataset = Path(args.base_dataset).resolve()
    output_root = Path(args.output_root).resolve()
    train_root = output_root / "bounding_box_train"
    output_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    samples, actual_revision = validate_samples(source, load_samples(source))
    if not samples:
        raise ValueError(f"No valid labeled samples found in {source}")
    if actual_revision != args.revision:
        raise ValueError(
            "Data Inbox batch changed after approval: "
            f"expected {args.revision}, received {actual_revision}"
        )

    for index, (path, pid, cam) in enumerate(samples):
        destination = train_root / f"{pid:06d}_c{cam}_inbox_{index:08d}.jpg"
        if not destination.exists():
            os.symlink(path, destination)
    for name in ("query", "bounding_box_test"):
        destination = output_root / name
        target = base_dataset / name
        if not destination.exists():
            os.symlink(target, destination, target_is_directory=True)

    identities = {pid for _, pid, _ in samples}
    cameras = {cam for _, _, cam in samples}
    config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    config["data"]["location"]["root"] = str(output_root)
    config["data"]["expected_train_images"] = len(samples)
    config["data"]["expected_train_identities"] = len(identities)
    config["evaluation"]["dataset_pointer"] = str(Path(args.output_pointer).resolve())
    source_release = getattr(args, "source_release", None)
    target_release = getattr(args, "target_release", None)
    base_checkpoint = getattr(args, "base_checkpoint", None)
    run_root = getattr(args, "run_root", None)
    if source_release:
        config.setdefault("release", {})["base_release_id"] = source_release
    if target_release:
        config.setdefault("release", {})["id"] = target_release
        config["experiment_name"] = target_release.replace(":", "-")
    if base_checkpoint:
        config.setdefault("transfer", {})["base_checkpoint"] = base_checkpoint
        if source_release and not source_release.startswith("market1501-vit-bnneck:v1"):
            config["transfer"]["expected_checkpoint_epoch"] = None
    if run_root:
        config.setdefault("artifacts", {})["run_root"] = run_root
    Path(args.output_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_config).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    Path(args.output_pointer).write_text(
        yaml.safe_dump({"outs": [{"md5": args.revision, "path": str(source)}]}),
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "dataset_revision": args.revision,
        "train_images": len(samples),
        "train_identities": len(identities),
        "train_cameras": len(cameras),
        "view": str(output_root),
        "config_sha256": hashlib.sha256(Path(args.output_config).read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    build_view(parse_args())


if __name__ == "__main__":
    main()
