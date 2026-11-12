from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml


REID_FILE = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)_")
DEFAULT_EVALUATION_HOLDOUT_IMAGES = 200


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
    parser.add_argument(
        "--evaluation-holdout-images",
        type=int,
        default=DEFAULT_EVALUATION_HOLDOUT_IMAGES,
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dataset-name")
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
    source: Path,
    samples: list[tuple[Path, int, int]],
    *,
    holdout_target: int = 200,
    split_seed: int = 42,
) -> tuple[list[tuple[Path, int, int]], str]:
    accepted: list[tuple[Path, int, int]] = []
    seen: set[str] = set()
    aggregate = hashlib.sha256()
    aggregate.update(
        f"reid-inbox-split:v2\0holdout={holdout_target}\0seed={split_seed}\n".encode()
    )
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


def split_identity_disjoint(
    samples: list[tuple[Path, int, int]],
    *,
    holdout_target: int = 200,
    seed: int = 42,
) -> dict[str, list[tuple[Path, int, int]]]:
    """Build deterministic ReID splits without sharing identities with train."""
    if holdout_target < 2 or len(samples) <= holdout_target:
        raise ValueError(
            f"Cannot reserve {holdout_target} evaluation images from "
            f"a batch containing {len(samples)} valid images."
        )
    grouped: dict[int, list[tuple[Path, int, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    eligible = [
        pid for pid, items in grouped.items()
        if len({item[2] for item in items}) >= 2
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    query: list[tuple[Path, int, int]] = []
    gallery: list[tuple[Path, int, int]] = []
    evaluation_ids: set[int] = set()
    target_queries = max(1, holdout_target // 4)

    def add_identity(pid: int) -> bool:
        items = list(grouped[pid])
        rng.shuffle(items)
        query_item = items[0]
        gallery_item = next(
            (item for item in items[1:] if item[2] != query_item[2]), None
        )
        if gallery_item is None:
            return False
        evaluation_ids.add(pid)
        query.append(query_item)
        gallery.append(gallery_item)
        return True

    remaining = iter(eligible)
    while len(query) < target_queries:
        try:
            pid = next(remaining)
        except StopIteration as exc:
            raise ValueError(
                "Batch does not contain enough identities observed by at least "
                f"two cameras; need {target_queries} evaluation identities."
            ) from exc
        add_identity(pid)

    used = {item[0] for item in [*query, *gallery]}
    for pid in list(evaluation_ids):
        query_camera = next(item[2] for item in query if item[1] == pid)
        extras = [
            item for item in grouped[pid]
            if item[0] not in used and item[2] != query_camera
        ]
        rng.shuffle(extras)
        for item in extras:
            if len(query) + len(gallery) >= holdout_target:
                break
            gallery.append(item)
            used.add(item[0])

    while len(query) + len(gallery) < holdout_target:
        try:
            pid = next(remaining)
        except StopIteration as exc:
            raise ValueError(
                "Batch cannot provide the requested cross-camera evaluation "
                f"holdout of {holdout_target} images."
            ) from exc
        add_identity(pid)

    train = [item for item in samples if item[1] not in evaluation_ids]
    if not train:
        raise ValueError("Evaluation identities consumed the complete training batch.")
    return {"train": train, "query": query, "gallery": gallery}


def build_view(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.source).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    holdout_target = int(getattr(args, "evaluation_holdout_images", 200))
    split_seed = int(getattr(args, "split_seed", 42))
    samples, actual_revision = validate_samples(
        source,
        load_samples(source),
        holdout_target=holdout_target,
        split_seed=split_seed,
    )
    if not samples:
        raise ValueError(f"No valid labeled samples found in {source}")
    if actual_revision != args.revision:
        raise ValueError(
            "Data Inbox batch changed after approval: "
            f"expected {args.revision}, received {actual_revision}"
        )

    splits = split_identity_disjoint(
        samples,
        holdout_target=holdout_target,
        seed=split_seed,
    )
    directories = {
        "train": output_root / "bounding_box_train",
        "query": output_root / "query",
        "gallery": output_root / "bounding_box_test",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        for index, (path, pid, cam) in enumerate(split_samples):
            destination = directories[split_name] / (
                f"{pid:06d}_c{cam}_inbox_{index:08d}.jpg"
            )
            if not destination.exists():
                os.symlink(path, destination)

    identities = {pid for _, pid, _ in splits["train"]}
    cameras = {cam for _, _, cam in splits["train"]}
    config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name:
        config["data"].setdefault("dataset", {})["name"] = dataset_name
        config["data"]["dataset"]["version"] = args.revision
    config["data"]["location"]["root"] = str(output_root)
    config["data"]["expected_train_images"] = len(splits["train"])
    config["data"]["expected_train_identities"] = len(identities)
    config["data"]["evaluation_split"] = {
        "strategy": "identity_disjoint_cross_camera",
        "seed": split_seed,
        "query_images": len(splits["query"]),
        "gallery_images": len(splits["gallery"]),
    }
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
        "source_images": len(samples),
        "train_images": len(splits["train"]),
        "train_identities": len(identities),
        "train_cameras": len(cameras),
        "query_images": len(splits["query"]),
        "gallery_images": len(splits["gallery"]),
        "evaluation_identities": len({pid for _, pid, _ in splits["query"]}),
        "identity_overlap": bool(
            {pid for _, pid, _ in splits["train"]}
            & {pid for _, pid, _ in [*splits["query"], *splits["gallery"]]}
        ),
        "view": str(output_root),
        "config_sha256": hashlib.sha256(Path(args.output_config).read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    build_view(parse_args())


if __name__ == "__main__":
    main()
