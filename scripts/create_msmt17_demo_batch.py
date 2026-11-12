from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CAMERAS = (1, 5, 7, 11, 13, 14)


def parse_sample(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected MSMT17 filename: {path.name}")
    return int(parts[0]), int(parts[2])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(
    source_root: Path,
    *,
    count: int,
    cameras: tuple[int, ...],
    seed: int,
) -> list[dict[str, Any]]:
    if count < len(cameras):
        raise ValueError("count must be at least the number of selected cameras")
    train_root = source_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"MSMT17 train split not found: {train_root}")
    by_camera: dict[int, list[tuple[Path, int]]] = {camera: [] for camera in cameras}
    for path in sorted(train_root.rglob("*.jpg")):
        person_id, camera_id = parse_sample(path)
        if camera_id in by_camera:
            by_camera[camera_id].append((path, person_id))

    base, remainder = divmod(count, len(cameras))
    quotas = {
        camera: base + (1 if index < remainder else 0)
        for index, camera in enumerate(cameras)
    }
    selected: list[dict[str, Any]] = []
    content_hashes: set[str] = set()
    for camera in cameras:
        candidates = list(by_camera[camera])
        random.Random(seed + camera).shuffle(candidates)
        camera_samples = []
        for path, person_id in candidates:
            content_sha256 = sha256(path)
            if content_sha256 in content_hashes:
                continue
            content_hashes.add(content_sha256)
            camera_samples.append(
                {
                    "source": path,
                    "person_id": person_id,
                    "camera_id": camera,
                    "sha256": content_sha256,
                }
            )
            if len(camera_samples) == quotas[camera]:
                break
        if len(camera_samples) != quotas[camera]:
            raise ValueError(
                f"Camera {camera} only provides {len(camera_samples)} unique images; "
                f"{quotas[camera]} are required."
            )
        selected.extend(camera_samples)
    return selected


def create_batch(
    source_root: Path,
    output_root: Path,
    *,
    count: int = 2000,
    cameras: tuple[int, ...] = DEFAULT_CAMERAS,
    seed: int = 20261110,
    dataset_label: str = "msmt17-demo",
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing batch: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    selected = select_samples(source_root, count=count, cameras=cameras, seed=seed)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        samples = []
        per_camera: Counter[int] = Counter()
        identities: set[int] = set()
        for item in selected:
            camera = int(item["camera_id"])
            source = Path(item["source"])
            relative = Path("images") / f"c{camera:02d}" / source.name
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
                link_mode = "hardlink"
            except OSError:
                shutil.copy2(source, destination)
                link_mode = "copy"
            samples.append(
                {
                    "image": relative.as_posix(),
                    "person_id": int(item["person_id"]),
                    "camera_id": camera,
                    "sha256": item["sha256"],
                    "source_image": source.relative_to(source_root).as_posix(),
                }
            )
            per_camera[camera] += 1
            identities.add(int(item["person_id"]))

        manifest = {
            "schema_version": 1,
            "batch_id": output_root.name,
            "dataset": dataset_label,
            "source_dataset": "MSMT17_V1",
            "purpose": "six-camera Data Inbox batch with automatic identity-disjoint evaluation split",
            "seed": seed,
            "selected_cameras": list(cameras),
            "selection_policy": {
                "method": "deterministic camera-stratified sampling without replacement",
                "camera_choice": "best six-camera subset from the exhaustive MSMT17 coverage analysis",
                "content_deduplication": "SHA256",
            },
            "summary": {
                "images": len(samples),
                "identities": len(identities),
                "cameras": len(per_camera),
                "images_per_camera": {str(key): per_camera[key] for key in sorted(per_camera)},
                "link_mode": link_mode,
            },
            "samples": samples,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic 2,000-image, six-camera MSMT17 Data Inbox batch."
    )
    parser.add_argument("--source", default="datasets/MSMT17_V1")
    parser.add_argument("--output", default="datasets/inbox/msmt17-demo-v3-2000-6cam")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--cameras", default=",".join(map(str, DEFAULT_CAMERAS)))
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--dataset-label", default="msmt17-demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_batch(
        Path(args.source),
        Path(args.output),
        count=args.count,
        cameras=tuple(int(value) for value in args.cameras.split(",")),
        seed=args.seed,
        dataset_label=args.dataset_label,
    )
    print(json.dumps({"output": args.output, **manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
