from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.common.utils import resolve_path
from src.data.reid_filename import parse_image_folder_reid_filename
from src.validate_v2_evaluation import canonical_dataset_name


def _resolve_split(root: Path, value: str) -> Path:
    split = Path(value)
    return split if split.is_absolute() else root / split


def validate_image_folder_dataset(payload: dict, *, config_path: Path) -> dict:
    """Scan evaluation inputs before GPU training starts.

    This deliberately validates the files with the same filename contract used
    by the runtime loader.  It catches malformed future inbox/evaluation views
    and also verifies that query identities can actually be retrieved from a
    different camera in the gallery.
    """

    data = payload.get("data", {})
    if str(data.get("source_type", "image_folder")) != "image_folder":
        raise ValueError(f"Unsupported evaluation source type in {config_path}")

    location = data.get("location", {})
    root_value = location.get("root") or data.get("root")
    splits = location.get("splits") or data.get("splits")
    if not root_value or not isinstance(splits, dict):
        raise ValueError(f"Evaluation dataset location is incomplete in {config_path}")

    root = resolve_path(str(root_value))
    split_defaults = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }
    parsed: dict[str, list[tuple[int, int]]] = {}
    for split_name, default in split_defaults.items():
        split_dir = _resolve_split(root, str(splits.get(split_name, default)))
        if not split_dir.is_dir():
            raise ValueError(f"Evaluation split does not exist: {split_dir}")
        images = sorted(split_dir.glob("*.jpg"))
        if not images:
            raise ValueError(f"Evaluation split contains no JPG images: {split_dir}")
        parsed[split_name] = [
            parsed_name
            for image in images
            if (parsed_name := parse_image_folder_reid_filename(image))[0] != -1
        ]
        if not parsed[split_name]:
            raise ValueError(f"Evaluation split contains no valid identities: {split_dir}")

    gallery_cameras: dict[int, set[int]] = {}
    for pid, camid in parsed["gallery"]:
        gallery_cameras.setdefault(pid, set()).add(camid)
    valid_queries = sum(
        1
        for pid, camid in parsed["query"]
        if any(gallery_cam != camid for gallery_cam in gallery_cameras.get(pid, set()))
    )
    if valid_queries == 0:
        raise ValueError(
            f"Evaluation dataset in {config_path} has no query identity visible "
            "from a different gallery camera"
        )

    return {
        "root": str(root),
        "train_images": len(parsed["train"]),
        "query_images": len(parsed["query"]),
        "gallery_images": len(parsed["gallery"]),
        "train_identities": len({pid for pid, _ in parsed["train"]}),
        "valid_queries": valid_queries,
    }


def validate_config(
    path: Path,
    *,
    release_id: str,
    dataset: str,
    report: Path,
    scan_dataset: bool = False,
) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config is not a mapping: {path}")
    actual_release = str(payload.get("release", {}).get("id", ""))
    if actual_release != release_id:
        raise ValueError(
            f"Evaluation config release mismatch in {path}: "
            f"{actual_release!r} != {release_id!r}"
        )
    actual_dataset = canonical_dataset_name(
        str(payload.get("data", {}).get("dataset", {}).get("name", ""))
    )
    expected_dataset = canonical_dataset_name(dataset)
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"Evaluation config dataset mismatch in {path}: "
            f"{actual_dataset!r} != {expected_dataset!r}"
        )
    run_root = Path(str(payload.get("artifacts", {}).get("run_root", "")))
    filename = str(payload.get("evaluation", {}).get("report_filename", ""))
    configured_report = run_root / "metrics" / filename
    if configured_report != report:
        raise ValueError(
            f"Evaluation report path mismatch in {path}: "
            f"{configured_report} != {report}"
        )
    result = {
        "config": str(path),
        "dataset": actual_dataset,
        "report": str(report),
    }
    if scan_dataset:
        result["dataset_contract"] = validate_image_folder_dataset(payload, config_path=path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail before GPU training if post-training evaluation wiring is inconsistent."
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--duke-config", required=True)
    parser.add_argument("--duke-report", required=True)
    parser.add_argument("--market-config", required=True)
    parser.add_argument("--market-report", required=True)
    parser.add_argument("--forgetting-policy", required=True)
    parser.add_argument("--validation-policy", required=True)
    parser.add_argument("--target-dataset", default="dukemtmc-reid")
    args = parser.parse_args()
    try:
        checks = [
            validate_config(
                Path(args.duke_config),
                release_id=args.release_id,
                dataset=args.target_dataset,
                report=Path(args.duke_report),
                scan_dataset=True,
            ),
            validate_config(
                Path(args.market_config),
                release_id=args.release_id,
                dataset="market1501",
                report=Path(args.market_report),
                scan_dataset=True,
            ),
        ]
        for policy in (Path(args.forgetting_policy), Path(args.validation_policy)):
            payload = yaml.safe_load(policy.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload:
                raise ValueError(f"Policy is empty or invalid: {policy}")
        print(json.dumps({"status": "PASS", "checks": checks}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
