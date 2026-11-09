from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, normalize_data_config
from src.common.utils import save_json
from src.validate_data import dvc_revision, git_commit, inspect_dataset_splits, sha256_file


def inspect_config(config_path: Path, pointer_path: Path) -> dict:
    config = normalize_data_config(load_config(config_path))
    splits = inspect_dataset_splits(config)
    return {
        "dataset": config["data"]["dataset"]["name"],
        "version": config["data"]["dataset"].get("version"),
        "dataset_root": str(Path(config["data"]["location"]["root"]).resolve()),
        "dataset_dvc_revision": dvc_revision(pointer_path),
        "pointer_path": str(pointer_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "splits": splits,
        "total_images": sum(item["num_images"] for item in splits.values()),
    }


def build_manifest(
    duke_config: Path,
    duke_pointer: Path,
    market1501_config: Path,
    market1501_pointer: Path,
) -> dict:
    return {
        "schema_version": 1,
        "passed": True,
        "git_commit": git_commit(),
        "datasets": {
            "duke": inspect_config(duke_config, duke_pointer),
            "market1501": inspect_config(market1501_config, market1501_pointer),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duke-config", required=True)
    parser.add_argument("--duke-pointer", required=True)
    parser.add_argument("--market1501-config", required=True)
    parser.add_argument("--market1501-pointer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_manifest(
        Path(args.duke_config).resolve(),
        Path(args.duke_pointer).resolve(),
        Path(args.market1501_config).resolve(),
        Path(args.market1501_pointer).resolve(),
    )
    save_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
