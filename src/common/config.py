from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")

    config["config_path"] = str(path)
    config["project_root"] = str(path.parent.parent.resolve())
    return config


def apply_config_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for dotted_key, value in overrides.items():
        if value is None:
            continue

        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    return config


def _resolve_split_path(dataset_root: str | Path, split_path: str) -> str:
    split = Path(split_path)
    if split.is_absolute():
        return str(split)
    return str(Path(dataset_root) / split)


def _normalize_image_folder_location(data: dict[str, Any], dataset: dict[str, Any]) -> None:
    location = data.setdefault("location", {})
    location_root = location.get("root") or dataset.get("root") or data.get("root") or "."

    splits = location.get("splits")
    if not isinstance(splits, dict):
        splits = data.get("splits")

    if not isinstance(splits, dict):
        splits = {
            "train": data.get("train_dir", "bounding_box_train"),
            "query": data.get("query_dir", "query"),
            "gallery": data.get("gallery_dir", "bounding_box_test"),
        }
    else:
        splits = {
            "train": splits.get("train", data.get("train_dir", "bounding_box_train")),
            "query": splits.get("query", data.get("query_dir", "query")),
            "gallery": splits.get("gallery", data.get("gallery_dir", "bounding_box_test")),
        }

    location["root"] = str(location_root)
    location["splits"] = splits
    dataset["root"] = str(location_root)
    data["root"] = str(location_root)
    data["splits"] = splits
    data["train_dir"] = _resolve_split_path(location_root, str(splits["train"]))
    data["query_dir"] = _resolve_split_path(location_root, str(splits["query"]))
    data["gallery_dir"] = _resolve_split_path(location_root, str(splits["gallery"]))


def normalize_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.setdefault("data", {})
    dataset = data.setdefault("dataset", {})

    dataset_name = dataset.get("name") or data.get("dataset_name") or "market1501"
    source_type = str(data.get("source_type") or dataset.get("source_type") or "image_folder")

    dataset["name"] = str(dataset_name)
    dataset["source_type"] = source_type
    data["source_type"] = source_type

    if source_type == "image_folder":
        _normalize_image_folder_location(data, dataset)
        return config

    raise ValueError(
        f"Unsupported data.source_type '{source_type}'. "
        "Currently supported: image_folder"
    )
    return config


def parse_config_overrides(override_pairs: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw_override in override_pairs or []:
        if "=" not in raw_override:
            raise ValueError(
                f"Invalid override '{raw_override}'. Expected format key=value, "
                "for example: data.batch_size=32"
            )

        dotted_key, raw_value = raw_override.split("=", 1)
        dotted_key = dotted_key.strip()
        if not dotted_key:
            raise ValueError(f"Invalid override '{raw_override}'. Key cannot be empty.")

        overrides[dotted_key] = yaml.safe_load(raw_value)

    return overrides


def load_runtime_config(
    config_path: str | Path,
    override_pairs: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    overrides = parse_config_overrides(override_pairs)
    if overrides:
        apply_config_overrides(config, overrides)
    normalize_data_config(config)
    return config
