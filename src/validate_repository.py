from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}.")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_training_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("data", {}).get("dataset", {}).get("name") != "market1501":
        errors.append("Training config must target Market-1501.")
    if config.get("model", {}).get("backbone") != "vit_b_16":
        errors.append("Training config must use vit_b_16.")
    if config.get("model", {}).get("embedding_dim") != 512:
        errors.append("Training config embedding_dim must be 512.")
    if config.get("data", {}).get("preserve_aspect_ratio") is not True:
        errors.append("Training config must preserve image aspect ratio.")
    if config.get("train", {}).get("early_stopping", {}).get("enabled") is not True:
        errors.append("Early stopping must remain enabled.")
    return errors


def validate_preprocessing(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != 1:
        errors.append("Preprocessing schema_version must be 1.")
    if contract.get("input", {}).get("shape") != [1, 3, 224, 224]:
        errors.append("Preprocessing input shape must be [1, 3, 224, 224].")
    if contract.get("resize", {}).get("preserve_aspect_ratio") is not True:
        errors.append("Preprocessing must preserve aspect ratio.")
    if contract.get("padding", {}).get("fill_rgb") != [124, 116, 104]:
        errors.append("Preprocessing padding fill must be [124, 116, 104].")
    return errors


def validate_release_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    release = manifest.get("release", {})
    source = manifest.get("source", {})
    model = manifest.get("model", {})
    contract = manifest.get("contract", {})
    if not release.get("release_id"):
        errors.append("Model manifest is missing release.release_id.")
    git_commit = source.get("git_commit", "")
    if not isinstance(git_commit, str) or len(git_commit) != 40:
        errors.append("Model manifest source.git_commit must be a full 40-character SHA.")
    if model.get("embedding_dim") != 512:
        errors.append("Model manifest embedding_dim must be 512.")

    input_contract = contract.get("input", {})
    output_contract = contract.get("output", {})
    if input_contract.get("shape") != [1, 3, 224, 224]:
        errors.append("Model manifest input shape is invalid.")
    if output_contract.get("shape") != [1, 512]:
        errors.append("Model manifest output shape is invalid.")

    files = manifest.get("files", [])
    roles = {item.get("role"): item for item in files if isinstance(item, dict)}
    for role in ("onnx_graph", "onnx_external_weights", "preprocessing_contract"):
        item = roles.get(role)
        checksum = item.get("sha256", "") if item else ""
        if not isinstance(checksum, str) or len(checksum) != 64:
            errors.append(f"Model manifest is missing a valid SHA256 for {role}.")

    preprocessing = roles.get("preprocessing_contract")
    if preprocessing:
        path = root / preprocessing.get("repository_path", "")
        if not path.is_file():
            errors.append(f"Tracked preprocessing contract does not exist: {path}")
        elif sha256(path) != preprocessing.get("sha256"):
            errors.append("Tracked preprocessing checksum does not match model manifest.")
    return errors


def validate_dvc_files(root: Path) -> list[str]:
    errors: list[str] = []
    pipeline = read_yaml(root / "dvc.yaml")
    expected_stages = {
        "validate_data",
        "train",
        "evaluate",
        "export_onnx",
        "verify_onnx",
        "extract_reference",
        "validate_release",
        "package_triton",
    }
    actual_stages = set(pipeline.get("stages", {}))
    missing = sorted(expected_stages - actual_stages)
    if missing:
        errors.append(f"DVC pipeline is missing stages: {', '.join(missing)}")

    params = read_yaml(root / "params.yaml")
    if not params.get("pipeline", {}).get("release_id"):
        errors.append("params.yaml is missing pipeline.release_id.")
    thresholds = read_yaml(root / "configs/model_validation.yaml")
    if thresholds.get("quality", {}).get("rank1_min") != 0.91:
        errors.append("Validation rank1_min must be 0.91.")
    if thresholds.get("contract", {}).get("embedding_dim") != 512:
        errors.append("Validation embedding_dim must be 512.")

    pointer = read_yaml(root / "datasets/Market-1501-v15.09.15.dvc")
    outputs = pointer.get("outs", [])
    if len(outputs) != 1 or not outputs[0].get("md5"):
        errors.append("Market-1501 DVC pointer is invalid.")
    return errors


def validate_repository(root: Path = ROOT) -> list[str]:
    errors = validate_training_config(read_yaml(root / "configs/vit_reid.yaml"))
    preprocessing_path = root / "model_releases/market1501-vit-bnneck/v1/preprocessing.json"
    errors.extend(validate_preprocessing(read_json(preprocessing_path)))
    manifest = read_json(root / "model_releases/market1501-vit-bnneck/v1/model_manifest.json")
    errors.extend(validate_release_manifest(manifest, root))
    errors.extend(validate_dvc_files(root))
    return errors


def main() -> None:
    errors = validate_repository()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Repository config, manifest, preprocessing contract and DVC metadata: PASS")


if __name__ == "__main__":
    main()
