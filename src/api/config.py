from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_SLUG = "market1501-vit-bnneck-v1-dvc"


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def resolve_repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ApiSettings:
    triton_url: str
    qdrant_url: str
    qdrant_collection: str
    model_name: str
    model_version: str
    serving_model_repository: Path
    model_manifest_path: Path
    max_upload_bytes: int
    max_image_pixels: int
    default_top_k: int
    max_top_k: int
    upstream_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "ApiSettings":
        manifest_default = (
            f"artifacts/pipeline/{DEFAULT_RELEASE_SLUG}/release/model_manifest.json"
        )
        return cls(
            triton_url=os.getenv("TRITON_URL", "http://localhost:8000").rstrip("/"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "reid_reference_v1"),
            model_name=os.getenv("TRITON_MODEL_NAME", "reid_embedding"),
            model_version=os.getenv("TRITON_MODEL_VERSION", "1"),
            serving_model_repository=resolve_repository_path(
                os.getenv(
                    "API_TRITON_MODEL_REPOSITORY",
                    f"artifacts/pipeline/{DEFAULT_RELEASE_SLUG}/triton/model_repository",
                )
            ),
            model_manifest_path=resolve_repository_path(
                os.getenv("MODEL_MANIFEST_PATH", manifest_default)
            ),
            max_upload_bytes=_env_int("API_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_env_int("API_MAX_IMAGE_PIXELS", 25_000_000),
            default_top_k=_env_int("API_DEFAULT_TOP_K", 5),
            max_top_k=_env_int("API_MAX_TOP_K", 100),
            upstream_timeout_seconds=_env_int("API_UPSTREAM_TIMEOUT_SECONDS", 30),
        )


@dataclass(frozen=True)
class ReleaseContract:
    release_id: str
    git_commit: str
    input_name: str
    output_name: str
    input_shape: tuple[int, int, int, int]
    embedding_dim: int
    preprocessing_path: Path
    preprocessing_version: int
    preprocessing_sha256: str
    preprocessing: dict[str, Any]
    onnx_sha256: str
    external_weights_sha256: str

    @classmethod
    def load(cls, manifest_path: Path) -> "ReleaseContract":
        manifest = read_json(manifest_path)
        release = manifest.get("release", {})
        source = manifest.get("source", {})
        contract = manifest.get("contract", {})
        validation = manifest.get("validation", {})
        release_id = release.get("release_id")
        git_commit = source.get("git_commit")
        input_shape = contract.get("input_shape")
        if not release_id or not git_commit:
            raise ValueError("Model manifest is missing release_id or source.git_commit.")
        if validation.get("status") != "PASS":
            raise ValueError("Model manifest has not passed the validation gate.")
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError("Model manifest input_shape must contain four dimensions.")

        preprocessing_value = contract.get("preprocessing_path")
        if not preprocessing_value:
            raise ValueError("Model manifest is missing contract.preprocessing_path.")
        preprocessing_path = resolve_repository_path(preprocessing_value)
        preprocessing = read_json(preprocessing_path)
        preprocessing_version = int(preprocessing.get("schema_version", 0))
        if preprocessing_version <= 0:
            raise ValueError("Preprocessing contract has no valid schema_version.")

        files = manifest.get("files", [])
        checksums = {
            item.get("role"): item.get("sha256")
            for item in files
            if isinstance(item, dict)
        }
        onnx_sha256 = checksums.get("onnx_graph")
        external_weights_sha256 = checksums.get("onnx_external_weights")
        preprocessing_expected = checksums.get("preprocessing_contract")
        if not onnx_sha256 or not external_weights_sha256 or not preprocessing_expected:
            raise ValueError("Model manifest is missing serving artifact checksums.")
        preprocessing_actual = sha256(preprocessing_path)
        if preprocessing_actual != preprocessing_expected:
            raise ValueError("Preprocessing contract checksum does not match the manifest.")

        return cls(
            release_id=str(release_id),
            git_commit=str(git_commit),
            input_name=str(contract.get("input_name", "images")),
            output_name=str(contract.get("output_name", "embeddings")),
            input_shape=tuple(int(value) for value in input_shape),
            embedding_dim=int(contract.get("embedding_dim", 0)),
            preprocessing_path=preprocessing_path,
            preprocessing_version=preprocessing_version,
            preprocessing_sha256=preprocessing_actual,
            preprocessing=preprocessing,
            onnx_sha256=str(onnx_sha256),
            external_weights_sha256=str(external_weights_sha256),
        )


def verify_serving_model_artifacts(
    contract: ReleaseContract,
    model_repository: Path,
    model_name: str,
    model_version: str,
) -> None:
    model_root = model_repository / model_name / model_version
    artifacts = (
        (model_root / "model.onnx", contract.onnx_sha256, "ONNX graph"),
        (
            model_root / "model_embedding.onnx.data",
            contract.external_weights_sha256,
            "ONNX external weights",
        ),
    )
    for path, expected, label in artifacts:
        if not path.is_file():
            raise ValueError(f"Serving {label} is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Serving {label} checksum mismatch: expected {expected}, got {actual}."
            )
