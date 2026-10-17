from __future__ import annotations

from typing import Any

import numpy as np

from src.api.clients import QdrantClient, TritonClient, UpstreamError
from src.api.config import ApiSettings, ReleaseContract
from src.api.image_processing import preprocess_image, validate_and_decode_image


class InferenceService:
    def __init__(
        self,
        settings: ApiSettings,
        *,
        triton: TritonClient | None = None,
        qdrant: QdrantClient | None = None,
        contract: ReleaseContract | None = None,
    ) -> None:
        self.settings = settings
        self.contract = contract or ReleaseContract.load(settings.model_manifest_path)
        self.triton = triton or TritonClient(
            settings.triton_url,
            settings.model_name,
            settings.upstream_timeout_seconds,
        )
        self.qdrant = qdrant or QdrantClient(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.upstream_timeout_seconds,
        )
        if self.contract.embedding_dim <= 0:
            raise ValueError("Model manifest has an invalid embedding dimension.")

    def model_info(self) -> dict[str, Any]:
        return {
            "release_id": self.contract.release_id,
            "git_commit": self.contract.git_commit,
            "model_name": self.settings.model_name,
            "embedding_dim": self.contract.embedding_dim,
            "input_shape": list(self.contract.input_shape),
            "preprocessing_version": self.contract.preprocessing_version,
            "preprocessing_sha256": self.contract.preprocessing_sha256,
            "qdrant_collection": self.settings.qdrant_collection,
        }

    def readiness(self) -> dict[str, bool]:
        checks = {
            "manifest": True,
            "triton_server": self.triton.server_ready(),
            "triton_model": self.triton.model_ready(),
            "qdrant_collection": False,
        }
        try:
            collection = self.qdrant.collection_info()
            vectors = (
                collection.get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
            vector_size = vectors.get("size") if isinstance(vectors, dict) else None
            points_count = int(collection.get("points_count", 0))
            checks["qdrant_collection"] = (
                collection.get("status") == "green"
                and vector_size == self.contract.embedding_dim
                and points_count > 0
            )
        except (UpstreamError, TypeError, ValueError):
            checks["qdrant_collection"] = False
        return checks

    def embed(self, raw: bytes, content_type: str | None) -> np.ndarray:
        image = validate_and_decode_image(
            raw,
            content_type,
            max_upload_bytes=self.settings.max_upload_bytes,
            max_image_pixels=self.settings.max_image_pixels,
        )
        tensor = preprocess_image(image, self.contract.preprocessing)
        if tuple(tensor.shape) != self.contract.input_shape:
            raise ValueError(
                f"Preprocessing produced {tuple(tensor.shape)}, expected {self.contract.input_shape}."
            )
        embedding = self.triton.infer(
            tensor,
            self.contract.input_name,
            self.contract.output_name,
        ).reshape(-1)
        if embedding.shape != (self.contract.embedding_dim,):
            raise UpstreamError(
                f"Triton returned embedding shape {embedding.shape}; "
                f"expected ({self.contract.embedding_dim},)."
            )
        if not np.all(np.isfinite(embedding)):
            raise UpstreamError("Triton returned NaN or Inf in the embedding.")
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise UpstreamError("Triton returned an embedding with invalid norm.")
        return (embedding / norm).astype(np.float32)

    def search(
        self, raw: bytes, content_type: str | None, top_k: int
    ) -> list[dict[str, Any]]:
        embedding = self.embed(raw, content_type)
        points = self.qdrant.search(embedding, top_k)
        results: list[dict[str, Any]] = []
        for point in points:
            payload = point.get("payload") or {}
            person_id = payload.get("pid", payload.get("person_id"))
            camera_id = payload.get("camid", payload.get("camera_id"))
            if person_id is None or camera_id is None or "score" not in point:
                raise UpstreamError("Qdrant result is missing pid, camid, or score.")
            source = payload.get("image_path") or payload.get("source_image")
            if not source:
                source = f"qdrant-point:{point.get('id', 'unknown')}"
            results.append(
                {
                    "person_id": int(person_id),
                    "camera_id": int(camera_id),
                    "similarity_score": float(point["score"]),
                    "source_image": str(source),
                }
            )
        return results
