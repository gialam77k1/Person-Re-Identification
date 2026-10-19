from __future__ import annotations

import io
import unittest
from dataclasses import replace
from typing import Any

import httpx2
import numpy as np
from PIL import Image

from src.api.config import ApiSettings
from src.api.main import create_app
from src.api.service import InferenceService
from tests.helpers import build_test_contract


class FakeTriton:
    def __init__(self) -> None:
        self.server_is_ready = True
        self.model_is_ready = True
        self.last_tensor: np.ndarray | None = None

    def server_ready(self) -> bool:
        return self.server_is_ready

    def model_ready(self) -> bool:
        return self.model_is_ready

    def infer(self, tensor: np.ndarray, input_name: str, output_name: str) -> np.ndarray:
        self.last_tensor = tensor
        embedding = np.arange(1, 513, dtype=np.float32)
        return embedding[None, :]


class FakeQdrant:
    def __init__(self) -> None:
        self.status = "green"
        self.points_count = 2
        self.vector_size = 512
        self.last_vector: np.ndarray | None = None
        self.last_limit: int | None = None

    def collection_info(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "points_count": self.points_count,
            "config": {"params": {"vectors": {"size": self.vector_size}}},
        }

    def search(self, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        self.last_vector = vector
        self.last_limit = limit
        points = [
            {
                "id": 11,
                "score": 0.98,
                "payload": {"pid": 150, "camid": 2, "image_path": "gallery/0150_c2.jpg"},
            },
            {
                "id": 12,
                "score": 0.91,
                "payload": {"pid": 151, "camid": 5},
            },
        ]
        return points[:limit]


def image_bytes(image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 160), color=(20, 80, 140)).save(buffer, format=image_format)
    return buffer.getvalue()


class InferenceApiTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_test_contract()

    def setUp(self) -> None:
        settings = replace(
            ApiSettings.from_env(),
            max_upload_bytes=1024,
            max_image_pixels=1_000_000,
            default_top_k=2,
            max_top_k=10,
        )
        self.triton = FakeTriton()
        self.qdrant = FakeQdrant()
        self.service = InferenceService(
            settings,
            triton=self.triton,
            qdrant=self.qdrant,
            contract=self.contract,
            verify_model_checksum=False,
        )
        self.app = create_app(self.service)

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx2.Response:
        transport = httpx2.ASGITransport(app=self.app)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    async def test_health_ready_and_model_info(self) -> None:
        health = await self.request("GET", "/health")
        self.assertEqual(health.json(), {"status": "ok"})

        ready = await self.request("GET", "/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertTrue(all(ready.json()["checks"].values()))

        model_info = await self.request("GET", "/model-info")
        self.assertEqual(model_info.status_code, 200)
        body = model_info.json()
        self.assertEqual(body["release_id"], "market1501-vit-bnneck:v1-dvc")
        self.assertEqual(body["git_commit"], self.contract.git_commit)
        self.assertEqual(body["preprocessing_version"], 1)
        self.assertEqual(body["embedding_dim"], 512)

    async def test_ready_fails_when_model_or_collection_is_not_ready(self) -> None:
        self.triton.model_is_ready = False
        response = await self.request("GET", "/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["checks"]["triton_model"])

        self.triton.model_is_ready = True
        self.qdrant.points_count = 0
        response = await self.request("GET", "/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["checks"]["qdrant_collection"])

    async def test_embed_accepts_jpeg_and_returns_normalized_vector(self) -> None:
        response = await self.request(
            "POST",
            "/embed",
            files={"image": ("query.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_release"], self.contract.release_id)
        self.assertEqual(body["embedding_dim"], 512)
        self.assertEqual(len(body["embedding"]), 512)
        self.assertAlmostEqual(float(np.linalg.norm(body["embedding"])), 1.0, places=6)
        self.assertEqual(tuple(self.triton.last_tensor.shape), (1, 3, 224, 224))

    async def test_search_returns_qdrant_top_k_with_required_metadata(self) -> None:
        response = await self.request(
            "POST",
            "/search?top_k=2",
            files={"image": ("query.png", image_bytes("PNG"), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_release"], self.contract.release_id)
        self.assertEqual(body["top_k"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["person_id"], 150)
        self.assertEqual(body["results"][0]["camera_id"], 2)
        self.assertEqual(body["results"][0]["source_image"], "gallery/0150_c2.jpg")
        self.assertEqual(body["results"][1]["source_image"], "qdrant-point:12")
        self.assertEqual(self.qdrant.last_limit, 2)
        self.assertAlmostEqual(float(np.linalg.norm(self.qdrant.last_vector)), 1.0, places=6)

    async def test_rejects_invalid_mime_corrupt_image_and_oversized_upload(self) -> None:
        wrong_mime = await self.request(
            "POST",
            "/embed",
            files={"image": ("query.gif", b"GIF89a", "image/gif")},
        )
        self.assertEqual(wrong_mime.status_code, 400)

        corrupt = await self.request(
            "POST",
            "/embed",
            files={"image": ("query.jpg", b"not-an-image", "image/jpeg")},
        )
        self.assertEqual(corrupt.status_code, 400)

        oversized = await self.request(
            "POST",
            "/embed",
            files={"image": ("query.jpg", b"x" * 1025, "image/jpeg")},
        )
        self.assertEqual(oversized.status_code, 413)

    async def test_rejects_top_k_above_limit(self) -> None:
        response = await self.request(
            "POST",
            "/search?top_k=11",
            files={"image": ("query.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
