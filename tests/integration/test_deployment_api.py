from __future__ import annotations

import base64
import io
import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.deployment_http import get_json, post_image_bytes


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/deployment_embedding_v1.json"
RUN_INTEGRATION = os.getenv("RUN_DEPLOYMENT_INTEGRATION") == "1"


def synthetic_png() -> bytes:
    height, width = 163, 79
    y, x = np.mgrid[:height, :width]
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = (x * 3 + y) % 256
    pixels[:, :, 1] = (x + y * 2) % 256
    pixels[:, :, 2] = (x * 7 + y * 5) % 256
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set RUN_DEPLOYMENT_INTEGRATION=1 to test the running Triton/Qdrant/API stack.",
)
class DeploymentApiIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api_url = os.getenv("DEPLOYMENT_API_URL", "http://localhost:8080").rstrip("/")
        cls.image = synthetic_png()
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        raw = base64.b64decode(cls.fixture["embedding_base64_f32_le"])
        cls.expected_embedding = np.frombuffer(raw, dtype="<f4")

    def test_health_readiness_and_model_contract(self) -> None:
        self.assertEqual(get_json(f"{self.api_url}/health")["status"], "ok")
        readiness = get_json(f"{self.api_url}/ready")
        self.assertEqual(readiness["status"], "ready")
        self.assertTrue(all(readiness["checks"].values()))
        info = get_json(f"{self.api_url}/model-info")
        self.assertEqual(info["release_id"], self.fixture["release_id"])
        self.assertEqual(info["git_commit"], self.fixture["git_commit"])
        self.assertEqual(info["preprocessing_sha256"], self.fixture["preprocessing_sha256"])
        self.assertEqual(info["embedding_dim"], 512)

    def test_embedding_is_finite_normalized_and_stable(self) -> None:
        embeddings = []
        for _ in range(3):
            response = post_image_bytes(
                f"{self.api_url}/embed",
                self.image,
                filename="deployment-regression.png",
                content_type="image/png",
            )
            embedding = np.asarray(response["embedding"], dtype=np.float32)
            self.assertEqual(embedding.shape, (512,))
            self.assertTrue(np.all(np.isfinite(embedding)))
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
            embeddings.append(embedding)

        for embedding in embeddings:
            cosine = float(np.dot(embedding, self.expected_embedding))
            max_abs_error = float(np.max(np.abs(embedding - self.expected_embedding)))
            self.assertGreaterEqual(cosine, 0.9999)
            self.assertLessEqual(max_abs_error, 0.0001)
        np.testing.assert_allclose(embeddings[0], embeddings[1], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(embeddings[1], embeddings[2], rtol=0.0, atol=1e-7)

    def test_search_returns_exact_qdrant_top_k_contract(self) -> None:
        response = post_image_bytes(
            f"{self.api_url}/search",
            self.image,
            filename="deployment-search.png",
            content_type="image/png",
            query={"top_k": 5},
        )
        self.assertEqual(response["model_release"], self.fixture["release_id"])
        self.assertEqual(response["top_k"], 5)
        self.assertEqual(len(response["results"]), 5)
        for result in response["results"]:
            self.assertIsInstance(result["person_id"], int)
            self.assertIsInstance(result["camera_id"], int)
            self.assertTrue(np.isfinite(result["similarity_score"]))
            self.assertTrue(result["source_image"])


if __name__ == "__main__":
    unittest.main()
