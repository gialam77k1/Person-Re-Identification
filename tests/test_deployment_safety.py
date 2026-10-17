from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.api.config import ApiSettings, ReleaseContract, sha256
from src.api.image_processing import preprocess_image
from src.api.service import InferenceService
from src.triton_infer import preprocess_image as legacy_preprocess_image
from tests.test_inference_api import FakeQdrant, FakeTriton


class DeploymentSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = ApiSettings.from_env()
        cls.contract = ReleaseContract.load(cls.settings.model_manifest_path)

    def test_contract_preprocessing_matches_triton_preprocessing(self) -> None:
        image = Image.new("RGB", (79, 163))
        pixels = np.asarray(image).copy()
        pixels[:, :, 0] = np.arange(79, dtype=np.uint8)[None, :]
        pixels[:, :, 1] = np.arange(163, dtype=np.uint8)[:, None]
        pixels[:, :, 2] = 127
        image = Image.fromarray(pixels)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "fixture.png"
            image.save(image_path)
            expected = legacy_preprocess_image(
                image_path,
                input_height=224,
                input_width=224,
                preserve_aspect_ratio=True,
            )
        actual = preprocess_image(image, self.contract.preprocessing)
        self.assertEqual(actual.shape, (1, 3, 224, 224))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.isfinite(actual)))

    def test_service_rejects_tampered_serving_model_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            model_root = repository / "reid_embedding" / "1"
            model_root.mkdir(parents=True)
            graph = model_root / "model.onnx"
            weights = model_root / "model_embedding.onnx.data"
            graph.write_bytes(b"known-graph")
            weights.write_bytes(b"known-weights")
            contract = replace(
                self.contract,
                onnx_sha256=sha256(graph),
                external_weights_sha256=sha256(weights),
            )
            settings = replace(self.settings, serving_model_repository=repository)

            InferenceService(
                settings,
                triton=FakeTriton(),
                qdrant=FakeQdrant(),
                contract=contract,
            )
            graph.write_bytes(b"tampered-graph")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                InferenceService(
                    settings,
                    triton=FakeTriton(),
                    qdrant=FakeQdrant(),
                    contract=contract,
                )


if __name__ == "__main__":
    unittest.main()
