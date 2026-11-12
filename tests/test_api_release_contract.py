from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.api.config import ReleaseContract


class ApiReleaseContractTest(unittest.TestCase):
    def write_manifest(
        self,
        root: Path,
        model_embedding_dim: int,
        contract_embedding_dim: int | None = None,
    ) -> Path:
        preprocessing = root / "preprocessing.json"
        preprocessing.write_text(
            json.dumps({"schema_version": 1, "normalization": {}}),
            encoding="utf-8",
        )
        preprocessing_sha = hashlib.sha256(preprocessing.read_bytes()).hexdigest()
        manifest = root / "model_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "release": {"release_id": "model:v2"},
                    "source": {"git_commit": "a" * 40},
                    "model": {"embedding_dim": model_embedding_dim},
                    "contract": {
                        "input_name": "images",
                        "input_shape": [1, 3, 224, 224],
                        "output_name": "embeddings",
                        "preprocessing_path": str(preprocessing),
                        **(
                            {"embedding_dim": contract_embedding_dim}
                            if contract_embedding_dim is not None
                            else {}
                        ),
                    },
                    "validation": {"status": "PASS"},
                    "files": [
                        {"role": "onnx_graph", "sha256": "b" * 64},
                        {"role": "onnx_external_weights", "sha256": "c" * 64},
                        {"role": "preprocessing_contract", "sha256": preprocessing_sha},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_legacy_v2_manifest_uses_validated_model_embedding_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract = ReleaseContract.load(self.write_manifest(Path(directory), 512))
            self.assertEqual(contract.embedding_dim, 512)

    def test_manifest_without_embedding_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "embedding dimension"):
                ReleaseContract.load(self.write_manifest(Path(directory), 0))

    def test_conflicting_model_and_contract_dimensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "do not match"):
                ReleaseContract.load(self.write_manifest(Path(directory), 512, 256))


if __name__ == "__main__":
    unittest.main()
