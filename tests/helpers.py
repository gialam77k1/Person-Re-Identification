from __future__ import annotations

import json

from src.api.config import ROOT, ReleaseContract, sha256


def build_test_contract() -> ReleaseContract:
    preprocessing_path = ROOT / "model_releases/market1501-vit-bnneck/v1/preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    return ReleaseContract(
        release_id="market1501-vit-bnneck:v1-dvc",
        git_commit="3bc5cd9e68d4b186900ba52e89976e9c94be28cc",
        input_name="images",
        output_name="embeddings",
        input_shape=(1, 3, 224, 224),
        embedding_dim=512,
        preprocessing_path=preprocessing_path,
        preprocessing_version=1,
        preprocessing_sha256=sha256(preprocessing_path),
        preprocessing=preprocessing,
        onnx_sha256="0" * 64,
        external_weights_sha256="1" * 64,
    )
