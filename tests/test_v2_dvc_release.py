from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.validate_v2_dvc_release import build_release, sha256_file


class V2DVCReleaseTest(unittest.TestCase):
    def write_json(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_manifest_contains_required_release_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"checkpoint")
            onnx = root / "model.onnx"
            onnx.write_bytes(b"onnx")
            (root / "model.onnx.data").write_bytes(b"weights")
            arrays = {
                "reference_embeddings": ("emb.npy", np.tile(np.eye(1, 512, dtype=np.float32), (10, 1))),
                "reference_pids": ("pids.npy", np.arange(10)),
                "reference_camids": ("cam.npy", np.zeros(10, dtype=np.int64)),
                "reference_paths": ("paths.npy", np.asarray([f"image-{index}.jpg" for index in range(10)])),
            }
            refs = []
            for role, (name, value) in arrays.items():
                path = root / name
                np.save(path, value)
                refs.append((role, path))
            checkpoint_manifest = self.write_json(root, "checkpoint.json", {"release_id": "market1501-duke-vit-bnneck:v2", "checkpoint_sha256": sha256_file(checkpoint), "mlflow_run_id": "run-1", "training_config_sha256": "c" * 64, "git_commit": "git", "training_git_commit": "train-git", "best_epoch": 9, "dataset_revisions": {"duke": "msmt.dir", "market1501": "market.dir"}})
            data = self.write_json(root, "data.json", {"passed": True})
            common = {"release_id": "market1501-duke-vit-bnneck:v2", "checkpoint_sha256": sha256_file(checkpoint)}
            msmt = self.write_json(root, "msmt.json", {**common, "rank1": 0.4, "rank5": 0.6, "mAP": 0.3, "mINP": 0.2})
            market = self.write_json(root, "market.json", {**common, "rank1": 0.9, "rank5": 0.95, "mAP": 0.85, "mINP": 0.7})
            forgetting = self.write_json(root, "forget.json", {"status": "PASS", "passed": True})
            onnx_manifest = self.write_json(root, "onnx.json", {"release_id": "market1501-duke-vit-bnneck:v2", "input_name": "images", "input_dtype": "float32", "input_shape": [1, 3, 224, 224], "output_name": "embeddings", "output_dtype": "float32", "embedding_dim": 512, "preserve_aspect_ratio": True, "files": {"graph": {"sha256": sha256_file(onnx)}, "external_weights": {"sha256": sha256_file(root / "model.onnx.data")}}})
            parity = self.write_json(root, "parity.json", {"passed": True, "input_name": "images", "input_shape": ["batch_size", 3, 224, 224], "output_name": "embeddings", "output_shape": ["batch_size", 512], "min_cosine_similarity": 0.99999, "max_abs_error": 0.00001})
            reference_files = {role.removeprefix("reference_"): {"sha256": sha256_file(path)} for role, path in refs}
            reference = self.write_json(root, "reference.json", {"release_id": "market1501-duke-vit-bnneck:v2", "feature_dim": 512, "num_samples": 10, "finite": True, "l2_normalized": True, "files": reference_files})
            preprocessing = self.write_json(root, "pre.json", {"schema_version": 1, "input": {"name": "images", "dtype": "float32", "layout": "NCHW", "shape": [1, 3, 224, 224]}, "resize": {"preserve_aspect_ratio": True, "target_height": 224, "target_width": 224}, "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "inference": {"reference_embeddings_l2_normalized": True}})
            policy = root / "policy.yaml"
            policy.write_text(yaml.safe_dump({"release_id": "market1501-duke-vit-bnneck:v2", "quality": {"duke": {"rank1_min": 0.3, "rank5_min": 0.5, "map_min": 0.15, "minp_min": 0.02}, "market1501": {"rank1_min": 0.87, "rank5_min": 0.92, "map_min": 0.8, "minp_min": 0.64}}, "onnx_parity": {"cosine_min": 0.9999, "max_abs_error": 0.0001}, "contract": {"input_name": "images", "input_dtype": "float32", "input_layout": "NCHW", "input_shape": [1, 3, 224, 224], "output_name": "embeddings", "output_dtype": "float32", "embedding_dim": 512, "preserve_aspect_ratio": True, "reference_l2_norm_atol": 0.0001}}), encoding="utf-8")
            report_path, manifest_path = root / "report.json", root / "manifest.json"
            report, manifest = build_release(
                checkpoint_manifest_path=checkpoint_manifest, data_manifest_path=data, duke_evaluation_path=msmt, market1501_evaluation_path=market, forgetting_report_path=forgetting, onnx_manifest_path=onnx_manifest, parity_report_path=parity, reference_manifest_path=reference, checkpoint_path=checkpoint, onnx_path=onnx, reference_files=refs, preprocessing_source_path=preprocessing, preprocessing_output_path=root / "pre-out.json", policy_path=policy, report_path=report_path, manifest_path=manifest_path,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(manifest["source"]["dataset_revisions"]["duke"], "msmt.dir")
            self.assertEqual(manifest["source"]["mlflow_run_id"], "run-1")
            self.assertEqual(manifest["source"]["config_sha256"], "c" * 64)
            self.assertEqual(manifest["release"]["status"], "validated")
            self.assertEqual(manifest["reference"]["embedding_dim"], 512)
            self.assertEqual(manifest["contract"]["embedding_dim"], 512)
            self.assertTrue(any(check["name"] == "onnx_parity_cosine" for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()
