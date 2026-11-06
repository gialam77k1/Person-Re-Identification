from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.validate_v2_dvc_release import build_release, sha256_file


class V2DVCReleaseTest(unittest.TestCase):
    def write_json(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_manifest_contains_required_release_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pth"; checkpoint.write_bytes(b"checkpoint")
            onnx = root / "model.onnx"; onnx.write_bytes(b"onnx")
            (root / "model.onnx.data").write_bytes(b"weights")
            refs = []
            for role, name in (("reference_embeddings", "emb.npy"), ("reference_pids", "pids.npy"), ("reference_camids", "cam.npy"), ("reference_paths", "paths.npy")):
                path = root / name; path.write_bytes(role.encode()); refs.append((role, path))
            checkpoint_manifest = self.write_json(root, "checkpoint.json", {"release_id": "market1501-msmt17-vit-bnneck:v2", "checkpoint_sha256": sha256_file(checkpoint), "mlflow_run_id": "run-1", "training_config_sha256": "c" * 64, "git_commit": "git", "training_git_commit": "train-git", "best_epoch": 9, "dataset_revisions": {"msmt17": "msmt.dir", "market1501": "market.dir"}})
            data = self.write_json(root, "data.json", {"passed": True})
            metrics = {"release_id": "market1501-msmt17-vit-bnneck:v2", "checkpoint_sha256": sha256_file(checkpoint), "rank1": 0.4, "rank5": 0.6, "mAP": 0.3, "mINP": 0.2}
            msmt = self.write_json(root, "msmt.json", metrics)
            market = self.write_json(root, "market.json", metrics)
            forgetting = self.write_json(root, "forget.json", {"status": "PASS", "passed": True})
            onnx_manifest = self.write_json(root, "onnx.json", {"input_shape": [1, 3, 224, 224], "embedding_dim": 512})
            parity = self.write_json(root, "parity.json", {"passed": True})
            reference = self.write_json(root, "reference.json", {"feature_dim": 512, "num_samples": 10})
            preprocessing = self.write_json(root, "pre.json", {"schema_version": 1})
            report_path, manifest_path = root / "report.json", root / "manifest.json"
            report, manifest = build_release(
                checkpoint_manifest_path=checkpoint_manifest, data_manifest_path=data, msmt17_evaluation_path=msmt, market1501_evaluation_path=market, forgetting_report_path=forgetting, onnx_manifest_path=onnx_manifest, parity_report_path=parity, reference_manifest_path=reference, checkpoint_path=checkpoint, onnx_path=onnx, reference_files=refs, preprocessing_source_path=preprocessing, preprocessing_output_path=root / "pre-out.json", report_path=report_path, manifest_path=manifest_path,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(manifest["source"]["dataset_revisions"]["msmt17"], "msmt.dir")
            self.assertEqual(manifest["source"]["mlflow_run_id"], "run-1")
            self.assertEqual(manifest["source"]["config_sha256"], "c" * 64)
            self.assertEqual(manifest["release"]["status"], "validated")


if __name__ == "__main__":
    unittest.main()
