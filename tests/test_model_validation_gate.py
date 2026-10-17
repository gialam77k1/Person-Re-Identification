from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.model_validation_gate import ROOT, build_validation
from src.prepare_triton_model import require_passed_validation
from src.publish_model_release import validation_allows_promotion


class ModelValidationGateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = ROOT / "tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="validation-gate-", dir=temp_root)
        self.root = Path(self.temp_dir.name)

        self.paths = {
            name: self.root / name
            for name in (
                "data.json",
                "training.json",
                "evaluation.json",
                "onnx_manifest.json",
                "parity.json",
                "reference.json",
                "deployment.json",
                "preprocessing.json",
                "checkpoint.pth",
                "model.onnx",
                "report.json",
                "manifest.json",
            )
        }
        self._write_json(self.paths["data.json"], {"passed": True, "dataset_dvc_revision": "dataset.dir"})
        self._write_json(self.paths["training.json"], {"best_epoch": 48})
        self._write_json(
            self.paths["evaluation.json"],
            {
                "rank1": 0.92,
                "rank5": 0.96,
                "mAP": 0.88,
                "mINP": 0.74,
                "rank1_base": 0.915,
                "mAP_base": 0.805,
                "mINP_base": 0.52,
            },
        )
        self._write_json(
            self.paths["onnx_manifest.json"],
            {"input_shape": [1, 3, 224, 224], "embedding_dim": 512},
        )
        self._write_json(
            self.paths["parity.json"],
            {
                "passed": True,
                "min_cosine_similarity": 0.99999,
                "max_abs_error": 0.00001,
                "input_name": "images",
                "output_name": "embeddings",
                "output_shape": [1, 512],
            },
        )
        self._write_json(
            self.paths["deployment.json"],
            {"rank1": 0.912, "mAP": 0.801, "mINP": 0.51},
        )
        self._write_json(
            self.paths["preprocessing.json"],
            {"input": {"name": "images", "shape": [1, 3, 224, 224]}},
        )
        self.paths["checkpoint.pth"].write_bytes(b"checkpoint")
        self.paths["model.onnx"].write_bytes(b"onnx")
        (self.root / "model.onnx.data").write_bytes(b"weights")
        self._write_reference(np.ones((3, 512), dtype=np.float32))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _write_reference(self, embeddings: np.ndarray) -> None:
        embeddings_path = self.root / "embeddings.npy"
        pids_path = self.root / "pids.npy"
        camids_path = self.root / "camids.npy"
        paths_path = self.root / "paths.npy"
        np.save(embeddings_path, embeddings)
        np.save(pids_path, np.arange(embeddings.shape[0], dtype=np.int32))
        np.save(camids_path, np.zeros(embeddings.shape[0], dtype=np.int32))
        np.save(paths_path, np.asarray(["image.jpg"] * embeddings.shape[0]))
        self._write_json(
            self.paths["reference.json"],
            {
                "embeddings_path": str(embeddings_path),
                "pids_path": str(pids_path),
                "camids_path": str(camids_path),
                "paths_path": str(paths_path),
                "num_samples": embeddings.shape[0],
                "feature_dim": embeddings.shape[1],
            },
        )

    def _run_gate(self) -> tuple[dict, dict]:
        return build_validation(
            release_id="test:v1",
            thresholds_path=ROOT / "configs/model_validation.yaml",
            data_manifest_path=self.paths["data.json"],
            training_metrics_path=self.paths["training.json"],
            evaluation_metrics_path=self.paths["evaluation.json"],
            onnx_manifest_path=self.paths["onnx_manifest.json"],
            parity_report_path=self.paths["parity.json"],
            reference_manifest_path=self.paths["reference.json"],
            checkpoint_path=self.paths["checkpoint.pth"],
            onnx_path=self.paths["model.onnx"],
            preprocessing_path=self.paths["preprocessing.json"],
            deployment_metrics_path=self.paths["deployment.json"],
            report_path=self.paths["report.json"],
            manifest_path=self.paths["manifest.json"],
        )

    def test_gate_passes_valid_candidate(self) -> None:
        report, manifest = self._run_gate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(manifest["release"]["status"], "validated")
        self.assertEqual(len(manifest["files"]), 8)

    def test_gate_reports_quality_and_non_finite_embedding_failures(self) -> None:
        evaluation = json.loads(self.paths["evaluation.json"].read_text(encoding="utf-8"))
        evaluation["mAP"] = 0.79
        self._write_json(self.paths["evaluation.json"], evaluation)
        invalid = np.ones((3, 512), dtype=np.float32)
        invalid[0, 0] = np.nan
        self._write_reference(invalid)

        report, manifest = self._run_gate()

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("quality_mAP", report["failed_checks"])
        self.assertIn("embedding_finite", report["failed_checks"])
        self.assertEqual(manifest["release"]["status"], "rejected")

    def test_packaging_rejects_failed_gate_report(self) -> None:
        failed_report = self.root / "failed_report.json"
        self._write_json(
            failed_report,
            {"status": "FAIL", "passed": False, "failed_checks": ["quality_mAP"]},
        )
        with self.assertRaisesRegex(RuntimeError, "packaging blocked: quality_mAP"):
            require_passed_validation(str(failed_report))

    def test_packaging_accepts_passed_gate_report(self) -> None:
        passed_report = self.root / "passed_report.json"
        self._write_json(
            passed_report,
            {"release_id": "test:v1", "status": "PASS", "passed": True},
        )
        report = require_passed_validation(str(passed_report))
        self.assertEqual(report["status"], "PASS")

    def test_promotion_requires_matching_pass_report(self) -> None:
        manifest = {"release": {"release_id": "test:v1"}}
        self.assertTrue(
            validation_allows_promotion(
                manifest,
                {"release_id": "test:v1", "status": "PASS", "passed": True},
            )
        )
        self.assertFalse(
            validation_allows_promotion(
                manifest,
                {"release_id": "test:v1", "status": "FAIL", "passed": False},
            )
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validation_allows_promotion(
                manifest,
                {"release_id": "other:v1", "status": "PASS", "passed": True},
            )


if __name__ == "__main__":
    unittest.main()
