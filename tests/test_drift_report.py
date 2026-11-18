from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.api.drift_sampling import EmbeddingSampler
from src.generate_drift_report import (
    build_drift_report,
    build_insufficient_data_report,
    load_current_embeddings,
    load_embeddings,
    model_artifact,
    prometheus_text,
)


class DriftReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "minimum_current_samples": 10,
            "thresholds": {
                "centroid_cosine_distance_max": 0.05,
                "standardized_mean_shift_max": 0.20,
                "norm_mean_delta_max": 0.05,
                "ks_statistic_max": 0.15,
            },
        }
        generator = np.random.default_rng(42)
        values = generator.normal(size=(200, 16))
        self.reference = values / np.linalg.norm(values, axis=1, keepdims=True)

    def test_identical_distribution_passes(self) -> None:
        report = build_drift_report(
            self.reference,
            self.reference.copy(),
            "model:v1",
            self.policy,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["drift_detected"])
        self.assertIn("reid_embedding_drift_detected", prometheus_text(report))
        self.assertIn("reid_embedding_drift_threshold", prometheus_text(report))
        self.assertEqual(report["thresholds"]["ks_statistic_max"], 0.15)

    def test_shifted_distribution_detects_drift(self) -> None:
        current = np.zeros((100, 16), dtype=np.float64)
        current[:, 0] = 1.0
        report = build_drift_report(self.reference, current, "model:v1", self.policy)
        self.assertEqual(report["status"], "DRIFT")
        self.assertTrue(report["drift_detected"])

    def test_sampler_writes_private_jsonl_consumable_by_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.jsonl"
            sampler = EmbeddingSampler(path, sample_rate=1.0, max_samples=10)
            sampler.observe(np.ones(16, dtype=np.float32), "model:v1")
            loaded = load_embeddings(path)
            self.assertEqual(loaded.shape, (1, 16))
            row = json.loads(path.read_text().strip())
            self.assertEqual(row["release_id"], "model:v1")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_missing_runtime_samples_are_a_successful_waiting_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.jsonl"
            current = load_current_embeddings(path, "model:v2", 16)
            report = build_insufficient_data_report(
                self.reference, current, "model:v2", self.policy
            )
            metrics = prometheus_text(report)

            self.assertEqual(report["status"], "INSUFFICIENT_DATA")
            self.assertIsNone(report["drift_detected"])
            self.assertEqual(report["samples_remaining"], 10)
            self.assertIn('state="INSUFFICIENT_DATA"', metrics)
            self.assertIn("reid_embedding_drift_ready", metrics)
            self.assertNotIn("reid_embedding_drift_detected{", metrics)

    def test_runtime_samples_are_filtered_by_active_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.jsonl"
            sampler = EmbeddingSampler(path, sample_rate=1.0, max_samples=10)
            sampler.observe(np.ones(16, dtype=np.float32), "model:v1")
            sampler.observe(np.ones(16, dtype=np.float32) * 2, "model:v2")

            loaded = load_current_embeddings(path, "model:v2", 16)
            self.assertEqual(loaded.shape, (1, 16))

    def test_sampler_rotates_window_when_serving_release_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current_embeddings.jsonl"
            first = EmbeddingSampler(
                path, sample_rate=1.0, max_samples=10, release_id="model:v2"
            )
            first.observe(np.ones(16, dtype=np.float32), "model:v2")

            second = EmbeddingSampler(
                path, sample_rate=1.0, max_samples=10, release_id="model:v3"
            )
            self.assertFalse(path.exists())
            archived = list((Path(directory) / "archive" / "model-v2").glob("*.jsonl"))
            self.assertEqual(len(archived), 1)
            marker = json.loads(
                (Path(directory) / ".active_release.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["release_id"], "model:v3")

            second.observe(np.ones(16, dtype=np.float32), "model:v3")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual({row["release_id"] for row in rows}, {"model:v3"})

    def test_sampler_migrates_a_legacy_window_without_release_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current_embeddings.jsonl"
            path.write_text(
                json.dumps({"release_id": "model:v2", "embedding": [1.0] * 16})
                + "\n",
                encoding="utf-8",
            )

            EmbeddingSampler(
                path, sample_rate=1.0, max_samples=10, release_id="model:v3"
            )

            self.assertFalse(path.exists())
            self.assertEqual(
                len(list((Path(directory) / "archive" / "model-v2").glob("*.jsonl"))),
                1,
            )

    def test_report_contains_reproducible_provenance_and_model_checksum(self) -> None:
        metadata = {
            "release_id": "model:v1",
            "git_commit": "abc123",
            "dataset": "dukemtmc-reid",
            "dataset_dvc_revision": "dataset.dir",
            "split": "query",
            "sample_seed": 42,
            "num_samples": 200,
        }
        model = model_artifact(
            {
                "release": {"release_id": "model:v1"},
                "files": [{"role": "onnx_graph", "repository_path": "model.onnx", "sha256": "a" * 64}],
            },
            "model:v1",
        )
        report = build_drift_report(
            self.reference,
            self.reference.copy(),
            "model:v1",
            self.policy,
            metadata=metadata,
            model=model,
            reference_sha256="b" * 64,
            current_sha256="c" * 64,
        )
        self.assertEqual(report["provenance"]["sample_seed"], 42)
        self.assertEqual(report["provenance"]["dataset_dvc_revision"], "dataset.dir")
        self.assertEqual(report["model_artifact"]["sha256"], "a" * 64)
        self.assertEqual(report["inputs"]["current_embeddings_sha256"], "c" * 64)


if __name__ == "__main__":
    unittest.main()
