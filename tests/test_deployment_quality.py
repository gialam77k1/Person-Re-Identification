from __future__ import annotations

import unittest

from src.benchmark_inference_api import percentile
from src.validate_deployment_quality import build_quality_report


class DeploymentQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.offline = {
            "dataset": "market1501",
            "valid_queries": 3368,
            "rank1_base": 0.91,
            "flip_test": True,
        }
        self.deployment = {
            "dataset": "market1501",
            "valid_queries": 3368,
            "rank1": 0.907,
            "embedding_dim": 512,
            "embeddings_finite": True,
            "embeddings_l2_normalized": True,
            "protocol": "serving-equivalent",
        }

    def test_quality_gate_passes_within_half_percent_drop(self) -> None:
        report = build_quality_report(
            self.offline, self.deployment, rank1_max_drop=0.005, embedding_dim=512
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["failed_checks"], [])

    def test_quality_gate_fails_rank1_shape_and_finite_checks(self) -> None:
        deployment = dict(
            self.deployment,
            rank1=0.90,
            embedding_dim=256,
            embeddings_finite=False,
        )
        report = build_quality_report(
            self.offline, deployment, rank1_max_drop=0.005, embedding_dim=512
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(
            set(report["failed_checks"]),
            {"rank1_drop", "embedding_dim", "embeddings_finite"},
        )

    def test_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)


if __name__ == "__main__":
    unittest.main()
