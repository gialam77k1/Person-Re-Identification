from __future__ import annotations

import math
import unittest

from src.catastrophic_forgetting_gate import build_forgetting_gate
from src.log_v2_evaluations_mlflow import validate_bundle


class CatastrophicForgettingGateTest(unittest.TestCase):
    def policy(self) -> dict:
        return {
            "release_id": "model:v2",
            "duke_improvement": {
                "rank1_min_absolute_gain": 0.05,
                "map_min_absolute_gain": 0.03,
            },
            "market1501_retention": {
                "baseline_protocol": "reranked",
                "rank1_max_absolute_drop": 0.05,
                "rank5_max_absolute_drop": 0.03,
                "map_max_absolute_drop": 0.08,
                "minp_max_absolute_drop": 0.10,
            },
            "minimum_counts": {"query": 1, "gallery": 1, "valid_queries": 1},
        }

    def candidate(self, dataset: str, metrics: dict) -> dict:
        return {
            "release_id": "model:v2",
            "dataset": dataset,
            "checkpoint_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "dataset_dvc_revision": f"{dataset}.dir",
            "query_count": 100,
            "gallery_count": 200,
            "valid_queries": 100,
            **metrics,
        }

    def build(self, market_rank1: float = 0.89, msmt_map: float = 0.25):
        return build_forgetting_gate(
            duke_v1={"rank1": 0.10, "rank5": 0.20, "mAP": 0.10, "mINP": 0.05},
            duke_v2=self.candidate(
                "dukemtmc-reid",
                {"rank1": 0.25, "rank5": 0.40, "mAP": msmt_map, "mINP": 0.15},
            ),
            market1501_v1_manifest={
                "quality_metrics": {
                    "reranked": {"rank1": 0.92, "rank5": 0.96, "mAP": 0.88, "mINP": 0.74}
                }
            },
            market1501_v2=self.candidate(
                "market1501", {"rank1": market_rank1, "rank5": 0.94, "mAP": 0.83, "mINP": 0.67}
            ),
            policy=self.policy(),
            policy_sha256="c" * 64,
            mlflow_run_id="run-1",
        )

    def test_passes_clear_improvement_and_bounded_forgetting(self) -> None:
        report = self.build()
        self.assertTrue(report["passed"])
        self.assertEqual(report["policy_sha256"], "c" * 64)

    def test_rejects_excessive_market1501_drop(self) -> None:
        report = self.build(market_rank1=0.80)
        self.assertFalse(report["passed"])
        self.assertIn("market1501_rank1_retention", report["failed_checks"])

    def test_rejects_non_finite_metrics(self) -> None:
        report = self.build(msmt_map=math.nan)
        self.assertFalse(report["passed"])
        self.assertIn("finite_duke_v2", report["failed_checks"])

    def test_mlflow_bundle_must_share_release_and_finished_run(self) -> None:
        run_id, release = validate_bundle(
            {"release_id": "model:v2"},
            {"release_id": "model:v2"},
            {"release_id": "model:v2", "mlflow_run_id": "run-1"},
            {"run_id": "run-1", "tracking_uri": "http://mlflow", "status": "FINISHED"},
        )
        self.assertEqual((run_id, release), ("run-1", "model:v2"))


if __name__ == "__main__":
    unittest.main()
