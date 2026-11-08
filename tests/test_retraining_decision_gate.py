from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.retraining_decision_gate import build_retraining_decision


ROOT = Path(__file__).resolve().parents[1]


class RetrainingDecisionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = yaml.safe_load((ROOT / "configs/retraining_decision.yaml").read_text(encoding="utf-8"))
        self.drift = {
            "release_id": "market1501-vit-bnneck:v1-dvc",
            "drift_detected": True,
        }
        self.evaluation = {
            "release_id": "market1501-vit-bnneck:v1-dvc",
            "dataset": "msmt17",
            "dataset_dvc_revision": "dataset-revision.dir",
            "num_query_samples": 11659,
            "num_gallery_samples": 82161,
            "query_coverage": 1.0,
            "embedding_dim": 512,
            "embeddings_finite": True,
            "rank1": 0.40,
            "mAP": 0.20,
        }

    def test_retraining_signal_requires_manual_review(self) -> None:
        result = build_retraining_decision(self.drift, self.evaluation, self.policy)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(result["retraining_signal"])

    def test_named_approval_allows_thirty_epoch_retraining(self) -> None:
        result = build_retraining_decision(self.drift, self.evaluation, self.policy, approved_by="gialam77k1")
        self.assertEqual(result["status"], "RETRAIN_APPROVED")
        self.assertEqual(result["target_epochs"], 30)

    def test_no_drift_does_not_request_retraining(self) -> None:
        self.drift["drift_detected"] = False
        result = build_retraining_decision(self.drift, self.evaluation, self.policy, approved_by="gialam77k1")
        self.assertEqual(result["status"], "NO_RETRAIN")

    def test_smoke_evaluation_is_insufficient_evidence(self) -> None:
        self.evaluation["num_query_samples"] = 100
        result = build_retraining_decision(self.drift, self.evaluation, self.policy, approved_by="gialam77k1")
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
