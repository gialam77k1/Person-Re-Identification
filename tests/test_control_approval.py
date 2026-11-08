from __future__ import annotations

import hashlib
import json
import unittest

from src.control_api.approval import ApprovalEvidenceError, create_approved_decision


class ControlApprovalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.release = "market1501-vit-bnneck:v1-dvc"
        self.drift = {"release_id": self.release, "drift_detected": True}
        self.evaluation = {
            "release_id": self.release, "dataset": "msmt17", "num_query_samples": 11659,
            "num_gallery_samples": 82161, "query_coverage": 1.0, "embedding_dim": 512,
            "embeddings_finite": True, "dataset_dvc_revision": "abc.dir", "rank1": 0.01,
            "mAP": 0.01, "mINP": 0.01,
        }
        self.policy = {
            "policy_name": "test", "candidate": {"base_release_id": self.release, "target_dataset": "msmt17", "target_dataset_version": "MSMT17_V1", "target_epochs": 50},
            "evidence": {"minimum_query_samples": 10000, "minimum_gallery_samples": 80000, "minimum_query_coverage": 0.99, "required_embedding_dim": 512, "require_drift_detected": True},
            "quality": {"retrain_when_any_below": {"rank1": 0.3, "mAP": 0.2, "mINP": 0.1}},
            "approval": {"manual_approval_required": True, "minimum_approver_name_length": 2},
        }

    def files(self) -> dict[str, bytes]:
        drift = json.dumps(self.drift).encode()
        evaluation = json.dumps(self.evaluation).encode()
        original = json.dumps({"status": "REVIEW_REQUIRED"}).encode()
        manifest = {
            "cycle_id": "cycle-1", "workflow_run_id": "123", "release_id": self.release,
            "dataset": "msmt17", "dataset_dvc_revision": "abc.dir", "source_git_commit": "deadbeef",
            "evidence": {
                "drift_report": {"path": "drift_report.json", "sha256": hashlib.sha256(drift).hexdigest()},
                "evaluation_report": {"path": "evaluation_report.json", "sha256": hashlib.sha256(evaluation).hexdigest()},
                "retraining_decision": {"path": "decision.json", "sha256": hashlib.sha256(original).hexdigest()},
            },
        }
        return {
            "evidence_manifest.json": json.dumps(manifest).encode(),
            "drift_report.json": drift, "evaluation_report.json": evaluation,
            "decision.json": original,
        }

    def test_backend_creates_approved_decision_with_auditable_evidence(self) -> None:
        decision = create_approved_decision(
            files=self.files(), policy=self.policy, cycle_id="cycle-1", actor="reviewer",
            source_release=self.release, dataset="MSMT17", evidence_run_id="123",
            evidence_artifact="mlops-evidence-cycle-1",
        )
        self.assertEqual(decision["status"], "RETRAIN_APPROVED")
        self.assertEqual(decision["approved_by"], "reviewer")
        self.assertEqual(decision["target_epochs"], 50)
        self.assertEqual(len(decision["evidence"]["manifest_sha256"]), 64)

    def test_backend_rejects_tampered_evidence(self) -> None:
        files = self.files()
        files["drift_report.json"] = b"{}"
        with self.assertRaisesRegex(ApprovalEvidenceError, "checksum"):
            create_approved_decision(
                files=files, policy=self.policy, cycle_id="cycle-1", actor="reviewer",
                source_release=self.release, dataset="MSMT17", evidence_run_id="123",
                evidence_artifact="mlops-evidence-cycle-1",
            )


if __name__ == "__main__":
    unittest.main()
