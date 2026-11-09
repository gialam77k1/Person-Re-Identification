from __future__ import annotations

import hashlib
import json
import unittest

from src.verify_approved_decision import canonical_sha256, verify_approved_decision


class VerifyApprovedDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {"cycle_id": "cycle-1", "workflow_run_id": "123"}
        self.manifest_sha = hashlib.sha256(json.dumps(self.manifest).encode()).hexdigest()
        self.decision = {
            "status": "RETRAIN_APPROVED",
            "cycle_id": "cycle-1",
            "base_release_id": "model:v1",
            "target_dataset": "dukemtmc-reid",
            "target_epochs": 30,
            "approved_by": "reviewer",
            "approved_at": "2026-11-06T10:00:00+00:00",
            "evidence": {"manifest_sha256": self.manifest_sha, "workflow_run_id": "123"},
        }

    def test_accepts_decision_bound_to_workflow_and_evidence(self) -> None:
        verify_approved_decision(
            self.decision,
            self.manifest,
            decision_sha256=canonical_sha256(self.decision),
            manifest_sha256=self.manifest_sha,
            cycle_id="cycle-1",
            source_release="model:v1",
            dataset="dukemtmc-reid",
            target_epochs=30,
        )

    def test_rejects_boolean_or_tampered_approval(self) -> None:
        tampered = {**self.decision, "status": "REVIEW_REQUIRED"}
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_approved_decision(
                tampered,
                self.manifest,
                decision_sha256=canonical_sha256(self.decision),
                manifest_sha256=self.manifest_sha,
                cycle_id="cycle-1",
                source_release="model:v1",
                dataset="dukemtmc-reid",
                target_epochs=30,
            )


if __name__ == "__main__":
    unittest.main()
