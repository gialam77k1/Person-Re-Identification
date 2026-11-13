from __future__ import annotations

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.control_api.progress import ProgressService
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class ProgressServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = CycleStore(Path(self.temporary.name) / "control.db")
        self.progress = ProgressService(self.store)
        self.cycle, _ = self.store.create_cycle(
            dataset="dukemtmc-reid",
            source_release="v1",
            target_release="v2",
            idempotency_key="progress-test",
            actor="tester",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_evaluation_pass_waits_for_retraining_approval(self) -> None:
        content = b"\xff\xd8\xffpreview"
        report = {
            "retrieval_preview": [{
                "query": {
                    "asset_id": hashlib.sha256(content).hexdigest(),
                    "asset_mime": "image/jpeg",
                    "asset_data": base64.b64encode(content).decode("ascii"),
                },
                "predictions": [],
            }]
        }
        cycle = self.progress.update(
            self.cycle["id"],
            "evaluate_v1",
            StageStatus.PASS,
            metrics={"evaluation": report},
        )
        approval = next(
            stage for stage in cycle["stages"] if stage["name"] == "retraining_approval"
        )
        self.assertEqual(cycle["status"], CycleStatus.WAITING_APPROVAL.value)
        self.assertEqual(approval["status"], StageStatus.WAITING_APPROVAL.value)
        events = self.store.events(cycle["id"])
        self.assertIn("stage.metrics", [event["kind"] for event in events])

    def test_evaluation_pass_rejects_missing_preview_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieval_preview"):
            self.progress.update(
                self.cycle["id"],
                "evaluate_v1",
                StageStatus.PASS,
                metrics={"evaluation": {"rank1": 0.5}},
            )
        stage = next(
            item for item in self.store.get_cycle(self.cycle["id"])["stages"]
            if item["name"] == "evaluate_v1"
        )
        self.assertEqual(stage["status"], StageStatus.PENDING.value)

    def test_candidate_evaluation_requires_preview_evidence_for_every_domain(self) -> None:
        content = b"\xff\xd8\xffcandidate-preview"
        report = {
            "retrieval_preview": [{
                "query": {
                    "asset_id": hashlib.sha256(content).hexdigest(),
                    "asset_mime": "image/jpeg",
                    "asset_data": base64.b64encode(content).decode("ascii"),
                },
                "predictions": [],
            }]
        }
        with self.assertRaisesRegex(ValueError, "market1501"):
            self.progress.update(
                self.cycle["id"],
                "evaluate_v2",
                StageStatus.PASS,
                metrics={"duke": report},
            )

    def test_staging_pass_waits_for_production_and_failure_stops_cycle(self) -> None:
        waiting = self.progress.update(
            self.cycle["id"], "staging_test", StageStatus.PASS
        )
        self.assertEqual(waiting["status"], CycleStatus.WAITING_APPROVAL.value)

        failed = self.progress.update(
            self.cycle["id"], "validate_release", StageStatus.FAIL, detail="parity failed"
        )
        self.assertEqual(failed["status"], CycleStatus.FAIL.value)
        self.assertEqual(failed["error"], "parity failed")

    def test_successful_retry_clears_previous_cycle_error(self) -> None:
        failed = self.progress.update(
            self.cycle["id"],
            "production_approval",
            StageStatus.FAIL,
            detail="previous release failed",
        )
        self.assertEqual(failed["error"], "previous release failed")

        recovered = self.progress.update(
            self.cycle["id"], "production_approval", StageStatus.PASS, progress=1.0
        )

        self.assertEqual(recovered["status"], CycleStatus.PASS.value)
        self.assertEqual(recovered["error"], "")


if __name__ == "__main__":
    unittest.main()
