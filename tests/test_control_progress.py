from __future__ import annotations

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
        cycle = self.progress.update(
            self.cycle["id"],
            "evaluate_v1",
            StageStatus.PASS,
            metrics={"rank1": 0.18, "mAP": 0.07},
        )
        approval = next(
            stage for stage in cycle["stages"] if stage["name"] == "retraining_approval"
        )
        self.assertEqual(cycle["status"], CycleStatus.WAITING_APPROVAL.value)
        self.assertEqual(approval["status"], StageStatus.WAITING_APPROVAL.value)
        events = self.store.events(cycle["id"])
        self.assertIn("stage.metrics", [event["kind"] for event in events])

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


if __name__ == "__main__":
    unittest.main()
