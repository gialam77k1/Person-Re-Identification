from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class CycleStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = CycleStore(Path(self.temporary.name) / "control.db")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cycle_creation_is_idempotent_and_initializes_stages(self) -> None:
        arguments = {
            "dataset": "MSMT17",
            "source_release": "market1501-vit-bnneck:v1",
            "target_release": "market1501-msmt17-vit-bnneck:v2",
            "idempotency_key": "msmt17-v2-epoch20",
            "actor": "gialam77k1",
        }
        first, created = self.store.create_cycle(**arguments)
        second, duplicated = self.store.create_cycle(**arguments)

        self.assertTrue(created)
        self.assertFalse(duplicated)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["status"], CycleStatus.PENDING)
        self.assertEqual(first["stages"][0]["name"], "validate_data")
        self.assertEqual(first["stages"][-1]["name"], "production_approval")

    def test_stage_and_cycle_updates_emit_events(self) -> None:
        cycle, _ = self.store.create_cycle(
            dataset="MSMT17",
            source_release="v1",
            target_release="v2",
            idempotency_key="cycle-2",
            actor="tester",
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.RUNNING)
        updated = self.store.update_stage(
            cycle["id"], "validate_data", StageStatus.RUNNING, progress=0.25, detail="checking"
        )

        self.assertEqual(updated["status"], CycleStatus.RUNNING)
        self.assertEqual(updated["stages"][0]["progress"], 0.25)
        self.assertEqual(self.store.events(cycle["id"])[-1]["payload"]["stage"], "validate_data")

    def test_retraining_decision_is_persisted_with_checksum(self) -> None:
        cycle, _ = self.store.create_cycle(
            dataset="MSMT17",
            source_release="v1",
            target_release="v2",
            idempotency_key="decision-store",
            actor="tester",
        )
        saved = self.store.save_retraining_decision(
            cycle["id"],
            {"status": "RETRAIN_APPROVED", "approved_by": "reviewer"},
            "a" * 64,
        )
        restored = self.store.get_retraining_decision(cycle["id"])
        self.assertEqual(saved["sha256"], "a" * 64)
        self.assertEqual(restored["decision"]["status"], "RETRAIN_APPROVED")


if __name__ == "__main__":
    unittest.main()
