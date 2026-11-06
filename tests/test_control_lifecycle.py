from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.control_api.config import ControlSettings
from src.control_api.lifecycle import InvalidTransitionError, LifecycleController
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict:
        self.calls.append((workflow, inputs))
        return {"workflow": workflow, "inputs": inputs}


class LifecycleControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = replace(
            ControlSettings.from_env(), database_path=Path(self.temporary.name) / "control.db"
        )
        self.store = CycleStore(self.settings.database_path)
        self.dispatcher = FakeDispatcher()
        self.lifecycle = LifecycleController(self.settings, self.store, self.dispatcher)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_start_is_idempotent_and_dispatches_preapproval_once(self) -> None:
        arguments = {
            "dataset": "MSMT17",
            "source_release": "v1",
            "target_release": "v2",
            "idempotency_key": "lifecycle-1",
            "actor": "tester",
        }
        first, created = self.lifecycle.start(**arguments)
        second, duplicated = self.lifecycle.start(**arguments)

        self.assertTrue(created)
        self.assertFalse(duplicated)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["status"], CycleStatus.RUNNING.value)
        self.assertEqual(len(self.dispatcher.calls), 1)

    def test_approval_requires_waiting_state_and_dispatches_retraining(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="MSMT17",
            source_release="v1",
            target_release="v2",
            idempotency_key="lifecycle-2",
            actor="tester",
        )
        with self.assertRaises(InvalidTransitionError):
            self.lifecycle.approve_retraining(cycle["id"], "reviewer")

        self.store.update_cycle(cycle["id"], status=CycleStatus.WAITING_APPROVAL)
        self.store.update_stage(
            cycle["id"], "retraining_approval", StageStatus.WAITING_APPROVAL
        )
        self.store.record_event(
            cycle["id"],
            "stage.metrics",
            "Evidence uploaded",
            {
                "stage": "evaluate_v1",
                "metrics": {
                    "workflow_run_id": "12345",
                    "artifact_name": f"mlops-evidence-{cycle['id']}",
                },
            },
        )
        approved = self.lifecycle.approve_retraining(cycle["id"], "reviewer")

        self.assertEqual(approved["status"], CycleStatus.RUNNING.value)
        train = next(stage for stage in approved["stages"] if stage["name"] == "train_v2")
        self.assertEqual(train["status"], StageStatus.RUNNING.value)
        self.assertEqual(self.dispatcher.calls[-1][0], self.settings.retraining_workflow)
        dispatched = self.dispatcher.calls[-1][1]
        self.assertEqual(dispatched["evidence_run_id"], "12345")
        self.assertEqual(dispatched["dataset"], "MSMT17")

    def test_approval_rejects_cycle_without_uploaded_evidence(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="MSMT17", source_release="v1", target_release="v2",
            idempotency_key="lifecycle-missing-evidence", actor="tester",
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.WAITING_APPROVAL)
        self.store.update_stage(cycle["id"], "retraining_approval", StageStatus.WAITING_APPROVAL)
        with self.assertRaisesRegex(InvalidTransitionError, "evidence"):
            self.lifecycle.approve_retraining(cycle["id"], "reviewer")
        self.assertEqual(len(self.dispatcher.calls), 1)


if __name__ == "__main__":
    unittest.main()
