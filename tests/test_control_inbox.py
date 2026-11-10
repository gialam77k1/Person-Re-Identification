from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.control_api.config import ControlSettings
from src.control_api.inbox import ActiveCycleError, DataInboxService, InvalidBatchError
from src.control_api.lifecycle import LifecycleController
from src.control_api.state import CycleStatus
from src.control_api.store import CycleStore


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict[str, Any]:
        self.calls.append((workflow, inputs))
        return {"workflow": workflow, "inputs": inputs}

    def download_artifact(self, run_id: str, artifact_name: str) -> dict[str, bytes]:
        return {}


def image(path: Path, marker: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff" + marker)


class DataInboxServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = replace(
            ControlSettings.from_env(),
            database_path=root / "control.db",
            dataset_root=root / "datasets",
            allowed_datasets=("dukemtmc-reid",),
            inbox_min_valid_images=3,
            inbox_min_identities=2,
            inbox_min_cameras=2,
            registry_model_name="",
        )
        self.store = CycleStore(self.settings.database_path)
        self.dispatcher = FakeDispatcher()
        self.lifecycle = LifecycleController(self.settings, self.store, self.dispatcher)
        self.inbox = DataInboxService(self.settings, self.store, self.lifecycle)
        train = self.settings.dataset_root / "dukemtmc-reid" / "bounding_box_train"
        image(train / "0001_c1_f000001.jpg", b"person-1-camera-1")
        image(train / "0001_c2_f000002.jpg", b"person-1-camera-2")
        image(train / "0002_c2_f000003.jpg", b"person-2-camera-2")
        image(
            self.settings.dataset_root / "dukemtmc-reid" / "query" / "9999_c9_query.jpg",
            b"held-out-query",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_batch_dispatches_once_and_ignores_held_out_splits(self) -> None:
        batch, created = self.inbox.import_batch(
            source_path="dukemtmc-reid/bounding_box_train",
            dataset="dukemtmc-reid",
            actor="operator",
        )

        self.assertTrue(created)
        self.assertEqual(batch["status"], "DISPATCHED")
        self.assertEqual(batch["valid_images"], 3)
        self.assertEqual(batch["identities"], 2)
        self.assertEqual(batch["cameras"], 2)
        self.assertEqual(len(self.dispatcher.calls), 1)

        self.store.update_cycle(batch["cycle_id"], status=CycleStatus.PASS)
        repeated, repeated_created = self.inbox.import_batch(
            source_path="dukemtmc-reid/bounding_box_train",
            dataset="dukemtmc-reid",
            actor="operator",
        )
        self.assertFalse(repeated_created)
        self.assertEqual(repeated["id"], batch["id"])
        self.assertEqual(len(self.dispatcher.calls), 1)

    def test_active_cycle_blocks_a_second_import(self) -> None:
        batch, _ = self.inbox.import_batch(
            source_path="dukemtmc-reid/bounding_box_train",
            dataset="dukemtmc-reid",
            actor="operator",
        )

        with self.assertRaises(ActiveCycleError) as raised:
            self.inbox.import_batch(
                source_path="dukemtmc-reid/bounding_box_train",
                dataset="dukemtmc-reid",
                actor="operator",
            )

        self.assertEqual(raised.exception.cycle["id"], batch["cycle_id"])
        self.assertEqual(len(self.dispatcher.calls), 1)

    def test_invalid_and_duplicate_images_do_not_count_toward_threshold(self) -> None:
        train = self.settings.dataset_root / "dukemtmc-reid" / "bounding_box_train"
        (train / "bad_name.jpg").write_bytes(b"\xff\xd8\xffbad")
        (train / "0003_c3_broken.jpg").write_bytes(b"not-an-image")
        (train / "0004_c4_duplicate.jpg").write_bytes(
            (train / "0001_c1_f000001.jpg").read_bytes()
        )
        scan = self.inbox._scan(train, {})

        self.assertEqual(scan["valid_images"], 3)
        self.assertEqual(scan["rejected_images"], 2)
        self.assertEqual(scan["duplicate_images"], 1)

    def test_manifest_batch_supports_new_paths_without_code_changes(self) -> None:
        batch_root = self.settings.dataset_root / "inbox" / "november-batch"
        image(batch_root / "images" / "person-a.jpg", b"new-a")
        image(batch_root / "images" / "person-b.jpg", b"new-b")
        image(batch_root / "images" / "person-c.jpg", b"new-c")
        (batch_root / "manifest.json").write_text(
            """{
              "batch_id": "november-batch",
              "dataset": "dukemtmc-reid",
              "samples": [
                {"image": "images/person-a.jpg", "person_id": 10, "camera_id": 1},
                {"image": "images/person-b.jpg", "person_id": 10, "camera_id": 2},
                {"image": "images/person-c.jpg", "person_id": 11, "camera_id": 2}
              ]
            }""",
            encoding="utf-8",
        )

        batch, created = self.inbox.import_batch(
            source_path="inbox/november-batch",
            dataset="dukemtmc-reid",
            actor="operator",
        )

        self.assertTrue(created)
        self.assertEqual(batch["summary"]["batch_id"], "november-batch")
        self.assertIn(
            "inbox/november-batch",
            [source["source_path"] for source in self.inbox.discover_sources()],
        )

    def test_blocked_batch_can_retry_after_failed_target_is_abandoned(self) -> None:
        failed, _ = self.store.create_cycle(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="failed-target-v2",
            actor="operator",
        )
        self.store.update_cycle(failed["id"], status=CycleStatus.FAIL)
        with self.assertRaises(InvalidBatchError):
            self.inbox.import_batch(
                source_path="dukemtmc-reid/bounding_box_train",
                dataset="dukemtmc-reid",
                actor="operator",
            )
        blocked = self.store.list_inbox_batches()[0]
        self.assertEqual(blocked["status"], "BLOCKED")

        self.lifecycle.abandon_failed(failed["id"], "operator")
        retried, created = self.inbox.import_batch(
            source_path="dukemtmc-reid/bounding_box_train",
            dataset="dukemtmc-reid",
            actor="operator",
        )

        self.assertTrue(created)
        self.assertEqual(retried["id"], blocked["id"])
        self.assertEqual(retried["status"], "DISPATCHED")
        self.assertIsNotNone(retried["cycle_id"])


if __name__ == "__main__":
    unittest.main()
