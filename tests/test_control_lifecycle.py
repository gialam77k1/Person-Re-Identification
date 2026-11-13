from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from src.control_api.config import ControlSettings
from src.control_api.lifecycle import InvalidTransitionError, LifecycleController
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.artifact: dict[str, bytes] = {}

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict:
        self.calls.append((workflow, inputs))
        return {"workflow": workflow, "inputs": inputs}

    def download_artifact(self, run_id: str, artifact_name: str) -> dict[str, bytes]:
        return self.artifact


def approval_artifact(cycle_id: str) -> dict[str, bytes]:
    release = "market1501-vit-bnneck:v1-dvc"
    drift = json.dumps({"release_id": release, "drift_detected": True}).encode()
    evaluation = json.dumps(
        {
            "release_id": release,
            "dataset": "dukemtmc-reid",
            "num_query_samples": 2228,
            "num_gallery_samples": 17661,
            "query_coverage": 1.0,
            "embedding_dim": 512,
            "embeddings_finite": True,
            "dataset_dvc_revision": "abc.dir",
            "rank1": 0.01,
            "mAP": 0.01,
        }
    ).encode()
    review = json.dumps({"status": "REVIEW_REQUIRED"}).encode()
    manifest = {
        "cycle_id": cycle_id,
        "workflow_run_id": "12345",
        "release_id": release,
        "dataset": "dukemtmc-reid",
        "dataset_dvc_revision": "abc.dir",
        "source_git_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "evidence": {
            "drift_report": {
                "path": "drift_report.json",
                "sha256": hashlib.sha256(drift).hexdigest(),
            },
            "evaluation_report": {
                "path": "evaluation_report.json",
                "sha256": hashlib.sha256(evaluation).hexdigest(),
            },
            "retraining_decision": {
                "path": "decision.json",
                "sha256": hashlib.sha256(review).hexdigest(),
            },
        },
    }
    return {
        "evidence_manifest.json": json.dumps(manifest).encode(),
        "drift_report.json": drift,
        "evaluation_report.json": evaluation,
        "decision.json": review,
    }


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
            "dataset": "dukemtmc-reid",
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
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
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
        self.dispatcher.artifact = approval_artifact(cycle["id"])
        approved = self.lifecycle.approve_retraining(cycle["id"], "reviewer")

        self.assertEqual(approved["status"], CycleStatus.RUNNING.value)
        train = next(stage for stage in approved["stages"] if stage["name"] == "train_v2")
        self.assertEqual(train["status"], StageStatus.RUNNING.value)
        self.assertEqual(self.dispatcher.calls[-1][0], self.settings.retraining_workflow)
        dispatched = self.dispatcher.calls[-1][1]
        self.assertEqual(dispatched["evidence_run_id"], "12345")
        self.assertEqual(dispatched["dataset"], "dukemtmc-reid")
        self.assertEqual(
            dispatched["evidence_git_commit"],
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        self.assertEqual(len(dispatched["decision_sha256"]), 64)
        stored = self.store.get_retraining_decision(cycle["id"])
        self.assertEqual(stored["decision"]["status"], "RETRAIN_APPROVED")
        self.assertEqual(stored["decision"]["target_dataset_version"], "dukemtmc-reid")

    def test_approval_rejects_cycle_without_uploaded_evidence(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid", source_release="v1", target_release="v2",
            idempotency_key="lifecycle-missing-evidence", actor="tester",
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.WAITING_APPROVAL)
        self.store.update_stage(cycle["id"], "retraining_approval", StageStatus.WAITING_APPROVAL)
        with self.assertRaisesRegex(InvalidTransitionError, "evidence"):
            self.lifecycle.approve_retraining(cycle["id"], "reviewer")
        self.assertEqual(len(self.dispatcher.calls), 1)

    def test_failed_evaluation_resumes_from_finished_mlflow_run_without_retraining(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="post-training-retry",
            actor="tester",
        )
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
        self.dispatcher.artifact = approval_artifact(cycle["id"])
        self.lifecycle.approve_retraining(cycle["id"], "reviewer")
        self.store.update_stage(cycle["id"], "train_v2", StageStatus.PASS)
        self.store.record_event(
            cycle["id"],
            "stage.metrics",
            "Training finished",
            {"stage": "train_v2", "metrics": {"mlflow_run_id": "run-finished"}},
        )
        self.store.update_stage(cycle["id"], "evaluate_v2", StageStatus.FAIL)
        self.store.update_cycle(cycle["id"], status=CycleStatus.FAIL, error="alias mismatch")

        resumed = self.lifecycle.retry_post_training(cycle["id"], "operator")

        self.assertEqual(resumed["status"], CycleStatus.RUNNING.value)
        self.assertEqual(
            self.dispatcher.calls[-1][1]["resume_training_run_id"], "run-finished"
        )
        self.assertEqual(
            next(stage for stage in resumed["stages"] if stage["name"] == "train_v2")["status"],
            StageStatus.PASS.value,
        )
        self.assertEqual(
            next(stage for stage in resumed["stages"] if stage["name"] == "evaluate_v2")["status"],
            StageStatus.RUNNING.value,
        )

    def test_post_training_retry_requires_completed_training_run(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid", source_release="v1", target_release="v2",
            idempotency_key="post-training-no-checkpoint", actor="tester",
        )
        self.store.update_stage(cycle["id"], "evaluate_v2", StageStatus.FAIL)
        self.store.update_cycle(cycle["id"], status=CycleStatus.FAIL)
        with self.assertRaisesRegex(InvalidTransitionError, "completed training"):
            self.lifecycle.retry_post_training(cycle["id"], "operator")

    def test_failed_release_reopens_production_approval_without_dispatch(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="production-release-retry",
            actor="tester",
        )
        for name in (
            "train_v2",
            "evaluate_v2",
            "validate_release",
            "export_onnx",
            "publish_artifacts",
            "register_candidate",
        ):
            self.store.update_stage(cycle["id"], name, StageStatus.PASS)
        self.store.update_stage(
            cycle["id"], "production_approval", StageStatus.FAIL
        )
        self.store.update_cycle(
            cycle["id"], status=CycleStatus.FAIL, error="release failed"
        )
        dispatch_count = len(self.dispatcher.calls)

        resumed = self.lifecycle.retry_production_release(cycle["id"], "operator")

        self.assertEqual(resumed["status"], CycleStatus.WAITING_APPROVAL.value)
        self.assertEqual(resumed["error"], "")
        self.assertEqual(len(self.dispatcher.calls), dispatch_count)
        self.assertEqual(
            next(
                stage
                for stage in resumed["stages"]
                if stage["name"] == "production_approval"
            )["status"],
            StageStatus.WAITING_APPROVAL.value,
        )
        self.assertEqual(
            next(
                stage for stage in resumed["stages"] if stage["name"] == "deploy_staging"
            )["status"],
            StageStatus.PENDING.value,
        )

    def test_failed_preflight_retries_with_the_approved_evidence_commit(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="training-setup-retry",
            actor="tester",
        )
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
        self.dispatcher.artifact = approval_artifact(cycle["id"])
        self.lifecycle.approve_retraining(cycle["id"], "reviewer")
        self.store.update_stage(
            cycle["id"],
            "train_v2",
            StageStatus.FAIL,
            detail="Retraining setup failed before training started",
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.FAIL)

        retried = self.lifecycle.retry_training_setup(cycle["id"], "operator")

        self.assertEqual(retried["status"], CycleStatus.RUNNING.value)
        self.assertEqual(
            self.dispatcher.calls[-1][1]["evidence_git_commit"],
            "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        )
        self.assertEqual(self.dispatcher.calls[-1][1]["resume_training_run_id"], "")
        self.assertEqual(self.dispatcher.calls[-1][1]["repair_retry"], "true")
        train = next(stage for stage in retried["stages"] if stage["name"] == "train_v2")
        self.assertEqual(train["status"], StageStatus.RUNNING.value)

    def test_training_setup_retry_rejects_cycle_with_finished_training_run(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid", source_release="v1", target_release="v2",
            idempotency_key="training-setup-has-run", actor="tester",
        )
        self.store.update_stage(cycle["id"], "retraining_approval", StageStatus.PASS)
        self.store.update_stage(cycle["id"], "train_v2", StageStatus.FAIL)
        self.store.record_event(
            cycle["id"], "stage.metrics", "Training run exists",
            {"stage": "train_v2", "metrics": {"mlflow_run_id": "finished-run"}},
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.FAIL)
        with self.assertRaisesRegex(InvalidTransitionError, "post-training recovery"):
            self.lifecycle.retry_training_setup(cycle["id"], "operator")

    def test_only_failed_cycle_can_be_abandoned_and_audit_is_preserved(self) -> None:
        cycle, _ = self.lifecycle.start(
            dataset="dukemtmc-reid", source_release="v1", target_release="v2",
            idempotency_key="lifecycle-failed", actor="tester",
        )
        with self.assertRaisesRegex(InvalidTransitionError, "Only a failed"):
            self.lifecycle.abandon_failed(cycle["id"], "operator")
        self.store.update_stage(
            cycle["id"], "validate_data", StageStatus.FAIL, detail="broken"
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.FAIL, error="broken")

        abandoned = self.lifecycle.abandon_failed(cycle["id"], "operator")

        self.assertEqual(abandoned["status"], CycleStatus.CANCELLED.value)
        validate = next(stage for stage in abandoned["stages"] if stage["name"] == "validate_data")
        pending = next(stage for stage in abandoned["stages"] if stage["name"] == "extract_embeddings")
        self.assertEqual(validate["status"], StageStatus.FAIL.value)
        self.assertEqual(pending["status"], StageStatus.SKIPPED.value)
        self.assertTrue(any(event["kind"] == "cycle.abandoned" for event in self.store.events(cycle["id"])))


if __name__ == "__main__":
    unittest.main()
