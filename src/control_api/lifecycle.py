from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from src.control_api.approval import create_approved_decision, load_policy
from src.control_api.config import ControlSettings
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore
from src.control_api.releases import release_slug


class Dispatcher(Protocol):
    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict[str, Any]: ...
    def download_artifact(self, run_id: str, artifact_name: str) -> dict[str, bytes]: ...


class InvalidTransitionError(RuntimeError):
    pass


class LifecycleController:
    def __init__(
        self, settings: ControlSettings, store: CycleStore, dispatcher: Dispatcher
    ) -> None:
        self.settings = settings
        self.store = store
        self.dispatcher = dispatcher

    def start(
        self,
        *,
        dataset: str,
        source_release: str,
        target_release: str,
        idempotency_key: str,
        actor: str,
        inbox_batch_id: str | None = None,
        dataset_revision: str | None = None,
        dataset_path: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        cycle, created = self.store.create_cycle(
            dataset=dataset,
            source_release=source_release,
            target_release=target_release,
            idempotency_key=idempotency_key,
            actor=actor,
            inbox_batch_id=inbox_batch_id,
            dataset_revision=dataset_revision,
            dataset_path=dataset_path,
        )
        if not created:
            return cycle, False
        cycle_id = cycle["id"]
        try:
            dispatch = self.dispatcher.dispatch(
                self.settings.preapproval_workflow,
                {
                    "cycle_id": cycle_id,
                    "dataset": dataset,
                    "sample_size": "1000",
                    "release_id": source_release,
                    "release_slug": release_slug(source_release),
                    "inbox_batch_id": inbox_batch_id or "",
                    "dataset_revision": dataset_revision or "",
                    "dataset_path": dataset_path or "",
                    "dataset_host_path": (
                        str(self.settings.host_dataset_root / dataset_path)
                        if dataset_path
                        else ""
                    ),
                },
            )
        except Exception as exc:
            self.store.update_stage(cycle_id, "validate_data", StageStatus.FAIL, detail=str(exc))
            self.store.update_cycle(cycle_id, status=CycleStatus.FAIL, error=str(exc))
            self.store.record_event(cycle_id, "workflow.dispatch_failed", str(exc))
            raise
        self.store.update_cycle(
            cycle_id,
            status=CycleStatus.RUNNING,
            workflow_run_id=str(dispatch.get("workflow", self.settings.preapproval_workflow)),
        )
        self.store.update_stage(
            cycle_id, "validate_data", StageStatus.RUNNING, detail="Pre-approval workflow dispatched"
        )
        self.store.record_event(cycle_id, "workflow.dispatched", "Pre-approval workflow dispatched", dispatch)
        return self.store.get_cycle(cycle_id), True

    def approve_retraining(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        approval = self._stage(cycle, "retraining_approval")
        if cycle["status"] != CycleStatus.WAITING_APPROVAL.value or approval["status"] not in {
            StageStatus.WAITING_APPROVAL.value,
            StageStatus.PENDING.value,
        }:
            raise InvalidTransitionError("Cycle is not waiting for retraining approval.")
        evidence = self._preapproval_evidence(cycle_id)
        if not evidence.get("workflow_run_id") or not evidence.get("artifact_name"):
            raise InvalidTransitionError(
                "Pre-approval evidence is missing its workflow run ID or artifact name."
            )
        evidence_files = self.dispatcher.download_artifact(
            str(evidence["workflow_run_id"]), str(evidence["artifact_name"])
        )
        policy = load_policy(
            Path(__file__).resolve().parents[2] / "configs/retraining_decision.yaml"
        )
        try:
            decision = create_approved_decision(
                files=evidence_files,
                policy=policy,
                cycle_id=cycle_id,
                actor=actor,
                source_release=cycle["source_release"],
                target_release=cycle["target_release"],
                dataset=cycle["dataset"],
                evidence_run_id=str(evidence["workflow_run_id"]),
                evidence_artifact=str(evidence["artifact_name"]),
            )
        except ValueError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        encoded_decision = json.dumps(
            decision, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        decision_sha256 = hashlib.sha256(encoded_decision).hexdigest()
        self.store.save_retraining_decision(cycle_id, decision, decision_sha256)
        dispatch = self.dispatcher.dispatch(
            self.settings.retraining_workflow,
            self._retraining_inputs(
                cycle,
                evidence=evidence,
                decision_sha256=decision_sha256,
                resume_training_run_id="",
            ),
        )
        self.store.audit(
            cycle_id,
            "retraining.approve",
            actor,
            {
                "dispatch": dispatch,
                "evidence": evidence,
                "decision_status": decision["status"],
                "decision_sha256": decision_sha256,
            },
        )
        self.store.update_stage(
            cycle_id, "retraining_approval", StageStatus.PASS, progress=1.0, detail=f"Approved by {actor}"
        )
        self.store.update_stage(
            cycle_id, "train_v2", StageStatus.RUNNING, detail="Retraining workflow dispatched"
        )
        self.store.update_cycle(cycle_id, status=CycleStatus.RUNNING)
        self.store.record_event(cycle_id, "workflow.dispatched", "Retraining workflow dispatched", dispatch)
        self.store.record_event(
            cycle_id,
            "retraining.decision_created",
            "Approved retraining decision created by Control API",
            {
                "status": decision["status"],
                "approved_by": decision["approved_by"],
                "approved_at": decision["approved_at"],
                "decision_sha256": decision_sha256,
                "evidence": decision["evidence"],
            },
        )
        return self.store.get_cycle(cycle_id)

    def retry_post_training(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        if cycle["status"] != CycleStatus.FAIL.value:
            raise InvalidTransitionError("Only a failed cycle can resume post-training stages.")
        if self._stage(cycle, "train_v2")["status"] != StageStatus.PASS.value:
            raise InvalidTransitionError(
                "A completed training stage is required; this retry never starts GPU training."
            )
        post_training = (
            "evaluate_v2",
            "validate_release",
            "export_onnx",
            "publish_artifacts",
            "register_candidate",
        )
        if not any(
            self._stage(cycle, name)["status"] == StageStatus.FAIL.value
            for name in post_training
        ):
            raise InvalidTransitionError("No failed post-training stage is available to retry.")
        training_run_id = self._latest_training_run_id(cycle_id)
        if not training_run_id:
            raise InvalidTransitionError(
                "The completed training stage has no MLflow run ID to restore."
            )
        evidence = self._preapproval_evidence(cycle_id)
        decision = self.store.get_retraining_decision(cycle_id)
        dispatch_inputs = self._retraining_inputs(
            cycle,
            evidence=evidence,
            decision_sha256=str(decision["sha256"]),
            resume_training_run_id=training_run_id,
        )
        dispatch = self.dispatcher.dispatch(
            self.settings.retraining_workflow, dispatch_inputs
        )
        for name in post_training:
            self.store.update_stage(
                cycle_id,
                name,
                StageStatus.RUNNING if name == "evaluate_v2" else StageStatus.PENDING,
                detail=(
                    f"Restoring MLflow run {training_run_id} and resuming at evaluation"
                    if name == "evaluate_v2"
                    else "Waiting for resumed post-training dependencies"
                ),
            )
        self.store.update_cycle(cycle_id, status=CycleStatus.RUNNING, error="")
        payload = {
            "dispatch": dispatch,
            "mlflow_run_id": training_run_id,
            "resume_stage": "evaluate_v2",
        }
        self.store.audit(cycle_id, "post_training.retry", actor, payload)
        self.store.record_event(
            cycle_id,
            "workflow.dispatched",
            "Post-training workflow resumed from the completed MLflow run",
            payload,
        )
        return self.store.get_cycle(cycle_id)

    def _preapproval_evidence(self, cycle_id: str) -> dict[str, Any]:
        for event in reversed(self.store.events(cycle_id)):
            if event["kind"] != "stage.metrics":
                continue
            payload = event.get("payload", {})
            if payload.get("stage") == "evaluate_v1":
                return dict(payload.get("metrics", {}))
        return {}

    def _latest_training_run_id(self, cycle_id: str) -> str:
        for event in reversed(self.store.events(cycle_id)):
            if event["kind"] != "stage.metrics":
                continue
            payload = event.get("payload", {})
            if payload.get("stage") != "train_v2":
                continue
            run_id = str(payload.get("metrics", {}).get("mlflow_run_id", "")).strip()
            if run_id:
                return run_id
        return ""

    def _retraining_inputs(
        self,
        cycle: dict[str, Any],
        *,
        evidence: dict[str, Any],
        decision_sha256: str,
        resume_training_run_id: str,
    ) -> dict[str, str]:
        return {
            "cycle_id": str(cycle["id"]),
            "source_release": str(cycle["source_release"]),
            "target_release": str(cycle["target_release"]),
            "dataset": str(cycle["dataset"]),
            "evidence_run_id": str(evidence["workflow_run_id"]),
            "evidence_artifact": str(evidence["artifact_name"]),
            "drift_report": "evidence/drift_report.json",
            "evaluation_report": "evidence/evaluation_report.json",
            "decision": "evidence/approved_decision.json",
            "decision_sha256": decision_sha256,
            "inbox_batch_id": str(cycle.get("inbox_batch_id") or ""),
            "dataset_revision": str(cycle.get("dataset_revision") or ""),
            "dataset_path": str(cycle.get("dataset_path") or ""),
            "dataset_host_path": (
                str(self.settings.host_dataset_root / str(cycle["dataset_path"]))
                if cycle.get("dataset_path")
                else ""
            ),
            "resume_training_run_id": resume_training_run_id,
        }

    def promote(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        if self._stage(cycle, "register_candidate")["status"] != StageStatus.PASS.value:
            raise InvalidTransitionError("A validated MLflow candidate is required before release.")
        dispatch = self.dispatcher.dispatch(
            self.settings.release_workflow,
            {
                "cycle_id": cycle_id,
                "release_id": cycle["target_release"],
                "release_slug": cycle["target_release"].replace(":", "-").replace("_", "-"),
            },
        )
        self.store.audit(cycle_id, "production.promote", actor, dispatch)
        self.store.update_stage(
            cycle_id,
            "production_approval",
            StageStatus.RUNNING,
            detail=f"Promotion requested by {actor}",
        )
        self.store.record_event(cycle_id, "workflow.dispatched", "Release workflow dispatched", dispatch)
        return self.store.get_cycle(cycle_id)

    def rollback(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        dispatch = self.dispatcher.dispatch(
            self.settings.rollback_workflow, {"environment": "production", "cycle_id": cycle_id}
        )
        self.store.audit(cycle_id, "production.rollback", actor, dispatch)
        self.store.record_event(cycle_id, "workflow.dispatched", "Rollback workflow dispatched", dispatch)
        return cycle

    def abandon_failed(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        if cycle["status"] != CycleStatus.FAIL.value:
            raise InvalidTransitionError("Only a failed cycle can be abandoned.")
        skipped = []
        for stage in cycle["stages"]:
            if stage["status"] in {
                StageStatus.PENDING.value,
                StageStatus.RUNNING.value,
                StageStatus.WAITING_APPROVAL.value,
            }:
                self.store.update_stage(
                    cycle_id,
                    stage["name"],
                    StageStatus.SKIPPED,
                    detail=f"Cycle abandoned by {actor}; previous evidence remains available",
                )
                skipped.append(stage["name"])
        self.store.update_cycle(cycle_id, status=CycleStatus.CANCELLED)
        payload = {
            "target_release": cycle["target_release"],
            "previous_error": cycle.get("error"),
            "skipped_stages": skipped,
        }
        self.store.audit(cycle_id, "cycle.abandon_failed", actor, payload)
        self.store.record_event(
            cycle_id,
            "cycle.abandoned",
            f"Failed cycle abandoned by {actor}; target release unlocked",
            payload,
        )
        return self.store.get_cycle(cycle_id)

    @staticmethod
    def _stage(cycle: dict[str, Any], name: str) -> dict[str, Any]:
        return next(stage for stage in cycle["stages"] if stage["name"] == name)
