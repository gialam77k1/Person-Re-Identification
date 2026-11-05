from __future__ import annotations

from typing import Any, Protocol

from src.control_api.config import ControlSettings
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class Dispatcher(Protocol):
    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict[str, Any]: ...


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
    ) -> tuple[dict[str, Any], bool]:
        cycle, created = self.store.create_cycle(
            dataset=dataset,
            source_release=source_release,
            target_release=target_release,
            idempotency_key=idempotency_key,
            actor=actor,
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
        dispatch = self.dispatcher.dispatch(
            self.settings.retraining_workflow,
            {
                "cycle_id": cycle_id,
                "approved_by": actor,
                "source_release": cycle["source_release"],
                "target_release": cycle["target_release"],
            },
        )
        self.store.audit(cycle_id, "retraining.approve", actor, dispatch)
        self.store.update_stage(
            cycle_id, "retraining_approval", StageStatus.PASS, progress=1.0, detail=f"Approved by {actor}"
        )
        self.store.update_stage(
            cycle_id, "train_v2", StageStatus.RUNNING, detail="Retraining workflow dispatched"
        )
        self.store.update_cycle(cycle_id, status=CycleStatus.RUNNING)
        self.store.record_event(cycle_id, "workflow.dispatched", "Retraining workflow dispatched", dispatch)
        return self.store.get_cycle(cycle_id)

    def promote(self, cycle_id: str, actor: str) -> dict[str, Any]:
        cycle = self.store.get_cycle(cycle_id)
        if self._stage(cycle, "staging_test")["status"] != StageStatus.PASS.value:
            raise InvalidTransitionError("Staging test must PASS before production promotion.")
        dispatch = self.dispatcher.dispatch(
            self.settings.release_workflow,
            {
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

    @staticmethod
    def _stage(cycle: dict[str, Any], name: str) -> dict[str, Any]:
        return next(stage for stage in cycle["stages"] if stage["name"] == name)
