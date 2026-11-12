from __future__ import annotations

from typing import Any

from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class ProgressService:
    def __init__(self, store: CycleStore) -> None:
        self.store = store

    def update(
        self,
        cycle_id: str,
        stage: str,
        status: StageStatus,
        *,
        progress: float | None = None,
        detail: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cycle = self.store.update_stage(
            cycle_id, stage, status, progress=progress, detail=detail
        )
        if metrics:
            self.store.record_event(
                cycle_id,
                "stage.metrics",
                f"Metrics updated for {stage}",
                {"stage": stage, "metrics": metrics},
            )
        if status == StageStatus.FAIL:
            for active_stage in cycle["stages"]:
                if active_stage["name"] == stage:
                    continue
                if active_stage["status"] == StageStatus.RUNNING.value:
                    self.store.update_stage(
                        cycle_id,
                        active_stage["name"],
                        StageStatus.FAIL,
                        progress=active_stage["progress"],
                        detail=f"Stopped because {stage} failed: {detail or 'no detail'}",
                    )
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.FAIL, error=detail or f"{stage} failed"
            )
        if stage == "evaluate_v1" and status == StageStatus.PASS:
            self.store.update_stage(
                cycle_id,
                "retraining_approval",
                StageStatus.WAITING_APPROVAL,
                detail="Review drift and evaluation evidence",
            )
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.WAITING_APPROVAL, error=""
            )
        if stage == "staging_test" and status == StageStatus.PASS:
            self.store.update_stage(
                cycle_id,
                "production_approval",
                StageStatus.WAITING_APPROVAL,
                detail="Staging passed; production approval required",
            )
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.WAITING_APPROVAL, error=""
            )
        if stage == "register_candidate" and status == StageStatus.PASS:
            self.store.update_stage(
                cycle_id,
                "production_approval",
                StageStatus.WAITING_APPROVAL,
                detail="Validated candidate is ready for staging and production approval",
            )
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.WAITING_APPROVAL, error=""
            )
        if stage == "production_approval" and status == StageStatus.PASS:
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.PASS, error=""
            )
        if status == StageStatus.RUNNING:
            return self.store.update_cycle(
                cycle_id, status=CycleStatus.RUNNING, error=""
            )
        return cycle
