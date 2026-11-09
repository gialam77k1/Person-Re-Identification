from __future__ import annotations

import secrets

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from src.control_api.config import ControlSettings
from src.control_api.evaluations import EvaluationService
from src.control_api.github_actions import GitHubActionsClient, WorkflowDispatchError
from src.control_api.inbox import ActiveCycleError, DataInboxService, InvalidBatchError
from src.control_api.lifecycle import InvalidTransitionError, LifecycleController
from src.control_api.metrics import render_control_metrics
from src.control_api.overview import OverviewService
from src.control_api.progress import ProgressService
from src.control_api.schemas import (
    ActorRequest,
    CycleCreate,
    CycleCreateResponse,
    EventListResponse,
    InboxImportResponse,
    InboxImportRequest,
    StageUpdateRequest,
)
from src.control_api.state import StageStatus
from src.control_api.store import CycleStore


def create_app(
    settings: ControlSettings | None = None,
    store: CycleStore | None = None,
    overview: OverviewService | None = None,
    lifecycle: LifecycleController | None = None,
) -> FastAPI:
    configured = settings or ControlSettings.from_env()
    cycle_store = store or CycleStore(configured.database_path)
    overview_service = overview or OverviewService(configured)
    lifecycle_service = lifecycle or LifecycleController(
        configured, cycle_store, GitHubActionsClient(configured)
    )
    progress_service = ProgressService(cycle_store)
    evaluation_service = EvaluationService(configured, cycle_store)
    inbox_service = DataInboxService(configured, cycle_store, lifecycle_service)
    api = FastAPI(
        title="Person ReID MLOps Control API",
        version="0.1.0",
        description="Persistent control plane for drift, retraining, release and rollback cycles.",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(
            content=render_control_metrics(cycle_store),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @api.get("/control/overview")
    async def get_overview() -> dict:
        payload = overview_service.overview()
        recent = cycle_store.list_cycles(limit=1)
        payload["latest_cycle"] = recent[0] if recent else None
        payload["allowed_datasets"] = list(configured.allowed_datasets)
        return payload

    @api.get("/control/services")
    async def get_services() -> dict:
        return {"services": overview_service.services()}

    @api.get("/control/evaluations")
    async def get_evaluations() -> dict:
        return evaluation_service.overview()

    @api.get("/control/inbox")
    async def get_inbox(limit: int = Query(default=25, ge=1, le=100)) -> dict:
        return {
            "batches": cycle_store.list_inbox_batches(limit),
            "sources": inbox_service.discover_sources(),
            "policy": {
                "allowed_datasets": list(configured.allowed_datasets),
                "min_valid_images": configured.inbox_min_valid_images,
                "min_identities": configured.inbox_min_identities,
                "min_cameras": configured.inbox_min_cameras,
            },
        }

    @api.post("/control/inbox/import", response_model=InboxImportResponse)
    async def import_batch(request: InboxImportRequest) -> InboxImportResponse:
        try:
            batch, created = inbox_service.import_batch(
                source_path=request.source_path,
                dataset=request.dataset,
                actor=request.actor,
            )
        except ActiveCycleError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "cycle_id": exc.cycle["id"]},
            ) from exc
        except (FileNotFoundError, InvalidBatchError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except WorkflowDispatchError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return InboxImportResponse(created=created, batch=batch)

    @api.get("/control/evaluation-images/{dataset}/{image_path:path}")
    async def get_evaluation_image(dataset: str, image_path: str) -> Response:
        try:
            path = evaluation_service.resolve_image(dataset, image_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Evaluation image not found") from exc
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @api.get("/control/cycles")
    async def list_cycles(limit: int = Query(default=25, ge=1, le=100)) -> dict:
        return {"cycles": cycle_store.list_cycles(limit=limit)}

    @api.get("/control/cycles/{cycle_id}")
    async def get_cycle(cycle_id: str) -> dict:
        try:
            return cycle_store.get_cycle(cycle_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Cycle not found") from exc

    @api.get("/control/cycles/{cycle_id}/events", response_model=EventListResponse)
    async def get_events(
        cycle_id: str, after_id: int = Query(default=0, ge=0)
    ) -> EventListResponse:
        try:
            cycle_store.get_cycle(cycle_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Cycle not found") from exc
        events = cycle_store.events(cycle_id, after_id=after_id)
        return EventListResponse(
            events=events,
            last_event_id=events[-1]["id"] if events else after_id,
        )

    @api.get("/control/cycles/{cycle_id}/audit")
    async def get_audit(cycle_id: str) -> dict:
        try:
            cycle_store.get_cycle(cycle_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Cycle not found") from exc
        return {"audit": cycle_store.audit_entries(cycle_id)}

    @api.get("/control/cycles/{cycle_id}/retraining-decision")
    async def get_retraining_decision(
        cycle_id: str,
        x_control_callback_token: str = Header(default=""),
    ) -> dict:
        if not secrets.compare_digest(
            x_control_callback_token, configured.callback_token
        ):
            raise HTTPException(status_code=401, detail="Invalid callback token")
        try:
            return cycle_store.get_retraining_decision(cycle_id)["decision"]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Retraining decision not found") from exc

    @api.post("/control/cycles", response_model=CycleCreateResponse)
    async def create_cycle(request: CycleCreate) -> CycleCreateResponse:
        if request.dataset not in configured.allowed_datasets:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset must be one of: {', '.join(configured.allowed_datasets)}",
            )
        try:
            cycle, created = lifecycle_service.start(
                dataset=request.dataset,
                source_release=request.source_release,
                target_release=request.target_release,
                idempotency_key=request.idempotency_key,
                actor=request.actor,
            )
        except WorkflowDispatchError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return CycleCreateResponse(created=created, cycle=cycle)

    @api.post("/control/cycles/{cycle_id}/approve-retraining")
    async def approve_retraining(cycle_id: str, request: ActorRequest) -> dict:
        return _lifecycle_action(lifecycle_service.approve_retraining, cycle_id, request.actor)

    @api.post("/control/cycles/{cycle_id}/promote")
    async def promote(cycle_id: str, request: ActorRequest) -> dict:
        return _lifecycle_action(lifecycle_service.promote, cycle_id, request.actor)

    @api.post("/control/cycles/{cycle_id}/rollback")
    async def rollback(cycle_id: str, request: ActorRequest) -> dict:
        return _lifecycle_action(lifecycle_service.rollback, cycle_id, request.actor)

    @api.post("/control/cycles/{cycle_id}/stages/{stage}")
    async def update_stage(
        cycle_id: str,
        stage: str,
        request: StageUpdateRequest,
        x_control_callback_token: str = Header(default=""),
    ) -> dict:
        if not secrets.compare_digest(
            x_control_callback_token, configured.callback_token
        ):
            raise HTTPException(status_code=401, detail="Invalid callback token")
        try:
            return progress_service.update(
                cycle_id,
                stage,
                StageStatus(request.status),
                progress=request.progress,
                detail=request.detail,
                metrics=request.metrics,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Cycle or stage not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return api


def _lifecycle_action(callback, cycle_id: str, actor: str) -> dict:
    try:
        return callback(cycle_id, actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Cycle not found") from exc
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WorkflowDispatchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


app = create_app()
