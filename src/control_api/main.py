from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.control_api.config import ControlSettings
from src.control_api.overview import OverviewService
from src.control_api.schemas import CycleCreate, CycleCreateResponse, EventListResponse
from src.control_api.store import CycleStore


def create_app(
    settings: ControlSettings | None = None,
    store: CycleStore | None = None,
    overview: OverviewService | None = None,
) -> FastAPI:
    configured = settings or ControlSettings.from_env()
    cycle_store = store or CycleStore(configured.database_path)
    overview_service = overview or OverviewService(configured)
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

    @api.post("/control/cycles", response_model=CycleCreateResponse)
    async def create_cycle(request: CycleCreate) -> CycleCreateResponse:
        if request.dataset not in configured.allowed_datasets:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset must be one of: {', '.join(configured.allowed_datasets)}",
            )
        cycle, created = cycle_store.create_cycle(
            dataset=request.dataset,
            source_release=request.source_release,
            target_release=request.target_release,
            idempotency_key=request.idempotency_key,
            actor=request.actor,
        )
        return CycleCreateResponse(created=created, cycle=cycle)

    return api


app = create_app()
