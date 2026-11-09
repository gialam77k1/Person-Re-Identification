from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CycleCreate(BaseModel):
    dataset: str
    source_release: str = "market1501-vit-bnneck:v1-dvc"
    target_release: str = "market1501-duke-vit-bnneck:v2"
    idempotency_key: str = Field(min_length=4, max_length=200)
    actor: str = Field(min_length=1, max_length=100)


class CycleCreateResponse(BaseModel):
    created: bool
    cycle: dict[str, Any]


class EventListResponse(BaseModel):
    events: list[dict[str, Any]]
    last_event_id: int


class ActorRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=100)


class InboxImportResponse(BaseModel):
    created: bool
    batch: dict[str, Any]


class InboxImportRequest(ActorRequest):
    dataset: str = Field(min_length=1, max_length=100)
    source_path: str = Field(min_length=1, max_length=500)


class StageUpdateRequest(BaseModel):
    status: Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "PASS", "FAIL", "SKIPPED"]
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str | None = Field(default=None, max_length=2000)
    # Drift and validation reports include nested checks and provenance. Keep
    # the complete JSON payload so the UI and audit trail receive all evidence.
    metrics: dict[str, Any] = Field(default_factory=dict)
