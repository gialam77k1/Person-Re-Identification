from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CycleCreate(BaseModel):
    dataset: str
    source_release: str = "market1501-vit-bnneck:v1"
    target_release: str = "market1501-msmt17-vit-bnneck:v2"
    idempotency_key: str = Field(min_length=4, max_length=200)
    actor: str = Field(min_length=1, max_length=100)


class CycleCreateResponse(BaseModel):
    created: bool
    cycle: dict[str, Any]


class EventListResponse(BaseModel):
    events: list[dict[str, Any]]
    last_event_id: int
