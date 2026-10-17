from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class ModelInfoResponse(BaseModel):
    release_id: str
    git_commit: str
    model_name: str
    embedding_dim: int
    input_shape: list[int]
    preprocessing_version: int
    preprocessing_sha256: str
    qdrant_collection: str


class EmbedResponse(BaseModel):
    model_release: str
    embedding_dim: int
    normalized: bool = True
    embedding: list[float]


class SearchResult(BaseModel):
    person_id: int
    camera_id: int
    similarity_score: float
    source_image: str


class SearchResponse(BaseModel):
    model_release: str
    top_k: int = Field(ge=1)
    results: list[SearchResult]
