from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.api.clients import UpstreamError
from src.api.config import ApiSettings
from src.api.image_processing import ImageValidationError
from src.api.schemas import (
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    SearchResponse,
)
from src.api.service import InferenceService


def create_app(service: InferenceService | None = None) -> FastAPI:
    inference = service or InferenceService(ApiSettings.from_env())
    api = FastAPI(
        title="Person Re-Identification API",
        version="1.1.0",
        description="Online embedding and retrieval through Triton and Qdrant.",
    )

    async def read_upload(image: UploadFile) -> bytes:
        limit = inference.settings.max_upload_bytes
        try:
            return await image.read(limit + 1)
        finally:
            await image.close()

    @api.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse | JSONResponse:
        checks = inference.readiness()
        response = ReadyResponse(
            status="ready" if all(checks.values()) else "not_ready", checks=checks
        )
        if response.status != "ready":
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @api.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> dict:
        return inference.model_info()

    @api.post("/embed", response_model=EmbedResponse)
    async def embed(image: UploadFile = File(...)) -> dict:
        raw = await read_upload(image)
        try:
            embedding = inference.embed(raw, image.content_type)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "model_release": inference.contract.release_id,
            "embedding_dim": inference.contract.embedding_dim,
            "normalized": True,
            "embedding": embedding.tolist(),
        }

    @api.post("/search", response_model=SearchResponse)
    async def search(
        image: UploadFile = File(...),
        top_k: int | None = Query(default=None, ge=1),
    ) -> dict:
        effective_top_k = top_k or inference.settings.default_top_k
        if effective_top_k > inference.settings.max_top_k:
            raise HTTPException(
                status_code=422,
                detail=f"top_k must not exceed {inference.settings.max_top_k}.",
            )
        raw = await read_upload(image)
        try:
            results = inference.search(raw, image.content_type, effective_top_k)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "model_release": inference.contract.release_id,
            "top_k": effective_top_k,
            "results": results,
        }

    return api


app = create_app()
