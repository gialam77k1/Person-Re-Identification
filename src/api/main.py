from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import Response

from src.api.clients import UpstreamError
from src.api.config import ApiSettings
from src.api.image_processing import ImageValidationError
from src.api.metrics import ApiMetrics
from src.api.schemas import (
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    SearchResponse,
)
from src.api.service import InferenceService


def create_app(service: InferenceService | None = None) -> FastAPI:
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        if service is None:
            api.state.inference = InferenceService(ApiSettings.from_env(), metrics=metrics)
        yield

    api = FastAPI(
        title="Person Re-Identification API",
        version="1.1.0",
        description="Online embedding and retrieval through Triton and Qdrant.",
        lifespan=lifespan,
    )
    if service is not None:
        service.set_metrics(metrics)
        api.state.inference = service

    @api.middleware("http")
    async def observe_requests(request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        started = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            endpoint = request.url.path if request.url.path in {
                "/health", "/ready", "/model-info", "/embed", "/search"
            } else "unmatched"
            metrics.requests.labels(request.method, endpoint, str(status)).inc()
            metrics.request_duration.labels(request.method, endpoint).observe(
                perf_counter() - started
            )

    def inference() -> InferenceService:
        configured = getattr(api.state, "inference", None)
        if configured is None:
            raise RuntimeError("Inference service has not completed startup.")
        return configured

    async def read_upload(image: UploadFile) -> bytes:
        limit = inference().settings.max_upload_bytes
        try:
            return await image.read(limit + 1)
        finally:
            await image.close()

    @api.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @api.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse | JSONResponse:
        checks = inference().readiness()
        response = ReadyResponse(
            status="ready" if all(checks.values()) else "not_ready", checks=checks
        )
        if response.status != "ready":
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @api.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> dict:
        return inference().model_info()

    @api.post("/embed", response_model=EmbedResponse)
    async def embed(image: UploadFile = File(...)) -> dict:
        raw = await read_upload(image)
        try:
            active = inference()
            embedding = active.embed(raw, image.content_type)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "model_release": active.contract.release_id,
            "embedding_dim": active.contract.embedding_dim,
            "normalized": True,
            "embedding": embedding.tolist(),
        }

    @api.post("/search", response_model=SearchResponse)
    async def search(
        image: UploadFile = File(...),
        top_k: int | None = Query(default=None, ge=1),
    ) -> dict:
        active = inference()
        effective_top_k = top_k or active.settings.default_top_k
        if effective_top_k > active.settings.max_top_k:
            raise HTTPException(
                status_code=422,
                detail=f"top_k must not exceed {active.settings.max_top_k}.",
            )
        raw = await read_upload(image)
        try:
            results = active.search(raw, image.content_type, effective_top_k)
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "model_release": active.contract.release_id,
            "top_k": effective_top_k,
            "results": results,
        }

    return api


app = create_app()
