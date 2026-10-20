from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "reid_api_requests_total",
            "HTTP requests handled by the ReID API.",
            ("method", "endpoint", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "reid_api_request_duration_seconds",
            "End-to-end API request latency.",
            ("method", "endpoint"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )
        self.upstream_duration = Histogram(
            "reid_upstream_duration_seconds",
            "Latency of Triton and Qdrant operations.",
            ("service", "operation", "outcome"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )
        self.embedding_norm = Histogram(
            "reid_embedding_norm",
            "L2 norm of raw model embeddings before normalization.",
            buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2, 5, 10, 25, 50),
            registry=self.registry,
        )
        self.similarity = Histogram(
            "reid_similarity_score",
            "Similarity scores returned by Qdrant.",
            buckets=(-1, -0.5, 0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1),
            registry=self.registry,
        )
        self.search_queries = Counter(
            "reid_search_queries_total",
            "Search queries grouped by whether a match crossed the configured threshold.",
            ("outcome",),
            registry=self.registry,
        )
        self.model_release = Gauge(
            "reid_model_release_info",
            "Currently served model release.",
            ("release_id", "git_commit"),
            registry=self.registry,
        )
        self.service_ready = Gauge(
            "reid_service_ready",
            "Readiness of serving dependencies (1 ready, 0 unavailable).",
            ("component",),
            registry=self.registry,
        )
        self.collection_points = Gauge(
            "reid_qdrant_collection_points",
            "Number of points in the configured Qdrant collection.",
            ("collection",),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
