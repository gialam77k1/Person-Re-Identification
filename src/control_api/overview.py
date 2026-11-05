from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.control_api.clients import ControlUpstreamError, HttpClient
from src.control_api.config import ControlSettings


class OverviewService:
    def __init__(self, settings: ControlSettings, client: HttpClient | None = None) -> None:
        self.settings = settings
        self.client = client or HttpClient()

    def services(self) -> list[dict[str, Any]]:
        definitions: list[tuple[str, str, Callable[[], Any]]] = [
            (
                "MLflow",
                self.settings.public_mlflow_url,
                lambda: self.client.get_text(f"{self.settings.mlflow_url}/health"),
            ),
            (
                "Prometheus",
                self.settings.public_grafana_url,
                lambda: self.client.get_text(f"{self.settings.prometheus_url}/-/ready"),
            ),
            (
                "Qdrant",
                self.settings.public_qdrant_url,
                lambda: self.client.get_json(
                    f"{self.settings.qdrant_url}/collections/{self.settings.qdrant_collection}"
                ),
            ),
            (
                "Triton",
                f"{self.settings.triton_url}/v2/health/ready",
                lambda: self.client.get_text(
                    f"{self.settings.triton_url}/v2/models/{self.settings.triton_model}/ready"
                ),
            ),
            (
                "Inference API",
                self.settings.inference_api_url,
                lambda: self.client.get_json(f"{self.settings.inference_api_url}/ready"),
            ),
        ]
        results = [self._probe(name, url, probe) for name, url, probe in definitions]
        results.extend(
            [
                {"name": "Grafana", "status": "LINK", "url": self.settings.public_grafana_url},
                {"name": "MinIO", "status": "LINK", "url": self.settings.public_minio_url},
                {
                    "name": "GitHub Actions",
                    "status": "LINK",
                    "url": self.settings.public_github_actions_url,
                },
            ]
        )
        return results

    def overview(self) -> dict[str, Any]:
        services = self.services()
        model = self._safe_json(f"{self.settings.inference_api_url}/model-info")
        collection = self._safe_json(
            f"{self.settings.qdrant_url}/collections/{self.settings.qdrant_collection}"
        )
        drift = self._drift()
        ready_count = sum(item["status"] == "READY" for item in services)
        checked_count = sum(item["status"] != "LINK" for item in services)
        return {
            "services": {"ready": ready_count, "checked": checked_count, "items": services},
            "model": model,
            "collection": collection.get("result", collection),
            "drift": drift,
        }

    def _drift(self) -> dict[str, Any]:
        try:
            response = self.client.prometheus_query(
                self.settings.prometheus_url, "reid_embedding_drift_detected"
            )
            results = response.get("data", {}).get("result", [])
            if not results:
                return {"status": "NO_DATA", "detected": None}
            sample = results[0]
            value = sample.get("value", [None, None])
            return {
                "status": "DRIFT" if float(value[1]) == 1.0 else "STABLE",
                "detected": float(value[1]) == 1.0,
                "labels": sample.get("metric", {}),
            }
        except (ControlUpstreamError, TypeError, ValueError, IndexError):
            return {"status": "UNAVAILABLE", "detected": None}

    def _safe_json(self, url: str) -> dict[str, Any]:
        try:
            return self.client.get_json(url)
        except ControlUpstreamError as exc:
            return {"status": "UNAVAILABLE", "error": str(exc)}

    @staticmethod
    def _probe(name: str, url: str, probe: Callable[[], Any]) -> dict[str, Any]:
        try:
            detail = probe()
            return {"name": name, "status": "READY", "url": url, "detail": detail}
        except ControlUpstreamError as exc:
            return {"name": name, "status": "UNAVAILABLE", "url": url, "error": str(exc)}
