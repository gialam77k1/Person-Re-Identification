from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np


class UpstreamError(RuntimeError):
    """Raised when Triton or Qdrant is unavailable or returns invalid data."""


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return json.loads(data.decode("utf-8")) if data else {}
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise UpstreamError(f"{url} returned HTTP {exc.code}: {response_body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise UpstreamError(f"Cannot reach upstream service at {url}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UpstreamError(f"Upstream service at {url} returned invalid JSON.") from exc


class TritonClient:
    def __init__(self, base_url: str, model_name: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def server_ready(self) -> bool:
        return self._plain_ready(f"{self.base_url}/v2/health/ready")

    def model_ready(self) -> bool:
        return self._plain_ready(
            f"{self.base_url}/v2/models/{self.model_name}/ready"
        )

    def _plain_ready(self, url: str) -> bool:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return False

    def infer(self, tensor: np.ndarray, input_name: str, output_name: str) -> np.ndarray:
        payload = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": list(tensor.shape),
                    "datatype": "FP32",
                    "data": tensor.reshape(-1).tolist(),
                }
            ],
            "outputs": [{"name": output_name}],
        }
        response = request_json(
            "POST",
            f"{self.base_url}/v2/models/{self.model_name}/infer",
            payload=payload,
            timeout=self.timeout,
        )
        outputs = response.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise UpstreamError("Triton response does not contain outputs.")
        output = next(
            (item for item in outputs if item.get("name") == output_name), outputs[0]
        )
        if "data" not in output:
            raise UpstreamError("Triton output does not contain inline tensor data.")
        embedding = np.asarray(output["data"], dtype=np.float32)
        shape = output.get("shape")
        if shape:
            embedding = embedding.reshape(shape)
        return embedding


class QdrantClient:
    def __init__(self, base_url: str, collection: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout

    def collection_info(self) -> dict[str, Any]:
        response = request_json(
            "GET",
            f"{self.base_url}/collections/{self.collection}",
            timeout=self.timeout,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise UpstreamError("Qdrant collection response has no result object.")
        return result

    def search(self, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        response = request_json(
            "POST",
            f"{self.base_url}/collections/{self.collection}/points/query",
            payload={
                "query": vector.astype(float).tolist(),
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
            },
            timeout=self.timeout,
        )
        result = response.get("result", {})
        points = result.get("points") if isinstance(result, dict) else None
        if not isinstance(points, list):
            raise UpstreamError("Qdrant query response has no result.points list.")
        return points
