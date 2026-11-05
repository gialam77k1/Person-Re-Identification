from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ControlUpstreamError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    def get_json(self, url: str) -> dict[str, Any]:
        payload = self._request(url)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlUpstreamError(f"{url} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ControlUpstreamError(f"{url} did not return a JSON object")
        return value

    def get_text(self, url: str) -> str:
        return self._request(url).decode("utf-8", errors="replace")

    def prometheus_query(self, base_url: str, query: str) -> dict[str, Any]:
        encoded = urllib.parse.urlencode({"query": query})
        return self.get_json(f"{base_url}/api/v1/query?{encoded}")

    def _request(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not 200 <= response.status < 300:
                    raise ControlUpstreamError(f"{url} returned HTTP {response.status}")
                return response.read()
        except urllib.error.HTTPError as exc:
            raise ControlUpstreamError(f"{url} returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ControlUpstreamError(f"Cannot reach {url}: {exc}") from exc
