from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from src.control_api.clients import ControlUpstreamError, HttpClient
from src.control_api.config import ControlSettings
from src.control_api.store import CycleStore


VERSION = re.compile(r"^(?P<name>[^:]+):v(?P<version>\d+)(?:[-._][A-Za-z0-9.-]+)?$")


@dataclass(frozen=True)
class ReleasePair:
    source: str
    target: str
    source_kind: str


def next_release(source: str, target_family: str) -> str:
    match = VERSION.fullmatch(source.strip())
    if match is None:
        raise ValueError(f"Cannot derive a model version from release {source!r}.")
    version = int(match.group("version")) + 1
    family = target_family.strip() or match.group("name")
    return f"{family}:v{version}"


def release_slug(release_id: str) -> str:
    return release_id.replace(":", "-").replace("_", "-")


class ReleaseResolver:
    """Resolve the immutable source/target pair for a new inbox cycle.

    MLflow's champion alias is authoritative. A successfully promoted local
    cycle is the offline fallback, and the configured bootstrap release is
    used only before the first promotion.
    """

    def __init__(
        self,
        settings: ControlSettings,
        store: CycleStore,
        client: HttpClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.client = client or HttpClient()

    def resolve(self) -> ReleasePair:
        source = self._registry_champion()
        source_kind = "mlflow_champion"
        if not source:
            source = self.store.latest_promoted_release()
            source_kind = "promoted_cycle"
        if not source:
            source = self.settings.inbox_bootstrap_release
            source_kind = "bootstrap"
        target = next_release(source, self.settings.inbox_target_family)
        existing = self.store.cycle_for_target_release(target)
        if existing:
            raise ValueError(
                f"Target release {target} already belongs to cycle {existing['id']} "
                f"({existing['status']}); promote or resolve that candidate before importing another batch."
            )
        return ReleasePair(
            source=source,
            target=target,
            source_kind=source_kind,
        )

    def _registry_champion(self) -> str | None:
        if not self.settings.registry_model_name:
            return None
        query = urllib.parse.urlencode(
            {
                "name": self.settings.registry_model_name,
                "alias": self.settings.registry_champion_alias,
            }
        )
        url = (
            f"{self.settings.mlflow_url}/api/2.0/mlflow/registered-models/"
            f"get-by-alias?{query}"
        )
        try:
            payload = self.client.get_json(url)
        except ControlUpstreamError:
            return None
        version = payload.get("model_version", payload)
        if not isinstance(version, dict):
            return None
        tags = self._tags(version.get("tags"))
        release_id = tags.get("release.id") or tags.get("release_id")
        if release_id and VERSION.fullmatch(release_id):
            return release_id
        return None

    @staticmethod
    def _tags(value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        if isinstance(value, list):
            return {
                str(item.get("key")): str(item.get("value"))
                for item in value
                if isinstance(item, dict) and item.get("key") is not None
            }
        return {}
