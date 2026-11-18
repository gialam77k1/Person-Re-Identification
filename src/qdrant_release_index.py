from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JsonRequest = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


class QdrantIndexError(RuntimeError):
    """Raised when an immutable release index cannot be prepared safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_collection_name(release_id: str, manifest_sha256: str) -> str:
    """Return a deterministic, immutable collection name for any model version."""
    slug = re.sub(r"[^a-z0-9_]+", "_", release_id.lower()).strip("_")
    if not slug or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise QdrantIndexError("Release ID or manifest checksum is invalid.")
    return f"reid_reference_{slug}_{manifest_sha256[:12]}"[:200]


def environment_alias(environment: str) -> str:
    if environment == "production":
        return "reid_reference_active"
    if environment == "staging":
        return "reid_reference_staging"
    raise QdrantIndexError(f"Unsupported deployment environment: {environment}")


def _repository_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise QdrantIndexError("Reference artifact has no repository path.")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise QdrantIndexError(f"Reference artifact escapes the repository: {value}") from exc
    return path


@dataclass(frozen=True)
class ReferenceBundle:
    release_id: str
    manifest_sha256: str
    model_git_commit: str
    embeddings: np.ndarray
    pids: np.ndarray
    camids: np.ndarray
    paths: np.ndarray

    @property
    def count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])


def load_reference_bundle(
    manifest_path: Path,
    expected_release_id: str,
    *,
    root: Path = ROOT,
) -> ReferenceBundle:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QdrantIndexError(f"Cannot read release manifest {manifest_path}: {exc}") from exc
    release_id = str(manifest.get("release", {}).get("release_id") or "")
    if release_id != expected_release_id:
        raise QdrantIndexError("Reference release does not match the deployment request.")
    if manifest.get("validation", {}).get("status") != "PASS":
        raise QdrantIndexError("Reference artifacts have not passed the validation gate.")

    files = {
        item.get("role"): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    roles = {
        "embeddings": "reference_embeddings",
        "pids": "reference_pids",
        "camids": "reference_camids",
        "paths": "reference_paths",
    }
    loaded: dict[str, np.ndarray] = {}
    for name, role in roles.items():
        item = files.get(role)
        if not isinstance(item, dict):
            raise QdrantIndexError(f"Release manifest is missing {role}.")
        path = _repository_path(item.get("repository_path"), root)
        expected_sha = str(item.get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise QdrantIndexError(f"Reference artifact checksum mismatch: {role}")
        try:
            loaded[name] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise QdrantIndexError(f"Cannot load reference artifact {role}: {exc}") from exc

    embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
    pids, camids, paths = loaded["pids"], loaded["camids"], loaded["paths"]
    expected_dim = int(manifest.get("contract", {}).get("embedding_dim") or 0)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] != expected_dim:
        raise QdrantIndexError(
            f"Reference embedding shape {embeddings.shape} does not match (*, {expected_dim})."
        )
    if pids.shape != camids.shape or pids.shape != paths.shape or pids.shape != (len(embeddings),):
        raise QdrantIndexError("Reference embeddings and metadata are not aligned.")
    if not np.isfinite(embeddings).all():
        raise QdrantIndexError("Reference embeddings contain NaN or Inf.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise QdrantIndexError("Reference embeddings are not L2-normalized.")
    if not all(math.isfinite(float(value)) for value in pids) or not all(
        math.isfinite(float(value)) for value in camids
    ):
        raise QdrantIndexError("Reference identity metadata contains invalid values.")

    return ReferenceBundle(
        release_id=release_id,
        manifest_sha256=sha256_file(manifest_path),
        model_git_commit=str(manifest.get("source", {}).get("git_commit") or ""),
        embeddings=embeddings,
        pids=pids,
        camids=camids,
        paths=paths,
    )


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
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
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise QdrantIndexError(f"Qdrant returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise QdrantIndexError(f"Cannot reach Qdrant at {url}: {exc}") from exc
    try:
        value = json.loads(data.decode("utf-8")) if data else {}
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise QdrantIndexError("Qdrant returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise QdrantIndexError("Qdrant response is not a JSON object.")
    return value


class QdrantReleaseIndex:
    def __init__(
        self,
        base_url: str,
        *,
        request: JsonRequest | None = None,
        batch_size: int = 128,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.request = request or (
            lambda method, url, payload=None: request_json(method, url, payload)
        )
        self.batch_size = batch_size

    def _call(
        self, method: str, endpoint: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.request(method, f"{self.base_url}{endpoint}", payload)

    def collection_info(self, collection: str) -> dict[str, Any] | None:
        try:
            result = self._call("GET", f"/collections/{collection}").get("result")
        except QdrantIndexError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return result if isinstance(result, dict) else None

    def alias_target(self, alias: str) -> str | None:
        aliases = self._call("GET", "/aliases").get("result", {}).get("aliases", [])
        for item in aliases if isinstance(aliases, list) else []:
            if isinstance(item, dict) and item.get("alias_name") == alias:
                target = item.get("collection_name")
                return str(target) if target else None
        return None

    def _validate_collection(
        self, collection: str, bundle: ReferenceBundle
    ) -> dict[str, Any] | None:
        info = self.collection_info(collection)
        if info is None:
            return None
        vectors = info.get("config", {}).get("params", {}).get("vectors", {})
        size = vectors.get("size") if isinstance(vectors, dict) else None
        if size != bundle.dimension or int(info.get("points_count", -1)) != bundle.count:
            raise QdrantIndexError(
                f"Existing immutable collection {collection} has the wrong size or point count."
            )
        sample = self._call(
            "POST",
            f"/collections/{collection}/points/scroll",
            {"limit": 1, "with_payload": True, "with_vector": False},
        )
        points = sample.get("result", {}).get("points", [])
        payload = points[0].get("payload", {}) if points else {}
        if (
            payload.get("release_id") != bundle.release_id
            or payload.get("manifest_sha256") != bundle.manifest_sha256
        ):
            raise QdrantIndexError(
                f"Existing immutable collection {collection} has incompatible lineage."
            )
        return info

    def ensure_collection(self, bundle: ReferenceBundle) -> str:
        collection = release_collection_name(bundle.release_id, bundle.manifest_sha256)
        if self._validate_collection(collection, bundle) is not None:
            return collection

        self._call(
            "PUT",
            f"/collections/{collection}",
            {"vectors": {"size": bundle.dimension, "distance": "Cosine"}},
        )
        try:
            for start in range(0, bundle.count, self.batch_size):
                stop = min(start + self.batch_size, bundle.count)
                points = []
                for index in range(start, stop):
                    points.append(
                        {
                            "id": index,
                            "vector": bundle.embeddings[index].astype(float).tolist(),
                            "payload": {
                                "pid": int(bundle.pids[index]),
                                "camid": int(bundle.camids[index]),
                                "image_path": str(bundle.paths[index]),
                                "release_id": bundle.release_id,
                                "model_git_commit": bundle.model_git_commit,
                                "manifest_sha256": bundle.manifest_sha256,
                            },
                        }
                    )
                self._call(
                    "PUT",
                    f"/collections/{collection}/points?wait=true",
                    {"points": points},
                )
            for attempt in range(30):
                info = self._validate_collection(collection, bundle)
                if info is not None and info.get("status") in {"green", "yellow"}:
                    return collection
                if attempt < 29:
                    time.sleep(1)
        except Exception:
            self._call("DELETE", f"/collections/{collection}")
            raise
        self._call("DELETE", f"/collections/{collection}")
        raise QdrantIndexError(f"Qdrant collection {collection} did not become ready.")

    def switch_alias(self, alias: str, collection: str) -> str | None:
        previous = self.alias_target(alias)
        if previous == collection:
            return previous
        actions: list[dict[str, Any]] = []
        if previous:
            actions.append({"delete_alias": {"alias_name": alias}})
        actions.append(
            {
                "create_alias": {
                    "collection_name": collection,
                    "alias_name": alias,
                }
            }
        )
        self._call("POST", "/collections/aliases", {"actions": actions})
        if self.alias_target(alias) != collection:
            raise QdrantIndexError(f"Qdrant alias {alias} did not switch to {collection}.")
        return previous

    def restore_alias(self, alias: str, previous: str | None) -> None:
        current = self.alias_target(alias)
        if current == previous:
            return
        actions: list[dict[str, Any]] = []
        if current:
            actions.append({"delete_alias": {"alias_name": alias}})
        if previous:
            actions.append(
                {
                    "create_alias": {
                        "collection_name": previous,
                        "alias_name": alias,
                    }
                }
            )
        if actions:
            self._call("POST", "/collections/aliases", {"actions": actions})
        if self.alias_target(alias) != previous:
            raise QdrantIndexError(f"Could not restore Qdrant alias {alias}.")
