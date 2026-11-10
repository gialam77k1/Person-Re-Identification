from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.control_api.config import ControlSettings
from src.control_api.lifecycle import LifecycleController
from src.control_api.releases import ReleaseResolver
from src.control_api.store import CycleStore


REID_FILE = re.compile(
    r"^(?P<person_id>-?\d+)_c(?P<camera_id>\d+)_.*\.(?:jpe?g|png)$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ActiveCycleError(RuntimeError):
    def __init__(self, cycle: dict[str, Any]) -> None:
        self.cycle = cycle
        super().__init__(
            f"Dataset already has active cycle {cycle['id']} ({cycle['status']})."
        )


class InvalidBatchError(ValueError):
    pass


class DataInboxService:
    """Register immutable labeled batches without coupling ingestion to one folder."""

    def __init__(
        self,
        settings: ControlSettings,
        store: CycleStore,
        lifecycle: LifecycleController,
        releases: ReleaseResolver | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.lifecycle = lifecycle
        self.releases = releases or ReleaseResolver(settings, store)

    def import_batch(
        self, *, source_path: str, dataset: str, actor: str
    ) -> tuple[dict[str, Any], bool]:
        source, relative_source = self._resolve_source(source_path)
        manifest = self._load_manifest(source)
        manifest_dataset = str(manifest.get("dataset", dataset)).strip().lower()
        if manifest_dataset != dataset.lower():
            raise InvalidBatchError(
                f"Manifest dataset {manifest_dataset!r} does not match {dataset!r}."
            )
        if dataset not in self.settings.allowed_datasets:
            raise InvalidBatchError(
                f"Dataset must be one of: {', '.join(self.settings.allowed_datasets)}"
            )
        active = self.store.active_cycle_for_dataset(dataset)
        if active:
            raise ActiveCycleError(active)

        scan = self._scan(source, manifest)
        failures = self._threshold_failures(scan)
        batch, created = self.store.create_inbox_batch(
            {
                **scan,
                "dataset": dataset,
                "source_split": relative_source,
                "status": "REJECTED" if failures else "VALIDATED",
                "actor": actor,
                "summary": {
                    "batch_id": str(manifest.get("batch_id", source.name)),
                    "source_path": relative_source,
                    "manifest": "manifest.json" if (source / "manifest.json").is_file() else None,
                    "thresholds": self.policy(),
                    "validation_failures": failures,
                },
            }
        )
        if (
            not created
            and batch["status"] in {"BLOCKED", "FAILED"}
            and not batch.get("cycle_id")
        ):
            # A release collision or transient dispatch error may be resolved by
            # the operator. Reuse the immutable batch record and fingerprint,
            # but allow the lifecycle dispatch to be attempted again.
            batch = self.store.update_inbox_batch(
                batch["id"], status="VALIDATED", error=None
            )
            created = True
        if not created or failures:
            return batch, created

        try:
            release_pair = self.releases.resolve()
        except ValueError as exc:
            self.store.update_inbox_batch(batch["id"], status="BLOCKED", error=str(exc))
            raise InvalidBatchError(str(exc)) from exc
        try:
            cycle, _ = self.lifecycle.start(
                dataset=dataset,
                source_release=release_pair.source,
                target_release=release_pair.target,
                idempotency_key=f"inbox:{scan['fingerprint']}",
                actor=actor,
                inbox_batch_id=batch["id"],
                dataset_revision=scan["fingerprint"],
                dataset_path=relative_source,
            )
        except Exception as exc:
            self.store.update_inbox_batch(batch["id"], status="FAILED", error=str(exc))
            raise
        self.store.record_event(
            cycle["id"],
            "release.resolved",
            "Source champion and next target release resolved",
            {
                "source_release": release_pair.source,
                "target_release": release_pair.target,
                "source_kind": release_pair.source_kind,
            },
        )
        return (
            self.store.update_inbox_batch(
                batch["id"], status="DISPATCHED", cycle_id=cycle["id"]
            ),
            True,
        )

    def policy(self) -> dict[str, int]:
        return {
            "min_valid_images": self.settings.inbox_min_valid_images,
            "min_identities": self.settings.inbox_min_identities,
            "min_cameras": self.settings.inbox_min_cameras,
        }

    def discover_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        inbox_root = self.settings.dataset_root / "inbox"
        if inbox_root.is_dir():
            for directory in sorted(path for path in inbox_root.iterdir() if path.is_dir()):
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    sources.append(
                        {
                            "source_path": directory.relative_to(
                                self.settings.dataset_root
                            ).as_posix(),
                            "batch_id": manifest.get("batch_id", directory.name),
                            "dataset": manifest.get("dataset", ""),
                        }
                    )
                except (OSError, json.JSONDecodeError):
                    continue
        legacy = self.settings.dataset_root / "dukemtmc-reid" / "bounding_box_train"
        if legacy.is_dir():
            sources.insert(
                0,
                {
                    "source_path": "dukemtmc-reid/bounding_box_train",
                    "batch_id": "duke-initial-train",
                    "dataset": "dukemtmc-reid",
                },
            )
        return sources

    def _resolve_source(self, source_path: str) -> tuple[Path, str]:
        relative = Path(source_path.strip())
        if not source_path.strip() or relative.is_absolute() or ".." in relative.parts:
            raise InvalidBatchError("source_path must be relative to CONTROL_DATASET_ROOT.")
        root = self.settings.dataset_root.resolve()
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_dir():
            raise InvalidBatchError(f"Batch directory not found: {source_path}")
        return source, source.relative_to(root).as_posix()

    @staticmethod
    def _load_manifest(source: Path) -> dict[str, Any]:
        path = source / "manifest.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidBatchError(f"Invalid manifest.json: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidBatchError("manifest.json must contain a JSON object.")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise InvalidBatchError("manifest.json must contain a non-empty samples list.")
        return payload

    def _scan(self, source: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        candidates = self._manifest_samples(source, manifest) if manifest else self._legacy_samples(source)
        aggregate = hashlib.sha256()
        identities: set[int] = set()
        cameras: set[int] = set()
        content_hashes: set[str] = set()
        valid_images = rejected_images = duplicate_images = total_bytes = 0

        for sample in candidates:
            path = sample["path"]
            person_id = sample["person_id"]
            camera_id = sample["camera_id"]
            if person_id < 0 or camera_id < 1 or path.suffix.lower() not in IMAGE_SUFFIXES:
                rejected_images += 1
                continue
            content_sha, size, valid_magic = self._hash_image(path)
            if not valid_magic:
                rejected_images += 1
                continue
            if content_sha in content_hashes:
                duplicate_images += 1
                continue
            content_hashes.add(content_sha)
            identities.add(person_id)
            cameras.add(camera_id)
            total_bytes += size
            valid_images += 1
            relative = path.relative_to(source).as_posix()
            aggregate.update(
                f"{relative}\0{person_id}\0{camera_id}\0{size}\0{content_sha}\n".encode()
            )

        if not candidates:
            raise InvalidBatchError(f"No image candidates found in {source}")
        return {
            "fingerprint": aggregate.hexdigest(),
            "valid_images": valid_images,
            "rejected_images": rejected_images,
            "duplicate_images": duplicate_images,
            "identities": len(identities),
            "cameras": len(cameras),
            "total_bytes": total_bytes,
        }

    @staticmethod
    def _manifest_samples(source: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        result = []
        for index, item in enumerate(manifest["samples"]):
            if not isinstance(item, dict):
                raise InvalidBatchError(f"manifest sample {index} must be an object.")
            relative = Path(str(item.get("image", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise InvalidBatchError(f"Unsafe image path in manifest sample {index}.")
            path = (source / relative).resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise InvalidBatchError(f"Missing image in manifest sample {index}: {relative}")
            try:
                person_id = int(item["person_id"])
                camera_id = int(item["camera_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidBatchError(
                    f"Invalid person_id/camera_id in manifest sample {index}."
                ) from exc
            result.append({"path": path, "person_id": person_id, "camera_id": camera_id})
        return result

    @staticmethod
    def _legacy_samples(source: Path) -> list[dict[str, Any]]:
        result = []
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = REID_FILE.match(path.name)
            if match:
                result.append(
                    {
                        "path": path,
                        "person_id": int(match.group("person_id")),
                        "camera_id": int(match.group("camera_id")),
                    }
                )
            else:
                result.append({"path": path, "person_id": -1, "camera_id": -1})
        return result

    def _threshold_failures(self, scan: dict[str, Any]) -> list[str]:
        failures = []
        for field, setting in (
            ("valid_images", "inbox_min_valid_images"),
            ("identities", "inbox_min_identities"),
            ("cameras", "inbox_min_cameras"),
        ):
            minimum = int(getattr(self.settings, setting))
            if int(scan[field]) < minimum:
                failures.append(f"{field} {scan[field]} < {minimum}")
        return failures

    @staticmethod
    def _hash_image(path: Path) -> tuple[str, int, bool]:
        digest = hashlib.sha256()
        size = 0
        prefix = b""
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if not prefix:
                    prefix = chunk[:8]
                size += len(chunk)
                digest.update(chunk)
        is_jpeg = prefix.startswith(b"\xff\xd8\xff")
        is_png = prefix.startswith(b"\x89PNG\r\n\x1a\n")
        return digest.hexdigest(), size, size > 0 and (is_jpeg or is_png)
