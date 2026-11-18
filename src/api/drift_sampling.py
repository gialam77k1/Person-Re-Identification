from __future__ import annotations

import json
import random
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class EmbeddingSampler:
    def __init__(
        self,
        path: Path | None,
        sample_rate: float,
        max_samples: int,
        release_id: str | None = None,
    ) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.release_id = release_id
        self._lock = threading.Lock()
        self._count = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if release_id:
                self._activate_release(release_id)
            try:
                self._count = self._count_release_rows(path, release_id)
            except FileNotFoundError:
                pass

    @staticmethod
    def _release_slug(release_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", release_id).strip("-.") or "unknown"

    @staticmethod
    def _count_release_rows(path: Path, release_id: str | None) -> int:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if release_id is None or json.loads(line).get("release_id") == release_id:
                    count += 1
        return count

    @staticmethod
    def _release_ids(paths: tuple[Path, ...]) -> set[str]:
        release_ids: set[str] = set()
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line).get("release_id")
                    if isinstance(value, str) and value:
                        release_ids.add(value)
        return release_ids

    def _activate_release(self, release_id: str) -> None:
        """Atomically start a fresh sampling window when serving release changes."""
        assert self.path is not None
        marker = self.path.parent / ".active_release.json"
        previous_release: str | None = None
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("release_id"), str):
                previous_release = value["release_id"]
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            previous_release = None

        sample_paths = (
            self.path,
            self.path.with_suffix(f"{self.path.suffix}.previous"),
        )
        observed_releases = self._release_ids(sample_paths)
        if previous_release is None and len(observed_releases) == 1:
            previous_release = next(iter(observed_releases))
        must_rotate = (
            (previous_release is not None and previous_release != release_id)
            or bool(observed_releases - {release_id})
        )
        if must_rotate:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_label = previous_release or "mixed-legacy-window"
            archive = self.path.parent / "archive" / self._release_slug(archive_label)
            archive.mkdir(parents=True, exist_ok=True, mode=0o700)
            for source, label in zip(sample_paths, ("current", "previous")):
                if source.is_file():
                    source.replace(archive / f"{timestamp}-{label}.jsonl")

        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "release_id": release_id,
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(marker)

    def observe(self, embedding: np.ndarray, release_id: str) -> None:
        if self.release_id is not None and release_id != self.release_id:
            raise ValueError(
                f"Sampler release mismatch: {release_id!r} != {self.release_id!r}"
            )
        if self.path is None or random.random() > self.sample_rate:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "release_id": release_id,
            "embedding": embedding.astype(float).tolist(),
        }
        with self._lock:
            if self._count >= self.max_samples:
                rotated = self.path.with_suffix(f"{self.path.suffix}.previous")
                self.path.replace(rotated)
                self._count = 0
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.path.chmod(0o600)
            self._count += 1
