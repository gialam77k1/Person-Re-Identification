from __future__ import annotations

import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class EmbeddingSampler:
    def __init__(self, path: Path | None, sample_rate: float, max_samples: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._count = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self._count = sum(1 for _ in path.open("r", encoding="utf-8"))
            except FileNotFoundError:
                pass

    def observe(self, embedding: np.ndarray, release_id: str) -> None:
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
