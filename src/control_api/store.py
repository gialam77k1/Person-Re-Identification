from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.control_api.state import PIPELINE_STAGES, CycleStatus, StageStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CycleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS cycles (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    dataset TEXT NOT NULL,
                    source_release TEXT NOT NULL,
                    target_release TEXT NOT NULL,
                    status TEXT NOT NULL,
                    workflow_run_id TEXT,
                    inbox_batch_id TEXT,
                    dataset_revision TEXT,
                    dataset_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stages (
                    cycle_id TEXT NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    detail TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    PRIMARY KEY (cycle_id, name)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retraining_decisions (
                    cycle_id TEXT PRIMARY KEY REFERENCES cycles(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inbox_batches (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    dataset TEXT NOT NULL,
                    source_split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    valid_images INTEGER NOT NULL,
                    rejected_images INTEGER NOT NULL,
                    duplicate_images INTEGER NOT NULL,
                    identities INTEGER NOT NULL,
                    cameras INTEGER NOT NULL,
                    total_bytes INTEGER NOT NULL,
                    cycle_id TEXT REFERENCES cycles(id),
                    actor TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cycle_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(cycles)").fetchall()
            }
            for name in ("inbox_batch_id", "dataset_revision", "dataset_path"):
                if name not in cycle_columns:
                    connection.execute(f"ALTER TABLE cycles ADD COLUMN {name} TEXT")

    def active_cycle_for_dataset(self, dataset: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM cycles
                WHERE dataset = ? AND status IN (?, ?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    dataset,
                    CycleStatus.PENDING.value,
                    CycleStatus.RUNNING.value,
                    CycleStatus.WAITING_APPROVAL.value,
                ),
            ).fetchone()
        return self.get_cycle(str(row["id"])) if row else None

    def create_inbox_batch(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        now = utc_now()
        batch_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM inbox_batches WHERE fingerprint = ?",
                (payload["fingerprint"],),
            ).fetchone()
            if existing:
                return self.get_inbox_batch(str(existing["id"])), False
            connection.execute(
                """
                INSERT INTO inbox_batches (
                    id, fingerprint, dataset, source_split, status, valid_images,
                    rejected_images, duplicate_images, identities, cameras,
                    total_bytes, actor, summary_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    payload["fingerprint"],
                    payload["dataset"],
                    payload["source_split"],
                    payload["status"],
                    payload["valid_images"],
                    payload["rejected_images"],
                    payload["duplicate_images"],
                    payload["identities"],
                    payload["cameras"],
                    payload["total_bytes"],
                    payload["actor"],
                    json.dumps(payload.get("summary", {}), sort_keys=True),
                    now,
                    now,
                ),
            )
        return self.get_inbox_batch(batch_id), True

    def update_inbox_batch(
        self,
        batch_id: str,
        *,
        status: str,
        cycle_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE inbox_batches
                SET status = ?, cycle_id = COALESCE(?, cycle_id), error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, cycle_id, error, utc_now(), batch_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(batch_id)
        return self.get_inbox_batch(batch_id)

    def get_inbox_batch(self, batch_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inbox_batches WHERE id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        payload = dict(row)
        payload["summary"] = json.loads(payload.pop("summary_json"))
        return payload

    def list_inbox_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM inbox_batches ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get_inbox_batch(str(row["id"])) for row in rows]

    def create_cycle(
        self,
        *,
        dataset: str,
        source_release: str,
        target_release: str,
        idempotency_key: str,
        actor: str,
        inbox_batch_id: str | None = None,
        dataset_revision: str | None = None,
        dataset_path: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        now = utc_now()
        cycle_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM cycles WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing:
                return self.get_cycle(str(existing["id"])), False
            connection.execute(
                """
                INSERT INTO cycles (
                    id, idempotency_key, dataset, source_release, target_release,
                    status, inbox_batch_id, dataset_revision, dataset_path,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    idempotency_key,
                    dataset,
                    source_release,
                    target_release,
                    CycleStatus.PENDING.value,
                    inbox_batch_id,
                    dataset_revision,
                    dataset_path,
                    now,
                    now,
                ),
            )
            connection.executemany(
                "INSERT INTO stages (cycle_id, name, position, status) VALUES (?, ?, ?, ?)",
                [
                    (cycle_id, name, position, StageStatus.PENDING.value)
                    for position, name in enumerate(PIPELINE_STAGES)
                ],
            )
            self._insert_event(connection, cycle_id, "cycle.created", "Cycle created", {})
            self._insert_audit(
                connection,
                cycle_id,
                "cycle.create",
                actor,
                {
                    "dataset": dataset,
                    "idempotency_key": idempotency_key,
                    "inbox_batch_id": inbox_batch_id,
                    "dataset_revision": dataset_revision,
                    "dataset_path": dataset_path,
                },
            )
        return self.get_cycle(cycle_id), True

    def list_cycles(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM cycles ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get_cycle(str(row["id"])) for row in rows]

    def get_cycle(self, cycle_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            cycle = connection.execute(
                "SELECT * FROM cycles WHERE id = ?", (cycle_id,)
            ).fetchone()
            if cycle is None:
                raise KeyError(cycle_id)
            stages = connection.execute(
                "SELECT * FROM stages WHERE cycle_id = ? ORDER BY position", (cycle_id,)
            ).fetchall()
        payload = dict(cycle)
        payload["stages"] = [dict(stage) for stage in stages]
        return payload

    def update_cycle(
        self,
        cycle_id: str,
        *,
        status: CycleStatus | None = None,
        workflow_run_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        updates: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        for column, value in (
            ("status", status),
            ("workflow_run_id", workflow_run_id),
            ("error", error),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(getattr(value, "value", value))
        values.append(cycle_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE cycles SET {', '.join(updates)} WHERE id = ?", values
            )
            if cursor.rowcount != 1:
                raise KeyError(cycle_id)
        return self.get_cycle(cycle_id)

    def update_stage(
        self,
        cycle_id: str,
        name: str,
        status: StageStatus,
        *,
        progress: float | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        if name not in PIPELINE_STAGES:
            raise ValueError(f"Unknown pipeline stage: {name}")
        progress_value = progress
        if progress_value is None:
            progress_value = 1.0 if status == StageStatus.PASS else 0.0
        if not 0.0 <= progress_value <= 1.0:
            raise ValueError("Stage progress must be between 0 and 1.")
        now = utc_now()
        started_at = now if status == StageStatus.RUNNING else None
        finished_at = now if status in {StageStatus.PASS, StageStatus.FAIL, StageStatus.SKIPPED} else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE stages SET status = ?, progress = ?, detail = COALESCE(?, detail),
                    started_at = COALESCE(started_at, ?), finished_at = COALESCE(?, finished_at)
                WHERE cycle_id = ? AND name = ?
                """,
                (status.value, progress_value, detail, started_at, finished_at, cycle_id, name),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"{cycle_id}:{name}")
            self._insert_event(
                connection,
                cycle_id,
                "stage.updated",
                f"{name}: {status.value}",
                {"stage": name, "status": status.value, "progress": progress_value, "detail": detail},
            )
        return self.get_cycle(cycle_id)

    def events(self, cycle_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE cycle_id = ? AND id > ? ORDER BY id",
                (cycle_id, after_id),
            ).fetchall()
        return [self._decode_payload(row) for row in rows]

    def latest_stage_metrics(self, stage: str) -> dict[str, Any] | None:
        """Return the newest metrics callback for a pipeline stage.

        Metrics are stored as lifecycle events so that runner callbacks remain the
        single source of truth.  Reading them here also avoids depending on a
        runner-local textfile path for Prometheus batch-job telemetry.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT events.*, cycles.dataset, cycles.source_release,
                    cycles.target_release
                FROM events
                JOIN cycles ON cycles.id = events.cycle_id
                WHERE events.kind = 'stage.metrics'
                ORDER BY events.id DESC
                """
            ).fetchall()
        for row in rows:
            decoded = self._decode_payload(row)
            payload = decoded.get("payload", {})
            if payload.get("stage") == stage and isinstance(payload.get("metrics"), dict):
                return decoded
        return None

    def latest_cycle_stage_metrics(
        self, cycle_id: str, stage: str
    ) -> dict[str, Any] | None:
        for event in reversed(self.events(cycle_id)):
            payload = event.get("payload", {})
            if (
                event.get("kind") == "stage.metrics"
                and payload.get("stage") == stage
                and isinstance(payload.get("metrics"), dict)
            ):
                return event
        return None

    def audit(self, cycle_id: str, action: str, actor: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._insert_audit(connection, cycle_id, action, actor, payload)

    def record_event(
        self, cycle_id: str, kind: str, message: str, payload: dict[str, Any] | None = None
    ) -> None:
        with self._connect() as connection:
            self._insert_event(connection, cycle_id, kind, message, payload or {})

    def audit_entries(self, cycle_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_log WHERE cycle_id = ? ORDER BY id", (cycle_id,)
            ).fetchall()
        return [self._decode_payload(row) for row in rows]

    def save_retraining_decision(
        self, cycle_id: str, decision: dict[str, Any], sha256: str
    ) -> dict[str, Any]:
        encoded = json.dumps(decision, sort_keys=True, separators=(",", ":"))
        created_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retraining_decisions (
                    cycle_id, status, decision_json, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cycle_id) DO UPDATE SET
                    status = excluded.status,
                    decision_json = excluded.decision_json,
                    sha256 = excluded.sha256,
                    created_at = excluded.created_at
                """,
                (cycle_id, decision["status"], encoded, sha256, created_at),
            )
        return self.get_retraining_decision(cycle_id)

    def get_retraining_decision(self, cycle_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM retraining_decisions WHERE cycle_id = ?", (cycle_id,)
            ).fetchone()
        if row is None:
            raise KeyError(cycle_id)
        result = dict(row)
        result["decision"] = json.loads(result.pop("decision_json"))
        return result

    @staticmethod
    def _decode_payload(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        cycle_id: str,
        kind: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events (cycle_id, kind, message, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (cycle_id, kind, message, json.dumps(payload), utc_now()),
        )

    @staticmethod
    def _insert_audit(
        connection: sqlite3.Connection,
        cycle_id: str,
        action: str,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO audit_log (cycle_id, action, actor, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (cycle_id, action, actor, json.dumps(payload), utc_now()),
        )
