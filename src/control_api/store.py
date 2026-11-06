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
                """
            )

    def create_cycle(
        self,
        *,
        dataset: str,
        source_release: str,
        target_release: str,
        idempotency_key: str,
        actor: str,
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
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    idempotency_key,
                    dataset,
                    source_release,
                    target_release,
                    CycleStatus.PENDING.value,
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
                {"dataset": dataset, "idempotency_key": idempotency_key},
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
