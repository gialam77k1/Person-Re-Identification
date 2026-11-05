from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx2

from src.control_api.config import ControlSettings
from src.control_api.lifecycle import LifecycleController
from src.control_api.main import create_app
from src.control_api.store import CycleStore


class FakeOverview:
    def services(self) -> list[dict[str, Any]]:
        return [{"name": "MLflow", "status": "READY", "url": "http://mlflow"}]

    def overview(self) -> dict[str, Any]:
        return {"services": {"ready": 1, "checked": 1}, "drift": {"status": "DRIFT"}}


class FakeDispatcher:
    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict[str, Any]:
        return {"workflow": workflow, "inputs": inputs}


class ControlApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        settings = replace(
            ControlSettings.from_env(),
            database_path=Path(self.temporary.name) / "control.db",
            allowed_datasets=("MSMT17",),
        )
        self.store = CycleStore(settings.database_path)
        lifecycle = LifecycleController(settings, self.store, FakeDispatcher())
        self.app = create_app(settings, self.store, FakeOverview(), lifecycle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx2.Response:
        transport = httpx2.ASGITransport(app=self.app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    async def test_create_list_get_events_and_restore_latest_cycle(self) -> None:
        payload = {
            "dataset": "MSMT17",
            "idempotency_key": "msmt17-v2-cycle",
            "actor": "gialam77k1",
        }
        created = await self.request("POST", "/control/cycles", json=payload)
        duplicated = await self.request("POST", "/control/cycles", json=payload)

        self.assertEqual(created.status_code, 200)
        self.assertTrue(created.json()["created"])
        self.assertFalse(duplicated.json()["created"])
        cycle_id = created.json()["cycle"]["id"]
        self.assertEqual((await self.request("GET", "/control/cycles")).json()["cycles"][0]["id"], cycle_id)
        self.assertEqual((await self.request("GET", f"/control/cycles/{cycle_id}")).status_code, 200)
        events = (await self.request("GET", f"/control/cycles/{cycle_id}/events")).json()
        self.assertEqual(events["events"][0]["kind"], "cycle.created")
        self.assertEqual((await self.request("GET", "/control/overview")).json()["latest_cycle"]["id"], cycle_id)

    async def test_rejects_unknown_dataset_and_missing_cycle(self) -> None:
        invalid = await self.request(
            "POST",
            "/control/cycles",
            json={"dataset": "Unknown", "idempotency_key": "unknown-ds", "actor": "tester"},
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual((await self.request("GET", "/control/cycles/missing")).status_code, 404)


if __name__ == "__main__":
    unittest.main()
