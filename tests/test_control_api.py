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

    def download_artifact(self, run_id: str, artifact_name: str) -> dict[str, bytes]:
        return {}


class ControlApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        settings = replace(
            ControlSettings.from_env(),
            database_path=Path(self.temporary.name) / "control.db",
            allowed_datasets=("dukemtmc-reid",),
            dataset_root=Path(self.temporary.name) / "datasets",
            evaluation_artifacts_root=Path(self.temporary.name) / "artifacts",
            inbox_min_valid_images=1,
            inbox_min_identities=1,
            inbox_min_cameras=1,
            registry_model_name="",
        )
        query_dir = settings.dataset_root / "dukemtmc-reid" / "query"
        query_dir.mkdir(parents=True)
        (query_dir / "0001_c2_sample.jpg").write_bytes(b"fake-jpeg")
        (query_dir / "0001_c3_sample.jpg").write_bytes(b"fake-jpeg")
        train_dir = settings.dataset_root / "dukemtmc-reid" / "bounding_box_train"
        train_dir.mkdir(parents=True)
        (train_dir / "0001_c2_train.jpg").write_bytes(b"\xff\xd8\xfftrain")
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
            "dataset": "dukemtmc-reid",
            "idempotency_key": "duke-v2-cycle",
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

    async def test_data_inbox_validates_and_dispatches_cycle(self) -> None:
        response = await self.request(
            "POST",
            "/control/inbox/import",
            json={
                "actor": "operator",
                "dataset": "dukemtmc-reid",
                "source_path": "dukemtmc-reid/bounding_box_train",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["created"])
        self.assertEqual(payload["batch"]["status"], "DISPATCHED")
        self.assertTrue(payload["batch"]["cycle_id"])
        inbox = (await self.request("GET", "/control/inbox")).json()
        self.assertEqual(inbox["batches"][0]["fingerprint"], payload["batch"]["fingerprint"])

    async def test_rejects_unknown_dataset_and_missing_cycle(self) -> None:
        invalid = await self.request(
            "POST",
            "/control/cycles",
            json={"dataset": "Unknown", "idempotency_key": "unknown-ds", "actor": "tester"},
        )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual((await self.request("GET", "/control/cycles/missing")).status_code, 404)

    async def test_runner_callback_requires_token_and_updates_live_progress(self) -> None:
        created = await self.request(
            "POST",
            "/control/cycles",
            json={"dataset": "dukemtmc-reid", "idempotency_key": "callback-cycle", "actor": "tester"},
        )
        cycle_id = created.json()["cycle"]["id"]
        payload = {
            "status": "RUNNING",
            "progress": 0.35,
            "detail": "Epoch 7/20",
            "metrics": {"epoch": 7, "mAP": 0.42},
        }
        unauthorized = await self.request(
            "POST", f"/control/cycles/{cycle_id}/stages/train_v2", json=payload
        )
        updated = await self.request(
            "POST",
            f"/control/cycles/{cycle_id}/stages/train_v2",
            json=payload,
            headers={"X-Control-Callback-Token": "change-me-control-callback-token"},
        )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(updated.status_code, 200)
        train = next(stage for stage in updated.json()["stages"] if stage["name"] == "train_v2")
        self.assertEqual(train["progress"], 0.35)
        events = (await self.request("GET", f"/control/cycles/{cycle_id}/events")).json()["events"]
        self.assertEqual(events[-1]["kind"], "stage.metrics")

    async def test_runner_callback_accepts_nested_report_metrics(self) -> None:
        created = await self.request(
            "POST",
            "/control/cycles",
            json={
                "dataset": "dukemtmc-reid",
                "idempotency_key": "nested-drift-report",
                "actor": "tester",
            },
        )
        cycle_id = created.json()["cycle"]["id"]
        updated = await self.request(
            "POST",
            f"/control/cycles/{cycle_id}/stages/detect_drift",
            headers={"X-Control-Callback-Token": "change-me-control-callback-token"},
            json={
                "status": "PASS",
                "progress": 1.0,
                "metrics": {
                    "drift_detected": True,
                    "checks": {
                        "centroid_cosine_distance": {
                            "value": 0.42,
                            "maximum": 0.15,
                            "passed": False,
                        }
                    },
                },
            },
        )

        self.assertEqual(updated.status_code, 200)
        events = (await self.request("GET", f"/control/cycles/{cycle_id}/events")).json()["events"]
        self.assertEqual(
            events[-1]["payload"]["metrics"]["checks"]
            ["centroid_cosine_distance"]["value"],
            0.42,
        )
        prometheus = await self.request("GET", "/metrics")
        self.assertEqual(prometheus.status_code, 200)
        self.assertIn(
            'reid_embedding_drift_detected{cycle_id="', prometheus.text
        )
        self.assertIn('dataset="dukemtmc-reid"', prometheus.text)
        self.assertIn('metric="centroid_cosine_distance"} 0.42', prometheus.text)

    async def test_failed_stage_closes_other_running_stages(self) -> None:
        created = await self.request(
            "POST",
            "/control/cycles",
            json={
                "dataset": "dukemtmc-reid",
                "idempotency_key": "terminal-failure-closes-running",
                "actor": "tester",
            },
        )
        cycle_id = created.json()["cycle"]["id"]
        headers = {"X-Control-Callback-Token": "change-me-control-callback-token"}
        await self.request(
            "POST",
            f"/control/cycles/{cycle_id}/stages/train_v2",
            headers=headers,
            json={"status": "RUNNING", "progress": 0.4, "detail": "Epoch 12/50"},
        )
        failed = await self.request(
            "POST",
            f"/control/cycles/{cycle_id}/stages/validate_release",
            headers=headers,
            json={"status": "FAIL", "detail": "Completion gate failed"},
        )

        self.assertEqual(failed.status_code, 200)
        self.assertEqual(failed.json()["status"], "FAIL")
        stages = {stage["name"]: stage for stage in failed.json()["stages"]}
        self.assertEqual(stages["validate_release"]["status"], "FAIL")
        self.assertEqual(stages["train_v2"]["status"], "FAIL")
        self.assertEqual(stages["train_v2"]["progress"], 0.4)
        self.assertIn("validate_release failed", stages["train_v2"]["detail"])

    async def test_evaluation_api_returns_real_callback_preview_and_safe_image(self) -> None:
        created = await self.request(
            "POST",
            "/control/cycles",
            json={"dataset": "dukemtmc-reid", "idempotency_key": "evaluation-ui", "actor": "tester"},
        )
        cycle_id = created.json()["cycle"]["id"]
        report = {
            "release_id": "market1501-vit-bnneck:v1-dvc",
            "dataset": "dukemtmc-reid",
            "rank1": 0.62,
            "rank5": 0.81,
            "mAP": 0.37,
            "mINP": 0.21,
            "retrieval_preview": [
                {
                    "query": {"image_path": "query/0001_c2_sample.jpg", "person_id": 1, "camera_id": 2},
                    "predictions": [],
                    "top1_correct": False,
                }
            ],
        }
        await self.request(
            "POST",
            f"/control/cycles/{cycle_id}/stages/evaluate_v1",
            headers={"X-Control-Callback-Token": "change-me-control-callback-token"},
            json={"status": "PASS", "metrics": {"evaluation": report}},
        )

        payload = (await self.request("GET", "/control/evaluations")).json()
        self.assertEqual(payload["versions"][0]["metrics"]["mAP"], 0.37)
        self.assertEqual(payload["identity_camera_groups"][0]["person_id"], 1)
        self.assertEqual(payload["identity_camera_groups"][0]["camera_count"], 2)
        self.assertEqual(
            [image["camera_id"] for image in payload["identity_camera_groups"][0]["images"]],
            [2, 3],
        )
        image = await self.request(
            "GET", "/control/evaluation-images/dukemtmc-reid/query/0001_c2_sample.jpg"
        )
        self.assertEqual(image.status_code, 200)
        self.assertEqual(
            (await self.request(
                "GET", "/control/evaluation-images/dukemtmc-reid/not-a-split/secret"
            )).status_code,
            404,
        )

    async def test_retraining_decision_endpoint_requires_runner_token(self) -> None:
        created = await self.request(
            "POST",
            "/control/cycles",
            json={"dataset": "dukemtmc-reid", "idempotency_key": "decision-api", "actor": "tester"},
        )
        cycle_id = created.json()["cycle"]["id"]
        self.store.save_retraining_decision(
            cycle_id,
            {"status": "RETRAIN_APPROVED", "approved_by": "reviewer"},
            "a" * 64,
        )
        unauthorized = await self.request(
            "GET", f"/control/cycles/{cycle_id}/retraining-decision"
        )
        authorized = await self.request(
            "GET",
            f"/control/cycles/{cycle_id}/retraining-decision",
            headers={"X-Control-Callback-Token": "change-me-control-callback-token"},
        )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.json()["status"], "RETRAIN_APPROVED")


if __name__ == "__main__":
    unittest.main()
