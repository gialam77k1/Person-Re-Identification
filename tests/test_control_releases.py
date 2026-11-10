from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.control_api.config import ControlSettings
from src.control_api.releases import ReleaseResolver, next_release
from src.control_api.state import CycleStatus, StageStatus
from src.control_api.store import CycleStore


class FakeHttpClient:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload

    def get_json(self, _url: str) -> dict:
        if self.payload is None:
            from src.control_api.clients import ControlUpstreamError

            raise ControlUpstreamError("offline")
        return self.payload


class ReleaseResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = replace(
            ControlSettings.from_env(),
            database_path=root / "control.db",
            mlflow_url="http://mlflow",
            inbox_bootstrap_release="market1501-vit-bnneck:v1-dvc",
            inbox_target_family="market1501-duke-vit-bnneck",
            registry_model_name="market1501-duke-vit-bnneck",
        )
        self.store = CycleStore(self.settings.database_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_bootstrap_and_version_increment(self) -> None:
        pair = ReleaseResolver(
            self.settings, self.store, FakeHttpClient()
        ).resolve()
        self.assertEqual(pair.source, "market1501-vit-bnneck:v1-dvc")
        self.assertEqual(pair.target, "market1501-duke-vit-bnneck:v2")
        self.assertEqual(next_release("market1501-duke-vit-bnneck:v9", "market1501-duke-vit-bnneck"), "market1501-duke-vit-bnneck:v10")

    def test_mlflow_champion_is_authoritative(self) -> None:
        pair = ReleaseResolver(
            self.settings,
            self.store,
            FakeHttpClient(
                {
                    "model_version": {
                        "tags": [{"key": "release.id", "value": "market1501-duke-vit-bnneck:v2"}]
                    }
                }
            ),
        ).resolve()
        self.assertEqual(pair.source, "market1501-duke-vit-bnneck:v2")
        self.assertEqual(pair.target, "market1501-duke-vit-bnneck:v3")
        self.assertEqual(pair.source_kind, "mlflow_champion")

    def test_promoted_cycle_is_offline_fallback(self) -> None:
        cycle, _ = self.store.create_cycle(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="promoted-cycle",
            actor="operator",
        )
        self.store.update_stage(
            cycle["id"], "production_approval", StageStatus.PASS, progress=1.0
        )
        pair = ReleaseResolver(
            self.settings, self.store, FakeHttpClient()
        ).resolve()
        self.assertEqual(pair.source, "market1501-duke-vit-bnneck:v2")
        self.assertEqual(pair.target, "market1501-duke-vit-bnneck:v3")
        self.assertEqual(pair.source_kind, "promoted_cycle")

    def test_existing_unpromoted_target_blocks_version_reuse(self) -> None:
        self.store.create_cycle(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="existing-v2",
            actor="operator",
        )
        with self.assertRaisesRegex(ValueError, "already belongs to cycle"):
            ReleaseResolver(self.settings, self.store, FakeHttpClient()).resolve()

    def test_cancelled_target_attempt_allows_safe_reuse(self) -> None:
        cycle, _ = self.store.create_cycle(
            dataset="dukemtmc-reid",
            source_release="market1501-vit-bnneck:v1-dvc",
            target_release="market1501-duke-vit-bnneck:v2",
            idempotency_key="abandoned-v2",
            actor="operator",
        )
        self.store.update_cycle(cycle["id"], status=CycleStatus.CANCELLED)

        pair = ReleaseResolver(self.settings, self.store, FakeHttpClient()).resolve()

        self.assertEqual(pair.target, "market1501-duke-vit-bnneck:v2")


if __name__ == "__main__":
    unittest.main()
