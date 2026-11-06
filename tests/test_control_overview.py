from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.control_api.config import ControlSettings
from src.control_api.overview import OverviewService


class FakeHttpClient:
    def get_text(self, url: str) -> str:
        return "ready"

    def get_json(self, url: str) -> dict:
        if url.endswith("/model-info"):
            return {"release_id": "market1501-vit-bnneck:v1", "embedding_dim": 512}
        if "/collections/" in url:
            return {"result": {"status": "green", "points_count": 3368}}
        return {"status": "ready"}

    def prometheus_query(self, base_url: str, query: str) -> dict:
        return {
            "data": {
                "result": [
                    {"metric": {"release_id": "v1"}, "value": [123.0, "1"]}
                ]
            }
        }


class OverviewServiceTest(unittest.TestCase):
    def test_aggregates_service_model_collection_and_drift_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = replace(
                ControlSettings.from_env(), database_path=Path(temporary) / "db"
            )
            overview = OverviewService(settings, FakeHttpClient()).overview()

        self.assertEqual(overview["services"]["ready"], 5)
        self.assertEqual(overview["services"]["checked"], 5)
        self.assertEqual(overview["model"]["embedding_dim"], 512)
        self.assertEqual(overview["collection"]["points_count"], 3368)
        self.assertEqual(overview["drift"]["status"], "DRIFT")


if __name__ == "__main__":
    unittest.main()
