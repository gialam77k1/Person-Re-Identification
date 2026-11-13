from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.validate_v2_evaluation import validate_report


class V2EvaluationTest(unittest.TestCase):
    def lineage(self) -> dict:
        content = b"\xff\xd8\xffpreview"
        return {
            "query_count": 10,
            "gallery_count": 20,
            "checkpoint_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "dataset_dvc_revision": "c.dir",
            "retrieval_preview": [{
                "query": {
                    "asset_id": hashlib.sha256(content).hexdigest(),
                    "asset_mime": "image/jpeg",
                    "asset_data": base64.b64encode(content).decode("ascii"),
                },
                "predictions": [],
            }],
        }

    def test_accepts_finite_metrics_for_requested_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({
                "release_id": "model:v2", "dataset": "dukemtmc-reid", "valid_queries": 10,
                "rank1": 0.5, "rank5": 0.7, "rank10": 0.8, "rank20": 0.9,
                "mAP": 0.4, "mINP": 0.2,
                **self.lineage(),
            }), encoding="utf-8")
            self.assertEqual(validate_report(path, "model:v2", "dukemtmc-reid")["rank1"], 0.5)

    def test_rejects_non_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({
                "release_id": "model:v2", "dataset": "dukemtmc-reid", "valid_queries": 10,
                "rank1": float("nan"), "rank5": 0.7, "rank10": 0.8, "rank20": 0.9,
                "mAP": 0.4, "mINP": 0.2,
                **self.lineage(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_report(path, "model:v2", "dukemtmc-reid")


if __name__ == "__main__":
    unittest.main()
