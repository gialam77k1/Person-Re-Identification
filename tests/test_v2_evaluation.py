from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.validate_v2_evaluation import validate_report


class V2EvaluationTest(unittest.TestCase):
    def test_accepts_finite_metrics_for_requested_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({
                "release_id": "model:v2", "dataset": "msmt17", "valid_queries": 10,
                "rank1": 0.5, "rank5": 0.7, "mAP": 0.4, "mINP": 0.2,
            }), encoding="utf-8")
            self.assertEqual(validate_report(path, "model:v2", "MSMT17")["rank1"], 0.5)

    def test_rejects_non_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps({
                "release_id": "model:v2", "dataset": "msmt17", "valid_queries": 10,
                "rank1": float("nan"), "rank5": 0.7, "mAP": 0.4, "mINP": 0.2,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_report(path, "model:v2", "MSMT17")


if __name__ == "__main__":
    unittest.main()
