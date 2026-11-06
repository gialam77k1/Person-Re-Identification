from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluate import load_evaluation_checkpoint
from src.validate_v2_evaluation import validate_report


class V2EvaluationReportTest(unittest.TestCase):
    def report(self) -> dict:
        return {
            "release_id": "market1501-msmt17-vit-bnneck:v2",
            "dataset": "msmt17",
            "rank1": 0.5,
            "rank5": 0.7,
            "mAP": 0.4,
            "mINP": 0.2,
            "valid_queries": 100,
            "query_count": 100,
            "gallery_count": 200,
            "checkpoint_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "dataset_dvc_revision": "c.dir",
        }

    def test_accepts_complete_lineage_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report()), encoding="utf-8")
            report = validate_report(
                path, "market1501-msmt17-vit-bnneck:v2", "msmt17"
            )
        self.assertEqual(report["query_count"], 100)

    def test_rejects_missing_checkpoint_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = self.report()
            report.pop("checkpoint_sha256")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                validate_report(path, "market1501-msmt17-vit-bnneck:v2", "msmt17")

    def test_evaluation_loader_only_skips_incompatible_task_head(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self, classes: int) -> None:
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.classifier = torch.nn.Linear(2, classes)

        source = TinyModel(classes=1041)
        target = TinyModel(classes=751)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save({"epoch": 8, "model_state_dict": source.state_dict()}, checkpoint)
            loaded, details = load_evaluation_checkpoint(
                target, checkpoint, torch.device("cpu")
            )
        self.assertEqual(loaded["epoch"], 8)
        self.assertIn("classifier.weight", details["skipped_task_tensors"])
        self.assertTrue(torch.equal(target.backbone.weight, source.backbone.weight))


if __name__ == "__main__":
    unittest.main()
