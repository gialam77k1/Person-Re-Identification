from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.import_v2_checkpoint import import_checkpoint


class V2DVCImportTest(unittest.TestCase):
    def write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_imports_checkpoint_with_release_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.pth"
            checkpoint.write_bytes(b"checkpoint-v2")
            training = self.write(root, "metrics.json", {"release_id": "market1501-msmt17-vit-bnneck:v2", "best_epoch": 7})
            mlflow = self.write(root, "mlflow.json", {"status": "FINISHED", "run_id": "run-1", "git_commit": "train-commit"})
            transfer = self.write(root, "transfer.json", {"source_epoch": 48, "reset_classifier_heads": True})
            config = self.write(root, "config.json", {"release": "v2"})
            data = self.write(root, "data.json", {"passed": True, "datasets": {"msmt17": {"dataset_dvc_revision": "msmt.dir"}, "market1501": {"dataset_dvc_revision": "market.dir"}}})
            output = root / "output"
            with patch("src.import_v2_checkpoint.current_commit", return_value="release-commit"):
                result = import_checkpoint(
                    checkpoint_path=checkpoint,
                    training_metrics_path=training,
                    mlflow_run_path=mlflow,
                    transfer_report_path=transfer,
                    effective_config_path=config,
                    data_manifest_path=data,
                    output_root=output,
                )
            self.assertEqual(result["mlflow_run_id"], "run-1")
            self.assertEqual(result["dataset_revisions"]["msmt17"], "msmt.dir")
            self.assertEqual(result["git_commit"], "release-commit")
            self.assertEqual(len(result["training_config_sha256"]), 64)
            self.assertEqual((output / "best_model.pth").read_bytes(), b"checkpoint-v2")

    def test_rejects_unfinished_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.pth"
            checkpoint.write_bytes(b"checkpoint-v2")
            paths = {
                "training_metrics_path": self.write(root, "metrics.json", {"release_id": "market1501-msmt17-vit-bnneck:v2"}),
                "mlflow_run_path": self.write(root, "mlflow.json", {"status": "FAILED", "run_id": "run-1"}),
                "transfer_report_path": self.write(root, "transfer.json", {"source_epoch": 48, "reset_classifier_heads": True}),
                "effective_config_path": self.write(root, "config.json", {}),
                "data_manifest_path": self.write(root, "data.json", {"passed": True}),
            }
            with self.assertRaisesRegex(ValueError, "FINISHED"):
                import_checkpoint(checkpoint_path=checkpoint, output_root=root / "output", **paths)


if __name__ == "__main__":
    unittest.main()
