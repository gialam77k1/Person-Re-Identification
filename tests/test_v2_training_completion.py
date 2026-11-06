from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.validate_v2_training_completion import validate_training_completion


ROOT = Path(__file__).resolve().parents[1]


class V2TrainingCompletionTest(unittest.TestCase):
    def create_run(self, root: Path, *, stopped_early: bool = True) -> Path:
        run_root = root / "run"
        for directory in ("checkpoints", "logs", "metrics"):
            (run_root / directory).mkdir(parents=True, exist_ok=True)
        (run_root / "checkpoints/best_model.pth").write_bytes(b"best")
        (run_root / "checkpoints/last_model.pth").write_bytes(b"last")
        config = yaml.safe_load(
            (ROOT / "configs/msmt17_v2_train.yaml").read_text(encoding="utf-8")
        )
        (run_root / "logs/effective_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        completed = 8 if stopped_early else 20
        stop_reason = (
            "early_stopping:mAP:no_improvement_3_epochs"
            if stopped_early
            else "max_epochs_completed"
        )
        (run_root / "logs/mlflow_run.json").write_text(
            json.dumps({"run_id": "run-123", "status": "FINISHED"}), encoding="utf-8"
        )
        (run_root / "logs/transfer_report.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "source_epoch": 48,
                    "reset_classifier_heads": True,
                    "loaded_backbone_tensors": 12,
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "release_id": "market1501-msmt17-vit-bnneck:v2",
            "base_release_id": "market1501-vit-bnneck:v1-dvc",
            "configured_epochs": 20,
            "completed_epochs": completed,
            "best_epoch": 5,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_rank1": 0.4,
            "best_mAP": 0.3,
            "history": [{"epoch": epoch, "mAP": 0.1 + epoch / 100} for epoch in range(1, completed + 1)],
        }
        (run_root / "metrics/metrics_v2.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (run_root / "logs/train.log").write_text("training completed\n", encoding="utf-8")
        return run_root

    def test_accepts_early_stopped_finished_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = validate_training_completion(self.create_run(Path(directory)))
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["stopped_early"])
        self.assertEqual(report["completed_epochs"], 8)

    def test_rejects_non_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.create_run(Path(directory))
            metrics_path = run_root / "metrics/metrics_v2.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["best_mAP"] = float("nan")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                validate_training_completion(run_root)

    def test_rejects_cuda_failure_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.create_run(Path(directory), stopped_early=False)
            (run_root / "logs/train.log").write_text(
                "RuntimeError: CUDA out of memory", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "CUDA failure marker"):
                validate_training_completion(run_root)

    def test_rejects_missing_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = self.create_run(Path(directory))
            (run_root / "logs/transfer_report.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "transfer_report"):
                validate_training_completion(run_root)


if __name__ == "__main__":
    unittest.main()
