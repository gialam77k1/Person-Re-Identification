from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.materialize_training_run import ARTIFACT_PATHS, materialize_training_run
from src.prepare_candidate_evaluation import prepare_config
from src.validate_post_training_contract import validate_config


class PostTrainingRecoveryTest(unittest.TestCase):
    def template(self, root: Path) -> Path:
        path = root / "template.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment_name": "old",
                    "release": {"id": "model:v2", "base_release_id": "model:v1"},
                    "data": {"dataset": {"name": "dukemtmc-reid"}},
                    "evaluation": {"report_filename": "evaluation_duke.json"},
                    "artifacts": {"run_root": "artifacts/old"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_version_neutral_config_targets_the_requested_release_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runtime.yaml"
            run_root = Path("artifacts/duke/model-v3/evaluations/duke")
            prepare_config(
                self.template(root),
                output,
                source_release="model:v2",
                target_release="model:v3",
                run_root=run_root,
            )
            result = validate_config(
                output,
                release_id="model:v3",
                dataset="duke",
                report=run_root / "metrics" / "evaluation_duke.json",
            )
        self.assertEqual(result["dataset"], "dukemtmc-reid")

    def test_contract_fails_before_training_when_report_path_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runtime.yaml"
            prepare_config(
                self.template(root),
                output,
                source_release="model:v1",
                target_release="model:v2",
                run_root=Path("artifacts/correct"),
            )
            with self.assertRaisesRegex(ValueError, "report path mismatch"):
                validate_config(
                    output,
                    release_id="model:v2",
                    dataset="dukemtmc-reid",
                    report=Path("artifacts/stale/metrics/evaluation_duke.json"),
                )

    def test_completed_mlflow_run_is_restored_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            remote.mkdir()
            by_uri: dict[str, str] = {}
            for index, artifact_path in enumerate(ARTIFACT_PATHS):
                source = remote / f"artifact-{index}"
                source.write_bytes(f"payload-{artifact_path}".encode())
                by_uri[f"runs:/run-123/{artifact_path}"] = str(source)
            run = SimpleNamespace(
                info=SimpleNamespace(status="FINISHED"),
                data=SimpleNamespace(
                    params={"release_id": "model:v2"}, tags={}
                ),
            )
            client = SimpleNamespace(get_run=lambda run_id: run)
            result = materialize_training_run(
                run_id="run-123",
                release_id="model:v2",
                run_root=root / "candidate",
                tracking_uri="http://mlflow",
                client=client,
                downloader=lambda uri: by_uri[uri],
            )
            report = json.loads(
                (root / "candidate/logs/training_restore_report.json").read_text()
            )
            best = (root / "candidate/checkpoints/best_model.pth").read_bytes()
            last = (root / "candidate/checkpoints/last_model.pth").read_bytes()
        self.assertEqual(result["mode"], "resume_post_training")
        self.assertEqual(report["run_id"], "run-123")
        self.assertEqual(report["last_checkpoint_source"], "checkpoints/best_model.pth (legacy recovery fallback)")
        self.assertEqual(best, last)

    def test_refuses_failed_or_wrong_release_mlflow_run(self) -> None:
        run = SimpleNamespace(
            info=SimpleNamespace(status="FAILED"),
            data=SimpleNamespace(params={"release_id": "model:v2"}, tags={}),
        )
        client = SimpleNamespace(get_run=lambda run_id: run)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not FINISHED"):
                materialize_training_run(
                    run_id="failed-run",
                    release_id="model:v2",
                    run_root=Path(directory),
                    tracking_uri="http://mlflow",
                    client=client,
                    downloader=lambda uri: uri,
                )


if __name__ == "__main__":
    unittest.main()
