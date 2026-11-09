from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.register_retraining_candidate import (
    build_lineage_tags,
    sha256 as registry_sha256,
    validate_candidate_inputs,
)
from src.retraining_validation_gate import build_gate, sha256


class RetrainingCandidateTest(unittest.TestCase):
    def test_gate_passes_complete_approved_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.pth"
            candidate = Path(directory) / "candidate.pth"
            base.write_bytes(b"base")
            candidate.write_bytes(b"candidate")
            report, manifest = build_gate(
                evidence={"source_git_commit": "abc", "release_id": "base:v1", "dataset": "dukemtmc-reid", "dataset_dvc_revision": "dvc.dir", "base_checkpoint": {"sha256": sha256(base)}},
                decision={"status": "RETRAIN_APPROVED", "base_release_id": "base:v1", "target_dataset": "dukemtmc-reid"},
                training={"release_id": "model:v2", "base_release_id": "base:v1"},
                evaluation={"release_id": "model:v2", "valid_queries": 10, "rank1": 0.6, "rank5": 0.8, "mAP": 0.5, "mINP": 0.3},
                baseline={"dataset_dvc_revision": "dvc.dir", "rank1": 0.5, "mAP": 0.4, "mINP": 0.2},
                mlflow_run={"run_id": "run-1", "status": "FINISHED"},
                config={"release": {"id": "model:v2", "base_release_id": "base:v1"}, "data": {"dataset": {"name": "dukemtmc-reid"}}, "model": {"backbone": "vit_b_16", "embedding_dim": 512}},
                policy={"quality": {"rank1_max_drop_from_v1": 0.005, "map_max_drop_from_v1": 0.005, "minp_max_drop_from_v1": 0.01, "minimum_valid_queries": 1}, "contract": {"approved_status": "RETRAIN_APPROVED"}},
                base_checkpoint=base, candidate_checkpoint=candidate, git_commit="abc",
            )
            self.assertTrue(report["passed"])
            self.assertEqual(manifest["source"]["mlflow_run_id"], "run-1")

    def test_candidate_requires_pass_and_matching_release(self) -> None:
        manifest = {
            "release": {"release_id": "model:v2", "status": "validated"},
            "validation": {"status": "PASS"},
            "source": {"mlflow_run_id": "run-1"},
        }
        self.assertEqual(validate_candidate_inputs({"status": "PASS", "passed": True, "release_id": "model:v2"}, manifest, "candidate"), ("model", "model:v2", "run-1"))
        with self.assertRaisesRegex(ValueError, "PASS"):
            validate_candidate_inputs({"status": "FAIL", "passed": False, "release_id": "model:v2"}, manifest, "candidate")

    def test_workflow_cannot_assign_champion(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate"):
            validate_candidate_inputs(
                {"status": "PASS", "passed": True, "release_id": "model:v2"},
                {"release": {"release_id": "model:v2", "status": "validated"}, "validation": {"status": "PASS"}, "source": {"mlflow_run_id": "run"}},
                "champion",
            )

    def test_registry_tags_bind_git_dvc_config_metrics_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            validation_path = root / "validation.json"
            validation_path.write_text('{"status":"PASS"}', encoding="utf-8")
            manifest = {
                "release": {"release_id": "model:v2"},
                "source": {
                    "git_commit": "pipeline-commit",
                    "training_git_commit": "training-commit",
                    "config_sha256": "a" * 64,
                    "dataset_revisions": {"duke": "msmt.dir", "market1501": "market.dir"},
                },
                "quality_metrics": {
                    "duke": {"rank1": 0.7, "rank5": 0.8, "mAP": 0.6, "mINP": 0.5},
                    "market1501": {"rank1": 0.9, "rank5": 0.95, "mAP": 0.8, "mINP": 0.7},
                },
                "files": [{"role": "pytorch_checkpoint", "sha256": "b" * 64}],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            publication = {
                "status": "PASS",
                "release_id": "model:v2",
                "model_manifest": {"sha256": registry_sha256(manifest_path)},
                "validation_report": {"sha256": registry_sha256(validation_path)},
                "dvc": {"remote": "minio", "stage": "v2_package_triton"},
            }
            tags = build_lineage_tags(manifest, validation_path, manifest_path, publication)
            self.assertEqual(tags["dvc.duke.revision"], "msmt.dir")
            self.assertEqual(tags["metrics.market1501.mAP"], "0.8")
            self.assertEqual(tags["registry.alias.policy"], "candidate-only")


if __name__ == "__main__":
    unittest.main()
