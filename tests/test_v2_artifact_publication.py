from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.verify_v2_artifact_publication import build_publication_report, sha256


class V2ArtifactPublicationTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[Path, Path, Path, list[Path]]:
        artifact = root / "model.onnx"
        artifact.write_bytes(b"onnx")
        pointers = [root / "best.pth.dvc", root / "last.pth.dvc"]
        for pointer in pointers:
            pointer.write_text("outs: []\n", encoding="utf-8")
        manifest = root / "model_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "release": {"release_id": "model:v2", "status": "validated"},
                    "validation": {"status": "PASS"},
                    "source": {
                        "git_commit": "abc",
                        "training_git_commit": "def",
                        "mlflow_run_id": "run-1",
                        "config_sha256": "a" * 64,
                        "dataset_revisions": {"msmt17": "msmt.dir", "market1501": "market.dir"},
                    },
                    "files": [{"role": "onnx_graph", "repository_path": str(artifact), "sha256": sha256(artifact)}],
                }
            ),
            encoding="utf-8",
        )
        validation = root / "validation.json"
        validation.write_text(json.dumps({"release_id": "model:v2", "status": "PASS", "passed": True}), encoding="utf-8")
        status = root / "dvc_status.json"
        status.write_text("{}", encoding="utf-8")
        return manifest, validation, status, pointers

    def test_clean_remote_and_matching_artifacts_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, validation, status, pointers = self.make_bundle(Path(directory))
            report = build_publication_report(manifest, validation, status, pointers, "minio", "v2_package_triton")
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["dvc"]["status"], "clean")
            self.assertEqual(len(report["artifacts"]), 1)

    def test_dirty_remote_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, validation, status, pointers = self.make_bundle(Path(directory))
            status.write_text(json.dumps({"not_in_remote": ["hash"]}), encoding="utf-8")
            report = build_publication_report(manifest, validation, status, pointers, "minio", "v2_package_triton")
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("dvc_remote_clean", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
