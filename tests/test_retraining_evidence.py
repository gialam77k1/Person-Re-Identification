from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.retraining_evidence import build_bundle, verify_bundle


class RetrainingEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.release = "market1501-vit-bnneck:v1-dvc"
        self._write("drift.json", {"release_id": self.release})
        self._write(
            "evaluation.json",
            {"release_id": self.release, "dataset": "msmt17", "dataset_dvc_revision": "abc.dir"},
        )
        self._write("decision.json", {"status": "REVIEW_REQUIRED"})
        self._write(
            "base.json",
            {
                "release": {"release_id": self.release},
                "files": [{"role": "pytorch_checkpoint", "repository_path": "base.pth", "sha256": "a" * 64}],
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, payload: dict) -> None:
        (self.root / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_build_and_verify_are_bound_to_cycle_commit_and_checksums(self) -> None:
        manifest = self.root / "bundle" / "evidence_manifest.json"
        build_bundle(
            cycle_id="cycle-1", release_id=self.release, dataset="MSMT17",
            drift_path=self.root / "drift.json", evaluation_path=self.root / "evaluation.json",
            decision_path=self.root / "decision.json", base_manifest_path=self.root / "base.json",
            output_path=manifest, source_commit="deadbeef", workflow_run_id="123",
        )
        verified = verify_bundle(
            manifest, cycle_id="cycle-1", release_id=self.release, dataset="MSMT17",
            current_commit="deadbeef",
        )
        self.assertEqual(verified["workflow_run_id"], "123")
        self.assertEqual(verified["dataset_dvc_revision"], "abc.dir")

    def test_verify_rejects_tampered_evidence(self) -> None:
        manifest = self.root / "bundle" / "evidence_manifest.json"
        build_bundle(
            cycle_id="cycle-1", release_id=self.release, dataset="MSMT17",
            drift_path=self.root / "drift.json", evaluation_path=self.root / "evaluation.json",
            decision_path=self.root / "decision.json", base_manifest_path=self.root / "base.json",
            output_path=manifest, source_commit="deadbeef",
        )
        (manifest.parent / "decision.json").write_text(
            json.dumps({"status": "NO_RETRAIN"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_bundle(
                manifest, cycle_id="cycle-1", release_id=self.release, dataset="MSMT17",
                current_commit="deadbeef",
            )


if __name__ == "__main__":
    unittest.main()
