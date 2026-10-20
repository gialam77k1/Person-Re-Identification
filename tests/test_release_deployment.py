from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.release_deployment import (
    ReleaseError,
    build_image_reference,
    select_rollback_record,
    validate_candidate,
    validate_image_reference,
    write_deployment_state,
    write_pending_state,
)


class ReleaseDeploymentTest(unittest.TestCase):
    def make_candidate(self, root: Path, status: str = "PASS") -> tuple[Path, Path]:
        artifact = root / "model.bin"
        artifact.write_bytes(b"model")
        checksum = hashlib.sha256(b"model").hexdigest()
        artifact_entry = {
            "role": "onnx_graph",
            "repository_path": str(artifact),
            "sha256": checksum,
        }
        manifest = {
            "release": {
                "release_id": "market1501-vit-bnneck:v1",
                "status": "validated" if status == "PASS" else "rejected",
            },
            "source": {"git_commit": "a" * 40},
            "validation": {"status": status},
            "files": [artifact_entry],
        }
        report = {
            "release_id": "market1501-vit-bnneck:v1",
            "status": status,
            "passed": status == "PASS",
            "failed_checks": [] if status == "PASS" else ["quality_rank1"],
            "validated_artifacts": [artifact_entry],
        }
        manifest_path = root / "manifest.json"
        report_path = root / "report.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, manifest_path

    def test_image_tag_contains_commit_and_release(self) -> None:
        image = build_image_reference("reid-api", "1ca60d8f00", "market1501-vit-bnneck:v1")
        self.assertEqual(image, "reid-api:1ca60d8-model-v1")
        validate_image_reference(image, "1ca60d8f00", "market1501-vit-bnneck:v1")
        with self.assertRaises(ReleaseError):
            validate_image_reference("reid-api:latest", "1ca60d8f00", "market1501-vit-bnneck:v1")

    def test_validation_gate_accepts_only_matching_pass_candidate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            report, manifest = self.make_candidate(root)
            validate_candidate(report, manifest, "market1501-vit-bnneck:v1")
            with self.assertRaises(ReleaseError):
                validate_candidate(report, manifest, "market1501-vit-bnneck:v2")

            failed_report, failed_manifest = self.make_candidate(root, status="FAIL")
            with self.assertRaises(ReleaseError):
                validate_candidate(failed_report, failed_manifest, "market1501-vit-bnneck:v1")

    def test_deployment_state_keeps_previous_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory)
            first = {"release_id": "model:v1", "image": "api:aaa-model-v1"}
            second = {"release_id": "model:v2", "image": "api:bbb-model-v2"}
            write_deployment_state(state_root, "staging", first)
            write_deployment_state(state_root, "staging", second)
            current = json.loads((state_root / "staging/current.json").read_text())
            previous = json.loads((state_root / "staging/previous.json").read_text())
            self.assertEqual(current, second)
            self.assertEqual(previous, first)

    def test_failed_deployment_rolls_back_to_current_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory)
            current = {"release_id": "model:v1", "image": "api:aaa-model-v1"}
            candidate = {"release_id": "model:v2", "image": "api:bbb-model-v2"}
            write_deployment_state(state_root, "production", current)
            write_pending_state(state_root, "production", candidate)
            self.assertEqual(select_rollback_record(state_root, "production"), current)


if __name__ == "__main__":
    unittest.main()
