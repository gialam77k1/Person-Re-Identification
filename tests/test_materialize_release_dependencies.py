from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from src.materialize_release_dependencies import (
    ReleaseDependencyError,
    materialize_validation_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class MaterializeReleaseDependenciesTest(unittest.TestCase):
    def manifest(self, digest: str, commit: str) -> dict:
        return {
            "release": {"release_id": "market1501-duke-vit-bnneck:v2"},
            "source": {"git_commit": commit},
            "files": [
                {
                    "role": "validation_policy",
                    "repository_path": "configs/model_validation_v2.yaml",
                    "sha256": digest,
                }
            ],
        }

    def test_restores_exact_policy_from_candidate_bundle(self) -> None:
        content = b"release_id: model:v2\nquality: {}\n"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            candidate = Path(temporary) / "candidate"
            (candidate / "configs").mkdir(parents=True)
            (candidate / "configs/model_validation_v2.yaml").write_bytes(content)
            destination = materialize_validation_policy(
                manifest=self.manifest(digest, "a" * 40),
                repository_root=root,
                candidate_root=candidate,
            )
            self.assertEqual(destination.read_bytes(), content)

    def test_reconstructs_legacy_candidate_policy_from_source_commit(self) -> None:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source = subprocess.run(
            ["git", "show", f"{source_commit}:configs/model_validation_v2.yaml"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        payload = yaml.safe_load(source)
        payload["release_id"] = "market1501-duke-vit-bnneck:v2"
        expected = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
        digest = hashlib.sha256(expected).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            destination = materialize_validation_policy(
                manifest=self.manifest(digest, source_commit),
                repository_root=Path(temporary),
                source_repository_root=ROOT,
            )
            self.assertEqual(destination.read_bytes(), expected)

    def test_rejects_tampered_candidate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            candidate = Path(temporary) / "candidate"
            (candidate / "configs").mkdir(parents=True)
            (candidate / "configs/model_validation_v2.yaml").write_text(
                "release_id: tampered:v2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ReleaseDependencyError, "checksum"):
                materialize_validation_policy(
                    manifest=self.manifest("0" * 64, "a" * 40),
                    repository_root=root,
                    candidate_root=candidate,
                )


if __name__ == "__main__":
    unittest.main()
