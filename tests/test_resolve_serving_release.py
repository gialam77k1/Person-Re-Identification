from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.resolve_serving_release import release_slug, resolve, resolve_reference


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ResolveServingReleaseTest(unittest.TestCase):
    def test_resolves_any_iterative_version_from_runtime(self) -> None:
        with patch(
            "src.resolve_serving_release.urllib.request.urlopen",
            return_value=FakeResponse({"release_id": "market1501-duke-vit-bnneck:v4"}),
        ):
            release_id, slug = resolve("http://localhost:8080")

        self.assertEqual(release_id, "market1501-duke-vit-bnneck:v4")
        self.assertEqual(slug, "market1501-duke-vit-bnneck-v4")

    def test_slug_supports_bootstrap_and_future_release_names(self) -> None:
        self.assertEqual(
            release_slug("market1501-vit-bnneck:v1-dvc"),
            "market1501-vit-bnneck-v1-dvc",
        )
        self.assertEqual(release_slug("family_name:v12"), "family-name-v12")

    def test_reference_is_resolved_from_immutable_serving_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "serving" / "digest" / "artifacts"
            reference = (
                root
                / "pipeline/model-family-v3/reference/embeddings/reference_embeddings.npy"
            )
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"reference")
            state = Path(directory) / "production/current.json"
            state.parent.mkdir(parents=True)
            state.write_text(
                json.dumps(
                    {
                        "release_id": "model-family:v3",
                        "serving_artifacts_root": str(root),
                    }
                ),
                encoding="utf-8",
            )

            resolved = resolve_reference(
                state, "model-family:v3", "model-family-v3"
            )

            self.assertEqual(resolved, reference)

    def test_reference_falls_back_to_active_container_artifact_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "serving" / "digest" / "artifacts"
            reference = (
                root
                / "pipeline/model-family-v4/reference/embeddings/reference_embeddings.npy"
            )
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"reference")

            resolved = resolve_reference(
                Path(directory) / "missing-production-state.json",
                "model-family:v4",
                "model-family-v4",
                root,
            )

            self.assertEqual(resolved, reference)

    def test_reference_falls_back_when_state_root_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "active-serving" / "artifacts"
            reference = (
                root
                / "pipeline/model-family-v5/reference/embeddings/reference_embeddings.npy"
            )
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"reference")
            state = Path(directory) / "production/current.json"
            state.parent.mkdir(parents=True)
            state.write_text(
                json.dumps(
                    {
                        "release_id": "model-family:v5",
                        "serving_artifacts_root": str(
                            Path(directory) / "retired-serving" / "artifacts"
                        ),
                    }
                ),
                encoding="utf-8",
            )

            resolved = resolve_reference(
                state,
                "model-family:v5",
                "model-family-v5",
                root,
            )

            self.assertEqual(resolved, reference)

    def test_scheduled_workflow_does_not_pin_a_model_version(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/drift.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("market1501-vit-bnneck:v1-dvc", workflow)
        self.assertIn("inputs.release_id || ''", workflow)
        self.assertIn("drift_report_local.sh", workflow)


if __name__ == "__main__":
    unittest.main()
