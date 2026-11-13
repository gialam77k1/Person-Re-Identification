import tempfile
import unittest
from pathlib import Path

from src.reset_candidate_workspace import (
    WorkspaceResetError,
    reset_candidate_workspace,
)


class ResetCandidateWorkspaceTests(unittest.TestCase):
    def test_reset_removes_only_exact_version_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "artifacts/duke/model-family-v3"
            pipeline_root = workspace / "artifacts/pipeline/model-family-v3"
            keep = workspace / "artifacts/duke/model-family-v2"
            for path in (run_root, pipeline_root, keep):
                path.mkdir(parents=True)
                (path / "evidence.txt").write_text("evidence", encoding="utf-8")

            removed = reset_candidate_workspace(
                workspace=workspace,
                release_id="model-family:v3",
                run_root=Path("artifacts/duke/model-family-v3"),
                pipeline_root=Path("artifacts/pipeline/model-family-v3"),
            )

            self.assertEqual(len(removed), 2)
            self.assertFalse(run_root.exists())
            self.assertFalse(pipeline_root.exists())
            self.assertTrue((keep / "evidence.txt").is_file())

    def test_reset_rejects_paths_outside_exact_release(self) -> None:
        invalid_paths = (
            ("artifacts/duke/model-family-v2", "artifacts/pipeline/model-family-v3"),
            ("artifacts/duke/model-family-v3", "artifacts/other/model-family-v3"),
            (".", "artifacts/pipeline/model-family-v3"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for run_root, pipeline_root in invalid_paths:
                with self.subTest(run_root=run_root, pipeline_root=pipeline_root):
                    with self.assertRaises(WorkspaceResetError):
                        reset_candidate_workspace(
                            workspace=Path(directory),
                            release_id="model-family:v3",
                            run_root=Path(run_root),
                            pipeline_root=Path(pipeline_root),
                        )


if __name__ == "__main__":
    unittest.main()
