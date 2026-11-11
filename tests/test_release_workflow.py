from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow_path = ROOT / ".github/workflows/release.yml"
        cls.text = cls.workflow_path.read_text(encoding="utf-8")
        cls.workflow = yaml.load(cls.text, Loader=yaml.BaseLoader)

    def test_candidate_artifacts_can_be_read_cross_run(self) -> None:
        self.assertEqual(self.workflow["permissions"]["actions"], "read")

    def test_each_self_hosted_release_job_restores_candidate_lock(self) -> None:
        for job_name in ("validate-candidate", "deploy-staging", "deploy-production"):
            with self.subTest(job=job_name):
                steps = self.workflow["jobs"][job_name]["steps"]
                names = [step["name"] for step in steps]
                locate = names.index("Locate immutable candidate artifact")
                download = names.index("Download immutable candidate lineage")
                pull = next(
                    index
                    for index, name in enumerate(names)
                    if name in {"Pull candidate artifacts from DVC", "Pull deployment artifacts"}
                )
                self.assertLess(locate, download)
                self.assertLess(download, pull)
                self.assertIn(
                    "materialize_release_artifacts.sh", steps[pull]["run"]
                )

    def test_release_never_pulls_an_untracked_parent_directory(self) -> None:
        self.assertNotIn(
            'dvc_local.sh pull "artifacts/pipeline/${RELEASE_SLUG}"', self.text
        )

    def test_release_failures_are_reported_to_control_center(self) -> None:
        job = self.workflow["jobs"]["report-release-failure"]
        self.assertEqual(job["runs-on"], ["self-hosted", "linux"])
        self.assertEqual(
            set(job["needs"]),
            {
                "validate-candidate",
                "build-image",
                "deploy-staging",
                "deploy-production",
            },
        )
        self.assertIn("always()", job["if"])
        self.assertIn("cancelled", job["if"])
        report_step = next(
            step
            for step in job["steps"]
            if step["name"] == "Mark production approval failed"
        )
        self.assertIn("production_approval FAIL", report_step["run"])


if __name__ == "__main__":
    unittest.main()
