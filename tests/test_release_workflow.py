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

    def test_retraining_artifact_keeps_exact_validation_policy(self) -> None:
        retraining = (ROOT / ".github/workflows/retrain-v2.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("configs/model_validation_v2.yaml", retraining)

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

    def test_every_release_callback_loads_local_credentials(self) -> None:
        callback_steps = []
        for job in self.workflow["jobs"].values():
            for step in job.get("steps", []):
                if "control_callback.py" in step.get("run", ""):
                    callback_steps.append(step)

        self.assertGreaterEqual(len(callback_steps), 3)
        for step in callback_steps:
            with self.subTest(step=step["name"]):
                self.assertIn("source .env.mlops", step["run"])

    def test_release_deploys_version_matched_qdrant_index(self) -> None:
        staging = next(
            step
            for step in self.workflow["jobs"]["deploy-staging"]["steps"]
            if step["name"] == "Deploy staging"
        )
        production = next(
            step
            for step in self.workflow["jobs"]["deploy-production"]["steps"]
            if step["name"] == "Deploy production"
        )
        self.assertIn("deploy_local.sh deploy", staging["run"])
        self.assertIn("deploy_local.sh deploy", production["run"])
        self.assertIn("Qdrant index", staging["run"])

    def test_rollback_reports_model_and_index_as_one_release(self) -> None:
        rollback = (ROOT / ".github/workflows/rollback.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("deploy_local.sh rollback", rollback)
        self.assertIn("previous model and matching Qdrant index", rollback)

    def test_compose_requires_release_managed_runtime(self) -> None:
        compose = (ROOT / "docker-compose.mlops.yml").read_text(encoding="utf-8")
        self.assertIn("RELEASE_RUNTIME_MANAGED:?Use ./scripts/mlops_stack.sh", compose)
        self.assertIn("RELEASE_ID:?Release runtime is not resolved", compose)

    def test_stack_startup_restores_production_state(self) -> None:
        stack = (ROOT / "scripts/mlops_stack.sh").read_text(encoding="utf-8")
        self.assertIn('production/current.json', stack)
        self.assertIn('release_deployment.py" runtime-env', stack)
        self.assertIn("restore-serving", stack)
        self.assertIn("verify_release", stack)


if __name__ == "__main__":
    unittest.main()
