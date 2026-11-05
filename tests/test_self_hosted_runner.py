from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SelfHostedRunnerSmokeTest(unittest.TestCase):
    def test_smoke_script_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/self_hosted_runner_smoke.sh")],
            check=True,
        )

    def test_workflow_is_manual_and_targets_linux_self_hosted(self) -> None:
        workflow = yaml.load(
            (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(workflow["jobs"]["smoke"]["runs-on"], ["self-hosted", "linux"])

    def test_workflow_cleans_up_temporary_environment_file(self) -> None:
        workflow_text = (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("secrets.MLOPS_ENV_FILE", workflow_text)
        self.assertIn("rm -f .env.mlops", workflow_text)

    def test_workflow_pulls_serving_release_before_preflight(self) -> None:
        workflow = yaml.load(
            (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        steps = workflow["jobs"]["smoke"]["steps"]
        step_names = [step["name"] for step in steps]
        pull_index = step_names.index("Pull serving artifacts from DVC")
        preflight_index = step_names.index("Check runner prerequisites")
        self.assertLess(pull_index, preflight_index)
        self.assertIn("inputs.release_slug", steps[pull_index]["env"]["RELEASE_SLUG"])
        self.assertIn("scripts/dvc_local.sh pull", steps[pull_index]["run"])


if __name__ == "__main__":
    unittest.main()
