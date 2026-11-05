from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SelfHostedRunnerSmokeTest(unittest.TestCase):
    def test_smoke_script_has_valid_shell_syntax(self) -> None:
        for script_name in (
            "self_hosted_runner_smoke.sh",
            "install_self_hosted_runner_service.sh",
        ):
            with self.subTest(script=script_name):
                subprocess.run(
                    ["bash", "-n", str(ROOT / "scripts" / script_name)],
                    check=True,
                )

    def test_workflow_is_manual_and_targets_linux_self_hosted(self) -> None:
        workflow = yaml.load(
            (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        self.assertEqual(workflow["jobs"]["smoke"]["runs-on"], ["self-hosted", "linux"])

    def test_workflow_verifies_persistent_stack_by_default(self) -> None:
        workflow = yaml.load(
            (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        inputs = workflow["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["start_stack"]["default"], "false")
        self.assertEqual(inputs["verify_services"]["default"], "true")

        job_env = workflow["jobs"]["smoke"]["env"]
        self.assertEqual(job_env["DOCKER_CONTEXT"], "default")
        self.assertEqual(job_env["DOCKER_HOST"], "unix:///var/run/docker.sock")

    def test_workflow_cleans_up_temporary_environment_file(self) -> None:
        workflow_text = (ROOT / ".github/workflows/self-hosted-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("secrets.MLOPS_ENV_FILE", workflow_text)
        self.assertIn("rm -f .env.mlops", workflow_text)

    def test_service_installer_pins_native_docker(self) -> None:
        installer = (ROOT / "scripts/install_self_hosted_runner_service.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("systemctl enable --now", installer)
        self.assertIn("DOCKER_CONTEXT=default", installer)
        self.assertIn("DOCKER_HOST=unix:///var/run/docker.sock", installer)


if __name__ == "__main__":
    unittest.main()
