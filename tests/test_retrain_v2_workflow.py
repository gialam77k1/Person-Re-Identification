from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RetrainV2WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / ".github/workflows/retrain-v2.yml"
        self.text = self.path.read_text(encoding="utf-8")
        self.workflow = yaml.load(self.text, Loader=yaml.BaseLoader)

    def test_runs_on_self_hosted_gpu_path_and_downloads_exact_evidence(self) -> None:
        job = self.workflow["jobs"]["retrain"]
        self.assertEqual(job["runs-on"], ["self-hosted", "linux"])
        self.assertIn("run-id: ${{ inputs.evidence_run_id }}", self.text)
        self.assertIn("retraining_evidence.py verify", self.text)

    def test_workflow_preflights_trains_evaluates_and_gates(self) -> None:
        for command in (
            "train_msmt17_v2_local.sh --check",
            "train_msmt17_v2_local.sh",
            "evaluate_msmt17_v2.sh",
            "retraining_validation_gate.py",
            "register_retraining_candidate.py",
        ):
            self.assertIn(command, self.text)

    def test_only_candidate_alias_is_assigned(self) -> None:
        self.assertIn("--alias candidate", self.text)
        self.assertNotIn("--alias champion", self.text)
        self.assertNotIn("deploy production", self.text.lower())

    def test_large_models_are_not_uploaded_as_actions_artifacts(self) -> None:
        artifact_section = self.text.split("name: msmt17-v2-retraining-reports", 1)[1]
        self.assertNotIn("*.pth\n", artifact_section)
        self.assertIn("dvc_local.sh push", self.text)

    def test_changed_shell_scripts_have_valid_syntax(self) -> None:
        for script in ("train_msmt17_v2_local.sh", "evaluate_msmt17_v2.sh"):
            subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)


if __name__ == "__main__":
    unittest.main()
