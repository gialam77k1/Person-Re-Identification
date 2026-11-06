from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class CrossDatasetDriftWorkflowTest(unittest.TestCase):
    def test_local_script_has_valid_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "scripts/cross_dataset_drift.sh")], check=True)

    def test_script_pulls_missing_inputs_separately(self) -> None:
        script = (ROOT / "scripts/cross_dataset_drift.sh").read_text(encoding="utf-8")
        self.assertIn("pull datasets/MSMT17_V1.dvc", script)
        self.assertIn("pull extract_reference", script)
        self.assertNotIn('pull datasets/MSMT17_V1.dvc "${REFERENCE}"', script)
        self.assertIn("pull validate_release", script)
        self.assertIn("--metadata", script)
        self.assertIn("--model-manifest", script)
        self.assertIn("textfile/reid_cross_dataset_drift.prom", script)

    def test_workflow_is_manual_and_uses_self_hosted_linux(self) -> None:
        workflow = yaml.load(
            (ROOT / ".github/workflows/cross-domain-drift.yml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
        job = workflow["jobs"]["detect-drift"]
        self.assertEqual(job["runs-on"], ["self-hosted", "linux"])

    def test_workflow_uploads_drift_evidence(self) -> None:
        workflow_text = (ROOT / ".github/workflows/cross-domain-drift.yml").read_text(encoding="utf-8")
        self.assertIn("drift_report.json", workflow_text)
        self.assertIn("evidence_manifest.json", workflow_text)
        self.assertIn("retention-days: 30", workflow_text)
        self.assertIn("retraining_evidence.py build", workflow_text)
        self.assertIn("dvc_local.sh pull validate_release", workflow_text)
        self.assertIn("mlops-evidence-${{ inputs.cycle_id }}", workflow_text)
        self.assertIn('"workflow_run_id":"${{ github.run_id }}"', workflow_text)


if __name__ == "__main__":
    unittest.main()
