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
        self.assertIn("src.retraining_evidence verify", self.text)
        self.assertIn("src.verify_approved_decision", self.text)
        self.assertIn("/retraining-decision", self.text)

    def test_workflow_preflights_trains_evaluates_and_gates(self) -> None:
        for command in (
            "train_duke_v2_local.sh --check",
            "train_duke_v2_local.sh",
            "evaluate_v2_dual_domain.sh",
            "src.catastrophic_forgetting_gate",
            "src.log_v2_evaluations_mlflow",
            "src.retraining_validation_gate",
            "publish_v2_artifacts.sh",
            "src.register_retraining_candidate",
        ):
            self.assertIn(command, self.text)

    def test_training_contract_is_memory_safe_and_capped_at_fifty_epochs(self) -> None:
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", self.text)
        self.assertIn("expandable_segments:True,max_split_size_mb:128", self.text)
        self.assertIn("--target-epochs 50", self.text)

    def test_only_candidate_alias_is_assigned(self) -> None:
        self.assertIn("--alias candidate", self.text)
        self.assertNotIn("--alias champion", self.text)
        self.assertNotIn("deploy production", self.text.lower())
        self.assertNotIn("retraining_decision.sh", self.text)

    def test_full_v2_release_gate_runs_before_candidate_registration(self) -> None:
        repro = self.text.index("repro v2_package_triton")
        registration = self.text.index("src.register_retraining_candidate")
        self.assertLess(repro, registration)
        self.assertIn("V2_PIPELINE_ROOT", self.text)
        self.assertIn("parity/metrics/onnx_parity.json", self.text)
        self.assertIn("${V2_PIPELINE_ROOT}/release/validation_report.json", self.text)

    def test_large_models_are_not_uploaded_as_actions_artifacts(self) -> None:
        artifact_section = self.text.split("name: duke-v2-retraining-reports", 1)[1]
        self.assertNotIn("*.pth\n", artifact_section)
        self.assertIn("publish_v2_artifacts.sh", self.text)

    def test_dvc_publication_is_verified_before_registry(self) -> None:
        publication = self.text.index("publish_v2_artifacts.sh")
        registration = self.text.index("src.register_retraining_candidate")
        self.assertLess(publication, registration)
        self.assertIn("--dvc-publication", self.text)
        self.assertIn("--mlflow-run", self.text)

    def test_forgetting_thresholds_are_versioned_outside_workflow(self) -> None:
        self.assertIn("configs/catastrophic_forgetting.yaml", self.text)
        self.assertNotIn("rank1_min_absolute_gain:", self.text)
        self.assertIn("model_releases/market1501-vit-bnneck/v1/model_manifest.json", self.text)
        self.assertIn("evaluation_duke.json", self.text)
        self.assertIn("evaluation_market1501.json", self.text)

    def test_changed_shell_scripts_have_valid_syntax(self) -> None:
        for script in (
            "train_duke_v2_local.sh",
            "evaluate_duke_v2.sh",
            "evaluate_market1501_v2.sh",
            "evaluate_v2_dual_domain.sh",
            "publish_v2_artifacts.sh",
        ):
            subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)

    def test_python_package_entrypoints_are_import_safe(self) -> None:
        self.assertNotIn("python src/", self.text)
        for module in (
            "retraining_evidence",
            "verify_approved_decision",
            "validate_v2_training_completion",
            "catastrophic_forgetting_gate",
            "log_v2_evaluations_mlflow",
            "retraining_validation_gate",
            "register_retraining_candidate",
        ):
            self.assertIn(f"python -m src.{module}", self.text)

    def test_runner_context_is_not_used_in_job_level_environment(self) -> None:
        job_environment = self.text.split("    steps:", 1)[0]
        self.assertNotIn("runner.temp", job_environment)
        self.assertNotIn("${{ runner.temp }}", self.text)
        self.assertIn('base_checkpoint="${RUNNER_TEMP}/reid-base-', self.text)


if __name__ == "__main__":
    unittest.main()
