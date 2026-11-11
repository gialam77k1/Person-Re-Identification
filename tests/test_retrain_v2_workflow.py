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
            "src.validate_post_training_contract",
        ):
            self.assertIn(command, self.text)

    def test_post_training_contract_is_checked_before_gpu_training(self) -> None:
        contract = self.text.index("src.validate_post_training_contract")
        training = self.text.index("./scripts/train_duke_v2_local.sh\n")
        self.assertLess(contract, training)

    def test_can_resume_a_finished_mlflow_run_without_retraining(self) -> None:
        self.assertIn("resume_training_run_id", self.text)
        self.assertIn("src.materialize_training_run", self.text)
        self.assertIn("inputs.resume_training_run_id == ''", self.text)
        self.assertIn("inputs.resume_training_run_id != ''", self.text)
        self.assertIn("--allow-source-commit-mismatch", self.text)
        self.assertIn("id: evidence_lineage", self.text)
        recovery_failure = self.text.index("Report post-training recovery failure")
        self.assertIn("evaluate_v2 FAIL", self.text[recovery_failure:])
        self.assertIn("steps.restore.outcome == 'failure'", self.text)
        self.assertIn("steps.evidence_lineage.outcome == 'failure'", self.text)

    def test_runtime_evaluation_configs_follow_each_target_version(self) -> None:
        self.assertIn("src.prepare_candidate_evaluation", self.text)
        self.assertIn("DUKE_EVAL_CONFIG=configs/runtime/", self.text)
        self.assertIn("MARKET1501_EVAL_CONFIG=configs/runtime/", self.text)

    def test_evaluation_wrappers_ignore_training_config_environment(self) -> None:
        duke = (ROOT / "scripts/evaluate_duke_v2.sh").read_text(encoding="utf-8")
        market = (ROOT / "scripts/evaluate_market1501_v2.sh").read_text(encoding="utf-8")
        self.assertIn("DUKE_EVAL_CONFIG", duke)
        self.assertIn("DUKE_V2_REPORT", duke)
        self.assertNotIn('${CONFIG_PATH', duke)
        self.assertIn("MARKET1501_EVAL_CONFIG", market)
        self.assertIn("MARKET1501_V2_REPORT", market)
        self.assertNotIn('${CONFIG_PATH', market)

    def test_dvc_release_wrapper_uses_valid_shell_identifiers(self) -> None:
        wrapper = (ROOT / "scripts/validate_v2_dvc_release.sh").read_text(encoding="utf-8")
        dvc_pipeline = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
        self.assertIn('DUKE_REPORT=', wrapper)
        self.assertNotIn('DukeMTMC-reID_REPORT', wrapper)
        self.assertIn(
            'env V2_PIPELINE_ROOT=artifacts/pipeline/${v2_pipeline.release_slug}',
            dvc_pipeline,
        )

    def test_failures_are_reported_against_the_stage_that_failed(self) -> None:
        self.assertIn('id: evaluate', self.text)
        self.assertIn('evaluate_v2 FAIL', self.text)
        self.assertIn('id: lineage_gate', self.text)
        self.assertIn('id: release_gate', self.text)
        self.assertIn('id: publish', self.text)
        self.assertIn('publish_artifacts FAIL', self.text)
        self.assertIn('id: registry', self.text)
        self.assertIn('register_candidate FAIL', self.text)
        self.assertNotIn('name: Report workflow failure', self.text)

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
            "validate_v2_dvc_release.sh",
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
