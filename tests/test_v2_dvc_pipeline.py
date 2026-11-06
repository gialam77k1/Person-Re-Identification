from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class V2DVCPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = yaml.load(
            (ROOT / "dvc.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        )["stages"]

    def test_contains_complete_v2_release_graph(self) -> None:
        expected = {
            "v2_validate_data", "v2_checkpoint", "v2_evaluate_msmt17",
            "v2_evaluate_market1501", "v2_export_onnx", "v2_verify_onnx",
            "v2_extract_reference", "v2_validate_release", "v2_package_triton",
        }
        self.assertTrue(expected.issubset(self.pipeline))

    def test_export_and_evaluation_do_not_trigger_each_other(self) -> None:
        export_deps = "\n".join(self.pipeline["v2_export_onnx"]["deps"])
        msmt_deps = "\n".join(self.pipeline["v2_evaluate_msmt17"]["deps"])
        market_deps = "\n".join(self.pipeline["v2_evaluate_market1501"]["deps"])
        self.assertNotIn("evaluate_", export_deps)
        self.assertNotIn("/onnx/", msmt_deps)
        self.assertNotIn("/onnx/", market_deps)

    def test_v2_outputs_never_target_v1_release_root(self) -> None:
        for stage_name, stage in self.pipeline.items():
            if not stage_name.startswith("v2_"):
                continue
            outputs = [*stage.get("outs", []), *stage.get("metrics", [])]
            serialized = yaml.safe_dump(outputs)
            self.assertNotIn("market1501-vit-bnneck-v1", serialized)
            self.assertIn("v2_pipeline.release_slug", serialized)

    def test_release_validation_precedes_triton_packaging(self) -> None:
        package_deps = self.pipeline["v2_package_triton"]["deps"]
        self.assertTrue(any("validation_report.json" in value for value in package_deps))
        self.assertTrue(any("model_manifest.json" in value for value in package_deps))

    def test_validation_wrapper_has_valid_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/validate_v2_dvc_release.sh")],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
