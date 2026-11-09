from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class MSMT17CrossDomainConfigTest(unittest.TestCase):
    def test_config_uses_official_list_splits(self) -> None:
        config = yaml.safe_load((ROOT / "configs/msmt17_v1_evaluation.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["dataset"]["name"], "msmt17")
        self.assertEqual(
            set(config["data"]["location"]["list_files"]),
            {"train", "query", "gallery"},
        )
        self.assertFalse(config["evaluation"]["use_rerank"])

    def test_local_evaluation_script_has_valid_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "scripts/evaluate_msmt17_v1.sh")], check=True)


if __name__ == "__main__":
    unittest.main()
