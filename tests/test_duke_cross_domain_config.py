from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DukeCrossDomainConfigTest(unittest.TestCase):
    def test_config_uses_official_image_folder_splits(self) -> None:
        config = yaml.safe_load((ROOT / "configs/duke_v1_evaluation.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["dataset"]["name"], "dukemtmc-reid")
        self.assertEqual(
            config["data"]["location"]["splits"],
            {
                "train": "bounding_box_train",
                "query": "query",
                "gallery": "bounding_box_test",
            },
        )
        self.assertFalse(config["evaluation"]["use_rerank"])

    def test_local_evaluation_script_has_valid_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "scripts/evaluate_duke_v1.sh")], check=True)


if __name__ == "__main__":
    unittest.main()
