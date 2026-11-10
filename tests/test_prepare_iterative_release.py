from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from src.prepare_iterative_release import prepare


ROOT = Path(__file__).resolve().parents[1]


class PrepareIterativeReleaseTest(unittest.TestCase):
    def test_arbitrary_version_updates_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            for relative in (
                "configs/duke_v2_train.yaml",
                "configs/duke_v2_eval.yaml",
                "configs/market1501_v2_eval.yaml",
                "configs/model_validation_v2.yaml",
                "configs/catastrophic_forgetting.yaml",
                "params.yaml",
            ):
                shutil.copy2(ROOT / relative, root / relative)
            previous = Path.cwd()
            os.chdir(root)
            try:
                result = prepare(
                    source_release="market1501-duke-vit-bnneck:v7",
                    target_release="market1501-duke-vit-bnneck:v8",
                    checkpoint="/tmp/champion-v7.pth",
                    run_root="artifacts/duke/market1501-duke-vit-bnneck-v8",
                    dataset_root="/tmp/inbox-v8",
                    dataset_pointer="/tmp/inbox-v8.dvc",
                )
            finally:
                os.chdir(previous)
            train = yaml.safe_load((root / "configs/duke_v2_train.yaml").read_text())
            params = yaml.safe_load((root / "params.yaml").read_text())
            policy = yaml.safe_load((root / "configs/model_validation_v2.yaml").read_text())
            self.assertEqual(result["target_version"], 8)
            self.assertEqual(train["release"]["base_release_id"], "market1501-duke-vit-bnneck:v7")
            self.assertEqual(train["release"]["id"], "market1501-duke-vit-bnneck:v8")
            self.assertIsNone(train["transfer"]["expected_checkpoint_epoch"])
            self.assertEqual(params["v2_pipeline"]["model_version"], 8)
            self.assertEqual(params["v2_pipeline"]["duke_root"], "/tmp/inbox-v8")
            self.assertEqual(policy["release_id"], "market1501-duke-vit-bnneck:v8")


if __name__ == "__main__":
    unittest.main()
