from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.reid_filename import parse_image_folder_reid_filename
from src.pipeline_failure import summarize_failure
from src.validate_post_training_contract import validate_image_folder_dataset


class ReIDFilenameTest(unittest.TestCase):
    def test_duke_and_market_names_share_one_camera_contract(self) -> None:
        self.assertEqual(parse_image_folder_reid_filename("0002_c1_f0044158.jpg"), (2, 0))
        self.assertEqual(parse_image_folder_reid_filename("0002_c6_inbox_0001.jpg"), (2, 5))
        self.assertEqual(parse_image_folder_reid_filename("0002_c1s1_000451_03.jpg"), (2, 0))

    def test_invalid_camera_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            parse_image_folder_reid_filename("0002_camera1_000451.jpg")
        with self.assertRaisesRegex(ValueError, "one-based"):
            parse_image_folder_reid_filename("0002_c0_000451.jpg")


class EvaluationDatasetPreflightTest(unittest.TestCase):
    def test_scans_every_split_and_requires_cross_camera_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "query", "gallery"):
                (root / split).mkdir()
            (root / "train/0001_c1s1_001.jpg").touch()
            (root / "query/0001_c1s2_002.jpg").touch()
            (root / "gallery/0001_c2s1_003.jpg").touch()
            payload = {
                "data": {
                    "source_type": "image_folder",
                    "location": {
                        "root": str(root),
                        "splits": {"train": "train", "query": "query", "gallery": "gallery"},
                    },
                }
            }

            result = validate_image_folder_dataset(payload, config_path=root / "eval.yaml")

            self.assertEqual(result["train_images"], 1)
            self.assertEqual(result["valid_queries"], 1)

    def test_rejects_query_without_cross_camera_gallery_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "query", "gallery"):
                (root / split).mkdir()
            (root / "train/0001_c1_001.jpg").touch()
            (root / "query/0001_c1_002.jpg").touch()
            (root / "gallery/0001_c1_003.jpg").touch()
            payload = {
                "data": {
                    "location": {
                        "root": str(root),
                        "splits": {"train": "train", "query": "query", "gallery": "gallery"},
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "different gallery camera"):
                validate_image_folder_dataset(payload, config_path=root / "eval.yaml")


class PipelineFailureSummaryTest(unittest.TestCase):
    def test_prefers_python_root_cause_over_conda_wrapper(self) -> None:
        log = """Traceback (most recent call last):
ValueError: Invalid image-folder ReID filename: broken.jpg
ERROR conda.cli.main_run:execute(148): command failed
"""
        self.assertEqual(
            summarize_failure(log),
            "ValueError: Invalid image-folder ReID filename: broken.jpg",
        )


if __name__ == "__main__":
    unittest.main()
