from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.validate_data import inspect_msmt17_split


class ValidateMSMT17Test(unittest.TestCase):
    def test_valid_list_returns_identity_and_camera_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "test/0007/0007_001_03_0303morning_0001_0.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            list_path = root / "list_query.txt"
            list_path.write_text(f"0007/{image.name} 7\n", encoding="utf-8")

            result = inspect_msmt17_split(root, "query", list_path)

            self.assertEqual(result["num_images"], 1)
            self.assertEqual(result["num_person_ids"], 1)
            self.assertEqual(result["num_cameras"], 1)

    def test_pid_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "test/0007/0007_001_03_0303morning_0001_0.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            list_path = root / "list_query.txt"
            list_path.write_text(f"0007/{image.name} 8\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "person ID mismatch"):
                inspect_msmt17_split(root, "query", list_path)


if __name__ == "__main__":
    unittest.main()
