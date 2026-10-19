from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.check_repository_hygiene import inspect_files
from src.validate_repository import validate_repository


class CiChecksTest(unittest.TestCase):
    def test_current_repository_metadata_is_valid(self) -> None:
        self.assertEqual(validate_repository(), [])

    def test_hygiene_rejects_large_binary_and_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            large = root / "large.txt"
            model = root / "model.onnx"
            secret = root / "secret.txt"
            large.write_text("x" * 11, encoding="utf-8")
            model.write_bytes(b"model")
            fake_token = "gh" + "p_" + "abcdefghijklmnopqrstuvwxyz1234567890"
            secret.write_text(f"token={fake_token}", encoding="utf-8")
            errors = inspect_files([large, model, secret], root, max_bytes=10)
        self.assertTrue(any("exceeds" in error for error in errors))
        self.assertTrue(any("binary" in error for error in errors))
        self.assertTrue(any("GitHub token" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
