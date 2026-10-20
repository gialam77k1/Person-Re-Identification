from __future__ import annotations

import builtins
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.check_repository_hygiene import inspect_files
from src.validate_repository import validate_repository


class CiChecksTest(unittest.TestCase):
    def test_common_utils_import_does_not_require_torch(self) -> None:
        module_name = "src.common.utils"
        existing = sys.modules.pop(module_name, None)
        real_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("torch intentionally unavailable in CPU CI")
            return real_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=import_without_torch):
                imported = importlib.import_module(module_name)
            self.assertTrue(callable(imported.save_json))
        finally:
            sys.modules.pop(module_name, None)
            if existing is not None:
                sys.modules[module_name] = existing

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
