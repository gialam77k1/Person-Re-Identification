from __future__ import annotations

import os
import sys
import json
import tempfile
import unittest
from unittest import mock

from scripts import control_callback


class ControlCallbackTest(unittest.TestCase):
    def test_metrics_file_is_merged_with_workflow_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "decision.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"status": "REVIEW_REQUIRED", "target_epochs": 20}, handle)
            metrics = control_callback.load_metrics(
                '{"workflow_run_id":"123"}', path
            )
        self.assertEqual(metrics["status"], "REVIEW_REQUIRED")
        self.assertEqual(metrics["workflow_run_id"], "123")

    def test_manual_workflow_skips_callback(self) -> None:
        with mock.patch.object(sys, "argv", ["control_callback.py", "manual", "train_v2", "RUNNING"]):
            self.assertEqual(control_callback.main(), 0)

    def test_callback_requires_server_side_token(self) -> None:
        with (
            mock.patch.object(sys, "argv", ["control_callback.py", "cycle-1", "train_v2", "RUNNING"]),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(control_callback.main(), 1)


if __name__ == "__main__":
    unittest.main()
