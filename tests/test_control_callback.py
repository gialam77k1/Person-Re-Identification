from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from scripts import control_callback


class ControlCallbackTest(unittest.TestCase):
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
