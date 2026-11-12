from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.create_msmt17_demo_batch import create_batch


class CreateMSMT17DemoBatchTest(unittest.TestCase):
    def test_creates_balanced_reproducible_six_camera_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "MSMT17_V1"
            for camera in range(1, 7):
                for index in range(3):
                    path = source / "train" / f"{index:04d}" / f"{index:04d}_{index:03d}_{camera:02d}_sample.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"\xff\xd8\xff" + bytes([camera, index]))
            output = root / "inbox" / "demo"
            manifest = create_batch(
                source,
                output,
                count=12,
                cameras=(1, 2, 3, 4, 5, 6),
                seed=7,
            )

            self.assertEqual(manifest["summary"]["images"], 12)
            self.assertEqual(manifest["summary"]["cameras"], 6)
            self.assertEqual(set(manifest["summary"]["images_per_camera"].values()), {2})
            self.assertEqual(manifest["source_dataset"], "MSMT17_V1")
            self.assertEqual(manifest["dataset"], "msmt17-demo")
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue(all((output / item["image"]).is_file() for item in manifest["samples"]))


if __name__ == "__main__":
    unittest.main()
