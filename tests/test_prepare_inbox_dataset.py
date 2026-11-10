from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.prepare_inbox_dataset import build_view, load_samples, validate_samples


class PrepareInboxDatasetTest(unittest.TestCase):
    def test_manifest_images_become_canonical_train_view_with_fixed_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            base = root / "duke"
            (source / "images").mkdir(parents=True)
            (source / "images" / "arbitrary-name.jpg").write_bytes(b"\xff\xd8\xffimage")
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "image": "images/arbitrary-name.jpg",
                                "person_id": 42,
                                "camera_id": 3,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (base / "query").mkdir(parents=True)
            (base / "bounding_box_test").mkdir()
            base_config = root / "base.yaml"
            base_config.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "location": {"root": "old"},
                            "expected_train_images": 0,
                            "expected_train_identities": 0,
                        },
                        "evaluation": {"dataset_pointer": "old.dvc"},
                    }
                ),
                encoding="utf-8",
            )
            view = root / "view"
            config = root / "runtime.yaml"
            pointer = root / "inbox.dvc"

            revision = validate_samples(source, load_samples(source))[1]
            result = build_view(
                argparse.Namespace(
                    source=str(source),
                    base_dataset=str(base),
                    base_config=str(base_config),
                    output_root=str(view),
                    output_config=str(config),
                    output_pointer=str(pointer),
                    revision=revision,
                )
            )

            self.assertEqual(result["train_images"], 1)
            self.assertTrue((view / "bounding_box_train" / "000042_c3_inbox_00000000.jpg").is_symlink())
            self.assertTrue((view / "query").is_symlink())
            runtime = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(runtime["data"]["expected_train_images"], 1)
            self.assertEqual(
                yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"],
                revision,
            )


if __name__ == "__main__":
    unittest.main()
