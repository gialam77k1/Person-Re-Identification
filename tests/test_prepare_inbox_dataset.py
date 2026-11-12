from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.prepare_inbox_dataset import (
    build_view,
    load_samples,
    split_identity_disjoint,
    validate_samples,
)


class PrepareInboxDatasetTest(unittest.TestCase):
    def test_manifest_images_become_canonical_train_view_with_fixed_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            base = root / "duke"
            (source / "images").mkdir(parents=True)
            samples = []
            for pid in (42, 43):
                for camera in (3, 11):
                    name = f"person-{pid}-camera-{camera}.jpg"
                    (source / "images" / name).write_bytes(
                        b"\xff\xd8\xff" + bytes([pid, camera])
                    )
                    samples.append(
                        {"image": f"images/{name}", "person_id": pid, "camera_id": camera}
                    )
            (source / "manifest.json").write_text(
                json.dumps({"samples": samples}),
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

            revision = validate_samples(
                source, load_samples(source), holdout_target=2, split_seed=7
            )[1]
            result = build_view(
                argparse.Namespace(
                    source=str(source),
                    base_dataset=str(base),
                    base_config=str(base_config),
                    output_root=str(view),
                    output_config=str(config),
                    output_pointer=str(pointer),
                    revision=revision,
                    evaluation_holdout_images=2,
                    split_seed=7,
                )
            )

            self.assertEqual(result["train_images"], 2)
            self.assertEqual(result["query_images"], 1)
            self.assertEqual(result["gallery_images"], 1)
            self.assertFalse(result["identity_overlap"])
            self.assertEqual(len(list((view / "bounding_box_train").glob("*.jpg"))), 2)
            self.assertEqual(len(list((view / "query").glob("*.jpg"))), 1)
            self.assertEqual(len(list((view / "bounding_box_test").glob("*.jpg"))), 1)
            runtime = yaml.safe_load(config.read_text(encoding="utf-8"))
            self.assertEqual(runtime["data"]["expected_train_images"], 2)
            self.assertEqual(
                yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"],
                revision,
            )

    def test_split_requires_cross_camera_identity_and_never_leaks_identity(self) -> None:
        samples = [
            (Path(f"/{pid}-{camera}.jpg"), pid, camera)
            for pid in range(10)
            for camera in (1, 11)
        ]
        result = split_identity_disjoint(samples, holdout_target=8, seed=3)
        train_ids = {item[1] for item in result["train"]}
        evaluation_ids = {item[1] for item in [*result["query"], *result["gallery"]]}
        self.assertFalse(train_ids & evaluation_ids)
        self.assertEqual(len(result["query"]) + len(result["gallery"]), 8)
        for query in result["query"]:
            self.assertTrue(
                any(item[1] == query[1] and item[2] != query[2] for item in result["gallery"])
            )


if __name__ == "__main__":
    unittest.main()
