from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation_assets import decode_preview_asset
from src.reid.evaluation import build_retrieval_preview


class RetrievalPreviewTest(unittest.TestCase):
    def test_filters_same_identity_same_camera_and_marks_correct_match(self) -> None:
        preview = build_retrieval_preview(
            query_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
            gallery_features=np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
            ),
            query_paths=["/runner/datasets/dukemtmc-reid/query/0001_c1.jpg"],
            query_pids=np.asarray([1]),
            query_camids=np.asarray([1]),
            gallery_paths=[
                "/runner/datasets/dukemtmc-reid/bounding_box_test/0001_c1.jpg",
                "/runner/datasets/dukemtmc-reid/bounding_box_test/0001_c2.jpg",
                "/runner/datasets/dukemtmc-reid/bounding_box_test/0002_c3.jpg",
            ],
            gallery_pids=np.asarray([1, 1, 2]),
            gallery_camids=np.asarray([1, 2, 3]),
            topk=2,
        )

        self.assertEqual(preview[0]["query"]["image_path"], "query/0001_c1.jpg")
        self.assertEqual(
            preview[0]["predictions"][0]["image_path"],
            "bounding_box_test/0001_c2.jpg",
        )
        self.assertTrue(preview[0]["predictions"][0]["correct"])
        self.assertTrue(preview[0]["top1_correct"])
        self.assertEqual(len(preview[0]["predictions"]), 2)

    def test_embeds_content_addressed_thumbnails_for_portable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = root / "query.jpg"
            gallery = root / "gallery.jpg"
            Image.new("RGB", (64, 128), (20, 40, 60)).save(query)
            Image.new("RGB", (64, 128), (60, 40, 20)).save(gallery)
            preview = build_retrieval_preview(
                query_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
                gallery_features=np.asarray([[0.9, 0.1]], dtype=np.float32),
                query_paths=[str(query)],
                query_pids=np.asarray([1]),
                query_camids=np.asarray([1]),
                gallery_paths=[str(gallery)],
                gallery_pids=np.asarray([1]),
                gallery_camids=np.asarray([2]),
                embed_images=True,
            )

        query_content, query_mime = decode_preview_asset(preview[0]["query"])
        gallery_content, gallery_mime = decode_preview_asset(preview[0]["predictions"][0])
        self.assertTrue(query_content.startswith(b"\xff\xd8"))
        self.assertTrue(gallery_content.startswith(b"\xff\xd8"))
        self.assertEqual(query_mime, "image/jpeg")
        self.assertEqual(gallery_mime, "image/jpeg")


if __name__ == "__main__":
    unittest.main()
