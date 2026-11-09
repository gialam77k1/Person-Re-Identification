from __future__ import annotations

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
