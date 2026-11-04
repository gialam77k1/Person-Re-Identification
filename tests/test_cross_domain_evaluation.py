from __future__ import annotations

import unittest

import numpy as np

from src.reid.cross_domain import evaluate_cross_domain_retrieval


class CrossDomainEvaluationTest(unittest.TestCase):
    def test_perfect_cross_camera_matches(self) -> None:
        query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        gallery = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

        result = evaluate_cross_domain_retrieval(
            query,
            gallery,
            query_pids=np.asarray([10, 20]),
            gallery_pids=np.asarray([10, 20, 30]),
            query_camids=np.asarray([0, 0]),
            gallery_camids=np.asarray([1, 1, 1]),
            query_batch_size=1,
        )

        self.assertEqual(result["rank1"], 1.0)
        self.assertEqual(result["mAP"], 1.0)
        self.assertEqual(result["valid_queries"], 2)

    def test_non_finite_embedding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            evaluate_cross_domain_retrieval(
                np.asarray([[np.nan, 0.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([1]),
                np.asarray([1]),
                np.asarray([0]),
                np.asarray([1]),
            )


if __name__ == "__main__":
    unittest.main()
