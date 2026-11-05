from __future__ import annotations

import unittest

from src.capture_external_embeddings import deterministic_sample


class CaptureExternalEmbeddingsTest(unittest.TestCase):
    def test_sampling_is_reproducible_and_keeps_source_order(self) -> None:
        samples = [{"index": index} for index in range(100)]
        first = deterministic_sample(samples, limit=10, seed=42)
        second = deterministic_sample(samples, limit=10, seed=42)

        self.assertEqual(first, second)
        self.assertEqual([item["index"] for item in first], sorted(item["index"] for item in first))

    def test_non_positive_limit_keeps_all_samples(self) -> None:
        samples = [{"index": index} for index in range(3)]
        self.assertEqual(deterministic_sample(samples, limit=0, seed=42), samples)


if __name__ == "__main__":
    unittest.main()
