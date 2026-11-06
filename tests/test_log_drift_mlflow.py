from __future__ import annotations

import unittest

from src.log_drift_mlflow import drift_metrics


class LogDriftMlflowTest(unittest.TestCase):
    def test_maps_detector_result_without_recomputing_decision(self) -> None:
        report = {
            "drift_detected": True,
            "reference_samples": 12936,
            "current_samples": 1000,
            "embedding_dim": 512,
            "checks": {
                "ks_statistic": {"value": 0.203, "maximum": 0.15, "passed": False}
            },
        }
        metrics = drift_metrics(report)
        self.assertEqual(metrics["drift_detected"], 1.0)
        self.assertEqual(metrics["drift_ks_statistic"], 0.203)
        self.assertEqual(metrics["threshold_ks_statistic"], 0.15)


if __name__ == "__main__":
    unittest.main()
